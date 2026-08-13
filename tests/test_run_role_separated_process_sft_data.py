from __future__ import annotations

from argparse import Namespace
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import pytest

from scripts import run_role_separated_process_sft_data as pipeline


CONFIG_PATH = Path("configs/experiments/role_separated_process_sft.json")
TRAIN_SHA = "a" * 64
OBSERVER_SHA = "5f050597da76f16ff28499fb75fcd6562a1fbf4bc20df83124b77709e9ee9d60"
GIT_HEAD = "1" * 40


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def _train600(path: Path) -> tuple[list[dict], str]:
    rows: list[dict] = []
    for dataset in pipeline.DATASETS:
        rows.extend(
            {
                "dataset": dataset,
                "sample_id": f"{dataset}-{index:03d}",
                "video": f"{dataset}/{index:03d}.mp4",
                "question": "Public question?",
                "choices": {"A": "one", "B": "two"},
                "answer": "A",
            }
            for index in range(200)
        )
    _write_jsonl(path, rows)
    return rows, hashlib.sha256(path.read_bytes()).hexdigest()


def _attribution(tmp_path: Path) -> tuple[Path, str, Path]:
    frozen = tmp_path / "frozen_role_inputs.jsonl"
    _write_jsonl(frozen, [{"sample_id": f"dev-{index}"} for index in range(30)])
    summary = {
        "schema_version": 1,
        "runtime_commit": "2" * 40,
        "decision": {
            "train_new_planner_lora": True,
            "train_observer_lora": False,
            "keep_verifier_base": True,
            "keep_answerer_base": True,
        },
        "frozen_input": {
            "path": str(frozen.resolve()),
            "samples": 30,
            "sha256": pipeline.file_sha256(frozen),
        },
    }
    path = tmp_path / "role_attribution_summary.json"
    path.write_text(json.dumps(summary), encoding="utf-8")
    return path, pipeline.file_sha256(path), frozen


def _source_rows(train_sha: str) -> list[dict]:
    return [
        {
            "dataset": "lvbench",
            "sample_id": "lvbench-000",
            "trajectory_id": "lvbench:lvbench-000:single:0",
            "train600_manifest_sha256": train_sha,
        },
        {
            "dataset": "lvbench",
            "sample_id": "lvbench-000",
            "trajectory_id": "lvbench:lvbench-000:hierarchical:0",
            "train600_manifest_sha256": train_sha,
        },
        {
            "dataset": "cgbench",
            "sample_id": "cgbench-001",
            "trajectory_id": "cgbench:cgbench-001:timestamp:0",
            "train600_manifest_sha256": train_sha,
        },
    ]


def _lock_fixture(tmp_path: Path) -> tuple[dict, dict]:
    train = tmp_path / "train600.jsonl"
    _rows, train_sha = _train600(train)
    shard_a = tmp_path / "shard-a.jsonl"
    shard_b = tmp_path / "shard-b.jsonl"
    sources = _source_rows(train_sha)
    _write_jsonl(shard_a, sources[:2])
    _write_jsonl(shard_b, sources[2:])
    attribution, attribution_sha, frozen = _attribution(tmp_path)
    implementation = tmp_path / "implementation.py"
    implementation.write_text("VALUE = 1\n", encoding="utf-8")
    config_sha = pipeline.file_sha256(CONFIG_PATH)
    kwargs = {
        "repo_root": tmp_path,
        "experiment_config": CONFIG_PATH.resolve(),
        "expected_experiment_config_sha256": config_sha,
        "train600": train,
        "expected_train600_sha256": train_sha,
        "source_shards": [shard_a, shard_b],
        "role_attribution_summary": attribution,
        "expected_role_attribution_sha256": attribution_sha,
        "frozen_role_input": frozen,
        "observer_model": "Qwen3.5-9B",
        "observer_artifact_sha256": OBSERVER_SHA,
        "verifier_model": "Qwen3.5-9B",
        "verifier_artifact_sha256": OBSERVER_SHA,
        "base_model_artifact_sha256": OBSERVER_SHA,
        "git_head": GIT_HEAD,
        "git_branch": "codex/role-separated",
        "implementation_files": {"implementation.py": implementation},
    }
    return kwargs, {"train": train, "shards": [shard_a, shard_b]}


def test_training_source_lock_is_deterministic_train_only_and_allows_replicas(
    tmp_path: Path,
) -> None:
    kwargs, _paths = _lock_fixture(tmp_path)

    first, source_by_id = pipeline.build_training_source_lock(**kwargs)
    second, second_sources = pipeline.build_training_source_lock(**kwargs)

    assert first == second
    assert source_by_id == second_sources
    assert first["train600"]["rows"] == 600
    assert first["train600"]["dataset_rows"] == {
        dataset: 200 for dataset in pipeline.DATASETS
    }
    assert first["source_shards"]["rows"] == 3
    assert first["source_shards"]["unique_sample_keys"] == 2
    assert first["role_attribution"]["trainable_roles"] == ["planner"]
    assert first["role_attribution"]["frozen_base_roles"] == [
        "observer",
        "verifier",
        "answerer",
    ]
    assert first["observer"]["artifact_sha256"] == OBSERVER_SHA
    assert first["verifier"]["artifact_sha256"] == OBSERVER_SHA
    assert first["training_base_model"]["artifact_sha256"] == OBSERVER_SHA
    assert first["role_separated_runtime_version"] == (
        pipeline.ROLE_SEPARATED_RUNTIME_VERSION
    )
    assert first["role_prompt_schema_bundle_sha256"] == (
        pipeline.role_prompt_schema_bundle_sha256()
    )
    assert first["controller_output_constraint_version"] == (
        pipeline.ROLE_SEPARATED_CONTROLLER_OUTPUT_CONSTRAINT_VERSION
    )
    serialized = json.dumps(first, ensure_ascii=False, sort_keys=True)
    assert "final_test" not in serialized
    assert not any(value in serialized for value in pipeline.TEST_MANIFEST_SHA256.values())


@pytest.mark.parametrize(
    ("mutate", "message"),
    (
        (
            lambda kwargs: kwargs.update(expected_train600_sha256="0" * 64),
            "Train600 SHA-256 changed",
        ),
        (
            lambda kwargs: kwargs.update(observer_artifact_sha256="3" * 64),
            "frozen Base attribution artifact",
        ),
        (
            lambda kwargs: kwargs.update(expected_experiment_config_sha256="0" * 64),
            "experiment config SHA-256 changed",
        ),
    ),
)
def test_training_source_lock_rejects_frozen_identity_drift(
    tmp_path: Path, mutate, message: str
) -> None:
    kwargs, _paths = _lock_fixture(tmp_path)
    mutate(kwargs)

    with pytest.raises(ValueError, match=message):
        pipeline.build_training_source_lock(**kwargs)


def test_training_source_lock_rejects_bad_train_scope_and_source_drift(
    tmp_path: Path,
) -> None:
    kwargs, paths = _lock_fixture(tmp_path)
    train_rows = pipeline.read_jsonl(paths["train"])
    train_rows.pop()
    _write_jsonl(paths["train"], train_rows)
    kwargs["expected_train600_sha256"] = pipeline.file_sha256(paths["train"])
    with pytest.raises(ValueError, match="600 rows/200 per dataset"):
        pipeline.build_training_source_lock(**kwargs)

    kwargs, paths = _lock_fixture(tmp_path / "second")
    source_rows = pipeline.read_jsonl(paths["shards"][1])
    source_rows[0]["sample_id"] = "outside-train600"
    _write_jsonl(paths["shards"][1], source_rows)
    with pytest.raises(ValueError, match="outside Train600"):
        pipeline.build_training_source_lock(**kwargs)


def test_training_source_lock_rejects_duplicate_trajectory_and_attribution_flip(
    tmp_path: Path,
) -> None:
    kwargs, paths = _lock_fixture(tmp_path)
    second = pipeline.read_jsonl(paths["shards"][1])
    second[0]["trajectory_id"] = _source_rows(kwargs["expected_train600_sha256"])[0][
        "trajectory_id"
    ]
    _write_jsonl(paths["shards"][1], second)
    with pytest.raises(ValueError, match="duplicate source trajectory_id"):
        pipeline.build_training_source_lock(**kwargs)

    kwargs, _paths = _lock_fixture(tmp_path / "flipped")
    attribution = pipeline._read_json(kwargs["role_attribution_summary"])
    attribution["decision"]["train_observer_lora"] = True
    kwargs["role_attribution_summary"].write_text(
        json.dumps(attribution), encoding="utf-8"
    )
    kwargs["expected_role_attribution_sha256"] = pipeline.file_sha256(
        kwargs["role_attribution_summary"]
    )
    with pytest.raises(ValueError, match="Planner-only"):
        pipeline.build_training_source_lock(**kwargs)


def test_training_source_lock_rejects_test_identity_but_allows_demo_test_path() -> None:
    pipeline._reject_test_identity(
        {"train600": {"path": "/data/Demo/test/train600.jsonl"}}
    )
    with pytest.raises(ValueError, match="Test field"):
        pipeline._reject_test_identity({"nested": {"final_test": "forbidden"}})
    with pytest.raises(ValueError, match="Test manifest"):
        pipeline._reject_test_identity(
            {"path": "/safe/lvbench_manifest_42_100.jsonl"}
        )
    with pytest.raises(ValueError, match="frozen Test SHA"):
        pipeline._reject_test_identity(
            {"sha256": next(iter(pipeline.TEST_MANIFEST_SHA256.values()))}
        )


def test_training_source_lock_validation_detects_file_and_payload_drift(
    tmp_path: Path,
) -> None:
    kwargs, _paths = _lock_fixture(tmp_path)
    lock, _source = pipeline.build_training_source_lock(**kwargs)
    pipeline.validate_training_source_lock(lock, **kwargs)

    changed = deepcopy(lock)
    changed["observer"]["model"] = "old-LoRA"
    with pytest.raises(ValueError, match="differs from current frozen inputs"):
        pipeline.validate_training_source_lock(changed, **kwargs)

    implementation = next(iter(kwargs["implementation_files"].values()))
    implementation.write_text("VALUE = 2\n", encoding="utf-8")
    with pytest.raises(ValueError, match="differs from current frozen inputs"):
        pipeline.validate_training_source_lock(lock, **kwargs)


def test_scope_requires_exact_locked_rows_and_accepts_rare_subset(tmp_path: Path) -> None:
    kwargs, paths = _lock_fixture(tmp_path)
    _lock, source_by_id = pipeline.build_training_source_lock(**kwargs)
    subset = tmp_path / "rare.jsonl"
    _write_jsonl(subset, [pipeline.read_jsonl(paths["shards"][0])[0]])

    resolved, report = pipeline._scope_inputs([subset], source_by_id)

    assert resolved == [subset.resolve()]
    assert report["mode"] == "explicit_scope"
    assert report["rows"] == 1
    assert report["locked_source_rows"] == 3

    changed = pipeline.read_jsonl(subset)
    changed[0]["sample_id"] = "changed"
    _write_jsonl(subset, changed)
    with pytest.raises(ValueError, match="not an exact locked source"):
        pipeline._scope_inputs([subset], source_by_id)


def test_endpoints_accept_one_to_four_and_reject_gpu_or_count_drift() -> None:
    pipeline._validate_endpoints(["http://127.0.0.1:8200/v1"], [3])
    pipeline._validate_endpoints(
        [f"http://127.0.0.1:{8200 + index}/v1" for index in range(4)],
        [0, 1, 2, 3],
    )
    with pytest.raises(ValueError, match="1-4"):
        pipeline._validate_endpoints(
            [f"http://127.0.0.1:{8200 + index}/v1" for index in range(5)],
            list(range(5)),
        )
    with pytest.raises(ValueError, match="uniquely bind"):
        pipeline._validate_endpoints(
            ["http://127.0.0.1:8200/v1", "http://127.0.0.1:8201/v1"],
            [0, 0],
        )


def test_replay_partition_keeps_failures_out_of_visual_csv_input(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    replay = tmp_path / "replay.jsonl"
    _write_jsonl(
        replay,
        [
            {"source_trajectory_id": "ok", "error": None, "error_type": None},
            {
                "source_trajectory_id": "failed",
                "error": "observer failed",
                "error_type": "observer_failure",
            },
        ],
    )
    monkeypatch.setattr(pipeline, "bind_visual_csv_jobs", lambda rows: [object()])
    success = tmp_path / "success.jsonl"
    failure = tmp_path / "failure.jsonl"

    summary = pipeline._partition_replays(
        [replay],
        success_path=success,
        failure_path=failure,
        summary_path=tmp_path / "summary.json",
    )

    assert summary["success"] == 1
    assert summary["failure"] == 1
    assert [row["source_trajectory_id"] for row in pipeline.read_jsonl(success)] == [
        "ok"
    ]
    assert [row["source_trajectory_id"] for row in pipeline.read_jsonl(failure)] == [
        "failed"
    ]


def test_replay_audit_rejects_runtime_contract_drift(tmp_path: Path) -> None:
    source = tmp_path / "source.jsonl"
    output = tmp_path / "output.jsonl"
    source_row = {
        "dataset": "lvbench",
        "sample_id": "sample",
        "trajectory_id": "source-id",
    }
    _write_jsonl(source, [source_row])
    result = {
        "source_trajectory_id": "source-id",
        "source_row_sha256": pipeline._canonical_sha256(source_row),
        "source_file_sha256": pipeline.file_sha256(source),
        "training_source_lock_sha256": "7" * 64,
        "experiment_config_sha256": "8" * 64,
        "role_separated_runtime_version": pipeline.ROLE_SEPARATED_RUNTIME_VERSION,
        "role_prompt_schema_bundle_sha256": (
            pipeline.role_prompt_schema_bundle_sha256()
        ),
        "controller_output_constraint_version": (
            pipeline.ROLE_SEPARATED_CONTROLLER_OUTPUT_CONSTRAINT_VERSION
        ),
        "role_separated_observer": True,
        "served_model_artifact_sha256": OBSERVER_SHA,
        "candidate_rerun": 0,
        "annotation_leak_check": "passed",
        "error": None,
        "error_type": None,
    }
    _write_jsonl(output, [result])
    kwargs = {
        "source_lock_sha256": "7" * 64,
        "experiment_config_sha256": "8" * 64,
        "observer_artifact_sha256": OBSERVER_SHA,
        "role_separated_runtime_version": pipeline.ROLE_SEPARATED_RUNTIME_VERSION,
        "role_prompt_schema_bundle_sha256": (
            pipeline.role_prompt_schema_bundle_sha256()
        ),
        "controller_output_constraint_version": (
            pipeline.ROLE_SEPARATED_CONTROLLER_OUTPUT_CONSTRAINT_VERSION
        ),
    }
    pipeline._audit_replay_output(source, output, **kwargs)

    result["controller_output_constraint_version"] = "drifted"
    _write_jsonl(output, [result])
    with pytest.raises(RuntimeError, match="controller_output_constraint_version"):
        pipeline._audit_replay_output(source, output, **kwargs)


def test_prepare_only_is_atomic_resumable_and_never_needs_endpoints(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    kwargs, paths = _lock_fixture(tmp_path / "inputs")
    args = Namespace(
        repo_root=Path.cwd(),
        run_root=tmp_path / "run",
        experiment_config=kwargs["experiment_config"],
        expected_experiment_config_sha256=kwargs[
            "expected_experiment_config_sha256"
        ],
        train600=kwargs["train600"],
        expected_train600_sha256=kwargs["expected_train600_sha256"],
        source_shard=paths["shards"],
        scope_source=None,
        role_attribution_summary=kwargs["role_attribution_summary"],
        expected_role_attribution_sha256=kwargs[
            "expected_role_attribution_sha256"
        ],
        frozen_role_input=kwargs["frozen_role_input"],
        observer_artifact_sha256=OBSERVER_SHA,
        verifier_artifact_sha256=OBSERVER_SHA,
        base_model_artifact_sha256=OBSERVER_SHA,
        expected_git_head=GIT_HEAD,
        expected_git_branch="codex/role-separated",
        base_url=[],
        gpu_id=[],
        python="python",
        sft_python=None,
        model_path=None,
        model="Qwen3.5-9B",
        seed=42,
        max_tokens=1024,
        max_frames_per_call=128,
        request_timeout=300.0,
        replay_concurrency=8,
        judge_concurrency=4,
        stop_after="length",
        prepare_only=True,
        resume=False,
    )
    monkeypatch.setattr(
        pipeline,
        "_git_identity",
        lambda *_args, **_kwargs: (GIT_HEAD, "codex/role-separated"),
    )
    monkeypatch.setattr(
        pipeline,
        "_run_command",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("prepare-only launched a command")
        ),
    )

    first = pipeline.run_pipeline(args)
    assert first["status"] == "stopped"
    assert first["stopped_after"] == "prepare"
    lock = args.run_root / "source_lock/training_source_lock.json"
    lock_bytes = lock.read_bytes()

    args.resume = True
    args.prepare_only = False
    args.stop_after = "prepare"
    second = pipeline.run_pipeline(args)
    assert second["status"] == "stopped"
    assert lock.read_bytes() == lock_bytes
