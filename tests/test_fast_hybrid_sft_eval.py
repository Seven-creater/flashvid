from __future__ import annotations

import json
from pathlib import Path

import pytest

from flashvid_eval.fast_hybrid_eval_protocol import (
    DATASETS,
    SPLITS,
    audit_result_file,
    build_protocol,
    freeze_json,
    load_protocol,
    sha256_file,
)
from scripts.freeze_fast_hybrid_sft_checkpoint import freeze_checkpoint
from scripts.run_fast_hybrid_sft_eval import _load_checkpoint, build_jobs


BASE_SHA = "a" * 64


def _jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def _fixture(tmp_path: Path) -> tuple[Path, str, dict[tuple[str, str], Path]]:
    datasets: dict[str, dict] = {}
    candidates: dict[tuple[str, str], Path] = {}
    for dataset in DATASETS:
        entry: dict[str, object] = {
            "annotations": str(tmp_path / f"{dataset}.json"),
            "video_root": str(tmp_path / "videos" / dataset),
        }
        for split in SPLITS:
            manifest = tmp_path / "manifests" / f"{split}_{dataset}.jsonl"
            rows = [{"dataset": dataset, "sample_id": f"{split}-{dataset}-0", "answer": "A"}]
            _jsonl(manifest, rows)
            entry[f"{split}_manifest"] = str(manifest)
            entry[f"{split}_manifest_sha256"] = sha256_file(manifest)
            candidate = tmp_path / "candidates" / f"{split}_{dataset}.jsonl"
            _jsonl(
                candidate,
                [
                    {
                        "dataset": dataset,
                        "sample_id": rows[0]["sample_id"],
                        "prediction": "A",
                        "baseline_mode": "direct",
                        "sampling_id": "uniform32",
                        "enable_thinking": False,
                        "protocol_request": {"max_tokens": 512, "temperature": 0.0},
                    }
                ],
            )
            candidates[(split, dataset)] = candidate
        datasets[dataset] = entry
    config = tmp_path / "experiment.json"
    config.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "teacher": {
                    "model_path": str(tmp_path / "model"),
                    "model_artifact_sha256": BASE_SHA,
                    "official_eva_commit": "o" * 40,
                },
                "datasets": datasets,
            }
        ),
        encoding="utf-8",
    )
    return config, sha256_file(config), candidates


def test_protocol_freezes_all_manifest_and_candidate_bytes(tmp_path: Path) -> None:
    config, config_sha, candidates = _fixture(tmp_path)
    protocol = build_protocol(
        experiment_config_path=config,
        expected_experiment_config_sha256=config_sha,
        candidate_paths=candidates,
    )
    output = tmp_path / "protocol.json"
    freeze_json(output, protocol)

    loaded = load_protocol(output, sha256_file(output))

    assert loaded["base_model_artifact_sha256"] == BASE_SHA
    assert loaded["parameters"]["max_total_visual_tokens"] == 24000
    assert loaded["splits"]["test"]["cgbench"]["candidate"]["count"] == 1


def test_checkpoint_and_eval_jobs_bind_base_adapter_manifest_and_candidate(
    tmp_path: Path,
) -> None:
    config, config_sha, candidates = _fixture(tmp_path)
    protocol = build_protocol(
        experiment_config_path=config,
        expected_experiment_config_sha256=config_sha,
        candidate_paths=candidates,
    )
    adapter = tmp_path / "checkpoint-7"
    adapter.mkdir()
    (adapter / "adapter_model.safetensors").write_bytes(b"adapter")
    (adapter / "adapter_config.json").write_text("{}", encoding="utf-8")
    (adapter / "trainer_state.json").write_text(
        json.dumps({"global_step": 7, "epoch": 1.0}), encoding="utf-8"
    )
    train_data = tmp_path / "sft.jsonl"
    train_data.write_text("{}\n", encoding="utf-8")
    checkpoint_path = tmp_path / "checkpoint.json"
    checkpoint = freeze_checkpoint(
        checkpoint_id="epoch-1",
        epoch=1,
        global_step=7,
        adapter_root=adapter,
        base_artifact_sha256=BASE_SHA,
        train_data=train_data,
        experiment_config=config,
        expected_experiment_config_sha256=config_sha,
        served_name="Qwen3.5-9B-fast-hybrid-epoch1",
        base_urls=["http://127.0.0.1:8200/v1", "http://127.0.0.1:8201/v1"],
        output=checkpoint_path,
    )
    loaded = _load_checkpoint(checkpoint_path, protocol)
    jobs = build_jobs(
        protocol=protocol,
        phase="dev",
        run_id="epoch-1",
        served_name=loaded["served_name"],
        served_model_sha256=loaded["served_stack_sha256"],
        teacher_model_sha256=BASE_SHA,
        output_root=tmp_path / "eval",
        python="python",
        repo_root=Path("/repo"),
        resume=True,
    )

    assert checkpoint["served_stack_sha256"] != BASE_SHA
    assert len(jobs) == 3
    command = list(jobs[0].command)
    assert command[command.index("--model-artifact-sha256") + 1] == checkpoint[
        "served_stack_sha256"
    ]
    assert command[command.index("--teacher-model-artifact-sha256") + 1] == BASE_SHA
    assert "--resume" in command
    assert "--retry-errors" in command


def test_result_audit_rejects_unbound_served_stack(tmp_path: Path) -> None:
    result = tmp_path / "result.jsonl"
    row = {
        "dataset": "lvbench",
        "sample_id": "one",
        "answer": "A",
        "prediction": "A",
        "manifest_sha256": "m" * 64,
        "candidate_results_sha256": "c" * 64,
        "experiment_config_sha256": "e" * 64,
        "model_artifact_sha256": "s" * 64,
        "teacher_model_sha256": BASE_SHA,
        "agent_version": "fast_hybrid_v2",
        "annotation_leak_check": "passed",
        "candidate_rerun": 0,
        "candidate_cost_complete": True,
        "end_to_end_total_tokens": 100,
        "end_to_end_total_tokens_complete": True,
        "end_to_end_visual_tokens": 80,
        "end_to_end_visual_tokens_complete": True,
    }
    _jsonl(result, [row])

    audit = audit_result_file(
        result,
        dataset="lvbench",
        expected_count=1,
        expected_sample_ids={"one"},
        manifest_sha256="m" * 64,
        candidate_sha256="c" * 64,
        experiment_config_sha256="e" * 64,
        served_model_sha256="s" * 64,
        teacher_model_sha256=BASE_SHA,
    )
    assert audit["rows"] == 1

    row["model_artifact_sha256"] = BASE_SHA
    _jsonl(result, [row])
    try:
        audit_result_file(
            result,
            dataset="lvbench",
            expected_count=1,
            expected_sample_ids={"one"},
            manifest_sha256="m" * 64,
            candidate_sha256="c" * 64,
            experiment_config_sha256="e" * 64,
            served_model_sha256="s" * 64,
            teacher_model_sha256=BASE_SHA,
        )
    except ValueError as error:
        assert "model_artifact_sha256" in str(error)
    else:
        raise AssertionError("changed served stack was accepted")


def test_result_audit_rejects_wrong_manifest_sample_ids(tmp_path: Path) -> None:
    result = tmp_path / "result.jsonl"
    row = {
        "dataset": "lvbench",
        "sample_id": "wrong",
        "manifest_sha256": "m" * 64,
        "candidate_results_sha256": "c" * 64,
        "experiment_config_sha256": "e" * 64,
        "model_artifact_sha256": "s" * 64,
        "teacher_model_sha256": BASE_SHA,
        "agent_version": "fast_hybrid_v2",
        "annotation_leak_check": "passed",
        "candidate_rerun": 0,
        "candidate_cost_complete": True,
        "end_to_end_total_tokens": 100,
        "end_to_end_total_tokens_complete": True,
        "end_to_end_visual_tokens": 80,
        "end_to_end_visual_tokens_complete": True,
    }
    _jsonl(result, [row])

    with pytest.raises(ValueError, match="sample IDs differ"):
        audit_result_file(
            result,
            dataset="lvbench",
            expected_count=1,
            expected_sample_ids={"expected"},
            manifest_sha256="m" * 64,
            candidate_sha256="c" * 64,
            experiment_config_sha256="e" * 64,
            served_model_sha256="s" * 64,
            teacher_model_sha256=BASE_SHA,
        )
