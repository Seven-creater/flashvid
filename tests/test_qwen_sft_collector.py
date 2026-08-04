from __future__ import annotations

import importlib.util
import json
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "scripts" / "collect_qwen_sft_inputs.py"
SPEC = importlib.util.spec_from_file_location("collect_qwen_sft_inputs", SCRIPT)
assert SPEC and SPEC.loader
module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )


def test_collector_accepts_hash_bound_rescue_without_counting_it_as_base(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(module, "SCHEDULES_PER_DATASET", 1)
    monkeypatch.setattr(module, "SAMPLES_PER_DATASET", 1)
    model_hash = "b" * 64
    train600_hash = "e" * 64
    dataset_hashes = {
        "lvbench": "1" * 64,
        "lsdbench": "2" * 64,
        "cgbench": "3" * 64,
    }
    config = {
        "models": {"q9": {"artifact_sha256": model_hash}},
        "sft": {"train600": {"sha256": train600_hash}},
        "datasets": {
            dataset: {"train": {"sha256": digest}}
            for dataset, digest in dataset_hashes.items()
        },
    }
    config_path = tmp_path / "config.json"
    _write_json(config_path, config)
    config_hash = module.canonical_sha256(config)
    tasks: list[dict[str, object]] = []
    for index, dataset in enumerate(module.DATASETS):
        output_dir = tmp_path / "base" / dataset
        agent_hash = f"{index + 4:x}" * 64
        runner_hash = f"{index + 7:x}" * 64
        result = output_dir / f"{dataset}_base.jsonl"
        _write_jsonl(
            result,
            [
                {
                    "sample_id": f"base-{dataset}",
                    "scoring_deferred": True,
                    "model": "Qwen3.5-9B",
                    "model_artifact_sha256": model_hash,
                    "dataset_manifest_sha256": dataset_hashes[dataset],
                    "train600_manifest_sha256": train600_hash,
                    "manifest_sha256": train600_hash,
                    "agent_config_sha256": agent_hash,
                    "trajectory_runner_fingerprint": runner_hash,
                    "runner_fingerprint": runner_hash,
                }
            ],
        )
        _write_json(
            output_dir / f"frozen_inputs_{dataset}_base.json",
            {
                "dataset": dataset,
                "manifest": {"sha256": dataset_hashes[dataset]},
                "model": "Qwen3.5-9B",
                "model_artifact_sha256": model_hash,
                "experiment_config_sha256": config_hash,
                "scoring_deferred": True,
                "train600_manifest_sha256": train600_hash,
                "agent_config": {"sha256": agent_hash},
                "run_fingerprint": runner_hash,
            },
        )
        tasks.append(
            {
                "dataset": dataset,
                "task_id": "schedule-0",
                "manifest_sha256": dataset_hashes[dataset],
                "agent_config_sha256": agent_hash,
                "output_dir": str(output_dir),
                "command": [
                    "python",
                    "evaluate_mcq.py",
                    "--train600-manifest-sha256",
                    train600_hash,
                    "--expected-agent-config-sha256",
                    agent_hash,
                ],
            }
        )
    plan = {
        "schema_version": 1,
        "phase": "trajectory",
        "config_sha256": config_hash,
        "tasks": tasks,
    }
    plan["plan_sha256"] = module.canonical_sha256(plan)
    plan_path = tmp_path / "plan.json"
    _write_json(plan_path, plan)
    base_bundle = module.collect_bundle(
        config_path=config_path,
        trajectory_plan=plan_path,
        counterfactual_paths=[],
    )
    base_bundle_path = tmp_path / "base_bundle.json"
    _write_json(base_bundle_path, base_bundle)

    rescue_agent = tmp_path / "rescue_agent.json"
    _write_json(rescue_agent, {"agent": {"strategy": "a4_independent_arbitration"}})
    rescue_agent_hash = module.sha256_file(rescue_agent)
    rescue_manifests: dict[str, dict[str, object]] = {}
    rescue_runner_hash = "f" * 64
    for dataset in module.DATASETS:
        manifest = tmp_path / "rescue" / f"{dataset}.jsonl"
        rows = [{"sample_id": "rescue-1"}] if dataset == "lvbench" else []
        _write_jsonl(manifest, rows)
        result_dir = tmp_path / "rescue/results" / dataset
        rescue_manifests[dataset] = {
            "path": str(manifest),
            "sha256": module.sha256_file(manifest),
            "count": len(rows),
            "result_dir": str(result_dir),
        }
        if not rows:
            continue
        _write_jsonl(
            result_dir / f"{dataset}_rescue.jsonl",
            [
                {
                    "sample_id": "rescue-1",
                    "variant_id": "rescue",
                    "scoring_deferred": True,
                    "model": "Qwen3.5-9B",
                    "model_artifact_sha256": model_hash,
                    "dataset_manifest_sha256": module.sha256_file(manifest),
                    "train600_manifest_sha256": train600_hash,
                    "manifest_sha256": train600_hash,
                    "agent_config_sha256": rescue_agent_hash,
                    "trajectory_runner_fingerprint": rescue_runner_hash,
                    "runner_fingerprint": rescue_runner_hash,
                }
            ],
        )
        _write_json(
            result_dir / f"frozen_inputs_{dataset}_rescue.json",
            {
                "dataset": dataset,
                "manifest": {"sha256": module.sha256_file(manifest)},
                "model": "Qwen3.5-9B",
                "model_artifact_sha256": model_hash,
                "experiment_config_sha256": config_hash,
                "scoring_deferred": True,
                "train600_manifest_sha256": train600_hash,
                "trajectory_schedule_id": "rescue_a4_dense_v1",
                "trajectory_variant_id": "rescue",
                "agent_config": {"sha256": rescue_agent_hash},
                "run_fingerprint": rescue_runner_hash,
            },
        )
    rescue_index = {
        "schema_version": 1,
        "experiment_config": {"canonical_sha256": config_hash},
        "source_base_bundle": {
            "path": str(base_bundle_path),
            "sha256": module.sha256_file(base_bundle_path),
            "bundle_sha256": base_bundle["bundle_sha256"],
        },
        "train600_manifest_sha256": train600_hash,
        "model_artifact_sha256": model_hash,
        "schedule_id": "rescue_a4_dense_v1",
        "variant_id": "rescue",
        "agent_config": {
            "path": str(rescue_agent),
            "sha256": rescue_agent_hash,
        },
        "manifests": rescue_manifests,
    }
    rescue_index["rescue_index_sha256"] = module.canonical_sha256(rescue_index)
    rescue_index_path = tmp_path / "rescue_index.json"
    _write_json(rescue_index_path, rescue_index)

    bundle = module.collect_bundle(
        config_path=config_path,
        trajectory_plan=plan_path,
        counterfactual_paths=[],
        rescue_index_path=rescue_index_path,
    )
    assert bundle["rescue_files"] == 1
    assert len(bundle["trajectory_files"]) == 4
    assert module.sha256_file(tmp_path / "rescue/lvbench.jsonl") in bundle[
        "expected_provenance"
    ]["dataset_manifest_sha256s"]
