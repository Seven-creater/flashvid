from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.build_qwen_agent_sft import _load_input_bundle
from flashvid_eval.qwen_sft import canonical_sha256, sha256_file


def _write(path: Path, content: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


def _bundle(tmp_path: Path) -> tuple[Path, Path, str]:
    config_path = _write(tmp_path / "config.json", '{"schema_version":1}\n')
    plan_path = _write(tmp_path / "trajectory_plan.json", '{"schema_version":1}\n')
    trajectory_path = _write(tmp_path / "trajectory.jsonl", '{"sample_id":"1"}\n')
    config_hash = "a" * 64
    payload = {
        "schema_version": 1,
        "config": {
            "path": str(config_path),
            "sha256": sha256_file(config_path),
        },
        "config_sha256": config_hash,
        "trajectory_run_plan": {
            "path": str(plan_path),
            "sha256": sha256_file(plan_path),
            "plan_sha256": "b" * 64,
        },
        "trajectory_files": [
            {
                "path": str(trajectory_path),
                "sha256": sha256_file(trajectory_path),
            }
        ],
        "expected_provenance": {
            "model": "Qwen3.5-9B",
            "model_artifact_sha256": "c" * 64,
            "dataset_manifest_sha256s": ["d" * 64],
            "agent_config_sha256s": ["e" * 64],
            "runner_fingerprints": ["f" * 64],
        },
    }
    payload["bundle_sha256"] = canonical_sha256(payload)
    bundle_path = tmp_path / "bundle.json"
    bundle_path.write_text(json.dumps(payload), encoding="utf-8")
    return bundle_path, trajectory_path, config_hash


def test_sft_input_bundle_is_self_hashed_and_binds_every_source_file(
    tmp_path: Path,
) -> None:
    bundle_path, trajectory_path, config_hash = _bundle(tmp_path)

    trajectories, provenance = _load_input_bundle(bundle_path, config_hash)

    assert trajectories == [trajectory_path]
    assert provenance.model == "Qwen3.5-9B"
    assert provenance.runner_fingerprints == frozenset({"f" * 64})

    trajectory_path.write_text('{"sample_id":"changed"}\n', encoding="utf-8")
    with pytest.raises(RuntimeError, match="trajectory file changed"):
        _load_input_bundle(bundle_path, config_hash)


def test_sft_input_bundle_rejects_self_hash_or_experiment_mismatch(
    tmp_path: Path,
) -> None:
    bundle_path, _, config_hash = _bundle(tmp_path)

    with pytest.raises(RuntimeError, match="experiment config mismatch"):
        _load_input_bundle(bundle_path, "0" * 64)

    payload = json.loads(bundle_path.read_text(encoding="utf-8"))
    payload["counterfactual_files"] = 1
    bundle_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(RuntimeError, match="self-hash mismatch"):
        _load_input_bundle(bundle_path, config_hash)
