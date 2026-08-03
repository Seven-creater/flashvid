from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest


SCRIPT = Path(__file__).parents[1] / "scripts" / "run_qwen_counterfactuals.py"
SPEC = importlib.util.spec_from_file_location("run_qwen_counterfactuals", SCRIPT)
assert SPEC and SPEC.loader
module = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = module
SPEC.loader.exec_module(module)


def test_counterfactual_specs_are_hash_bound_and_dataset_filtered(tmp_path: Path) -> None:
    manifest_hash = "a" * 64
    dataset_manifest_hash = "d" * 64
    config_hash = "b" * 64
    base = {
        "schema_version": 2,
        "dataset": "lvbench",
        "sample_id": "s1",
        "schedule_id": "schedule-1",
        "variant_id": "frames_50",
        "family_id": "schedule-1~frames_50",
        "base_trajectory_id": "base",
        "manifest_sha256": manifest_hash,
        "train600_manifest_sha256": manifest_hash,
        "dataset_manifest_sha256": dataset_manifest_hash,
        "config_sha256": config_hash,
        "planned_calls": [
            {
                "start_time": 1.0,
                "end_time": 3.0,
                "nframes": 4,
                "resize": 0.5,
            }
        ],
    }
    base["counterfactual_fingerprint"] = module.canonical_sha256(base)
    other = {**base, "dataset": "cgbench", "sample_id": "c1"}
    other["counterfactual_fingerprint"] = module.canonical_sha256(
        {key: value for key, value in other.items() if key != "counterfactual_fingerprint"}
    )
    path = tmp_path / "specs.jsonl"
    path.write_text(json.dumps(base) + "\n" + json.dumps(other) + "\n", encoding="utf-8")
    loaded = module.load_specs(
        path,
        dataset="lvbench",
        train600_manifest_sha256=manifest_hash,
        dataset_manifest_sha256=dataset_manifest_hash,
        config_sha256=config_hash,
    )
    assert [row["sample_id"] for row in loaded] == ["s1"]
    request = module.frame_requests(loaded[0])[0]
    assert request.nframes == 4 and request.fps is None

    tampered = dict(base)
    tampered["planned_calls"] = [{**base["planned_calls"][0], "nframes": 2}]
    path.write_text(json.dumps(tampered) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="fingerprint mismatch"):
        module.load_specs(
            path,
            dataset="lvbench",
            train600_manifest_sha256=manifest_hash,
            dataset_manifest_sha256=dataset_manifest_hash,
            config_sha256=config_hash,
        )


def test_counterfactual_replica_seeds_are_unique_and_reproducible() -> None:
    spec = {"counterfactual_fingerprint": "c" * 64}
    first = [module.replica_seed(42, spec, index) for index in range(3)]
    second = [module.replica_seed(42, spec, index) for index in range(3)]
    assert first == second
    assert len(set(first)) == 3
