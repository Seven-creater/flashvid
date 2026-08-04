from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


SCRIPT = Path(__file__).parents[1] / "scripts" / "run_fast_hybrid_replays.py"
SPEC = importlib.util.spec_from_file_location("run_fast_hybrid_replays", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)


def _manifest(path: Path) -> dict[str, object]:
    row = {
        "dataset": "lsdbench",
        "sample_id": "one",
        "video": "one.mp4",
        "question": "What happens?",
        "choices": {"A": "first", "B": "second"},
        "answer": "B",
        "time_range": "PRIVATE",
    }
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")
    return row


def test_public_manifest_loader_drops_labels(tmp_path: Path) -> None:
    path = tmp_path / "manifest.jsonl"
    _manifest(path)
    samples, digest = module.load_public_samples(path, "lsdbench")
    assert len(digest) == 64
    sample = samples["one"]
    assert sample.candidate_answer is None
    assert not hasattr(sample, "answer")
    assert "PRIVATE" not in json.dumps(sample.__dict__)


def test_candidate_loader_requires_clean_uniform32_protocol(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.jsonl"
    _manifest(manifest)
    samples, _ = module.load_public_samples(manifest, "lsdbench")
    candidate = {
        "sample_id": "one",
        "prediction": "B",
        "baseline_mode": "direct",
        "sampling_id": "uniform32",
        "enable_thinking": False,
        "protocol_request": {"max_tokens": 512, "temperature": 0.0},
    }
    path = tmp_path / "candidate.jsonl"
    path.write_text(json.dumps(candidate) + "\n", encoding="utf-8")
    answers, records, digest = module.load_candidates(path, samples)
    assert answers == {"one": "B"}
    assert records["one"]["prediction"] == "B"
    assert len(digest) == 64

    candidate["protocol_request"]["max_tokens"] = 32
    path.write_text(json.dumps(candidate) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="frozen protocol"):
        module.load_candidates(path, samples)


def test_ready_specs_reject_parallel_nodes_for_same_sample(tmp_path: Path) -> None:
    train_hash = "a" * 64
    dataset_hash = "b" * 64
    config_hash = "c" * 64
    base = {
        "dataset": "lsdbench",
        "sample_id": "one",
        "train600_manifest_sha256": train_hash,
        "dataset_manifest_sha256": dataset_hash,
        "config_sha256": config_hash,
        "planned_calls": [{"start_time": 0, "end_time": 2, "nframes": 2, "resize": 1}],
        "replica_trajectory_ids": ["a", "b", "c"],
    }
    first = dict(base)
    first["counterfactual_fingerprint"] = module.canonical_sha256(first)
    second = {**base, "variant_id": "other"}
    second["counterfactual_fingerprint"] = module.canonical_sha256(second)
    path = tmp_path / "specs.jsonl"
    path.write_text(json.dumps(first) + "\n" + json.dumps(second) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="one DAG node per sample"):
        module.load_ready_specs(
            path,
            dataset="lsdbench",
            train600_sha256=train_hash,
            dataset_manifest_sha256=dataset_hash,
            config_sha256=config_hash,
        )
