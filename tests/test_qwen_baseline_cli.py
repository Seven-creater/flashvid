from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from flashvid_eval.schemas import Sample
from scripts.evaluate_mcq import (
    _load_validated_mismatch_map,
    _stable_model_slug,
)


def _samples() -> list[Sample]:
    return [
        Sample("demo", "s1", "a.mp4", "Q1", {"A": "a", "B": "b"}, "A"),
        Sample("demo", "s2", "b.mp4", "Q2", {"A": "a", "B": "b"}, "B"),
    ]


def _write_map(path: Path, manifest_hash: str, video_root: Path) -> None:
    payload = {
        "schema_version": 1,
        "manifest_sha256": manifest_hash,
        "video_root": str(video_root.resolve()),
        "duration_bucket_edges_s": [0.0, 300.0, 900.0, 1800.0, 3600.0, 7200.0, "inf"],
        "mapping": {"s1": "b.mp4", "s2": "a.mp4"},
        "assignments": [
            {
                "dataset": "demo",
                "sample_id": "s1",
                "source_video": "a.mp4",
                "target_video": "b.mp4",
                "duration_bucket": 1,
            },
            {
                "dataset": "demo",
                "sample_id": "s2",
                "source_video": "b.mp4",
                "target_video": "a.mp4",
                "duration_bucket": 1,
            },
        ],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_mismatch_map_validates_manifest_assignments_and_bucket(tmp_path: Path) -> None:
    manifest_hash = hashlib.sha256(b"manifest").hexdigest()
    path = tmp_path / "map.json"
    _write_map(path, manifest_hash, tmp_path)
    mapping, map_hash, _payload = _load_validated_mismatch_map(
        path,
        samples=_samples(),
        manifest_hash=manifest_hash,
        video_root=tmp_path,
    )
    assert mapping == {"s1": "b.mp4", "s2": "a.mp4"}
    assert len(map_hash) == 64

    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["assignments"][1]["duration_bucket"] = 2
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="crosses an explicit duration bucket"):
        _load_validated_mismatch_map(
            path,
            samples=_samples(),
            manifest_hash=manifest_hash,
            video_root=tmp_path,
        )


def test_model_slug_is_stable_and_distinguishes_served_models() -> None:
    q4 = _stable_model_slug("Qwen/Qwen3.5-4B")
    q9 = _stable_model_slug("Qwen/Qwen3.5-9B")
    assert q4 == _stable_model_slug("Qwen/Qwen3.5-4B")
    assert q4 != q9
    assert "4b" in q4
    assert "9b" in q9
