from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from flashvid_eval.qwen_mismatch_maps import (
    build_mismatch_map_payload,
    write_frozen_json,
)


def _manifest(tmp_path: Path) -> tuple[Path, str]:
    rows = [
        {
            "dataset": "demo",
            "sample_id": "s1",
            "video": "a.mp4",
            "question": "PRIVATE QUESTION ONE",
            "choices": {"A": "x", "B": "y"},
            "answer": "A",
            "metadata": {"time_range": "PRIVATE TIME"},
        },
        {
            "dataset": "demo",
            "sample_id": "s2",
            "video": "b.mp4",
            "question": "PRIVATE QUESTION TWO",
            "choices": {"A": "x", "B": "y"},
            "answer": "B",
            "metadata": {"clue_intervals": "PRIVATE CLUE"},
        },
        {
            "dataset": "demo",
            "sample_id": "s3",
            "video": "c.mp4",
            "question": "PRIVATE QUESTION THREE",
            "choices": {"A": "x", "B": "y"},
            "answer": "A",
            "metadata": {},
        },
    ]
    path = tmp_path / "manifest.jsonl"
    text = "".join(json.dumps(row) + "\n" for row in rows)
    path.write_text(text, encoding="utf-8")
    return path, hashlib.sha256(path.read_bytes()).hexdigest()


def test_builds_same_bucket_wrong_video_map_without_private_fields(
    tmp_path: Path,
) -> None:
    manifest, manifest_hash = _manifest(tmp_path)
    video_root = tmp_path / "videos"
    video_root.mkdir()
    for name in ("a.mp4", "b.mp4", "c.mp4"):
        (video_root / name).write_bytes(b"video")
    durations = {"a.mp4": 100.0, "b.mp4": 200.0, "c.mp4": 1000.0}

    payload = build_mismatch_map_payload(
        dataset="demo",
        manifest_path=manifest,
        expected_manifest_sha256=manifest_hash,
        video_root=video_root,
        seed=17,
        experiment_config_sha256="a" * 64,
        probe=lambda path: {"duration": durations[path.name]},
    )

    assert payload["mapping"] == {"s1": "b.mp4", "s2": "a.mp4"}
    assert payload["unmapped_sample_ids"] == ["s3"]
    serialized = json.dumps(payload)
    assert "PRIVATE" not in serialized
    assert all(
        row["source_video"] != row["target_video"]
        for row in payload["assignments"]
    )


def test_frozen_writer_is_idempotent_and_refuses_changed_payload(
    tmp_path: Path,
) -> None:
    path = tmp_path / "frozen" / "map.json"
    payload = {"schema_version": 1, "mapping": {"s1": "b.mp4"}}
    first = write_frozen_json(path, payload)
    second = write_frozen_json(path, payload)
    assert first == second
    with pytest.raises(RuntimeError, match="refusing to overwrite"):
        write_frozen_json(path, {"schema_version": 1, "mapping": {}})


def test_manifest_hash_mismatch_is_rejected_before_video_probe(tmp_path: Path) -> None:
    manifest, _manifest_hash = _manifest(tmp_path)
    with pytest.raises(RuntimeError, match="manifest SHA-256 mismatch"):
        build_mismatch_map_payload(
            dataset="demo",
            manifest_path=manifest,
            expected_manifest_sha256="0" * 64,
            video_root=tmp_path,
            seed=17,
            experiment_config_sha256="a" * 64,
        )
