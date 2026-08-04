from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.repair_direct_candidate_file import (
    merge_candidates,
    prepare_subset,
    read_jsonl,
    sha256_file,
)
from scripts.extract_hf_mirror_zip_member import HttpRangeReader


def _jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def _fixture(tmp_path: Path) -> tuple[Path, Path, list[dict]]:
    manifest = tmp_path / "manifest.jsonl"
    original = tmp_path / "original.jsonl"
    manifest_rows = [
        {"dataset": "lvbench", "sample_id": "ok", "video": "ok.mp4"},
        {"dataset": "lvbench", "sample_id": "missing", "video": "missing.mp4"},
    ]
    _jsonl(manifest, manifest_rows)
    _jsonl(
        original,
        [
            {
                "sample_id": "ok",
                "video": "ok.mp4",
                "prediction": "A",
                "baseline_mode": "direct",
                "sampling_id": "uniform32",
                "enable_thinking": False,
                "protocol_request": {"max_tokens": 512, "temperature": 0.0},
            },
            {
                "sample_id": "missing",
                "video": "missing.mp4",
                "prediction": None,
                "data_unavailable": True,
                "failure_class": "data_unavailable",
                "baseline_mode": "direct",
                "sampling_id": "uniform32",
                "enable_thinking": False,
                "protocol_request": {"max_tokens": 512, "temperature": 0.0},
            },
        ],
    )
    return manifest, original, manifest_rows


def _patch_row() -> dict:
    return {
        "sample_id": "missing",
        "video": "missing.mp4",
        "prediction": "B",
        "baseline_mode": "direct",
        "sampling_id": "uniform32",
        "enable_thinking": False,
        "protocol_request": {"max_tokens": 512, "temperature": 0.0},
        "model_parse_failure": False,
        "failure_class": None,
        "annotation_leak_check": "passed",
        "candidate_rerun": 0,
        "visual_usage_complete": True,
        "visual_tokens": 100,
        "total_tokens": 120,
    }


def test_prepare_and_merge_only_replace_original_unavailable_ids(tmp_path: Path) -> None:
    manifest, original, manifest_rows = _fixture(tmp_path)
    subset = tmp_path / "subset.jsonl"
    prepared = prepare_subset(
        manifest,
        original,
        subset,
        expected_manifest_sha256=sha256_file(manifest),
        expected_original_sha256=sha256_file(original),
        expected_count=1,
    )
    assert prepared["target_ids"] == ["missing"]
    assert read_jsonl(subset) == [manifest_rows[1]]

    patch = tmp_path / "patch.jsonl"
    _jsonl(patch, [_patch_row()])
    output = tmp_path / "repaired.jsonl"
    merged = merge_candidates(
        manifest,
        original,
        patch,
        output,
        expected_manifest_sha256=sha256_file(manifest),
        expected_original_sha256=sha256_file(original),
        expected_count=1,
    )
    assert merged["rows"] == 2
    rows = read_jsonl(output)
    assert [row["sample_id"] for row in rows] == ["ok", "missing"]
    assert rows[0]["prediction"] == "A"
    assert rows[1]["prediction"] == "B"


def test_merge_rejects_nonvisual_or_wrong_sample_patch(tmp_path: Path) -> None:
    manifest, original, _ = _fixture(tmp_path)
    patch = tmp_path / "patch.jsonl"
    bad = _patch_row()
    bad["visual_usage_complete"] = False
    _jsonl(patch, [bad])
    with pytest.raises(ValueError, match="Token accounting"):
        merge_candidates(
            manifest,
            original,
            patch,
            tmp_path / "out.jsonl",
            expected_manifest_sha256=sha256_file(manifest),
            expected_original_sha256=sha256_file(original),
            expected_count=1,
        )


def test_range_extractor_rejects_non_mirror_sources_before_network() -> None:
    with pytest.raises(ValueError, match="hf-mirror"):
        HttpRangeReader("https://huggingface.co/datasets/example/archive.zip")
