from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from flashvid_eval.perception_memory_eva import explicit_time_rescue_request
from scripts.run_perception_memory_explicit_time_rescue import (
    _canonical_sha256,
    _run_one,
    _validate_frozen_scope,
    _validate_manifest_counts,
    _validate_manifest_row,
    _rescue_trajectory_id,
)


def _repair_row() -> dict[str, object]:
    source = {
        "dataset": "lvbench",
        "sample_id": "long-video",
        "trajectory_id": "lvbench:long-video:base:0",
        "candidate_answer": "A",
        "scoring_deferred": True,
        "train600_manifest_sha256": "3" * 64,
        "tool_steps": [
            {
                "start_time": 64.02,
                "end_time": 66.02,
                "frame_paths": ["/cache/wrong.jpg"],
            },
            {
                "start_time": 3901.0,
                "end_time": 3903.0,
                "frame_paths": ["/cache/correct.jpg"],
            },
        ],
    }
    return {
        "dataset": "lvbench",
        "sample_id": "long-video",
        "source_trajectory_id": source["trajectory_id"],
        "source_row_sha256": _canonical_sha256(source),
        "source_file_sha256": "4" * 64,
        "base_file_sha256": "5" * 64,
        "reason": "explicit_time_source_interval_mismatch",
        "parsed_time_range": [3901.0, 3903.0],
        "failed_step_index": 0,
        "failed_step_requested_interval": [64.02, 66.02],
        "source_requested_intervals": [[64.02, 66.02], [3901.0, 3903.0]],
        "public_sample": {
            "dataset": "lvbench",
            "sample_id": "long-video",
            "video": "long-video.mp4",
            "question": "What happens at 65:02?",
            "choices": {"A": "First", "B": "Second"},
        },
        "source_row": source,
    }


def test_repair_manifest_row_binds_public_time_source_and_frozen_candidate() -> None:
    sample, intervals = _validate_manifest_row(_repair_row())

    assert sample.question == "What happens at 65:02?"
    assert sample.candidate_answer == "A"
    assert intervals == ((64.02, 66.02), (3901.0, 3903.0))
    assert not hasattr(sample, "answer")
    assert not hasattr(sample, "metadata")


def test_repair_manifest_row_rejects_private_or_forged_inputs() -> None:
    private = _repair_row()
    private["public_sample"] = {
        **private["public_sample"],  # type: ignore[arg-type]
        "time_range": "SECRET",
    }
    with pytest.raises(ValueError, match="annotation-free public_sample"):
        _validate_manifest_row(private)

    forged = _repair_row()
    forged["source_row_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="source_row_sha256"):
        _validate_manifest_row(forged)


def test_repair_manifest_row_rejects_tampered_failed_step_lineage() -> None:
    tampered_interval = _repair_row()
    tampered_interval["failed_step_requested_interval"] = [3901.0, 3903.0]
    with pytest.raises(
        ValueError, match="failed_step_requested_interval differs"
    ):
        _validate_manifest_row(tampered_interval)

    tampered_index = _repair_row()
    tampered_index["failed_step_index"] = 1
    with pytest.raises(
        ValueError, match="failed_step_requested_interval differs"
    ):
        _validate_manifest_row(tampered_index)


def test_runtime_proves_mismatch_from_failed_step_not_later_correct_step(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    captured: dict[str, object] = {}

    class FakeEvaluator:
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            pass

        def run(self, sample: object, **kwargs: object) -> dict[str, object]:
            captured.update(kwargs)
            _request, audit = explicit_time_rescue_request(
                sample.question, 4000.0, kwargs["rescue_source_requested_intervals"]
            )
            return {
                "error": None,
                "explicit_time_rescue_audit": audit,
            }

    monkeypatch.setattr(
        "scripts.run_perception_memory_explicit_time_rescue.PerceptionMemoryEvaEvaluator",
        FakeEvaluator,
    )
    row = _repair_row()
    result = _run_one(
        row,
        base_url="http://127.0.0.1:8200/v1",
        api_key="no",
        model="Qwen3.5-9B",
        video_root=tmp_path,
        frame_root=tmp_path / "frames",
        timeout=1.0,
        local_media_paths=True,
        max_turns=6,
        seed=42,
        manifest_sha256="a" * 64,
        frozen_scope_sha256="b" * 64,
    )

    assert captured["rescue_source_requested_intervals"] == ((64.02, 66.02),)
    assert result["failed_step_index"] == 0
    assert result["failed_step_requested_interval"] == [64.02, 66.02]
    assert result["source_requested_intervals_audit"] == [
        [64.02, 66.02],
        [3901.0, 3903.0],
    ]


def test_repair_scope_requires_37_trajectories_across_6_samples() -> None:
    rows = [
        {
            "dataset": "lvbench",
            "sample_id": f"sample-{index % 6}",
            "source_trajectory_id": f"source-{index}",
        }
        for index in range(37)
    ]

    _validate_manifest_counts(rows, expected_rows=37, expected_samples=6)
    rescue_ids = [
        _rescue_trajectory_id(str(row["source_trajectory_id"])) for row in rows
    ]
    assert len(rescue_ids) == len(set(rescue_ids)) == 37
    assert all(item.endswith(":explicit_time_rescue_v1") for item in rescue_ids)

    with pytest.raises(ValueError, match="37 rows"):
        _validate_manifest_counts(rows[:-1], expected_rows=37, expected_samples=6)
    five_sample_rows = [
        {**row, "sample_id": f"sample-{index % 5}"}
        for index, row in enumerate(rows)
    ]
    with pytest.raises(ValueError, match="6 unique samples"):
        _validate_manifest_counts(
            five_sample_rows, expected_rows=37, expected_samples=6
        )


def test_frozen_scope_cryptographically_binds_explicit_time_manifest(
    tmp_path: Path,
) -> None:
    manifest = tmp_path / "explicit_time_mismatch.jsonl"
    manifest.write_text(
        json.dumps(_repair_row(), ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    manifest_sha = hashlib.sha256(manifest.read_bytes()).hexdigest()
    scope = tmp_path / "frozen_scope.json"
    scope.write_text(
        json.dumps(
            {
                "artifacts": {
                    manifest.name: {
                        "sha256": manifest_sha,
                    }
                }
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    assert _validate_frozen_scope(scope, manifest, manifest_sha) == hashlib.sha256(
        scope.read_bytes()
    ).hexdigest()
    with pytest.raises(ValueError, match="manifest SHA-256 mismatch"):
        _validate_frozen_scope(scope, manifest, "0" * 64)
