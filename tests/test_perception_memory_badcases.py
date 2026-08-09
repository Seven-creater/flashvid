from __future__ import annotations

import json
from pathlib import Path

import pytest

from flashvid_eval.perception_memory_badcases import (
    attach_target_diagnostics,
    pair_badcases,
    read_result_files,
    read_manifest_annotations,
    summarize_pairs,
    write_badcase_report,
)
from scripts.evaluate_mcq import _validate_perception_memory_diagnostics_gate


def _row(sample_id: str, prediction: str, *, calls: int, candidate: str = "A") -> dict:
    return {
        "dataset": "lvbench",
        "sample_id": sample_id,
        "answer": "A",
        "prediction": prediction,
        "candidate_answer": candidate,
        "tool_calls": [
            {
                "stage": "verification",
                "start_time": index * 10.0,
                "end_time": index * 10.0 + 5.0,
                "nframes": 8,
                "timestamps": [index * 10.0, index * 10.0 + 5.0],
                "frame_paths": [f"/frames/{index}.png"],
            }
            for index in range(calls)
        ],
        "request_trace": [
            {
                "stage": "verification",
                "assistant_content": f"turn {index}",
                "usage": {"total_tokens": 10},
            }
            for index in range(calls)
        ],
        "visual_tokens": calls * 100,
        "total_tokens": calls * 120,
    }


def test_pair_badcases_marks_early_stop_and_candidate_regression() -> None:
    untrained = {("lvbench", "x"): _row("x", "A", calls=3)}
    sft = {("lvbench", "x"): _row("x", "B", calls=1)}
    paired = pair_badcases(untrained, sft)
    assert paired[0]["flip_type"] == "untrained_correct_sft_wrong"
    assert paired[0]["failure_modes"] == [
        "possible_early_stop",
        "correct_candidate_regressed",
    ]
    assert paired[0]["untrained_tool_trace"][0]["nframes_returned"] == 1
    assert paired[0]["sft_request_trace"][0]["assistant_output"] == "turn 0"
    summary = summarize_pairs(paired)
    assert summary["datasets"]["lvbench"]["untrained_correct"] == 1
    assert summary["datasets"]["lvbench"]["sft_correct"] == 0
    assert summary["status"] == "failed"
    assert summary["scope_passed"] is False
    assert set(summary["flip_totals"]) == {
        "untrained_correct_sft_wrong",
        "untrained_wrong_sft_correct",
        "both_correct",
        "both_wrong",
    }
    assert set(summary["failure_mode_totals"]) == {
        "localization",
        "visual_fact_extraction",
        "cross_interval_memory",
        "incomplete_evidence_early_stop",
        "judging",
        "candidate_gate",
        "engineering",
    }


def test_read_results_rejects_duplicate_identity(tmp_path: Path) -> None:
    path = tmp_path / "rows.jsonl"
    row = _row("x", "A", calls=1)
    path.write_text(json.dumps(row) + "\n" + json.dumps(row) + "\n")
    with pytest.raises(ValueError, match="duplicate result identity"):
        read_result_files([path])


def test_pair_badcases_requires_identical_scope() -> None:
    with pytest.raises(ValueError, match="paired result IDs differ"):
        pair_badcases({("lvbench", "x"): _row("x", "A", calls=1)}, {})


def test_write_report_creates_auditable_outputs(tmp_path: Path) -> None:
    rows = pair_badcases(
        {("lvbench", "x"): _row("x", "B", calls=2, candidate="B")},
        {("lvbench", "x"): _row("x", "A", calls=2, candidate="B")},
    )
    summary = summarize_pairs(rows)
    write_badcase_report(tmp_path, rows, summary)
    assert (tmp_path / "paired_badcases.jsonl").is_file()
    assert json.loads((tmp_path / "summary.json").read_text())["samples"] == 1
    assert "Untrained" in (tmp_path / "badcases.md").read_text()


def test_posthoc_manifest_join_computes_target_hit_without_model_leak(
    tmp_path: Path,
) -> None:
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(
        json.dumps(
            {
                "dataset": "lvbench",
                "sample_id": "x",
                "video": "x.mp4",
                "question": "q",
                "choices": {"A": "a", "B": "b"},
                "answer": "A",
                "metadata": {"time_reference": "00:09-00:11"},
            }
        )
        + "\n"
    )
    rows = pair_badcases(
        {("lvbench", "x"): _row("x", "A", calls=2)},
        {("lvbench", "x"): _row("x", "A", calls=2)},
    )
    annotations = read_manifest_annotations([manifest])
    attach_target_diagnostics(rows, annotations)
    assert rows[0]["sft_target_frame_count"] == 1
    assert rows[0]["sft_funnel"]["target_hit"] is True


def test_diagnostics_gate_requires_exact_test300_scope(tmp_path: Path) -> None:
    rows = []
    for dataset in ("lvbench", "lsdbench", "cgbench"):
        for index in range(100):
            untrained = _row(str(index), "A", calls=1)
            tuned = _row(str(index), "A", calls=1)
            untrained["dataset"] = dataset
            tuned["dataset"] = dataset
            rows.extend(pair_badcases({(dataset, str(index)): untrained}, {(dataset, str(index)): tuned}))
    summary = summarize_pairs(rows)
    path = tmp_path / "summary.json"
    path.write_text(json.dumps(summary), encoding="utf-8")
    assert summary["status"] == "passed"
    assert len(_validate_perception_memory_diagnostics_gate(path)) == 64

    summary["datasets"]["lvbench"]["samples"] = 99
    path.write_text(json.dumps(summary), encoding="utf-8")
    with pytest.raises(ValueError, match="100 samples per dataset"):
        _validate_perception_memory_diagnostics_gate(path)
