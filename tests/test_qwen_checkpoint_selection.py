from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest

from flashvid_eval import qwen_checkpoint_selection as selection


def _aggregate(*, phase: str, correct: int, total_tokens: float, visual_tokens: float):
    rows = []
    for index in range(150):
        is_correct = index < correct
        rows.append(
            {
                "_dataset_identity": ("lvbench", "lsdbench", "cgbench")[index // 50],
                "sample_id": f"sample-{index % 50}",
                "prediction": "A" if is_correct else "B",
                "answer": "A",
                "correct": is_correct,
            }
        )
    return {
        "schema_version": 1,
        "phase": phase,
        "manifest_sha256": "m" * 64,
        "sample_ids_sha256": "s" * 64,
        "aggregate": {
            "total": 150,
            "correct": correct,
            "failure_rate": 0.0,
            "annotation_leak": 0,
            "mean_total_tokens": total_tokens,
            "mean_visual_tokens": visual_tokens,
        },
        "rows": rows,
        "frozen_checkpoint": (
            {
                "path": f"/{correct}.json",
                "sha256": f"{correct:064x}",
                "checkpoint_id": f"epoch-{correct}",
                "epoch": 1,
            }
            if phase == "sft_dev"
            else None
        ),
    }


def test_sft_checkpoint_requires_accuracy_and_both_token_gates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    teacher = _aggregate(
        phase="teacher_dev", correct=90, total_tokens=1000, visual_tokens=600
    )
    passing = _aggregate(
        phase="sft_dev", correct=91, total_tokens=700, visual_tokens=420
    )
    equal_accuracy = _aggregate(
        phase="sft_dev", correct=90, total_tokens=500, visual_tokens=300
    )
    equal_accuracy["frozen_checkpoint"]["checkpoint_id"] = "epoch-equal"

    def fake_aggregate(path: Path, _config_hash: str, expected_phase: str):
        if expected_phase == "teacher_dev":
            return deepcopy(teacher)
        return deepcopy(passing if path.name == "passing" else equal_accuracy)

    monkeypatch.setattr(selection, "_aggregate_plan", fake_aggregate)
    report = selection.select_sft_checkpoint(
        config_sha256="a" * 64,
        teacher_plan=Path("teacher"),
        checkpoint_plans=[Path("passing"), Path("equal")],
    )
    assert report["status"] == "passed"
    assert report["selected"]["checkpoint_id"] == "epoch-91"
    rejected = next(
        point for point in report["checkpoints"] if point["checkpoint_id"] == "epoch-equal"
    )
    assert rejected["gate"]["conditions"]["accuracy_strictly_higher"] is False


def test_sft_checkpoint_report_blocks_when_no_checkpoint_is_dual_gain(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    teacher = _aggregate(
        phase="teacher_dev", correct=90, total_tokens=1000, visual_tokens=600
    )
    candidate = _aggregate(
        phase="sft_dev", correct=91, total_tokens=701, visual_tokens=420
    )

    def fake_aggregate(_path: Path, _config_hash: str, expected_phase: str):
        return deepcopy(teacher if expected_phase == "teacher_dev" else candidate)

    monkeypatch.setattr(selection, "_aggregate_plan", fake_aggregate)
    report = selection.select_sft_checkpoint(
        config_sha256="a" * 64,
        teacher_plan=Path("teacher"),
        checkpoint_plans=[Path("candidate")],
    )
    assert report["status"] == "blocked"
    assert report["selected"] is None
