from __future__ import annotations

import pytest

from flashvid_eval.qwen_final_reporting import (
    build_final_report,
    exact_mcnemar_p_value,
    paired_method_compare,
    summarize_method,
)


def _row(sample_id: str, *, correct: bool, dataset: str = "lvbench"):
    return {
        "_dataset_identity": dataset,
        "sample_id": sample_id,
        "prediction": "A" if correct else "B",
        "answer": "A",
        "correct": correct,
        "prompt_tokens": 80,
        "completion_tokens": 20,
        "reasoning_tokens": 10,
        "visual_tokens": 50,
        "total_tokens": 100,
        "latency_s": 1.0,
        "annotation_leak_check": "passed",
    }


def test_final_summary_and_paired_mcnemar_use_common_valid_samples() -> None:
    baseline = [_row("a", correct=True), _row("b", correct=False), _row("c", correct=True)]
    candidate = [_row("a", correct=False), _row("b", correct=True), _row("c", correct=True)]
    candidate[2]["data_unavailable"] = True
    summary = summarize_method(baseline)
    assert summary["nominal"] == {"denominator": 3, "correct": 2, "accuracy": 2 / 3}
    assert summary["mean_total_tokens"] == 100
    paired = paired_method_compare(baseline, candidate)
    assert paired["common_valid"] == 2
    assert paired["corrected"] == paired["regressed"] == 1
    assert paired["gain"] == 0
    assert paired["mcnemar_exact_p"] == 1.0


def test_exact_mcnemar_rejects_negative_counts() -> None:
    assert exact_mcnemar_p_value(0, 0) == 1.0
    with pytest.raises(ValueError):
        exact_mcnemar_p_value(-1, 2)


def test_final_report_without_sft_still_reports_untrained_matrix(
    monkeypatch,
) -> None:
    wrong = [_row("sample", correct=False)]
    correct = [_row("sample", correct=True)]
    methods = {
        "q9_no_video": wrong,
        "q9_direct_uniform64": wrong,
        "q9_eva_clean": wrong,
        "q9_best_untrained": correct,
        "q4_no_video": wrong,
        "q4_direct_uniform64": correct,
        "q4_eva_clean": wrong,
        "q4_best_untrained": correct,
    }
    monkeypatch.setattr(
        "flashvid_eval.qwen_final_reporting.load_final_matrix",
        lambda _paths, _config_hash: (methods, {"plans": []}),
    )
    report, _rows = build_final_report([], "a" * 64)
    assert report["sft_evaluated"] is False
    assert report["comparisons"]["q9_direct_vs_untrained_agent"]["gain"] == 1
    assert "q9_untrained_vs_sft" not in report["comparisons"]
    assert report["sft_token_ratios"] == {"total": None, "visual": None}
    assert report["overall_success"] is False
