from __future__ import annotations

from flashvid_eval.qwen_reporting import (
    choose_best_framework,
    paired_compare,
    promotion_decision,
    summarize,
)


def test_summary_and_paired_comparison_keep_failures_visible() -> None:
    baseline = [
        {"sample_id": "1", "prediction": "A", "correct": True, "total_tokens": 100},
        {"sample_id": "2", "prediction": "B", "correct": False, "total_tokens": 100},
        {"sample_id": "3", "prediction": None, "correct": False, "error": "parse"},
    ]
    candidate = [
        {"sample_id": "1", "prediction": "B", "correct": False},
        {"sample_id": "2", "prediction": "C", "correct": True},
        {"sample_id": "3", "prediction": "A", "correct": True},
    ]
    report = summarize(baseline)
    assert report["correct"] == 1
    assert report["errors"] == 1
    assert report["mean_total_tokens"] == 100
    paired = paired_compare(baseline, candidate)
    assert paired == {
        "shared": 3,
        "common_valid": 2,
        "baseline_correct": 1,
        "candidate_correct": 1,
        "gain": 0,
        "corrected": 1,
        "regressed": 1,
    }


def test_promotion_requires_mean_gain_and_two_seed_wins() -> None:
    passed = promotion_decision(
        {17: 60, 42: 60, 73: 60},
        {17: 63, 42: 62, 73: 61},
    )
    assert passed.accepted
    assert passed.mean_gain == 2
    failed = promotion_decision(
        {17: 60, 42: 60, 73: 60},
        {17: 66, 42: 59, 73: 59},
    )
    assert not failed.accepted
    assert failed.seed_wins == 1


def test_best_framework_tiebreaks_variance_regressions_then_tokens() -> None:
    best = choose_best_framework(
        [
            {
                "id": "unstable",
                "mean_accuracy": 0.6,
                "accuracy_stdev": 0.02,
                "failure_rate": 0,
            },
            {
                "id": "stable-expensive",
                "mean_accuracy": 0.6,
                "accuracy_stdev": 0.01,
                "regressed": 1,
                "mean_total_tokens": 200,
                "failure_rate": 0,
            },
            {
                "id": "stable-cheap",
                "mean_accuracy": 0.6,
                "accuracy_stdev": 0.01,
                "regressed": 1,
                "mean_total_tokens": 100,
                "failure_rate": 0,
            },
        ]
    )
    assert best["id"] == "stable-cheap"
