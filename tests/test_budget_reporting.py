from __future__ import annotations

import json
from pathlib import Path

from flashvid_eval.budget_reporting import (
    MethodSpec,
    choose_sft_checkpoint,
    exact_mcnemar,
    fixed_budget_curve,
    pareto_acceptance,
    read_jsonl_latest,
    score_checkpoint,
    summarize_records,
)


def _row(
    sample_id: str,
    *,
    answer: str = "A",
    prediction: str = "A",
    candidate: str = "B",
    retained: float = 100,
    ratio: float = 0.5,
) -> dict[str, object]:
    return {
        "dataset": "lvbench",
        "sample_id": sample_id,
        "answer": answer,
        "prediction": prediction,
        "final_prediction": prediction,
        "correct": prediction == answer,
        "candidate_answer": candidate,
        "candidate_source": "normalized",
        "candidate_rerun": 0,
        "retained_visual_tokens": retained,
        "raw_visual_tokens": 200,
        "latency_s": 2,
        "annotation_leak_check": "passed",
        "tool_steps": [
            {
                "retention_ratio": ratio,
                "retained_visual_tokens": retained,
                "raw_visual_tokens": 200,
            }
        ],
    }


def test_jsonl_latest_reports_duplicates_and_invalid_lines(tmp_path: Path) -> None:
    path = tmp_path / "run.jsonl"
    path.write_text(
        json.dumps(_row("one", prediction="B"))
        + "\nnot-json\n"
        + json.dumps(_row("one", prediction="A"))
        + "\n",
        encoding="utf-8",
    )
    loaded = read_jsonl_latest(path)
    assert loaded.records[0]["prediction"] == "A"
    assert loaded.duplicate_sample_ids == ("one",)
    assert loaded.invalid_lines == 1


def test_record_summary_audits_candidate_changes_and_budget() -> None:
    rows = [
        _row("fixed", answer="A", prediction="A", candidate="B", ratio=0.1),
        _row("harmed", answer="A", prediction="B", candidate="A", ratio=0.5),
        _row("same", answer="A", prediction="A", candidate="A", ratio=1.0),
    ]
    summary = summarize_records(rows, expected=3)
    assert summary["correct"] == 2
    assert summary["candidate"]["changes_fixed"] == 1
    assert summary["candidate"]["changes_harmed"] == 1
    assert summary["candidate"]["normalized_recovered"] == 3
    assert summary["budget"]["step_ratio_counts"] == {
        "0.10": 1,
        "0.50": 1,
        "1.00": 1,
    }


def test_record_summary_separates_source_data_unavailable_accuracy() -> None:
    available = _row("available", answer="A", prediction="A")
    unavailable = {
        **_row("unavailable", answer="A", prediction="A"),
        "data_unavailable": True,
        "error": "FileNotFoundError: source video unavailable",
    }
    summary = summarize_records([available, unavailable], expected=2)
    assert summary["accuracy"] == 1.0
    assert summary["data_unavailable"] == 1
    assert summary["available_records"] == 1
    assert summary["available_correct"] == 1
    assert summary["accuracy_available"] == 1.0
    assert summary["failures"] == 1
    assert summary["engineering_failures"] == 0
    assert summary["engineering_failure_rate"] == 0.0


def test_exact_mcnemar_uses_only_jointly_valid_rows() -> None:
    baseline = [
        _row("win", prediction="B"),
        _row("loss", prediction="A"),
        _row("same", prediction="A"),
        _row("invalid", prediction=""),
    ]
    method = [
        _row("win", prediction="A"),
        _row("loss", prediction="B"),
        _row("same", prediction="A"),
        _row("invalid", prediction="A"),
    ]
    result = exact_mcnemar(baseline, method)
    assert result["paired_common_valid"] == 3
    assert result["wins_baseline_wrong_method_correct"] == 1
    assert result["losses_baseline_correct_method_wrong"] == 1
    assert result["mcnemar_exact_two_sided_p"] == 1.0


def test_checkpoint_selection_prioritizes_correct_then_lower_cost() -> None:
    expected = {"lvbench": 1, "lsdbench": 1, "cgbench": 1}

    def score(name: str, wrong: bool, cost: float) -> dict[str, object]:
        by_dataset = {
            dataset: [
                {
                    **_row(
                        dataset,
                        prediction="B" if wrong and dataset == "cgbench" else "A",
                        retained=cost,
                        ratio={"lvbench": 0.1, "lsdbench": 0.5, "cgbench": 1.0}[dataset],
                    ),
                    "dataset": dataset,
                }
            ]
            for dataset in expected
        }
        return score_checkpoint(name, by_dataset, expected_counts=expected)

    scores = [
        score("epoch-1", True, 10),
        score("epoch-2", False, 30),
        score("epoch-3", False, 20),
    ]
    selection = choose_sft_checkpoint(scores)
    assert selection["selected_checkpoint"] == "epoch-3"
    assert [item["checkpoint"] for item in selection["ranking"]] == [
        "epoch-3",
        "epoch-2",
        "epoch-1",
    ]


def test_checkpoint_with_incomplete_dataset_is_not_selected() -> None:
    score = score_checkpoint(
        "epoch-1",
        {"lvbench": [_row("one")]},
        expected_counts={"lvbench": 1, "lsdbench": 1, "cgbench": 1},
    )
    assert score["status"] == "ineligible"
    assert choose_sft_checkpoint([score])["status"] == "pending"


def test_checkpoint_engineering_gates_reject_leaks_and_constant_budget() -> None:
    expected = {"lvbench": 1, "lsdbench": 1, "cgbench": 1}
    by_dataset = {
        dataset: [{**_row(dataset, ratio=0.5), "dataset": dataset}]
        for dataset in expected
    }
    by_dataset["lvbench"][0]["annotation_leak_check"] = "failed"
    score = score_checkpoint("epoch-1", by_dataset, expected_counts=expected)
    assert score["status"] == "ineligible"
    assert "aggregate:annotation_leak" in score["ineligible_reasons"]
    assert "aggregate:constant_single_budget_policy" in score["ineligible_reasons"]


def test_fixed_curve_and_pareto_acceptance() -> None:
    datasets_fixed = {
        dataset: summarize_records(
            [{**_row(dataset, retained=100), "dataset": dataset}],
            expected=1,
        )
        for dataset in ("lvbench", "lsdbench", "cgbench")
    }
    datasets_untrained = {
        dataset: summarize_records(
            [{**_row(dataset, retained=80), "dataset": dataset}],
            expected=1,
        )
        for dataset in ("lvbench", "lsdbench", "cgbench")
    }
    datasets_sft = {
        dataset: summarize_records(
            [
                {
                    **_row(dataset, retained=60, ratio=0.5 if dataset == "lvbench" else 0.1),
                    "dataset": dataset,
                }
            ],
            expected=1,
        )
        for dataset in ("lvbench", "lsdbench", "cgbench")
    }
    from flashvid_eval.budget_reporting import aggregate_dataset_metrics

    methods = {
        "fixed": {
            "datasets": datasets_fixed,
            "aggregate": aggregate_dataset_metrics(datasets_fixed),
        },
        "untrained": {
            "datasets": datasets_untrained,
            "aggregate": aggregate_dataset_metrics(datasets_untrained),
        },
        "sft": {
            "datasets": datasets_sft,
            "aggregate": aggregate_dataset_metrics(datasets_sft),
        },
    }
    specs = {
        "fixed": MethodSpec("fixed", "validation", {}, role="fixed_100", fixed_ratio=1.0),
        "untrained": MethodSpec("untrained", "validation", {}, role="untrained_dynamic"),
        "sft": MethodSpec("sft", "validation", {}, role="sft_dynamic"),
    }
    acceptance = pareto_acceptance(methods, specs, "validation")
    assert acceptance["passed"] is True
    curve = fixed_budget_curve(methods, specs, "validation")
    assert curve["points"][0]["pareto_efficient"] is True
