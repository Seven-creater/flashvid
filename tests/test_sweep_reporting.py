from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from flashvid_eval.sweep_reporting import (
    budget_distribution,
    bootstrap_mean_ci,
    build_sweep_report,
    exact_mcnemar,
    extract_frozen_budget_distribution,
    load_sweep_config,
    render_markdown,
    candidate_only_rows,
)
from scripts.summarize_budget_sweep import main


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )


def _candidate_rows() -> list[dict[str, object]]:
    return [
        {
            "dataset": "dev",
            "sample_id": str(index),
            "answer": "A",
            "candidate_answer": "A" if index < 63 else "B",
            "candidate_source": "normalized",
        }
        for index in range(150)
    ]


def _method_rows(
    correct: int,
    retained: float,
    *,
    parse_failures: int = 0,
    fallbacks: int = 0,
    ratio: float = 0.1,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for index in range(150):
        candidate = "A" if index < 63 else "B"
        prediction = "A" if index < correct else "B"
        rows.append(
            {
                "dataset": "dev",
                "sample_id": str(index),
                "answer": "A",
                "candidate_answer": candidate,
                "prediction": prediction,
                "final_prediction": prediction,
                "retained_visual_tokens": retained,
                "raw_visual_tokens": retained * 2,
                "controller_model": "observed-model",
                "controller_usage": {"total_tokens": 100},
                "perception_usage": {"total_tokens": 200},
                "usage": {"total_tokens": 300},
                "latency_s": 2.0,
                "tool_steps": [{"retention_ratio": ratio}],
                "parse_error": "invalid_controller_output" if index < parse_failures else None,
                "failure_stage": "controller_parse" if index < parse_failures else None,
                "fallback_to_candidate": index < fallbacks,
            }
        )
    return rows


def _config(tmp_path: Path) -> Path:
    _write_jsonl(tmp_path / "candidate.jsonl", _candidate_rows())
    _write_jsonl(tmp_path / "ck39.jsonl", _method_rows(63, 295.5))
    _write_jsonl(
        tmp_path / "untrained.jsonl",
        _method_rows(65, 660.8, parse_failures=82, fallbacks=74),
    )
    _write_jsonl(tmp_path / "q9_fixed10.jsonl", _method_rows(61, 276.2))
    config = {
        "output_dir": "report",
        "bootstrap": {"seed": 42, "samples": 100, "confidence": 0.95},
        "methods": [
            {
                "name": "candidate_only",
                "kind": "candidate_only",
                "controller_model": "frozen-qwen3.5-9b-direct-candidate",
                "paths": {"dev": "candidate.jsonl"},
            },
            {
                "name": "checkpoint_39",
                "controller_group": "q4_controller",
                "controller_model": "Qwen3.5-4B-checkpoint-39",
                "paths": {"dev": "ck39.jsonl"},
            },
            {
                "name": "q4_untrained",
                "controller_group": "q4_controller",
                "controller_model": "Qwen3.5-4B",
                "paths": {"dev": "untrained.jsonl"},
            },
            {
                "name": "q9_fixed10",
                "controller_group": "q9_controller",
                "controller_model": "Qwen3.5-9B",
                "paths": {"dev": "q9_fixed10.jsonl"},
            },
        ],
    }
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    return path


def _point(report: dict[str, object], group: str, name: str) -> dict[str, object]:
    pareto = report["pareto"]
    assert isinstance(pareto, dict)
    points = pareto[group]["points"]
    return next(point for point in points if point["method"] == name)


def test_old_dev_numbers_are_grouped_and_candidate_only_is_global_origin(
    tmp_path: Path,
) -> None:
    report = build_sweep_report(_config(tmp_path))
    candidate = report["methods"]["candidate_only"]["aggregate"]
    assert candidate["correct"] == 63
    assert candidate["records"] == 150
    assert candidate["tokens"]["retained_visual"]["mean"] == 0.0

    ck39 = _point(report, "system", "checkpoint_39")
    assert ck39["pareto_efficient"] is False
    assert "candidate_only" in ck39["dominated_by"]
    untrained = _point(report, "system", "q4_untrained")
    assert untrained["pareto_efficient"] is True
    assert report["methods"]["q4_untrained"]["aggregate"]["correct"] == 65
    assert (
        report["methods"]["q4_untrained"]["aggregate"]["tokens"]
        ["retained_visual"]["mean"]
        == pytest.approx(660.8)
    )
    assert report["methods"]["q4_untrained"]["aggregate"]["parse_failure_count"] == 82
    assert report["methods"]["q4_untrained"]["aggregate"]["fallback_to_candidate"] == 74

    assert "candidate_only" not in {
        point["method"] for point in report["pareto"]["q4_controller"]["points"]
    }
    assert set(report["pareto"]["q4_controller"]["non_dominated_methods"]) == {
        "checkpoint_39",
        "q4_untrained",
    }
    assert report["pareto"]["q9_controller"]["non_dominated_methods"] == [
        "q9_fixed10"
    ]


def test_common_valid_exact_mcnemar_and_change_audit(tmp_path: Path) -> None:
    report = build_sweep_report(_config(tmp_path))
    comparison = report["paired_common_valid"][
        "system:q4_untrained__vs__candidate_only"
    ]
    assert comparison["paired_common_valid"] == 150
    assert comparison["wins_baseline_wrong_method_correct"] == 2
    assert comparison["losses_baseline_correct_method_wrong"] == 0
    assert comparison["mcnemar_exact_two_sided_p"] == 0.5
    candidate = report["methods"]["q4_untrained"]["aggregate"]["candidate"]
    assert candidate["changes_fixed"] == 2
    assert candidate["changes_harmed"] == 0

    direct = _candidate_rows()[:3]
    synthesized = [
        {
            **row,
            "prediction": row["candidate_answer"],
            "final_prediction": row["candidate_answer"],
        }
        for row in direct
    ]
    assert exact_mcnemar(synthesized, synthesized)["mcnemar_exact_two_sided_p"] == 1.0


def test_bootstrap_is_reproducible_and_markdown_names_models(tmp_path: Path) -> None:
    first = bootstrap_mean_ci([1.0, 2.0, 3.0], seed=17, samples=100)
    second = bootstrap_mean_ci([1.0, 2.0, 3.0], seed=17, samples=100)
    assert first == second
    report = build_sweep_report(_config(tmp_path))
    markdown = render_markdown(report)
    assert "frozen-qwen3.5-9b-direct-candidate" in markdown
    assert "Qwen3.5-4B-checkpoint-39" in markdown
    assert "Candidate-only therefore has zero incremental visual cost" in markdown
    assert "Executed budget distributions" in markdown


def test_budget_distribution_can_be_frozen_for_cost_matched_random() -> None:
    rows = [
        {
            "sample_id": "one",
            "retained_visual_tokens": 10,
            "tool_steps": [
                {"retention_ratio": 0.10},
                {"retention_ratio": 0.25},
            ],
        },
        {
            "sample_id": "two",
            "retained_visual_tokens": 20,
            "budget_sequence": [0.50, 1.00],
        },
    ]
    distribution = budget_distribution(rows)
    assert distribution["counts"] == {
        "R010": 1,
        "R025": 1,
        "R050": 1,
        "R100": 1,
    }
    assert distribution["probabilities"] == {
        "R010": 0.25,
        "R025": 0.25,
        "R050": 0.25,
        "R100": 0.25,
    }
    assert distribution["mean_retained_visual_tokens_per_sample"] == 15
    frozen = extract_frozen_budget_distribution(rows, source_method="adaptive_best")
    assert frozen["source_method"] == "adaptive_best"
    assert len(frozen["distribution_sha256"]) == 64
    assert frozen == extract_frozen_budget_distribution(
        rows, source_method="adaptive_best"
    )


def test_frozen_distribution_rejects_unsupported_budget() -> None:
    with pytest.raises(ValueError, match="unsupported"):
        extract_frozen_budget_distribution(
            [{"tool_steps": [{"retention_ratio": 0.33}]}],
            source_method="bad",
        )


def test_cli_writes_json_and_markdown(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config_path = _config(tmp_path)
    monkeypatch.setattr(sys, "argv", ["summarize_budget_sweep.py", "--config", str(config_path)])
    assert main() == 0
    payload = json.loads((tmp_path / "report" / "summary.json").read_text(encoding="utf-8"))
    assert payload["methods"]["candidate_only"]["aggregate"]["correct"] == 63
    assert (tmp_path / "report" / "summary.md").is_file()


def test_controller_model_is_required(tmp_path: Path) -> None:
    path = tmp_path / "bad.json"
    path.write_text(
        json.dumps({"methods": [{"name": "bad", "paths": {}}]}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="controller_model"):
        load_sweep_config(path)


def test_candidate_only_never_uses_visual_prediction_as_candidate() -> None:
    rows = candidate_only_rows(
        [
            {
                "dataset": "dev",
                "sample_id": "missing",
                "answer": "A",
                "candidate_answer": None,
                "final_prediction": "A",
                "prediction": "A",
            }
        ]
    )
    assert rows[0]["final_prediction"] is None
    assert rows[0]["correct"] is False


def test_declared_incomplete_method_is_excluded_from_pareto(tmp_path: Path) -> None:
    _write_jsonl(tmp_path / "partial.jsonl", _method_rows(100, 100.0)[:-1])
    config = {
        "expected_count": {"dev": 150},
        "bootstrap": {"samples": 10},
        "methods": [
            {
                "name": "partial",
                "controller_group": "q9_controller",
                "controller_model": "Qwen3.5-9B",
                "paths": {"dev": "partial.jsonl"},
            }
        ],
    }
    path = tmp_path / "partial_config.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    report = build_sweep_report(path)
    point = report["pareto"]["q9_controller"]["points"][0]
    assert point["selection_eligible"] is False
    assert point["pareto_efficient"] is False
    assert point["exclusion_reason"] == "incomplete_result"
    assert report["pareto"]["q9_controller"]["non_dominated_methods"] == []


def test_cost_match_binds_frozen_base_summary_and_rejects_source_change(
    tmp_path: Path,
) -> None:
    target_path = tmp_path / "adaptive.jsonl"
    _write_jsonl(target_path, _method_rows(70, 300.0, ratio=0.25))
    base_config = {
        "expected_count": {"dev": 150},
        "bootstrap": {"samples": 10},
        "matched_random_source_method": "adaptive",
        "methods": [
            {
                "name": "adaptive",
                "controller_group": "q9_controller",
                "controller_model": "Qwen3.5-9B",
                "paths": {"dev": "adaptive.jsonl"},
            }
        ],
    }
    base_config_path = tmp_path / "base_config.json"
    base_config_path.write_text(json.dumps(base_config), encoding="utf-8")
    base_report_path = tmp_path / "base_summary.json"
    base_report_path.write_text(
        json.dumps(build_sweep_report(base_config_path)), encoding="utf-8"
    )
    base_sha = __import__("hashlib").sha256(base_report_path.read_bytes()).hexdigest()

    _write_jsonl(tmp_path / "random.jsonl", _method_rows(60, 301.0, ratio=0.25))
    combined = {
        "expected_count": {"dev": 150},
        "bootstrap": {"samples": 10},
        "matched_random_from_summary": {
            "path": "base_summary.json",
            "sha256": base_sha,
        },
        "cost_match": {
            "target_from_matched_random": True,
            "candidate_methods": ["random"],
        },
        "methods": [
            {
                "name": "adaptive",
                "controller_group": "q9_controller",
                "controller_model": "Qwen3.5-9B",
                "paths": {"dev": "adaptive.jsonl"},
            },
            {
                "name": "random",
                "controller_group": "q9_controller",
                "controller_model": "Qwen3.5-9B",
                "paths": {"dev": "random.jsonl"},
            },
        ],
    }
    combined_path = tmp_path / "combined.json"
    combined_path.write_text(json.dumps(combined), encoding="utf-8")
    report = build_sweep_report(combined_path)
    assert report["matched_random"]["source_policy_id"] == "adaptive"
    assert report["matched_random"]["bound_base_summary_sha256"] == base_sha
    assert report["cost_matched_random"]["selected_method"] == "random"

    changed = _method_rows(70, 300.0, ratio=0.25)
    changed[0]["raw_response"] = "source changed without changing distribution"
    _write_jsonl(target_path, changed)
    with pytest.raises(RuntimeError, match="source result SHA-256 changed"):
        build_sweep_report(combined_path)


def test_report_freezes_adaptive_distribution_and_selects_random_by_cost(
    tmp_path: Path,
) -> None:
    target_rows = _method_rows(70, 300.0)
    for index, row in enumerate(target_rows):
        row["tool_steps"] = [
            {"retention_ratio": 0.10 if index % 2 == 0 else 0.50}
        ]
    _write_jsonl(tmp_path / "target.jsonl", target_rows)
    _write_jsonl(tmp_path / "seed17.jsonl", _method_rows(90, 360.0))
    _write_jsonl(tmp_path / "seed42.jsonl", _method_rows(20, 305.0))
    _write_jsonl(tmp_path / "seed73.jsonl", _method_rows(80, 280.0))
    config = {
        "output_dir": "report",
        "bootstrap": {"seed": 42, "samples": 20},
        "matched_random_source_method": "adaptive",
        "cost_match": {
            "target_method": "adaptive",
            "candidate_methods": ["seed17", "seed42", "seed73"],
        },
        "methods": [
            {
                "name": name,
                "controller_group": "q9_controller",
                "controller_model": "Qwen3.5-9B",
                "paths": {"dev": filename},
            }
            for name, filename in (
                ("adaptive", "target.jsonl"),
                ("seed17", "seed17.jsonl"),
                ("seed42", "seed42.jsonl"),
                ("seed73", "seed73.jsonl"),
            )
        ],
    }
    config_path = tmp_path / "matched.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    report = build_sweep_report(config_path)

    assert report["matched_random"]["source_policy_id"] == "adaptive"
    assert report["matched_random"]["budget_distribution"] == {
        "0.10": 0.5,
        "0.25": 0.0,
        "0.50": 0.5,
        "1.00": 0.0,
    }
    assert report["cost_matched_random"]["selected_method"] == "seed42"
    assert report["cost_matched_random"]["accuracy_used_for_selection"] is False
    assert report["methods"]["adaptive"]["source_sha256"]
