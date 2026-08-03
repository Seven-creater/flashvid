from __future__ import annotations

import math
import statistics
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .qwen_dev_selection import (
    DATASETS,
    DevRun,
    canonical_sha256,
    file_sha256,
    load_frozen_run_plan,
    load_task_result,
)


def _failed(row: Mapping[str, Any]) -> bool:
    return bool(row.get("error") or row.get("error_type") or row.get("parse_error"))


def _accessible(row: Mapping[str, Any]) -> bool:
    return not row.get("data_unavailable") and not row.get("control_unavailable")


def _valid(row: Mapping[str, Any]) -> bool:
    return _accessible(row) and not _failed(row) and row.get("prediction") is not None


def _mean(rows: Sequence[Mapping[str, Any]], field: str) -> float | None:
    values = [
        float(row[field])
        for row in rows
        if isinstance(row.get(field), (int, float)) and not isinstance(row.get(field), bool)
    ]
    return statistics.fmean(values) if values else None


def summarize_method(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    nominal = list(rows)
    accessible = [row for row in rows if _accessible(row)]
    valid = [row for row in rows if _valid(row)]

    def accuracy(group: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        correct = sum(bool(row.get("correct")) for row in group)
        return {
            "denominator": len(group),
            "correct": correct,
            "accuracy": correct / len(group) if group else None,
        }

    by_dataset = {}
    for dataset in DATASETS:
        dataset_rows = [
            row for row in nominal if row.get("_dataset_identity") == dataset
        ]
        by_dataset[dataset] = {
            "nominal": accuracy(dataset_rows),
            "accessible": accuracy([row for row in dataset_rows if _accessible(row)]),
            "common_valid_self": accuracy([row for row in dataset_rows if _valid(row)]),
        }
    return {
        "nominal": accuracy(nominal),
        "accessible": accuracy(accessible),
        "common_valid_self": accuracy(valid),
        "by_dataset": by_dataset,
        "failure_count": sum(_failed(row) for row in nominal),
        "data_unavailable": sum(bool(row.get("data_unavailable")) for row in nominal),
        "annotation_leak": sum(
            row.get("annotation_leak_check") == "failed"
            or row.get("failure_class") == "annotation_leak"
            for row in nominal
        ),
        "mean_prompt_tokens": _mean(nominal, "prompt_tokens"),
        "mean_completion_tokens": _mean(nominal, "completion_tokens"),
        "mean_reasoning_tokens": _mean(nominal, "reasoning_tokens"),
        "mean_visual_tokens": _mean(nominal, "visual_tokens"),
        "mean_total_tokens": _mean(nominal, "total_tokens"),
        "mean_latency_s": _mean(nominal, "latency_s"),
        "visual_token_rows": sum(
            isinstance(row.get("visual_tokens"), (int, float))
            and not isinstance(row.get("visual_tokens"), bool)
            for row in nominal
        ),
    }


def exact_mcnemar_p_value(corrected: int, regressed: int) -> float:
    if corrected < 0 or regressed < 0:
        raise ValueError("McNemar discordant counts cannot be negative")
    discordant = corrected + regressed
    if discordant == 0:
        return 1.0
    tail = sum(
        math.comb(discordant, value)
        for value in range(min(corrected, regressed) + 1)
    ) / (2**discordant)
    return min(1.0, 2.0 * tail)


def paired_method_compare(
    baseline: Sequence[Mapping[str, Any]],
    candidate: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    def keyed(rows: Sequence[Mapping[str, Any]]) -> dict[tuple[str, str], Mapping[str, Any]]:
        result: dict[tuple[str, str], Mapping[str, Any]] = {}
        for row in rows:
            key = (str(row["_dataset_identity"]), str(row["sample_id"]))
            if key in result:
                raise ValueError(f"duplicate paired identity: {key}")
            result[key] = row
        return result

    left, right = keyed(baseline), keyed(candidate)
    shared = sorted(left.keys() & right.keys())
    valid = [key for key in shared if _valid(left[key]) and _valid(right[key])]
    corrected = sum(
        not bool(left[key].get("correct")) and bool(right[key].get("correct"))
        for key in valid
    )
    regressed = sum(
        bool(left[key].get("correct")) and not bool(right[key].get("correct"))
        for key in valid
    )
    baseline_correct = sum(bool(left[key].get("correct")) for key in valid)
    candidate_correct = sum(bool(right[key].get("correct")) for key in valid)
    return {
        "shared": len(shared),
        "common_valid": len(valid),
        "baseline_correct": baseline_correct,
        "candidate_correct": candidate_correct,
        "gain": candidate_correct - baseline_correct,
        "corrected": corrected,
        "regressed": regressed,
        "mcnemar_exact_p": exact_mcnemar_p_value(corrected, regressed),
    }


def _method_id(task: Mapping[str, Any]) -> str:
    value = str(task.get("task_id") or "")
    if not value:
        raise ValueError("final task has no task_id")
    return value


def load_final_matrix(
    plan_paths: Sequence[Path], experiment_config_sha256: str
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    if len(plan_paths) != 3:
        raise ValueError("final matrix requires exactly q9, q4, and sft9 run plans")
    groups: set[str] = set()
    methods: dict[str, list[dict[str, Any]]] = defaultdict(list)
    plan_records: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for path in plan_paths:
        plan = load_frozen_run_plan(path, experiment_config_sha256)
        if plan.get("phase") != "final_matrix":
            raise ValueError(f"not a final_matrix plan: {path}")
        group = str(plan.get("final_model_group") or "")
        if group not in {"q9", "q4", "sft9"} or group in groups:
            raise ValueError("final plans must contain q9/q4/sft9 exactly once")
        groups.add(group)
        tasks = plan.get("tasks")
        if not isinstance(tasks, list):
            raise ValueError("final run plan has no tasks")
        for task in tasks:
            run: DevRun = load_task_result(plan, task)
            method = _method_id(task)
            identity = (method, run.dataset)
            if identity in seen:
                raise ValueError(f"duplicate final method/dataset task: {identity}")
            seen.add(identity)
            methods[method].extend(
                {**row, "_dataset_identity": run.dataset} for row in run.rows
            )
        plan_records.append(
            {
                "path": str(path.resolve()),
                "sha256": file_sha256(path),
                "plan_sha256": plan["plan_sha256"],
                "group": group,
            }
        )
    expected_prefixes = {
        "q9_no_video",
        "q9_eva_clean",
        "q9_best_untrained",
        "q4_no_video",
        "q4_eva_clean",
        "q4_best_untrained",
    }
    if not expected_prefixes <= set(methods):
        raise ValueError(f"final matrix is missing methods: {sorted(expected_prefixes - set(methods))}")
    for method, rows in methods.items():
        if len(rows) != 300:
            raise ValueError(f"final method {method} expected 300 rows, found {len(rows)}")
    return dict(methods), {"plans": plan_records}


def build_final_report(
    plan_paths: Sequence[Path], experiment_config_sha256: str
) -> tuple[dict[str, Any], dict[str, list[dict[str, Any]]]]:
    methods, provenance = load_final_matrix(plan_paths, experiment_config_sha256)

    def exactly_one(prefix: str) -> str:
        matches = [name for name in methods if name.startswith(prefix)]
        if len(matches) != 1:
            raise ValueError(f"expected one final method with prefix {prefix!r}, found {matches}")
        return matches[0]

    q9_direct = exactly_one("q9_direct_")
    q4_direct = exactly_one("q4_direct_")
    sft9 = exactly_one("sft9_")
    comparisons = {
        "q9_no_video_vs_direct": paired_method_compare(
            methods["q9_no_video"], methods[q9_direct]
        ),
        "q9_direct_vs_untrained_agent": paired_method_compare(
            methods[q9_direct], methods["q9_best_untrained"]
        ),
        "q4_no_video_vs_direct": paired_method_compare(
            methods["q4_no_video"], methods[q4_direct]
        ),
        "q4_direct_vs_untrained_agent": paired_method_compare(
            methods[q4_direct], methods["q4_best_untrained"]
        ),
        "q9_untrained_vs_sft": paired_method_compare(
            methods["q9_best_untrained"], methods[sft9]
        ),
    }
    summaries = {name: summarize_method(rows) for name, rows in sorted(methods.items())}
    teacher = summaries["q9_best_untrained"]
    student = summaries[sft9]
    total_ratio = (
        student["mean_total_tokens"] / teacher["mean_total_tokens"]
        if teacher["mean_total_tokens"]
        else None
    )
    visual_ratio = (
        student["mean_visual_tokens"] / teacher["mean_visual_tokens"]
        if teacher["mean_visual_tokens"]
        else None
    )
    success = {
        "untrained_q9_agent_strictly_beats_direct": comparisons[
            "q9_direct_vs_untrained_agent"
        ]["gain"]
        > 0,
        "sft_q9_strictly_beats_untrained": comparisons["q9_untrained_vs_sft"][
            "gain"
        ]
        > 0,
        "sft_total_tokens_at_most_70pct": total_ratio is not None
        and total_ratio <= 0.70,
        "sft_visual_tokens_at_most_70pct": visual_ratio is not None
        and visual_ratio <= 0.70,
    }
    report: dict[str, Any] = {
        "schema_version": 1,
        "experiment_config_sha256": experiment_config_sha256,
        "scope": "fixed engineering test set; not a statistical blind test",
        "provenance": provenance,
        "methods": summaries,
        "comparisons": comparisons,
        "sft_token_ratios": {"total": total_ratio, "visual": visual_ratio},
        "success_conditions": success,
        "overall_success": all(success.values()),
    }
    report["report_sha256"] = canonical_sha256(report)
    return report, methods


__all__ = [
    "build_final_report",
    "exact_mcnemar_p_value",
    "paired_method_compare",
    "summarize_method",
]
