from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DATASETS = ("lvbench", "lsdbench", "cgbench")
OPTION_LETTERS = frozenset("ABCDEFGH")
ERROR_FIELDS = (
    "error",
    "api_error",
    "frame_error",
    "transcode_error",
    "parse_error",
    "verifier_error",
)


@dataclass(frozen=True)
class MethodSpec:
    name: str
    stage: str
    paths: Mapping[str, tuple[str, ...]]
    label: str | None = None
    role: str | None = None
    fixed_ratio: float | None = None
    reference: str | None = None


@dataclass(frozen=True)
class LoadedRecords:
    records: tuple[dict[str, Any], ...]
    duplicate_sample_ids: tuple[str, ...]
    invalid_lines: int


def _number(value: Any) -> float | None:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        number = float(value)
        if math.isfinite(number):
            return number
    return None


def _prediction(record: Mapping[str, Any]) -> str | None:
    value = record.get("final_prediction", record.get("prediction"))
    answer = str(value or "").strip().upper()
    return answer if answer in OPTION_LETTERS else None


def _answer(record: Mapping[str, Any]) -> str | None:
    for key in ("answer", "correct_answer", "right_answer"):
        value = str(record.get(key) or "").strip().upper()
        if value in OPTION_LETTERS:
            return value
    return None


def _record_correct(record: Mapping[str, Any]) -> bool:
    prediction = _prediction(record)
    answer = _answer(record)
    if answer is not None:
        return prediction == answer
    return prediction is not None and bool(record.get("correct"))


def _record_failed(record: Mapping[str, Any]) -> bool:
    if any(record.get(key) for key in ERROR_FIELDS):
        return True
    if _prediction(record) is None:
        return True
    leak = record.get("annotation_leak_check")
    return leak is not None and leak != "passed"


def _nested_number(record: Mapping[str, Any], *keys: str) -> float | None:
    current: Any = record
    for key in keys:
        if not isinstance(current, Mapping):
            return None
        current = current.get(key)
    return _number(current)


def _first_number(record: Mapping[str, Any], paths: Sequence[tuple[str, ...]]) -> float | None:
    for path in paths:
        value = _nested_number(record, *path)
        if value is not None:
            return value
    return None


def _usage_total(record: Mapping[str, Any], prefix: str) -> float | None:
    direct = _number(record.get(f"{prefix}_total_tokens"))
    if direct is not None:
        return direct
    usage = record.get(f"{prefix}_usage")
    if not isinstance(usage, Mapping):
        return None
    total = _number(usage.get("total_tokens"))
    if total is not None:
        return total
    prompt = _number(usage.get("prompt_tokens"))
    completion = _number(usage.get("completion_tokens"))
    if prompt is None and completion is None:
        return None
    return (prompt or 0.0) + (completion or 0.0)


def _mean(values: Iterable[float | None]) -> float | None:
    present = [value for value in values if value is not None]
    return sum(present) / len(present) if present else None


def _percentile(values: Iterable[float | None], quantile: float) -> float | None:
    present = sorted(value for value in values if value is not None)
    if not present:
        return None
    if len(present) == 1:
        return present[0]
    location = (len(present) - 1) * quantile
    lower = math.floor(location)
    upper = math.ceil(location)
    if lower == upper:
        return present[lower]
    weight = location - lower
    return present[lower] * (1.0 - weight) + present[upper] * weight


def read_jsonl_latest(path: str | Path) -> LoadedRecords:
    """Read a resumable JSONL, retaining the last row for each sample."""

    source = Path(path)
    latest: dict[str, dict[str, Any]] = {}
    seen: Counter[str] = Counter()
    invalid_lines = 0
    for line in source.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            invalid_lines += 1
            continue
        if not isinstance(record, dict):
            invalid_lines += 1
            continue
        sample_id = str(record.get("sample_id") or "").strip()
        if not sample_id:
            invalid_lines += 1
            continue
        seen[sample_id] += 1
        latest[sample_id] = record
    duplicates = tuple(sorted(sample_id for sample_id, count in seen.items() if count > 1))
    return LoadedRecords(
        records=tuple(latest[sample_id] for sample_id in sorted(latest)),
        duplicate_sample_ids=duplicates,
        invalid_lines=invalid_lines,
    )


def read_jsonl_all(path: str | Path) -> tuple[dict[str, Any], ...]:
    """Read every valid object row, including intentional same-sample trajectories."""

    records: list[dict[str, Any]] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(record, dict):
            records.append(record)
    return tuple(records)


def _tool_steps(record: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    value = record.get("tool_steps")
    if not isinstance(value, list):
        value = record.get("tool_calls")
    if not isinstance(value, list):
        return []
    return [step for step in value if isinstance(step, Mapping)]


def retained_visual_tokens(record: Mapping[str, Any]) -> float | None:
    direct = _number(record.get("retained_visual_tokens"))
    if direct is not None:
        return direct
    values = [_number(step.get("retained_visual_tokens")) for step in _tool_steps(record)]
    return sum(value for value in values if value is not None) if any(
        value is not None for value in values
    ) else None


def raw_visual_tokens(record: Mapping[str, Any]) -> float | None:
    direct = _number(record.get("raw_visual_tokens"))
    if direct is not None:
        return direct
    values = [_number(step.get("raw_visual_tokens")) for step in _tool_steps(record)]
    return sum(value for value in values if value is not None) if any(
        value is not None for value in values
    ) else None


def _budget_metrics(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    ratios: Counter[str] = Counter()
    sequences: Counter[str] = Counter()
    samples_with_steps = 0
    for record in records:
        sequence: list[str] = []
        for step in _tool_steps(record):
            ratio = _number(step.get("retention_ratio"))
            if ratio is None:
                continue
            key = f"{ratio:.2f}"
            ratios[key] += 1
            sequence.append(key)
        if sequence:
            samples_with_steps += 1
            sequences[">".join(sequence)] += 1
    total_steps = sum(ratios.values())
    largest_share = max(ratios.values()) / total_steps if total_steps else None
    return {
        "samples_with_budget_steps": samples_with_steps,
        "step_count": total_steps,
        "step_ratio_counts": dict(sorted(ratios.items())),
        "step_ratio_shares": {
            key: count / total_steps for key, count in sorted(ratios.items())
        } if total_steps else {},
        "sequence_counts": dict(sorted(sequences.items())),
        "distinct_ratios": len(ratios),
        "largest_ratio_share": largest_share,
        "constant_single_budget": len(ratios) == 1 and total_steps > 0,
    }


def summarize_records(
    records: Sequence[Mapping[str, Any]],
    *,
    expected: int | None = None,
    duplicates: Sequence[str] = (),
    invalid_lines: int = 0,
) -> dict[str, Any]:
    total = len(records)
    correct = sum(_record_correct(record) for record in records)
    data_unavailable = sum(bool(record.get("data_unavailable")) for record in records)
    available_records = total - data_unavailable
    available_correct = sum(
        _record_correct(record)
        for record in records
        if not bool(record.get("data_unavailable"))
    )
    valid = sum(_prediction(record) is not None for record in records)
    failures = sum(_record_failed(record) for record in records)
    engineering_failures = sum(
        _record_failed(record)
        for record in records
        if not bool(record.get("data_unavailable"))
    )
    leak_failures = sum(
        record.get("annotation_leak_check") not in (None, "passed")
        for record in records
    )
    leak_passed = sum(
        record.get("annotation_leak_check") == "passed"
        for record in records
    )
    candidate_valid = 0
    candidate_correct = 0
    changed = 0
    changes_fixed = 0
    changes_harmed = 0
    changed_wrong_to_wrong = 0
    candidate_sources: Counter[str] = Counter()
    fallback = 0
    candidate_reruns = 0
    for record in records:
        candidate = str(record.get("candidate_answer") or "").strip().upper()
        prediction = _prediction(record)
        answer = _answer(record)
        source = str(record.get("candidate_source") or "none")
        candidate_sources[source] += 1
        fallback += bool(record.get("fallback_to_candidate"))
        candidate_reruns += int(_number(record.get("candidate_rerun")) or 0)
        if candidate not in OPTION_LETTERS:
            continue
        candidate_valid += 1
        was_correct = answer is not None and candidate == answer
        candidate_correct += was_correct
        if prediction is None or prediction == candidate:
            continue
        changed += 1
        is_correct = answer is not None and prediction == answer
        if not was_correct and is_correct:
            changes_fixed += 1
        elif was_correct and not is_correct:
            changes_harmed += 1
        elif not was_correct and not is_correct:
            changed_wrong_to_wrong += 1

    retained = [retained_visual_tokens(record) for record in records]
    raw = [raw_visual_tokens(record) for record in records]
    latency = [_number(record.get("latency_s")) for record in records]
    controller_latency = [_number(record.get("controller_latency_s")) for record in records]
    perception_latency = [_number(record.get("perception_latency_s")) for record in records]
    rounds = [
        _first_number(record, (("turn_count",), ("rounds",)))
        for record in records
    ]
    calls = [
        float(len(_tool_steps(record)))
        if _tool_steps(record)
        else _first_number(record, (("tool_call_count",), ("perception_requests",)))
        for record in records
    ]
    controller_tokens = [
        _usage_total(record, "controller")
        for record in records
    ]
    perception_tokens = [
        _usage_total(record, "perception")
        for record in records
    ]
    total_tokens = [
        _first_number(record, (("total_tokens",), ("usage", "total_tokens")))
        for record in records
    ]
    if expected is None:
        status = "complete" if total else "pending"
    elif total == expected:
        status = "complete"
    elif total == 0:
        status = "pending"
    else:
        status = "partial"
    return {
        "status": status,
        "records": total,
        "expected": expected,
        "completion_rate": total / expected if expected else None,
        "correct": correct,
        "accuracy": correct / total if total else None,
        "accuracy_over_expected": correct / expected if expected else None,
        "data_unavailable": data_unavailable,
        "available_records": available_records,
        "available_correct": available_correct,
        "accuracy_available": (
            available_correct / available_records if available_records else None
        ),
        "valid_predictions": valid,
        "failures": failures,
        "failure_rate": failures / total if total else None,
        "engineering_failures": engineering_failures,
        "engineering_failure_rate": (
            engineering_failures / available_records if available_records else None
        ),
        "invalid_lines": invalid_lines,
        "duplicate_sample_ids": list(duplicates),
        "duplicate_sample_id_count": len(duplicates),
        "annotation_leak_failures": leak_failures,
        "annotation_leak_passed": leak_passed,
        "candidate": {
            "valid": candidate_valid,
            "correct": candidate_correct,
            "source_counts": dict(sorted(candidate_sources.items())),
            "normalized_recovered": candidate_sources.get("normalized", 0),
            "changed": changed,
            "changes_fixed": changes_fixed,
            "changes_harmed": changes_harmed,
            "changed_wrong_to_wrong": changed_wrong_to_wrong,
            "fallback_to_candidate": fallback,
            "candidate_rerun_total": candidate_reruns,
        },
        "runtime": {
            "mean_rounds": _mean(rounds),
            "p50_rounds": _percentile(rounds, 0.5),
            "p90_rounds": _percentile(rounds, 0.9),
            "mean_tool_calls": _mean(calls),
            "p50_tool_calls": _percentile(calls, 0.5),
            "p90_tool_calls": _percentile(calls, 0.9),
            "mean_latency_s": _mean(latency),
            "p50_latency_s": _percentile(latency, 0.5),
            "p90_latency_s": _percentile(latency, 0.9),
            "mean_controller_latency_s": _mean(controller_latency),
            "mean_perception_latency_s": _mean(perception_latency),
            "latency_coverage": sum(value is not None for value in latency),
            "controller_latency_coverage": sum(
                value is not None for value in controller_latency
            ),
            "perception_latency_coverage": sum(
                value is not None for value in perception_latency
            ),
            "round_coverage": sum(value is not None for value in rounds),
            "tool_call_coverage": sum(value is not None for value in calls),
        },
        "tokens": {
            "mean_raw_visual_tokens": _mean(raw),
            "total_raw_visual_tokens": sum(value for value in raw if value is not None),
            "raw_token_coverage": sum(value is not None for value in raw),
            "mean_retained_visual_tokens": _mean(retained),
            "total_retained_visual_tokens": sum(
                value for value in retained if value is not None
            ),
            "retained_token_coverage": sum(value is not None for value in retained),
            "mean_controller_tokens": _mean(controller_tokens),
            "mean_perception_tokens": _mean(perception_tokens),
            "mean_total_tokens": _mean(total_tokens),
            "controller_token_coverage": sum(
                value is not None for value in controller_tokens
            ),
            "perception_token_coverage": sum(
                value is not None for value in perception_tokens
            ),
            "total_token_coverage": sum(
                value is not None for value in total_tokens
            ),
        },
        "budget": _budget_metrics(records),
    }


def aggregate_dataset_metrics(
    datasets: Mapping[str, Mapping[str, Any]],
    expected_datasets: Sequence[str] = DATASETS,
) -> dict[str, Any]:
    available = [
        metrics for dataset, metrics in datasets.items()
        if dataset in expected_datasets and metrics.get("status") != "pending"
    ]
    if not available:
        return {"status": "pending", "datasets_complete": 0}
    complete = sum(metrics.get("status") == "complete" for metrics in available)
    records = sum(int(metrics.get("records") or 0) for metrics in available)
    expected_values = [metrics.get("expected") for metrics in available]
    expected = (
        sum(int(value) for value in expected_values)
        if expected_values and all(isinstance(value, int) for value in expected_values)
        else None
    )
    correct = sum(int(metrics.get("correct") or 0) for metrics in available)
    data_unavailable = sum(
        int(metrics.get("data_unavailable") or 0) for metrics in available
    )
    available_records = sum(
        int(metrics.get("available_records") or 0) for metrics in available
    )
    available_correct = sum(
        int(metrics.get("available_correct") or 0) for metrics in available
    )
    failures = sum(int(metrics.get("failures") or 0) for metrics in available)
    engineering_failures = sum(
        int(metrics.get("engineering_failures") or 0) for metrics in available
    )
    retained_total = sum(
        float(metrics["tokens"].get("total_retained_visual_tokens") or 0)
        for metrics in available
    )
    retained_coverage = sum(
        int(metrics["tokens"].get("retained_token_coverage") or 0)
        for metrics in available
    )
    raw_total = sum(
        float(metrics["tokens"].get("total_raw_visual_tokens") or 0)
        for metrics in available
    )
    raw_coverage = sum(
        int(metrics["tokens"].get("raw_token_coverage") or 0)
        for metrics in available
    )
    ratios: Counter[str] = Counter()
    for metrics in available:
        ratios.update(metrics["budget"].get("step_ratio_counts") or {})
    ratio_total = sum(ratios.values())
    status = (
        "complete"
        if complete == len(expected_datasets) and len(available) == len(expected_datasets)
        else "partial"
    )

    def weighted(section: str, mean_key: str, coverage_key: str) -> float | None:
        coverage = sum(
            int(metrics[section].get(coverage_key) or 0)
            for metrics in available
        )
        if not coverage:
            return None
        total = sum(
            float(metrics[section].get(mean_key) or 0)
            * int(metrics[section].get(coverage_key) or 0)
            for metrics in available
        )
        return total / coverage

    return {
        "status": status,
        "datasets_complete": complete,
        "records": records,
        "expected": expected,
        "correct": correct,
        "accuracy": correct / records if records else None,
        "accuracy_over_expected": correct / expected if expected else None,
        "data_unavailable": data_unavailable,
        "available_records": available_records,
        "available_correct": available_correct,
        "accuracy_available": (
            available_correct / available_records if available_records else None
        ),
        "failures": failures,
        "failure_rate": failures / records if records else None,
        "engineering_failures": engineering_failures,
        "engineering_failure_rate": (
            engineering_failures / available_records if available_records else None
        ),
        "mean_retained_visual_tokens": (
            retained_total / retained_coverage if retained_coverage else None
        ),
        "mean_raw_visual_tokens": raw_total / raw_coverage if raw_coverage else None,
        "total_raw_visual_tokens": raw_total,
        "raw_token_coverage": raw_coverage,
        "total_retained_visual_tokens": retained_total,
        "retained_token_coverage": retained_coverage,
        "candidate_changes_fixed": sum(
            int(metrics["candidate"].get("changes_fixed") or 0) for metrics in available
        ),
        "candidate_changes_harmed": sum(
            int(metrics["candidate"].get("changes_harmed") or 0) for metrics in available
        ),
        "candidate_normalized_recovered": sum(
            int(metrics["candidate"].get("normalized_recovered") or 0)
            for metrics in available
        ),
        "annotation_leak_failures": sum(
            int(metrics.get("annotation_leak_failures") or 0) for metrics in available
        ),
        "annotation_leak_passed": sum(
            int(metrics.get("annotation_leak_passed") or 0) for metrics in available
        ),
        "duplicate_sample_id_count": sum(
            int(metrics.get("duplicate_sample_id_count") or 0) for metrics in available
        ),
        "candidate_rerun_total": sum(
            int(metrics["candidate"].get("candidate_rerun_total") or 0)
            for metrics in available
        ),
        "budget": {
            "step_ratio_counts": dict(sorted(ratios.items())),
            "distinct_ratios": len(ratios),
            "largest_ratio_share": (
                max(ratios.values()) / ratio_total if ratio_total else None
            ),
            "constant_single_budget": len(ratios) == 1 and ratio_total > 0,
        },
        "runtime": {
            "mean_rounds": weighted("runtime", "mean_rounds", "round_coverage"),
            "mean_tool_calls": weighted(
                "runtime", "mean_tool_calls", "tool_call_coverage"
            ),
            "mean_latency_s": weighted(
                "runtime", "mean_latency_s", "latency_coverage"
            ),
            "mean_controller_latency_s": weighted(
                "runtime",
                "mean_controller_latency_s",
                "controller_latency_coverage",
            ),
            "mean_perception_latency_s": weighted(
                "runtime",
                "mean_perception_latency_s",
                "perception_latency_coverage",
            ),
        },
        "tokens": {
            "mean_raw_visual_tokens": raw_total / raw_coverage if raw_coverage else None,
            "mean_retained_visual_tokens": (
                retained_total / retained_coverage if retained_coverage else None
            ),
            "mean_controller_tokens": weighted(
                "tokens", "mean_controller_tokens", "controller_token_coverage"
            ),
            "mean_perception_tokens": weighted(
                "tokens", "mean_perception_tokens", "perception_token_coverage"
            ),
            "mean_total_tokens": weighted(
                "tokens", "mean_total_tokens", "total_token_coverage"
            ),
        },
    }


def exact_mcnemar(
    baseline_records: Sequence[Mapping[str, Any]],
    method_records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Compute an exact paired McNemar test on jointly valid sample IDs."""

    baseline = {str(record.get("sample_id")): record for record in baseline_records}
    method = {str(record.get("sample_id")): record for record in method_records}
    paired = 0
    baseline_correct = 0
    method_correct = 0
    wins = 0
    losses = 0
    answer_mismatch = 0
    for sample_id in sorted(set(baseline) & set(method)):
        before = baseline[sample_id]
        after = method[sample_id]
        if _record_failed(before) or _record_failed(after):
            continue
        before_answer = _answer(before)
        after_answer = _answer(after)
        if before_answer is None or after_answer is None:
            continue
        if before_answer != after_answer:
            answer_mismatch += 1
            continue
        before_correct = _prediction(before) == before_answer
        after_correct = _prediction(after) == after_answer
        paired += 1
        baseline_correct += before_correct
        method_correct += after_correct
        wins += (not before_correct) and after_correct
        losses += before_correct and (not after_correct)
    discordant = wins + losses
    if discordant:
        tail = sum(
            math.comb(discordant, index)
            for index in range(min(wins, losses) + 1)
        ) / (2 ** discordant)
        p_value = min(1.0, 2.0 * tail)
    else:
        p_value = 1.0
    return {
        "status": "complete" if paired else "pending",
        "paired_common_valid": paired,
        "baseline_correct": baseline_correct,
        "method_correct": method_correct,
        "wins_baseline_wrong_method_correct": wins,
        "losses_baseline_correct_method_wrong": losses,
        "net_correct": wins - losses,
        "discordant": discordant,
        "mcnemar_exact_two_sided_p": p_value,
        "answer_mismatches_excluded": answer_mismatch,
    }


def _sample_signature(records_by_dataset: Mapping[str, Sequence[Mapping[str, Any]]]) -> str:
    identities = sorted(
        f"{dataset}\0{record.get('sample_id')}"
        for dataset, records in records_by_dataset.items()
        for record in records
    )
    return hashlib.sha256("\n".join(identities).encode("utf-8")).hexdigest()


def score_checkpoint(
    name: str,
    records_by_dataset: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    expected_counts: Mapping[str, int],
    duplicate_counts: Mapping[str, int] | None = None,
    invalid_line_counts: Mapping[str, int] | None = None,
) -> dict[str, Any]:
    duplicates = duplicate_counts or {}
    invalid_lines = invalid_line_counts or {}
    dataset_metrics = {
        dataset: summarize_records(
            records_by_dataset.get(dataset, ()),
            expected=expected,
            duplicates=tuple(f"duplicate-{index}" for index in range(duplicates.get(dataset, 0))),
            invalid_lines=invalid_lines.get(dataset, 0),
        )
        for dataset, expected in expected_counts.items()
    }
    reasons: list[str] = []
    for dataset, expected in expected_counts.items():
        records = records_by_dataset.get(dataset, ())
        if len(records) != expected:
            reasons.append(f"{dataset}:expected_{expected}_got_{len(records)}")
        if duplicates.get(dataset, 0):
            reasons.append(f"{dataset}:duplicate_sample_ids")
        if invalid_lines.get(dataset, 0):
            reasons.append(f"{dataset}:invalid_jsonl_lines")
        if any(retained_visual_tokens(record) is None for record in records):
            reasons.append(f"{dataset}:missing_retained_visual_tokens")
    aggregate = aggregate_dataset_metrics(dataset_metrics, tuple(expected_counts))
    total_records = int(aggregate.get("records") or 0)
    maximum_failures = math.floor(total_records * 0.01)
    if int(aggregate.get("failures") or 0) > maximum_failures:
        reasons.append(
            "aggregate:failure_rate_above_1_percent"
        )
    if int(aggregate.get("annotation_leak_failures") or 0):
        reasons.append("aggregate:annotation_leak")
    if int(aggregate.get("candidate_rerun_total") or 0):
        reasons.append("aggregate:candidate_rerun_nonzero")
    if aggregate.get("budget", {}).get("constant_single_budget"):
        reasons.append("aggregate:constant_single_budget_policy")
    for dataset, metrics in dataset_metrics.items():
        if int(metrics.get("candidate", {}).get("changes_harmed") or 0) > 1:
            reasons.append(f"{dataset}:candidate_changes_harmed_above_1")
    return {
        "checkpoint": name,
        "status": "eligible" if not reasons else ("pending" if not records_by_dataset else "ineligible"),
        "ineligible_reasons": reasons,
        "sample_signature": _sample_signature(records_by_dataset),
        "datasets": dataset_metrics,
        "aggregate": aggregate,
    }


def choose_sft_checkpoint(scores: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Select by correct count first and retained visual tokens second."""

    eligible = [dict(score) for score in scores if score.get("status") == "eligible"]
    if not eligible:
        return {
            "status": "pending",
            "selected_checkpoint": None,
            "reason": "no_complete_eligible_checkpoint",
        }
    signatures = {str(score.get("sample_signature")) for score in eligible}
    if len(signatures) != 1:
        return {
            "status": "invalid",
            "selected_checkpoint": None,
            "reason": "eligible_checkpoints_use_different_validation_samples",
            "sample_signatures": sorted(signatures),
        }
    ordered = sorted(
        eligible,
        key=lambda score: (
            -int(score["aggregate"]["correct"]),
            float(score["aggregate"]["total_retained_visual_tokens"]),
            str(score["checkpoint"]),
        ),
    )
    selected = ordered[0]
    return {
        "status": "complete",
        "selected_checkpoint": selected["checkpoint"],
        "correct": selected["aggregate"]["correct"],
        "total_retained_visual_tokens": selected["aggregate"][
            "total_retained_visual_tokens"
        ],
        "tie_break_rule": (
            "higher total correct, then lower total retained visual tokens, "
            "then lexical checkpoint name"
        ),
        "ranking": [
            {
                "checkpoint": score["checkpoint"],
                "correct": score["aggregate"]["correct"],
                "total_retained_visual_tokens": score["aggregate"][
                    "total_retained_visual_tokens"
                ],
            }
            for score in ordered
        ],
    }


def fixed_budget_curve(
    methods: Mapping[str, Mapping[str, Any]],
    specs: Mapping[str, MethodSpec],
    stage: str,
) -> dict[str, Any]:
    points: list[dict[str, Any]] = []
    for name, spec in specs.items():
        if spec.stage != stage or spec.fixed_ratio is None:
            continue
        aggregate = methods.get(name, {}).get("aggregate", {"status": "pending"})
        point = {
            "method": name,
            "retention_ratio": spec.fixed_ratio,
            "status": aggregate.get("status", "pending"),
            "correct": aggregate.get("correct"),
            "accuracy": aggregate.get("accuracy"),
            "mean_retained_visual_tokens": aggregate.get(
                "mean_retained_visual_tokens"
            ),
            "mean_raw_visual_tokens": methods.get(name, {}).get("mean_raw_visual_tokens"),
        }
        points.append(point)
    points.sort(key=lambda point: point["retention_ratio"])
    complete = [
        point for point in points
        if point["status"] == "complete"
        and point["correct"] is not None
        and point["mean_retained_visual_tokens"] is not None
    ]
    for point in points:
        point["pareto_efficient"] = None
        if point not in complete:
            continue
        point["pareto_efficient"] = not any(
            other is not point
            and int(other["correct"]) >= int(point["correct"])
            and float(other["mean_retained_visual_tokens"])
            <= float(point["mean_retained_visual_tokens"])
            and (
                int(other["correct"]) > int(point["correct"])
                or float(other["mean_retained_visual_tokens"])
                < float(point["mean_retained_visual_tokens"])
            )
            for other in complete
        )
    return {
        "status": "complete" if complete and len(complete) == len(points) else (
            "partial" if complete else "pending"
        ),
        "points": points,
    }


def pareto_acceptance(
    methods: Mapping[str, Mapping[str, Any]],
    specs: Mapping[str, MethodSpec],
    stage: str,
) -> dict[str, Any]:
    roles: dict[str, str] = {}
    for name, spec in specs.items():
        if spec.stage == stage and spec.role:
            roles[spec.role] = name
    required = ("sft_dynamic", "untrained_dynamic", "fixed_100")
    missing = [
        role for role in required
        if role not in roles
        or methods.get(roles[role], {}).get("aggregate", {}).get("status") != "complete"
    ]
    if missing:
        return {
            "status": "pending",
            "passed": None,
            "missing_roles": missing,
        }
    sft_method = methods[roles["sft_dynamic"]]
    untrained_method = methods[roles["untrained_dynamic"]]
    fixed_method = methods[roles["fixed_100"]]
    sft = sft_method["aggregate"]
    untrained = untrained_method["aggregate"]
    fixed = fixed_method["aggregate"]
    costs = (
        sft.get("mean_retained_visual_tokens"),
        untrained.get("mean_retained_visual_tokens"),
        fixed.get("mean_retained_visual_tokens"),
    )
    if any(cost is None for cost in costs):
        return {
            "status": "pending",
            "passed": None,
            "missing_metrics": ["mean_retained_visual_tokens"],
        }
    sft_cost, untrained_cost, fixed_cost = (float(cost) for cost in costs)
    untrained_correct_delta = int(sft["correct"]) - int(untrained["correct"])
    fixed_correct_delta = int(sft["correct"]) - int(fixed["correct"])
    option_accuracy_gain = untrained_correct_delta >= 3 and sft_cost <= untrained_cost
    option_cost_gain = (
        untrained_correct_delta >= 0
        and untrained_cost > 0
        and sft_cost <= 0.8 * untrained_cost
    )
    fixed_accuracy_ok = fixed_correct_delta >= -3
    fixed_cost_ok = fixed_cost > 0 and sft_cost <= 0.7 * fixed_cost
    per_dataset_drops = {
        dataset: (
            int(sft_method["datasets"][dataset]["correct"])
            - int(fixed_method["datasets"][dataset]["correct"])
        )
        for dataset in DATASETS
    }
    per_dataset_ok = all(delta >= -2 for delta in per_dataset_drops.values())
    dynamic_budget_ok = not bool(sft.get("budget", {}).get("constant_single_budget"))
    conditions = {
        "improves_untrained_accuracy_without_more_cost": option_accuracy_gain,
        "or_preserves_untrained_accuracy_with_20pct_cost_reduction": option_cost_gain,
        "within_3_correct_of_fixed_100": fixed_accuracy_ok,
        "at_least_30pct_cost_reduction_vs_fixed_100": fixed_cost_ok,
        "no_dataset_drops_more_than_2": per_dataset_ok,
        "dynamic_budget_not_single_constant": dynamic_budget_ok,
    }
    passed = (
        (option_accuracy_gain or option_cost_gain)
        and fixed_accuracy_ok
        and fixed_cost_ok
        and per_dataset_ok
        and dynamic_budget_ok
    )
    return {
        "status": "complete",
        "passed": passed,
        "methods": roles,
        "sft_correct_delta_vs_untrained": untrained_correct_delta,
        "sft_correct_delta_vs_fixed_100": fixed_correct_delta,
        "sft_retained_token_reduction_vs_untrained": (
            1.0 - sft_cost / untrained_cost if untrained_cost else None
        ),
        "sft_retained_token_reduction_vs_fixed_100": (
            1.0 - sft_cost / fixed_cost if fixed_cost else None
        ),
        "per_dataset_correct_delta_vs_fixed_100": per_dataset_drops,
        "conditions": conditions,
    }


def default_method_specs() -> tuple[MethodSpec, ...]:
    def dataset_paths(*templates: str) -> dict[str, tuple[str, ...]]:
        return {dataset: tuple(template.format(dataset=dataset) for template in templates)
                for dataset in DATASETS}

    specs = [
        MethodSpec(
            "qwen9b_direct",
            "final",
            dataset_paths(
                "frozen/qwen9b_direct/{dataset}_direct.jsonl",
                "frozen/{dataset}_direct.jsonl",
                "../{dataset}_direct.jsonl",
            ),
            "Frozen Qwen3.5-9B Direct",
            reference="qwen9b_direct",
        ),
        MethodSpec(
            "qwen9b_v3c_frozen",
            "final",
            dataset_paths(
                "native_baselines/qwen9b_v3c_frozen/{dataset}/{dataset}_hybrid_frozen.jsonl",
            ),
            "Frozen-candidate Qwen3.5-9B v3c",
            reference="qwen9b_direct",
        ),
        MethodSpec(
            "qwen4b_direct",
            "final",
            dataset_paths(
                "native_baselines/qwen4b_direct/{dataset}/{dataset}_direct.jsonl",
            ),
            "Native Qwen3.5-4B Direct",
        ),
        MethodSpec(
            "qwen4b_v3c",
            "final",
            dataset_paths(
                "native_baselines/qwen4b_v3c_frozen/{dataset}/{dataset}_hybrid_frozen.jsonl",
                "native_baselines/qwen4b_v3c/{dataset}/{dataset}_hybrid_frozen.jsonl",
            ),
            "Native Qwen3.5-4B v3c",
        ),
    ]
    for stage, prefix in (("validation", "fixed_budget"), ("final", "final_test/fixed_budget")):
        for ratio, slug in ((0.10, "r010"), (0.50, "r050"), (1.00, "r100")):
            name = f"{stage}_q9p4_fixed_{slug}"
            specs.append(
                MethodSpec(
                    name,
                    stage,
                    dataset_paths(
                        f"{prefix}/{slug}/{{dataset}}/{{dataset}}_flashvid_hybrid.jsonl",
                    ),
                    f"9B Agent + FlashVID-4B fixed {ratio:.0%}",
                    role="fixed_100" if ratio == 1.0 else None,
                    fixed_ratio=ratio,
                    reference="qwen9b_direct" if stage == "final" else None,
                )
            )
    for stage, prefix in (("validation", "validation"), ("final", "final_test")):
        for name, directory, label, role in (
            ("q9p4_model_untrained", "q9p4_model_untrained", "9B Agent + P4 untrained dynamic", "untrained_dynamic"),
            ("q4p4_model_untrained", "q4p4_model_untrained", "4B Agent + P4 untrained dynamic", None),
            ("sft4p4_model", "sft4p4_model", "SFT-4B Agent + P4 dynamic", "sft_dynamic"),
        ):
            specs.append(
                MethodSpec(
                    f"{stage}_{name}",
                    stage,
                    dataset_paths(
                        f"{prefix}/{directory}/{{dataset}}/{{dataset}}_flashvid_hybrid.jsonl",
                    ),
                    label,
                    role=role,
                    reference="qwen9b_direct" if stage == "final" else None,
                )
            )
    return tuple(specs)


def load_method_specs(path: str | Path | None) -> tuple[MethodSpec, ...]:
    if path is None:
        return default_method_specs()
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    source = payload.get("methods") if isinstance(payload, Mapping) else payload
    if not isinstance(source, list):
        raise ValueError("method config must contain a methods list")
    specs: list[MethodSpec] = []
    for item in source:
        if not isinstance(item, Mapping):
            raise ValueError("each method config entry must be an object")
        raw_paths = item.get("paths")
        if not isinstance(raw_paths, Mapping):
            raise ValueError(f"method {item.get('name')} must define paths by dataset")
        paths: dict[str, tuple[str, ...]] = {}
        for dataset in DATASETS:
            value = raw_paths.get(dataset)
            if value is None:
                paths[dataset] = ()
            elif isinstance(value, str):
                paths[dataset] = (value,)
            elif isinstance(value, list) and all(isinstance(entry, str) for entry in value):
                paths[dataset] = tuple(value)
            else:
                raise ValueError(f"invalid path list for {item.get('name')}:{dataset}")
        ratio = item.get("fixed_ratio")
        specs.append(
            MethodSpec(
                name=str(item["name"]),
                stage=str(item.get("stage") or "final"),
                paths=paths,
                label=str(item.get("label") or item["name"]),
                role=str(item["role"]) if item.get("role") else None,
                fixed_ratio=float(ratio) if ratio is not None else None,
                reference=str(item["reference"]) if item.get("reference") else None,
            )
        )
    if len({spec.name for spec in specs}) != len(specs):
        raise ValueError("method names must be unique")
    return tuple(specs)


def resolve_result_path(root: Path, candidates: Sequence[str]) -> Path | None:
    for value in candidates:
        candidate = Path(value)
        if not candidate.is_absolute():
            candidate = root / candidate
        if candidate.is_file():
            return candidate
    return None
