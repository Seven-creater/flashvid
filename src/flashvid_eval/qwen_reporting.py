from __future__ import annotations

import json
import math
import statistics
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any


def _number(value: Any) -> float | None:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        number = float(value)
        return number if math.isfinite(number) else None
    return None


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    ids = [str(row.get("sample_id")) for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError(f"duplicate sample_id in {path}")
    return rows


def summarize(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    total = len(rows)
    correct = sum(bool(row.get("correct")) for row in rows)
    errors = sum(
        bool(row.get("error") or row.get("error_type") or row.get("parse_error"))
        for row in rows
    )
    length_stops = sum(str(row.get("finish_reason") or "") == "length" for row in rows)
    fallback = sum(bool(row.get("fallback_used") or row.get("fallback_to_candidate")) for row in rows)
    token_fields = (
        "prompt_tokens",
        "completion_tokens",
        "reasoning_tokens",
        "visual_tokens",
        "total_tokens",
        "latency_s",
    )
    means: dict[str, float | None] = {}
    for field in token_fields:
        values = [_number(row.get(field)) for row in rows]
        clean = [value for value in values if value is not None]
        means[f"mean_{field}"] = statistics.fmean(clean) if clean else None
    return {
        "total": total,
        "correct": correct,
        "accuracy": correct / total if total else None,
        "errors": errors,
        "failure_rate": errors / total if total else None,
        "length_finish": length_stops,
        "fallback": fallback,
        **means,
    }


def paired_compare(
    baseline: Sequence[Mapping[str, Any]],
    candidate: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    baseline_by_id = {str(row["sample_id"]): row for row in baseline}
    candidate_by_id = {str(row["sample_id"]): row for row in candidate}
    shared = sorted(baseline_by_id.keys() & candidate_by_id.keys())
    common_valid = [
        sample_id
        for sample_id in shared
        if baseline_by_id[sample_id].get("prediction") is not None
        and candidate_by_id[sample_id].get("prediction") is not None
        and not baseline_by_id[sample_id].get("data_unavailable")
        and not candidate_by_id[sample_id].get("data_unavailable")
    ]
    base_correct = sum(bool(baseline_by_id[item].get("correct")) for item in common_valid)
    candidate_correct = sum(bool(candidate_by_id[item].get("correct")) for item in common_valid)
    corrected = sum(
        not bool(baseline_by_id[item].get("correct"))
        and bool(candidate_by_id[item].get("correct"))
        for item in common_valid
    )
    regressed = sum(
        bool(baseline_by_id[item].get("correct"))
        and not bool(candidate_by_id[item].get("correct"))
        for item in common_valid
    )
    return {
        "shared": len(shared),
        "common_valid": len(common_valid),
        "baseline_correct": base_correct,
        "candidate_correct": candidate_correct,
        "gain": candidate_correct - base_correct,
        "corrected": corrected,
        "regressed": regressed,
    }


@dataclass(frozen=True)
class PromotionDecision:
    accepted: bool
    seed_gains: dict[int, int]
    mean_gain: float
    seed_wins: int
    reason: str


def promotion_decision(
    baseline_correct_by_seed: Mapping[int, int],
    candidate_correct_by_seed: Mapping[int, int],
    *,
    minimum_mean_gain: float = 2.0,
    minimum_seed_wins: int = 2,
) -> PromotionDecision:
    if set(baseline_correct_by_seed) != set(candidate_correct_by_seed):
        raise ValueError("baseline and candidate seed sets differ")
    if not baseline_correct_by_seed:
        raise ValueError("at least one seed is required")
    gains = {
        int(seed): int(candidate_correct_by_seed[seed] - baseline_correct_by_seed[seed])
        for seed in sorted(baseline_correct_by_seed)
    }
    mean_gain = statistics.fmean(gains.values())
    wins = sum(gain > 0 for gain in gains.values())
    accepted = mean_gain >= minimum_mean_gain and wins >= minimum_seed_wins
    reason = (
        "promotion_gate_passed"
        if accepted
        else f"mean_gain={mean_gain:.3f},seed_wins={wins}"
    )
    return PromotionDecision(accepted, gains, mean_gain, wins, reason)


def choose_best_framework(points: Iterable[Mapping[str, Any]]) -> Mapping[str, Any]:
    eligible = [
        point
        for point in points
        if float(point.get("failure_rate") or 0.0) <= 0.01
        and point.get("annotation_leak", 0) == 0
    ]
    if not eligible:
        raise ValueError("no eligible framework points")

    def key(point: Mapping[str, Any]) -> tuple[float, float, float, float]:
        accuracy = float(point.get("mean_accuracy") or point.get("accuracy") or 0.0)
        variance = float(point.get("accuracy_stdev") or 0.0)
        regressions = float(point.get("regressed") or 0.0)
        tokens = float(point.get("mean_total_tokens") or math.inf)
        return (-accuracy, variance, regressions, tokens)

    return min(eligible, key=key)
