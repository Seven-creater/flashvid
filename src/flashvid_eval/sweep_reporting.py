from __future__ import annotations

import hashlib
import json
import math
import random
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any


OPTION_LETTERS = frozenset("ABCDEFGH")
ANALYSIS_GROUPS = ("system", "q9_controller", "q4_controller")
BUDGET_LEVELS = (
    ("R010", 0.10),
    ("R025", 0.25),
    ("R050", 0.50),
    ("R100", 1.00),
)


@dataclass(frozen=True)
class SweepMethod:
    name: str
    kind: str
    controller_group: str | None
    controller_model: str
    paths: Mapping[str, tuple[Path, ...]]
    expected_counts: Mapping[str, int] | None = None
    expected_total: int | None = None
    label: str | None = None


@dataclass(frozen=True)
class LoadedRows:
    rows: tuple[dict[str, Any], ...]
    source_files: tuple[str, ...]
    source_sha256: Mapping[str, str]
    duplicate_ids: tuple[str, ...]
    invalid_lines: int


def _finite_number(value: Any) -> float | None:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        number = float(value)
        return number if math.isfinite(number) else None
    return None


def _letter(value: Any) -> str | None:
    letter = str(value or "").strip().upper()
    return letter if letter in OPTION_LETTERS else None


def _prediction(row: Mapping[str, Any]) -> str | None:
    return _letter(row.get("final_prediction", row.get("prediction")))


def _answer(row: Mapping[str, Any]) -> str | None:
    for key in ("answer", "correct_answer", "right_answer"):
        answer = _letter(row.get(key))
        if answer is not None:
            return answer
    return None


def _candidate(row: Mapping[str, Any]) -> str | None:
    for key in (
        "candidate_answer",
        "normalized_candidate_answer",
        "normalized_candidate",
        "normalized_prediction",
    ):
        candidate = _letter(row.get(key))
        if candidate is not None:
            return candidate
    return None


def _parse_expected_count(
    value: Any,
    *,
    label: str,
) -> tuple[dict[str, int] | None, int | None]:
    if value is None:
        return None, None
    if isinstance(value, int) and not isinstance(value, bool) and value > 0:
        return None, value
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a positive integer or dataset mapping")
    raw_by_dataset = value.get("by_dataset", value)
    explicit_total = value.get("total") if "by_dataset" in value else None
    if not isinstance(raw_by_dataset, Mapping) or not raw_by_dataset:
        raise ValueError(f"{label}.by_dataset must be a non-empty mapping")
    by_dataset: dict[str, int] = {}
    for dataset, expected in raw_by_dataset.items():
        if not isinstance(expected, int) or isinstance(expected, bool) or expected <= 0:
            raise ValueError(f"{label}.{dataset} must be a positive integer")
        by_dataset[str(dataset)] = expected
    total = sum(by_dataset.values())
    if explicit_total is not None:
        if not isinstance(explicit_total, int) or isinstance(explicit_total, bool):
            raise ValueError(f"{label}.total must be a positive integer")
        if explicit_total != total:
            raise ValueError(
                f"{label}.total={explicit_total} does not equal per-dataset sum {total}"
            )
    return by_dataset, total


def _nested_number(row: Mapping[str, Any], *path: str) -> float | None:
    value: Any = row
    for key in path:
        if not isinstance(value, Mapping):
            return None
        value = value.get(key)
    return _finite_number(value)


def _first_number(row: Mapping[str, Any], paths: Sequence[tuple[str, ...]]) -> float | None:
    for path in paths:
        value = _nested_number(row, *path)
        if value is not None:
            return value
    return None


def _tool_steps(row: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    steps = row.get("tool_steps", row.get("tool_calls"))
    if not isinstance(steps, list):
        return []
    return [step for step in steps if isinstance(step, Mapping)]


def _budget_label(value: Any) -> str | None:
    ratio = _finite_number(value)
    if ratio is None:
        return None
    for label, expected in BUDGET_LEVELS:
        if math.isclose(ratio, expected, rel_tol=0.0, abs_tol=1e-6):
            return label
    return None


def _row_budget_labels(row: Mapping[str, Any]) -> tuple[list[str], int]:
    steps = _tool_steps(row)
    if steps and any("retention_ratio" in step for step in steps):
        raw_values = [step.get("retention_ratio") for step in steps]
    else:
        sequence = row.get("budget_sequence")
        raw_values = list(sequence) if isinstance(sequence, list) else []
    labels: list[str] = []
    invalid = 0
    for value in raw_values:
        label = _budget_label(value)
        if label is None:
            invalid += 1
        else:
            labels.append(label)
    return labels, invalid


def budget_distribution(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Summarize the four executed FlashVID budget levels and sample cost."""

    counts: Counter[str] = Counter()
    samples_with_steps = 0
    invalid_steps = 0
    for row in rows:
        labels, invalid = _row_budget_labels(row)
        counts.update(labels)
        samples_with_steps += bool(labels)
        invalid_steps += invalid
    total_steps = sum(counts.values())
    complete_counts = {label: counts[label] for label, _ in BUDGET_LEVELS}
    probabilities = {
        label: complete_counts[label] / total_steps if total_steps else 0.0
        for label, _ in BUDGET_LEVELS
    }
    costs = [
        cost for row in rows if (cost := _retained_tokens(row)) is not None
    ]
    return {
        "counts": complete_counts,
        "probabilities": probabilities,
        "step_count": total_steps,
        "sample_count": len(rows),
        "samples_with_budget_steps": samples_with_steps,
        "invalid_budget_step_count": invalid_steps,
        "mean_retention_ratio": (
            sum(
                counts[label] * ratio
                for label, ratio in BUDGET_LEVELS
            )
            / total_steps
            if total_steps
            else None
        ),
        "mean_retained_visual_tokens_per_sample": (
            sum(costs) / len(costs) if costs else None
        ),
        "retained_visual_token_coverage": len(costs),
    }


def extract_frozen_budget_distribution(
    rows: Sequence[Mapping[str, Any]],
    *,
    source_method: str,
) -> dict[str, Any]:
    """Create the immutable four-level distribution for cost-matched random runs."""

    distribution = budget_distribution(rows)
    if distribution["invalid_budget_step_count"]:
        raise ValueError(
            "cannot freeze a budget distribution containing unsupported budget levels"
        )
    if distribution["step_count"] == 0:
        raise ValueError("cannot freeze a budget distribution with no executed budget steps")
    payload = {
        "schema_version": 1,
        "source_method": source_method,
        "budget_levels": [
            {"label": label, "retention_ratio": ratio}
            for label, ratio in BUDGET_LEVELS
        ],
        **distribution,
    }
    canonical = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return {
        **payload,
        "distribution_sha256": hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
    }


def _summed_step_number(row: Mapping[str, Any], *keys: str) -> float | None:
    values: list[float] = []
    for step in _tool_steps(row):
        value = _first_number(step, tuple((key,) for key in keys))
        if value is not None:
            values.append(value)
    return sum(values) if values else None


def _retained_tokens(row: Mapping[str, Any]) -> float | None:
    direct = _first_number(
        row,
        (
            ("retained_visual_tokens",),
            ("retained_visual_tokens_estimated",),
            ("visual_tokens",),
        ),
    )
    return direct if direct is not None else _summed_step_number(
        row, "retained_visual_tokens", "retained_visual_tokens_estimated"
    )


def _raw_tokens(row: Mapping[str, Any]) -> float | None:
    direct = _first_number(
        row,
        (("raw_visual_tokens",), ("raw_visual_tokens_estimated",)),
    )
    return direct if direct is not None else _summed_step_number(
        row, "raw_visual_tokens", "raw_visual_tokens_estimated"
    )


def _usage_tokens(row: Mapping[str, Any], prefix: str) -> float | None:
    total = _first_number(
        row,
        (
            (f"{prefix}_total_tokens",),
            (f"{prefix}_usage", "total_tokens"),
        ),
    )
    if total is not None:
        return total
    prompt = _nested_number(row, f"{prefix}_usage", "prompt_tokens")
    completion = _nested_number(row, f"{prefix}_usage", "completion_tokens")
    if prompt is None and completion is None:
        return None
    return (prompt or 0.0) + (completion or 0.0)


def _total_tokens(row: Mapping[str, Any]) -> float | None:
    total = _first_number(row, (("total_tokens",), ("usage", "total_tokens")))
    if total is not None:
        return total
    controller = _usage_tokens(row, "controller")
    perception = _usage_tokens(row, "perception")
    if controller is None and perception is None:
        return None
    return (controller or 0.0) + (perception or 0.0)


def _latency(row: Mapping[str, Any]) -> float | None:
    return _first_number(row, (("latency_s",), ("elapsed_s",)))


def _component_latency(row: Mapping[str, Any], component: str) -> float | None:
    return _finite_number(row.get(f"{component}_latency_s"))


def _parse_failed(row: Mapping[str, Any]) -> bool:
    failure_stage = str(row.get("failure_stage") or "").lower()
    return bool(row.get("parse_error")) or "parse" in failure_stage


def _is_evaluable(row: Mapping[str, Any]) -> bool:
    return (
        _prediction(row) is not None
        and _answer(row) is not None
        and not bool(row.get("data_unavailable"))
    )


def _qualified_id(row: Mapping[str, Any]) -> str:
    return f"{row.get('dataset', '')}:{row.get('sample_id', '')}"


def _read_jsonl(paths: Sequence[Path], dataset: str) -> LoadedRows:
    latest: dict[str, dict[str, Any]] = {}
    seen: Counter[str] = Counter()
    invalid_lines = 0
    source_files: list[str] = []
    source_sha256: dict[str, str] = {}
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)
        resolved = str(path.resolve())
        source_files.append(resolved)
        source_sha256[resolved] = hashlib.sha256(path.read_bytes()).hexdigest()
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                invalid_lines += 1
                continue
            if not isinstance(row, dict):
                invalid_lines += 1
                continue
            sample_id = str(row.get("sample_id") or "").strip()
            if not sample_id:
                invalid_lines += 1
                continue
            row = dict(row)
            row["dataset"] = str(row.get("dataset") or dataset)
            identity = _qualified_id(row)
            seen[identity] += 1
            latest[identity] = row
    return LoadedRows(
        rows=tuple(latest[key] for key in sorted(latest)),
        source_files=tuple(source_files),
        source_sha256=source_sha256,
        duplicate_ids=tuple(sorted(key for key, count in seen.items() if count > 1)),
        invalid_lines=invalid_lines,
    )


def candidate_only_rows(rows: Sequence[Mapping[str, Any]]) -> tuple[dict[str, Any], ...]:
    """Turn frozen normalized candidates into a zero-incremental-cost baseline."""

    synthesized: list[dict[str, Any]] = []
    for source in rows:
        candidate = _candidate(source)
        answer = _answer(source)
        synthesized.append(
            {
                "dataset": source.get("dataset"),
                "sample_id": source.get("sample_id"),
                "answer": answer,
                "prediction": candidate,
                "final_prediction": candidate,
                "candidate_answer": candidate,
                "candidate_source": source.get("candidate_source", "normalized"),
                "correct": candidate is not None and answer is not None and candidate == answer,
                "retained_visual_tokens": 0.0,
                "raw_visual_tokens": 0.0,
                "controller_usage": {"total_tokens": 0.0},
                "perception_usage": {"total_tokens": 0.0},
                "usage": {"total_tokens": 0.0},
                "latency_s": 0.0,
                "fallback_to_candidate": False,
                "candidate_rerun": 0,
                "_candidate_only": True,
            }
        )
    return tuple(synthesized)


def bootstrap_mean_ci(
    values: Sequence[float],
    *,
    seed: int = 42,
    samples: int = 10_000,
    confidence: float = 0.95,
) -> dict[str, float | int | None]:
    if not values:
        return {
            "mean": None,
            "lower": None,
            "upper": None,
            "confidence": confidence,
            "bootstrap_samples": samples,
            "seed": seed,
        }
    if samples < 1:
        raise ValueError("bootstrap samples must be positive")
    if not 0 < confidence < 1:
        raise ValueError("bootstrap confidence must be between zero and one")
    finite = [float(value) for value in values if math.isfinite(float(value))]
    if not finite:
        return bootstrap_mean_ci((), seed=seed, samples=samples, confidence=confidence)
    rng = random.Random(seed)
    length = len(finite)
    means = sorted(
        sum(finite[rng.randrange(length)] for _ in range(length)) / length
        for _ in range(samples)
    )

    def percentile(quantile: float) -> float:
        location = (len(means) - 1) * quantile
        lower = math.floor(location)
        upper = math.ceil(location)
        if lower == upper:
            return means[lower]
        weight = location - lower
        return means[lower] * (1.0 - weight) + means[upper] * weight

    alpha = (1.0 - confidence) / 2.0
    return {
        "mean": sum(finite) / len(finite),
        "lower": percentile(alpha),
        "upper": percentile(1.0 - alpha),
        "confidence": confidence,
        "bootstrap_samples": samples,
        "seed": seed,
    }


def _token_metric(
    rows: Sequence[Mapping[str, Any]],
    extractor: Any,
    *,
    bootstrap_seed: int,
    bootstrap_samples: int,
    bootstrap_confidence: float,
) -> dict[str, Any]:
    values = [value for row in rows if (value := extractor(row)) is not None]
    return {
        "mean": sum(values) / len(values) if values else None,
        "total": sum(values),
        "coverage": len(values),
        "bootstrap_mean_ci": bootstrap_mean_ci(
            values,
            seed=bootstrap_seed,
            samples=bootstrap_samples,
            confidence=bootstrap_confidence,
        ),
    }


def summarize_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    bootstrap_seed: int = 42,
    bootstrap_samples: int = 10_000,
    bootstrap_confidence: float = 0.95,
) -> dict[str, Any]:
    correct = sum(_is_evaluable(row) and _prediction(row) == _answer(row) for row in rows)
    valid = sum(_is_evaluable(row) for row in rows)
    parse_failures = sum(_parse_failed(row) for row in rows)
    fallback = sum(bool(row.get("fallback_to_candidate")) for row in rows)
    invalid_predictions = sum(_prediction(row) is None for row in rows)
    candidate_valid = candidate_correct = changed = fixed = harmed = wrong_to_wrong = kept_wrong = 0
    for row in rows:
        candidate = _candidate(row)
        prediction = _prediction(row)
        answer = _answer(row)
        if candidate is None:
            continue
        candidate_valid += 1
        was_correct = answer is not None and candidate == answer
        candidate_correct += was_correct
        if prediction is None or prediction == candidate:
            kept_wrong += answer is not None and not was_correct
            continue
        changed += 1
        is_correct = answer is not None and prediction == answer
        fixed += not was_correct and is_correct
        harmed += was_correct and not is_correct
        wrong_to_wrong += not was_correct and not is_correct
    model_counts = Counter(
        str(row.get("controller_model"))
        for row in rows
        if row.get("controller_model")
    )
    return {
        "records": len(rows),
        "valid_common_eligible": valid,
        "invalid_prediction_count": invalid_predictions,
        "correct": correct,
        "accuracy": correct / len(rows) if rows else None,
        "accuracy_on_valid": correct / valid if valid else None,
        "parse_failure_count": parse_failures,
        "parse_failure_rate": parse_failures / len(rows) if rows else None,
        "fallback_to_candidate": fallback,
        "fallback_rate": fallback / len(rows) if rows else None,
        "candidate": {
            "valid": candidate_valid,
            "correct": candidate_correct,
            "changed": changed,
            "changes_fixed": fixed,
            "changes_harmed": harmed,
            "changed_wrong_to_wrong": wrong_to_wrong,
            "kept_wrong": kept_wrong,
        },
        "tokens": {
            "retained_visual": _token_metric(
                rows,
                _retained_tokens,
                bootstrap_seed=bootstrap_seed,
                bootstrap_samples=bootstrap_samples,
                bootstrap_confidence=bootstrap_confidence,
            ),
            "raw_visual": _token_metric(
                rows,
                _raw_tokens,
                bootstrap_seed=bootstrap_seed + 1,
                bootstrap_samples=bootstrap_samples,
                bootstrap_confidence=bootstrap_confidence,
            ),
            "controller": _token_metric(
                rows,
                lambda row: _usage_tokens(row, "controller"),
                bootstrap_seed=bootstrap_seed + 2,
                bootstrap_samples=bootstrap_samples,
                bootstrap_confidence=bootstrap_confidence,
            ),
            "perception": _token_metric(
                rows,
                lambda row: _usage_tokens(row, "perception"),
                bootstrap_seed=bootstrap_seed + 3,
                bootstrap_samples=bootstrap_samples,
                bootstrap_confidence=bootstrap_confidence,
            ),
            "total": _token_metric(
                rows,
                _total_tokens,
                bootstrap_seed=bootstrap_seed + 4,
                bootstrap_samples=bootstrap_samples,
                bootstrap_confidence=bootstrap_confidence,
            ),
        },
        "latency_s": _token_metric(
            rows,
            _latency,
            bootstrap_seed=bootstrap_seed + 5,
            bootstrap_samples=bootstrap_samples,
            bootstrap_confidence=bootstrap_confidence,
        ),
        "controller_latency_s": _token_metric(
            rows,
            lambda row: _component_latency(row, "controller"),
            bootstrap_seed=bootstrap_seed + 6,
            bootstrap_samples=bootstrap_samples,
            bootstrap_confidence=bootstrap_confidence,
        ),
        "perception_latency_s": _token_metric(
            rows,
            lambda row: _component_latency(row, "perception"),
            bootstrap_seed=bootstrap_seed + 7,
            bootstrap_samples=bootstrap_samples,
            bootstrap_confidence=bootstrap_confidence,
        ),
        "observed_controller_models": dict(sorted(model_counts.items())),
        "budget_distribution": budget_distribution(rows),
    }


def _completion_validation(
    *,
    actual_count: int,
    expected_count: int | None,
    duplicate_count: int,
    invalid_lines: int,
) -> dict[str, Any]:
    count_complete = expected_count is None or actual_count == expected_count
    clean = duplicate_count == 0 and invalid_lines == 0
    return {
        "status": (
            "unchecked"
            if expected_count is None and clean
            else "complete"
            if count_complete and clean
            else "incomplete"
        ),
        "expected_count": expected_count,
        "actual_count": actual_count,
        "count_complete": count_complete,
        "duplicate_sample_id_count": duplicate_count,
        "invalid_jsonl_lines": invalid_lines,
        "selection_eligible": count_complete and clean,
    }


def exact_mcnemar(
    baseline: Sequence[Mapping[str, Any]],
    method: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Exact two-sided McNemar on common evaluable rows; no scipy required."""

    before = {_qualified_id(row): row for row in baseline if _is_evaluable(row)}
    after = {_qualified_id(row): row for row in method if _is_evaluable(row)}
    paired = wins = losses = before_correct = after_correct = answer_mismatch = 0
    for identity in sorted(before.keys() & after.keys()):
        left, right = before[identity], after[identity]
        if _answer(left) != _answer(right):
            answer_mismatch += 1
            continue
        left_correct = _prediction(left) == _answer(left)
        right_correct = _prediction(right) == _answer(right)
        paired += 1
        before_correct += left_correct
        after_correct += right_correct
        wins += not left_correct and right_correct
        losses += left_correct and not right_correct
    discordant = wins + losses
    if discordant:
        lower_tail = sum(
            math.comb(discordant, index)
            for index in range(min(wins, losses) + 1)
        ) / (2**discordant)
        p_value = min(1.0, 2.0 * lower_tail)
    else:
        p_value = 1.0
    return {
        "paired_common_valid": paired,
        "baseline_correct": before_correct,
        "method_correct": after_correct,
        "wins_baseline_wrong_method_correct": wins,
        "losses_baseline_correct_method_wrong": losses,
        "net_correct": wins - losses,
        "discordant": discordant,
        "mcnemar_exact_two_sided_p": p_value,
        "answer_mismatches_excluded": answer_mismatch,
    }


def pareto_frontier(
    method_summaries: Mapping[str, Mapping[str, Any]],
    memberships: Mapping[str, Sequence[str]],
) -> dict[str, Any]:
    frontiers: dict[str, Any] = {}
    for group in ANALYSIS_GROUPS:
        names = sorted(name for name, groups in memberships.items() if group in groups)
        points: list[dict[str, Any]] = []
        for name in names:
            summary = method_summaries[name]["aggregate"]
            point = {
                "method": name,
                "accuracy": summary.get("accuracy"),
                "mean_retained_visual_tokens": summary["tokens"]["retained_visual"]["mean"],
                "selection_eligible": summary.get("selection_eligible", True),
                "exclusion_reason": None,
                "pareto_efficient": None,
                "dominated_by": [],
            }
            points.append(point)
        for point in points:
            if not point["selection_eligible"]:
                point["pareto_efficient"] = False
                point["exclusion_reason"] = "incomplete_result"
                continue
            if point["accuracy"] is None or point["mean_retained_visual_tokens"] is None:
                point["exclusion_reason"] = "missing_accuracy_or_cost"
                continue
            dominators = [
                other["method"]
                for other in points
                if other is not point
                and other["selection_eligible"]
                and other["accuracy"] is not None
                and other["mean_retained_visual_tokens"] is not None
                and float(other["accuracy"]) >= float(point["accuracy"])
                and float(other["mean_retained_visual_tokens"])
                <= float(point["mean_retained_visual_tokens"])
                and (
                    float(other["accuracy"]) > float(point["accuracy"])
                    or float(other["mean_retained_visual_tokens"])
                    < float(point["mean_retained_visual_tokens"])
                )
            ]
            point["dominated_by"] = sorted(dominators)
            point["pareto_efficient"] = not dominators
        frontiers[group] = {
            "points": points,
            "non_dominated_methods": [
                point["method"] for point in points if point["pareto_efficient"] is True
            ],
        }
    return frontiers


def _method_groups(method: SweepMethod) -> tuple[str, ...]:
    if method.kind == "candidate_only" or method.controller_group is None:
        return ("system",)
    if method.controller_group not in {"q9_controller", "q4_controller"}:
        raise ValueError(
            f"method {method.name!r} must declare q9_controller or q4_controller"
        )
    return ("system", str(method.controller_group))


def load_sweep_config(path: str | Path) -> tuple[dict[str, Any], tuple[SweepMethod, ...]]:
    config_path = Path(path).resolve()
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError("sweep config must be a JSON object")
    default_expected = payload.get("expected_count")
    methods: list[SweepMethod] = []
    seen: set[str] = set()
    for item in payload.get("methods") or []:
        if not isinstance(item, dict):
            raise TypeError("each method must be an object")
        name = str(item.get("name") or "").strip()
        if not name or name in seen:
            raise ValueError(f"method name is empty or duplicated: {name!r}")
        seen.add(name)
        kind = str(item.get("kind") or "result")
        if kind not in {"result", "candidate_only"}:
            raise ValueError(f"unsupported method kind: {kind}")
        model = str(item.get("controller_model") or "").strip()
        if not model:
            raise ValueError(f"method {name!r} must declare controller_model")
        resolved: dict[str, tuple[Path, ...]] = {}
        for dataset, entries in (item.get("paths") or {}).items():
            values = entries if isinstance(entries, list) else [entries]
            resolved[str(dataset)] = tuple(
                (config_path.parent / str(value)).resolve()
                if not Path(str(value)).is_absolute()
                else Path(str(value)).resolve()
                for value in values
            )
        expected_counts, expected_total = _parse_expected_count(
            item.get("expected_count", default_expected),
            label=f"methods.{name}.expected_count",
        )
        if expected_counts is not None and set(expected_counts) != set(resolved):
            raise ValueError(
                f"methods.{name}.expected_count datasets must exactly match paths"
            )
        methods.append(
            SweepMethod(
                name=name,
                label=str(item.get("label") or name),
                kind=kind,
                controller_group=item.get("controller_group"),
                controller_model=model,
                paths=resolved,
                expected_counts=expected_counts,
                expected_total=expected_total,
            )
        )
    if not methods:
        raise ValueError("sweep config has no methods")
    return payload, tuple(methods)


def build_sweep_report(config_path: str | Path) -> dict[str, Any]:
    config, specs = load_sweep_config(config_path)
    bootstrap = config.get("bootstrap") or {}
    seed = int(bootstrap.get("seed", 42))
    samples = int(bootstrap.get("samples", 10_000))
    confidence = float(bootstrap.get("confidence", 0.95))
    methods: dict[str, Any] = {}
    records: dict[str, tuple[dict[str, Any], ...]] = {}
    memberships: dict[str, tuple[str, ...]] = {}
    for index, spec in enumerate(specs):
        by_dataset: dict[str, Any] = {}
        all_rows: list[dict[str, Any]] = []
        all_sources: list[str] = []
        all_source_sha256: dict[str, str] = {}
        duplicate_ids: list[str] = []
        invalid_lines = 0
        for dataset, paths in sorted(spec.paths.items()):
            loaded = _read_jsonl(paths, dataset)
            dataset_rows = (
                candidate_only_rows(loaded.rows)
                if spec.kind == "candidate_only"
                else loaded.rows
            )
            dataset_summary = summarize_rows(
                dataset_rows,
                bootstrap_seed=seed + index * 100 + len(by_dataset),
                bootstrap_samples=samples,
                bootstrap_confidence=confidence,
            )
            dataset_completion = _completion_validation(
                actual_count=len(dataset_rows),
                expected_count=(
                    spec.expected_counts.get(dataset)
                    if spec.expected_counts is not None
                    else None
                ),
                duplicate_count=len(loaded.duplicate_ids),
                invalid_lines=loaded.invalid_lines,
            )
            dataset_summary["completion"] = dataset_completion
            dataset_summary["selection_eligible"] = dataset_completion[
                "selection_eligible"
            ]
            by_dataset[dataset] = dataset_summary
            all_rows.extend(dataset_rows)
            all_sources.extend(loaded.source_files)
            all_source_sha256.update(loaded.source_sha256)
            duplicate_ids.extend(loaded.duplicate_ids)
            invalid_lines += loaded.invalid_lines
        records[spec.name] = tuple(all_rows)
        memberships[spec.name] = _method_groups(spec)
        aggregate_summary = summarize_rows(
            all_rows,
            bootstrap_seed=seed + index * 100 + 99,
            bootstrap_samples=samples,
            bootstrap_confidence=confidence,
        )
        datasets_eligible = all(
            summary["selection_eligible"] for summary in by_dataset.values()
        )
        aggregate_completion = _completion_validation(
            actual_count=len(all_rows),
            expected_count=spec.expected_total,
            duplicate_count=len(set(duplicate_ids)),
            invalid_lines=invalid_lines,
        )
        aggregate_completion["selection_eligible"] = (
            aggregate_completion["selection_eligible"] and datasets_eligible
        )
        if not aggregate_completion["selection_eligible"]:
            aggregate_completion["status"] = "incomplete"
        aggregate_summary["completion"] = aggregate_completion
        aggregate_summary["selection_eligible"] = aggregate_completion[
            "selection_eligible"
        ]
        methods[spec.name] = {
            "label": spec.label or spec.name,
            "kind": spec.kind,
            "controller_group": spec.controller_group,
            "controller_model": spec.controller_model,
            "analysis_groups": list(memberships[spec.name]),
            "source_files": all_sources,
            "source_sha256": dict(sorted(all_source_sha256.items())),
            "duplicate_sample_ids": sorted(set(duplicate_ids)),
            "invalid_jsonl_lines": invalid_lines,
            "datasets": by_dataset,
            "aggregate": aggregate_summary,
        }
    frontiers = pareto_frontier(methods, memberships)
    matched_random: dict[str, Any] | None = None
    matched_source = config.get("matched_random_source_method")
    matched_selection_rule = "explicit_source_method"
    bound_summary = config.get("matched_random_from_summary")
    if bound_summary is not None and (
        matched_source is not None or config.get("matched_random_source_candidates")
    ):
        raise ValueError(
            "matched_random_from_summary is mutually exclusive with local source selection"
        )
    if bound_summary is not None:
        if not isinstance(bound_summary, Mapping):
            raise TypeError("matched_random_from_summary must be an object")
        summary_path = Path(str(bound_summary.get("path") or ""))
        if not summary_path.is_absolute():
            summary_path = Path(config_path).resolve().parent / summary_path
        expected_sha = str(bound_summary.get("sha256") or "").lower()
        if len(expected_sha) != 64 or any(
            character not in "0123456789abcdef" for character in expected_sha
        ):
            raise ValueError(
                "matched_random_from_summary.sha256 must be a frozen hexadecimal SHA-256"
            )
        actual_sha = hashlib.sha256(summary_path.read_bytes()).hexdigest()
        if actual_sha != expected_sha:
            raise RuntimeError(
                "matched_random_from_summary SHA-256 mismatch; refusing cost matching"
            )
        base_report = json.loads(summary_path.read_text(encoding="utf-8"))
        base_matched = base_report.get("matched_random")
        if not isinstance(base_matched, Mapping):
            raise ValueError("bound base summary has no matched_random object")
        matched_source = str(base_matched.get("source_policy_id") or "")
        if matched_source not in records:
            raise ValueError(
                f"bound matched-random source is absent from report: {matched_source}"
            )
        if not methods[matched_source]["aggregate"]["selection_eligible"]:
            raise ValueError("bound matched-random source result is incomplete")
        frozen = extract_frozen_budget_distribution(
            records[matched_source], source_method=matched_source
        )
        current_distribution = {
            "0.10": frozen["probabilities"]["R010"],
            "0.25": frozen["probabilities"]["R025"],
            "0.50": frozen["probabilities"]["R050"],
            "1.00": frozen["probabilities"]["R100"],
        }
        if base_matched.get("budget_distribution") != current_distribution:
            raise RuntimeError(
                "matched-random source distribution changed after base summary freeze"
            )
        if base_matched.get("distribution_sha256") != frozen["distribution_sha256"]:
            raise RuntimeError(
                "matched-random source distribution fingerprint changed"
            )
        source_sha = methods[matched_source]["source_sha256"]
        if base_matched.get("source_result_sha256") != source_sha:
            raise RuntimeError("matched-random source result SHA-256 changed")
        matched_random = {
            **dict(base_matched),
            "source_selection_rule": "frozen_from_base_summary",
            "bound_base_summary": str(summary_path.resolve()),
            "bound_base_summary_sha256": actual_sha,
        }
    elif matched_source is None and config.get("matched_random_source_candidates"):
        candidates = [
            str(name)
            for name in config["matched_random_source_candidates"]
        ]
        unknown = set(candidates) - set(methods)
        if not candidates or unknown:
            raise ValueError(
                f"invalid matched_random_source_candidates; unknown={sorted(unknown)}"
            )
        eligible: list[tuple[str, float, float]] = []
        for name in candidates:
            summary = methods[name]["aggregate"]
            if not summary["selection_eligible"]:
                continue
            accuracy = summary.get("accuracy")
            cost = summary["tokens"]["retained_visual"]["mean"]
            if accuracy is not None and cost is not None:
                eligible.append((name, float(accuracy), float(cost)))
        if not eligible:
            raise ValueError("no matched-random source candidate has accuracy and cost")
        matched_source = min(
            eligible,
            key=lambda item: (-item[1], item[2], item[0]),
        )[0]
        matched_selection_rule = (
            "highest_dev_accuracy_then_lowest_mean_retained_visual_tokens"
        )
    if matched_source is not None and matched_random is None:
        matched_source = str(matched_source)
        if matched_source not in records:
            raise ValueError(
                f"unknown matched_random_source_method: {matched_source}"
            )
        if not methods[matched_source]["aggregate"]["selection_eligible"]:
            raise ValueError("matched-random source result is incomplete")
        frozen = extract_frozen_budget_distribution(
            records[matched_source],
            source_method=matched_source,
        )
        probabilities = frozen["probabilities"]
        matched_random = {
            "source_policy_id": matched_source,
            "source_selection_rule": matched_selection_rule,
            "budget_distribution": {
                "0.10": probabilities["R010"],
                "0.25": probabilities["R025"],
                "0.50": probabilities["R050"],
                "1.00": probabilities["R100"],
            },
            "source_step_count": frozen["step_count"],
            "source_mean_retained_visual_tokens": frozen[
                "mean_retained_visual_tokens_per_sample"
            ],
            "distribution_sha256": frozen["distribution_sha256"],
            "source_result_sha256": methods[matched_source]["source_sha256"],
        }
    cost_match: dict[str, Any] | None = None
    cost_match_config = config.get("cost_match")
    if cost_match_config is not None:
        if not isinstance(cost_match_config, Mapping):
            raise TypeError("cost_match must be an object")
        target = str(cost_match_config.get("target_method") or "")
        if not target and cost_match_config.get("target_from_matched_random"):
            if matched_random is None:
                raise ValueError(
                    "cost_match target_from_matched_random requires matched_random"
                )
            target = str(matched_random["source_policy_id"])
        candidates = [
            str(name) for name in cost_match_config.get("candidate_methods") or []
        ]
        unknown = ({target} | set(candidates)) - set(methods)
        if not target or not candidates or unknown:
            raise ValueError(
                f"invalid cost_match methods; unknown={sorted(unknown)}"
            )
        target_cost = methods[target]["aggregate"]["tokens"][
            "retained_visual"
        ]["mean"]
        if target_cost is None:
            raise ValueError("cost_match target has no retained visual-token cost")
        if not methods[target]["aggregate"]["selection_eligible"]:
            raise ValueError("cost_match target result is incomplete")
        points: list[dict[str, Any]] = []
        for candidate in candidates:
            if not methods[candidate]["aggregate"]["selection_eligible"]:
                raise ValueError(f"cost_match candidate result is incomplete: {candidate}")
            candidate_cost = methods[candidate]["aggregate"]["tokens"][
                "retained_visual"
            ]["mean"]
            if candidate_cost is None:
                raise ValueError(
                    f"cost_match candidate has no cost: {candidate}"
                )
            points.append(
                {
                    "method": candidate,
                    "mean_retained_visual_tokens": candidate_cost,
                    "absolute_cost_gap": abs(
                        float(candidate_cost) - float(target_cost)
                    ),
                }
            )
        selected = min(
            points,
            key=lambda point: (point["absolute_cost_gap"], point["method"]),
        )
        cost_match = {
            "selection_rule": "minimum_absolute_mean_retained_visual_token_gap",
            "accuracy_used_for_selection": False,
            "target_method": target,
            "target_mean_retained_visual_tokens": target_cost,
            "candidates": points,
            "selected_method": selected["method"],
        }
    pairwise: dict[str, Any] = {}
    for group in ANALYSIS_GROUPS:
        names = sorted(name for name, groups in memberships.items() if group in groups)
        for left_index, baseline in enumerate(names):
            for method in names[left_index + 1 :]:
                key = f"{group}:{method}__vs__{baseline}"
                pairwise[key] = {
                    "group": group,
                    "baseline": baseline,
                    "method": method,
                    **exact_mcnemar(records[baseline], records[method]),
                }
    canonical = json.dumps(config, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return {
        "schema_version": 1,
        "config_path": str(Path(config_path).resolve()),
        "config_sha256": hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
        "cost_definition": "incremental retained visual tokens beyond frozen Direct candidate",
        "bootstrap": {"seed": seed, "samples": samples, "confidence": confidence},
        "methods": methods,
        "pareto": frontiers,
        "paired_common_valid": pairwise,
        "matched_random": matched_random,
        "cost_matched_random": cost_match,
    }


def render_markdown(report: Mapping[str, Any]) -> str:
    def fmt(value: Any, digits: int = 4) -> str:
        number = _finite_number(value)
        return "-" if number is None else f"{number:.{digits}f}"

    lines = [
        "# FlashVID budget sweep: offline Pareto report",
        "",
        f"Config SHA-256: `{report['config_sha256']}`",
        "",
        "Costs are incremental beyond the frozen Direct candidate. Candidate-only therefore has zero incremental visual cost.",
        "",
    ]
    for group in ANALYSIS_GROUPS:
        lines.extend(
            [
                f"## {group} Pareto frontier",
                "",
                "| Method | Controller | Correct / N | Accuracy | Mean retained visual tokens | Parse failures | Fallbacks | Pareto | Dominated by |",
                "|---|---|---:|---:|---:|---:|---:|---|---|",
            ]
        )
        for point in report["pareto"][group]["points"]:
            method = report["methods"][point["method"]]
            summary = method["aggregate"]
            accuracy = summary["accuracy"]
            cost = summary["tokens"]["retained_visual"]["mean"]
            lines.append(
                f"| {point['method']} | {method['controller_model']} | "
                f"{summary['correct']} / {summary['records']} | "
                f"{fmt(accuracy)} | {fmt(cost, 2)} | {summary['parse_failure_count']} | "
                f"{summary['fallback_to_candidate']} | "
                f"{'yes' if point['pareto_efficient'] else 'no'} | "
                f"{', '.join(point['dominated_by']) or '-'} |"
            )
        lines.append("")
    lines.extend(
        [
            "## Executed budget distributions",
            "",
            "| Method | R010 | R025 | R050 | R100 | Mean ratio | Mean retained visual tokens |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for name, method in report["methods"].items():
        distribution = method["aggregate"]["budget_distribution"]
        cells = []
        for label, _ in BUDGET_LEVELS:
            count = distribution["counts"][label]
            probability = distribution["probabilities"][label]
            cells.append(f"{count} ({probability:.1%})")
        lines.append(
            f"| {name} | {' | '.join(cells)} | "
            f"{fmt(distribution['mean_retention_ratio'])} | "
            f"{fmt(distribution['mean_retained_visual_tokens_per_sample'], 2)} |"
        )
    lines.extend(
        [
            "",
            "## Candidate change diagnostics",
            "",
            "| Method | Changed | Fixed | Harmed | Wrong-to-wrong | Kept wrong |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for name, method in report["methods"].items():
        candidate = method["aggregate"]["candidate"]
        lines.append(
            f"| {name} | {candidate['changed']} | {candidate['changes_fixed']} | "
            f"{candidate['changes_harmed']} | {candidate['changed_wrong_to_wrong']} | "
            f"{candidate['kept_wrong']} |"
        )
    lines.extend(["", "## Paired common-valid McNemar tests", ""])
    lines.extend(
        [
            "| Group | Method vs baseline | Common valid | Wins | Losses | Exact p |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    for result in report["paired_common_valid"].values():
        lines.append(
            f"| {result['group']} | {result['method']} vs {result['baseline']} | "
            f"{result['paired_common_valid']} | "
            f"{result['wins_baseline_wrong_method_correct']} | "
            f"{result['losses_baseline_correct_method_wrong']} | "
            f"{result['mcnemar_exact_two_sided_p']:.6g} |"
        )
    if report.get("cost_matched_random"):
        match = report["cost_matched_random"]
        lines.extend(
            [
                "",
                "## Cost-matched random selection",
                "",
                (
                    f"Selected `{match['selected_method']}` for "
                    f"`{match['target_method']}` by minimum absolute mean visual-token "
                    "cost gap; accuracy was not used for selection."
                ),
            ]
        )
    return "\n".join(lines) + "\n"
