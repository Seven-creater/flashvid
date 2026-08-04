"""Deterministic control plane for Fast Hybrid EVA trajectory SFT data.

This module does not run the model.  It freezes base/rescue jobs, validates
completed unscored trajectories after an offline label join, selects the
lowest-cost stable positive per sample, and emits replay specifications for
counterfactual compression.
"""

from __future__ import annotations

import hashlib
import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import Any, Iterable, Mapping, Sequence

from .privacy import assert_deferred_result_public
from .qwen_sft import (
    canonical_sha256,
    composite_trajectory_id,
)


SCHEMA_VERSION = 1
BASE_VISUAL_BUDGETS = (6_000, 12_000, 18_000, 24_000)
BASE_PLANNER_SEEDS = (17, 42, 73)
JUDGE_SEEDS = (17, 42, 73)
FRAME_FRACTIONS = (0.75, 0.50, 0.25)
SFT_MIN_TOTAL = 300
SFT_MIN_PER_DATASET = 80
_DATASETS = ("lvbench", "lsdbench", "cgbench")
_PRIVATE_OUTPUT_KEYS = {
    "answer",
    "correct_answer",
    "right_answer",
    "ground_truth",
    "gt",
    "correct",
    "time_range",
    "clue_intervals",
    "question_type",
}


@dataclass(frozen=True)
class SelectionOutcome:
    selected: tuple[dict[str, Any], ...]
    no_positive_sample_ids: tuple[str, ...]
    rejected: dict[str, int]
    gate: dict[str, Any]


@dataclass(frozen=True)
class CompressionExecutionState:
    """One fail-closed scheduling snapshot for a compression replay DAG."""

    ready: tuple[dict[str, Any], ...]
    completed_fingerprints: tuple[str, ...]
    skipped_fingerprints: tuple[str, ...]
    blocked_fingerprints: tuple[str, ...]
    complete: bool


@dataclass(frozen=True)
class PreJudgeOutcome:
    eligible_specs: tuple[dict[str, Any], ...]
    eligible_trajectories: tuple[dict[str, Any], ...]
    completion_index: tuple[dict[str, Any], ...]
    rejected: dict[str, int]


@dataclass(frozen=True)
class CompressionReplayAnalysis:
    """Offline 3/3 decisions for compression nodes that have finished."""

    outcomes: dict[str, str]
    representatives: tuple[dict[str, Any], ...]
    rejection_reasons: dict[str, str]
    incomplete_fingerprints: tuple[str, ...]


@dataclass(frozen=True)
class CompressionFinalization:
    """One resumable snapshot across every selected trajectory DAG."""

    ready: tuple[dict[str, Any], ...]
    outcomes: dict[str, str]
    representatives: tuple[dict[str, Any], ...]
    rejection_reasons: dict[str, str]
    incomplete_fingerprints: tuple[str, ...]
    selected_pruned: tuple[dict[str, Any], ...]
    gate: dict[str, Any]
    complete: bool


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _require_sha256(value: str, label: str) -> str:
    normalized = str(value).strip().lower()
    if len(normalized) != 64 or any(c not in "0123456789abcdef" for c in normalized):
        raise ValueError(f"{label} must be a SHA-256")
    return normalized


def controller_fingerprint(
    *,
    manifest_sha256: str,
    config_sha256: str,
    budgets: Sequence[int] = BASE_VISUAL_BUDGETS,
    planner_seeds: Sequence[int] = BASE_PLANNER_SEEDS,
    dataset_manifest_sha256s: Mapping[str, str] | None = None,
) -> str:
    """Bind every job and resume row to immutable controller inputs."""

    payload = {
        "schema_version": SCHEMA_VERSION,
        "manifest_sha256": _require_sha256(manifest_sha256, "manifest_sha256"),
        "config_sha256": _require_sha256(config_sha256, "config_sha256"),
        "budgets": _validated_budgets(budgets),
        "planner_seeds": _validated_seeds(planner_seeds, "planner_seeds"),
        "dataset_manifest_sha256s": _dataset_hashes(dataset_manifest_sha256s),
        "judge_seeds": list(JUDGE_SEEDS),
        "implementation_sha256": _sha256_file(Path(__file__)),
    }
    return canonical_sha256(payload)


def _dataset_hashes(values: Mapping[str, str] | None) -> dict[str, str]:
    if not values:
        return {}
    if set(values) != set(_DATASETS):
        raise ValueError(
            "dataset manifest hashes must cover lvbench, lsdbench and cgbench"
        )
    return {
        dataset: _require_sha256(values[dataset], f"{dataset}_manifest_sha256")
        for dataset in _DATASETS
    }


def _validated_budgets(values: Sequence[int]) -> list[int]:
    budgets = [int(value) for value in values]
    if (
        len(budgets) != 4
        or len(set(budgets)) != 4
        or any(value <= 0 for value in budgets)
    ):
        raise ValueError("exactly four unique positive visual budgets are required")
    return budgets


def _validated_seeds(values: Sequence[int], label: str) -> list[int]:
    seeds = [int(value) for value in values]
    if len(seeds) != 3 or len(set(seeds)) != 3:
        raise ValueError(f"{label} must contain exactly three unique seeds")
    return seeds


def _samples(records: Iterable[Mapping[str, Any]]) -> tuple[dict[str, str], ...]:
    samples: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for index, row in enumerate(records):
        dataset = str(row.get("dataset") or "").strip().lower()
        sample_id = str(row.get("sample_id") or "").strip()
        if dataset not in _DATASETS or not sample_id:
            raise ValueError(f"invalid Train600 row {index}")
        identity = (dataset, sample_id)
        if identity in seen:
            raise ValueError(f"duplicate Train600 sample: {dataset}/{sample_id}")
        seen.add(identity)
        samples.append({"dataset": dataset, "sample_id": sample_id})
    return tuple(sorted(samples, key=lambda row: (row["dataset"], row["sample_id"])))


def _run_spec(payload: dict[str, Any]) -> dict[str, Any]:
    spec = dict(payload)
    spec["run_spec_fingerprint"] = canonical_sha256(spec)
    return spec


def build_base_run_specs(
    records: Iterable[Mapping[str, Any]],
    *,
    manifest_sha256: str,
    config_sha256: str,
    budgets: Sequence[int] = BASE_VISUAL_BUDGETS,
    planner_seeds: Sequence[int] = BASE_PLANNER_SEEDS,
    dataset_manifest_sha256s: Mapping[str, str] | None = None,
) -> tuple[dict[str, Any], ...]:
    """Create exactly four budgets x three seeds for every Train600 sample."""

    manifest_hash = _require_sha256(manifest_sha256, "manifest_sha256")
    config_hash = _require_sha256(config_sha256, "config_sha256")
    budget_values = _validated_budgets(budgets)
    seed_values = _validated_seeds(planner_seeds, "planner_seeds")
    dataset_hashes = _dataset_hashes(dataset_manifest_sha256s)
    fingerprint = controller_fingerprint(
        manifest_sha256=manifest_hash,
        config_sha256=config_hash,
        budgets=budget_values,
        planner_seeds=seed_values,
        dataset_manifest_sha256s=dataset_hashes,
    )
    specs: list[dict[str, Any]] = []
    for sample in _samples(records):
        for budget in budget_values:
            for seed in seed_values:
                schedule_id = f"budget_{budget:06d}_seed_{seed}"
                trajectory_id = composite_trajectory_id(
                    sample["dataset"], sample["sample_id"], schedule_id, 0
                )
                specs.append(
                    _run_spec(
                        {
                            "schema_version": SCHEMA_VERSION,
                            "phase": "base",
                            **sample,
                            "schedule_id": schedule_id,
                            "variant_id": "base",
                            "family_id": schedule_id,
                            "replica_id": "0",
                            "trajectory_id": trajectory_id,
                            "planner_seed": seed,
                            "max_total_visual_tokens": budget,
                            "max_call_visual_tokens": min(12_000, budget),
                            "max_turns": 6,
                            "required_judge_seeds": list(JUDGE_SEEDS),
                            "manifest_sha256": manifest_hash,
                            "train600_manifest_sha256": manifest_hash,
                            "dataset_manifest_sha256": dataset_hashes.get(
                                sample["dataset"], manifest_hash
                            ),
                            "config_sha256": config_hash,
                            "controller_fingerprint": fingerprint,
                        }
                    )
                )
    return tuple(specs)


def pending_run_specs(
    specs: Sequence[Mapping[str, Any]],
    completed: Iterable[Mapping[str, Any]],
    *,
    expected_controller_fingerprint: str,
) -> tuple[dict[str, Any], ...]:
    """Validate a resume file and return only jobs that have not completed."""

    expected = {str(spec["trajectory_id"]): dict(spec) for spec in specs}
    if len(expected) != len(specs):
        raise ValueError("run plan contains duplicate trajectory_id")
    seen: set[str] = set()
    for row in completed:
        if row.get("controller_fingerprint") != expected_controller_fingerprint:
            raise RuntimeError("resume controller fingerprint mismatch")
        trajectory_id = str(row.get("trajectory_id") or "")
        if trajectory_id not in expected:
            raise RuntimeError(
                f"resume contains an unknown trajectory: {trajectory_id}"
            )
        if trajectory_id in seen:
            raise RuntimeError(f"duplicate resume trajectory: {trajectory_id}")
        if row.get("run_spec_fingerprint") != expected[trajectory_id].get(
            "run_spec_fingerprint"
        ):
            raise RuntimeError(f"resume run-spec fingerprint mismatch: {trajectory_id}")
        seen.add(trajectory_id)
    return tuple(expected[key] for key in sorted(set(expected) - seen))


def _prediction(row: Mapping[str, Any]) -> str | None:
    value = (
        str(row.get("final_prediction", row.get("prediction")) or "").strip().upper()
    )
    return value if len(value) == 1 and "A" <= value <= "H" else None


def _frame_steps(row: Mapping[str, Any]) -> list[dict[str, Any]]:
    raw_steps = row.get("tool_steps", row.get("tool_calls"))
    if not isinstance(raw_steps, list):
        return []
    steps: list[dict[str, Any]] = []
    for raw in raw_steps:
        if not isinstance(raw, Mapping):
            continue
        try:
            start = float(raw["start_time"])
            end = float(raw["end_time"])
            nframes = int(raw["nframes"])
            resize = float(raw.get("resize", 1.0))
        except (KeyError, TypeError, ValueError):
            continue
        timestamps = raw.get("actual_timestamps", raw.get("timestamps"))
        if (
            end <= start
            or nframes <= 0
            or resize <= 0
            or not isinstance(timestamps, list)
            or not timestamps
        ):
            continue
        steps.append(
            {
                **dict(raw),
                "start_time": start,
                "end_time": end,
                "nframes": nframes,
                "resize": resize,
                "actual_timestamps": [float(value) for value in timestamps],
            }
        )
    return steps


def positive_rejection_reason(
    row: Mapping[str, Any],
    correct_answer: str,
    *,
    judge_seeds: Sequence[int] = JUDGE_SEEDS,
) -> str | None:
    """Return why a scored trajectory cannot become a positive SFT source."""

    answer = str(correct_answer).strip().upper()
    if len(answer) != 1 or not "A" <= answer <= "H":
        raise ValueError("correct_answer must be A-H")
    if row.get("annotation_leak_check") != "passed":
        return "annotation_leak"
    if int(row.get("candidate_rerun") or 0) != 0:
        return "candidate_rerun"
    if any(
        row.get(key) for key in ("error", "api_error", "frame_error", "parse_error")
    ):
        return "engineering_or_parse_error"
    if _prediction(row) != answer:
        return "incorrect"
    if not _frame_steps(row):
        return "no_frame_select"
    candidate = str(row.get("candidate_answer") or "").strip().upper()
    if candidate and candidate != answer and row.get("fallback_to_candidate") is True:
        return "wrong_candidate_fallback"

    expected_seeds = {int(seed) for seed in judge_seeds}
    confirmations = row.get("judge_confirmations")
    if not isinstance(confirmations, list) or len(confirmations) != len(expected_seeds):
        return "incomplete_judge_confirmation"
    observed: set[int] = set()
    for confirmation in confirmations:
        if not isinstance(confirmation, Mapping):
            return "incomplete_judge_confirmation"
        try:
            seed = int(confirmation["judge_seed"])
        except (KeyError, TypeError, ValueError):
            return "incomplete_judge_confirmation"
        if seed in observed or seed not in expected_seeds:
            return "incomplete_judge_confirmation"
        observed.add(seed)
        if confirmation.get("annotation_leak_check") != "passed":
            return "annotation_leak"
        if (
            confirmation.get("fallback_used") is True
            or confirmation.get("fallback_to_candidate") is True
        ):
            return "judge_fallback"
        if any(
            confirmation.get(key)
            for key in ("error", "api_error", "frame_error", "parse_error")
        ):
            return "judge_failure"
        if _prediction(confirmation) != answer:
            return "unstable_judges"
    if observed != expected_seeds:
        return "incomplete_judge_confirmation"
    return None


def _prejudge_rejection_reason(
    row: Mapping[str, Any], correct_answer: str
) -> str | None:
    if row.get("scoring_deferred") is not True:
        return "not_deferred"
    if row.get("annotation_leak_check") != "passed":
        return "annotation_leak"
    if int(row.get("candidate_rerun") or 0) != 0:
        return "candidate_rerun"
    if row.get("candidate_cost_complete") is False:
        return "candidate_cost_incomplete"
    if any(
        row.get(key) for key in ("error", "api_error", "frame_error", "parse_error")
    ):
        return "engineering_or_parse_error"
    if _prediction(row) != str(correct_answer).strip().upper():
        return "incorrect"
    if not _frame_steps(row):
        return "no_frame_select"
    candidate = str(row.get("candidate_answer") or "").strip().upper()
    if (
        candidate
        and candidate != str(correct_answer).strip().upper()
        and row.get("fallback_to_candidate") is True
    ):
        return "wrong_candidate_fallback"
    return None


def prepare_prejudge_candidates(
    specs: Sequence[Mapping[str, Any]],
    raw_trajectories: Iterable[Mapping[str, Any]],
    answers: Mapping[tuple[str, str], str],
) -> PreJudgeOutcome:
    """Offline-label raw runs and expose only public successful traces to Judges."""

    spec_by_key: dict[tuple[str, str, str, int], dict[str, Any]] = {}
    for raw_spec in specs:
        spec = dict(raw_spec)
        key = (
            str(spec.get("dataset") or "").lower(),
            str(spec.get("sample_id") or ""),
            str(spec.get("schedule_id") or ""),
            int(spec.get("planner_seed")),
        )
        if not all(key[:3]) or key in spec_by_key:
            raise ValueError("pre-Judge specs contain a missing/duplicate identity")
        spec_by_key[key] = spec

    raw_by_key: dict[tuple[str, str, str, int], dict[str, Any]] = {}
    for raw_row in raw_trajectories:
        row = dict(raw_row)
        assert_deferred_result_public(row)
        key = (
            str(row.get("dataset") or "").lower(),
            str(row.get("sample_id") or ""),
            str(
                row.get("trajectory_schedule_id", row.get("schedule_id")) or ""
            ),
            int(row.get("generation_seed", row.get("planner_seed", -1))),
        )
        if key not in spec_by_key:
            raise ValueError(f"raw trajectory is outside its frozen run plan: {key}")
        if key in raw_by_key:
            raise ValueError(f"duplicate raw trajectory: {key}")
        raw_by_key[key] = row
    missing = sorted(set(spec_by_key) - set(raw_by_key))
    if missing:
        raise RuntimeError(f"raw trajectory matrix is incomplete: {missing[:3]}")

    eligible_specs: list[dict[str, Any]] = []
    eligible_rows: list[dict[str, Any]] = []
    completion: list[dict[str, Any]] = []
    rejected: Counter[str] = Counter()
    provenance_fields = (
        "schema_version",
        "phase",
        "dataset",
        "sample_id",
        "schedule_id",
        "variant_id",
        "family_id",
        "replica_id",
        "trajectory_id",
        "planner_seed",
        "max_total_visual_tokens",
        "max_call_visual_tokens",
        "max_turns",
        "required_judge_seeds",
        "manifest_sha256",
        "train600_manifest_sha256",
        "dataset_manifest_sha256",
        "config_sha256",
        "controller_fingerprint",
        "run_spec_fingerprint",
    )
    for key in sorted(spec_by_key):
        spec = spec_by_key[key]
        row = raw_by_key[key]
        identity = (key[0], key[1])
        answer = answers.get(identity)
        if answer is None:
            raise ValueError(f"run plan is outside Train600: {identity}")
        reason = _prejudge_rejection_reason(row, answer)
        trajectory_id = str(spec["trajectory_id"])
        index_row = {
            field: spec.get(field)
            for field in provenance_fields
            if field in spec
        }
        index_row.update(
            {
                "trajectory_id": trajectory_id,
                "prejudge_status": "eligible" if reason is None else "rejected",
                "prejudge_rejection_reason": reason,
            }
        )
        completion.append(index_row)
        if reason is not None:
            rejected[reason] += 1
            continue
        public_row = dict(row)
        public_row.update(
            {
                field: spec.get(field)
                for field in provenance_fields
                if field in spec
            }
        )
        public_row["trajectory_id"] = trajectory_id
        public_row["tool_steps"] = _frame_steps(row)
        public_row["_offline_prejudge_passed"] = True
        assert_deferred_result_public(public_row)
        eligible_specs.append(spec)
        eligible_rows.append(public_row)
    return PreJudgeOutcome(
        eligible_specs=tuple(eligible_specs),
        eligible_trajectories=tuple(eligible_rows),
        completion_index=tuple(completion),
        rejected=dict(sorted(rejected.items())),
    )


def validate_prejudge_coverage(
    specs: Sequence[Mapping[str, Any]],
    completion_index: Iterable[Mapping[str, Any]],
    judged_trajectories: Iterable[Mapping[str, Any]],
) -> None:
    expected = {str(spec.get("trajectory_id") or "") for spec in specs}
    if "" in expected or len(expected) != len(specs):
        raise ValueError("run plan contains missing/duplicate trajectory IDs")
    indexed: dict[str, str] = {}
    for row in completion_index:
        trajectory_id = str(row.get("trajectory_id") or "")
        status = str(row.get("prejudge_status") or "")
        if trajectory_id not in expected or trajectory_id in indexed:
            raise ValueError("pre-Judge index has an unknown/duplicate trajectory")
        if status not in {"eligible", "rejected"}:
            raise ValueError("pre-Judge index has an invalid status")
        indexed[trajectory_id] = status
    if set(indexed) != expected:
        raise RuntimeError("pre-Judge index does not cover the frozen run plan")
    judged_ids: set[str] = set()
    for row in judged_trajectories:
        trajectory_id = str(row.get("trajectory_id") or "")
        if trajectory_id in judged_ids:
            raise ValueError("Judged trajectories contain a duplicate ID")
        judged_ids.add(trajectory_id)
    eligible = {key for key, status in indexed.items() if status == "eligible"}
    if judged_ids != eligible:
        raise RuntimeError("Judged trajectories do not exactly cover eligible raw runs")


def _without_labels(row: Mapping[str, Any]) -> dict[str, Any]:
    clean = {
        key: value for key, value in row.items() if key not in _PRIVATE_OUTPUT_KEYS
    }
    clean["scoring_deferred"] = True
    clean["_offline_selection_passed"] = True
    clean["_selection_stable"] = True
    clean["_selection_confirmation_count"] = len(JUDGE_SEEDS)
    clean["tool_steps"] = _frame_steps(row)
    return clean


def _cost_number(row: Mapping[str, Any], key: str) -> float:
    value = row.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"trajectory requires numeric {key}")
    result = float(value)
    if not math.isfinite(result) or result < 0:
        raise ValueError(f"trajectory has invalid {key}")
    return result


def _end_to_end_cost_key(row: Mapping[str, Any]) -> tuple[float, float, int, float, str]:
    """Rank complete-system cost, including the frozen Direct candidate."""

    return (
        _cost_number(row, "end_to_end_total_tokens"),
        _cost_number(row, "end_to_end_visual_tokens"),
        len(_frame_steps(row)),
        _cost_number(row, "end_to_end_latency_s"),
        str(row.get("trajectory_id") or ""),
    )


def sft_start_gate(
    selected: Iterable[Mapping[str, Any]],
    *,
    min_total: int = SFT_MIN_TOTAL,
    min_per_dataset: int = SFT_MIN_PER_DATASET,
) -> dict[str, Any]:
    counts = Counter(str(row.get("dataset") or "").lower() for row in selected)
    conditions = {
        "total_at_least_300": sum(counts.values()) >= min_total,
        "each_dataset_at_least_80": all(
            counts.get(dataset, 0) >= min_per_dataset for dataset in _DATASETS
        ),
    }
    return {
        "passed": all(conditions.values()),
        "selected_total": sum(counts.values()),
        "selected_by_dataset": {
            dataset: counts.get(dataset, 0) for dataset in _DATASETS
        },
        "minimum_total": min_total,
        "minimum_per_dataset": min_per_dataset,
        "conditions": conditions,
    }


def select_lowest_cost_positives(
    trajectories: Iterable[Mapping[str, Any]],
    answers: Mapping[tuple[str, str], str],
) -> SelectionOutcome:
    """Offline-join labels and choose the cheapest 3/3-confirmed trajectory."""

    positive: dict[tuple[str, str], list[dict[str, Any]]] = {}
    rejected: Counter[str] = Counter()
    seen_ids: set[str] = set()
    for raw in trajectories:
        row = dict(raw)
        trajectory_id = str(row.get("trajectory_id") or "")
        if not trajectory_id or trajectory_id in seen_ids:
            raise ValueError(f"missing or duplicate trajectory_id: {trajectory_id}")
        seen_ids.add(trajectory_id)
        identity = (
            str(row.get("dataset") or "").lower(),
            str(row.get("sample_id") or ""),
        )
        answer = answers.get(identity)
        if answer is None:
            raise ValueError(f"trajectory is outside Train600: {identity}")
        reason = positive_rejection_reason(row, answer)
        if reason is not None:
            rejected[reason] += 1
            continue
        positive.setdefault(identity, []).append(row)

    selected: list[dict[str, Any]] = []
    no_positive: list[str] = []
    for identity in sorted(answers):
        choices = positive.get(identity, [])
        if not choices:
            no_positive.append(f"{identity[0]}/{identity[1]}")
            continue
        winner = min(
            choices,
            key=_end_to_end_cost_key,
        )
        selected_row = _without_labels(winner)
        selected_row["_selection_cost_basis"] = "end_to_end"
        selected_row["_selection_source"] = "base"
        selected.append(selected_row)
    gate = sft_start_gate(selected)
    return SelectionOutcome(
        selected=tuple(selected),
        no_positive_sample_ids=tuple(no_positive),
        rejected=dict(sorted(rejected.items())),
        gate=gate,
    )


def build_rescue_run_specs(
    records: Iterable[Mapping[str, Any]],
    *,
    no_positive_sample_ids: Iterable[str],
    manifest_sha256: str,
    config_sha256: str,
    controller_sha256: str,
    dataset_manifest_sha256s: Mapping[str, str] | None = None,
) -> tuple[dict[str, Any], ...]:
    """Create exactly four deterministic higher-coverage jobs per failed sample."""

    samples = {f"{row['dataset']}/{row['sample_id']}": row for row in _samples(records)}
    missing = sorted(set(no_positive_sample_ids))
    unknown = sorted(set(missing) - set(samples))
    if unknown:
        raise ValueError(f"unknown rescue samples: {unknown[:3]}")
    dataset_hashes = _dataset_hashes(dataset_manifest_sha256s)
    profiles = tuple(
        (budget, seed) for budget in (32_000, 48_000) for seed in (101, 211)
    )
    specs: list[dict[str, Any]] = []
    for identity in missing:
        sample = samples[identity]
        for index, (total_budget, seed) in enumerate(profiles):
            schedule_id = f"rescue_{index + 1}_budget_{total_budget:06d}_seed_{seed}"
            specs.append(
                _run_spec(
                    {
                        "schema_version": SCHEMA_VERSION,
                        "phase": "rescue",
                        **sample,
                        "schedule_id": schedule_id,
                        "variant_id": "rescue",
                        "family_id": f"{schedule_id}~rescue",
                        "replica_id": "0",
                        "trajectory_id": composite_trajectory_id(
                            sample["dataset"],
                            sample["sample_id"],
                            f"{schedule_id}~rescue",
                            0,
                        ),
                        "planner_seed": seed,
                        "max_total_visual_tokens": total_budget,
                        "max_call_visual_tokens": total_budget // 2,
                        "max_turns": 8,
                        "required_judge_seeds": list(JUDGE_SEEDS),
                        "manifest_sha256": _require_sha256(
                            manifest_sha256, "manifest_sha256"
                        ),
                        "train600_manifest_sha256": _require_sha256(
                            manifest_sha256, "manifest_sha256"
                        ),
                        "dataset_manifest_sha256": dataset_hashes.get(
                            sample["dataset"],
                            _require_sha256(manifest_sha256, "manifest_sha256"),
                        ),
                        "config_sha256": _require_sha256(
                            config_sha256, "config_sha256"
                        ),
                        "controller_fingerprint": _require_sha256(
                            controller_sha256, "controller_sha256"
                        ),
                    }
                )
            )
    return tuple(specs)


def _normalized_for_replay(row: Mapping[str, Any]) -> dict[str, Any]:
    normalized = dict(row)
    normalized["tool_steps"] = _frame_steps(row)
    if not normalized["tool_steps"]:
        raise ValueError("compression source has no frame_select step")
    return normalized


def _extra_replay_spec(
    trajectory: Mapping[str, Any],
    *,
    variant_id: str,
    planned_calls: Sequence[Mapping[str, Any]],
    stage: str,
    stage_order: int,
    execution_order: int,
    parent_fingerprint: str,
    stop_policy: str,
    requires_failure_of: str | None = None,
    scale_fraction: float | None = None,
) -> dict[str, Any]:
    dataset = str(trajectory["dataset"])
    sample_id = str(trajectory["sample_id"])
    schedule_id = str(trajectory["schedule_id"])
    family_id = f"{schedule_id}~{variant_id}"
    payload = {
        "schema_version": SCHEMA_VERSION,
        "dataset": dataset,
        "sample_id": sample_id,
        "schedule_id": schedule_id,
        "variant_id": variant_id,
        "family_id": family_id,
        "base_trajectory_id": trajectory["trajectory_id"],
        "manifest_sha256": trajectory["manifest_sha256"],
        "train600_manifest_sha256": trajectory["train600_manifest_sha256"],
        "dataset_manifest_sha256": trajectory.get("dataset_manifest_sha256"),
        "config_sha256": trajectory["config_sha256"],
        "stage": stage,
        "stage_order": stage_order,
        "execution_order": execution_order,
        "execution_policy": "sequential_dependency_gated",
        "parallel_safe": False,
        "parent_fingerprint": parent_fingerprint,
        "requires_parent_status": "passed",
        "on_dependency_failure": "skip_branch",
        "stop_policy": stop_policy,
        "planned_calls": [dict(call) for call in planned_calls],
        "replica_trajectory_ids": [
            composite_trajectory_id(dataset, sample_id, family_id, replica)
            for replica in range(3)
        ],
        "required_judge_confirmations": 3,
        "required_judge_seeds": list(JUDGE_SEEDS),
    }
    if requires_failure_of is not None:
        payload["requires_failure_of"] = requires_failure_of
    if scale_fraction is not None:
        payload["scale_fraction"] = scale_fraction
        payload["scale_fields"] = ["nframes", "resize"]
    payload["counterfactual_fingerprint"] = canonical_sha256(payload)
    return payload


def _planned_calls(steps: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "start_time": step["start_time"],
            "end_time": step["end_time"],
            "nframes": step["nframes"],
            "resize": step["resize"],
            "source_actual_timestamps": step["actual_timestamps"],
            **(
                {"evidence_request": step["evidence_request"]}
                if step.get("evidence_request") is not None
                else {}
            ),
        }
        for step in steps
    ]


def generate_compression_replay_specs(
    trajectory: Mapping[str, Any],
    *,
    frame_fractions: Sequence[float] = FRAME_FRACTIONS,
) -> tuple[dict[str, Any], ...]:
    """Emit an ordered dependency DAG for deterministic trajectory compression.

    Tail deletion is tried one call at a time.  Exactly one early-stop branch
    activates at the last successful tail node, after which joint nframes and
    resize reductions run as a 75% -> 50% -> 25% success chain.  Executors must
    skip a node when its parent failed or its ``requires_failure_of`` node did
    not fail.
    """

    source = _normalized_for_replay(trajectory)
    steps = source["tool_steps"]
    fractions = tuple(float(value) for value in frame_fractions)
    if fractions != FRAME_FRACTIONS:
        raise ValueError("compression fractions are frozen at 0.75, 0.50 and 0.25")
    root_calls = _planned_calls(steps)
    root_fingerprint = canonical_sha256(
        {
            "base_trajectory_id": source["trajectory_id"],
            "planned_calls": root_calls,
        }
    )
    specs: list[dict[str, Any]] = []
    execution_order = 0

    # Stage 1: delete exactly one additional tail call after each success.
    tail_nodes: list[tuple[str, list[dict[str, Any]]]] = [
        (root_fingerprint, root_calls)
    ]
    parent = root_fingerprint
    for drop_count in range(1, len(steps)):
        execution_order += 1
        calls = _planned_calls(steps[: len(steps) - drop_count])
        spec = _extra_replay_spec(
            source,
            variant_id=f"tail_drop_{drop_count}",
            planned_calls=calls,
            stage="tail_delete",
            stage_order=1,
            execution_order=execution_order,
            parent_fingerprint=parent,
            stop_policy="replay_then_judge",
        )
        specs.append(spec)
        parent = str(spec["counterfactual_fingerprint"])
        tail_nodes.append((parent, calls))

    # Stage 2: one branch activates for the last successful tail node.  A
    # shallower branch requires the next tail attempt to have failed.
    early_nodes: list[tuple[dict[str, Any], list[dict[str, Any]]]] = []
    tail_specs = [spec for spec in specs if spec["stage"] == "tail_delete"]
    for index, (parent_fingerprint, calls) in enumerate(tail_nodes):
        execution_order += 1
        next_tail = (
            str(tail_specs[index]["counterfactual_fingerprint"])
            if index < len(tail_specs)
            else None
        )
        early = _extra_replay_spec(
            source,
            variant_id=f"early_stop_after_tail_{index}",
            planned_calls=calls,
            stage="early_stop",
            stage_order=2,
            execution_order=execution_order,
            parent_fingerprint=parent_fingerprint,
            requires_failure_of=next_tail,
            stop_policy="force_judge_after_parent",
        )
        specs.append(early)
        early_nodes.append((early, calls))

    # Stage 3: only the active early-stop branch is reduced.  Each fraction is
    # measured from that branch's retained calls, but depends on the preceding
    # fraction succeeding, so a failure skips all more aggressive descendants.
    for early, baseline_calls in early_nodes:
        parent = str(early["counterfactual_fingerprint"])
        for fraction in fractions:
            execution_order += 1
            scaled = [
                {
                    **call,
                    "nframes": max(
                        1, int(math.floor(int(call["nframes"]) * fraction + 0.5))
                    ),
                    "resize": float(call["resize"]) * fraction,
                }
                for call in baseline_calls
            ]
            suffix = int(round(fraction * 100))
            scale = _extra_replay_spec(
                source,
                variant_id=f"{early['variant_id']}_scale_{suffix:03d}",
                planned_calls=scaled,
                stage="evidence_scale",
                stage_order=3,
                execution_order=execution_order,
                parent_fingerprint=parent,
                stop_policy="fixed_replay_then_judge",
                scale_fraction=fraction,
            )
            specs.append(scale)
            parent = str(scale["counterfactual_fingerprint"])
    return tuple(specs)


def _passed(value: bool | str) -> bool:
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().casefold()
    if normalized == "passed":
        return True
    if normalized == "failed":
        return False
    raise ValueError("compression outcome must be bool, 'passed', or 'failed'")


def compression_execution_state(
    specs: Sequence[Mapping[str, Any]],
    outcomes: Mapping[str, bool | str],
) -> CompressionExecutionState:
    """Release at most one dependency-safe replay job.

    ``outcomes`` contains offline stability decisions for jobs that have
    actually run.  External parent fingerprints refer to the already selected
    stable source and are therefore treated as passed.  An attempted child
    without a successful parent is rejected, while pending descendants remain
    blocked.  This is the executor-side guard that prevents callers from
    submitting the static DAG as an unordered parallel matrix.
    """

    ordered = sorted(
        (dict(spec) for spec in specs), key=lambda row: row["execution_order"]
    )
    if not ordered:
        return CompressionExecutionState((), (), (), (), True)
    orders = [int(spec["execution_order"]) for spec in ordered]
    if orders != list(range(1, len(ordered) + 1)):
        raise ValueError("compression execution_order must be contiguous from 1")
    by_fingerprint: dict[str, dict[str, Any]] = {}
    for spec in ordered:
        fingerprint = _require_sha256(
            str(spec.get("counterfactual_fingerprint") or ""),
            "counterfactual_fingerprint",
        )
        if fingerprint in by_fingerprint:
            raise ValueError("duplicate compression counterfactual_fingerprint")
        if spec.get("execution_policy") != "sequential_dependency_gated":
            raise ValueError("compression spec is not dependency gated")
        if spec.get("parallel_safe") is not False:
            raise ValueError("compression spec must be marked parallel_safe=false")
        by_fingerprint[fingerprint] = spec

    normalized_outcomes = {
        _require_sha256(str(fingerprint), "outcome fingerprint"): _passed(status)
        for fingerprint, status in outcomes.items()
    }
    unknown = sorted(set(normalized_outcomes) - set(by_fingerprint))
    if unknown:
        raise ValueError(f"compression outcomes contain unknown nodes: {unknown[:3]}")

    ready: list[dict[str, Any]] = []
    completed: list[str] = []
    skipped: set[str] = set()
    blocked: list[str] = []
    for fingerprint, spec in (
        (str(row["counterfactual_fingerprint"]), row) for row in ordered
    ):
        parent = _require_sha256(
            str(spec.get("parent_fingerprint") or ""), "parent_fingerprint"
        )
        required_failure = spec.get("requires_failure_of")
        if required_failure is not None:
            required_failure = _require_sha256(
                str(required_failure), "requires_failure_of"
            )

        parent_is_internal = parent in by_fingerprint
        parent_failed = parent_is_internal and (
            parent in skipped or normalized_outcomes.get(parent) is False
        )
        parent_pending = parent_is_internal and parent not in normalized_outcomes
        failure_pending = (
            required_failure is not None and required_failure not in normalized_outcomes
        )
        failure_condition_rejected = (
            required_failure is not None
            and normalized_outcomes.get(required_failure) is True
        )

        if fingerprint in normalized_outcomes:
            if parent_failed or parent_pending:
                raise RuntimeError(
                    "compression result exists without a successful completed parent"
                )
            if failure_pending or failure_condition_rejected:
                raise RuntimeError(
                    "compression result violated its requires_failure_of dependency"
                )
            completed.append(fingerprint)
            continue
        if parent_failed or failure_condition_rejected:
            skipped.add(fingerprint)
            continue
        if parent_pending or failure_pending:
            blocked.append(fingerprint)
            continue
        ready.append(spec)

    if len(ready) > 1:
        raise RuntimeError("compression DAG exposed multiple parallel-ready jobs")
    return CompressionExecutionState(
        ready=tuple(ready),
        completed_fingerprints=tuple(completed),
        skipped_fingerprints=tuple(
            str(row["counterfactual_fingerprint"])
            for row in ordered
            if str(row["counterfactual_fingerprint"]) in skipped
        ),
        blocked_fingerprints=tuple(blocked),
        complete=not ready and not blocked,
    )


def _normalized_call_schedule(
    calls: Any, *, require_observed_frames: bool
) -> tuple[dict[str, Any], ...]:
    if not isinstance(calls, list) or not calls:
        raise ValueError("frame-select schedule must be a non-empty list")
    normalized: list[dict[str, Any]] = []
    for index, raw in enumerate(calls):
        if not isinstance(raw, Mapping):
            raise ValueError(f"frame-select call {index} must be an object")
        arguments = raw.get("arguments") if raw.get("tool") == "frame_select" else raw
        if not isinstance(arguments, Mapping):
            raise ValueError(f"frame-select call {index} has invalid arguments")
        try:
            start = float(arguments["start_time"])
            end = float(arguments["end_time"])
            nframes = int(arguments["nframes"])
            resize = float(arguments.get("resize", 1.0))
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"frame-select call {index} is malformed") from error
        if (
            not math.isfinite(start)
            or not math.isfinite(end)
            or not math.isfinite(resize)
            or end <= start
            or nframes <= 0
            or resize <= 0
        ):
            raise ValueError(f"frame-select call {index} has invalid values")
        if require_observed_frames:
            timestamps = arguments.get("actual_timestamps", arguments.get("timestamps"))
            frame_paths = arguments.get("frame_paths")
            if not isinstance(timestamps, list) or len(timestamps) != nframes:
                raise ValueError(f"frame-select call {index} timestamp count mismatch")
            if not isinstance(frame_paths, list) or len(frame_paths) != nframes:
                raise ValueError(f"frame-select call {index} frame count mismatch")
        normalized.append(
            {
                "start_time": start,
                "end_time": end,
                "nframes": nframes,
                "resize": resize,
            }
        )
    return tuple(normalized)


def _same_call_schedule(
    expected: Sequence[Mapping[str, Any]], actual: Sequence[Mapping[str, Any]]
) -> bool:
    if len(expected) != len(actual):
        return False
    for left, right in zip(expected, actual):
        if int(left["nframes"]) != int(right["nframes"]):
            return False
        for key in ("start_time", "end_time", "resize"):
            if not math.isclose(
                float(left[key]), float(right[key]), rel_tol=0.0, abs_tol=1e-6
            ):
                return False
    return True


def _compression_row_rejection_reason(
    row: Mapping[str, Any], spec: Mapping[str, Any], correct_answer: str
) -> str | None:
    try:
        assert_deferred_result_public(row)
    except ValueError:
        return "annotation_leak"
    if row.get("scoring_deferred") is not True:
        return "not_deferred"
    if row.get("annotation_leak_check") != "passed":
        return "annotation_leak"
    if int(row.get("candidate_rerun") or 0) != 0:
        return "candidate_rerun"
    if row.get("candidate_cost_complete") is not True:
        return "candidate_cost_incomplete"
    if any(
        row.get(key)
        for key in (
            "error",
            "api_error",
            "frame_error",
            "parse_error",
            "strict_replay_violation",
        )
    ):
        return "engineering_or_parse_error"
    if row.get("fallback_to_candidate") is True or row.get("fallback_used") is True:
        return "fallback"
    if row.get("planned_calls_completed") is not True:
        return "incomplete_schedule"
    if _prediction(row) != str(correct_answer).strip().upper():
        return "incorrect"
    try:
        expected = _normalized_call_schedule(
            list(spec.get("planned_calls") or []), require_observed_frames=False
        )
        declared = _normalized_call_schedule(
            row.get("planned_calls"), require_observed_frames=False
        )
        observed = _normalized_call_schedule(
            row.get("tool_steps", row.get("tool_calls")),
            require_observed_frames=True,
        )
    except ValueError:
        return "invalid_tool_trace"
    if not _same_call_schedule(expected, declared) or not _same_call_schedule(
        expected, observed
    ):
        return "schedule_mismatch"
    try:
        _end_to_end_cost_key(row)
    except ValueError:
        return "invalid_end_to_end_cost"
    return None


def analyze_compression_replays(
    specs: Sequence[Mapping[str, Any]],
    replay_results: Iterable[Mapping[str, Any]],
    answers: Mapping[tuple[str, str], str],
) -> CompressionReplayAnalysis:
    """Apply the offline 3/3 correctness gate to completed replay nodes."""

    by_fingerprint: dict[str, dict[str, Any]] = {}
    expected_trajectory_ids: dict[str, tuple[str, int]] = {}
    for raw_spec in specs:
        spec = dict(raw_spec)
        fingerprint = _require_sha256(
            str(spec.get("counterfactual_fingerprint") or ""),
            "counterfactual_fingerprint",
        )
        if fingerprint != canonical_sha256(
            {key: value for key, value in spec.items() if key != "counterfactual_fingerprint"}
        ):
            raise ValueError("compression spec fingerprint mismatch")
        if fingerprint in by_fingerprint:
            raise ValueError("duplicate compression spec fingerprint")
        replica_ids = spec.get("replica_trajectory_ids")
        seeds = spec.get("required_judge_seeds")
        if (
            not isinstance(replica_ids, list)
            or len(replica_ids) != 3
            or len(set(map(str, replica_ids))) != 3
            or list(seeds or []) != list(JUDGE_SEEDS)
        ):
            raise ValueError("compression spec must freeze three replica IDs and seeds")
        by_fingerprint[fingerprint] = spec
        for index, trajectory_id in enumerate(replica_ids):
            identity = str(trajectory_id)
            if identity in expected_trajectory_ids:
                raise ValueError("duplicate compression replica trajectory ID")
            expected_trajectory_ids[identity] = (fingerprint, index)

    rows_by_fingerprint: dict[str, list[dict[str, Any]]] = {}
    seen_ids: set[str] = set()
    for raw_row in replay_results:
        row = dict(raw_row)
        trajectory_id = str(row.get("trajectory_id") or "")
        if not trajectory_id or trajectory_id in seen_ids:
            raise ValueError("replay results contain a missing/duplicate trajectory ID")
        seen_ids.add(trajectory_id)
        expected = expected_trajectory_ids.get(trajectory_id)
        if expected is None:
            raise ValueError(f"replay result is outside the frozen DAG: {trajectory_id}")
        fingerprint, replica_index = expected
        spec = by_fingerprint[fingerprint]
        if row.get("counterfactual_fingerprint") != fingerprint:
            raise ValueError("replay result counterfactual fingerprint mismatch")
        if str(row.get("base_trajectory_id") or "") != str(
            spec.get("base_trajectory_id") or ""
        ):
            raise ValueError("replay result base trajectory mismatch")
        for key in ("dataset", "sample_id", "schedule_id", "variant_id"):
            if str(row.get(key) or "") != str(spec.get(key) or ""):
                raise ValueError(f"replay result {key} mismatch")
        if int(row.get("judge_seed", -1)) != JUDGE_SEEDS[replica_index]:
            raise ValueError("replay result Judge seed mismatch")
        if str(row.get("replica_id") or "") != str(replica_index):
            raise ValueError("replay result replica ID mismatch")
        rows_by_fingerprint.setdefault(fingerprint, []).append(row)

    outcomes: dict[str, str] = {}
    representatives: list[dict[str, Any]] = []
    rejection_reasons: dict[str, str] = {}
    incomplete: list[str] = []
    for fingerprint, rows in sorted(rows_by_fingerprint.items()):
        spec = by_fingerprint[fingerprint]
        expected_ids = {str(value) for value in spec["replica_trajectory_ids"]}
        observed_ids = {str(row["trajectory_id"]) for row in rows}
        if len(rows) < 3:
            incomplete.append(fingerprint)
            continue
        if len(rows) != 3 or observed_ids != expected_ids:
            raise ValueError("completed compression node does not contain exact replicas")
        answer_key = (
            str(spec.get("dataset") or "").lower(),
            str(spec.get("sample_id") or ""),
        )
        answer = answers.get(answer_key)
        if answer is None:
            raise ValueError(f"compression spec is outside Train600: {answer_key}")
        reasons = [
            _compression_row_rejection_reason(row, spec, answer) for row in rows
        ]
        reason = next((value for value in reasons if value is not None), None)
        if reason is not None:
            outcomes[fingerprint] = "failed"
            rejection_reasons[fingerprint] = reason
            continue
        costs = [_cost_number(row, "end_to_end_total_tokens") for row in rows]
        middle = float(median(costs))
        representative = min(
            rows,
            key=lambda row: (
                abs(_cost_number(row, "end_to_end_total_tokens") - middle),
                _cost_number(row, "end_to_end_visual_tokens"),
                len(_frame_steps(row)),
                _cost_number(row, "end_to_end_latency_s"),
                str(row["trajectory_id"]),
            ),
        )
        selected = _without_labels(representative)
        selected.update(
            {
                "_selection_source": "compression_replay",
                "_selection_cost_basis": "end_to_end",
                "_selection_family_id": spec["family_id"],
                "_selection_median_end_to_end_total_tokens": middle,
                "_compression_fingerprint": fingerprint,
            }
        )
        outcomes[fingerprint] = "passed"
        representatives.append(selected)
    return CompressionReplayAnalysis(
        outcomes=dict(sorted(outcomes.items())),
        representatives=tuple(
            sorted(representatives, key=lambda row: str(row["_compression_fingerprint"]))
        ),
        rejection_reasons=dict(sorted(rejection_reasons.items())),
        incomplete_fingerprints=tuple(sorted(incomplete)),
    )


def finalize_compression_dags(
    base_selected: Iterable[Mapping[str, Any]],
    specs: Sequence[Mapping[str, Any]],
    replay_results: Iterable[Mapping[str, Any]],
    answers: Mapping[tuple[str, str], str],
) -> CompressionFinalization:
    """Release the next safe nodes or finalize the cheapest stable trajectories."""

    bases: dict[str, dict[str, Any]] = {}
    base_by_sample: dict[tuple[str, str], str] = {}
    for raw in base_selected:
        row = dict(raw)
        assert_deferred_result_public(row)
        if row.get("_selection_stable") is not True:
            raise ValueError("base trajectory is not marked 3/3 stable")
        trajectory_id = str(row.get("trajectory_id") or "")
        identity = (
            str(row.get("dataset") or "").lower(),
            str(row.get("sample_id") or ""),
        )
        if not trajectory_id or trajectory_id in bases or identity in base_by_sample:
            raise ValueError("base selection has a missing/duplicate identity")
        if identity not in answers:
            raise ValueError(f"base selection is outside Train600: {identity}")
        _end_to_end_cost_key(row)
        bases[trajectory_id] = row
        base_by_sample[identity] = trajectory_id
    if not bases:
        raise ValueError("base stable selection is empty")

    specs_by_base: dict[str, list[dict[str, Any]]] = {}
    for raw_spec in specs:
        spec = dict(raw_spec)
        base_id = str(spec.get("base_trajectory_id") or "")
        if base_id not in bases:
            raise ValueError("compression spec references an unknown base trajectory")
        base = bases[base_id]
        if (
            str(spec.get("dataset") or "").lower()
            != str(base.get("dataset") or "").lower()
            or str(spec.get("sample_id") or "") != str(base.get("sample_id") or "")
        ):
            raise ValueError("compression spec sample differs from its base trajectory")
        specs_by_base.setdefault(base_id, []).append(spec)
    if set(specs_by_base) != set(bases):
        raise ValueError("compression specs do not cover every stable base trajectory")

    analysis = analyze_compression_replays(specs, replay_results, answers)
    representative_by_fingerprint = {
        str(row["_compression_fingerprint"]): row
        for row in analysis.representatives
    }
    ready: list[dict[str, Any]] = []
    selected: list[dict[str, Any]] = []
    all_complete = True
    for base_id in sorted(bases):
        group = specs_by_base[base_id]
        group_fingerprints = {
            str(spec["counterfactual_fingerprint"]) for spec in group
        }
        group_outcomes = {
            fingerprint: outcome
            for fingerprint, outcome in analysis.outcomes.items()
            if fingerprint in group_fingerprints
        }
        state = compression_execution_state(group, group_outcomes)
        ready.extend(state.ready)
        if not state.complete:
            all_complete = False
            continue
        candidates = [bases[base_id]] + [
            representative_by_fingerprint[fingerprint]
            for fingerprint, outcome in group_outcomes.items()
            if outcome == "passed" and fingerprint in representative_by_fingerprint
        ]
        winner = dict(min(candidates, key=_end_to_end_cost_key))
        winner["_selection_stable"] = True
        winner["_selection_confirmation_count"] = len(JUDGE_SEEDS)
        winner["_selection_cost_basis"] = "end_to_end"
        winner["_compression_finalized"] = True
        selected.append(winner)

    ready_samples = [
        (str(spec.get("dataset") or "").lower(), str(spec.get("sample_id") or ""))
        for spec in ready
    ]
    if len(ready_samples) != len(set(ready_samples)):
        raise RuntimeError("compression scheduler released multiple nodes for one sample")
    if not all_complete:
        selected = []
    gate = sft_start_gate(selected)
    gate["conditions"] = {
        **dict(gate["conditions"]),
        "compression_dags_complete": all_complete,
    }
    gate["passed"] = all(gate["conditions"].values())
    return CompressionFinalization(
        ready=tuple(sorted(ready, key=lambda row: (row["dataset"], row["sample_id"]))),
        outcomes=analysis.outcomes,
        representatives=analysis.representatives,
        rejection_reasons=analysis.rejection_reasons,
        incomplete_fingerprints=analysis.incomplete_fingerprints,
        selected_pruned=tuple(
            sorted(selected, key=lambda row: (row["dataset"], row["sample_id"]))
        ),
        gate=gate,
        complete=all_complete,
    )
