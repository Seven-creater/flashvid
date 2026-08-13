"""Strict Dev/Test gates for Perception-Memory EVA SFT results."""

from __future__ import annotations

import hashlib
import json
import math
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from statistics import fmean
from typing import Any, Mapping, Sequence


DATASETS = ("lvbench", "lsdbench", "cgbench")
TEST_FLOORS = {"lvbench": 45, "lsdbench": 63, "cgbench": 43}
DEV_SEEDS = frozenset({17, 42, 73})
TEST_SEEDS = frozenset({42})
_MODEL_ARTIFACT_FIELD = "model_artifact_sha256"
_ROLE_NAMES = ("planner", "observer", "verifier", "answerer")
_RUNTIME_FIELDS = (
    "backend",
    "agent_version",
    "implementation_sha256",
    "implementation_bundle_sha256",
    "frame_tool_identity",
    "manifest_sha256",
    "candidate_results_sha256",
    "experiment_config_sha256",
    "diagnostics_gate_sha256",
    "max_turns",
    "max_frames_per_call",
    "controller_max_tokens",
    "perception_max_tokens",
    "judge_max_tokens",
    "role_config_sha256",
    "role_separated_runtime_version",
    "role_prompt_schema_bundle_sha256",
)
_SPLIT_SOURCE_FIELDS = ("manifest_sha256", "candidate_results_sha256")
_STABLE_RUNTIME_FIELDS = tuple(
    field for field in _RUNTIME_FIELDS if field not in _SPLIT_SOURCE_FIELDS
)


class PerceptionMemoryGateError(ValueError):
    pass


@dataclass(frozen=True)
class SeedRun:
    seed: int
    result_paths: Mapping[str, Path]


@dataclass(frozen=True)
class MethodRuns:
    method_id: str
    runs: tuple[SeedRun, ...]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _validated_sha256(value: Any, field: str) -> str:
    result = str(value or "")
    if len(result) != 64 or any(character not in "0123456789abcdef" for character in result):
        raise PerceptionMemoryGateError(f"{field} must be a lowercase SHA-256")
    return result


def _validated_manifest_sha256(
    values: Mapping[str, str],
) -> dict[str, str]:
    if set(values) != set(DATASETS):
        raise PerceptionMemoryGateError(
            "expected_manifest_sha256 must cover all datasets"
        )
    return {
        dataset: _validated_sha256(
            values[dataset], f"expected_manifest_sha256.{dataset}"
        )
        for dataset in DATASETS
    }


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _validate_role_audit_fields(
    row: Mapping[str, Any], identity: tuple[str, str]
) -> None:
    if row.get("role_separated_runtime_version") != "role_separated_visual_csv_v2":
        raise PerceptionMemoryGateError(
            f"{identity}: role-separated runtime version changed"
        )
    _validated_sha256(
        row.get("role_prompt_schema_bundle_sha256"),
        f"{identity}: role_prompt_schema_bundle_sha256",
    )
    models = row.get("role_models")
    artifacts = row.get("role_artifact_sha256s")
    if not isinstance(models, Mapping) or set(models) != set(_ROLE_NAMES):
        raise PerceptionMemoryGateError(f"{identity}: invalid role_models")
    if not isinstance(artifacts, Mapping) or set(artifacts) != set(_ROLE_NAMES):
        raise PerceptionMemoryGateError(
            f"{identity}: invalid role_artifact_sha256s"
        )
    for role in _ROLE_NAMES:
        if not str(models[role] or "").strip():
            raise PerceptionMemoryGateError(
                f"{identity}: role_models.{role} must be non-empty"
            )
        _validated_sha256(
            artifacts[role], f"{identity}: role_artifact_sha256s.{role}"
        )


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, start=1):
            if not raw.strip():
                continue
            try:
                row = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise PerceptionMemoryGateError(
                    f"{path}:{line_number}: invalid JSON"
                ) from exc
            if not isinstance(row, dict):
                raise PerceptionMemoryGateError(
                    f"{path}:{line_number}: record must be an object"
                )
            rows.append(row)
    return rows


def _letter(value: Any, field: str, identity: tuple[str, str]) -> str:
    result = str(value or "").strip().upper()
    if len(result) != 1 or not "A" <= result <= "H":
        raise PerceptionMemoryGateError(f"{identity}: invalid {field}")
    return result


def _prediction(row: Mapping[str, Any], identity: tuple[str, str]) -> str | None:
    value = row.get("prediction")
    if value in {None, ""}:
        value = row.get("final_prediction")
    if value in {None, ""}:
        return None
    return _letter(value, "prediction", identity)


def _number(value: Any, field: str, identity: tuple[str, str]) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) < 0
    ):
        raise PerceptionMemoryGateError(f"{identity}: invalid {field}")
    return float(value)


def _cost(row: Mapping[str, Any], kind: str, identity: tuple[str, str]) -> float | None:
    if row.get("candidate_cost_complete") is not True:
        return None
    if kind == "total":
        value_key = "end_to_end_total_tokens"
        complete_key = "end_to_end_total_tokens_complete"
    else:
        value_key = "end_to_end_visual_tokens"
        complete_key = "end_to_end_visual_tokens_complete"
    if row.get(complete_key) is not True or value_key not in row:
        return None
    return _number(row.get(value_key), value_key, identity)


def _engineering_failure(row: Mapping[str, Any], identity: tuple[str, str]) -> bool:
    if any(row.get(key) for key in ("api_error", "frame_error", "parse_error")):
        return True
    if row.get("failure_stage") or row.get("trajectory_valid") is False:
        return True
    if row.get("error"):
        return True
    if _prediction(row, identity) is None:
        return True
    return _cost(row, "total", identity) is None or _cost(row, "visual", identity) is None


def _incomplete_stop(row: Mapping[str, Any]) -> bool:
    reason = str(row.get("stop_reason") or "").strip().lower()
    return reason in {
        "incomplete_evidence",
        "max_turns_incomplete",
        "stopped_with_incomplete_evidence",
    } or (
        row.get("evidence_complete") is False
        and reason not in {"runtime_error", "annotation_leak", "error"}
    )


def _validate_method(method: MethodRuns) -> None:
    if not method.method_id.strip():
        raise PerceptionMemoryGateError("method_id must be non-empty")
    if not method.runs:
        raise PerceptionMemoryGateError(f"{method.method_id}: no seed runs")
    seeds = [run.seed for run in method.runs]
    if len(seeds) != len(set(seeds)):
        raise PerceptionMemoryGateError(f"{method.method_id}: duplicate seed")
    for run in method.runs:
        if set(run.result_paths) != set(DATASETS):
            raise PerceptionMemoryGateError(
                f"{method.method_id}/seed{run.seed}: must contain all three datasets"
            )


def _require_seed_scope(
    method: MethodRuns, expected_seeds: frozenset[int], phase: str
) -> None:
    actual = {run.seed for run in method.runs}
    if actual != set(expected_seeds):
        raise PerceptionMemoryGateError(
            f"{method.method_id}: {phase} requires seeds "
            f"{sorted(expected_seeds)}, found {sorted(actual)}"
        )


def _load_seed_run(
    method_id: str,
    run: SeedRun,
    expected_counts: Mapping[str, int],
    expected_manifest_sha256: Mapping[str, str],
) -> tuple[dict[tuple[str, str], dict[str, Any]], dict[str, dict[str, str]]]:
    records: dict[tuple[str, str], dict[str, Any]] = {}
    files: dict[str, dict[str, str]] = {}
    for dataset in DATASETS:
        path = Path(run.result_paths[dataset]).resolve()
        rows = _read_jsonl(path)
        expected = int(expected_counts[dataset])
        if len(rows) != expected:
            raise PerceptionMemoryGateError(
                f"{path}: expected {expected} rows, found {len(rows)}"
            )
        seen: set[str] = set()
        for row in rows:
            if str(row.get("dataset") or "").strip().lower() != dataset:
                raise PerceptionMemoryGateError(f"{path}: wrong dataset scope")
            sample_id = str(row.get("sample_id") or "").strip()
            if not sample_id or sample_id in seen:
                raise PerceptionMemoryGateError(
                    f"{path}: missing or duplicate sample_id {sample_id!r}"
                )
            seen.add(sample_id)
            identity = (dataset, sample_id)
            _letter(row.get("answer"), "answer", identity)
            if row.get("backend") != "perception_memory_eva":
                raise PerceptionMemoryGateError(
                    f"{identity}: result is not perception_memory_eva"
                )
            if int(row.get("seed", -1)) != run.seed:
                raise PerceptionMemoryGateError(f"{identity}: seed does not match run")
            for field in _RUNTIME_FIELDS:
                value = row.get(field)
                if value is None or value == "":
                    raise PerceptionMemoryGateError(
                        f"{identity}: missing runtime audit field {field}"
                    )
            _validate_role_audit_fields(row, identity)
            if row.get("manifest_sha256") != expected_manifest_sha256[dataset]:
                raise PerceptionMemoryGateError(
                    f"{identity}: manifest SHA-256 does not match frozen config"
                )
            _validated_sha256(
                row.get(_MODEL_ARTIFACT_FIELD),
                f"{identity}: {_MODEL_ARTIFACT_FIELD}",
            )
            for complete_field in (
                "candidate_cost_complete",
                "end_to_end_total_tokens_complete",
                "end_to_end_visual_tokens_complete",
            ):
                if row.get(complete_field) is not True:
                    raise PerceptionMemoryGateError(
                        f"{identity}: {complete_field} must be true"
                    )
            _cost(row, "total", identity)
            _cost(row, "visual", identity)
            records[identity] = dict(row)
        files[dataset] = {"path": str(path), "sha256": _sha256(path)}
    return records, files


def _audit_scope(
    reference: Mapping[tuple[str, str], Mapping[str, Any]],
    other: Mapping[tuple[str, str], Mapping[str, Any]],
    label: str,
    *,
    allow_planner_treatment: bool = False,
) -> None:
    if set(other) != set(reference):
        missing = sorted(set(reference) - set(other))
        extra = sorted(set(other) - set(reference))
        raise PerceptionMemoryGateError(
            f"{label}: sample scope mismatch (missing={missing[:3]}, extra={extra[:3]})"
        )
    for identity in reference:
        if _letter(reference[identity].get("answer"), "answer", identity) != _letter(
            other[identity].get("answer"), "answer", identity
        ):
            raise PerceptionMemoryGateError(f"{label}: answer mismatch at {identity}")
        for field in _RUNTIME_FIELDS:
            if field == "role_config_sha256" and allow_planner_treatment:
                continue
            if other[identity].get(field) != reference[identity].get(field):
                raise PerceptionMemoryGateError(
                    f"{label}: runtime field mismatch at {identity}: {field}"
                )
        for field in ("role_models", "role_artifact_sha256s"):
            current = other[identity][field]
            baseline = reference[identity][field]
            roles = ("observer", "verifier", "answerer") if allow_planner_treatment else _ROLE_NAMES
            if any(current[role] != baseline[role] for role in roles):
                raise PerceptionMemoryGateError(
                    f"{label}: runtime field mismatch at {identity}: {field}"
                )


def _seed_summary(rows: Mapping[tuple[str, str], Mapping[str, Any]]) -> dict[str, Any]:
    ordered = [(identity, rows[identity]) for identity in sorted(rows)]
    costs: list[tuple[float, float]] = []
    correct_by_dataset = {dataset: 0 for dataset in DATASETS}
    failures = leaks = reruns = incomplete_stops = 0
    flips = {
        "candidate_changed": 0,
        "candidate_wrong_to_right": 0,
        "candidate_right_to_wrong": 0,
        "fallback_to_candidate": 0,
    }
    for identity, row in ordered:
        answer = _letter(row.get("answer"), "answer", identity)
        prediction = _prediction(row, identity)
        correct = prediction == answer
        correct_by_dataset[identity[0]] += int(correct)
        failure = _engineering_failure(row, identity)
        failures += int(failure)
        if not failure:
            total = _cost(row, "total", identity)
            visual = _cost(row, "visual", identity)
            assert total is not None and visual is not None
            costs.append((total, visual))
        leaks += int(row.get("annotation_leak_check") != "passed")
        try:
            reruns += int(row.get("candidate_rerun") or 0)
        except (TypeError, ValueError) as exc:
            raise PerceptionMemoryGateError(
                f"{identity}: invalid candidate_rerun"
            ) from exc
        incomplete_stops += int(_incomplete_stop(row))
        candidate_raw = str(row.get("candidate_answer") or "").strip().upper()
        candidate = candidate_raw if len(candidate_raw) == 1 and "A" <= candidate_raw <= "H" else None
        changed = bool(candidate and prediction and prediction != candidate)
        flips["candidate_changed"] += int(changed)
        flips["candidate_wrong_to_right"] += int(
            changed and candidate != answer and prediction == answer
        )
        flips["candidate_right_to_wrong"] += int(
            changed and candidate == answer and prediction != answer
        )
        flips["fallback_to_candidate"] += int(
            row.get("fallback_to_candidate") is True
        )
    if not costs:
        raise PerceptionMemoryGateError("no complete cost rows")
    return {
        "samples": len(ordered),
        "correct": sum(correct_by_dataset.values()),
        "correct_by_dataset": correct_by_dataset,
        "mean_total_tokens": fmean(item[0] for item in costs),
        "mean_visual_tokens": fmean(item[1] for item in costs),
        "cost_complete_samples": len(costs),
        "cost_excluded_samples": len(ordered) - len(costs),
        "engineering_failures": failures,
        "failure_rate": failures / len(ordered),
        "annotation_leaks": leaks,
        "candidate_reruns": reruns,
        "stopped_with_incomplete_evidence": incomplete_stops,
        "candidate_flips": flips,
    }


def _method_summary(seed_summaries: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return {
        "seed_count": len(seed_summaries),
        "mean_correct": fmean(float(item["correct"]) for item in seed_summaries),
        "mean_correct_by_dataset": {
            dataset: fmean(
                float(item["correct_by_dataset"][dataset]) for item in seed_summaries
            )
            for dataset in DATASETS
        },
        "mean_total_tokens": fmean(
            float(item["mean_total_tokens"]) for item in seed_summaries
        ),
        "mean_visual_tokens": fmean(
            float(item["mean_visual_tokens"]) for item in seed_summaries
        ),
        "mean_failure_rate": fmean(
            float(item["failure_rate"]) for item in seed_summaries
        ),
        "total_annotation_leaks": sum(
            int(item["annotation_leaks"]) for item in seed_summaries
        ),
        "total_candidate_reruns": sum(
            int(item["candidate_reruns"]) for item in seed_summaries
        ),
        "mean_stopped_with_incomplete_evidence": fmean(
            float(item["stopped_with_incomplete_evidence"])
            for item in seed_summaries
        ),
        "candidate_flips": {
            key: sum(int(item["candidate_flips"][key]) for item in seed_summaries)
            for key in (
                "candidate_changed",
                "candidate_wrong_to_right",
                "candidate_right_to_wrong",
                "fallback_to_candidate",
            )
        },
    }


def _require_valid_baseline(report: Mapping[str, Any]) -> None:
    summary = report["summary"]
    if float(summary["mean_failure_rate"]) > 0.01:
        raise PerceptionMemoryGateError("untrained baseline failure rate exceeds 1%")
    if int(summary["total_annotation_leaks"]):
        raise PerceptionMemoryGateError("untrained baseline contains annotation leaks")
    if int(summary["total_candidate_reruns"]):
        raise PerceptionMemoryGateError("untrained baseline reran frozen candidates")


def _method_runtime_binding(
    method_id: str,
    rows_by_seed: Mapping[int, Mapping[tuple[str, str], Mapping[str, Any]]],
) -> tuple[dict[str, Any], dict[str, dict[str, str]]]:
    runtime_identity: dict[str, Any] | None = None
    split_sources: dict[str, dict[str, str]] = {}
    for seed in sorted(rows_by_seed):
        for identity in sorted(rows_by_seed[seed]):
            row = rows_by_seed[seed][identity]
            current_identity = {
                field: deepcopy(row[field]) for field in _STABLE_RUNTIME_FIELDS
            }
            current_identity.update(
                {
                    _MODEL_ARTIFACT_FIELD: row[_MODEL_ARTIFACT_FIELD],
                    "role_models": deepcopy(row["role_models"]),
                    "role_artifact_sha256s": deepcopy(
                        row["role_artifact_sha256s"]
                    ),
                }
            )
            if runtime_identity is None:
                runtime_identity = current_identity
            elif current_identity != runtime_identity:
                raise PerceptionMemoryGateError(
                    f"{method_id}: runtime identity changed within method at "
                    f"seed{seed}/{identity}"
                )
            dataset = identity[0]
            current_sources = {
                field: _validated_sha256(
                    row[field], f"{method_id}/seed{seed}/{identity}: {field}"
                )
                for field in _SPLIT_SOURCE_FIELDS
            }
            previous_sources = split_sources.setdefault(dataset, current_sources)
            if current_sources != previous_sources:
                raise PerceptionMemoryGateError(
                    f"{method_id}: split source changed within {dataset}"
                )
    if runtime_identity is None:
        raise PerceptionMemoryGateError(f"{method_id}: no result rows")
    if set(split_sources) != set(DATASETS):
        raise PerceptionMemoryGateError(
            f"{method_id}: split sources must cover all datasets"
        )
    return runtime_identity, split_sources


def _load_method(
    method: MethodRuns,
    expected_counts: Mapping[str, int],
    expected_manifest_sha256: Mapping[str, str],
) -> tuple[
    dict[int, dict[tuple[str, str], dict[str, Any]]],
    dict[str, Any],
    dict[str, Any],
]:
    _validate_method(method)
    rows_by_seed: dict[int, dict[tuple[str, str], dict[str, Any]]] = {}
    seed_reports: dict[str, Any] = {}
    files: dict[str, Any] = {}
    reference: dict[tuple[str, str], dict[str, Any]] | None = None
    model_artifacts: set[str] = set()
    for run in sorted(method.runs, key=lambda item: item.seed):
        rows, run_files = _load_seed_run(
            method.method_id,
            run,
            expected_counts,
            expected_manifest_sha256,
        )
        if reference is None:
            reference = rows
        else:
            _audit_scope(reference, rows, f"{method.method_id}/seed{run.seed}")
        rows_by_seed[run.seed] = rows
        model_artifacts.update(
            str(row[_MODEL_ARTIFACT_FIELD]) for row in rows.values()
        )
        seed_reports[str(run.seed)] = _seed_summary(rows)
        files[str(run.seed)] = run_files
    if len(model_artifacts) != 1:
        raise PerceptionMemoryGateError(
            f"{method.method_id}: method must use exactly one model artifact, "
            f"found {sorted(model_artifacts)}"
        )
    model_artifact_sha256 = next(iter(model_artifacts))
    runtime_identity, split_sources = _method_runtime_binding(
        method.method_id,
        rows_by_seed,
    )
    return rows_by_seed, {
        "method_id": method.method_id,
        "model_artifact_sha256": model_artifact_sha256,
        "runtime_identity": runtime_identity,
        "runtime_identity_sha256": _canonical_sha256(runtime_identity),
        "split_sources": split_sources,
        "summary": _method_summary(list(seed_reports.values())),
        "seeds": seed_reports,
    }, files


def _audit_methods(
    baseline_rows: Mapping[int, Mapping[tuple[str, str], Mapping[str, Any]]],
    candidate_rows: Mapping[int, Mapping[tuple[str, str], Mapping[str, Any]]],
    label: str,
) -> None:
    if set(candidate_rows) != set(baseline_rows):
        raise PerceptionMemoryGateError(f"{label}: seed scope mismatch")
    for seed in baseline_rows:
        baseline_seed = baseline_rows[seed]
        candidate_seed = candidate_rows[seed]
        _audit_scope(
            baseline_seed,
            candidate_seed,
            f"{label}/seed{seed}",
            allow_planner_treatment=True,
        )
        for identity in baseline_seed:
            baseline_roles = baseline_seed[identity]["role_artifact_sha256s"]
            candidate_roles = candidate_seed[identity]["role_artifact_sha256s"]
            for role in ("observer", "verifier", "answerer"):
                if candidate_roles[role] != baseline_roles[role]:
                    raise PerceptionMemoryGateError(
                        f"{label}/seed{seed}: frozen {role} artifact changed at "
                        f"{identity}"
                    )


def _paired_costs(
    baseline_rows: Mapping[int, Mapping[tuple[str, str], Mapping[str, Any]]],
    candidate_rows: Mapping[int, Mapping[tuple[str, str], Mapping[str, Any]]],
) -> dict[str, float | int]:
    baseline_total: list[float] = []
    baseline_visual: list[float] = []
    candidate_total: list[float] = []
    candidate_visual: list[float] = []
    for seed in sorted(baseline_rows):
        for identity in sorted(baseline_rows[seed]):
            baseline = baseline_rows[seed][identity]
            candidate = candidate_rows[seed][identity]
            if _engineering_failure(baseline, identity) or _engineering_failure(
                candidate, identity
            ):
                continue
            base_total = _cost(baseline, "total", identity)
            base_visual = _cost(baseline, "visual", identity)
            current_total = _cost(candidate, "total", identity)
            current_visual = _cost(candidate, "visual", identity)
            assert None not in (base_total, base_visual, current_total, current_visual)
            baseline_total.append(float(base_total))
            baseline_visual.append(float(base_visual))
            candidate_total.append(float(current_total))
            candidate_visual.append(float(current_visual))
    if not baseline_total:
        raise PerceptionMemoryGateError("no jointly complete rows for token comparison")
    return {
        "joint_complete_samples": len(baseline_total),
        "baseline_mean_total_tokens": fmean(baseline_total),
        "baseline_mean_visual_tokens": fmean(baseline_visual),
        "candidate_mean_total_tokens": fmean(candidate_total),
        "candidate_mean_visual_tokens": fmean(candidate_visual),
        "total_token_ratio": fmean(candidate_total) / fmean(baseline_total),
        "visual_token_ratio": fmean(candidate_visual) / fmean(baseline_visual),
    }


def _dev_point(
    baseline: Mapping[str, Any],
    candidate: Mapping[str, Any],
    paired_costs: Mapping[str, float | int],
) -> dict[str, Any]:
    base = baseline["summary"]
    current = candidate["summary"]
    dataset_deltas = {
        dataset: current["mean_correct_by_dataset"][dataset]
        - base["mean_correct_by_dataset"][dataset]
        for dataset in DATASETS
    }
    total_ratio = float(paired_costs["total_token_ratio"])
    visual_ratio = float(paired_costs["visual_token_ratio"])
    conditions = {
        "mean_accuracy_gain_at_least_3": current["mean_correct"]
        >= base["mean_correct"] + 3.0,
        "each_dataset_nondecrease": all(value >= 0 for value in dataset_deltas.values()),
        "total_token_ratio_at_most_0_70": total_ratio <= 0.70,
        "visual_token_ratio_at_most_0_70": visual_ratio <= 0.70,
        "incomplete_stops_decrease": current[
            "mean_stopped_with_incomplete_evidence"
        ]
        < base["mean_stopped_with_incomplete_evidence"],
        "failure_rate_at_most_1pct": current["mean_failure_rate"] <= 0.01,
        "annotation_leak_zero": current["total_annotation_leaks"] == 0,
        "candidate_rerun_zero": current["total_candidate_reruns"] == 0,
        "duplicate_sample_id_zero": True,
    }
    return {
        "method_id": candidate["method_id"],
        "model_artifact_sha256": candidate["model_artifact_sha256"],
        "runtime_identity": deepcopy(candidate["runtime_identity"]),
        "runtime_identity_sha256": candidate["runtime_identity_sha256"],
        "split_sources": deepcopy(candidate["split_sources"]),
        "summary": current,
        "dataset_deltas": dataset_deltas,
        "total_token_ratio": total_ratio,
        "visual_token_ratio": visual_ratio,
        "paired_costs": dict(paired_costs),
        "conditions": conditions,
        "passed": all(conditions.values()),
    }


def evaluate_dev_gate(
    *,
    baseline: MethodRuns,
    candidates: Sequence[MethodRuns],
    expected_manifest_sha256: Mapping[str, str],
    expected_counts: Mapping[str, int] | None = None,
) -> dict[str, Any]:
    counts = dict(expected_counts or {dataset: 50 for dataset in DATASETS})
    if set(counts) != set(DATASETS) or any(int(value) <= 0 for value in counts.values()):
        raise PerceptionMemoryGateError("expected_counts must cover all datasets")
    if not candidates:
        raise PerceptionMemoryGateError("at least one checkpoint candidate is required")
    manifests = _validated_manifest_sha256(expected_manifest_sha256)
    _require_seed_scope(baseline, DEV_SEEDS, "Dev gate")
    baseline_rows, baseline_report, baseline_files = _load_method(
        baseline, counts, manifests
    )
    _require_valid_baseline(baseline_report)
    points: list[dict[str, Any]] = []
    input_files: dict[str, Any] = {baseline.method_id: baseline_files}
    seen = {baseline.method_id}
    for candidate in candidates:
        if candidate.method_id in seen:
            raise PerceptionMemoryGateError("method IDs must be unique")
        seen.add(candidate.method_id)
        _require_seed_scope(candidate, DEV_SEEDS, "Dev gate")
        rows, report, files = _load_method(candidate, counts, manifests)
        _audit_methods(baseline_rows, rows, candidate.method_id)
        points.append(
            _dev_point(
                baseline_report,
                report,
                _paired_costs(baseline_rows, rows),
            )
        )
        input_files[candidate.method_id] = files
    eligible = [item for item in points if item["passed"]]
    winner = (
        min(
            eligible,
            key=lambda item: (
                -float(item["summary"]["mean_correct"]),
                -min(float(value) for value in item["dataset_deltas"].values()),
                float(item["total_token_ratio"]),
                float(item["visual_token_ratio"]),
                float(item["summary"]["mean_stopped_with_incomplete_evidence"]),
                float(item["summary"]["mean_failure_rate"]),
                str(item["method_id"]),
            ),
        )
        if eligible
        else None
    )
    return {
        "schema_version": 1,
        "phase": "dev",
        "status": "passed" if winner else "blocked",
        "passed": winner is not None,
        "expected_counts": counts,
        "expected_manifest_sha256": manifests,
        "baseline": baseline_report,
        "candidates": points,
        "selected_method_id": winner["method_id"] if winner else None,
        "selection_order": (
            "mean_correct_desc,min_dataset_delta_desc,total_ratio_asc,"
            "visual_ratio_asc,incomplete_stops_asc,failure_rate_asc,method_id_asc"
        ),
        "inputs": input_files,
        "blocking_errors": [] if winner else ["no_checkpoint_passed_strict_dev_gate"],
    }


def evaluate_test_gate(
    *,
    baseline: MethodRuns,
    candidate: MethodRuns,
    expected_manifest_sha256: Mapping[str, str],
    dev_gate_report_path: Path,
    dev_gate_report_sha256: str,
    expected_counts: Mapping[str, int] | None = None,
) -> dict[str, Any]:
    counts = dict(expected_counts or {dataset: 100 for dataset in DATASETS})
    if counts != {dataset: 100 for dataset in DATASETS}:
        raise PerceptionMemoryGateError("final Test gate requires exactly 100 samples per dataset")
    manifests = _validated_manifest_sha256(expected_manifest_sha256)
    _require_seed_scope(baseline, TEST_SEEDS, "Test gate")
    _require_seed_scope(candidate, TEST_SEEDS, "Test gate")
    expected_dev_sha256 = _validated_sha256(
        dev_gate_report_sha256, "dev_gate_report_sha256"
    )
    dev_report_path = Path(dev_gate_report_path).resolve()
    actual_dev_sha256 = _sha256(dev_report_path)
    if actual_dev_sha256 != expected_dev_sha256:
        raise PerceptionMemoryGateError(
            "Dev gate report SHA-256 does not match the pinned value"
        )
    try:
        dev_report = json.loads(dev_report_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise PerceptionMemoryGateError("Dev gate report is not valid JSON") from exc
    if not isinstance(dev_report, Mapping):
        raise PerceptionMemoryGateError("Dev gate report must be an object")
    if (
        dev_report.get("phase") != "dev"
        or dev_report.get("status") != "passed"
        or dev_report.get("passed") is not True
    ):
        raise PerceptionMemoryGateError("Test requires a passed Dev gate report")
    selected_method_id = str(dev_report.get("selected_method_id") or "").strip()
    if not selected_method_id or selected_method_id != candidate.method_id:
        raise PerceptionMemoryGateError(
            "Test candidate must match the Dev gate selected_method_id"
        )
    raw_dev_candidates = dev_report.get("candidates")
    if not isinstance(raw_dev_candidates, list):
        raise PerceptionMemoryGateError("Dev gate report candidates must be an array")
    selected_entries = [
        item
        for item in raw_dev_candidates
        if isinstance(item, Mapping) and item.get("method_id") == selected_method_id
    ]
    if len(selected_entries) != 1 or selected_entries[0].get("passed") is not True:
        raise PerceptionMemoryGateError(
            "Dev gate report must contain one passed selected method"
        )
    selected_entry = selected_entries[0]
    dev_baseline = dev_report.get("baseline")
    if not isinstance(dev_baseline, Mapping) or dev_baseline.get("method_id") != baseline.method_id:
        raise PerceptionMemoryGateError(
            "Test baseline must match the Dev gate baseline method"
        )
    baseline_rows, baseline_report, baseline_files = _load_method(
        baseline, counts, manifests
    )
    _require_valid_baseline(baseline_report)
    candidate_rows, candidate_report, candidate_files = _load_method(
        candidate, counts, manifests
    )
    if dev_baseline.get("model_artifact_sha256") != baseline_report.get(
        "model_artifact_sha256"
    ):
        raise PerceptionMemoryGateError(
            "Test baseline model artifact does not match the Dev gate report"
        )
    if selected_entry.get("model_artifact_sha256") != candidate_report.get(
        "model_artifact_sha256"
    ):
        raise PerceptionMemoryGateError(
            "Test candidate model artifact does not match the Dev gate selection"
        )
    for label, dev_method, test_method in (
        ("baseline", dev_baseline, baseline_report),
        ("candidate", selected_entry, candidate_report),
    ):
        runtime_identity = dev_method.get("runtime_identity")
        runtime_identity_sha256 = dev_method.get("runtime_identity_sha256")
        if not isinstance(runtime_identity, Mapping):
            raise PerceptionMemoryGateError(
                f"Dev gate report is missing {label} runtime identity"
            )
        if _canonical_sha256(runtime_identity) != _validated_sha256(
            runtime_identity_sha256,
            f"Dev gate {label} runtime_identity_sha256",
        ):
            raise PerceptionMemoryGateError(
                f"Dev gate {label} runtime identity hash is invalid"
            )
        if (
            dict(runtime_identity) != test_method.get("runtime_identity")
            or runtime_identity_sha256
            != test_method.get("runtime_identity_sha256")
        ):
            raise PerceptionMemoryGateError(
                f"Test {label} runtime identity does not match the Dev gate report"
            )
    _audit_methods(baseline_rows, candidate_rows, candidate.method_id)
    base = baseline_report["summary"]
    current = candidate_report["summary"]
    dataset_deltas = {
        dataset: current["mean_correct_by_dataset"][dataset]
        - base["mean_correct_by_dataset"][dataset]
        for dataset in DATASETS
    }
    paired_costs = _paired_costs(baseline_rows, candidate_rows)
    total_ratio = float(paired_costs["total_token_ratio"])
    visual_ratio = float(paired_costs["visual_token_ratio"])
    conditions = {
        "accuracy_gain_at_least_6": current["mean_correct"] >= base["mean_correct"] + 6,
        "total_correct_at_least_157": current["mean_correct"] >= 157,
        "each_dataset_nondecrease": all(value >= 0 for value in dataset_deltas.values()),
        "dataset_floors_met": all(
            current["mean_correct_by_dataset"][dataset] >= TEST_FLOORS[dataset]
            for dataset in DATASETS
        ),
        "total_token_ratio_at_most_0_70": total_ratio <= 0.70,
        "visual_token_ratio_at_most_0_70": visual_ratio <= 0.70,
        "failure_rate_at_most_1pct": current["mean_failure_rate"] <= 0.01,
        "annotation_leak_zero": current["total_annotation_leaks"] == 0,
        "candidate_rerun_zero": current["total_candidate_reruns"] == 0,
        "duplicate_sample_id_zero": True,
    }
    passed = all(conditions.values())
    return {
        "schema_version": 1,
        "phase": "test",
        "status": "passed" if passed else "failed",
        "passed": passed,
        "expected_counts": counts,
        "expected_manifest_sha256": manifests,
        "dev_gate_binding": {
            "path": str(dev_report_path),
            "sha256": actual_dev_sha256,
            "selected_method_id": selected_method_id,
        },
        "baseline": baseline_report,
        "candidate": candidate_report,
        "dataset_deltas": dataset_deltas,
        "total_token_ratio": total_ratio,
        "visual_token_ratio": visual_ratio,
        "paired_costs": paired_costs,
        "conditions": conditions,
        "inputs": {
            baseline.method_id: baseline_files,
            candidate.method_id: candidate_files,
        },
        "blocking_errors": []
        if passed
        else [name for name, value in conditions.items() if not value],
    }


def method_from_config(payload: Mapping[str, Any]) -> MethodRuns:
    method_id = str(payload.get("method_id") or "").strip()
    raw_runs = payload.get("runs")
    if not isinstance(raw_runs, list):
        raise PerceptionMemoryGateError(f"{method_id or 'method'}: runs must be an array")
    runs: list[SeedRun] = []
    for raw in raw_runs:
        if not isinstance(raw, Mapping) or not isinstance(raw.get("paths"), Mapping):
            raise PerceptionMemoryGateError(f"{method_id}: invalid run config")
        try:
            seed = int(raw["seed"])
        except (KeyError, TypeError, ValueError) as exc:
            raise PerceptionMemoryGateError(f"{method_id}: invalid seed") from exc
        runs.append(
            SeedRun(
                seed,
                {str(key).lower(): Path(str(value)) for key, value in raw["paths"].items()},
            )
        )
    return MethodRuns(method_id, tuple(runs))


__all__ = [
    "DATASETS",
    "MethodRuns",
    "PerceptionMemoryGateError",
    "SeedRun",
    "TEST_FLOORS",
    "evaluate_dev_gate",
    "evaluate_test_gate",
    "method_from_config",
]
