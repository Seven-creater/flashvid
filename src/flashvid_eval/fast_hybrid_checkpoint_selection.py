"""Select a Fast Hybrid EVA SFT checkpoint on frozen Dev150 results."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from statistics import fmean
from typing import Any, Mapping, Sequence

from .fast_hybrid_eval_protocol import is_engineering_failure, is_model_fallback
from .qwen_sft import read_jsonl, sha256_file


DATASETS = ("lvbench", "lsdbench", "cgbench")


@dataclass(frozen=True)
class CheckpointRun:
    checkpoint_id: str
    epoch: int
    result_paths: Mapping[str, Path]


def _number(row: Mapping[str, Any], key: str) -> float:
    value = row.get(key)
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) < 0
    ):
        raise ValueError(f"{key} must be a finite non-negative number")
    return float(value)


def _complete_cost(row: Mapping[str, Any]) -> bool:
    try:
        _number(row, "end_to_end_total_tokens")
        _number(row, "end_to_end_visual_tokens")
        _number(row, "end_to_end_latency_s")
    except ValueError:
        return False
    return bool(
        row.get("candidate_cost_complete") is True
        and row.get("end_to_end_total_tokens_complete") is True
        and row.get("end_to_end_visual_tokens_complete") is True
    )


def _load_dataset(
    path: Path, dataset: str, *, allow_incomplete_failures: bool = False
) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for row in read_jsonl(path):
        if str(row.get("dataset") or "").lower() != dataset:
            raise ValueError(f"{path}: row belongs to a different dataset")
        sample_id = str(row.get("sample_id") or "").strip()
        if not sample_id or sample_id in rows:
            raise ValueError(f"{path}: missing or duplicate sample_id {sample_id!r}")
        answer = str(row.get("answer") or "").strip().upper()
        prediction = str(row.get("prediction") or "").strip().upper()
        if len(answer) != 1 or not "A" <= answer <= "H":
            raise ValueError(f"{path}: {sample_id} has no valid answer")
        if prediction and (len(prediction) != 1 or not "A" <= prediction <= "H"):
            raise ValueError(f"{path}: {sample_id} has an invalid prediction")
        if row.get("candidate_cost_complete") is not True:
            raise ValueError(f"{path}: {sample_id} has incomplete frozen-candidate cost")
        if not _complete_cost(row) and not (
            allow_incomplete_failures and is_engineering_failure(row)
        ):
            for key in (
                "end_to_end_total_tokens",
                "end_to_end_visual_tokens",
                "end_to_end_latency_s",
            ):
                try:
                    _number(row, key)
                except ValueError as error:
                    raise ValueError(f"{path}: {sample_id} has invalid {key}") from error
            if row.get("end_to_end_total_tokens_complete") is not True:
                raise ValueError(
                    f"{path}: {sample_id} has incomplete end_to_end_total_tokens"
                )
            raise ValueError(
                f"{path}: {sample_id} has incomplete end_to_end_visual_tokens"
            )
        rows[sample_id] = dict(row)
    if len(rows) != 50:
        raise ValueError(f"{path}: expected 50 Dev rows, found {len(rows)}")
    return rows


def _load_run(
    paths: Mapping[str, Path], *, allow_incomplete_failures: bool = False
) -> tuple[dict[tuple[str, str], dict[str, Any]], dict[str, str]]:
    if set(paths) != set(DATASETS):
        raise ValueError("a Dev run must contain LVBench, LSDBench and CG-Bench")
    rows: dict[tuple[str, str], dict[str, Any]] = {}
    hashes: dict[str, str] = {}
    for dataset in DATASETS:
        path = Path(paths[dataset])
        dataset_rows = _load_dataset(
            path, dataset, allow_incomplete_failures=allow_incomplete_failures
        )
        hashes[dataset] = sha256_file(path)
        rows.update({(dataset, sample_id): row for sample_id, row in dataset_rows.items()})
    return rows, hashes


def _failure(row: Mapping[str, Any]) -> bool:
    return is_engineering_failure(row)


def _audit_against_teacher(
    teacher: Mapping[tuple[str, str], Mapping[str, Any]],
    candidate: Mapping[tuple[str, str], Mapping[str, Any]],
) -> None:
    if set(candidate) != set(teacher):
        raise ValueError("Teacher and checkpoint Dev sample identities differ")
    for identity in sorted(teacher):
        baseline = teacher[identity]
        row = candidate[identity]
        if row.get("answer") != baseline.get("answer"):
            raise ValueError(f"{identity}: Teacher/checkpoint answers differ")
        for key in (
            "candidate_answer",
            "candidate_results_sha256",
            "agent_version",
            "official_eva_commit",
            "experiment_config_sha256",
        ):
            if baseline.get(key) is not None and row.get(key) != baseline.get(key):
                raise ValueError(f"{identity}: frozen field changed: {key}")
        baseline_hash = (baseline.get("prompt_hashes") or {}).get("verification")
        row_hash = (row.get("prompt_hashes") or {}).get("verification")
        if baseline_hash is not None and baseline_hash != row_hash:
            raise ValueError(f"{identity}: verification prompt hash changed")


def _summarize(
    rows: Mapping[tuple[str, str], Mapping[str, Any]],
    *,
    cost_identities: set[tuple[str, str]] | None = None,
) -> dict[str, Any]:
    ordered = [rows[key] for key in sorted(rows)]
    if cost_identities is None:
        cost_identities = {key for key, row in rows.items() if _complete_cost(row)}
    cost_rows = [rows[key] for key in sorted(cost_identities)]
    if not cost_rows:
        raise ValueError("no complete rows are available for cost comparison")
    if any(not _complete_cost(row) for row in cost_rows):
        raise ValueError("cost comparison includes an incomplete row")
    per_dataset = {
        dataset: sum(
            str(row.get("prediction") or "").upper()
            == str(row.get("answer") or "").upper()
            for (row_dataset, _sample_id), row in rows.items()
            if row_dataset == dataset
        )
        for dataset in DATASETS
    }
    errors = sum(_failure(row) for row in ordered)
    leaks = sum(
        row.get("annotation_leak_check") != "passed" and not _failure(row)
        for row in ordered
    )
    reruns = sum(int(row.get("candidate_rerun", 0) or 0) for row in ordered)
    tool_calls = [len(row.get("tool_calls") or []) for row in ordered]
    return {
        "samples": len(ordered),
        "correct": sum(per_dataset.values()),
        "correct_by_dataset": per_dataset,
        "mean_total_tokens": fmean(
            _number(row, "end_to_end_total_tokens") for row in cost_rows
        ),
        "mean_visual_tokens": fmean(
            _number(row, "end_to_end_visual_tokens") for row in cost_rows
        ),
        "mean_latency_s": fmean(
            _number(row, "end_to_end_latency_s") for row in cost_rows
        ),
        "mean_tool_calls": fmean(tool_calls),
        "engineering_failures": errors,
        "model_fallbacks": sum(is_model_fallback(row) for row in ordered),
        "failure_rate": errors / len(ordered),
        "cost_samples": len(cost_rows),
        "cost_excluded": len(ordered) - len(cost_rows),
        "annotation_leaks": leaks,
        "candidate_reruns": reruns,
    }


def _paired_counts(
    teacher: Mapping[tuple[str, str], Mapping[str, Any]],
    candidate: Mapping[tuple[str, str], Mapping[str, Any]],
) -> dict[str, int]:
    fixed = harmed = 0
    for identity, baseline in teacher.items():
        row = candidate[identity]
        answer = str(baseline["answer"])
        baseline_correct = str(baseline.get("prediction") or "") == answer
        candidate_correct = str(row.get("prediction") or "") == answer
        fixed += int(not baseline_correct and candidate_correct)
        harmed += int(baseline_correct and not candidate_correct)
    return {"teacher_wrong_sft_right": fixed, "teacher_right_sft_wrong": harmed}


def select_fast_hybrid_checkpoint(
    *,
    teacher_paths: Mapping[str, Path],
    checkpoints: Sequence[CheckpointRun],
) -> dict[str, Any]:
    if not checkpoints:
        raise ValueError("at least one SFT checkpoint is required")
    teacher_rows, teacher_hashes = _load_run(
        teacher_paths, allow_incomplete_failures=True
    )
    teacher_cost_identities = {
        identity for identity, row in teacher_rows.items() if _complete_cost(row)
    }
    teacher_summary = _summarize(
        teacher_rows, cost_identities=teacher_cost_identities
    )
    points: list[dict[str, Any]] = []
    seen: set[str] = set()
    for checkpoint in checkpoints:
        if not checkpoint.checkpoint_id.strip() or checkpoint.checkpoint_id in seen:
            raise ValueError("checkpoint IDs must be unique and non-empty")
        if checkpoint.epoch not in {1, 2, 3}:
            raise ValueError("checkpoint epoch must be 1, 2 or 3")
        seen.add(checkpoint.checkpoint_id)
        rows, hashes = _load_run(checkpoint.result_paths)
        _audit_against_teacher(teacher_rows, rows)
        summary = _summarize(rows, cost_identities=teacher_cost_identities)
        dataset_deltas = {
            dataset: summary["correct_by_dataset"][dataset]
            - teacher_summary["correct_by_dataset"][dataset]
            for dataset in DATASETS
        }
        conditions = {
            "overall_accuracy_strictly_higher": summary["correct"]
            > teacher_summary["correct"],
            "each_dataset_not_lower": all(value >= 0 for value in dataset_deltas.values()),
            "total_tokens_strictly_lower": summary["mean_total_tokens"]
            < teacher_summary["mean_total_tokens"],
            "visual_tokens_strictly_lower": summary["mean_visual_tokens"]
            < teacher_summary["mean_visual_tokens"],
            "failure_rate_at_most_1pct": summary["failure_rate"] <= 0.01,
            "annotation_leak_zero": summary["annotation_leaks"] == 0,
            "candidate_rerun_zero": summary["candidate_reruns"] == 0,
        }
        total_ratio = summary["mean_total_tokens"] / teacher_summary["mean_total_tokens"]
        visual_ratio = summary["mean_visual_tokens"] / teacher_summary["mean_visual_tokens"]
        points.append(
            {
                "checkpoint_id": checkpoint.checkpoint_id,
                "epoch": checkpoint.epoch,
                "summary": summary,
                "dataset_deltas": dataset_deltas,
                "paired": _paired_counts(teacher_rows, rows),
                "conditions": conditions,
                "passed": all(conditions.values()),
                "total_token_ratio": total_ratio,
                "visual_token_ratio": visual_ratio,
                "clearly_reduced_30pct": total_ratio <= 0.70 and visual_ratio <= 0.70,
                "files": hashes,
            }
        )
    eligible = [point for point in points if point["passed"]]
    selected = (
        min(
            eligible,
            key=lambda point: (
                -int(point["summary"]["correct"]),
                -min(int(value) for value in point["dataset_deltas"].values()),
                float(point["summary"]["mean_total_tokens"]),
                float(point["summary"]["mean_visual_tokens"]),
                float(point["summary"]["mean_tool_calls"]),
                float(point["summary"]["mean_latency_s"]),
                int(point["epoch"]),
            ),
        )
        if eligible
        else None
    )
    return {
        "schema_version": 1,
        "status": "passed" if selected is not None else "blocked",
        "teacher": {"summary": teacher_summary, "files": teacher_hashes},
        "checkpoints": points,
        "selected": (
            {
                "checkpoint_id": selected["checkpoint_id"],
                "epoch": selected["epoch"],
                "clearly_reduced_30pct": selected["clearly_reduced_30pct"],
            }
            if selected is not None
            else None
        ),
        "blocking_errors": (
            []
            if selected is not None
            else ["no_checkpoint_passed_accuracy_and_strict_token_reduction_gate"]
        ),
    }


__all__ = ["CheckpointRun", "select_fast_hybrid_checkpoint"]
