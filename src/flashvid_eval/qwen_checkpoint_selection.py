from __future__ import annotations

import math
import statistics
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
    write_frozen_json,
)
from .qwen_reporting import paired_compare
from .qwen_sft import checkpoint_gate


def _number(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} is not numeric")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{field} is not finite")
    return number


def _aggregate_plan(path: Path, config_sha256: str, expected_phase: str) -> dict[str, Any]:
    plan = load_frozen_run_plan(path, config_sha256)
    if plan.get("phase") != expected_phase:
        raise ValueError(f"expected {expected_phase} plan, found {plan.get('phase')}")
    tasks = plan.get("tasks")
    if not isinstance(tasks, list):
        raise ValueError("run plan has no task list")
    runs: list[DevRun] = [load_task_result(plan, task) for task in tasks]
    if sorted(run.dataset for run in runs) != sorted(DATASETS):
        raise ValueError(f"{expected_phase} must contain exactly one task per dataset")
    identities: set[tuple[str, str]] = set()
    rows: list[dict[str, Any]] = []
    manifest_hashes: dict[str, str] = {}
    result_hashes: dict[str, str] = {}
    for run in runs:
        manifest_hash = str(next(
            task["manifest_sha256"]
            for task in tasks
            if task["task_id"] == run.task_id
        ))
        manifest_hashes[run.dataset] = manifest_hash
        result_hashes[run.dataset] = run.result_sha256
        for row in run.rows:
            identity = (run.dataset, str(row["sample_id"]))
            if identity in identities:
                raise ValueError(f"duplicate checkpoint sample identity: {identity}")
            identities.add(identity)
            rows.append({**row, "_dataset_identity": run.dataset})
    if len(rows) != 150:
        raise ValueError(f"{expected_phase} expected 150 rows, found {len(rows)}")

    failures = sum(
        bool(row.get("error") or row.get("error_type") or row.get("parse_error"))
        for row in rows
    )
    leaks = sum(
        row.get("annotation_leak_check") == "failed"
        or row.get("failure_class") == "annotation_leak"
        for row in rows
    )
    token_fields = ("total_tokens", "visual_tokens")
    means: dict[str, float] = {}
    for field in token_fields:
        values = [_number(row.get(field), field) for row in rows]
        means[f"mean_{field}"] = statistics.fmean(values)
    sample_digest = canonical_sha256(sorted(identities))
    return {
        "schema_version": 1,
        "phase": expected_phase,
        "run_plan": {
            "path": str(path.resolve()),
            "sha256": file_sha256(path),
            "plan_sha256": plan["plan_sha256"],
        },
        "manifest_sha256": canonical_sha256(manifest_hashes),
        "manifest_sha256_by_dataset": manifest_hashes,
        "sample_ids_sha256": sample_digest,
        "result_sha256_by_dataset": result_hashes,
        "aggregate": {
            "total": len(rows),
            "correct": sum(bool(row.get("correct")) for row in rows),
            "failure_rate": failures / len(rows),
            "annotation_leak": leaks,
            **means,
        },
        "rows": rows,
        "frozen_checkpoint": plan.get("frozen_checkpoint"),
    }


def _paired_rows(
    teacher: Mapping[str, Any], checkpoint: Mapping[str, Any]
) -> dict[str, Any]:
    def public_rows(source: Mapping[str, Any]) -> list[dict[str, Any]]:
        return [
            {
                **row,
                "sample_id": f"{row['_dataset_identity']}:{row['sample_id']}",
            }
            for row in source["rows"]
        ]

    return paired_compare(public_rows(teacher), public_rows(checkpoint))


def select_sft_checkpoint(
    *,
    config_sha256: str,
    teacher_plan: Path,
    checkpoint_plans: Sequence[Path],
) -> dict[str, Any]:
    if not checkpoint_plans:
        raise ValueError("at least one checkpoint plan is required")
    teacher = _aggregate_plan(teacher_plan, config_sha256, "teacher_dev")
    if teacher["aggregate"]["failure_rate"] > 0.01 or teacher["aggregate"]["annotation_leak"]:
        raise ValueError("Teacher Dev run failed engineering gates")
    points: list[dict[str, Any]] = []
    seen_checkpoints: set[str] = set()
    for plan_path in checkpoint_plans:
        checkpoint = _aggregate_plan(plan_path, config_sha256, "sft_dev")
        frozen = checkpoint.get("frozen_checkpoint")
        if not isinstance(frozen, Mapping):
            raise ValueError("SFT Dev plan has no frozen checkpoint identity")
        checkpoint_id = str(frozen.get("checkpoint_id") or "")
        if not checkpoint_id or checkpoint_id in seen_checkpoints:
            raise ValueError("checkpoint plans contain a duplicate/empty checkpoint id")
        seen_checkpoints.add(checkpoint_id)
        if checkpoint["manifest_sha256"] != teacher["manifest_sha256"]:
            raise ValueError("Teacher/checkpoint Dev manifests differ")
        if checkpoint["sample_ids_sha256"] != teacher["sample_ids_sha256"]:
            raise ValueError("Teacher/checkpoint Dev sample identities differ")
        engineering_passed = (
            checkpoint["aggregate"]["failure_rate"] <= 0.01
            and checkpoint["aggregate"]["annotation_leak"] == 0
        )
        gate = checkpoint_gate(teacher, checkpoint)
        points.append(
            {
                "checkpoint_id": checkpoint_id,
                "epoch": int(frozen.get("epoch")),
                "checkpoint_config": {
                    "path": str(frozen.get("path")),
                    "sha256": str(frozen.get("sha256")),
                },
                "run": {key: value for key, value in checkpoint.items() if key != "rows"},
                "paired": _paired_rows(teacher, checkpoint),
                "engineering_passed": engineering_passed,
                "gate": gate,
                "eligible": engineering_passed and bool(gate["passed"]),
            }
        )
    eligible = [point for point in points if point["eligible"]]
    selected = (
        min(
            eligible,
            key=lambda point: (
                -point["run"]["aggregate"]["correct"],
                point["run"]["aggregate"]["mean_total_tokens"],
                point["run"]["aggregate"]["mean_visual_tokens"],
                point["epoch"],
            ),
        )
        if eligible
        else None
    )
    report: dict[str, Any] = {
        "schema_version": 1,
        "status": "passed" if selected is not None else "blocked",
        "experiment_config_sha256": config_sha256,
        "policy": {
            "accuracy_strictly_higher": True,
            "mean_total_tokens_ratio_max": 0.70,
            "mean_visual_tokens_ratio_max": 0.70,
            "failure_rate_max": 0.01,
            "annotation_leak_max": 0,
            "selection_order": ["correct", "total_tokens", "visual_tokens", "epoch"],
        },
        "teacher": {key: value for key, value in teacher.items() if key != "rows"},
        "checkpoints": points,
        "selected": selected,
        "blocking_errors": [] if selected is not None else ["no_checkpoint_passed_strict_dual_gain_gate"],
    }
    report["selection_state_sha256"] = canonical_sha256(report)
    return report


def freeze_sft_winner(
    report: Mapping[str, Any], *, report_path: Path, report_sha256: str
) -> dict[str, Any]:
    selected = report.get("selected")
    if report.get("status") != "passed" or not isinstance(selected, Mapping):
        raise ValueError("cannot freeze an SFT winner from a blocked report")
    payload: dict[str, Any] = {
        "schema_version": 1,
        "experiment_config_sha256": report["experiment_config_sha256"],
        "checkpoint_id": selected["checkpoint_id"],
        "epoch": selected["epoch"],
        "checkpoint_config": selected["checkpoint_config"],
        "selection_report": {
            "path": str(report_path.resolve()),
            "sha256": report_sha256,
            "selection_state_sha256": report["selection_state_sha256"],
        },
    }
    payload["winner_fingerprint"] = canonical_sha256(payload)
    return payload


__all__ = [
    "freeze_sft_winner",
    "select_sft_checkpoint",
    "write_frozen_json",
]
