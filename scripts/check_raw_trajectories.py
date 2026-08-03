#!/usr/bin/env python3
"""Read-only hard gate for raw FlashVID budget trajectories.

Selection and SFT must not start until all three training datasets are complete,
trajectory identities are unique, and the union of engineering-failure rows is
at most the preregistered threshold.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import sys
import tempfile
from typing import Any, Mapping, Sequence


DATASETS = ("lvbench", "lsdbench", "cgbench")
OPTION_RE = re.compile(r"^[A-H]$")
FAILURE_FIELDS = (
    "error",
    "api_error",
    "frame_error",
    "transcode_error",
    "perception_error",
    "controller_error",
    "verifier_error",
    "parse_error",
    "failure_stage",
    "data_unavailable",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_jsonl(
    path: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], int]:
    if not path.is_file():
        raise FileNotFoundError(path)
    records: list[dict[str, Any]] = []
    invalid: list[dict[str, Any]] = []
    nonempty_lines = 0
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            nonempty_lines += 1
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                invalid.append({"line": line_number, "reason": f"invalid JSON: {exc}"})
                continue
            if not isinstance(value, dict):
                invalid.append({"line": line_number, "reason": "row must be an object"})
                continue
            records.append(value)
    return records, invalid, nonempty_lines


def _candidate_reruns(record: Mapping[str, Any]) -> int:
    value = record.get("candidate_rerun")
    try:
        count = int(value or 0)
    except (TypeError, ValueError):
        return 1
    return count if count >= 0 else 1


def _record_failure_reasons(
    record: Mapping[str, Any], expected_dataset: str
) -> set[str]:
    reasons = {field for field in FAILURE_FIELDS if record.get(field)}
    if str(record.get("dataset") or "").strip() != expected_dataset:
        reasons.add("dataset_mismatch")
    if not str(record.get("trajectory_id") or "").strip():
        reasons.add("missing_trajectory_id")
    if record.get("trajectory_valid") is not True:
        reasons.add("trajectory_not_valid")
    if record.get("annotation_leak_check") != "passed":
        reasons.add("annotation_leak")
    if _candidate_reruns(record):
        reasons.add("candidate_rerun")
    prediction = str(
        record.get("final_prediction") or record.get("prediction") or ""
    ).strip().upper()
    if OPTION_RE.fullmatch(prediction) is None:
        reasons.add("invalid_prediction")
    messages = record.get("training_messages")
    if not isinstance(messages, list) or not messages:
        reasons.add("missing_training_messages")
    steps = record.get("tool_steps")
    if not isinstance(steps, list) or not steps:
        reasons.add("missing_tool_steps")
    else:
        for step in steps:
            if not isinstance(step, Mapping):
                reasons.add("invalid_tool_step")
                continue
            measured = step.get("api_multimodal_video_tokens_actual")
            if (
                isinstance(measured, bool)
                or not isinstance(measured, int)
                or measured < 0
            ):
                reasons.add("missing_api_multimodal_measurement")
            if (
                step.get("raw_visual_tokens_actual") is not None
                or step.get("retained_visual_tokens_actual") is not None
            ):
                reasons.add("unsupported_visual_actual_claim")
            if "estimate" not in str(step.get("token_count_source") or ""):
                reasons.add("missing_visual_token_provenance")
            for field in FAILURE_FIELDS:
                if step.get(field):
                    reasons.add(f"tool_step_{field}")
    return reasons


def validate_raw_trajectories(
    records_by_dataset: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    invalid_lines_by_dataset: Mapping[str, int] | None = None,
    expected_per_dataset: int = 4800,
    maximum_failure_rate: float = 0.01,
) -> dict[str, Any]:
    """Validate already-loaded records without modifying any trajectory file."""
    invalid_lines_by_dataset = invalid_lines_by_dataset or {}
    supplied = set(records_by_dataset)
    expected = set(DATASETS)
    dataset_counts: dict[str, dict[str, Any]] = {}
    failure_reasons: Counter[str] = Counter()
    failure_examples: list[dict[str, Any]] = []
    trajectory_ids: Counter[str] = Counter()
    missing_trajectory_ids = 0
    dataset_mismatches = 0
    candidate_rerun_total = 0
    leak_failures = 0
    failed_records = 0

    for dataset in DATASETS:
        records = records_by_dataset.get(dataset, ())
        invalid_count = int(invalid_lines_by_dataset.get(dataset, 0))
        actual = len(records)
        dataset_counts[dataset] = {
            "actual_valid_records": actual,
            "invalid_json_lines": invalid_count,
            "expected": expected_per_dataset,
            "passed": actual == expected_per_dataset and invalid_count == 0,
        }
        failed_records += invalid_count
        if invalid_count:
            failure_reasons["invalid_json"] += invalid_count
        for record in records:
            trajectory_id = str(record.get("trajectory_id") or "").strip()
            if trajectory_id:
                trajectory_ids[trajectory_id] += 1
            else:
                missing_trajectory_ids += 1
            if str(record.get("dataset") or "").strip() != dataset:
                dataset_mismatches += 1
            reruns = _candidate_reruns(record)
            candidate_rerun_total += reruns
            if record.get("annotation_leak_check") != "passed":
                leak_failures += 1
            reasons = _record_failure_reasons(record, dataset)
            if not reasons:
                continue
            failed_records += 1
            failure_reasons.update(reasons)
            if len(failure_examples) < 100:
                failure_examples.append(
                    {
                        "dataset": dataset,
                        "sample_id": str(record.get("sample_id") or ""),
                        "trajectory_id": trajectory_id,
                        "reasons": sorted(reasons),
                    }
                )

    valid_record_count = sum(len(records_by_dataset.get(name, ())) for name in DATASETS)
    invalid_line_count = sum(int(invalid_lines_by_dataset.get(name, 0)) for name in DATASETS)
    evaluated_rows = valid_record_count + invalid_line_count
    expected_total = expected_per_dataset * len(DATASETS)
    failure_rate = failed_records / evaluated_rows if evaluated_rows else None
    duplicate_ids = sorted(
        trajectory_id for trajectory_id, count in trajectory_ids.items() if count > 1
    )
    visual_audit_counts = {
        name: int(failure_reasons.get(name, 0))
        for name in (
            "missing_api_multimodal_measurement",
            "unsupported_visual_actual_claim",
            "missing_visual_token_provenance",
        )
    }

    constraints = {
        "dataset_inputs": {
            "actual": sorted(supplied),
            "expected": list(DATASETS),
            "passed": supplied == expected,
        },
        "dataset_record_counts": {
            "datasets": dataset_counts,
            "passed": all(item["passed"] for item in dataset_counts.values()),
        },
        "total_record_count": {
            "actual_valid_records": valid_record_count,
            "invalid_json_lines": invalid_line_count,
            "expected": expected_total,
            "passed": valid_record_count == expected_total and invalid_line_count == 0,
        },
        "engineering_failure_rate": {
            "failed_records": failed_records,
            "evaluated_rows": evaluated_rows,
            "actual": failure_rate,
            "maximum": maximum_failure_rate,
            "unit": "union_of_rows_with_any_engineering_failure",
            "passed": failure_rate is not None and failure_rate <= maximum_failure_rate,
        },
        "duplicate_trajectory_ids": {
            "actual": len(duplicate_ids),
            "maximum": 0,
            "passed": not duplicate_ids,
        },
        "trajectory_identity": {
            "missing_trajectory_ids": missing_trajectory_ids,
            "dataset_mismatches": dataset_mismatches,
            "passed": missing_trajectory_ids == 0 and dataset_mismatches == 0,
        },
        "annotation_leak_failures": {
            "actual": leak_failures,
            "maximum": 0,
            "passed": leak_failures == 0,
        },
        "candidate_reruns": {
            "actual": candidate_rerun_total,
            "maximum": 0,
            "passed": candidate_rerun_total == 0,
        },
        "visual_token_audit": {
            "counts": visual_audit_counts,
            "maximum_each": 0,
            "passed": all(value == 0 for value in visual_audit_counts.values()),
        },
    }
    failed_constraints = [
        name for name, value in constraints.items() if not value["passed"]
    ]
    return {
        "status": "passed" if not failed_constraints else "failed",
        "constraints": constraints,
        "counts": {
            "valid_records": valid_record_count,
            "invalid_json_lines": invalid_line_count,
            "evaluated_rows": evaluated_rows,
            "failed_records": failed_records,
            "unique_trajectory_ids": len(trajectory_ids),
            "duplicate_trajectory_id_count": len(duplicate_ids),
        },
        "failure_breakdown": dict(sorted(failure_reasons.items())),
        "duplicate_trajectory_ids": duplicate_ids[:100],
        "duplicates_truncated": len(duplicate_ids) > 100,
        "failure_examples": failure_examples,
        "failure_examples_truncated": failed_records > len(failure_examples),
        "failed_constraints": failed_constraints,
    }


def _parse_input(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise ValueError(f"trajectory input must be DATASET=PATH: {value}")
    dataset, raw_path = value.split("=", 1)
    dataset = dataset.strip()
    if dataset not in DATASETS:
        raise ValueError(f"unsupported dataset: {dataset}")
    if not raw_path.strip():
        raise ValueError(f"trajectory path is empty for {dataset}")
    return dataset, Path(raw_path).expanduser()


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    temporary.replace(path)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Hard-gate all raw trajectories before selection and SFT."
    )
    parser.add_argument(
        "--trajectory",
        action="append",
        required=True,
        metavar="DATASET=PATH",
    )
    parser.add_argument("--expected-per-dataset", type=int, default=4800)
    parser.add_argument("--maximum-failure-rate", type=float, default=0.01)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)

    try:
        if args.expected_per_dataset < 1:
            raise ValueError("expected-per-dataset must be positive")
        if not 0 <= args.maximum_failure_rate <= 1:
            raise ValueError("maximum-failure-rate must be in [0, 1]")
        paths: dict[str, Path] = {}
        for value in args.trajectory:
            dataset, path = _parse_input(value)
            if dataset in paths:
                raise ValueError(f"duplicate trajectory input for {dataset}")
            paths[dataset] = path.resolve()
        if set(paths) != set(DATASETS):
            raise ValueError(
                f"expected trajectory inputs for {list(DATASETS)}, got {sorted(paths)}"
            )

        records_by_dataset: dict[str, list[dict[str, Any]]] = {}
        invalid_counts: dict[str, int] = {}
        input_report: dict[str, dict[str, Any]] = {}
        for dataset in DATASETS:
            records, invalid, nonempty_lines = _read_jsonl(paths[dataset])
            records_by_dataset[dataset] = records
            invalid_counts[dataset] = len(invalid)
            input_report[dataset] = {
                "path": str(paths[dataset]),
                "sha256": _sha256(paths[dataset]),
                "nonempty_lines": nonempty_lines,
                "invalid_lines": invalid[:100],
                "invalid_lines_truncated": len(invalid) > 100,
            }
        report = validate_raw_trajectories(
            records_by_dataset,
            invalid_lines_by_dataset=invalid_counts,
            expected_per_dataset=args.expected_per_dataset,
            maximum_failure_rate=args.maximum_failure_rate,
        )
        report["inputs"] = input_report
    except Exception as exc:
        report = {
            "status": "failed",
            "failed_constraints": ["checker_execution"],
            "error": f"{type(exc).__name__}: {exc}",
        }

    _atomic_write_json(args.output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report.get("status") == "passed" else 1


if __name__ == "__main__":
    sys.exit(main())
