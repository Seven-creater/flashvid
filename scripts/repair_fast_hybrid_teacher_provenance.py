#!/usr/bin/env python3
"""Repair provenance omitted from failed Teacher rows and emit a new audit.

The evaluator used to omit immutable run metadata when an exception escaped
before the backend returned its normal result object.  This utility repairs
only those failed rows.  It never changes the error, prediction, model trace,
or scoring state, and it preserves the original failed audit as an immutable
record.
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

try:
    from fast_hybrid_bulk_common import file_sha256, freeze_json
    from launch_fast_hybrid_teacher_matrix import TeacherJob, audit_outputs
except ModuleNotFoundError:  # imported as a module by tests
    from scripts.fast_hybrid_bulk_common import file_sha256, freeze_json
    from scripts.launch_fast_hybrid_teacher_matrix import TeacherJob, audit_outputs


_REPAIRABLE_ERROR_TYPES = {"TimeoutError"}


def _load_object(path: Path, label: str) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object: {path}")
    return value


def _jobs_from_plan(plan: Mapping[str, Any]) -> list[TeacherJob]:
    if plan.get("schema_version") != 1 or plan.get("kind") != "fast_hybrid_teacher_matrix":
        raise ValueError("Teacher plan must be schema-v1 fast_hybrid_teacher_matrix")
    raw_jobs = plan.get("jobs")
    if not isinstance(raw_jobs, list) or not raw_jobs:
        raise ValueError("Teacher plan has no jobs")
    jobs: list[TeacherJob] = []
    for raw in raw_jobs:
        if not isinstance(raw, Mapping):
            raise ValueError("Teacher plan job must be an object")
        jobs.append(
            TeacherJob(
                job_id=str(raw["job_id"]),
                endpoint=str(raw["endpoint"]),
                command=tuple(str(value) for value in raw["command"]),
                log_path=Path(str(raw["log_path"])),
                output_path=Path(str(raw["output_path"])),
                expected_ids=tuple(str(value) for value in raw["expected_ids"]),
                dataset=str(raw["dataset"]),
                phase=str(raw["phase"]),
                schedule_id=str(raw["schedule_id"]),
                planner_seed=int(raw["planner_seed"]),
                active_manifest_sha256=str(raw["active_manifest_sha256"]),
            )
        )
    if len({job.job_id for job in jobs}) != len(jobs):
        raise ValueError("Teacher plan contains duplicate job_id values")
    return jobs


def _frozen_inputs(job: TeacherJob) -> dict[str, Any]:
    path = job.output_path.parent / f"frozen_inputs_{job.dataset}.json"
    frozen = _load_object(path, "Teacher frozen inputs")
    manifest = frozen.get("manifest")
    candidates = frozen.get("candidate_results")
    if (
        frozen.get("dataset") != job.dataset
        or not isinstance(manifest, Mapping)
        or manifest.get("sha256") != job.active_manifest_sha256
        or not isinstance(candidates, Mapping)
        or int(candidates.get("direct_rerun", -1)) != 0
        or frozen.get("trajectory_schedule_id") != job.schedule_id
        or int(frozen.get("generation_seed", -1)) != job.planner_seed
        or frozen.get("scoring_deferred") is not True
    ):
        raise RuntimeError(f"Teacher frozen inputs disagree with plan: {path}")
    return frozen


def _expected_fields(job: TeacherJob, frozen: Mapping[str, Any]) -> dict[str, Any]:
    candidate = frozen["candidate_results"]
    model_hash = frozen.get("model_artifact_sha256")
    return {
        "candidate_results_sha256": candidate.get("sha256"),
        "teacher_model_sha256": model_hash,
        "model_artifact_sha256": model_hash,
        "experiment_config_sha256": frozen.get("experiment_config_sha256"),
        "manifest_sha256": job.active_manifest_sha256,
        "train600_manifest_sha256": frozen.get("train600_manifest_sha256"),
        "trajectory_schedule_id": job.schedule_id,
        "trajectory_variant_id": frozen.get("trajectory_variant_id"),
        "trajectory_replica_id": frozen.get("trajectory_replica_id"),
        "teacher_temperature": frozen.get("teacher_temperature"),
        "generation_seed": job.planner_seed,
        "scoring_deferred": True,
    }


def _row_needs_repair(row: Mapping[str, Any], expected: Mapping[str, Any]) -> bool:
    return any(row.get(key) != value for key, value in expected.items())


def _repairable_row(
    row: Mapping[str, Any], *, job: TeacherJob, frozen: Mapping[str, Any]
) -> None:
    if (
        row.get("error_type") not in _REPAIRABLE_ERROR_TYPES
        or not str(row.get("error") or "").startswith("TimeoutError:")
        or row.get("scoring_deferred") is not True
        or int(row.get("candidate_rerun", -1)) != 0
        or row.get("annotation_leak_check") != "not_run"
        or row.get("run_fingerprint") != frozen.get("run_fingerprint")
    ):
        raise RuntimeError(
            f"refusing non-timeout or identity-mismatched repair: "
            f"{job.job_id}/{row.get('sample_id')}"
        )


def _atomic_lines(path: Path, lines: Sequence[str]) -> None:
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False, newline=""
    ) as handle:
        temporary = Path(handle.name)
        handle.writelines(lines)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _validate_existing_report(
    report: Mapping[str, Any], *, plan_path: Path, failed_audit_path: Path
) -> None:
    if (
        report.get("status") != "passed"
        or report.get("source_plan_sha256") != file_sha256(plan_path)
        or report.get("source_failed_audit_sha256") != file_sha256(failed_audit_path)
        or report.get("issues")
    ):
        raise RuntimeError("existing repaired audit does not match immutable inputs")
    for item in report.get("files", []):
        path = Path(str(item.get("path") or ""))
        if not path.is_file() or file_sha256(path) != item.get("after_sha256"):
            raise RuntimeError(f"repaired Teacher result changed after audit: {path}")


def repair_teacher_provenance(
    *, plan_path: Path, failed_audit_path: Path, output_audit_path: Path
) -> dict[str, Any]:
    """Repair strictly eligible rows and freeze a separately named passed audit."""

    if output_audit_path.exists():
        report = _load_object(output_audit_path, "repaired Teacher audit")
        _validate_existing_report(
            report, plan_path=plan_path, failed_audit_path=failed_audit_path
        )
        return report

    plan = _load_object(plan_path, "Teacher plan")
    failed = _load_object(failed_audit_path, "failed Teacher audit")
    if (
        failed.get("status") != "failed"
        or not isinstance(failed.get("issues"), list)
        or not failed["issues"]
        or any(
            not isinstance(issue, Mapping)
            or issue.get("reason") != "frozen_provenance_mismatch"
            for issue in failed["issues"]
        )
    ):
        raise RuntimeError("source audit is not an exclusively provenance-failed audit")

    jobs = _jobs_from_plan(plan)
    issue_jobs = {str(issue.get("job_id") or "") for issue in failed["issues"]}
    known_jobs = {job.job_id for job in jobs}
    if not issue_jobs or not issue_jobs <= known_jobs:
        raise RuntimeError("source audit refers to unknown Teacher jobs")

    repair_items: list[dict[str, Any]] = []
    changed_files: list[dict[str, Any]] = []
    for job in jobs:
        if job.job_id not in issue_jobs:
            continue
        frozen = _frozen_inputs(job)
        expected = _expected_fields(job, frozen)
        before_sha = file_sha256(job.output_path)
        lines = job.output_path.read_text(encoding="utf-8").splitlines(keepends=True)
        seen: set[str] = set()
        changed = False
        for index, line in enumerate(lines):
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{job.output_path}:{index + 1} is not an object")
            sample_id = str(row.get("sample_id") or "")
            if not sample_id or sample_id in seen:
                raise ValueError(f"{job.output_path} has duplicate/empty sample_id")
            seen.add(sample_id)
            if not _row_needs_repair(row, expected):
                continue
            _repairable_row(row, job=job, frozen=frozen)
            added: list[str] = []
            for key, value in expected.items():
                current = row.get(key)
                if current not in (None, value):
                    raise RuntimeError(
                        f"refusing conflicting provenance repair: "
                        f"{job.job_id}/{sample_id}/{key}"
                    )
                if current != value:
                    row[key] = value
                    added.append(key)
            lines[index] = json.dumps(row, ensure_ascii=False) + "\n"
            repair_items.append(
                {
                    "job_id": job.job_id,
                    "sample_id": sample_id,
                    "error_type": row.get("error_type"),
                    "added_fields": sorted(added),
                }
            )
            changed = True
        if seen != set(job.expected_ids):
            raise RuntimeError(f"Teacher result coverage mismatch: {job.job_id}")
        if changed:
            _atomic_lines(job.output_path, lines)
            changed_files.append(
                {
                    "path": str(job.output_path),
                    "before_sha256": before_sha,
                    "after_sha256": file_sha256(job.output_path),
                }
            )

    if not repair_items:
        raise RuntimeError("failed audit contained no strictly repairable rows")
    audit = audit_outputs(jobs)
    if audit.get("status") != "passed":
        raise RuntimeError(f"Teacher audit still fails after repair: {audit['issues'][:3]}")
    report = {
        "schema_version": 1,
        "kind": "fast_hybrid_teacher_repaired_audit",
        **audit,
        "source_plan": str(plan_path),
        "source_plan_sha256": file_sha256(plan_path),
        "source_failed_audit": str(failed_audit_path),
        "source_failed_audit_sha256": file_sha256(failed_audit_path),
        "repair_policy": "timeout_rows_provenance_only_v1",
        "repaired_rows": len(repair_items),
        "repairs": repair_items,
        "files": changed_files,
    }
    freeze_json(output_audit_path, report)
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--failed-audit", type=Path, required=True)
    parser.add_argument("--output-audit", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        report = repair_teacher_provenance(
            plan_path=args.plan,
            failed_audit_path=args.failed_audit,
            output_audit_path=args.output_audit,
        )
    except (OSError, ValueError, TypeError, KeyError, RuntimeError) as error:
        print(json.dumps({"status": "failed", "error": str(error)}, ensure_ascii=False))
        return 1
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
