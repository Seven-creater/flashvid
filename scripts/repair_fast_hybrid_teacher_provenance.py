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
import re
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
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def _require_sha256(value: Any, label: str) -> str:
    digest = str(value or "").strip().lower()
    if not _SHA256_RE.fullmatch(digest):
        raise ValueError(f"{label} must be a SHA-256 digest")
    return digest


def _command_arg(command: Sequence[str], flag: str) -> str:
    indexes = [index for index, value in enumerate(command) if value == flag]
    if len(indexes) != 1 or indexes[0] + 1 >= len(command):
        raise ValueError(f"Teacher command must contain exactly one {flag}")
    return str(command[indexes[0] + 1])


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


def _frozen_inputs(job: TeacherJob, plan: Mapping[str, Any]) -> dict[str, Any]:
    path = job.output_path.parent / f"frozen_inputs_{job.dataset}.json"
    frozen = _load_object(path, "Teacher frozen inputs")
    manifest = frozen.get("manifest")
    candidates = frozen.get("candidate_results")
    candidate_hashes = plan.get("candidate_sha256s")
    if not isinstance(candidate_hashes, Mapping):
        raise ValueError("Teacher plan has no candidate_sha256s")
    config_hash = _require_sha256(plan.get("config_sha256"), "plan config_sha256")
    expected_candidate_hash = _require_sha256(
        candidate_hashes.get(job.dataset), f"plan {job.dataset} candidate SHA-256"
    )
    model_hash = _require_sha256(
        frozen.get("model_artifact_sha256"), "frozen model_artifact_sha256"
    )
    train_hash = _require_sha256(
        frozen.get("train600_manifest_sha256"), "frozen Train600 SHA-256"
    )
    run_fingerprint = _require_sha256(
        frozen.get("run_fingerprint"), "frozen run_fingerprint"
    )
    if (
        frozen.get("dataset") != job.dataset
        or not isinstance(manifest, Mapping)
        or manifest.get("sha256") != job.active_manifest_sha256
        or not isinstance(candidates, Mapping)
        or int(candidates.get("direct_rerun", -1)) != 0
        or frozen.get("trajectory_schedule_id") != job.schedule_id
        or int(frozen.get("generation_seed", -1)) != job.planner_seed
        or frozen.get("scoring_deferred") is not True
        or frozen.get("experiment_config_sha256") != config_hash
        or candidates.get("sha256") != expected_candidate_hash
        or _command_arg(job.command, "--expected-manifest-sha256")
        != job.active_manifest_sha256
        or int(_command_arg(job.command, "--seed")) != job.planner_seed
        or _command_arg(job.command, "--trajectory-schedule-id") != job.schedule_id
        or _command_arg(job.command, "--trajectory-variant-id")
        != frozen.get("trajectory_variant_id")
        or int(_command_arg(job.command, "--trajectory-replica-id"))
        != int(frozen.get("trajectory_replica_id", -1))
        or _command_arg(job.command, "--experiment-config-sha256") != config_hash
        or _command_arg(job.command, "--model-artifact-sha256") != model_hash
        or _command_arg(job.command, "--train600-manifest-sha256") != train_hash
    ):
        raise RuntimeError(f"Teacher frozen inputs disagree with plan: {path}")
    frozen["run_fingerprint"] = run_fingerprint
    return frozen


def _expected_fields(job: TeacherJob, frozen: Mapping[str, Any]) -> dict[str, Any]:
    candidate = frozen["candidate_results"]
    model_hash = _require_sha256(
        frozen.get("model_artifact_sha256"), "frozen model_artifact_sha256"
    )
    teacher_model_hash = _require_sha256(
        frozen.get("teacher_model_artifact_sha256") or model_hash,
        "frozen teacher_model_artifact_sha256",
    )
    return {
        "candidate_results_sha256": _require_sha256(
            candidate.get("sha256"), "frozen candidate SHA-256"
        ),
        "teacher_model_sha256": teacher_model_hash,
        "model_artifact_sha256": model_hash,
        "experiment_config_sha256": _require_sha256(
            frozen.get("experiment_config_sha256"),
            "frozen experiment_config_sha256",
        ),
        "manifest_sha256": _require_sha256(
            job.active_manifest_sha256, "job active_manifest_sha256"
        ),
        "train600_manifest_sha256": _require_sha256(
            frozen.get("train600_manifest_sha256"), "frozen Train600 SHA-256"
        ),
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


def _stage_lines(path: Path, lines: Sequence[str]) -> tuple[Path, str]:
    staged = path.with_name(path.name + ".provenance-repair.partial")
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False, newline=""
    ) as handle:
        temporary = Path(handle.name)
        handle.writelines(lines)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, staged)
    return staged, file_sha256(staged)


def _intent_path(output_audit_path: Path) -> Path:
    return output_audit_path.with_name(output_audit_path.stem + "_intent.json")


def _validate_existing_report(
    report: Mapping[str, Any], *, plan_path: Path, failed_audit_path: Path
) -> None:
    repairs = report.get("repairs")
    files = report.get("files")
    if (
        report.get("schema_version") != 1
        or report.get("kind") != "fast_hybrid_teacher_repaired_audit"
        or report.get("status") != "passed"
        or report.get("source_plan_sha256") != file_sha256(plan_path)
        or report.get("source_failed_audit_sha256") != file_sha256(failed_audit_path)
        or report.get("issues")
        or not isinstance(repairs, list)
        or not repairs
        or int(report.get("repaired_rows", -1)) != len(repairs)
        or not isinstance(files, list)
        or not files
    ):
        raise RuntimeError("existing repaired audit does not match immutable inputs")
    intent_path = Path(str(report.get("repair_intent") or ""))
    if (
        not intent_path.is_file()
        or file_sha256(intent_path) != report.get("repair_intent_sha256")
    ):
        raise RuntimeError("existing repaired audit has a missing or changed intent")
    for item in files:
        if not isinstance(item, Mapping):
            raise RuntimeError("existing repaired audit has an invalid file entry")
        path = Path(str(item.get("path") or ""))
        if not path.is_file() or file_sha256(path) != item.get("after_sha256"):
            raise RuntimeError(f"repaired Teacher result changed after audit: {path}")


def _validate_failed_audit(failed: Mapping[str, Any]) -> set[str]:
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
    return {str(issue.get("job_id") or "") for issue in failed["issues"]}


def _validate_intent(
    intent: Mapping[str, Any], *, plan_path: Path, failed_audit_path: Path
) -> None:
    if (
        intent.get("schema_version") != 1
        or intent.get("kind") != "fast_hybrid_teacher_provenance_repair_intent"
        or intent.get("source_plan_sha256") != file_sha256(plan_path)
        or intent.get("source_failed_audit_sha256") != file_sha256(failed_audit_path)
        or not isinstance(intent.get("repairs"), list)
        or not intent["repairs"]
        or not isinstance(intent.get("files"), list)
        or not intent["files"]
    ):
        raise RuntimeError("repair intent does not match immutable inputs")


def _apply_intent(
    intent: Mapping[str, Any], *, jobs: Sequence[TeacherJob], output_audit_path: Path
) -> dict[str, Any]:
    for item in intent["files"]:
        if not isinstance(item, Mapping):
            raise RuntimeError("repair intent has an invalid file entry")
        path = Path(str(item.get("path") or ""))
        staged = Path(str(item.get("staged_path") or ""))
        before_sha = str(item.get("before_sha256") or "")
        after_sha = str(item.get("after_sha256") or "")
        current_sha = file_sha256(path)
        if current_sha == after_sha:
            continue
        if current_sha != before_sha:
            raise RuntimeError(f"Teacher result changed during provenance repair: {path}")
        if not staged.is_file() or file_sha256(staged) != after_sha:
            raise RuntimeError(f"staged provenance repair is missing or changed: {staged}")
        os.replace(staged, path)

    audit = audit_outputs(jobs)
    if audit.get("status") != "passed":
        raise RuntimeError(f"Teacher audit still fails after repair: {audit['issues'][:3]}")
    intent_path = Path(str(intent["repair_intent"]))
    report = {
        "schema_version": 1,
        "kind": "fast_hybrid_teacher_repaired_audit",
        **audit,
        "source_plan": intent["source_plan"],
        "source_plan_sha256": intent["source_plan_sha256"],
        "source_failed_audit": intent["source_failed_audit"],
        "source_failed_audit_sha256": intent["source_failed_audit_sha256"],
        "repair_intent": str(intent_path),
        "repair_intent_sha256": file_sha256(intent_path),
        "repair_policy": "timeout_rows_provenance_only_v2",
        "repaired_rows": len(intent["repairs"]),
        "repairs": intent["repairs"],
        "files": intent["files"],
    }
    freeze_json(output_audit_path, report)
    for item in intent["files"]:
        staged = Path(str(item.get("staged_path") or ""))
        if staged.is_file():
            staged.unlink()
    return report


def repair_teacher_provenance(
    *, plan_path: Path, failed_audit_path: Path, output_audit_path: Path
) -> dict[str, Any]:
    """Repair strictly eligible rows and freeze a separately named passed audit."""

    plan = _load_object(plan_path, "Teacher plan")
    failed = _load_object(failed_audit_path, "failed Teacher audit")
    jobs = _jobs_from_plan(plan)
    issue_jobs = _validate_failed_audit(failed)
    known_jobs = {job.job_id for job in jobs}
    if not issue_jobs or not issue_jobs <= known_jobs:
        raise RuntimeError("source audit refers to unknown Teacher jobs")

    if output_audit_path.exists():
        report = _load_object(output_audit_path, "repaired Teacher audit")
        _validate_existing_report(
            report, plan_path=plan_path, failed_audit_path=failed_audit_path
        )
        if audit_outputs(jobs).get("status") != "passed":
            raise RuntimeError("current Teacher results no longer pass repaired audit")
        return report

    intent_path = _intent_path(output_audit_path)
    if intent_path.exists():
        intent = _load_object(intent_path, "Teacher provenance repair intent")
        _validate_intent(
            intent, plan_path=plan_path, failed_audit_path=failed_audit_path
        )
        return _apply_intent(intent, jobs=jobs, output_audit_path=output_audit_path)

    repair_items: list[dict[str, Any]] = []
    changed_files: list[dict[str, Any]] = []
    for job in jobs:
        frozen = _frozen_inputs(job, plan) if job.job_id in issue_jobs else None
        expected = _expected_fields(job, frozen) if frozen is not None else None
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
            mismatch = (
                row.get("scoring_deferred") is not True
                or row.get("trajectory_schedule_id") != job.schedule_id
                or int(row.get("generation_seed", -1)) != job.planner_seed
                or row.get("manifest_sha256") != job.active_manifest_sha256
                or int(row.get("candidate_rerun", -1)) != 0
            )
            if mismatch:
                if expected is None or frozen is None:
                    raise RuntimeError(
                        f"unreported Teacher provenance mismatch: {job.job_id}/{sample_id}"
                    )
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
                if _row_needs_repair(row, expected):
                    raise RuntimeError(
                        f"provenance repair remained incomplete: {job.job_id}/{sample_id}"
                    )
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
            staged, after_sha = _stage_lines(job.output_path, lines)
            changed_files.append(
                {
                    "path": str(job.output_path),
                    "before_sha256": before_sha,
                    "after_sha256": after_sha,
                    "staged_path": str(staged),
                }
            )

    if not repair_items:
        raise RuntimeError("failed audit contained no strictly repairable rows")
    if {item["job_id"] for item in repair_items} != issue_jobs:
        raise RuntimeError("failed audit job set differs from full repair preflight")
    intent = {
        "schema_version": 1,
        "kind": "fast_hybrid_teacher_provenance_repair_intent",
        "source_plan": str(plan_path),
        "source_plan_sha256": file_sha256(plan_path),
        "source_failed_audit": str(failed_audit_path),
        "source_failed_audit_sha256": file_sha256(failed_audit_path),
        "repair_intent": str(intent_path),
        "repair_policy": "timeout_rows_provenance_only_v2",
        "repairs": repair_items,
        "files": changed_files,
    }
    freeze_json(intent_path, intent)
    return _apply_intent(intent, jobs=jobs, output_audit_path=output_audit_path)


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
