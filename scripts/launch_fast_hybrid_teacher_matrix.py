#!/usr/bin/env python3
"""Validate and run the frozen Fast Hybrid Teacher matrix on two endpoints."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

try:
    from fast_hybrid_bulk_common import (
        BulkJob,
        detached_shell_line,
        execute_jobs,
        file_sha256,
        freeze_json,
        freeze_jsonl,
        load_frozen_config,
        parse_dataset_paths,
        read_jsonl,
        safe_id,
        shell_line,
        validate_inputs,
    )
except ModuleNotFoundError:  # imported as a module by tests
    from scripts.fast_hybrid_bulk_common import (  # type: ignore[no-redef]
        BulkJob,
        detached_shell_line,
        execute_jobs,
        file_sha256,
        freeze_json,
        freeze_jsonl,
        load_frozen_config,
        parse_dataset_paths,
        read_jsonl,
        safe_id,
        shell_line,
        validate_inputs,
    )


@dataclass(frozen=True)
class TeacherJob(BulkJob):
    dataset: str
    phase: str
    schedule_id: str
    planner_seed: int
    active_manifest_sha256: str


def _candidate_rows(path: Path) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    rows = read_jsonl(path)
    indexed: dict[str, dict[str, Any]] = {}
    for row in rows:
        sample_id = str(row.get("sample_id") or "")
        if not sample_id or sample_id in indexed:
            raise ValueError(f"{path} has a missing or duplicate candidate sample_id")
        request = row.get("protocol_request") or {}
        if (
            row.get("baseline_mode") != "direct"
            or row.get("sampling_id") != "uniform32"
            or row.get("enable_thinking") is not False
            or request.get("max_tokens") != 512
            or float(request.get("temperature", -1.0)) != 0.0
        ):
            raise ValueError(f"{path} is not a frozen clean Direct candidate file")
        indexed[sample_id] = row
    return rows, indexed


def _active_artifacts(
    *,
    result_root: Path,
    phase: str,
    schedule_id: str,
    dataset: str,
    expected_ids: set[str],
    source: Mapping[str, Any],
    candidate_path: Path,
    write: bool,
) -> tuple[Path, str, Path, str]:
    source_ids = set(source["index"])
    _, candidates = _candidate_rows(candidate_path)
    missing = sorted(expected_ids - set(candidates))
    if missing:
        raise RuntimeError(
            f"{dataset} frozen candidate file is missing {len(missing)} planned samples: {missing[:3]}"
        )
    if expected_ids == source_ids and set(candidates) == source_ids:
        return (
            Path(source["path"]),
            str(source["sha256"]),
            candidate_path,
            file_sha256(candidate_path),
        )

    directory = result_root / "frozen" / "run_subsets" / phase / safe_id(schedule_id)
    manifest_path = directory / f"{dataset}_manifest.jsonl"
    candidate_subset_path = directory / f"{dataset}_candidates.jsonl"
    manifest_rows = [
        row for row in source["rows"] if str(row.get("sample_id") or "") in expected_ids
    ]
    candidate_rows = [
        candidates[str(row["sample_id"])] for row in manifest_rows
    ]
    manifest_text = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in manifest_rows)
    candidate_text = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in candidate_rows)
    if write:
        freeze_jsonl(manifest_path, manifest_rows)
        freeze_jsonl(candidate_subset_path, candidate_rows)
    import hashlib

    return (
        manifest_path,
        hashlib.sha256(manifest_text.encode("utf-8")).hexdigest(),
        candidate_subset_path,
        hashlib.sha256(candidate_text.encode("utf-8")).hexdigest(),
    )


def build_jobs(
    *,
    config: Mapping[str, Any],
    config_sha256: str,
    specs: Sequence[Mapping[str, Any]],
    source_rows: Mapping[str, Mapping[str, Any]],
    candidate_paths: Mapping[str, Path],
    python: str,
    repo_root: Path,
    concurrency: int,
    timeout: float,
    resume: bool,
    write_artifacts: bool,
) -> list[TeacherJob]:
    teacher = config["teacher"]
    endpoints = [str(value).rstrip("/") for value in teacher.get("base_urls") or []]
    if not endpoints or len(set(endpoints)) != len(endpoints):
        raise ValueError("teacher.base_urls must contain distinct endpoints")
    grouped: dict[tuple[str, str], list[Mapping[str, Any]]] = {}
    for spec in specs:
        key = (str(spec["schedule_id"]), str(spec["dataset"]).lower())
        grouped.setdefault(key, []).append(spec)
    result_root = Path(str(config["result_root"]))
    jobs: list[TeacherJob] = []
    for index, ((schedule_id, dataset), rows) in enumerate(sorted(grouped.items())):
        expected_ids = {str(row["sample_id"]) for row in rows}
        if len(expected_ids) != len(rows):
            raise ValueError(f"duplicate samples in {schedule_id}/{dataset}")
        common = {
            key: {str(row.get(key)) for row in rows}
            for key in (
                "phase",
                "variant_id",
                "replica_id",
                "planner_seed",
                "max_total_visual_tokens",
                "max_call_visual_tokens",
                "max_turns",
            )
        }
        if any(len(values) != 1 for values in common.values()):
            raise ValueError(f"mixed schedule parameters in {schedule_id}/{dataset}")
        phase = next(iter(common["phase"]))
        if phase == "base" and expected_ids != set(source_rows[dataset]["index"]):
            raise RuntimeError(
                "base plans must be complete Train200 schedules; use the frozen full plan with --resume"
            )
        active_manifest, active_hash, active_candidates, _ = _active_artifacts(
            result_root=result_root,
            phase=phase,
            schedule_id=schedule_id,
            dataset=dataset,
            expected_ids=expected_ids,
            source=source_rows[dataset],
            candidate_path=candidate_paths[dataset],
            write=write_artifacts,
        )
        endpoint = endpoints[index % len(endpoints)]
        output_dir = result_root / "trajectories" / "raw" / phase / safe_id(schedule_id) / dataset
        frame_root = result_root / "frames" / phase / safe_id(schedule_id) / dataset
        output_path = output_dir / f"{dataset}_fast_hybrid_eva.jsonl"
        command = [
            python,
            str(repo_root / "scripts" / "evaluate_mcq.py"),
            "--dataset",
            dataset,
            "--backend",
            "fast_hybrid_eva",
            "--annotations",
            str(config["datasets"][dataset]["annotations"]),
            "--video-root",
            str(config["datasets"][dataset]["video_root"]),
            "--base-url",
            endpoint,
            "--api-key",
            "no",
            "--model",
            str(teacher["model"]),
            "--sample",
            str(len(expected_ids)),
            "--seed",
            next(iter(common["planner_seed"])),
            "--output-dir",
            str(output_dir),
            "--manifest",
            str(active_manifest),
            "--expected-manifest-sha256",
            active_hash,
            "--frame-root",
            str(frame_root),
            "--concurrency",
            str(concurrency),
            "--max-turns",
            next(iter(common["max_turns"])),
            "--max-call-visual-tokens",
            next(iter(common["max_call_visual_tokens"])),
            "--max-total-visual-tokens",
            next(iter(common["max_total_visual_tokens"])),
            "--agent-version",
            str(teacher["agent_version"]),
            "--timeout",
            str(timeout),
            "--defer-scoring",
            "--trajectory-schedule-id",
            schedule_id,
            "--train600-manifest-sha256",
            str(config["train600"]["sha256"]),
            "--trajectory-replica-id",
            next(iter(common["replica_id"])),
            "--trajectory-variant-id",
            next(iter(common["variant_id"])),
            "--candidate-results",
            str(active_candidates),
            "--controller-temperature",
            str(teacher["temperature"]),
            "--experiment-config-sha256",
            config_sha256,
            "--model-artifact-sha256",
            str(teacher["model_artifact_sha256"]),
        ]
        if resume:
            command.append("--resume")
        jobs.append(
            TeacherJob(
                job_id=f"{phase}:{schedule_id}:{dataset}",
                endpoint=endpoint,
                command=tuple(command),
                log_path=result_root / "logs" / "teacher" / phase / f"{safe_id(schedule_id)}_{dataset}.log",
                output_path=output_path,
                expected_ids=tuple(sorted(expected_ids)),
                dataset=dataset,
                phase=phase,
                schedule_id=schedule_id,
                planner_seed=int(next(iter(common["planner_seed"]))),
                active_manifest_sha256=active_hash,
            )
        )
    return jobs


def audit_outputs(jobs: Sequence[TeacherJob]) -> dict[str, Any]:
    issues: list[dict[str, Any]] = []
    total = 0
    for job in jobs:
        if not job.output_path.is_file():
            issues.append({"job_id": job.job_id, "reason": "result_missing"})
            continue
        rows = read_jsonl(job.output_path)
        ids = [str(row.get("sample_id") or "") for row in rows]
        total += len(rows)
        if len(ids) != len(set(ids)) or set(ids) != set(job.expected_ids):
            issues.append({"job_id": job.job_id, "reason": "sample_matrix_mismatch"})
            continue
        for row in rows:
            if (
                row.get("scoring_deferred") is not True
                or row.get("trajectory_schedule_id") != job.schedule_id
                or int(row.get("generation_seed", -1)) != job.planner_seed
                or row.get("manifest_sha256") != job.active_manifest_sha256
                or int(row.get("candidate_rerun", -1)) != 0
            ):
                issues.append(
                    {
                        "job_id": job.job_id,
                        "sample_id": row.get("sample_id"),
                        "reason": "frozen_provenance_mismatch",
                    }
                )
                break
    return {
        "status": "passed" if not issues else "failed",
        "jobs": len(jobs),
        "rows": total,
        "issues": issues,
    }


def _strip_detached_args(argv: Sequence[str]) -> list[str]:
    result: list[str] = []
    skip = False
    for value in argv:
        if skip:
            skip = False
            continue
        if value == "--print-nohup-command":
            continue
        if value == "--detached-log":
            skip = True
            continue
        result.append(value)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--expected-config-sha256", required=True)
    parser.add_argument("--specs", type=Path, required=True)
    parser.add_argument(
        "--candidate-results",
        action="append",
        default=[],
        metavar="DATASET=PATH",
        required=True,
    )
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--concurrency-per-endpoint", type=int, default=16)
    parser.add_argument("--timeout", type=float, default=80.0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--print-nohup-command", action="store_true")
    parser.add_argument("--detached-log", type=Path)
    args = parser.parse_args(argv)
    try:
        if args.concurrency_per_endpoint <= 0 or args.timeout <= 0:
            raise ValueError("concurrency and timeout must be positive")
        config, config_hash = load_frozen_config(
            args.config, args.expected_config_sha256
        )
        candidates = parse_dataset_paths(args.candidate_results, "candidate-results")
        for dataset, path in candidates.items():
            if not path.is_file():
                raise FileNotFoundError(f"{dataset} candidate file not found: {path}")
        specs, sources, specs_sha = validate_inputs(config, config_hash, args.specs)
        jobs = build_jobs(
            config=config,
            config_sha256=config_hash,
            specs=specs,
            source_rows=sources,
            candidate_paths=candidates,
            python=args.python,
            repo_root=args.repo_root,
            concurrency=args.concurrency_per_endpoint,
            timeout=args.timeout,
            resume=args.resume,
            write_artifacts=not args.dry_run and not args.print_nohup_command,
        )
        detached_log = args.detached_log or (
            Path(str(config["result_root"])) / "logs" / f"teacher_launcher_{specs_sha[:12]}.log"
        )
        if args.print_nohup_command:
            raw_argv = list(argv) if argv is not None else sys.argv[1:]
            print(
                detached_shell_line(
                    repo_root=args.repo_root,
                    python=args.python,
                    script=Path(__file__).resolve(),
                    argv=_strip_detached_args(raw_argv),
                    log=detached_log,
                )
            )
            return 0
        plan = {
            "schema_version": 1,
            "kind": "fast_hybrid_teacher_matrix",
            "config_sha256": config_hash,
            "specs_path": str(args.specs.resolve()),
            "specs_sha256": specs_sha,
            "candidate_sha256s": {
                dataset: file_sha256(path) for dataset, path in candidates.items()
            },
            "jobs": [
                {
                    **asdict(job),
                    "command": list(job.command),
                    "log_path": str(job.log_path),
                    "output_path": str(job.output_path),
                }
                for job in jobs
            ],
        }
        plan_path = (
            Path(str(config["result_root"]))
            / "run_plans"
            / f"teacher_{specs_sha[:12]}.json"
        )
        print(json.dumps({**plan, "plan_path": str(plan_path)}, ensure_ascii=False, indent=2, default=str))
        for job in jobs:
            print(shell_line(job.command))
        if args.dry_run:
            return 0
        freeze_json(plan_path, plan)
        execute_jobs(
            jobs,
            repo_root=args.repo_root,
            retry_failed_processes=False,
        )
        audit = audit_outputs(jobs)
        freeze_json(plan_path.with_name(plan_path.stem + "_audit.json"), audit)
        print(json.dumps(audit, ensure_ascii=False, indent=2))
        return 0 if audit["status"] == "passed" else 1
    except (OSError, ValueError, TypeError, KeyError, RuntimeError) as error:
        print(
            json.dumps(
                {"status": "failed", "error": f"{type(error).__name__}: {error}"},
                ensure_ascii=False,
            )
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
