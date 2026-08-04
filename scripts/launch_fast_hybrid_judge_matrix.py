#!/usr/bin/env python3
"""Validate and run candidate-blind Fast Hybrid Judges by frozen schedule."""

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
        load_frozen_config,
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
        load_frozen_config,
        read_jsonl,
        safe_id,
        shell_line,
        validate_inputs,
    )


@dataclass(frozen=True)
class JudgeJob(BulkJob):
    phase: str
    schedule_id: str
    required_judge_seeds: tuple[int, ...]


def _trajectory_index(paths: Sequence[Path]) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for path in paths:
        for row in read_jsonl(path):
            trajectory_id = str(row.get("trajectory_id") or "")
            if not trajectory_id or trajectory_id in indexed:
                raise ValueError("Judge trajectories have a missing or duplicate trajectory_id")
            forbidden = {
                key
                for key in ("answer", "correct", "time_range", "clue_intervals", "question_type")
                if key in row
            }
            if forbidden:
                raise ValueError(
                    f"private scoring fields in Judge trajectory {trajectory_id}: {sorted(forbidden)}"
                )
            indexed[trajectory_id] = row
    return indexed


def build_jobs(
    *,
    config: Mapping[str, Any],
    specs: Sequence[Mapping[str, Any]],
    specs_path: Path,
    trajectory_paths: Sequence[Path],
    python: str,
    repo_root: Path,
    concurrency: int,
    timeout: float,
    max_tokens: int,
    resume: bool,
) -> list[JudgeJob]:
    teacher = config["teacher"]
    endpoints = [str(value).rstrip("/") for value in teacher.get("base_urls") or []]
    if not endpoints or len(set(endpoints)) != len(endpoints):
        raise ValueError("teacher.base_urls must contain distinct endpoints")
    trajectories = _trajectory_index(trajectory_paths)
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for spec in specs:
        grouped.setdefault(str(spec["schedule_id"]), []).append(spec)
    generation = config["trajectory_generation"]
    configured_seeds = tuple(int(seed) for seed in generation["judge_seeds"])
    temperature = float(generation["judge_temperature"])
    result_root = Path(str(config["result_root"]))
    jobs: list[JudgeJob] = []
    for index, (schedule_id, rows) in enumerate(sorted(grouped.items())):
        expected_ids = tuple(sorted(str(row["trajectory_id"]) for row in rows))
        if len(expected_ids) != len(set(expected_ids)):
            raise ValueError(f"duplicate trajectory in Judge schedule {schedule_id}")
        missing = sorted(set(expected_ids) - set(trajectories))
        if missing:
            raise RuntimeError(
                f"Judge schedule {schedule_id} is missing {len(missing)} trajectories: {missing[:3]}"
            )
        phases = {str(row.get("phase") or "") for row in rows}
        seed_sets = {tuple(int(seed) for seed in row["required_judge_seeds"]) for row in rows}
        if len(phases) != 1 or seed_sets != {configured_seeds}:
            raise ValueError(f"mixed phase or Judge seeds in {schedule_id}")
        phase = next(iter(phases))
        endpoint = endpoints[index % len(endpoints)]
        output = result_root / "trajectories" / "judged" / phase / f"{safe_id(schedule_id)}.jsonl"
        command = [
            python,
            str(repo_root / "scripts" / "judge_fast_hybrid_trajectories.py"),
            "--specs",
            str(specs_path),
            "--trajectories",
            *[str(path) for path in trajectory_paths],
            "--schedule-id",
            schedule_id,
            "--output",
            str(output),
            "--base-url",
            endpoint,
            "--api-key",
            "no",
            "--model",
            str(teacher["model"]),
            "--max-tokens",
            str(max_tokens),
            "--temperature",
            str(temperature),
            "--timeout",
            str(timeout),
            "--concurrency",
            str(concurrency),
        ]
        for seed in configured_seeds:
            command.extend(("--judge-seed", str(seed)))
        if resume:
            command.append("--resume")
        jobs.append(
            JudgeJob(
                job_id=f"{phase}:{schedule_id}",
                endpoint=endpoint,
                command=tuple(command),
                log_path=result_root / "logs" / "judge" / phase / f"{safe_id(schedule_id)}.log",
                output_path=output,
                expected_ids=expected_ids,
                phase=phase,
                schedule_id=schedule_id,
                required_judge_seeds=configured_seeds,
            )
        )
    return jobs


def audit_outputs(jobs: Sequence[JudgeJob]) -> dict[str, Any]:
    issues: list[dict[str, Any]] = []
    total = 0
    complete = 0
    failed = 0
    for job in jobs:
        if not job.output_path.is_file():
            issues.append({"job_id": job.job_id, "reason": "result_missing"})
            continue
        rows = read_jsonl(job.output_path)
        ids = [str(row.get("trajectory_id") or "") for row in rows]
        total += len(rows)
        if len(ids) != len(set(ids)) or set(ids) != set(job.expected_ids):
            issues.append({"job_id": job.job_id, "reason": "trajectory_matrix_mismatch"})
            continue
        for row in rows:
            confirmations = row.get("judge_confirmations")
            seeds = (
                tuple(sorted(int(item["judge_seed"]) for item in confirmations))
                if isinstance(confirmations, list)
                else ()
            )
            if seeds != tuple(sorted(job.required_judge_seeds)):
                issues.append(
                    {
                        "job_id": job.job_id,
                        "trajectory_id": row.get("trajectory_id"),
                        "reason": "judge_seed_matrix_mismatch",
                    }
                )
                continue
            status = row.get("judge_status")
            complete += status == "complete"
            failed += status == "complete_with_failures"
            if status not in {"complete", "complete_with_failures"}:
                issues.append(
                    {
                        "job_id": job.job_id,
                        "trajectory_id": row.get("trajectory_id"),
                        "reason": "judge_status_incomplete",
                    }
                )
    return {
        "status": "passed" if not issues else "failed",
        "jobs": len(jobs),
        "rows": total,
        "complete": complete,
        "complete_with_failures": failed,
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
    parser.add_argument("--trajectories", type=Path, nargs="+", required=True)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--concurrency-per-endpoint", type=int, default=16)
    parser.add_argument("--timeout", type=float, default=80.0)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--retry-failed-processes", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--print-nohup-command", action="store_true")
    parser.add_argument("--detached-log", type=Path)
    args = parser.parse_args(argv)
    try:
        if args.concurrency_per_endpoint <= 0 or args.timeout <= 0 or args.max_tokens <= 0:
            raise ValueError("concurrency, timeout and max-tokens must be positive")
        config, config_hash = load_frozen_config(
            args.config, args.expected_config_sha256
        )
        for path in args.trajectories:
            if not path.is_file():
                raise FileNotFoundError(path)
        specs, _, specs_sha = validate_inputs(config, config_hash, args.specs)
        jobs = build_jobs(
            config=config,
            specs=specs,
            specs_path=args.specs,
            trajectory_paths=args.trajectories,
            python=args.python,
            repo_root=args.repo_root,
            concurrency=args.concurrency_per_endpoint,
            timeout=args.timeout,
            max_tokens=args.max_tokens,
            resume=args.resume or args.retry_failed_processes,
        )
        detached_log = args.detached_log or (
            Path(str(config["result_root"])) / "logs" / f"judge_launcher_{specs_sha[:12]}.log"
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
            "kind": "fast_hybrid_judge_matrix",
            "config_sha256": config_hash,
            "specs_path": str(args.specs.resolve()),
            "specs_sha256": specs_sha,
            "trajectory_sha256s": {
                str(path.resolve()): file_sha256(path) for path in args.trajectories
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
            / f"judge_{specs_sha[:12]}.json"
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
            retry_failed_processes=args.retry_failed_processes,
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
