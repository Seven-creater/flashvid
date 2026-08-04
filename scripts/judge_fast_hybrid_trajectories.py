#!/usr/bin/env python3
"""Candidate-blind 3-seed Qwen Judge for frozen Fast Hybrid trajectories."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from flashvid_eval.client import OpenAICompatibleClient
from flashvid_eval.fast_hybrid_trajectory_judge import (
    FastHybridTrajectoryJudge,
    TrajectoryJudgeConfig,
    bind_trajectory_jobs,
)
from flashvid_eval.qwen_sft import read_jsonl


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False
    ) as handle:
        temporary = Path(handle.name)
        for row in sorted(rows, key=lambda item: str(item["trajectory_id"])):
            handle.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
    os.replace(temporary, path)


def _load_many(paths: Sequence[Path]) -> list[dict[str, Any]]:
    return [row for path in paths for row in read_jsonl(path)]


def run(args: argparse.Namespace) -> dict[str, Any]:
    specs = read_jsonl(args.specs)
    trajectories = _load_many(args.trajectories)
    jobs = bind_trajectory_jobs(
        specs,
        trajectories,
        schedule_id=args.schedule_id,
    )
    config = TrajectoryJudgeConfig(
        model=args.model,
        judge_seeds=tuple(args.judge_seed),
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        enable_thinking=False,
    )
    client = OpenAICompatibleClient(
        args.base_url,
        api_key=args.api_key,
        timeout=args.timeout,
    )
    judge = FastHybridTrajectoryJudge(client, config)

    existing_rows = read_jsonl(args.output) if args.output.is_file() else []
    if existing_rows and not args.resume:
        raise RuntimeError("output exists; pass --resume to continue it")
    rows: dict[str, dict[str, Any]] = {}
    for row in existing_rows:
        trajectory_id = str(row.get("trajectory_id") or "")
        if not trajectory_id or trajectory_id in rows:
            raise RuntimeError("resume has missing or duplicate trajectory_id")
        rows[trajectory_id] = row
    expected_ids = {job.trajectory_id for job in jobs}
    extras = sorted(set(rows) - expected_ids)
    if extras:
        raise RuntimeError(f"resume contains trajectories outside this schedule: {extras[:3]}")

    lock = threading.Lock()

    def update(row: dict[str, Any]) -> None:
        with lock:
            rows[str(row["trajectory_id"])] = row
            _write_jsonl(args.output, rows.values())

    def execute(job: Any) -> dict[str, Any]:
        result = judge.judge(
            job,
            existing=rows.get(job.trajectory_id),
            on_update=update,
        )
        update(result)
        return result

    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures = [pool.submit(execute, job) for job in jobs]
        for future in as_completed(futures):
            future.result()

    completed = sum(
        1 for row in rows.values() if row.get("judge_status") == "complete"
    )
    failed = sum(
        1 for row in rows.values() if row.get("judge_status") == "complete_with_failures"
    )
    return {
        "schedule_id": args.schedule_id,
        "judge_config_sha256": config.fingerprint(),
        "trajectories": len(jobs),
        "completed": completed,
        "complete_with_failures": failed,
        "output": str(args.output.resolve()),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--specs", type=Path, required=True)
    parser.add_argument("--trajectories", type=Path, nargs="+", required=True)
    parser.add_argument("--schedule-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY", "no"))
    parser.add_argument("--model", default="Qwen3.5-9B")
    parser.add_argument("--judge-seed", type=int, action="append", default=None)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--timeout", type=float, default=80.0)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    args.judge_seed = args.judge_seed or [17, 42, 73]
    if args.concurrency <= 0:
        parser.error("--concurrency must be positive")
    if args.temperature < 0:
        parser.error("--temperature cannot be negative")
    try:
        summary = run(args)
    except (OSError, ValueError, TypeError, KeyError, RuntimeError) as error:
        print(
            json.dumps(
                {"status": "failed", "error": f"{type(error).__name__}: {error}"},
                ensure_ascii=False,
            )
        )
        return 1
    print(json.dumps({"status": "passed", **summary}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
