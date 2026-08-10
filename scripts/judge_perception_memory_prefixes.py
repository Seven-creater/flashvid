#!/usr/bin/env python3
"""Run resumable, high-concurrency text-only Judges over evidence prefixes."""

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
from flashvid_eval.perception_memory_prefix_judge import (
    PerceptionMemoryPrefixJudge,
    PrefixJudgeConfig,
    bind_prefix_jobs,
)
from flashvid_eval.qwen_sft import read_jsonl


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False
    ) as handle:
        temporary = Path(handle.name)
        for row in sorted(rows, key=lambda item: str(item["prefix_id"])):
            handle.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
    os.replace(temporary, path)


def _append_progress(path: Path, row: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _latest_rows(paths: Iterable[Path]) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for path in paths:
        if not path.is_file():
            continue
        for row in read_jsonl(path):
            prefix_id = str(row.get("prefix_id") or "")
            if not prefix_id:
                raise RuntimeError(f"row in {path} has no prefix_id")
            rows[prefix_id] = row
    return rows


def run(args: argparse.Namespace) -> dict[str, Any]:
    trajectories = [
        row for path in args.trajectories for row in read_jsonl(path)
    ]
    jobs = bind_prefix_jobs(trajectories)
    config = PrefixJudgeConfig(
        model=args.model,
        judge_seeds=tuple(args.judge_seed),
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        enable_thinking=False,
    )
    configured_urls = getattr(args, "base_urls", None)
    if configured_urls is None:
        legacy_url = getattr(args, "base_url", None)
        configured_urls = [legacy_url or "http://127.0.0.1:8200/v1"]
    clients = [
        OpenAICompatibleClient(url, api_key=args.api_key, timeout=args.timeout)
        for url in configured_urls
    ]
    judge = PerceptionMemoryPrefixJudge(clients, config)

    progress_path = args.output.with_suffix(args.output.suffix + ".progress.jsonl")
    if (args.output.is_file() or progress_path.is_file()) and not args.resume:
        raise RuntimeError("output/progress exists; pass --resume to continue")
    rows = _latest_rows((args.output, progress_path)) if args.resume else {}
    retry_errors = bool(getattr(args, "retry_errors", False))
    if retry_errors and not args.resume:
        raise ValueError("--retry-errors requires --resume")
    expected = {job.prefix_id for job in jobs}
    extras = sorted(set(rows) - expected)
    if extras:
        raise RuntimeError(f"resume contains prefixes outside the input: {extras[:3]}")
    # Validate source/config fingerprints without issuing sequential API calls.
    # With --retry-errors this also drops only failed/invalid seeds, preserving
    # successful confirmations for the concurrent retry pass below.
    for job in jobs:
        if job.prefix_id in rows:
            rows[job.prefix_id] = judge.prepare_resume(
                job,
                rows[job.prefix_id],
                retry_errors=retry_errors,
            )

    lock = threading.Lock()

    def update(row: dict[str, Any]) -> None:
        # Append-only progress makes each seed durable without repeatedly rewriting
        # the full result matrix.  The compact output is written once at the end.
        with lock:
            rows[str(row["prefix_id"])] = row
            _append_progress(progress_path, row)

    def execute(job: Any) -> dict[str, Any]:
        result = judge.judge(
            job,
            existing=rows.get(job.prefix_id),
            on_update=update,
            retry_errors=retry_errors,
        )
        with lock:
            rows[job.prefix_id] = result
        return result

    pending = [
        job
        for job in jobs
        if rows.get(job.prefix_id, {}).get("judge_status") != "complete"
        and (
            retry_errors
            or rows.get(job.prefix_id, {}).get("judge_status")
            != "complete_with_failures"
        )
    ]
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures = [pool.submit(execute, job) for job in pending]
        for future in as_completed(futures):
            future.result()

    # Compact both old output and append-only progress into one unique row/prefix.
    _write_jsonl(args.output, (rows[prefix_id] for prefix_id in sorted(expected)))
    progress_path.write_text("", encoding="utf-8")
    return {
        "prefix_judge_config_sha256": config.fingerprint(),
        "trajectories": len(trajectories),
        "prefixes": len(jobs),
        "complete": sum(
            row.get("judge_status") == "complete" for row in rows.values()
        ),
        "complete_with_failures": sum(
            row.get("judge_status") == "complete_with_failures"
            for row in rows.values()
        ),
        "retry_errors": retry_errors,
        "endpoint_count": judge.endpoint_count,
        "output": str(args.output.resolve()),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trajectories", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--base-url",
        action="append",
        dest="base_urls",
        help=(
            "OpenAI-compatible endpoint; repeat to distribute prefixes "
            "deterministically across endpoints. Defaults to 8200."
        ),
    )
    parser.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY", "no"))
    parser.add_argument("--model", default="Qwen3.5-9B")
    parser.add_argument("--judge-seed", type=int, action="append", default=None)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--timeout", type=float, default=80.0)
    parser.add_argument("--concurrency", type=int, default=32)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--retry-errors",
        action="store_true",
        help=(
            "With --resume, rerun only seeds whose persisted confirmation has "
            "error!=null or parsed_valid=false; successful seeds are preserved."
        ),
    )
    args = parser.parse_args(argv)
    args.judge_seed = args.judge_seed or [17, 42, 73]
    if args.concurrency <= 0:
        parser.error("--concurrency must be positive")
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
