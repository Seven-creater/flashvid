#!/usr/bin/env python3
"""Run resumable three-seed visual-only completeness checks over cached prefixes."""

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
from flashvid_eval.perception_memory_visual_csv import (
    VisualCsvConfig,
    VisualCsvVerifier,
    bind_visual_csv_jobs,
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


def _append(path: Path, row: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _latest(paths: Iterable[Path]) -> dict[str, dict[str, Any]]:
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
    trajectories = [row for path in args.trajectories for row in read_jsonl(path)]
    jobs = bind_visual_csv_jobs(trajectories)
    config = VisualCsvConfig(
        model=args.model,
        seeds=tuple(args.judge_seed),
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        enable_thinking=False,
    )
    urls = args.base_urls or ["http://127.0.0.1:8200/v1"]
    clients = [
        OpenAICompatibleClient(
            url,
            api_key=args.api_key,
            timeout=args.timeout,
            local_file_urls_as_paths=args.local_media_paths,
        )
        for url in urls
    ]
    verifier = VisualCsvVerifier(clients, config)
    progress = args.output.with_suffix(args.output.suffix + ".progress.jsonl")
    if (args.output.is_file() or progress.is_file()) and not args.resume:
        raise RuntimeError("output/progress exists; pass --resume to continue")
    rows = _latest((args.output, progress)) if args.resume else {}
    expected = {job.prefix_id for job in jobs}
    extras = sorted(set(rows) - expected)
    if extras:
        raise RuntimeError(f"resume contains prefixes outside input: {extras[:3]}")
    if args.retry_errors and not args.resume:
        raise ValueError("--retry-errors requires --resume")
    if args.resume:
        for job in jobs:
            if job.prefix_id in rows:
                rows[job.prefix_id] = verifier.prepare_resume(
                    job,
                    rows[job.prefix_id],
                    retry_errors=args.retry_errors,
                )
    lock = threading.Lock()

    def update(row: dict[str, Any]) -> None:
        with lock:
            rows[str(row["prefix_id"])] = row
            _append(progress, row)

    def execute(job: Any) -> dict[str, Any]:
        result = verifier.verify(
            job,
            existing=rows.get(job.prefix_id),
            on_update=update,
            retry_errors=args.retry_errors,
        )
        with lock:
            rows[job.prefix_id] = result
        return result

    pending = [
        job
        for job in jobs
        if rows.get(job.prefix_id, {}).get("visual_csv_status") != "complete"
        and (
            args.retry_errors
            or rows.get(job.prefix_id, {}).get("visual_csv_status")
            != "complete_with_failures"
        )
    ]
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures = [pool.submit(execute, job) for job in pending]
        for future in as_completed(futures):
            future.result()
    _write_jsonl(args.output, (rows[prefix_id] for prefix_id in sorted(expected)))
    progress.write_text("", encoding="utf-8")
    return {
        "visual_csv_config_sha256": config.fingerprint(),
        "trajectories": len(trajectories),
        "prefixes": len(jobs),
        "complete": sum(
            row.get("visual_csv_status") == "complete" for row in rows.values()
        ),
        "complete_with_failures": sum(
            row.get("visual_csv_status") == "complete_with_failures"
            for row in rows.values()
        ),
        "retry_errors": args.retry_errors,
        "endpoint_count": len(clients),
        "output": str(args.output.resolve()),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trajectories", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--base-url", action="append", dest="base_urls")
    parser.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY", "no"))
    parser.add_argument("--model", default="Qwen3.5-9B")
    parser.add_argument("--judge-seed", type=int, action="append", default=None)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--local-media-paths", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--retry-errors", action="store_true")
    args = parser.parse_args(argv)
    args.judge_seed = args.judge_seed or [17, 42, 73]
    if args.concurrency <= 0:
        parser.error("--concurrency must be positive")
    try:
        summary = run(args)
    except (OSError, ValueError, TypeError, KeyError, RuntimeError) as error:
        print(json.dumps({"status": "failed", "error": str(error)}))
        return 1
    print(json.dumps({"status": "passed", **summary}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
