#!/usr/bin/env python3
"""Run the post-base Perception-Memory repair lane entirely on the server.

The caller must launch this script from a clean, pre-synchronised Git checkout.
It never pulls or changes code.  It waits for the named base replay workers to
exit, freezes the complete 2,917-row scope, repairs both failure lanes, and
materialises a new success-only merged artifact.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from flashvid_eval.perception_memory_repair import (
    merge_replay_results,
    prepare_repair_scope,
)
from flashvid_eval.perception_memory_replay import file_sha256


PIPELINE_VERSION = "perception_memory_post_base_repair_v1"
SOURCE_ROWS = 2917
EXPLICIT_TIME_ROWS = 37
EXPLICIT_TIME_SAMPLES = 6


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False
    ) as handle:
        temporary = Path(handle.name)
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid JSON: {path}") from error
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def _read_jsonl_count(path: Path) -> int:
    if not path.is_file():
        raise FileNotFoundError(path)
    count = 0
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"{path}:{line_number}: invalid JSON") from error
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: row must be an object")
            count += 1
    return count


def _file_records(paths: Sequence[Path]) -> list[dict[str, Any]]:
    return [
        {
            "path": str(path.resolve()),
            "sha256": file_sha256(path),
            "rows": _read_jsonl_count(path),
        }
        for path in paths
    ]


def _validate_prepared(
    prepared: Path, source_paths: Sequence[Path], base_paths: Sequence[Path]
) -> dict[str, Any]:
    scope_path = prepared / "frozen_scope.json"
    scope = _read_json(scope_path)
    if scope.get("expected_rows") != SOURCE_ROWS:
        raise RuntimeError("prepared scope is not the registered 2917-row run")
    if (
        scope.get("expected_explicit_time_rows") != EXPLICIT_TIME_ROWS
        or scope.get("expected_explicit_time_samples") != EXPLICIT_TIME_SAMPLES
    ):
        raise RuntimeError(
            "prepared explicit-time scope differs from 37 rows/6 samples"
        )
    if (scope.get("source") or {}).get("files") != _file_records(source_paths):
        raise RuntimeError("prepared source shard lineage changed")
    if (scope.get("base") or {}).get("files") != _file_records(base_paths):
        raise RuntimeError("prepared base shard lineage changed")
    artifacts = scope.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise RuntimeError("prepared scope has no artifact records")
    for name in ("cached_repair_input.jsonl", "explicit_time_mismatch.jsonl"):
        path = prepared / name
        record = artifacts.get(name)
        if not isinstance(record, Mapping):
            raise RuntimeError(f"prepared scope lacks {name}")
        if _read_jsonl_count(path) != int(record.get("rows", -1)):
            raise RuntimeError(f"prepared artifact row count changed: {name}")
        if file_sha256(path) != record.get("sha256"):
            raise RuntimeError(f"prepared artifact hash changed: {name}")
    if int(artifacts["explicit_time_mismatch.jsonl"]["rows"]) != (EXPLICIT_TIME_ROWS):
        raise RuntimeError("prepared explicit-time artifact must contain 37 rows")
    return scope


def _validate_completed_output(
    path: Path, *, expected_rows: int, expected_sha256: str | None = None
) -> str:
    if _read_jsonl_count(path) != expected_rows:
        raise RuntimeError(f"completed output row count changed: {path}")
    digest = file_sha256(path)
    if expected_sha256 is not None and digest != expected_sha256:
        raise RuntimeError(f"completed output hash changed: {path}")
    return digest


def _validate_merged(merged: Path) -> dict[str, Any]:
    summary = _read_json(merged / "merge_summary.json")
    if (
        summary.get("source_rows") != SOURCE_ROWS
        or summary.get("prefix_bind_passed") is not True
    ):
        raise RuntimeError("merged output did not pass the registered scope/bind gate")
    artifacts = summary.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise RuntimeError("merged summary lacks artifact hashes")
    for name in ("merged_success.jsonl", "double_failures.jsonl"):
        record = artifacts.get(name)
        if not isinstance(record, Mapping):
            raise RuntimeError(f"merged summary lacks {name}")
        _validate_completed_output(
            merged / name,
            expected_rows=int(record.get("rows", -1)),
            expected_sha256=str(record.get("sha256") or ""),
        )
    return summary


def _check_endpoints(base_urls: Sequence[str], timeout: float = 10.0) -> None:
    for base_url in base_urls:
        url = base_url.rstrip("/") + "/models"
        try:
            with urllib.request.urlopen(url, timeout=timeout) as response:
                if response.status != 200:
                    raise RuntimeError(
                        f"endpoint returned HTTP {response.status}: {url}"
                    )
        except OSError as error:
            raise RuntimeError(f"model endpoint is unavailable: {url}") from error


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _log(run_root: Path, message: str) -> None:
    line = f"{_now()} {message}\n"
    with (run_root / "pipeline.log").open("a", encoding="utf-8") as handle:
        handle.write(line)
        handle.flush()
        os.fsync(handle.fileno())


def _git_output(repo_root: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    return result.stdout.strip()


def _verify_git(repo_root: Path, expected_branch: str, expected_head: str) -> None:
    head = _git_output(repo_root, "rev-parse", "HEAD")
    branch = _git_output(repo_root, "branch", "--show-current")
    dirty = _git_output(repo_root, "status", "--porcelain")
    if head != expected_head:
        raise RuntimeError(f"Git HEAD mismatch: expected {expected_head}, found {head}")
    if branch != expected_branch:
        raise RuntimeError(
            f"Git branch mismatch: expected {expected_branch}, found {branch}"
        )
    if dirty:
        raise RuntimeError("repair checkout is dirty; use a clean immutable worktree")


def _matching_worker_pids(required_fragments: Sequence[str]) -> list[int]:
    """Return Linux processes whose command contains every registered fragment."""

    if not required_fragments or any(not item.strip() for item in required_fragments):
        raise ValueError("base worker command fragments must be non-empty")
    proc = Path("/proc")
    if not proc.is_dir():
        raise RuntimeError("base-worker waiting requires Linux /proc")
    own_pid = os.getpid()
    matches: list[int] = []
    for entry in proc.iterdir():
        if not entry.name.isdigit() or int(entry.name) == own_pid:
            continue
        try:
            command = (
                (entry / "cmdline")
                .read_bytes()
                .replace(b"\0", b" ")
                .decode("utf-8", errors="replace")
            )
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        if command and all(fragment in command for fragment in required_fragments):
            matches.append(int(entry.name))
    return sorted(matches)


def _wait_for_base_workers(
    *,
    fragments: Sequence[str],
    poll_seconds: float,
    timeout_seconds: float,
    on_poll: Callable[[list[int]], None],
) -> None:
    if poll_seconds <= 0 or timeout_seconds <= 0:
        raise ValueError("worker wait intervals must be positive")
    deadline = time.monotonic() + timeout_seconds
    while True:
        pids = _matching_worker_pids(fragments)
        on_poll(pids)
        if not pids:
            return
        if time.monotonic() >= deadline:
            raise TimeoutError(f"base replay workers did not exit: {pids}")
        time.sleep(min(poll_seconds, max(0.0, deadline - time.monotonic())))


def _run_command(command: Sequence[str], *, cwd: Path, log_path: Path) -> int:
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(f"{_now()} COMMAND {json.dumps(list(command))}\n")
        handle.flush()
        result = subprocess.run(
            list(command),
            cwd=cwd,
            stdout=handle,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
        handle.write(f"{_now()} EXIT {result.returncode}\n")
        handle.flush()
        os.fsync(handle.fileno())
    return int(result.returncode)


def _cached_command(
    args: argparse.Namespace, prepared: Path, output: Path
) -> list[str]:
    command = [
        sys.executable,
        str(args.repo_root / "scripts/replay_perception_memory_trajectories.py"),
        "--input",
        str(prepared / "cached_repair_input.jsonl"),
        "--output",
        str(output),
        "--audit-summary",
        str(args.audit_summary),
        "--model",
        args.model,
        "--seed",
        str(args.seed),
        "--max-tokens",
        str(args.max_tokens),
        "--max-frames-per-call",
        str(args.max_frames_per_call),
        "--timeout",
        str(args.request_timeout),
        "--concurrency",
        "8",
        "--local-media-paths",
        "--resume",
    ]
    for endpoint in args.base_url:
        command.extend(["--base-url", endpoint])
    return command


def _rescue_command(
    args: argparse.Namespace, prepared: Path, output: Path
) -> list[str]:
    command = [
        sys.executable,
        str(args.repo_root / "scripts/run_perception_memory_explicit_time_rescue.py"),
        "--repair-manifest",
        str(prepared / "explicit_time_mismatch.jsonl"),
        "--frozen-scope",
        str(prepared / "frozen_scope.json"),
        "--output",
        str(output),
        "--video-root",
        str(args.video_root),
        "--frame-root",
        str(args.frame_root),
        "--model",
        args.model,
        "--seed",
        str(args.seed),
        "--max-turns",
        str(args.max_turns),
        "--timeout",
        str(args.request_timeout),
        "--concurrency",
        "8",
        "--expected-rows",
        str(EXPLICIT_TIME_ROWS),
        "--expected-samples",
        str(EXPLICIT_TIME_SAMPLES),
        "--local-media-paths",
        "--resume",
    ]
    for endpoint in args.base_url:
        command.extend(["--base-url", endpoint])
    return command


def _fingerprint(args: argparse.Namespace) -> str:
    return _canonical_sha256(
        {
            "pipeline_version": PIPELINE_VERSION,
            "repo_root": str(args.repo_root.resolve()),
            "expected_git_branch": args.expected_git_branch,
            "expected_git_head": args.expected_git_head,
            "source_shards": [str(path.resolve()) for path in args.source],
            "base_shards": [str(path.resolve()) for path in args.base_results],
            "audit_summary": str(args.audit_summary.resolve()),
            "video_root": str(args.video_root.resolve()),
            "frame_root": str(args.frame_root.resolve()),
            "base_urls": list(args.base_url),
            "model": args.model,
            "seed": args.seed,
            "max_tokens": args.max_tokens,
            "max_frames_per_call": args.max_frames_per_call,
            "max_turns": args.max_turns,
            "request_timeout": args.request_timeout,
            "worker_match": list(args.base_worker_match),
            "registered_scope": {
                "source_rows": SOURCE_ROWS,
                "explicit_time_rows": EXPLICIT_TIME_ROWS,
                "explicit_time_samples": EXPLICIT_TIME_SAMPLES,
            },
        }
    )


def _initial_state(args: argparse.Namespace, fingerprint: str) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "pipeline_version": PIPELINE_VERSION,
        "config_sha256": fingerprint,
        "status": "running",
        "current_stage": "wait_base",
        "created_at": _now(),
        "updated_at": _now(),
        "expected_git_branch": args.expected_git_branch,
        "expected_git_head": args.expected_git_head,
        "stages": {},
        "error": None,
    }


def _save_state(path: Path, state: dict[str, Any]) -> None:
    state["updated_at"] = _now()
    _write_json_atomic(path, state)


def _stage_done(state: Mapping[str, Any], name: str) -> bool:
    stage = (state.get("stages") or {}).get(name)
    return isinstance(stage, Mapping) and stage.get("status") == "passed"


def run_pipeline(args: argparse.Namespace) -> dict[str, Any]:
    if len(args.source) != 8 or len(args.base_results) != 8:
        raise ValueError("repair pipeline requires exactly 8 source and 8 base shards")
    if len(args.base_url) != 8 or len(set(args.base_url)) != 8:
        raise ValueError("repair pipeline requires exactly 8 unique model endpoints")
    args.repo_root = args.repo_root.resolve()
    run_root = args.run_root.resolve()
    state_path = run_root / "status.json"
    fingerprint = _fingerprint(args)
    if run_root.exists():
        if not args.resume:
            raise FileExistsError("run root exists; pass --resume")
        state = _read_json(state_path)
        if state.get("config_sha256") != fingerprint:
            raise RuntimeError("resume configuration fingerprint mismatch")
    else:
        run_root.mkdir(parents=True)
        state = _initial_state(args, fingerprint)
        _save_state(state_path, state)

    prepared = run_root / "repair_scope"
    cached_output = run_root / "cached_repair_results.jsonl"
    rescue_output = run_root / "explicit_time_rescue_results.jsonl"
    merged = run_root / "merged"
    try:
        _verify_git(args.repo_root, args.expected_git_branch, args.expected_git_head)
        if not _stage_done(state, "wait_base"):
            state["current_stage"] = "wait_base"

            def on_poll(pids: list[int]) -> None:
                state["active_base_worker_pids"] = pids
                _save_state(state_path, state)

            _wait_for_base_workers(
                fragments=args.base_worker_match,
                poll_seconds=args.poll_seconds,
                timeout_seconds=args.wait_timeout,
                on_poll=on_poll,
            )
            state["stages"]["wait_base"] = {"status": "passed", "ended_at": _now()}
            _save_state(state_path, state)
            _log(run_root, "all registered base replay workers exited")

        _verify_git(args.repo_root, args.expected_git_branch, args.expected_git_head)
        if not _stage_done(state, "prepare"):
            state["current_stage"] = "prepare"
            _save_state(state_path, state)
            if prepared.exists():
                _validate_prepared(prepared, args.source, args.base_results)
                summary = {"status": "adopted_atomic_prepare"}
            else:
                summary = prepare_repair_scope(
                    source_paths=args.source,
                    base_results_paths=args.base_results,
                    output_dir=prepared,
                    expected_rows=SOURCE_ROWS,
                    expected_explicit_time_rows=EXPLICIT_TIME_ROWS,
                    expected_explicit_time_samples=EXPLICIT_TIME_SAMPLES,
                )
            state["stages"]["prepare"] = {
                "status": "passed",
                "ended_at": _now(),
                "summary": summary,
                "frozen_scope_sha256": file_sha256(prepared / "frozen_scope.json"),
            }
            _save_state(state_path, state)
        _validate_prepared(prepared, args.source, args.base_results)

        if not _stage_done(state, "cached_repair"):
            state["current_stage"] = "cached_repair"
            _save_state(state_path, state)
            _check_endpoints(args.base_url)
            code = _run_command(
                _cached_command(args, prepared, cached_output),
                cwd=args.repo_root,
                log_path=run_root / "cached_repair.log",
            )
            expected = _read_jsonl_count(prepared / "cached_repair_input.jsonl")
            completed = _read_jsonl_count(cached_output)
            if completed != expected:
                raise RuntimeError(
                    f"cached repair incomplete: expected {expected}, found {completed}, exit={code}"
                )
            state["stages"]["cached_repair"] = {
                "status": "passed",
                "ended_at": _now(),
                "command_exit_code": code,
                "rows": completed,
                "output_sha256": file_sha256(cached_output),
            }
            _save_state(state_path, state)
        else:
            expected = _read_jsonl_count(prepared / "cached_repair_input.jsonl")
            _validate_completed_output(
                cached_output,
                expected_rows=expected,
                expected_sha256=state["stages"]["cached_repair"].get("output_sha256"),
            )

        if not _stage_done(state, "explicit_time_rescue"):
            state["current_stage"] = "explicit_time_rescue"
            _save_state(state_path, state)
            _check_endpoints(args.base_url)
            code = _run_command(
                _rescue_command(args, prepared, rescue_output),
                cwd=args.repo_root,
                log_path=run_root / "explicit_time_rescue.log",
            )
            completed = _read_jsonl_count(rescue_output)
            if completed != EXPLICIT_TIME_ROWS:
                raise RuntimeError(
                    "explicit-time rescue incomplete: "
                    f"expected {EXPLICIT_TIME_ROWS}, found {completed}, exit={code}"
                )
            state["stages"]["explicit_time_rescue"] = {
                "status": "passed",
                "ended_at": _now(),
                "command_exit_code": code,
                "rows": completed,
                "output_sha256": file_sha256(rescue_output),
            }
            _save_state(state_path, state)
        else:
            _validate_completed_output(
                rescue_output,
                expected_rows=EXPLICIT_TIME_ROWS,
                expected_sha256=state["stages"]["explicit_time_rescue"].get(
                    "output_sha256"
                ),
            )

        _verify_git(args.repo_root, args.expected_git_branch, args.expected_git_head)
        if not _stage_done(state, "merge"):
            state["current_stage"] = "merge"
            _save_state(state_path, state)
            if merged.exists():
                summary = _validate_merged(merged)
            else:
                summary = merge_replay_results(
                    source_paths=args.source,
                    base_results_paths=args.base_results,
                    frozen_scope_path=prepared / "frozen_scope.json",
                    replacement_result_paths=[cached_output, rescue_output],
                    output_dir=merged,
                )
            state["stages"]["merge"] = {
                "status": "passed",
                "ended_at": _now(),
                "summary": summary,
                "merged_success_sha256": file_sha256(merged / "merged_success.jsonl"),
            }
            _save_state(state_path, state)
        _validate_merged(merged)

        state["status"] = "passed"
        state["current_stage"] = "complete"
        state["error"] = None
        _save_state(state_path, state)
        _log(run_root, "post-base repair pipeline complete")
        return state
    except BaseException as error:
        state["status"] = "failed"
        state["error"] = f"{type(error).__name__}: {error}"
        _save_state(state_path, state)
        _log(run_root, f"FAILED {state['current_stage']}: {state['error']}")
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--expected-git-branch", required=True)
    parser.add_argument("--expected-git-head", required=True)
    parser.add_argument("--source", type=Path, nargs="+", required=True)
    parser.add_argument("--base-results", type=Path, nargs="+", required=True)
    parser.add_argument("--audit-summary", type=Path, required=True)
    parser.add_argument("--video-root", type=Path, required=True)
    parser.add_argument("--frame-root", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--base-url", action="append", required=True)
    parser.add_argument("--base-worker-match", action="append", required=True)
    parser.add_argument("--model", default="Qwen3.5-9B")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--max-frames-per-call", type=int, default=128)
    parser.add_argument("--max-turns", type=int, default=6)
    parser.add_argument("--request-timeout", type=float, default=300.0)
    parser.add_argument("--poll-seconds", type=float, default=30.0)
    parser.add_argument("--wait-timeout", type=float, default=43200.0)
    parser.add_argument("--resume", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        state = run_pipeline(args)
    except BaseException as error:
        print(
            json.dumps(
                {"status": "failed", "error": f"{type(error).__name__}: {error}"},
                ensure_ascii=False,
            )
        )
        return 1
    print(json.dumps(state, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
