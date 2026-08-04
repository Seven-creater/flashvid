#!/usr/bin/env python3
"""Run one experiment command and stop it after a bounded progress stall.

Progress is defined as a new or changed result file under ``--watch-root`` or
an atomic update to the optional ``--heartbeat-file``.
The wrapper is intentionally generic and never retries a stalled command: the
phase launcher can be invoked again with ``--resume`` after the cause is
inspected or the protocol is changed.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Sequence


STALL_EXIT_CODE = 124


def progress_snapshot(root: Path, pattern: str) -> tuple[tuple[str, int, int], ...]:
    if not root.exists():
        return ()
    rows: list[tuple[str, int, int]] = []
    for path in root.rglob(pattern):
        if not path.is_file():
            continue
        stat = path.stat()
        rows.append((str(path.resolve()), stat.st_size, stat.st_mtime_ns))
    return tuple(sorted(rows))


def heartbeat_snapshot(path: Path | None) -> tuple[int, int, int] | None:
    if path is None or not path.is_file():
        return None
    stat = path.stat()
    return (stat.st_size, stat.st_mtime_ns, stat.st_ino)


def _terminate_group(process: subprocess.Popen[bytes], grace_seconds: float) -> None:
    if process.poll() is not None:
        return
    if os.name == "posix":
        os.killpg(process.pid, signal.SIGTERM)
    else:  # pragma: no cover - production launcher is Linux
        process.terminate()
    try:
        process.wait(timeout=grace_seconds)
        return
    except subprocess.TimeoutExpired:
        pass
    if os.name == "posix":
        os.killpg(process.pid, signal.SIGKILL)
    else:  # pragma: no cover - production launcher is Linux
        process.kill()
    process.wait()


def run_with_watchdog(
    command: Sequence[str],
    *,
    watch_root: Path,
    pattern: str,
    stall_seconds: float,
    poll_seconds: float,
    terminate_grace_seconds: float,
    heartbeat_file: Path | None = None,
) -> int:
    if not command:
        raise ValueError("command cannot be empty")
    if stall_seconds <= 0 or poll_seconds <= 0 or terminate_grace_seconds <= 0:
        raise ValueError("watchdog durations must be positive")
    before = progress_snapshot(watch_root, pattern)
    heartbeat_before = heartbeat_snapshot(heartbeat_file)
    process = subprocess.Popen(list(command), start_new_session=(os.name == "posix"))
    last_snapshot = before
    last_heartbeat = heartbeat_before
    last_progress = time.monotonic()
    try:
        while process.poll() is None:
            time.sleep(poll_seconds)
            current = progress_snapshot(watch_root, pattern)
            current_heartbeat = heartbeat_snapshot(heartbeat_file)
            if current != last_snapshot or current_heartbeat != last_heartbeat:
                last_snapshot = current
                last_heartbeat = current_heartbeat
                last_progress = time.monotonic()
                continue
            stalled_for = time.monotonic() - last_progress
            if stalled_for < stall_seconds:
                continue
            print(
                json.dumps(
                    {
                        "event": "progress_stall",
                        "stall_seconds": round(stalled_for, 3),
                        "watch_root": str(watch_root.resolve()),
                        "pattern": pattern,
                        "heartbeat_file": (
                            str(heartbeat_file.resolve())
                            if heartbeat_file is not None
                            else None
                        ),
                        "child_pid": process.pid,
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
            _terminate_group(process, terminate_grace_seconds)
            return STALL_EXIT_CODE
        return int(process.returncode or 0)
    except KeyboardInterrupt:
        _terminate_group(process, terminate_grace_seconds)
        return 130


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Stop a resumable experiment if no result file changes in time."
    )
    parser.add_argument("--watch-root", type=Path, required=True)
    parser.add_argument("--pattern", default="*.jsonl")
    parser.add_argument("--heartbeat-file", type=Path)
    parser.add_argument("--stall-seconds", type=float, default=90.0)
    parser.add_argument("--poll-seconds", type=float, default=5.0)
    parser.add_argument("--terminate-grace-seconds", type=float, default=15.0)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    command = list(args.command)
    if command and command[0] == "--":
        command = command[1:]
    try:
        return run_with_watchdog(
            command,
            watch_root=args.watch_root,
            pattern=args.pattern,
            stall_seconds=args.stall_seconds,
            poll_seconds=args.poll_seconds,
            terminate_grace_seconds=args.terminate_grace_seconds,
            heartbeat_file=args.heartbeat_file,
        )
    except (OSError, ValueError) as error:
        print(json.dumps({"event": "watchdog_error", "error": str(error)}), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
