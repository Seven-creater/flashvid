from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import pytest


SCRIPT = Path(__file__).parents[1] / "scripts" / "run_with_progress_watchdog.py"
SPEC = importlib.util.spec_from_file_location("run_with_progress_watchdog", SCRIPT)
assert SPEC and SPEC.loader
module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)


def test_progress_snapshot_changes_only_for_matching_result_files(tmp_path: Path) -> None:
    assert module.progress_snapshot(tmp_path, "*.jsonl") == ()
    (tmp_path / "ignored.log").write_text("x", encoding="utf-8")
    assert module.progress_snapshot(tmp_path, "*.jsonl") == ()
    result = tmp_path / "nested" / "rows.jsonl"
    result.parent.mkdir()
    result.write_text("{}\n", encoding="utf-8")
    first = module.progress_snapshot(tmp_path, "*.jsonl")
    result.write_text("{}\n{}\n", encoding="utf-8")
    assert module.progress_snapshot(tmp_path, "*.jsonl") != first


def test_heartbeat_snapshot_changes_after_atomic_replace(tmp_path: Path) -> None:
    heartbeat = tmp_path / "phase.heartbeat.json"
    assert module.heartbeat_snapshot(heartbeat) is None
    heartbeat.write_text('{"sequence":1}\n', encoding="utf-8")
    first = module.heartbeat_snapshot(heartbeat)
    replacement = heartbeat.with_suffix(".partial")
    replacement.write_text('{"sequence":2}\n', encoding="utf-8")
    replacement.replace(heartbeat)
    assert module.heartbeat_snapshot(heartbeat) != first


@pytest.mark.skipif(os.name != "posix", reason="production watchdog uses POSIX process groups")
def test_watchdog_terminates_a_stalled_child(tmp_path: Path) -> None:
    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--watch-root",
            str(tmp_path),
            "--stall-seconds",
            "0.15",
            "--poll-seconds",
            "0.02",
            "--terminate-grace-seconds",
            "0.1",
            "--",
            sys.executable,
            "-c",
            "import time; time.sleep(10)",
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=3,
    )
    assert completed.returncode == module.STALL_EXIT_CODE
    assert '"event": "progress_stall"' in completed.stdout


@pytest.mark.skipif(os.name != "posix", reason="production watchdog uses POSIX process groups")
def test_heartbeat_keeps_a_multi_step_child_alive(tmp_path: Path) -> None:
    heartbeat = tmp_path / "phase.heartbeat.json"
    child = (
        "import pathlib,time\n"
        f"p=pathlib.Path({str(heartbeat)!r})\n"
        "for i in range(8):\n"
        " p.write_text(str(i), encoding='utf-8')\n"
        " time.sleep(0.04)\n"
    )
    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--watch-root",
            str(tmp_path / "results"),
            "--heartbeat-file",
            str(heartbeat),
            "--stall-seconds",
            "0.12",
            "--poll-seconds",
            "0.02",
            "--terminate-grace-seconds",
            "0.1",
            "--",
            sys.executable,
            "-c",
            child,
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=3,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
