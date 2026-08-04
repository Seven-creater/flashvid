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
