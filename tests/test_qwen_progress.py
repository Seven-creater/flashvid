from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from flashvid_eval.qwen_progress import HEARTBEAT_ENV, emit_progress


def test_progress_is_disabled_without_launcher_environment(monkeypatch) -> None:
    monkeypatch.delenv(HEARTBEAT_ENV, raising=False)
    assert emit_progress("ignored") is False


def test_concurrent_progress_updates_leave_one_atomic_json_file(
    tmp_path: Path, monkeypatch
) -> None:
    heartbeat = tmp_path / "nested" / "phase.heartbeat.json"
    monkeypatch.setenv(HEARTBEAT_ENV, str(heartbeat))

    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(lambda index: emit_progress("step", index=index), range(64)))

    assert all(results)
    payload = json.loads(heartbeat.read_text(encoding="utf-8"))
    assert payload["event"] == "step"
    assert payload["index"] in range(64)
    assert payload["sequence"] >= 1
    assert not list(heartbeat.parent.glob("*.partial"))
