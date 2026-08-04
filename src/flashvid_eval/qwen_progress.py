from __future__ import annotations

import itertools
import json
import os
import threading
import time
from pathlib import Path
from typing import Any


HEARTBEAT_ENV = "QWEN_PROGRESS_HEARTBEAT"

_WRITE_LOCK = threading.Lock()
_SEQUENCE = itertools.count(1)


def emit_progress(event: str, **details: Any) -> bool:
    """Atomically publish internal Qwen progress when a launcher requests it.

    The heartbeat is deliberately best-effort and external to result files. It
    must never change inference semantics or turn a filesystem issue into an
    experiment failure.
    """

    raw_path = os.environ.get(HEARTBEAT_ENV, "").strip()
    if not raw_path:
        return False
    path = Path(raw_path)
    payload = {
        "event": str(event),
        "pid": os.getpid(),
        "thread_id": threading.get_ident(),
        "sequence": next(_SEQUENCE),
        "time_ns": time.time_ns(),
        **details,
    }
    try:
        with _WRITE_LOCK:
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_name(f".{path.name}.{os.getpid()}.partial")
            temporary.write_text(
                json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str)
                + "\n",
                encoding="utf-8",
            )
            os.replace(temporary, path)
    except OSError:
        return False
    return True
