from __future__ import annotations

import json
import subprocess
from pathlib import Path


def probe_video(path: Path) -> dict[str, float | int]:
    command = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration:stream=width,height",
        "-of",
        "json",
        str(path),
    ]
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    payload = json.loads(result.stdout)
    stream = next((item for item in payload.get("streams", []) if item.get("width")), None)
    if stream is None:
        raise RuntimeError(f"ffprobe returned no video stream for {path}")
    return {
        "duration": float(payload.get("format", {}).get("duration") or 0.0),
        "width": int(stream["width"]),
        "height": int(stream["height"]),
    }


def estimate_visual_tokens(metadata: dict[str, float | int], nframes: int, resize: float) -> int:
    h = max(1, round(float(metadata["height"]) * resize / 28))
    w = max(1, round(float(metadata["width"]) * resize / 28))
    return int(nframes * h * w)
