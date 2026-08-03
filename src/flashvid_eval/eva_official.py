"""Small adapter around the vendored official EVA frame-selection tool.

The model/service remains Qwen3.5-9B.  This module only reuses the official
EVA frame decoding, timestamp sampling, clamping, resize-rounding and image
serialization implementation; it does not load EVA weights.
"""

from __future__ import annotations

import hashlib
import importlib.util
import os
import subprocess
from pathlib import Path
from typing import Any


def _tool_path() -> Path:
    configured = os.environ.get("FLASHVID_EVA_TOOL_PATH")
    if configured:
        return Path(configured).expanduser().resolve()
    return (
        Path(__file__).resolve().parents[2]
        / "third_party"
        / "EfficientVideoAgent"
        / "select_frame_fallback.py"
    )


def frame_tool_identity() -> dict[str, str | None]:
    """Return the exact official frame-tool artifact used by this process."""

    path = _tool_path()
    if not path.is_file():
        raise FileNotFoundError(f"official EVA frame tool is missing: {path}")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    commit: str | None = None
    try:
        completed = subprocess.run(
            ["git", "-C", str(path.parent), "rev-parse", "HEAD"],
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
        candidate = completed.stdout.strip().lower()
        if completed.returncode == 0 and len(candidate) == 40 and all(
            character in "0123456789abcdef" for character in candidate
        ):
            commit = candidate
    except (OSError, subprocess.SubprocessError):
        pass
    return {
        "path": str(path),
        "sha256": digest,
        "git_commit": commit,
    }


def _load_tool() -> Any:
    path = _tool_path()
    if not path.is_file():
        raise FileNotFoundError(
            f"official EVA frame tool is missing: {path}. "
            "Vendor EfficientVideoAgent/select_frame_fallback.py first."
        )
    spec = importlib.util.spec_from_file_location("flashvid_official_eva_frame_tool", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load official EVA frame tool: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def select_frames(
    video: Path,
    start_time: float,
    end_time: float,
    nframes: int,
    resize: float,
    output_dir: Path,
) -> tuple[list[Path], list[float], str]:
    """Run the official EVA fallback selector and save its frames.

    ``factor=28`` and ``clamp_to_stream=True`` match the official
    ``eval-eva.py`` fallback path.  The returned timestamps are the actual
    sampled timestamps, rather than estimates based on the requested range.
    """

    tool = _load_tool()
    frames, timestamps, backend = tool.extract_frames(
        video_path=str(video),
        start_time=float(start_time),
        end_time=float(end_time),
        nframes=int(nframes),
        resize=float(resize),
        factor=28,
        backend="auto",
        clamp_to_stream=True,
    )
    _, saved_count, names = tool._save_frames(
        frames=frames,
        video_path=str(video),
        save_root=str(output_dir.parent),
        save_dir=str(output_dir),
    )
    if saved_count <= 0 or not names:
        raise RuntimeError(f"official EVA frame tool returned no frames for {video}")
    return [Path(name).resolve() for name in names], [float(ts) for ts in timestamps], str(backend)
