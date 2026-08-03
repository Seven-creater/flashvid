"""Utilities shared by FlashVID budget-aware perception backends."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import tempfile
from typing import Any, Mapping, Sequence
from urllib.parse import urlsplit


ALLOWED_RETENTION_RATIOS = (0.10, 0.25, 0.50, 1.00)
_ALLOWED_RATIO_DECIMALS = tuple(
    Decimal(str(value)) for value in ALLOWED_RETENTION_RATIOS
)


def _normalize_ratio(value: float | str | Decimal) -> float:
    if isinstance(value, bool):
        raise ValueError(f"unsupported retention ratio: {value!r}")
    try:
        ratio = Decimal(str(value))
    except (InvalidOperation, ValueError) as exc:
        raise ValueError(f"unsupported retention ratio: {value!r}") from exc
    if not ratio.is_finite() or ratio not in _ALLOWED_RATIO_DECIMALS:
        allowed = ", ".join(f"{item:.2f}" for item in ALLOWED_RETENTION_RATIOS)
        raise ValueError(
            f"unsupported retention ratio {value!r}; expected one of {allowed}"
        )
    return float(ratio)


@dataclass(frozen=True, slots=True)
class BudgetEndpoint:
    """One FlashVID perception service with a fixed retention ratio."""

    retention_ratio: float
    base_url: str
    model: str
    api_key: str = "no"
    max_concurrency: int = 8
    backend_revision: str = "flashvid_v1"

    def __post_init__(self) -> None:
        ratio = _normalize_ratio(self.retention_ratio)
        base_url = self.base_url.strip().rstrip("/")
        parsed = urlsplit(base_url)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError(
                f"base_url must be an absolute HTTP(S) URL: {self.base_url!r}"
            )
        model = self.model.strip()
        if not model:
            raise ValueError("model must not be empty")
        if (
            isinstance(self.max_concurrency, bool)
            or not isinstance(self.max_concurrency, int)
            or self.max_concurrency <= 0
        ):
            raise ValueError("max_concurrency must be a positive integer")
        if not isinstance(self.api_key, str):
            raise ValueError("api_key must be a string")
        backend_revision = self.backend_revision.strip()
        if not backend_revision:
            raise ValueError("backend_revision must not be empty")

        object.__setattr__(self, "retention_ratio", ratio)
        object.__setattr__(self, "base_url", base_url)
        object.__setattr__(self, "model", model)
        object.__setattr__(self, "backend_revision", backend_revision)


class BudgetEndpointPool:
    """Validated lookup table for the four fixed FlashVID budget services."""

    def __init__(self, endpoints: Sequence[BudgetEndpoint]):
        by_ratio: dict[float, BudgetEndpoint] = {}
        for endpoint in endpoints:
            ratio = _normalize_ratio(endpoint.retention_ratio)
            if ratio in by_ratio:
                raise ValueError(f"duplicate endpoint for retention ratio {ratio:.2f}")
            by_ratio[ratio] = endpoint

        expected = set(ALLOWED_RETENTION_RATIOS)
        actual = set(by_ratio)
        if actual != expected:
            missing = sorted(expected - actual)
            extra = sorted(actual - expected)
            details = []
            if missing:
                details.append(
                    "missing=" + ",".join(f"{value:.2f}" for value in missing)
                )
            if extra:
                details.append(
                    "extra=" + ",".join(f"{value:.2f}" for value in extra)
                )
            raise ValueError(
                "budget endpoint pool must define every allowed ratio"
                + (f" ({'; '.join(details)})" if details else "")
            )
        self._by_ratio = by_ratio

    @classmethod
    def load(cls, path: str | Path) -> "BudgetEndpointPool":
        """Load either a standalone ``endpoints`` or combined ``perception`` list."""

        config_path = Path(path)
        try:
            payload = json.loads(config_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid endpoint JSON in {config_path}: {exc}") from exc
        if not isinstance(payload, dict):
            raise ValueError("endpoint configuration must be a JSON object")

        keys = [key for key in ("endpoints", "perception") if key in payload]
        if len(keys) != 1:
            raise ValueError(
                "endpoint configuration must contain exactly one of "
                "'endpoints' or 'perception'"
            )
        records = payload[keys[0]]
        if not isinstance(records, list):
            raise ValueError(f"{keys[0]} must be a JSON list")

        allowed_fields = {
            "retention_ratio",
            "base_url",
            "model",
            "api_key",
            "max_concurrency",
            "backend_revision",
        }
        endpoints: list[BudgetEndpoint] = []
        for index, record in enumerate(records):
            if not isinstance(record, dict):
                raise ValueError(f"{keys[0]}[{index}] must be a JSON object")
            unknown = set(record) - allowed_fields
            if unknown:
                raise ValueError(
                    f"{keys[0]}[{index}] has unknown fields: {sorted(unknown)}"
                )
            missing = {"retention_ratio", "base_url", "model"} - set(record)
            if missing:
                raise ValueError(
                    f"{keys[0]}[{index}] is missing fields: {sorted(missing)}"
                )
            endpoints.append(BudgetEndpoint(**record))
        return cls(endpoints)

    @property
    def endpoints(self) -> tuple[BudgetEndpoint, ...]:
        return tuple(self._by_ratio[ratio] for ratio in ALLOWED_RETENTION_RATIOS)

    def choose(self, ratio: float | str | Decimal) -> BudgetEndpoint:
        return self._by_ratio[_normalize_ratio(ratio)]


def perception_cache_key(
    video: str | Path,
    start_time: float,
    end_time: float,
    nframes: int,
    resize: float,
    retention_ratio: float | str | Decimal,
    model: str,
    prompt_hash: str,
    evidence_request: str = "",
    backend_revision: str = "flashvid_v1",
) -> str:
    """Return a stable SHA-256 key for one perception observation."""

    start = _finite_float("start_time", start_time)
    end = _finite_float("end_time", end_time)
    if start < 0 or end < start:
        raise ValueError("time interval must satisfy 0 <= start_time <= end_time")
    if isinstance(nframes, bool) or not isinstance(nframes, int) or nframes <= 0:
        raise ValueError("nframes must be a positive integer")
    scale = _finite_float("resize", resize)
    if scale <= 0:
        raise ValueError("resize must be positive")
    if not model.strip():
        raise ValueError("model must not be empty")
    if not prompt_hash.strip():
        raise ValueError("prompt_hash must not be empty")
    if not backend_revision.strip():
        raise ValueError("backend_revision must not be empty")

    payload = {
        "video": Path(video).expanduser().resolve(strict=False).as_posix(),
        "interval": [_canonical_float(start), _canonical_float(end)],
        "nframes": nframes,
        "resize": _canonical_float(scale),
        "retention_ratio": f"{_normalize_ratio(retention_ratio):.2f}",
        "model": model.strip(),
        "backend_revision": backend_revision.strip(),
        "prompt_hash": prompt_hash.strip(),
        "evidence_request": evidence_request,
    }
    serialized = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def flashvid_token_counts(
    metadata: Mapping[str, float | int],
    nframes: int,
    resize: float,
    ratio: float | str | Decimal,
) -> tuple[int, int, float]:
    """Estimate raw and retained visual tokens like the deployed vLLM path.

    Spatial dimensions follow the existing Qwen visual-token estimate: resized
    height and width are rounded to a 28-pixel grid.  The deployed model floors
    the aggregate retention target and guarantees at least one frame's tokens.
    """

    if isinstance(nframes, bool) or not isinstance(nframes, int) or nframes <= 0:
        raise ValueError("nframes must be a positive integer")
    scale = _finite_float("resize", resize)
    if scale <= 0:
        raise ValueError("resize must be positive")
    height = _positive_dimension(metadata, "height")
    width = _positive_dimension(metadata, "width")
    normalized_ratio = _normalize_ratio(ratio)

    patch_height = max(1, round(height * scale / 28))
    patch_width = max(1, round(width * scale / 28))
    tokens_per_frame = patch_height * patch_width
    raw = nframes * tokens_per_frame
    retained = (
        raw
        if normalized_ratio == 1.0
        else max(tokens_per_frame, int(raw * normalized_ratio))
    )
    return raw, retained, retained / raw


def pack_frames_to_mp4(
    frames: Sequence[str | Path],
    timestamps: Sequence[float],
    output_mp4: str | Path,
    sidecar_path: str | Path,
) -> dict[str, Any]:
    """Pack ordered image files into an MP4 and atomically write its sidecar."""

    frame_paths = [Path(item).expanduser().resolve() for item in frames]
    if not frame_paths:
        raise ValueError("at least one frame is required")
    if len(frame_paths) != len(timestamps):
        raise ValueError("frames and timestamps must have equal length")
    missing = [path for path in frame_paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"frame does not exist: {missing[0]}")

    normalized_timestamps = [
        _finite_float(f"timestamps[{index}]", value)
        for index, value in enumerate(timestamps)
    ]
    if any(
        right < left
        for left, right in zip(
            normalized_timestamps, normalized_timestamps[1:]
        )
    ):
        raise ValueError("timestamps must be in nondecreasing order")

    output = Path(output_mp4).expanduser().resolve()
    sidecar = Path(sidecar_path).expanduser().resolve()
    if output == sidecar:
        raise ValueError("output_mp4 and sidecar_path must be different files")
    if output.suffix.lower() != ".mp4":
        raise ValueError("output_mp4 must use the .mp4 extension")
    output.parent.mkdir(parents=True, exist_ok=True)
    sidecar.parent.mkdir(parents=True, exist_ok=True)

    manifest_lines = ["ffconcat version 1.0"]
    for frame in frame_paths:
        manifest_lines.append(f"file '{_ffconcat_quote(frame)}'")
        manifest_lines.append("duration 1.0")
    manifest = "\n".join(manifest_lines) + "\n"

    temporary_output: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            prefix=f".{output.stem}.",
            suffix=".mp4",
            dir=output.parent,
            delete=False,
        ) as handle:
            temporary_output = Path(handle.name)
        command = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-protocol_whitelist",
            "file,pipe",
            "-i",
            "pipe:0",
            "-an",
            "-vf",
            "pad=ceil(iw/2)*2:ceil(ih/2)*2,format=yuv420p",
            "-r",
            "1",
            "-frames:v",
            str(len(frame_paths)),
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-movflags",
            "+faststart",
            str(temporary_output),
        ]
        try:
            subprocess.run(
                command,
                input=manifest,
                text=True,
                capture_output=True,
                check=True,
            )
        except FileNotFoundError as exc:
            raise RuntimeError("ffmpeg executable was not found") from exc
        except subprocess.CalledProcessError as exc:
            detail = (exc.stderr or exc.stdout or "").strip()
            raise RuntimeError(f"ffmpeg failed to pack frames: {detail[:1000]}") from exc
        if not temporary_output.is_file() or temporary_output.stat().st_size == 0:
            raise RuntimeError("ffmpeg produced an empty MP4")
        os.replace(temporary_output, output)
        temporary_output = None
    finally:
        if temporary_output is not None:
            temporary_output.unlink(missing_ok=True)

    payload: dict[str, Any] = {
        "version": 1,
        "video": output.as_posix(),
        "frame_count": len(frame_paths),
        "frames": [
            {
                "index": index,
                "timestamp_s": timestamp,
                "source_path": frame.as_posix(),
            }
            for index, (frame, timestamp) in enumerate(
                zip(frame_paths, normalized_timestamps)
            )
        ],
    }
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        prefix=f".{sidecar.name}.",
        suffix=".tmp",
        dir=sidecar.parent,
        delete=False,
    ) as handle:
        temporary_sidecar = Path(handle.name)
        handle.write(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    try:
        os.replace(temporary_sidecar, sidecar)
    finally:
        temporary_sidecar.unlink(missing_ok=True)
    return payload


def _finite_float(name: str, value: float) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be finite")
    try:
        converted = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be finite") from exc
    if not math.isfinite(converted):
        raise ValueError(f"{name} must be finite")
    return converted


def _positive_dimension(
    metadata: Mapping[str, float | int], key: str
) -> float:
    if key not in metadata:
        raise ValueError(f"metadata is missing {key!r}")
    value = _finite_float(f"metadata[{key!r}]", metadata[key])
    if value <= 0:
        raise ValueError(f"metadata[{key!r}] must be positive")
    return value


def _canonical_float(value: float) -> str:
    return format(value, ".12g")


def _ffconcat_quote(path: Path) -> str:
    value = path.as_posix()
    if "\n" in value or "\r" in value:
        raise ValueError("frame paths must not contain newlines")
    return value.replace("'", "'\\''")
