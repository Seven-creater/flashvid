from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from .baseline_diagnostics import (
    DEFAULT_DURATION_BUCKET_EDGES_S,
    VideoDiagnosticItem,
    assign_mismatched_videos,
)
from .datasets import VideoIndex
from .media import probe_video
from .schemas import Sample


def _canonical_sha256(payload: object) -> str:
    serialized = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_edges(edges: Sequence[float]) -> list[float | str]:
    return ["inf" if math.isinf(value) else float(value) for value in edges]


def _load_manifest(path: Path) -> list[Sample]:
    rows = [
        Sample(**json.loads(line))
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    ids = [sample.sample_id for sample in rows]
    if len(ids) != len(set(ids)):
        raise ValueError(f"manifest contains duplicate sample_id values: {path}")
    return rows


def build_mismatch_map_payload(
    *,
    dataset: str,
    manifest_path: Path,
    expected_manifest_sha256: str,
    video_root: Path,
    seed: int,
    experiment_config_sha256: str,
    probe: Callable[[Path], Mapping[str, Any]] = probe_video,
) -> dict[str, Any]:
    """Create an auditable same-dataset, same-duration-bucket wrong-video map."""

    manifest_path = manifest_path.resolve()
    video_root = video_root.resolve()
    actual_manifest_sha256 = _file_sha256(manifest_path)
    if actual_manifest_sha256.lower() != expected_manifest_sha256.lower():
        raise RuntimeError(
            f"{dataset} Dev manifest SHA-256 mismatch: expected "
            f"{expected_manifest_sha256}, got {actual_manifest_sha256}"
        )
    samples = _load_manifest(manifest_path)
    index = VideoIndex(video_root)
    durations: dict[str, float] = {}
    items: list[VideoDiagnosticItem] = []
    for sample in samples:
        if sample.dataset != dataset:
            raise ValueError(
                f"manifest sample {sample.sample_id} belongs to {sample.dataset}, "
                f"not {dataset}"
            )
        if sample.video not in durations:
            resolved = index.resolve(sample.video)
            metadata = probe(resolved)
            duration = float(metadata["duration"])
            if not math.isfinite(duration) or duration <= 0:
                raise ValueError(f"invalid duration for {sample.video}: {duration}")
            durations[sample.video] = duration
        items.append(
            VideoDiagnosticItem(
                dataset=sample.dataset,
                sample_id=sample.sample_id,
                video=sample.video,
                duration_s=durations[sample.video],
            )
        )

    assignments = assign_mismatched_videos(items, seed=seed)
    mapping = {
        assignment.sample_id: assignment.target_video for assignment in assignments
    }
    payload: dict[str, Any] = {
        "schema_version": 1,
        "dataset": dataset,
        "seed": int(seed),
        "manifest": str(manifest_path),
        "manifest_sha256": actual_manifest_sha256,
        "video_root": str(video_root),
        "experiment_config_sha256": experiment_config_sha256,
        "duration_bucket_edges_s": _json_edges(DEFAULT_DURATION_BUCKET_EDGES_S),
        "sample_count": len(samples),
        "mapped_sample_count": len(mapping),
        "unmapped_sample_ids": sorted(
            set(sample.sample_id for sample in samples) - set(mapping)
        ),
        "mapping": dict(sorted(mapping.items())),
        "assignments": [asdict(assignment) for assignment in assignments],
    }
    payload["payload_sha256"] = _canonical_sha256(payload)
    return payload


def write_frozen_json(path: Path, payload: Mapping[str, Any]) -> str:
    """Atomically create a frozen JSON artifact or verify an identical one."""

    serialized = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    if path.is_file():
        existing = path.read_text(encoding="utf-8")
        if json.loads(existing) != dict(payload):
            raise RuntimeError(f"refusing to overwrite different frozen artifact: {path}")
        return _file_sha256(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(serialized, encoding="utf-8")
    temporary.replace(path)
    return _file_sha256(path)


def freeze_mismatch_maps(
    config_path: Path,
    *,
    output_root: Path | None = None,
) -> dict[str, Any]:
    config_path = config_path.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config_sha256 = _canonical_sha256(config)
    seeds = tuple(int(seed) for seed in config["diagnostics"]["seeds"])
    if not seeds:
        raise ValueError("diagnostics.seeds must not be empty")
    root = (
        output_root.resolve()
        if output_root is not None
        else Path(config["result_root"]).resolve()
        / "frozen"
        / "mismatched_video_maps"
    )
    artifacts: list[dict[str, Any]] = []
    for dataset, dataset_config in config["datasets"].items():
        dev = dataset_config["dev"]
        for seed in seeds:
            payload = build_mismatch_map_payload(
                dataset=dataset,
                manifest_path=Path(dev["path"]),
                expected_manifest_sha256=str(dev["sha256"]),
                video_root=Path(dataset_config["video_root"]),
                seed=seed,
                experiment_config_sha256=config_sha256,
            )
            path = root / f"{dataset}_dev_seed{seed}.json"
            artifact_sha256 = write_frozen_json(path, payload)
            artifacts.append(
                {
                    "dataset": dataset,
                    "seed": seed,
                    "path": str(path),
                    "sha256": artifact_sha256,
                    "mapped_sample_count": payload["mapped_sample_count"],
                    "unmapped_sample_count": len(payload["unmapped_sample_ids"]),
                }
            )
    return {
        "schema_version": 1,
        "config": str(config_path),
        "config_sha256": config_sha256,
        "output_root": str(root),
        "artifacts": artifacts,
    }
