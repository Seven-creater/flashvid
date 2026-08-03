#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PureWindowsPath
import tempfile
from typing import Any, Mapping


DATASETS = ("lvbench", "lsdbench", "cgbench")
DEFAULT_SPLIT_COUNTS = {"train": 200, "dev": 50, "final": 100}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    serialized = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: row must be an object")
            rows.append(value)
    return rows


def _video_id(value: Any) -> str:
    video = str(value or "").strip().replace("\\", "/")
    if not video:
        raise ValueError("manifest row has no video")
    while video.startswith("./"):
        video = video[2:]
    return video


def _is_absolute_video(video: str) -> bool:
    return Path(video).is_absolute() or PureWindowsPath(video).is_absolute()


def _validate_rows(
    rows: list[dict[str, Any]],
    *,
    dataset: str,
    split: str,
    expected_count: int,
) -> tuple[set[str], set[str]]:
    if len(rows) != expected_count:
        raise ValueError(
            f"{dataset} {split} expected {expected_count} rows, found {len(rows)}"
        )
    sample_ids: set[str] = set()
    videos: set[str] = set()
    for index, row in enumerate(rows):
        if str(row.get("dataset") or "").strip().lower() != dataset:
            raise ValueError(f"{dataset} {split} row {index} has wrong dataset")
        sample_id = str(row.get("sample_id") or "").strip()
        if not sample_id or sample_id in sample_ids:
            raise ValueError(f"{dataset} {split} has duplicate/empty sample_id")
        answer = str(
            row.get("answer")
            or row.get("correct_answer")
            or row.get("right_answer")
            or ""
        ).strip().upper()
        if len(answer) != 1 or not "A" <= answer <= "H":
            raise ValueError(f"{dataset} {split} row {index} has no A-H answer")
        sample_ids.add(sample_id)
        videos.add(_video_id(row.get("video")))
    return sample_ids, videos


def _atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="wb",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".partial",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        handle.write(content.encode("utf-8"))
    temporary.replace(path)


def _freeze(path: Path, content: str) -> str:
    if path.exists():
        if path.read_text(encoding="utf-8") != content:
            raise RuntimeError(f"refusing to overwrite changed frozen artifact: {path}")
        return "existing_identical"
    _atomic_write(path, content)
    return "created"


def freeze_train600(
    *,
    config_path: Path,
    output_path: Path,
    metadata_path: Path,
    expected_split_counts: Mapping[str, int] = DEFAULT_SPLIT_COUNTS,
) -> dict[str, Any]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    datasets = config.get("datasets") if isinstance(config, dict) else None
    if not isinstance(datasets, dict) or set(datasets) != set(DATASETS):
        raise ValueError(f"config must contain exactly {DATASETS}")
    if set(expected_split_counts) != {"train", "dev", "final"}:
        raise ValueError("expected split counts must contain train/dev/final")

    merged: list[dict[str, Any]] = []
    source_report: dict[str, Any] = {}
    composite_ids: set[tuple[str, str]] = set()
    absolute_video_owners: dict[str, str] = {}
    for dataset in DATASETS:
        dataset_config = datasets[dataset]
        if not isinstance(dataset_config, dict):
            raise ValueError(f"datasets.{dataset} must be an object")
        split_rows: dict[str, list[dict[str, Any]]] = {}
        split_ids: dict[str, set[str]] = {}
        split_videos: dict[str, set[str]] = {}
        split_report: dict[str, Any] = {}
        for split in ("train", "dev", "final"):
            spec = dataset_config.get(split)
            if not isinstance(spec, dict):
                raise ValueError(f"datasets.{dataset}.{split} must be an object")
            path = Path(str(spec.get("path") or ""))
            if not path.is_file():
                raise FileNotFoundError(path)
            actual_hash = sha256_file(path)
            expected_hash = str(spec.get("sha256") or "").lower()
            if actual_hash != expected_hash:
                raise RuntimeError(f"{dataset} {split} manifest SHA-256 mismatch")
            rows = _read_jsonl(path)
            ids, videos = _validate_rows(
                rows,
                dataset=dataset,
                split=split,
                expected_count=int(expected_split_counts[split]),
            )
            split_rows[split] = rows
            split_ids[split] = ids
            split_videos[split] = videos
            split_report[split] = {
                "path": str(path.resolve()),
                "sha256": actual_hash,
                "samples": len(rows),
                "videos": len(videos),
            }
        for left, right in (("train", "dev"), ("train", "final"), ("dev", "final")):
            if split_ids[left] & split_ids[right]:
                raise ValueError(f"{dataset} sample overlap: {left}/{right}")
            if split_videos[left] & split_videos[right]:
                raise ValueError(f"{dataset} video overlap: {left}/{right}")

        for row in split_rows["train"]:
            sample_id = str(row["sample_id"]).strip()
            identity = (dataset, sample_id)
            if identity in composite_ids:
                raise ValueError(f"duplicate composite sample identity: {identity}")
            composite_ids.add(identity)
            video = _video_id(row.get("video"))
            if _is_absolute_video(video):
                absolute_video = os.path.normcase(os.path.normpath(video))
                owner = absolute_video_owners.setdefault(absolute_video, dataset)
                if owner != dataset:
                    raise ValueError(
                        f"absolute video is shared across datasets: {owner}/{dataset}"
                    )
            merged.append(row)
        source_report[dataset] = split_report

    output_content = "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
        for row in merged
    )
    output_sha256 = hashlib.sha256(output_content.encode("utf-8")).hexdigest()
    metadata = {
        "schema_version": 1,
        "artifact": "qwen_train600",
        "config": {
            "path": str(config_path.resolve()),
            "sha256": sha256_file(config_path),
        },
        "sources": source_report,
        "counts": {dataset: int(expected_split_counts["train"]) for dataset in DATASETS},
        "total": len(merged),
        "composite_sample_ids_sha256": _canonical_sha256(sorted(composite_ids)),
        "video_isolation": "passed",
        "output": {
            "path": str(output_path.resolve()),
            "sha256": output_sha256,
        },
    }
    metadata_content = json.dumps(
        metadata,
        ensure_ascii=False,
        sort_keys=True,
        indent=2,
    ) + "\n"
    output_status = _freeze(output_path, output_content)
    metadata_status = _freeze(metadata_path, metadata_content)
    return {
        **metadata,
        "output_status": output_status,
        "metadata_status": metadata_status,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Freeze the three read-only Train200 manifests as one audited Train600."
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    args = parser.parse_args()
    report = freeze_train600(
        config_path=args.config,
        output_path=args.output,
        metadata_path=args.metadata,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
