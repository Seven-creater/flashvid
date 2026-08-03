from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from flashvid_eval.datasets import VideoIndex, load_samples
from flashvid_eval.offline_budget import (
    route_counts,
    split_samples_by_video,
    video_group_key,
)
from flashvid_eval.schemas import Sample


def _read_manifest(path: Path) -> list[Sample]:
    return [
        Sample(**json.loads(line))
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_jsonl(path: Path, samples: tuple[Sample, ...]) -> None:
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(
        "".join(
            json.dumps(sample.to_dict(), ensure_ascii=False) + "\n"
            for sample in samples
        ),
        encoding="utf-8",
    )
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare deterministic video-disjoint train/dev manifests."
    )
    parser.add_argument(
        "--dataset",
        choices=("cgbench", "lvbench", "lsdbench"),
        required=True,
    )
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--video-root", type=Path, required=True)
    parser.add_argument("--test-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--train-count", type=int, default=200)
    parser.add_argument("--dev-count", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    all_candidates = load_samples(args.dataset, args.annotations)
    video_index = VideoIndex(args.video_root)
    candidates: list[Sample] = []
    inaccessible_sample_ids: list[str] = []
    for sample in all_candidates:
        try:
            video_index.resolve(sample.video)
        except FileNotFoundError:
            inaccessible_sample_ids.append(sample.sample_id)
            continue
        candidates.append(sample)
    frozen_test = _read_manifest(args.test_manifest)
    split = split_samples_by_video(
        candidates,
        frozen_test,
        train_count=args.train_count,
        dev_count=args.dev_count,
        seed=args.seed,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    train_path = args.output_dir / f"{args.dataset}_train.jsonl"
    dev_path = args.output_dir / f"{args.dataset}_dev.jsonl"
    summary_path = args.output_dir / f"{args.dataset}_split_summary.json"
    _write_jsonl(train_path, split.train)
    _write_jsonl(dev_path, split.dev)

    summary: dict[str, Any] = {
        "dataset": args.dataset,
        "seed": args.seed,
        "requested": {"train": args.train_count, "dev": args.dev_count},
        "counts": {
            "annotation_samples": len(all_candidates),
            "candidate_samples": len(candidates),
            "inaccessible_samples": len(inaccessible_sample_ids),
            "train_samples": len(split.train),
            "dev_samples": len(split.dev),
            "excluded_test_samples": len(split.excluded_test_sample_ids),
            "unused_samples": len(split.unused_sample_ids),
            "train_videos": len({video_group_key(sample) for sample in split.train}),
            "dev_videos": len({video_group_key(sample) for sample in split.dev}),
        },
        "route_counts": {
            "train": route_counts(split.train),
            "dev": route_counts(split.dev),
        },
        "inputs": {
            "annotations": {
                "path": str(args.annotations.resolve()),
                "sha256": _sha256(args.annotations),
            },
            "video_root": str(args.video_root.resolve()),
            "frozen_test_manifest": {
                "path": str(args.test_manifest.resolve()),
                "sha256": _sha256(args.test_manifest),
            },
        },
        "outputs": {
            "train": {"path": str(train_path.resolve()), "sha256": _sha256(train_path)},
            "dev": {"path": str(dev_path.resolve()), "sha256": _sha256(dev_path)},
        },
        "video_overlap": {
            "train_dev": 0,
            "train_test": 0,
            "dev_test": 0,
        },
        "inaccessible_sample_ids": sorted(inaccessible_sample_ids),
    }
    temporary = summary_path.with_suffix(summary_path.suffix + ".partial")
    temporary.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(summary_path)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
