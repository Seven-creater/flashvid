from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def _rows(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _videos(path: Path) -> set[str]:
    return {str(row["video"]) for row in _rows(path)}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate frozen train/dev candidate files and video-disjoint splits."
    )
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--test-manifest-root", type=Path, required=True)
    args = parser.parse_args()

    failures: list[str] = []
    expected_counts = {"train": 200, "dev": 50}
    for dataset in ("lvbench", "lsdbench", "cgbench"):
        for split, expected in expected_counts.items():
            path = (
                args.root
                / "candidates"
                / "qwen9b"
                / dataset
                / split
                / f"{dataset}_direct.jsonl"
            )
            rows = _rows(path)
            sample_ids = [str(row.get("sample_id")) for row in rows]
            valid = sum(
                bool(row.get("prediction") or row.get("final_prediction"))
                for row in rows
            )
            errors = sum(bool(row.get("error")) for row in rows)
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            print(
                f"{dataset}/{split}: rows={len(rows)} unique={len(set(sample_ids))} "
                f"valid={valid} errors={errors} sha256={digest}"
            )
            if len(rows) != expected:
                failures.append(f"{dataset}/{split}: expected {expected}, got {len(rows)}")
            if len(sample_ids) != len(set(sample_ids)):
                failures.append(f"{dataset}/{split}: duplicate sample_id")

        split_root = args.root / "splits"
        train_videos = _videos(split_root / f"{dataset}_train.jsonl")
        dev_videos = _videos(split_root / f"{dataset}_dev.jsonl")
        test_videos = _videos(
            args.test_manifest_root / f"{dataset}_manifest_42_100.jsonl"
        )
        overlaps = {
            "train_dev": train_videos & dev_videos,
            "train_test": train_videos & test_videos,
            "dev_test": dev_videos & test_videos,
        }
        print(
            f"{dataset} video overlaps: "
            + ", ".join(f"{name}={len(items)}" for name, items in overlaps.items())
        )
        for name, items in overlaps.items():
            if items:
                failures.append(
                    f"{dataset}: {name} overlap, first={sorted(items)[0]}"
                )

    if failures:
        raise SystemExit("\n".join(failures))


if __name__ == "__main__":
    main()
