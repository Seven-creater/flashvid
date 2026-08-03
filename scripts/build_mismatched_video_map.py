from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from flashvid_eval.baseline_diagnostics import (
    DEFAULT_DURATION_BUCKET_EDGES_S,
    VideoDiagnosticItem,
    assign_mismatched_videos,
)
from flashvid_eval.datasets import VideoIndex
from flashvid_eval.media import probe_video


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Freeze same-dataset, same-duration-bucket wrong-video controls."
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--video-root", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    rows = [
        json.loads(line)
        for line in args.manifest.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    index = VideoIndex(args.video_root)
    items: list[VideoDiagnosticItem] = []
    unavailable: list[str] = []
    for row in rows:
        sample_id = str(row["sample_id"])
        try:
            video = index.resolve(str(row["video"]))
        except FileNotFoundError:
            unavailable.append(sample_id)
            continue
        items.append(
            VideoDiagnosticItem(
                dataset=str(row["dataset"]),
                sample_id=sample_id,
                video=str(row["video"]),
                duration_s=float(probe_video(video)["duration"]),
            )
        )
    assignments = assign_mismatched_videos(items, seed=args.seed)
    mapping = {item.sample_id: item.target_video for item in assignments}
    excluded = sorted(
        {item.sample_id for item in items} - set(mapping)
    )
    payload = {
        "schema_version": 1,
        "manifest": str(args.manifest.resolve()),
        "manifest_sha256": sha256(args.manifest),
        "video_root": str(args.video_root.resolve()),
        "seed": args.seed,
        "duration_bucket_edges_s": [
            "inf" if edge == float("inf") else float(edge)
            for edge in DEFAULT_DURATION_BUCKET_EDGES_S
        ],
        "mapping": mapping,
        "assignments": [item.__dict__ for item in assignments],
        "source_video_unavailable": unavailable,
        "duration_bucket_without_alternative": excluded,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".partial")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(args.output)
    print(json.dumps({"mapped": len(mapping), "excluded": len(excluded), "unavailable": len(unavailable)}))


if __name__ == "__main__":
    main()
