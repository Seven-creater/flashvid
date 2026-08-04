#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path
from typing import Iterable, Mapping

from flashvid_eval.fast_hybrid_trajectory_control import prepare_prejudge_candidates
from flashvid_eval.qwen_sft import load_training_manifest, read_jsonl, sha256_file


def _load_many(paths: list[Path]) -> list[dict]:
    return [row for path in paths for row in read_jsonl(path)]


def _jsonl(path: Path, rows: Iterable[Mapping]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False
    ) as handle:
        temporary = Path(handle.name)
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train600", type=Path, required=True)
    parser.add_argument("--expected-train600-sha256", required=True)
    parser.add_argument("--specs", type=Path, required=True)
    parser.add_argument("--raw-trajectories", type=Path, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    try:
        manifest = load_training_manifest(args.train600)
        if manifest.sha256 != args.expected_train600_sha256:
            raise RuntimeError("Train600 SHA-256 mismatch")
        specs = read_jsonl(args.specs)
        raw = _load_many(args.raw_trajectories)
        outcome = prepare_prejudge_candidates(specs, raw, manifest.answers)
        _jsonl(args.output_dir / "judge_specs.jsonl", outcome.eligible_specs)
        _jsonl(
            args.output_dir / "judge_trajectories.jsonl",
            outcome.eligible_trajectories,
        )
        _jsonl(
            args.output_dir / "prejudge_completion_index.jsonl",
            outcome.completion_index,
        )
        summary = {
            "schema_version": 1,
            "status": "passed",
            "train600_sha256": manifest.sha256,
            "specs_sha256": sha256_file(args.specs),
            "raw_trajectory_sha256s": {
                str(path.resolve()): sha256_file(path)
                for path in args.raw_trajectories
            },
            "planned": len(specs),
            "eligible": len(outcome.eligible_specs),
            "rejected": outcome.rejected,
        }
        args.output_dir.mkdir(parents=True, exist_ok=True)
        (args.output_dir / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    except (OSError, ValueError, TypeError, KeyError, RuntimeError) as error:
        print(json.dumps({"status": "failed", "error": str(error)}, ensure_ascii=False))
        return 1
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
