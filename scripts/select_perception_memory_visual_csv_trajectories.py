#!/usr/bin/env python3
"""Offline-label Visual-CSV prefixes and select stable process-SFT traces."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from flashvid_eval.perception_memory_selection import load_frozen_answers
from flashvid_eval.perception_memory_visual_csv import (
    select_visual_csv_trajectories,
)
from flashvid_eval.qwen_sft import read_jsonl


_TRAIN_COUNTS = {"lvbench": 200, "lsdbench": 200, "cgbench": 200}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_many(paths: Sequence[Path]) -> list[dict[str, Any]]:
    return [row for path in paths for row in read_jsonl(path)]


def _load_train600(
    path: Path, expected_sha256: str
) -> tuple[dict[tuple[str, str], str], str]:
    digest = _sha256(path)
    if digest.lower() != expected_sha256.lower():
        raise ValueError(
            f"Train600 answer SHA-256 mismatch: expected {expected_sha256}, got {digest}"
        )
    rows = read_jsonl(path)
    counts = {dataset: 0 for dataset in _TRAIN_COUNTS}
    for row in rows:
        dataset = str(row.get("dataset") or "").strip()
        if dataset not in counts:
            raise ValueError(f"Train600 answer table has unexpected dataset: {dataset!r}")
        counts[dataset] += 1
    if len(rows) != 600 or counts != _TRAIN_COUNTS:
        raise ValueError(
            f"Train600 answer table must be 600 rows/200 per dataset, found {counts}"
        )
    return load_frozen_answers(rows), digest


def _temporary(path: Path, lines: Iterable[str]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False
    ) as handle:
        temporary = Path(handle.name)
        for line in lines:
            handle.write(line)
        handle.flush()
        os.fsync(handle.fileno())
    return temporary


def _jsonl_lines(rows: Iterable[Mapping[str, Any]]) -> Iterable[str]:
    for row in rows:
        yield json.dumps(row, ensure_ascii=False, default=str) + "\n"


def _publish_outputs(
    labeled_path: Path,
    labeled: Iterable[Mapping[str, Any]],
    selected_path: Path,
    selected: Iterable[Mapping[str, Any]],
    summary_path: Path,
    summary: Mapping[str, Any],
) -> None:
    temporary: list[tuple[Path, Path]] = []
    try:
        temporary.append(
            (labeled_path, _temporary(labeled_path, _jsonl_lines(labeled)))
        )
        temporary.append(
            (selected_path, _temporary(selected_path, _jsonl_lines(selected)))
        )
        temporary.append(
            (
                summary_path,
                _temporary(
                    summary_path,
                    (
                        json.dumps(
                            summary, ensure_ascii=False, indent=2, sort_keys=True
                        )
                        + "\n",
                    ),
                ),
            )
        )
        for destination, source in temporary:
            os.replace(source, destination)
    finally:
        for _destination, source in temporary:
            if source.exists():
                source.unlink()


def run(args: argparse.Namespace) -> dict[str, Any]:
    outputs = (args.labeled_output, args.selected_output, args.summary)
    if any(path.exists() for path in outputs) and not args.overwrite:
        raise RuntimeError("selection output exists; pass --overwrite")
    trajectories = _load_many(args.trajectories)
    visual_csv_results = _load_many(args.visual_csv_results)
    answers, answers_sha256 = _load_train600(
        args.answers, args.expected_answers_sha256
    )
    for index, trajectory in enumerate(trajectories):
        if trajectory.get("train600_manifest_sha256") != answers_sha256:
            raise ValueError(
                f"trajectory[{index}] is not bound to the supplied Train600 artifact"
            )
    labeled, selected, selection_summary = select_visual_csv_trajectories(
        trajectories,
        visual_csv_results,
        answers,
        verifier_artifact_sha256=args.verifier_artifact_sha256,
    )
    report = {
        **selection_summary,
        "train600_answers": {
            "path": str(args.answers.resolve()),
            "sha256": answers_sha256,
            "samples": 600,
            "per_dataset": dict(_TRAIN_COUNTS),
        },
        "labeled_output": str(args.labeled_output.resolve()),
        "selected_output": str(args.selected_output.resolve()),
        "summary": str(args.summary.resolve()),
    }
    _publish_outputs(
        args.labeled_output,
        labeled,
        args.selected_output,
        selected,
        args.summary,
        report,
    )
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trajectories", type=Path, nargs="+", required=True)
    parser.add_argument(
        "--visual-csv-results", type=Path, nargs="+", required=True
    )
    parser.add_argument("--verifier-artifact-sha256", required=True)
    parser.add_argument("--answers", type=Path, required=True)
    parser.add_argument("--expected-answers-sha256", required=True)
    parser.add_argument("--labeled-output", type=Path, required=True)
    parser.add_argument("--selected-output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    try:
        summary = run(args)
    except (OSError, ValueError, TypeError, KeyError, RuntimeError) as error:
        print(
            json.dumps(
                {"status": "failed", "error": f"{type(error).__name__}: {error}"},
                ensure_ascii=False,
            )
        )
        return 1
    print(json.dumps({"status": "passed", **summary}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
