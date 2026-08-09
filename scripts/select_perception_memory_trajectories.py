#!/usr/bin/env python3
"""Offline-score prefix Judges and select stable Perception-Memory trajectories."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from flashvid_eval.perception_memory_selection import (
    label_and_select_trajectories,
    load_frozen_answers,
)
from flashvid_eval.qwen_sft import read_jsonl


_TRAIN_COUNTS = {"lvbench": 200, "lsdbench": 200, "cgbench": 200}


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_bound_train600_answers(
    path: Path, expected_sha256: str
) -> tuple[dict[tuple[str, str], str], str]:
    _rows, answers, digest = _load_bound_train600_source(path, expected_sha256)
    return answers, digest


def _load_bound_train600_source(
    path: Path, expected_sha256: str
) -> tuple[list[dict[str, Any]], dict[tuple[str, str], str], str]:
    digest = _file_sha256(path)
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
    return rows, load_frozen_answers(rows), digest


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False
    ) as handle:
        temporary = Path(handle.name)
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
    os.replace(temporary, path)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False
    ) as handle:
        temporary = Path(handle.name)
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, path)


def _load_many(paths: Sequence[Path]) -> list[dict[str, Any]]:
    return [row for path in paths for row in read_jsonl(path)]


def _rescue_paths(directory: Path) -> dict[str, Path]:
    return {
        dataset: directory / f"{dataset}_no_stable.jsonl"
        for dataset in _TRAIN_COUNTS
    }


def _build_rescue_rows(
    train600_rows: Sequence[Mapping[str, Any]],
    selection_summary: Mapping[str, Any],
) -> dict[str, list[dict[str, Any]]]:
    raw_ids = selection_summary.get("no_stable_sample_ids")
    raw_counts = selection_summary.get("no_stable_by_dataset")
    if not isinstance(raw_ids, Mapping) or not isinstance(raw_counts, Mapping):
        raise ValueError("selection summary has no no-stable sample scope")
    unexpected = set(raw_ids) - set(_TRAIN_COUNTS)
    if unexpected:
        raise ValueError(f"no-stable scope has unexpected datasets: {sorted(unexpected)}")

    source: dict[tuple[str, str], dict[str, Any]] = {}
    for row in train600_rows:
        identity = (
            str(row.get("dataset") or "").strip(),
            str(row.get("sample_id") or "").strip(),
        )
        if identity in source:
            raise ValueError("Train600 source contains a duplicate sample identity")
        source[identity] = dict(row)

    manifests: dict[str, list[dict[str, Any]]] = {}
    for dataset in _TRAIN_COUNTS:
        values = raw_ids.get(dataset, [])
        if not isinstance(values, list) or any(
            not isinstance(value, str) or not value.strip() for value in values
        ):
            raise ValueError(f"invalid no-stable sample IDs for {dataset}")
        sample_ids = [value.strip() for value in values]
        if sample_ids != sorted(set(sample_ids)):
            raise ValueError(f"no-stable sample IDs are not unique/sorted for {dataset}")
        if int(raw_counts.get(dataset, 0)) != len(sample_ids):
            raise ValueError(f"no-stable count mismatch for {dataset}")
        rows: list[dict[str, Any]] = []
        for sample_id in sample_ids:
            identity = (dataset, sample_id)
            if identity not in source:
                raise ValueError(
                    f"no-stable sample is absent from bound Train600 source: {identity}"
                )
            rows.append(dict(source[identity]))
        manifests[dataset] = rows
    if sum(len(rows) for rows in manifests.values()) != int(
        selection_summary.get("no_stable", -1)
    ):
        raise ValueError("no-stable total count mismatch")
    return manifests


def run(args: argparse.Namespace) -> dict[str, Any]:
    rescue_directory = getattr(args, "rescue_manifest_dir", None)
    rescue_paths = (
        _rescue_paths(Path(rescue_directory)) if rescue_directory is not None else {}
    )
    outputs = [args.labeled_output, args.selected_output, args.summary]
    outputs.extend(rescue_paths.values())
    for output in outputs:
        if output.exists() and not args.overwrite:
            raise RuntimeError(f"output exists; pass --overwrite: {output}")
    trajectories = _load_many(args.trajectories)
    judgments = _load_many(args.prefix_judgments)
    train600_rows, answers, answers_sha256 = _load_bound_train600_source(
        args.answers, args.expected_answers_sha256
    )
    for index, trajectory in enumerate(trajectories):
        if trajectory.get("train600_manifest_sha256") != answers_sha256:
            raise ValueError(
                f"trajectory[{index}] is not bound to the supplied Train600 artifact"
            )
    labeled, selected, summary = label_and_select_trajectories(
        trajectories, judgments, answers
    )
    rescue_rows = (
        _build_rescue_rows(train600_rows, summary) if rescue_paths else {}
    )
    _write_jsonl(args.labeled_output, labeled)
    _write_jsonl(args.selected_output, selected)
    rescue_manifests: dict[str, dict[str, Any]] = {}
    for dataset, path in rescue_paths.items():
        _write_jsonl(path, rescue_rows[dataset])
        rescue_manifests[dataset] = {
            "path": str(path.resolve()),
            "sha256": _file_sha256(path),
            "samples": len(rescue_rows[dataset]),
            "sample_ids": [str(row["sample_id"]) for row in rescue_rows[dataset]],
        }
    report = {
        **summary,
        "train600_answers": {
            "path": str(args.answers.resolve()),
            "sha256": answers_sha256,
            "samples": 600,
            "per_dataset": dict(_TRAIN_COUNTS),
        },
        "labeled_output": str(args.labeled_output.resolve()),
        "selected_output": str(args.selected_output.resolve()),
        "summary": str(args.summary.resolve()),
        "rescue_manifests": rescue_manifests,
    }
    _write_json(args.summary, report)
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trajectories", type=Path, nargs="+", required=True)
    parser.add_argument("--prefix-judgments", type=Path, nargs="+", required=True)
    parser.add_argument("--answers", type=Path, required=True)
    parser.add_argument("--expected-answers-sha256", required=True)
    parser.add_argument("--labeled-output", type=Path, required=True)
    parser.add_argument("--selected-output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--rescue-manifest-dir", type=Path)
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
