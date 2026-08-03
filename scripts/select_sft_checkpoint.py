from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
import uuid
from collections import defaultdict
from pathlib import Path
from typing import Any

from flashvid_eval.budget_reporting import (
    DATASETS,
    choose_sft_checkpoint,
    read_jsonl_latest,
    score_checkpoint,
)


def _checkpoint_inputs(
    validation_root: Path,
    explicit: list[str],
) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for value in explicit:
        if "=" not in value:
            raise ValueError("--checkpoint must use NAME=PATH")
        name, raw_path = value.split("=", 1)
        if not name or not raw_path:
            raise ValueError("--checkpoint must use NAME=PATH")
        result[name] = Path(raw_path)
    if result:
        return result
    if not validation_root.is_dir():
        return {}
    return {
        path.name: path
        for path in sorted(validation_root.iterdir())
        if path.is_dir() and path.name.startswith(("checkpoint-", "epoch-"))
    }


def _load_checkpoint(path: Path) -> tuple[
    dict[str, list[dict[str, Any]]],
    dict[str, int],
    dict[str, int],
    list[str],
]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    duplicate_counts: defaultdict[str, int] = defaultdict(int)
    invalid_counts: defaultdict[str, int] = defaultdict(int)
    files = [path] if path.is_file() else sorted(path.rglob("*.jsonl"))
    used: list[str] = []
    seen_paths: set[Path] = set()
    for result_path in files:
        resolved = result_path.resolve()
        if resolved in seen_paths:
            continue
        seen_paths.add(resolved)
        loaded = read_jsonl_latest(result_path)
        evaluable = tuple(
            record
            for record in loaded.records
            if (
                record.get("prediction") is not None
                or record.get("final_prediction") is not None
            )
            and (
                record.get("answer") is not None
                or record.get("correct_answer") is not None
                or record.get("right_answer") is not None
            )
        )
        if not evaluable:
            continue
        datasets = {
            str(record.get("dataset") or "").strip().lower()
            for record in evaluable
        }
        datasets.discard("")
        if len(datasets) != 1 or next(iter(datasets)) not in DATASETS:
            continue
        dataset = next(iter(datasets))
        grouped[dataset].extend(evaluable)
        duplicate_counts[dataset] += len(loaded.duplicate_sample_ids)
        invalid_counts[dataset] += loaded.invalid_lines
        used.append(str(result_path.resolve()))

    # A sample repeated across separate shards is also a duplicate.
    for dataset, records in tuple(grouped.items()):
        latest: dict[str, dict[str, Any]] = {}
        counts: defaultdict[str, int] = defaultdict(int)
        for record in records:
            sample_id = str(record.get("sample_id"))
            counts[sample_id] += 1
            latest[sample_id] = record
        duplicate_counts[dataset] += sum(count > 1 for count in counts.values())
        grouped[dataset] = [latest[sample_id] for sample_id in sorted(latest)]
    return dict(grouped), dict(duplicate_counts), dict(invalid_counts), used


def _parse_expected(values: list[str]) -> dict[str, int]:
    expected = {dataset: 50 for dataset in DATASETS}
    for value in values:
        if "=" not in value:
            raise ValueError("--expected must use DATASET=COUNT")
        dataset, raw_count = value.split("=", 1)
        dataset = dataset.lower()
        if dataset not in DATASETS:
            raise ValueError(f"unsupported dataset in --expected: {dataset}")
        count = int(raw_count)
        if count <= 0:
            raise ValueError("expected count must be positive")
        expected[dataset] = count
    return expected


def _tree_hashes(
    root: Path,
    *,
    excluded_names: frozenset[str] = frozenset(),
) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        relative = path.relative_to(root).as_posix()
        if relative in excluded_names:
            continue
        hashes[relative] = hashlib.sha256(
            path.read_bytes()
        ).hexdigest()
    return hashes


def _materialize_selected(
    source: Path,
    target: Path,
    checkpoint: str,
) -> None:
    payload = {
        "selected_checkpoint": checkpoint,
        "source": str(source.resolve()),
        "files": _tree_hashes(source),
    }
    metadata_name = "selected_checkpoint.json"
    if target.is_dir():
        metadata_path = target / metadata_name
        if metadata_path.is_file():
            try:
                current_metadata = json.loads(
                    metadata_path.read_text(encoding="utf-8")
                )
            except (OSError, UnicodeError, json.JSONDecodeError):
                current_metadata = None
            current_files = _tree_hashes(
                target,
                excluded_names=frozenset({metadata_name}),
            )
            if current_metadata == payload and current_files == payload["files"]:
                return
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{target.name}.", dir=target.parent)
    )
    previous: Path | None = None
    try:
        for item in source.iterdir():
            destination = temporary / item.name
            if item.is_dir():
                shutil.copytree(item, destination)
            elif item.is_file():
                shutil.copy2(item, destination)
        (temporary / metadata_name).write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        if target.exists() or target.is_symlink():
            previous = target.parent / (
                f".{target.name}.previous.{uuid.uuid4().hex}"
            )
            os.replace(target, previous)
        os.replace(temporary, target)
        if previous is not None:
            stale = previous
            previous = None
            try:
                if stale.is_dir() and not stale.is_symlink():
                    shutil.rmtree(stale)
                else:
                    stale.unlink()
            except OSError:
                # The new materialization and its source are already intact.
                # A stale hidden backup can be cleaned up separately.
                pass
    except Exception:
        if previous is not None and previous.exists() and not target.exists():
            os.replace(previous, target)
            previous = None
        raise
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Select an SFT checkpoint from end-to-end validation JSONL. "
            "Only complete checkpoints with identical sample sets are comparable."
        )
    )
    parser.add_argument(
        "--validation-root",
        type=Path,
        default=Path("results/eval/flashvid_budget_v1/checkpoints/validation"),
    )
    parser.add_argument(
        "--checkpoint",
        action="append",
        default=[],
        metavar="NAME=PATH",
        help="Explicit checkpoint result directory/file; repeat as needed.",
    )
    parser.add_argument(
        "--expected",
        action="append",
        default=[],
        metavar="DATASET=COUNT",
        help="Expected validation samples (default: 50 for each dataset).",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--materialize-selected-dir",
        type=Path,
        help=(
            "Copy the selected checkpoint's validation results into this "
            "stable, immutable directory for reporting."
        ),
    )
    args = parser.parse_args()

    expected = _parse_expected(args.expected)
    inputs = _checkpoint_inputs(args.validation_root, args.checkpoint)
    scores: list[dict[str, Any]] = []
    source_files: dict[str, list[str]] = {}
    for name, path in sorted(inputs.items()):
        records, duplicates, invalid, used = _load_checkpoint(path)
        scores.append(
            score_checkpoint(
                name,
                records,
                expected_counts=expected,
                duplicate_counts=duplicates,
                invalid_line_counts=invalid,
            )
        )
        source_files[name] = used
    selection = choose_sft_checkpoint(scores)
    result = {
        "schema_version": 1,
        "expected_counts": expected,
        "source_files": source_files,
        "checkpoints": scores,
        "selection": selection,
    }
    output = args.output or args.validation_root / "checkpoint_selection.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".partial")
    temporary.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(output)
    selected_name = selection.get("selected_checkpoint")
    if args.materialize_selected_dir is not None:
        if not selected_name:
            raise RuntimeError(
                "cannot materialize selected results because no checkpoint is eligible"
            )
        _materialize_selected(
            inputs[str(selected_name)],
            args.materialize_selected_dir,
            str(selected_name),
        )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
