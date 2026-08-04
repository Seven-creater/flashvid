#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

from flashvid_eval.fast_hybrid_checkpoint_selection import (
    CheckpointRun,
    select_fast_hybrid_checkpoint,
)


def _mapping(values: list[str], label: str) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"{label} must use DATASET=PATH")
        dataset, raw_path = value.split("=", 1)
        dataset = dataset.strip().lower()
        if dataset in result:
            raise ValueError(f"duplicate {label} dataset: {dataset}")
        result[dataset] = Path(raw_path)
    return result


def _checkpoint(value: str) -> CheckpointRun:
    parts = value.split("|", 2)
    if len(parts) != 3:
        raise ValueError("--checkpoint must use ID|EPOCH|JSON_MAPPING")
    checkpoint_id, raw_epoch, raw_mapping = parts
    mapping = json.loads(raw_mapping)
    if not isinstance(mapping, dict):
        raise ValueError("checkpoint result mapping must be a JSON object")
    return CheckpointRun(
        checkpoint_id=checkpoint_id,
        epoch=int(raw_epoch),
        result_paths={str(key): Path(str(path)) for key, path in mapping.items()},
    )


def _write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False
    ) as handle:
        temporary = Path(handle.name)
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teacher", action="append", required=True)
    parser.add_argument("--checkpoint", action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        report = select_fast_hybrid_checkpoint(
            teacher_paths=_mapping(args.teacher, "teacher"),
            checkpoints=[_checkpoint(value) for value in args.checkpoint],
        )
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError) as error:
        report = {"schema_version": 1, "status": "failed", "error": str(error)}
    _write(args.output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["status"] == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
