#!/usr/bin/env python3
"""Freeze label-free Dev role-ablation inputs from one Base runtime result set."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from flashvid_eval.qwen_sft import read_jsonl
from flashvid_eval.role_ablation_replay import (
    freeze_role_ablation_inputs,
    validate_role_ablation_dev30_scope,
)


def _jsonl_bytes(rows: Iterable[Mapping[str, Any]]) -> bytes:
    return b"".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        .encode("utf-8")
        + b"\n"
        for row in rows
    )


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("wb", dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def run(inputs: Sequence[Path], output: Path, *, overwrite: bool) -> dict[str, Any]:
    if output.exists() and not overwrite:
        raise FileExistsError(f"output exists: {output}")
    sources = [row for path in inputs for row in read_jsonl(path)]
    frozen = freeze_role_ablation_inputs(sources)
    counts = validate_role_ablation_dev30_scope(frozen)
    payload = _jsonl_bytes(frozen)
    _atomic_write(output, payload)
    return {
        "inputs": [str(path.resolve()) for path in inputs],
        "samples": len(frozen),
        "datasets": counts,
        "output": str(output.resolve()),
        "output_sha256": hashlib.sha256(payload).hexdigest(),
        "labels_serialized": False,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    try:
        summary = run(args.input, args.output, overwrite=args.overwrite)
    except (OSError, TypeError, ValueError, KeyError) as error:
        print(json.dumps({"status": "failed", "error": str(error)}))
        return 1
    print(json.dumps({"status": "passed", **summary}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
