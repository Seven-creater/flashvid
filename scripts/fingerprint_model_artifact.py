#!/usr/bin/env python3
"""Create a reproducible identity for every file in a local model artifact."""

from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
from pathlib import Path
from typing import Any


IGNORED_NAMES = frozenset({"READY.txt"})
IGNORED_DIRECTORIES = frozenset({".cache"})


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def fingerprint_model(root: Path) -> dict[str, Any]:
    resolved = root.resolve()
    if not resolved.is_dir():
        raise FileNotFoundError(resolved)
    files = [
        path
        for path in sorted(resolved.rglob("*"))
        if path.is_file()
        and path.name not in IGNORED_NAMES
        and not any(
            part in IGNORED_DIRECTORIES
            for part in path.relative_to(resolved).parts
        )
    ]
    if not files or not any(path.suffix == ".safetensors" for path in files):
        raise ValueError(f"model artifact contains no safetensors weights: {resolved}")
    records = [
        {
            "path": path.relative_to(resolved).as_posix(),
            "size": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in files
    ]
    canonical = json.dumps(
        records,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return {
        "schema_version": 1,
        "model_root": str(resolved),
        "artifact_sha256": hashlib.sha256(canonical).hexdigest(),
        "file_count": len(records),
        "total_bytes": sum(record["size"] for record in records),
        "files": records,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-sha256")
    args = parser.parse_args()
    payload = fingerprint_model(args.model_root)
    if (
        args.expected_sha256 is not None
        and payload["artifact_sha256"].lower() != args.expected_sha256.lower()
    ):
        raise RuntimeError(
            "model artifact SHA-256 mismatch: "
            f"expected {args.expected_sha256}, got {payload['artifact_sha256']}"
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=args.output.parent,
        prefix=f".{args.output.name}.",
        suffix=".partial",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    temporary.replace(args.output)
    print(json.dumps({key: payload[key] for key in ("model_root", "artifact_sha256", "file_count", "total_bytes")}))


if __name__ == "__main__":
    main()
