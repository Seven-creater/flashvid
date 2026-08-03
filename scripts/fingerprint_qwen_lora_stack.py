#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
from pathlib import Path

from scripts.fingerprint_model_artifact import fingerprint_model


def validated_sha256(value: str, label: str) -> str:
    digest = value.lower()
    if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        raise ValueError(f"{label} must be a SHA-256")
    return digest


def fingerprint_stack(base_artifact_sha256: str, adapter_root: Path) -> dict:
    base_hash = validated_sha256(base_artifact_sha256, "base-artifact-sha256")
    adapter = fingerprint_model(adapter_root)
    identity = {
        "base_artifact_sha256": base_hash,
        "adapter_artifact_sha256": adapter["artifact_sha256"],
        "adapter_file_count": adapter["file_count"],
        "adapter_total_bytes": adapter["total_bytes"],
    }
    serialized = json.dumps(identity, sort_keys=True, separators=(",", ":"))
    return {
        "schema_version": 1,
        **identity,
        "served_stack_sha256": hashlib.sha256(serialized.encode("utf-8")).hexdigest(),
        "adapter_root": adapter["model_root"],
        "adapter_files": adapter["files"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fingerprint a frozen Qwen base artifact plus one LoRA checkpoint."
    )
    parser.add_argument("--base-artifact-sha256", required=True)
    parser.add_argument("--adapter-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = fingerprint_stack(args.base_artifact_sha256, args.adapter_root)
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
    print(
        json.dumps(
            {
                "served_stack_sha256": payload["served_stack_sha256"],
                "adapter_artifact_sha256": payload["adapter_artifact_sha256"],
            }
        )
    )


if __name__ == "__main__":
    main()
