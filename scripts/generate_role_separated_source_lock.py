#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
from pathlib import Path

from flashvid_eval.role_separated_orchestration import build_source_lock, load_config


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a deterministic seven-repository source lock."
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config_bytes = args.config.read_bytes()
    config = load_config(args.config)
    lock = build_source_lock(
        config, config_sha256=hashlib.sha256(config_bytes).hexdigest()
    )
    _write_json(args.output, lock)
    print(json.dumps(lock, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
