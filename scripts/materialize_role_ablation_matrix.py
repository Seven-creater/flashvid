#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
from pathlib import Path

from flashvid_eval.role_separated_orchestration import (
    load_config,
    materialize_role_ablation,
)


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
        description="Materialize the frozen Dev-only 2^3 role ablation matrix."
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split", default="dev")
    args = parser.parse_args()
    config = load_config(args.config)
    plan = materialize_role_ablation(config, split=args.split)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    files: list[dict[str, str]] = []
    for run in plan["runs"]:
        path = args.output_dir / f"{run['id']}.json"
        _write_json(path, run)
        files.append(
            {
                "id": run["id"],
                "path": str(path.resolve()),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        )
    plan["files"] = files
    _write_json(args.output_dir / "matrix.json", plan)
    print(json.dumps(plan, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
