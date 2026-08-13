#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
from pathlib import Path

from flashvid_eval.role_separated_orchestration import (
    freeze_role_ablation_dev30,
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
    parser.add_argument(
        "--derive-dev10",
        action="store_true",
        help="Verify each frozen Dev50 SHA and atomically derive its first 10 rows.",
    )
    args = parser.parse_args()
    config = load_config(args.config)
    plan = materialize_role_ablation(config, split=args.split)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    derived_manifests = (
        freeze_role_ablation_dev30(
            config, output_dir=args.output_dir / "frozen"
        )
        if args.derive_dev10
        else None
    )
    files: list[dict[str, str]] = []
    for run in plan["runs"]:
        run_dir = args.output_dir / run["id"]
        path = run_dir / "run.json"
        role_config_path = run_dir / "role_config.json"
        _write_json(role_config_path, run["pm_role_config"])
        materialized_run = dict(run)
        materialized_run["pm_role_config_sha256"] = hashlib.sha256(
            role_config_path.read_bytes()
        ).hexdigest()
        if derived_manifests is not None:
            materialized_run["manifests"] = derived_manifests
        materialized_run["pm_role_config_path"] = str(role_config_path.resolve())
        _write_json(path, materialized_run)
        files.append(
            {
                "id": run["id"],
                "path": str(path.resolve()),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "role_config_path": str(role_config_path.resolve()),
                "role_config_sha256": hashlib.sha256(
                    role_config_path.read_bytes()
                ).hexdigest(),
            }
        )
    plan["files"] = files
    _write_json(args.output_dir / "matrix.json", plan)
    print(json.dumps(plan, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
