#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from flashvid_eval.qwen_plan_index import (
    build_selection_plan_index,
    canonical_sha256,
    write_frozen_index,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Freeze the exact run-plan allowlist used for Qwen Dev selection."
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-plan", type=Path, action="append", required=True)
    parser.add_argument("--reject-q4-think-from-smoke", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    config = json.loads(args.config.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("experiment config must be a JSON object")
    config_hash = canonical_sha256(config)
    payload = build_selection_plan_index(
        config_path=args.config,
        config_sha256=config_hash,
        run_plan_paths=args.run_plan,
        q4_think_smoke_rejection=args.reject_q4_think_from_smoke,
    )
    digest = write_frozen_index(args.output, payload)
    print(
        json.dumps(
            {
                "selection_plan_index": str(args.output.resolve()),
                "sha256": digest,
                "run_plan_count": len(payload["run_plans"]),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
