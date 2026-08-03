#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from flashvid_eval.qwen_dev_selection import (
    build_dev_selection_report,
    build_frozen_winner,
    canonical_sha256,
    load_dev_runs,
    write_frozen_json,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Summarize frozen Qwen Dev runs and freeze a promotion-gated winner."
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-plan", type=Path, action="append", default=[])
    parser.add_argument(
        "--run-plan-dir",
        type=Path,
        action="append",
        default=[],
        help="Recursively include protocol_audit/direct_dev/agent_dev plan JSON files.",
    )
    parser.add_argument("--teacher-model-key", choices=("q9", "q4"), default="q9")
    parser.add_argument("--summary-output", type=Path, required=True)
    parser.add_argument("--winner-output", type=Path, required=True)
    args = parser.parse_args()

    plan_paths = list(args.run_plan)
    for directory in args.run_plan_dir:
        if not directory.is_dir():
            raise FileNotFoundError(directory)
        for prefix in ("protocol_audit", "direct_dev", "agent_dev"):
            plan_paths.extend(directory.rglob(f"{prefix}*.json"))
    unique_plans = sorted({path.resolve() for path in plan_paths})
    if not unique_plans:
        parser.error("at least one --run-plan or --run-plan-dir is required")

    config = json.loads(args.config.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("experiment config must be a JSON object")
    config_hash = canonical_sha256(config)
    runs = load_dev_runs(unique_plans, config_hash)
    report = build_dev_selection_report(
        config,
        runs,
        teacher_model_key=args.teacher_model_key,
    )
    report_sha = write_frozen_json(args.summary_output, report)
    if report["status"] != "passed":
        print(
            json.dumps(
                {
                    "status": "blocked",
                    "summary": str(args.summary_output.resolve()),
                    "summary_sha256": report_sha,
                    "blocking_errors": report["blocking_errors"],
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        raise SystemExit(2)
    winner = build_frozen_winner(
        report,
        report_path=args.summary_output,
        report_sha256=report_sha,
    )
    winner_sha = write_frozen_json(args.winner_output, winner)
    print(
        json.dumps(
            {
                "status": "passed",
                "summary": str(args.summary_output.resolve()),
                "summary_sha256": report_sha,
                "winner": str(args.winner_output.resolve()),
                "winner_sha256": winner_sha,
                "winner_id": winner["winner_id"],
                "agent_config_sha256": winner["agent_config"]["sha256"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
