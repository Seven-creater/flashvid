#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from flashvid_eval.qwen_checkpoint_selection import (
    freeze_sft_winner,
    select_sft_checkpoint,
    write_frozen_json,
)
from flashvid_eval.qwen_dev_selection import canonical_sha256


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Apply the frozen accuracy-plus-30%-Token gate to three 9B SFT checkpoints."
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--teacher-run-plan", type=Path, required=True)
    parser.add_argument(
        "--checkpoint-run-plan", type=Path, action="append", required=True
    )
    parser.add_argument("--summary-output", type=Path, required=True)
    parser.add_argument("--winner-output", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("experiment config must be an object")
    report = select_sft_checkpoint(
        config_sha256=canonical_sha256(config),
        teacher_plan=args.teacher_run_plan,
        checkpoint_plans=args.checkpoint_run_plan,
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
    winner = freeze_sft_winner(
        report,
        report_path=args.summary_output,
        report_sha256=report_sha,
    )
    winner_sha = write_frozen_json(args.winner_output, winner)
    print(
        json.dumps(
            {
                "status": "passed",
                "checkpoint_id": winner["checkpoint_id"],
                "epoch": winner["epoch"],
                "winner": str(args.winner_output.resolve()),
                "winner_sha256": winner_sha,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
