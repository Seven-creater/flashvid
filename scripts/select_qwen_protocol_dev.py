#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from flashvid_eval.qwen_dev_selection import (
    build_dev_selection_report,
    canonical_sha256,
    load_dev_runs,
    load_protocol_smoke_rejection,
    write_frozen_json,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Freeze the q4/q9 protocol choices after the protocol-audit Dev150 matrix."
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-plan", type=Path, action="append", required=True)
    parser.add_argument(
        "--reject-q4-think-from-smoke",
        type=Path,
        help=(
            "Allow q4 think to be explicitly rejected from a protocol_smoke plan "
            "whose engineering failure rate exceeds 1%%."
        ),
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("experiment config must be an object")
    config_hash = canonical_sha256(config)
    runs = load_dev_runs(args.run_plan, config_hash)
    if not runs or any(run.phase != "protocol_audit" for run in runs):
        raise ValueError("protocol selector only accepts protocol_audit plans")
    protocol_rejections = {}
    if args.reject_q4_think_from_smoke is not None:
        evidence = load_protocol_smoke_rejection(
            args.reject_q4_think_from_smoke,
            config_hash,
            model_key="q4",
            protocol="think",
        )
        protocol_rejections = {"q4": {"think": evidence}}
    full_report = build_dev_selection_report(
        config,
        runs,
        protocol_rejections=protocol_rejections,
    )
    selection = full_report["protocol_selection"]
    if any(not (selection.get(key) or {}).get("protocol") for key in ("q9", "q4")):
        raise RuntimeError("one or both models have no eligible protocol")
    protocol_blockers = [
        item
        for item in full_report["blocking_errors"]
        if item.startswith(("missing_protocol", "incomplete_protocol", "no_eligible_protocol"))
    ]
    if protocol_blockers:
        raise RuntimeError(f"protocol audit is incomplete: {protocol_blockers}")
    report = {
        "schema_version": 1,
        "status": "passed",
        "experiment_config_sha256": config_hash,
        "selection": selection,
        "source_run_plans": full_report["source_run_plans"],
        "policy": full_report["policy"],
    }
    if protocol_rejections:
        report["protocol_rejections"] = protocol_rejections
    report["selection_state_sha256"] = canonical_sha256(report)
    digest = write_frozen_json(args.output, report)
    print(
        json.dumps(
            {
                "status": "passed",
                "q9_protocol": selection["q9"]["protocol"],
                "q4_protocol": selection["q4"]["protocol"],
                "output": str(args.output.resolve()),
                "sha256": digest,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
