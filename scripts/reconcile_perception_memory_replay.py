#!/usr/bin/env python3
"""Prepare and reconcile an immutable Perception-Memory replay repair lane."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from flashvid_eval.perception_memory_repair import (
    merge_replay_results,
    prepare_repair_scope,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser(
        "prepare", help="freeze the failed subset of one complete base replay"
    )
    prepare.add_argument("--source", type=Path, nargs="+", required=True)
    prepare.add_argument("--base-results", type=Path, nargs="+", required=True)
    prepare.add_argument("--output-dir", type=Path, required=True)
    prepare.add_argument(
        "--expected-rows",
        type=int,
        default=2917,
        help="formal Train600 replay scope (default: 2917)",
    )
    prepare.add_argument("--expected-explicit-time-rows", type=int, default=37)
    prepare.add_argument("--expected-explicit-time-samples", type=int, default=6)

    merge = subparsers.add_parser(
        "merge", help="merge successful repairs without changing the base replay"
    )
    merge.add_argument("--source", type=Path, nargs="+", required=True)
    merge.add_argument("--base-results", type=Path, nargs="+", required=True)
    merge.add_argument("--frozen-scope", type=Path, required=True)
    merge.add_argument("--replacement-results", type=Path, nargs="+", required=True)
    merge.add_argument("--output-dir", type=Path, required=True)
    return parser


def run(args: argparse.Namespace) -> dict:
    if args.command == "prepare":
        return prepare_repair_scope(
            source_paths=args.source,
            base_results_paths=args.base_results,
            output_dir=args.output_dir,
            expected_rows=args.expected_rows,
            expected_explicit_time_rows=args.expected_explicit_time_rows,
            expected_explicit_time_samples=args.expected_explicit_time_samples,
        )
    if args.command == "merge":
        return merge_replay_results(
            source_paths=args.source,
            base_results_paths=args.base_results,
            frozen_scope_path=args.frozen_scope,
            replacement_result_paths=args.replacement_results,
            output_dir=args.output_dir,
        )
    raise ValueError(f"unsupported command: {args.command}")


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        summary = run(args)
    except (OSError, ValueError, TypeError, KeyError, RuntimeError) as error:
        print(
            json.dumps(
                {"status": "failed", "error": f"{type(error).__name__}: {error}"},
                ensure_ascii=False,
            )
        )
        return 1
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
