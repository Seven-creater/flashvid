#!/usr/bin/env python3
"""Advance or finalize the dependency-gated Fast Hybrid compression DAGs."""

from __future__ import annotations

import argparse
import json
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from flashvid_eval.fast_hybrid_trajectory_control import finalize_compression_dags
from flashvid_eval.qwen_sft import load_training_manifest, read_jsonl, sha256_file


def _atomic_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False
    ) as handle:
        temporary = Path(handle.name)
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False) + "\n")
    temporary.replace(path)


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False
    ) as handle:
        temporary = Path(handle.name)
        json.dump(dict(value), handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    temporary.replace(path)


def run(args: argparse.Namespace) -> dict[str, Any]:
    manifest = load_training_manifest(args.train600)
    if (
        args.expected_train600_sha256
        and manifest.sha256 != args.expected_train600_sha256
    ):
        raise RuntimeError("Train600 SHA-256 mismatch")
    base_selected = read_jsonl(args.base_selected)
    specs = read_jsonl(args.specs)
    replay_rows = [
        row for path in args.replay_results for row in read_jsonl(path)
    ]
    finalized = finalize_compression_dags(
        base_selected, specs, replay_rows, manifest.answers
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    _atomic_jsonl(args.output_dir / "compression_ready.jsonl", finalized.ready)
    _atomic_jsonl(
        args.output_dir / "compression_representatives.jsonl",
        finalized.representatives,
    )
    _atomic_jsonl(
        args.output_dir / "compression_outcomes.jsonl",
        (
            {
                "counterfactual_fingerprint": fingerprint,
                "status": status,
                "rejection_reason": finalized.rejection_reasons.get(fingerprint),
            }
            for fingerprint, status in sorted(finalized.outcomes.items())
        ),
    )
    _atomic_jsonl(
        args.output_dir / "selected_pruned.jsonl", finalized.selected_pruned
    )
    counts = Counter(finalized.outcomes.values())
    summary = {
        "schema_version": 1,
        "train600_sha256": manifest.sha256,
        "base_selected_sha256": sha256_file(args.base_selected),
        "compression_specs_sha256": sha256_file(args.specs),
        "replay_result_sha256s": {
            str(path.resolve()): sha256_file(path) for path in args.replay_results
        },
        "specs": len(specs),
        "replay_rows": len(replay_rows),
        "decided_nodes": len(finalized.outcomes),
        "passed_nodes": counts.get("passed", 0),
        "failed_nodes": counts.get("failed", 0),
        "incomplete_nodes": len(finalized.incomplete_fingerprints),
        "incomplete_fingerprints": list(finalized.incomplete_fingerprints),
        "ready_nodes": len(finalized.ready),
        "compression_complete": finalized.complete,
        "selected_pruned": len(finalized.selected_pruned),
        "sft_start_gate": finalized.gate,
    }
    _atomic_json(args.output_dir / "compression_summary.json", summary)
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train600", type=Path, required=True)
    parser.add_argument("--expected-train600-sha256")
    parser.add_argument("--base-selected", type=Path, required=True)
    parser.add_argument("--specs", type=Path, required=True)
    parser.add_argument("--replay-results", type=Path, nargs="*", default=[])
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        report = run(args)
    except (OSError, ValueError, TypeError, KeyError, RuntimeError) as error:
        print(
            json.dumps(
                {"status": "failed", "error": f"{type(error).__name__}: {error}"},
                ensure_ascii=False,
            )
        )
        return 1
    print(json.dumps({"status": "passed", **report}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
