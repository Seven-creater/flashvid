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


LEDGER_NAME = "compression_ledger.json"


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


def _result_hashes(paths: Iterable[Path]) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for path in paths:
        resolved = str(path.resolve())
        if resolved in hashes:
            raise ValueError(f"duplicate replay result path: {resolved}")
        hashes[resolved] = sha256_file(path)
    return dict(sorted(hashes.items()))


def _validate_monotonic_ledger(
    previous: Mapping[str, Any] | None,
    *,
    train600_sha256: str,
    base_selected_sha256: str,
    compression_specs_sha256: str,
    replay_result_sha256s: Mapping[str, str],
    outcomes: Mapping[str, str],
    rejection_reasons: Mapping[str, str],
) -> None:
    if previous is None:
        return
    frozen = {
        "train600_sha256": train600_sha256,
        "base_selected_sha256": base_selected_sha256,
        "compression_specs_sha256": compression_specs_sha256,
    }
    for key, expected in frozen.items():
        if previous.get(key) != expected:
            raise RuntimeError(f"compression ledger frozen input changed: {key}")
    old_files = previous.get("replay_result_sha256s")
    if not isinstance(old_files, Mapping):
        raise RuntimeError("compression ledger has invalid replay_result_sha256s")
    if not set(old_files).issubset(replay_result_sha256s):
        raise RuntimeError("replay result set is not a monotonic superset of the ledger")
    for path, digest in old_files.items():
        if replay_result_sha256s.get(path) != digest:
            raise RuntimeError(f"previous replay result changed: {path}")
    old_outcomes = previous.get("outcomes")
    if not isinstance(old_outcomes, Mapping):
        raise RuntimeError("compression ledger has invalid outcomes")
    for fingerprint, status in old_outcomes.items():
        if outcomes.get(fingerprint) != status:
            raise RuntimeError(
                f"compression outcome is not monotonic: {fingerprint}"
            )
    old_reasons = previous.get("rejection_reasons")
    if not isinstance(old_reasons, Mapping):
        raise RuntimeError("compression ledger has invalid rejection_reasons")
    for fingerprint, reason in old_reasons.items():
        if rejection_reasons.get(fingerprint) != reason:
            raise RuntimeError(
                f"compression rejection reason changed: {fingerprint}"
            )


def run(args: argparse.Namespace) -> dict[str, Any]:
    manifest = load_training_manifest(args.train600)
    if (
        args.expected_train600_sha256
        and manifest.sha256 != args.expected_train600_sha256
    ):
        raise RuntimeError("Train600 SHA-256 mismatch")
    base_selected = read_jsonl(args.base_selected)
    specs = read_jsonl(args.specs)
    replay_hashes = _result_hashes(args.replay_results)
    replay_rows = [
        row for path in args.replay_results for row in read_jsonl(path)
    ]
    finalized = finalize_compression_dags(
        base_selected, specs, replay_rows, manifest.answers
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    ledger_path = args.output_dir / LEDGER_NAME
    previous_ledger = (
        json.loads(ledger_path.read_text(encoding="utf-8"))
        if ledger_path.is_file()
        else None
    )
    base_sha = sha256_file(args.base_selected)
    specs_sha = sha256_file(args.specs)
    _validate_monotonic_ledger(
        previous_ledger,
        train600_sha256=manifest.sha256,
        base_selected_sha256=base_sha,
        compression_specs_sha256=specs_sha,
        replay_result_sha256s=replay_hashes,
        outcomes=finalized.outcomes,
        rejection_reasons=finalized.rejection_reasons,
    )
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
        "base_selected_sha256": base_sha,
        "compression_specs_sha256": specs_sha,
        "replay_result_sha256s": replay_hashes,
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
    _atomic_json(
        ledger_path,
        {
            "schema_version": 1,
            "train600_sha256": manifest.sha256,
            "base_selected_sha256": base_sha,
            "compression_specs_sha256": specs_sha,
            "replay_result_sha256s": replay_hashes,
            "outcomes": dict(sorted(finalized.outcomes.items())),
            "rejection_reasons": dict(
                sorted(finalized.rejection_reasons.items())
            ),
        },
    )
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
