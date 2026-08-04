#!/usr/bin/env python3
"""Freeze, resume, score, rescue and compress Fast Hybrid Train600 jobs."""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from flashvid_eval.fast_hybrid_trajectory_control import (
    BASE_PLANNER_SEEDS,
    BASE_VISUAL_BUDGETS,
    build_base_run_specs,
    build_rescue_run_specs,
    controller_fingerprint,
    generate_compression_replay_specs,
    pending_run_specs,
    select_lowest_cost_positives,
    validate_prejudge_coverage,
)
from flashvid_eval.qwen_sft import load_training_manifest, read_jsonl, sha256_file


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False
    ) as handle:
        temporary = Path(handle.name)
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(path)


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False
    ) as handle:
        temporary = Path(handle.name)
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    temporary.replace(path)


def _dataset_hashes(values: Sequence[str]) -> dict[str, str]:
    result: dict[str, str] = {}
    for value in values:
        if "=" not in value:
            raise ValueError("dataset manifest hashes use DATASET=SHA256")
        dataset, digest = value.split("=", 1)
        dataset = dataset.strip().lower()
        if dataset in result:
            raise ValueError(f"duplicate dataset manifest hash: {dataset}")
        result[dataset] = digest.strip().lower()
    if set(result) != {"lvbench", "lsdbench", "cgbench"}:
        raise ValueError("dataset manifest hashes must cover all three datasets")
    return result


def _load_many(paths: Sequence[Path]) -> list[dict[str, Any]]:
    return [row for path in paths for row in read_jsonl(path)]


def _phase_rows(rows: Iterable[Mapping[str, Any]], phase: str) -> list[dict[str, Any]]:
    return [dict(row) for row in rows if str(row.get("phase") or "base") == phase]


def run(args: argparse.Namespace) -> dict[str, Any]:
    train_rows = read_jsonl(args.train600)
    manifest = load_training_manifest(args.train600)
    if (
        args.expected_train600_sha256
        and manifest.sha256 != args.expected_train600_sha256
    ):
        raise RuntimeError("Train600 SHA-256 mismatch")
    dataset_hashes = _dataset_hashes(args.dataset_manifest_sha256)
    fingerprint = controller_fingerprint(
        manifest_sha256=manifest.sha256,
        config_sha256=args.config_sha256,
        budgets=BASE_VISUAL_BUDGETS,
        planner_seeds=BASE_PLANNER_SEEDS,
        dataset_manifest_sha256s=dataset_hashes,
    )
    base_specs = build_base_run_specs(
        train_rows,
        manifest_sha256=manifest.sha256,
        config_sha256=args.config_sha256,
        dataset_manifest_sha256s=dataset_hashes,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.phase == "plan-base":
        completed = (
            _phase_rows(_load_many(args.completed), "base") if args.completed else []
        )
        pending = pending_run_specs(
            base_specs,
            completed,
            expected_controller_fingerprint=fingerprint,
        )
        _write_jsonl(args.output_dir / "base_plan.jsonl", base_specs)
        _write_jsonl(args.output_dir / "base_pending.jsonl", pending)
        summary = {
            "phase": args.phase,
            "controller_fingerprint": fingerprint,
            "train600_sha256": manifest.sha256,
            "planned": len(base_specs),
            "completed": len(base_specs) - len(pending),
            "pending": len(pending),
        }
    else:
        if not args.trajectories:
            raise ValueError(f"{args.phase} requires --trajectories")
        trajectories = _load_many(args.trajectories)
        base_rows = _phase_rows(trajectories, "base")
        if args.prejudge_index:
            index_rows = _load_many(args.prejudge_index)
            validate_prejudge_coverage(
                base_specs,
                _phase_rows(index_rows, "base"),
                base_rows,
            )
        else:
            base_pending = pending_run_specs(
                base_specs,
                base_rows,
                expected_controller_fingerprint=fingerprint,
            )
            if base_pending:
                raise RuntimeError(
                    f"base matrix is incomplete: {len(base_pending)} jobs missing"
                )
        base_selection = select_lowest_cost_positives(base_rows, manifest.answers)
        rescue_specs = build_rescue_run_specs(
            train_rows,
            no_positive_sample_ids=base_selection.no_positive_sample_ids,
            manifest_sha256=manifest.sha256,
            config_sha256=args.config_sha256,
            controller_sha256=fingerprint,
            dataset_manifest_sha256s=dataset_hashes,
        )
        if args.phase == "plan-rescue":
            rescue_rows = (
                _phase_rows(_load_many(args.completed), "rescue")
                if args.completed
                else []
            )
            pending = pending_run_specs(
                rescue_specs,
                rescue_rows,
                expected_controller_fingerprint=fingerprint,
            )
            _write_jsonl(args.output_dir / "rescue_plan.jsonl", rescue_specs)
            _write_jsonl(args.output_dir / "rescue_pending.jsonl", pending)
            summary = {
                "phase": args.phase,
                "controller_fingerprint": fingerprint,
                "no_positive_after_base": len(base_selection.no_positive_sample_ids),
                "planned": len(rescue_specs),
                "completed": len(rescue_specs) - len(pending),
                "pending": len(pending),
            }
        else:
            rescue_rows = _phase_rows(trajectories, "rescue")
            if args.prejudge_index:
                validate_prejudge_coverage(
                    rescue_specs,
                    _phase_rows(index_rows, "rescue"),
                    rescue_rows,
                )
            else:
                rescue_pending = pending_run_specs(
                    rescue_specs,
                    rescue_rows,
                    expected_controller_fingerprint=fingerprint,
                )
                if rescue_pending:
                    raise RuntimeError(
                        f"required rescue matrix is incomplete: {len(rescue_pending)} jobs missing"
                    )
            selection = select_lowest_cost_positives(
                [*base_rows, *rescue_rows], manifest.answers
            )
            compression = [
                spec
                for trajectory in selection.selected
                for spec in generate_compression_replay_specs(trajectory)
            ]
            _write_jsonl(args.output_dir / "selected.jsonl", selection.selected)
            _write_jsonl(
                args.output_dir / "compression_replay_specs.jsonl", compression
            )
            summary = {
                "phase": args.phase,
                "controller_fingerprint": fingerprint,
                "train600_sha256": manifest.sha256,
                "trajectory_source_sha256": {
                    str(path.resolve()): sha256_file(path) for path in args.trajectories
                },
                "selected": len(selection.selected),
                "no_positive": len(selection.no_positive_sample_ids),
                "no_positive_sample_ids": list(selection.no_positive_sample_ids),
                "rejected": selection.rejected,
                "compression_replay_specs": len(compression),
                "sft_start_gate": selection.gate,
            }
    _write_json(args.output_dir / f"{args.phase}_summary.json", summary)
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phase", choices=("plan-base", "plan-rescue", "select"), required=True
    )
    parser.add_argument("--train600", type=Path, required=True)
    parser.add_argument("--expected-train600-sha256")
    parser.add_argument("--config-sha256", required=True)
    parser.add_argument(
        "--dataset-manifest-sha256",
        action="append",
        default=[],
        metavar="DATASET=SHA256",
        required=True,
    )
    parser.add_argument("--trajectories", type=Path, nargs="*")
    parser.add_argument("--completed", type=Path, nargs="*", default=[])
    parser.add_argument(
        "--prejudge-index",
        type=Path,
        nargs="*",
        default=[],
        help="Offline completion indexes proving why non-Judged runs were rejected.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        summary = run(args)
    except (OSError, ValueError, TypeError, KeyError, RuntimeError) as error:
        print(
            json.dumps(
                {"status": "failed", "error": f"{type(error).__name__}: {error}"}
            )
        )
        return 1
    print(json.dumps({"status": "passed", **summary}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
