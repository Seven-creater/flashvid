#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import tempfile
from typing import Any, Mapping

from flashvid_eval.fast_hybrid_sft import validate_fast_hybrid_frame_files
from flashvid_eval.privacy import assert_deferred_result_public
from flashvid_eval.qwen_sft import (
    TRAIN_COUNTS,
    load_training_manifest,
    read_jsonl,
    sha256_file,
    validate_exported_sft_record,
)


DATASETS = ("lvbench", "lsdbench", "cgbench")
ERROR_FIELDS = ("error", "api_error", "frame_error", "parse_error")


def _sha256(value: Any, label: str) -> str:
    digest = str(value or "").strip().lower()
    if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        raise ValueError(f"{label} must be a SHA-256 digest")
    return digest


def _prediction(row: Mapping[str, Any]) -> str:
    value = row.get("final_prediction", row.get("prediction"))
    answer = str(value or "").strip().upper()
    if len(answer) != 1 or not "A" <= answer <= "H":
        raise ValueError("selected trajectory has no valid A-H prediction")
    return answer


def _atomic_write(path: Path, payload: Mapping[str, Any]) -> None:
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


def validate_fast_hybrid_sft_data(
    *,
    train_manifest_path: Path,
    selected_path: Path,
    sft_path: Path,
    summary_path: Path,
    config_sha256: str,
    expected_counts: Mapping[str, int] = TRAIN_COUNTS,
    minimum_total: int = 300,
    minimum_per_dataset: int = 80,
) -> dict[str, Any]:
    config_hash = _sha256(config_sha256, "config_sha256")
    manifest = load_training_manifest(
        train_manifest_path,
        expected_counts=expected_counts,
    )
    selected_rows = read_jsonl(selected_path)
    sft_rows = read_jsonl(sft_path)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if not isinstance(summary, Mapping):
        raise ValueError("Fast Hybrid SFT summary must be an object")

    selected: dict[str, dict[str, Any]] = {}
    selected_samples: set[tuple[str, str]] = set()
    selected_counts: Counter[str] = Counter()
    for index, row in enumerate(selected_rows):
        assert_deferred_result_public(row)
        dataset = str(row.get("dataset") or "").strip().lower()
        sample_id = str(row.get("sample_id") or "").strip()
        trajectory_id = str(row.get("trajectory_id") or "").strip()
        identity = (dataset, sample_id)
        if dataset not in DATASETS or not sample_id or not trajectory_id:
            raise ValueError(f"selected[{index}] has invalid dataset/sample/trajectory identity")
        if trajectory_id in selected or identity in selected_samples:
            raise ValueError(f"selected[{index}] has a duplicate trajectory or sample identity")
        if identity not in manifest.answers:
            raise ValueError(f"selected trajectory is outside Train600: {dataset}/{sample_id}")
        if row.get("scoring_deferred") is not True:
            raise ValueError(f"selected trajectory is not deferred: {trajectory_id}")
        if row.get("_selection_stable") is not True:
            raise ValueError(f"selected trajectory is not stable: {trajectory_id}")
        if row.get("_selection_confirmation_count") != 3:
            raise ValueError(f"selected trajectory lacks 3/3 confirmation: {trajectory_id}")
        if row.get("_compression_finalized") is not True:
            raise ValueError(f"selected trajectory is not compression-finalized: {trajectory_id}")
        if row.get("annotation_leak_check") != "passed":
            raise ValueError(f"selected trajectory failed leak audit: {trajectory_id}")
        if int(row.get("candidate_rerun") or 0) != 0:
            raise ValueError(f"selected trajectory reran its frozen candidate: {trajectory_id}")
        if any(row.get(field) for field in ERROR_FIELDS):
            raise ValueError(f"selected trajectory has an engineering error: {trajectory_id}")
        prediction = _prediction(row)
        answer = manifest.answers[identity]
        if prediction != answer:
            raise ValueError(f"selected trajectory is not positive: {trajectory_id}")
        if row.get("fallback_to_candidate") is True or row.get("fallback_used") is True:
            candidate = str(row.get("candidate_answer") or "").strip().upper()
            if candidate != answer:
                raise ValueError(f"selected trajectory used a wrong candidate fallback: {trajectory_id}")
        if row.get("manifest_sha256") != manifest.sha256:
            raise ValueError(f"selected trajectory Train600 manifest hash mismatch: {trajectory_id}")
        if row.get("train600_manifest_sha256") != manifest.sha256:
            raise ValueError(f"selected trajectory train600 hash mismatch: {trajectory_id}")
        _sha256(row.get("dataset_manifest_sha256"), "dataset_manifest_sha256")
        if row.get("config_sha256") != config_hash:
            raise ValueError(f"selected trajectory config hash mismatch: {trajectory_id}")
        validate_fast_hybrid_frame_files(row)
        selected[trajectory_id] = dict(row)
        selected_samples.add(identity)
        selected_counts[dataset] += 1

    if len(selected) < minimum_total:
        raise ValueError(
            f"SFT start gate requires at least {minimum_total} selected trajectories, "
            f"found {len(selected)}"
        )
    for dataset in DATASETS:
        if selected_counts[dataset] < minimum_per_dataset:
            raise ValueError(
                f"SFT start gate requires at least {minimum_per_dataset} {dataset} trajectories, "
                f"found {selected_counts[dataset]}"
            )

    episode_ids: set[str] = set()
    turns: dict[str, set[int]] = {}
    targets: dict[str, Counter[str]] = {}
    for index, row in enumerate(sft_rows):
        validate_exported_sft_record(row)
        metadata = row["metadata"]
        trajectory_id = str(metadata.get("trajectory_id") or "").strip()
        source = selected.get(trajectory_id)
        if source is None:
            raise ValueError(f"sft[{index}] has no selected source trajectory: {trajectory_id}")
        episode_id = str(metadata.get("episode_id") or "").strip()
        if not episode_id or episode_id in episode_ids:
            raise ValueError(f"sft[{index}] has a missing or duplicate episode_id")
        if not episode_id.startswith(f"{trajectory_id}#turn-"):
            raise ValueError(f"sft[{index}] episode_id is not bound to its trajectory")
        turn_index = metadata.get("turn_index")
        if isinstance(turn_index, bool) or not isinstance(turn_index, int) or turn_index < 0:
            raise ValueError(f"sft[{index}] has an invalid turn_index")
        trajectory_turns = turns.setdefault(trajectory_id, set())
        if turn_index in trajectory_turns:
            raise ValueError(f"sft[{index}] duplicates a trajectory turn_index")
        target = str(metadata.get("episode_target_type") or "")
        if target not in {"tool", "final"}:
            raise ValueError(f"sft[{index}] has a non-Fast-Hybrid episode target")
        for key in (
            "dataset",
            "sample_id",
            "manifest_sha256",
            "train600_manifest_sha256",
            "dataset_manifest_sha256",
            "config_sha256",
        ):
            if metadata.get(key) != source.get(key):
                raise ValueError(f"sft[{index}] metadata differs from selected source: {key}")
        episode_ids.add(episode_id)
        trajectory_turns.add(turn_index)
        targets.setdefault(trajectory_id, Counter())[target] += 1

    if set(targets) != set(selected):
        raise ValueError("selected and SFT trajectory coverage differs")
    for trajectory_id, counts in targets.items():
        if counts["final"] != 1:
            raise ValueError(
                f"Fast Hybrid trajectory must have exactly one final episode: {trajectory_id}"
            )
        if counts["tool"] < 1:
            raise ValueError(
                f"Fast Hybrid trajectory must have at least one tool episode: {trajectory_id}"
            )

    selected_hash = sha256_file(selected_path)
    sft_hash = sha256_file(sft_path)
    derived_summary = {
        "selected": len(selected_rows),
        "selected_by_dataset": dict(sorted(selected_counts.items())),
        "sft_records": len(sft_rows),
        "selected_sha256": selected_hash,
        "sft_sha256": sft_hash,
        "final_targets": sum(counts["final"] for counts in targets.values()),
        "tool_targets": sum(counts["tool"] for counts in targets.values()),
    }
    for key, expected in derived_summary.items():
        if summary.get(key) != expected:
            raise ValueError(f"Fast Hybrid SFT summary mismatch: {key}")

    return {
        "schema_version": 1,
        "status": "passed",
        "manifest_sha256": manifest.sha256,
        "config_sha256": config_hash,
        "selected_count": len(selected),
        "selected_by_dataset": dict(sorted(selected_counts.items())),
        "sft_record_count": len(sft_rows),
        "episode_target_counts": {
            "final": derived_summary["final_targets"],
            "tool": derived_summary["tool_targets"],
        },
        "selected_sha256": selected_hash,
        "sft_sha256": sft_hash,
        "summary_sha256": sha256_file(summary_path),
        "frame_contract_checked": True,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate finalized Fast Hybrid EVA SFT data.")
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--selected", type=Path, required=True)
    parser.add_argument("--sft-data", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--config-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--minimum-total", type=int, default=300)
    parser.add_argument("--minimum-per-dataset", type=int, default=80)
    args = parser.parse_args()
    try:
        report = validate_fast_hybrid_sft_data(
            train_manifest_path=args.train_manifest,
            selected_path=args.selected,
            sft_path=args.sft_data,
            summary_path=args.summary,
            config_sha256=args.config_sha256,
            minimum_total=args.minimum_total,
            minimum_per_dataset=args.minimum_per_dataset,
        )
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError) as error:
        report = {
            "schema_version": 1,
            "status": "failed",
            "error": f"{type(error).__name__}: {error}",
        }
    _atomic_write(args.output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
