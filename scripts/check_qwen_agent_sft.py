#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

from flashvid_eval.qwen_sft import (
    SCHEMA_VERSION,
    TRAIN_COUNTS,
    checkpoint_gate,
    load_training_manifest,
    read_jsonl,
    sha256_file,
    trajectory_rejection_reason,
    validate_exported_sft_record,
    validate_trajectory_schema,
)


def _atomic_write(path: Path, value: Mapping[str, Any]) -> None:
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
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    temporary.replace(path)


def validate_data(
    *,
    train_manifest_path: Path,
    selected_path: Path,
    sft_path: Path,
    config_sha256: str,
    state_path: Path | None = None,
    expected_counts: Mapping[str, int] = TRAIN_COUNTS,
) -> dict[str, Any]:
    manifest = load_training_manifest(
        train_manifest_path,
        expected_counts=expected_counts,
    )
    selected = read_jsonl(selected_path)
    sft = read_jsonl(sft_path)
    if not selected:
        raise ValueError("selected SFT trajectories are empty")

    selected_by_id: dict[str, dict[str, Any]] = {}
    for row in selected:
        validate_trajectory_schema(
            row,
            manifest_sha256=manifest.sha256,
            config_sha256=config_sha256,
        )
        trajectory_id = str(row.get("trajectory_id") or "")
        if trajectory_id in selected_by_id:
            raise ValueError(f"duplicate selected trajectory_id: {trajectory_id}")
        if row.get("_selection_stable") is not True:
            raise ValueError(f"selected trajectory is not marked stable: {trajectory_id}")
        if row.get("_selection_confirmation_count") != 3:
            raise ValueError(f"selected trajectory lacks 3/3 confirmation: {trajectory_id}")
        identity = (str(row["dataset"]).lower(), str(row["sample_id"]))
        if identity not in manifest.answers:
            raise ValueError(f"selected trajectory is outside Train600: {trajectory_id}")
        rejection = trajectory_rejection_reason(row, manifest.answers[identity])
        if rejection is not None:
            raise ValueError(f"selected trajectory is not eligible ({rejection}): {trajectory_id}")
        selected_by_id[trajectory_id] = row

    sft_by_id: dict[str, dict[str, Any]] = {}
    sft_trajectory_ids: set[str] = set()
    episode_target_counts: dict[str, int] = {}
    trajectory_targets: dict[str, list[str]] = {}
    for row in sft:
        validate_exported_sft_record(row)
        metadata = row["metadata"]
        trajectory_id = str(metadata.get("trajectory_id") or "")
        record_id = str(metadata.get("episode_id") or trajectory_id)
        if not trajectory_id or not record_id:
            raise ValueError("SFT record lacks trajectory/episode identity")
        if record_id in sft_by_id:
            raise ValueError(f"duplicate SFT record identity: {record_id}")
        if metadata.get("manifest_sha256") != manifest.sha256:
            raise ValueError(f"SFT manifest hash mismatch: {trajectory_id}")
        if metadata.get("config_sha256") != config_sha256:
            raise ValueError(f"SFT config hash mismatch: {trajectory_id}")
        target = metadata.get("episode_target_type")
        if target is not None:
            target_name = str(target)
            episode_target_counts[target_name] = episode_target_counts.get(target_name, 0) + 1
        assistant_targets = metadata.get("assistant_target_types")
        if not isinstance(assistant_targets, list):
            raise ValueError(f"SFT record lacks assistant target metadata: {record_id}")
        trajectory_targets.setdefault(trajectory_id, []).extend(
            str(item) for item in assistant_targets
        )
        sft_by_id[record_id] = row
        sft_trajectory_ids.add(trajectory_id)
    if set(selected_by_id) != sft_trajectory_ids:
        raise ValueError("selected and SFT trajectory coverage differs")
    for trajectory_id, targets in trajectory_targets.items():
        if targets.count("final") != 1:
            raise ValueError(
                f"SFT trajectory must contain exactly one final target: {trajectory_id}"
            )
        if not any(target in {"plan", "tool", "memory", "stop"} for target in targets):
            raise ValueError(
                f"SFT trajectory has no trainable agent action: {trajectory_id}"
            )

    state: dict[str, Any] | None = None
    if state_path is not None:
        state = json.loads(state_path.read_text(encoding="utf-8"))
        output_hashes = state.get("output_sha256")
        if not isinstance(output_hashes, Mapping):
            raise ValueError("selection state has no output hashes")
        if output_hashes.get(selected_path.name) != sha256_file(selected_path):
            raise ValueError("selected output hash differs from build state")
        if output_hashes.get(sft_path.name) != sha256_file(sft_path):
            raise ValueError("SFT output hash differs from build state")

    datasets: dict[str, int] = {}
    for row in selected:
        dataset = str(row["dataset"])
        datasets[dataset] = datasets.get(dataset, 0) + 1
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "passed",
        "manifest_sha256": manifest.sha256,
        "config_sha256": config_sha256,
        "selected_count": len(selected),
        "sft_count": len(sft),
        "sft_records_per_selected_trajectory": len(sft) / len(selected),
        "episode_target_counts": dict(sorted(episode_target_counts.items())),
        "selected_by_dataset": dict(sorted(datasets.items())),
        "selected_sha256": sha256_file(selected_path),
        "sft_sha256": sha256_file(sft_path),
        "state_checked": state is not None,
    }


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Validate Qwen Agent SFT data or enforce the strict checkpoint gate."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    data = subparsers.add_parser("data")
    data.add_argument("--train-manifest", type=Path, required=True)
    data.add_argument("--selected", type=Path, required=True)
    data.add_argument("--sft-data", type=Path, required=True)
    data.add_argument("--config-sha256", required=True)
    data.add_argument("--state", type=Path)
    data.add_argument("--output", type=Path, required=True)

    checkpoint = subparsers.add_parser("checkpoint")
    checkpoint.add_argument("--teacher-summary", type=Path, required=True)
    checkpoint.add_argument("--checkpoint-summary", type=Path, required=True)
    checkpoint.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)

    try:
        if args.command == "data":
            report = validate_data(
                train_manifest_path=args.train_manifest,
                selected_path=args.selected,
                sft_path=args.sft_data,
                config_sha256=args.config_sha256,
                state_path=args.state,
            )
        else:
            report = {
                "schema_version": SCHEMA_VERSION,
                "status": "passed",
                **checkpoint_gate(
                    _load_json(args.teacher_summary),
                    _load_json(args.checkpoint_summary),
                ),
            }
            if not report["passed"]:
                report["status"] = "failed"
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError) as error:
        report = {
            "schema_version": SCHEMA_VERSION,
            "status": "failed",
            "passed": False,
            "error": f"{type(error).__name__}: {error}",
        }
    _atomic_write(args.output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report.get("status") == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
