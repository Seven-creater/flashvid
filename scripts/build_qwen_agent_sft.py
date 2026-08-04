#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

from flashvid_eval.qwen_sft import (
    ExpectedTrajectoryProvenance,
    SCHEMA_VERSION,
    TRAIN_COUNTS,
    build_sft_records,
    canonical_sha256,
    generate_counterfactual_specs,
    load_training_manifest,
    load_trajectory_input_bundle,
    read_jsonl,
    select_stable_correct_trajectories,
    sha256_file,
    source_fingerprint,
    trajectory_total_tokens,
    trajectory_visual_tokens,
)


_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def _atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
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


def _atomic_write_jsonl(path: Path, records: Sequence[Mapping[str, Any]]) -> None:
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
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    temporary.replace(path)


def _output_names(phase: str) -> tuple[str, ...]:
    if phase == "counterfactuals":
        return (
            "counterfactual_sources.jsonl",
            "counterfactual_specs.jsonl",
            "counterfactuals_summary.json",
        )
    return ("selected.jsonl", "sft.jsonl", "selection_summary.json")


def _summary_name(phase: str) -> str:
    return (
        "counterfactuals_summary.json"
        if phase == "counterfactuals"
        else "selection_summary.json"
    )


def _load_input_bundle(
    path: Path, config_sha256: str
) -> tuple[list[Path], ExpectedTrajectoryProvenance]:
    return load_trajectory_input_bundle(path, config_sha256)


def _resume_or_refuse(
    output_dir: Path,
    *,
    phase: str,
    fingerprint: str,
    resume: bool,
) -> dict[str, Any] | None:
    names = _output_names(phase)
    state_path = output_dir / f"{phase}_state.json"
    existing = [path for name in names for path in [output_dir / name] if path.exists()]
    if not resume:
        if state_path.exists() or existing:
            raise FileExistsError(
                f"{output_dir} already contains {phase} outputs; use --resume only for an identical source fingerprint"
            )
        return None
    if not state_path.is_file():
        if existing:
            raise RuntimeError("resume refused: outputs exist without a trusted state file")
        return None
    state = json.loads(state_path.read_text(encoding="utf-8"))
    if state.get("source_fingerprint") != fingerprint:
        raise RuntimeError("resume refused: source fingerprint changed")
    hashes = state.get("output_sha256")
    if not isinstance(hashes, dict):
        raise RuntimeError("resume refused: state has no output hashes")
    for name in names:
        path = output_dir / name
        if not path.is_file() or hashes.get(name) != sha256_file(path):
            raise RuntimeError(f"resume refused: output is missing or changed: {name}")
    return json.loads((output_dir / _summary_name(phase)).read_text(encoding="utf-8"))


def build(
    *,
    phase: str,
    train_manifest_path: Path,
    trajectory_paths: Sequence[Path],
    output_dir: Path,
    config_sha256: str,
    expected_provenance: ExpectedTrajectoryProvenance,
    resume: bool,
    expected_counts: Mapping[str, int] = TRAIN_COUNTS,
    expected_schedules: int = 12,
) -> dict[str, Any]:
    if phase not in {"counterfactuals", "select"}:
        raise ValueError("phase must be counterfactuals or select")
    if not _SHA256_RE.fullmatch(config_sha256):
        raise ValueError("config_sha256 must be 64 lowercase hexadecimal characters")
    if not trajectory_paths:
        raise ValueError("at least one trajectory file is required")
    manifest = load_training_manifest(train_manifest_path, expected_counts=expected_counts)
    trajectory_hashes = {
        str(path.resolve()): sha256_file(path) for path in sorted(trajectory_paths)
    }
    fingerprint = source_fingerprint(
        manifest_sha256=manifest.sha256,
        config_sha256=config_sha256,
        trajectory_files=trajectory_hashes,
        phase=phase,
        expected_provenance=expected_provenance.to_dict(),
    )
    resumed = _resume_or_refuse(
        output_dir,
        phase=phase,
        fingerprint=fingerprint,
        resume=resume,
    )
    if resumed is not None:
        resumed["resumed_without_changes"] = True
        return resumed

    trajectories = [
        record
        for path in trajectory_paths
        for record in read_jsonl(path)
    ]
    selection_input = trajectories
    if phase == "counterfactuals":
        selection_input = [
            row
            for row in trajectories
            if str(row.get("variant_id") or "base") in {"base", "rescue"}
        ]
    selection = select_stable_correct_trajectories(
        selection_input,
        manifest,
        config_sha256=config_sha256,
        expected_schedules=expected_schedules,
        expected_provenance=expected_provenance,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    if phase == "counterfactuals":
        specs = [
            spec
            for trajectory in selection.stable_candidates
            for spec in generate_counterfactual_specs(trajectory)
        ]
        _atomic_write_jsonl(
            output_dir / "counterfactual_sources.jsonl",
            selection.stable_candidates,
        )
        _atomic_write_jsonl(output_dir / "counterfactual_specs.jsonl", specs)
        output_counts = {
            "stable_source_trajectories": len(selection.stable_candidates),
            "counterfactual_specs": len(specs),
        }
    else:
        sft_records = [
            record
            for trajectory in selection.selected
            for record in build_sft_records(trajectory)
        ]
        _atomic_write_jsonl(output_dir / "selected.jsonl", selection.selected)
        _atomic_write_jsonl(output_dir / "sft.jsonl", sft_records)
        output_counts = {
            "selected_trajectories": len(selection.selected),
            "sft_records": len(sft_records),
            "mean_sft_records_per_trajectory": (
                len(sft_records) / len(selection.selected) if selection.selected else None
            ),
            "selected_total_tokens": sum(
                trajectory_total_tokens(row) for row in selection.selected
            ),
            "selected_visual_tokens": sum(
                trajectory_visual_tokens(row) for row in selection.selected
            ),
        }

    summary: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "phase": phase,
        "source_fingerprint": fingerprint,
        "manifest": {
            "path": str(train_manifest_path.resolve()),
            "sha256": manifest.sha256,
            "counts": manifest.counts,
        },
        "config_sha256": config_sha256,
        "expected_provenance": expected_provenance.to_dict(),
        "trajectory_files": trajectory_hashes,
        "trajectory_rows": len(trajectories),
        "stable_family_count": selection.stable_family_count,
        "rejected_families": selection.rejected_families,
        "no_stable_sample_count": len(selection.no_stable_sample_ids),
        "no_stable_sample_ids": list(selection.no_stable_sample_ids),
        "outputs": output_counts,
        "resumed_without_changes": False,
    }
    _atomic_write_json(output_dir / _summary_name(phase), summary)
    output_hashes = {
        name: sha256_file(output_dir / name) for name in _output_names(phase)
    }
    _atomic_write_json(
        output_dir / f"{phase}_state.json",
        {
            "schema_version": SCHEMA_VERSION,
            "phase": phase,
            "source_fingerprint": fingerprint,
            "output_sha256": output_hashes,
        },
    )
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Build Qwen3.5-9B Agent counterfactual schedules or stable, "
            "ground-truth-filtered SFT data from the frozen Train600 manifest."
        )
    )
    parser.add_argument("--phase", choices=("counterfactuals", "select"), required=True)
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--trajectories", type=Path, nargs="+")
    parser.add_argument(
        "--input-bundle",
        type=Path,
        help="Frozen trajectory/provenance bundle produced by collect_qwen_sft_inputs.py.",
    )
    parser.add_argument("--config-sha256", required=True)
    parser.add_argument("--expected-model", default="Qwen3.5-9B")
    parser.add_argument("--expected-model-artifact-sha256")
    parser.add_argument(
        "--expected-dataset-manifest-sha256",
        action="append",
        dest="expected_dataset_manifest_sha256s",
    )
    parser.add_argument(
        "--expected-agent-config-sha256",
        action="append",
        dest="expected_agent_config_sha256s",
    )
    parser.add_argument(
        "--expected-runner-fingerprint",
        action="append",
        dest="expected_runner_fingerprints",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    try:
        if (args.input_bundle is None) == (args.trajectories is None):
            raise ValueError("provide exactly one of --input-bundle or --trajectories")
        if args.input_bundle is not None:
            trajectory_paths, expected_provenance = _load_input_bundle(
                args.input_bundle, args.config_sha256
            )
        else:
            if not (
                args.expected_model_artifact_sha256
                and args.expected_dataset_manifest_sha256s
                and args.expected_agent_config_sha256s
                and args.expected_runner_fingerprints
            ):
                raise ValueError("manual trajectories require all expected provenance flags")
            trajectory_paths = list(args.trajectories or [])
            expected_provenance = ExpectedTrajectoryProvenance(
                model=args.expected_model,
                model_artifact_sha256=args.expected_model_artifact_sha256,
                dataset_manifest_sha256s=frozenset(
                    args.expected_dataset_manifest_sha256s
                ),
                agent_config_sha256s=frozenset(args.expected_agent_config_sha256s),
                runner_fingerprints=frozenset(args.expected_runner_fingerprints),
            )
        summary = build(
            phase=args.phase,
            train_manifest_path=args.train_manifest,
            trajectory_paths=trajectory_paths,
            output_dir=args.output_dir,
            config_sha256=args.config_sha256,
            expected_provenance=expected_provenance,
            resume=args.resume,
        )
    except (OSError, RuntimeError, ValueError, TypeError, KeyError, json.JSONDecodeError) as error:
        print(json.dumps({"status": "failed", "error": str(error)}, ensure_ascii=False, indent=2))
        return 1
    print(json.dumps({"status": "passed", **summary}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
