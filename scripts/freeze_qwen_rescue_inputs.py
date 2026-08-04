#!/usr/bin/env python3
"""Freeze dense A4 rescue inputs for Train600 items with no stable base trace."""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

from flashvid_eval.qwen_sft import (
    canonical_sha256,
    load_training_manifest,
    load_trajectory_input_bundle,
    read_jsonl,
    select_stable_correct_trajectories,
    sha256_file,
)


DATASETS = ("lvbench", "lsdbench", "cgbench")
SCHEDULE_ID = "rescue_a4_dense_v1"
VARIANT_ID = "rescue"


def _atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_text(encoding="utf-8") != content:
            raise RuntimeError(f"refusing to overwrite changed frozen artifact: {path}")
        return
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".partial",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        handle.write(content)
    temporary.replace(path)


def _json_content(value: Mapping[str, Any]) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n"


def _load_winner(path: Path, config_hash: str) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if (
        not isinstance(payload, dict)
        or payload.get("schema_version") != 2
        or payload.get("experiment_config_sha256") != config_hash
        or payload.get("model_key") != "q9"
        or payload.get("protocol") not in {"no_think", "think"}
    ):
        raise RuntimeError("rescue requires the passed Qwen3.5-9B frozen winner")
    report = payload.get("selection_report")
    if not isinstance(report, Mapping):
        raise ValueError("frozen winner has no selection report reference")
    report_path = Path(str(report.get("path") or ""))
    if not report_path.is_file() or sha256_file(report_path) != report.get("sha256"):
        raise RuntimeError("frozen winner selection report changed")
    report_payload = json.loads(report_path.read_text(encoding="utf-8"))
    report_state = str(report_payload.get("selection_state_sha256") or "")
    if report_state != canonical_sha256(
        {
            key: value
            for key, value in report_payload.items()
            if key != "selection_state_sha256"
        }
    ):
        raise RuntimeError("frozen winner selection report self-hash is invalid")
    winner_fields = {
        key: payload.get(key)
        for key in (
            "winner_id",
            "model_key",
            "protocol",
            "seed",
            "strategy",
            "variant_id",
            "agent_config",
        )
    }
    if (
        report_payload.get("status") != "passed"
        or report_payload.get("blocking_errors")
        or payload.get("selection_state_sha256") != report_state
        or report_payload.get("winner") != winner_fields
        or report_payload.get("source_run_plans") != payload.get("source_run_plans")
    ):
        raise RuntimeError("frozen winner selection report is not passed")
    agent = payload.get("agent_config")
    if not isinstance(agent, Mapping):
        raise ValueError("frozen winner has no Agent config reference")
    agent_path = Path(str(agent.get("path") or ""))
    if not agent_path.is_file() or sha256_file(agent_path) != agent.get("sha256"):
        raise RuntimeError("frozen winner Agent config changed")
    return payload


def _dense_a4_config(config: Mapping[str, Any], output_dir: Path) -> dict[str, str]:
    source = Path(str(config["source_workspace"])) / "configs/agents/a4_independent_arbitration.json"
    payload = json.loads(source.read_text(encoding="utf-8"))
    settings = dict(payload.get("agent", payload))
    settings.update(
        {
            "strategy": "a4_independent_arbitration",
            "overview_frames": 128,
            "local_fps": 2.0,
            "local_window_s": 120.0,
            "max_intervals": 6,
            "max_turns": 8,
            "max_frames_per_call": 128,
            "resize": 0.75,
            "hierarchy_nodes": 8,
            "hierarchy_depth": 3,
            "evidence_strategy": "a3_hierarchical_search",
            "direct_sampling": "uniform128",
        }
    )
    frozen = {
        "schema_version": 1,
        "config_id": SCHEDULE_ID,
        "experiment_config_sha256": canonical_sha256(config),
        "purpose": "teacher_error_dense_a4_rescue",
        "source_agent_config": {"path": str(source), "sha256": sha256_file(source)},
        "agent": settings,
    }
    path = output_dir / "a4_dense_rescue.json"
    _atomic_write(path, _json_content(frozen))
    return {"path": str(path.resolve()), "sha256": sha256_file(path)}


def freeze_rescue_inputs(
    *,
    config_path: Path,
    base_bundle_path: Path,
    frozen_winner_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("experiment config must be an object")
    config_hash = canonical_sha256(config)
    bundle = json.loads(base_bundle_path.read_text(encoding="utf-8"))
    if bundle.get("counterfactual_files") not in {None, 0} or bundle.get(
        "rescue_files"
    ) not in {None, 0}:
        raise ValueError("rescue discovery requires the base 12-schedule bundle only")
    paths, provenance = load_trajectory_input_bundle(base_bundle_path, config_hash)
    train600 = Path(str(config["sft"]["train600"]["path"]))
    manifest = load_training_manifest(train600)
    if manifest.sha256 != config["sft"]["train600"]["sha256"]:
        raise RuntimeError("Train600 changed")
    trajectories = [row for path in paths for row in read_jsonl(path)]
    selection = select_stable_correct_trajectories(
        trajectories,
        manifest,
        config_sha256=config_hash,
        expected_schedules=int(config["sft"]["trajectories_per_sample"]),
        expected_provenance=provenance,
    )
    rescue_by_dataset: dict[str, set[str]] = {dataset: set() for dataset in DATASETS}
    for identity in selection.no_stable_sample_ids:
        dataset, separator, sample_id = identity.partition("/")
        if not separator or dataset not in rescue_by_dataset or not sample_id:
            raise ValueError(f"invalid no-stable identity: {identity}")
        rescue_by_dataset[dataset].add(sample_id)

    winner = _load_winner(frozen_winner_path, config_hash)
    agent_config = _dense_a4_config(config, output_dir)
    manifests: dict[str, dict[str, Any]] = {}
    for dataset in DATASETS:
        source = Path(str(config["datasets"][dataset]["train"]["path"]))
        expected_hash = str(config["datasets"][dataset]["train"]["sha256"])
        if sha256_file(source) != expected_hash:
            raise RuntimeError(f"{dataset} Train200 changed")
        selected: list[dict[str, Any]] = []
        for line in source.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("sample_id") in rescue_by_dataset[dataset]:
                selected.append(row)
        found = {str(row.get("sample_id") or "") for row in selected}
        if found != rescue_by_dataset[dataset]:
            raise RuntimeError(f"{dataset} rescue IDs are not exactly present in Train200")
        path = output_dir / f"{dataset}_rescue_manifest.jsonl"
        content = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in selected)
        _atomic_write(path, content)
        manifests[dataset] = {
            "path": str(path.resolve()),
            "sha256": sha256_file(path),
            "count": len(selected),
            "source_train200_sha256": expected_hash,
            "result_dir": str(
                (Path(str(config["result_root"])) / "trajectories/rescue" / dataset).resolve()
            ),
        }

    index: dict[str, Any] = {
        "schema_version": 1,
        "experiment_config": {
            "path": str(config_path.resolve()),
            "file_sha256": sha256_file(config_path),
            "canonical_sha256": config_hash,
        },
        "source_base_bundle": {
            "path": str(base_bundle_path.resolve()),
            "sha256": sha256_file(base_bundle_path),
            "bundle_sha256": bundle["bundle_sha256"],
        },
        "frozen_winner": {
            "path": str(frozen_winner_path.resolve()),
            "sha256": sha256_file(frozen_winner_path),
            "winner_id": winner["winner_id"],
            "protocol": winner["protocol"],
            "seed": int(winner["seed"]),
        },
        "train600_manifest_sha256": manifest.sha256,
        "model_artifact_sha256": str(config["models"]["q9"]["artifact_sha256"]),
        "schedule_id": SCHEDULE_ID,
        "variant_id": VARIANT_ID,
        "agent_config": agent_config,
        "manifests": manifests,
        "no_stable_sample_count": len(selection.no_stable_sample_ids),
        "stable_base_sample_count": len(selection.selected),
        "rejected_base_families": selection.rejected_families,
    }
    index["rescue_index_sha256"] = canonical_sha256(index)
    index_path = output_dir / "rescue_index.json"
    _atomic_write(index_path, _json_content(index))
    return index


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--base-bundle", type=Path, required=True)
    parser.add_argument("--frozen-winner", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        index = freeze_rescue_inputs(
            config_path=args.config,
            base_bundle_path=args.base_bundle,
            frozen_winner_path=args.frozen_winner,
            output_dir=args.output_dir,
        )
    except (OSError, RuntimeError, ValueError, TypeError, KeyError, json.JSONDecodeError) as error:
        print(json.dumps({"status": "failed", "error": str(error)}, ensure_ascii=False))
        return 1
    print(
        json.dumps(
            {
                "status": "passed",
                "rescue_samples": index["no_stable_sample_count"],
                "rescue_index_sha256": index["rescue_index_sha256"],
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
