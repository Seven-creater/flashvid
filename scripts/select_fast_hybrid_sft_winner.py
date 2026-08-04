#!/usr/bin/env python3
"""Select and immutably freeze the sole Fast Hybrid SFT Dev150 winner."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Mapping

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from flashvid_eval.fast_hybrid_checkpoint_selection import (
    CheckpointRun,
    select_fast_hybrid_checkpoint,
)
from flashvid_eval.fast_hybrid_eval_protocol import (
    DATASETS,
    audit_result_file,
    freeze_json,
    load_protocol,
    manifest_sample_ids,
    require_sha256,
    sha256_file,
)
try:
    from scripts.run_fast_hybrid_sft_eval import _load_checkpoint
except ModuleNotFoundError:  # direct `python scripts/...` execution
    from run_fast_hybrid_sft_eval import _load_checkpoint  # type: ignore[no-redef]


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain an object")
    return value


def _parse_checkpoints(values: list[str]) -> dict[Path, Path]:
    result: dict[Path, Path] = {}
    for value in values:
        if "=" not in value:
            raise ValueError("--checkpoint must use CONFIG=DEV_RUN_ROOT")
        raw_config, raw_root = value.split("=", 1)
        config = Path(raw_config)
        if config in result:
            raise ValueError(f"duplicate checkpoint config: {config}")
        result[config] = Path(raw_root)
    if not result:
        raise ValueError("at least one checkpoint Dev run is required")
    return result


def _audit_run(
    root: Path,
    *,
    protocol: Mapping[str, Any],
    protocol_sha: str,
    mode: str,
    served_hash: str,
    checkpoint_config: Path | None,
) -> dict[str, Path]:
    metadata_path = root / "evaluation_run.json"
    metadata = _load_json(metadata_path)
    if (
        metadata.get("kind") != "fast_hybrid_sft_eval_run"
        or metadata.get("status") != "passed"
        or metadata.get("phase") != "dev"
        or metadata.get("mode") != mode
        or metadata.get("evaluation_protocol_sha256") != protocol_sha
        or metadata.get("served_model_sha256") != served_hash
        or metadata.get("teacher_model_sha256") != protocol["base_model_artifact_sha256"]
    ):
        raise RuntimeError(f"invalid or mismatched Dev run metadata: {metadata_path}")
    if checkpoint_config is None:
        if metadata.get("checkpoint_config") is not None:
            raise RuntimeError("Teacher Dev run unexpectedly references a checkpoint")
    else:
        reference = metadata.get("checkpoint_config") or {}
        if (
            reference.get("path") != str(checkpoint_config.resolve())
            or reference.get("sha256") != sha256_file(checkpoint_config)
        ):
            raise RuntimeError("checkpoint Dev run references another checkpoint config")
    paths: dict[str, Path] = {}
    for dataset in DATASETS:
        entry = protocol["splits"]["dev"][dataset]
        file_reference = metadata["files"][dataset]
        path = Path(file_reference["path"])
        audit = audit_result_file(
            path,
            dataset=dataset,
            expected_count=int(entry["manifest"]["count"]),
            expected_sample_ids=manifest_sample_ids(
                Path(str(entry["manifest"]["path"]))
            ),
            manifest_sha256=str(entry["manifest"]["sha256"]),
            candidate_sha256=str(entry["candidate"]["sha256"]),
            experiment_config_sha256=str(protocol["experiment_config"]["sha256"]),
            served_model_sha256=served_hash,
            teacher_model_sha256=str(protocol["base_model_artifact_sha256"]),
        )
        if file_reference.get("sha256") != audit["sha256"]:
            raise RuntimeError(f"Dev result changed after run audit: {path}")
        paths[dataset] = path
    return paths


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--expected-protocol-sha256", required=True)
    parser.add_argument("--teacher-run-root", type=Path, required=True)
    parser.add_argument("--checkpoint", action="append", default=[], required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        protocol_sha = require_sha256(
            args.expected_protocol_sha256, "expected_protocol_sha256"
        )
        protocol = load_protocol(args.protocol, protocol_sha)
        base_hash = str(protocol["base_model_artifact_sha256"])
        teacher_paths = _audit_run(
            args.teacher_run_root,
            protocol=protocol,
            protocol_sha=protocol_sha,
            mode="teacher",
            served_hash=base_hash,
            checkpoint_config=None,
        )
        checkpoint_runs: list[CheckpointRun] = []
        checkpoint_payloads: dict[str, tuple[Path, dict[str, Any], Path]] = {}
        for config_path, run_root in _parse_checkpoints(args.checkpoint).items():
            checkpoint = _load_checkpoint(config_path, protocol)
            checkpoint_id = str(checkpoint["checkpoint_id"])
            if checkpoint_id in checkpoint_payloads:
                raise ValueError(f"duplicate checkpoint ID: {checkpoint_id}")
            paths = _audit_run(
                run_root,
                protocol=protocol,
                protocol_sha=protocol_sha,
                mode="checkpoint",
                served_hash=str(checkpoint["served_stack_sha256"]),
                checkpoint_config=config_path,
            )
            checkpoint_runs.append(
                CheckpointRun(checkpoint_id, int(checkpoint["epoch"]), paths)
            )
            checkpoint_payloads[checkpoint_id] = (config_path, checkpoint, run_root)
        selection = select_fast_hybrid_checkpoint(
            teacher_paths=teacher_paths,
            checkpoints=checkpoint_runs,
        )
        selected = selection.get("selected")
        payload: dict[str, Any] = {
            "schema_version": 1,
            "kind": "fast_hybrid_sft_winner",
            "status": "passed" if selected else "blocked",
            "evaluation_protocol": {
                "path": str(args.protocol.resolve()),
                "sha256": protocol_sha,
            },
            "evaluation_protocol_sha256": protocol_sha,
            "teacher_run": {
                "path": str(args.teacher_run_root.resolve()),
                "metadata_sha256": sha256_file(args.teacher_run_root / "evaluation_run.json"),
            },
            "selection": selection,
            "checkpoint_config": None,
            "checkpoint_dev_run": None,
        }
        if selected:
            checkpoint_id = str(selected["checkpoint_id"])
            config_path, checkpoint, run_root = checkpoint_payloads[checkpoint_id]
            payload["checkpoint_config"] = {
                "path": str(config_path.resolve()),
                "sha256": sha256_file(config_path),
                "checkpoint_id": checkpoint_id,
                "served_stack_sha256": checkpoint["served_stack_sha256"],
            }
            payload["checkpoint_dev_run"] = {
                "path": str(run_root.resolve()),
                "metadata_sha256": sha256_file(run_root / "evaluation_run.json"),
            }
        freeze_json(args.output, payload)
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0 if selected else 2
    except (OSError, ValueError, TypeError, KeyError, RuntimeError, json.JSONDecodeError) as error:
        print(json.dumps({"status": "failed", "error": f"{type(error).__name__}: {error}"}, ensure_ascii=False))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
