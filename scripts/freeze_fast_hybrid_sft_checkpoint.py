#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from flashvid_eval.fast_hybrid_eval_protocol import (
    canonical_sha256,
    freeze_json,
    require_sha256,
    sha256_file,
)
try:
    from scripts.fingerprint_qwen_lora_stack import fingerprint_stack
except ModuleNotFoundError:  # direct `python scripts/...` execution
    from fingerprint_qwen_lora_stack import fingerprint_stack  # type: ignore[no-redef]


def freeze_checkpoint(
    *,
    checkpoint_id: str,
    epoch: int,
    global_step: int,
    adapter_root: Path,
    base_artifact_sha256: str,
    train_data: Path,
    experiment_config: Path,
    expected_experiment_config_sha256: str,
    served_name: str,
    base_urls: list[str],
    output: Path,
) -> dict:
    if not checkpoint_id.strip() or not served_name.strip():
        raise ValueError("checkpoint-id and served-name are required")
    if epoch not in {1, 2, 3} or global_step <= 0:
        raise ValueError("epoch must be 1..3 and global-step must be positive")
    if len(base_urls) != 2 or len(set(base_urls)) != 2:
        raise ValueError("exactly two distinct base URLs are required")
    if not adapter_root.is_dir() or not train_data.is_file():
        raise FileNotFoundError("adapter root or SFT data is missing")
    config_sha = require_sha256(
        expected_experiment_config_sha256, "expected_experiment_config_sha256"
    )
    if not experiment_config.is_file() or sha256_file(experiment_config) != config_sha:
        raise RuntimeError("Fast Hybrid experiment config is missing or changed")
    trainer_state_path = adapter_root / "trainer_state.json"
    if not trainer_state_path.is_file():
        raise FileNotFoundError(trainer_state_path)
    trainer_state = json.loads(trainer_state_path.read_text(encoding="utf-8"))
    if int(trainer_state.get("global_step") or -1) != global_step:
        raise ValueError("checkpoint global-step differs from trainer_state.json")
    recorded_epoch = float(trainer_state.get("epoch") or 0.0)
    if not math.isclose(recorded_epoch, float(epoch), abs_tol=0.05):
        raise ValueError("checkpoint epoch differs from trainer_state.json")
    stack = fingerprint_stack(base_artifact_sha256, adapter_root)
    payload = {
        "schema_version": 1,
        "kind": "fast_hybrid_sft_checkpoint",
        "checkpoint_id": checkpoint_id.strip(),
        "epoch": epoch,
        "global_step": global_step,
        "experiment_config": {
            "path": str(experiment_config.resolve()),
            "sha256": config_sha,
        },
        "base_model_artifact_sha256": require_sha256(
            stack["base_artifact_sha256"], "base_model_artifact_sha256"
        ),
        "adapter": {
            "path": str(adapter_root.resolve()),
            "artifact_sha256": stack["adapter_artifact_sha256"],
            "file_count": stack["adapter_file_count"],
            "total_bytes": stack["adapter_total_bytes"],
            "files": stack["adapter_files"],
        },
        "served_stack_sha256": stack["served_stack_sha256"],
        "served_name": served_name.strip(),
        "base_urls": [value.rstrip("/") for value in base_urls],
        "train_data": {
            "path": str(train_data.resolve()),
            "sha256": sha256_file(train_data),
        },
        "trainer_state": {
            "path": str(trainer_state_path.resolve()),
            "sha256": sha256_file(trainer_state_path),
        },
    }
    payload["checkpoint_fingerprint"] = canonical_sha256(payload)
    freeze_json(output, payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-id", required=True)
    parser.add_argument("--epoch", type=int, required=True)
    parser.add_argument("--global-step", type=int, required=True)
    parser.add_argument("--adapter-root", type=Path, required=True)
    parser.add_argument("--base-artifact-sha256", required=True)
    parser.add_argument("--train-data", type=Path, required=True)
    parser.add_argument("--experiment-config", type=Path, required=True)
    parser.add_argument("--expected-experiment-config-sha256", required=True)
    parser.add_argument("--served-name", required=True)
    parser.add_argument("--base-url", action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        payload = freeze_checkpoint(
            checkpoint_id=args.checkpoint_id,
            epoch=args.epoch,
            global_step=args.global_step,
            adapter_root=args.adapter_root,
            base_artifact_sha256=args.base_artifact_sha256,
            train_data=args.train_data,
            experiment_config=args.experiment_config,
            expected_experiment_config_sha256=args.expected_experiment_config_sha256,
            served_name=args.served_name,
            base_urls=args.base_url,
            output=args.output,
        )
        report = {
            "status": "passed",
            "checkpoint_id": payload["checkpoint_id"],
            "served_stack_sha256": payload["served_stack_sha256"],
            "path": str(args.output.resolve()),
            "sha256": sha256_file(args.output),
        }
    except (OSError, ValueError, TypeError, KeyError, RuntimeError, json.JSONDecodeError) as error:
        report = {"status": "failed", "error": f"{type(error).__name__}: {error}"}
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
