#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
from pathlib import Path
from typing import Any, Mapping

try:
    from scripts.fingerprint_qwen_lora_stack import fingerprint_stack
except ModuleNotFoundError:  # Direct ``python scripts/...`` execution.
    from fingerprint_qwen_lora_stack import fingerprint_stack


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    content = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def _atomic_freeze(path: Path, payload: Mapping[str, Any]) -> None:
    content = json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing != payload:
            raise RuntimeError(f"refusing to overwrite changed frozen checkpoint: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="wb",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".partial",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        handle.write(content.encode("utf-8"))
    temporary.replace(path)


def freeze_checkpoint(
    *,
    checkpoint_id: str,
    epoch: int,
    adapter_root: Path,
    base_artifact_sha256: str,
    train_data: Path,
    frozen_winner: Path,
    experiment_config_sha256: str,
    served_name: str,
    base_url: str,
    output: Path,
) -> dict[str, Any]:
    if not checkpoint_id.strip() or not served_name.strip() or not base_url.strip():
        raise ValueError("checkpoint id, served name, and base URL are required")
    if epoch not in {1, 2, 3}:
        raise ValueError("epoch must be 1, 2, or 3")
    if not adapter_root.is_dir() or not train_data.is_file() or not frozen_winner.is_file():
        raise FileNotFoundError("adapter, train data, or frozen winner is missing")
    winner = json.loads(frozen_winner.read_text(encoding="utf-8"))
    if not isinstance(winner, dict) or winner.get("schema_version") != 2:
        raise ValueError("frozen winner must be a schema-v2 Agent winner")
    if winner.get("experiment_config_sha256") != experiment_config_sha256:
        raise RuntimeError("frozen winner belongs to a different experiment config")
    stack = fingerprint_stack(base_artifact_sha256, adapter_root)
    payload: dict[str, Any] = {
        "schema_version": 1,
        "checkpoint_id": checkpoint_id,
        "epoch": epoch,
        "experiment_config_sha256": experiment_config_sha256,
        "base_model_artifact_sha256": stack["base_artifact_sha256"],
        "adapter": {
            "path": str(adapter_root.resolve()),
            "artifact_sha256": stack["adapter_artifact_sha256"],
            "file_count": stack["adapter_file_count"],
            "total_bytes": stack["adapter_total_bytes"],
            "files": stack["adapter_files"],
        },
        "served_stack_sha256": stack["served_stack_sha256"],
        "served_name": served_name,
        "base_url": base_url,
        "train_data": {
            "path": str(train_data.resolve()),
            "sha256": sha256_file(train_data),
        },
        "frozen_winner": {
            "path": str(frozen_winner.resolve()),
            "sha256": sha256_file(frozen_winner),
            "winner_id": winner.get("winner_id"),
            "agent_config": winner.get("agent_config"),
        },
    }
    payload["checkpoint_fingerprint"] = canonical_sha256(payload)
    _atomic_freeze(output, payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Freeze one auditable Qwen3.5-9B LoRA checkpoint.")
    parser.add_argument("--checkpoint-id", required=True)
    parser.add_argument("--epoch", type=int, required=True)
    parser.add_argument("--adapter-root", type=Path, required=True)
    parser.add_argument("--base-artifact-sha256", required=True)
    parser.add_argument("--train-data", type=Path, required=True)
    parser.add_argument("--frozen-winner", type=Path, required=True)
    parser.add_argument("--experiment-config-sha256", required=True)
    parser.add_argument("--served-name", required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:8200/v1")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = freeze_checkpoint(
        checkpoint_id=args.checkpoint_id,
        epoch=args.epoch,
        adapter_root=args.adapter_root,
        base_artifact_sha256=args.base_artifact_sha256,
        train_data=args.train_data,
        frozen_winner=args.frozen_winner,
        experiment_config_sha256=args.experiment_config_sha256,
        served_name=args.served_name,
        base_url=args.base_url,
        output=args.output,
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
