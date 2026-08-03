from __future__ import annotations

import argparse
import hashlib
import json
import math
import tempfile
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open


DEFAULT_STEPS = (13, 26, 39)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
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


def _tensor_schema_and_finiteness(path: Path) -> tuple[dict[str, Any], int]:
    schema: dict[str, Any] = {}
    tensor_count = 0
    with safe_open(path, framework="pt", device="cpu") as checkpoint:
        for key in checkpoint.keys():
            tensor = checkpoint.get_tensor(key)
            tensor_count += 1
            schema[key] = {
                "shape": list(tensor.shape),
                "dtype": str(tensor.dtype),
            }
            if tensor.is_floating_point() and not bool(torch.isfinite(tensor).all()):
                raise ValueError(f"{path}: tensor {key} contains NaN or Inf")
    if tensor_count == 0:
        raise ValueError(f"{path}: adapter has no tensors")
    return schema, tensor_count


def verify_checkpoints(root: Path, expected_steps: tuple[int, ...]) -> dict[str, Any]:
    if not root.is_dir():
        raise ValueError(f"checkpoint root not found: {root}")
    checkpoints: list[dict[str, Any]] = []
    reference_schema: dict[str, Any] | None = None
    for ordinal, step in enumerate(expected_steps, start=1):
        checkpoint = root / f"checkpoint-{step}"
        if not checkpoint.is_dir():
            raise ValueError(f"missing checkpoint directory: {checkpoint}")
        adapter = checkpoint / "adapter_model.safetensors"
        config_path = checkpoint / "adapter_config.json"
        trainer_state_path = checkpoint / "trainer_state.json"
        for path in (adapter, config_path, trainer_state_path):
            if not path.is_file() or path.stat().st_size <= 0:
                raise ValueError(f"missing or empty checkpoint artifact: {path}")

        config = json.loads(config_path.read_text(encoding="utf-8"))
        rank = int(config.get("r") or 0)
        if rank != 16:
            raise ValueError(f"{config_path}: expected LoRA rank 16, got {rank}")
        if str(config.get("task_type") or "").upper() != "CAUSAL_LM":
            raise ValueError(f"{config_path}: task_type must be CAUSAL_LM")

        trainer_state = json.loads(trainer_state_path.read_text(encoding="utf-8"))
        actual_step = int(trainer_state.get("global_step") or -1)
        if actual_step != step:
            raise ValueError(
                f"{trainer_state_path}: expected global_step {step}, got {actual_step}"
            )
        epoch = float(trainer_state.get("epoch") or 0.0)
        if not math.isclose(epoch, float(ordinal), abs_tol=0.05):
            raise ValueError(
                f"{trainer_state_path}: expected epoch {ordinal}, got {epoch}"
            )

        schema, tensor_count = _tensor_schema_and_finiteness(adapter)
        if reference_schema is None:
            reference_schema = schema
        elif schema != reference_schema:
            raise ValueError(f"{adapter}: tensor schema differs from checkpoint-{expected_steps[0]}")
        checkpoints.append(
            {
                "name": checkpoint.name,
                "global_step": actual_step,
                "epoch": epoch,
                "lora_rank": rank,
                "tensor_count": tensor_count,
                "adapter_sha256": _sha256(adapter),
                "adapter_config_sha256": _sha256(config_path),
                "trainer_state_sha256": _sha256(trainer_state_path),
            }
        )

    extras = sorted(
        path.name
        for path in root.glob("checkpoint-*")
        if path.is_dir() and path.name not in {f"checkpoint-{step}" for step in expected_steps}
    )
    if extras:
        raise ValueError(f"unexpected checkpoint directories: {extras}")
    return {
        "schema_version": 1,
        "status": "complete",
        "checkpoint_root": str(root.resolve()),
        "expected_steps": list(expected_steps),
        "checkpoints": checkpoints,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Verify that an SFT run completed and all LoRA checkpoints are usable."
    )
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--expected-step", type=int, action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    expected_steps = tuple(args.expected_step) or DEFAULT_STEPS
    try:
        payload = verify_checkpoints(args.checkpoint_root, expected_steps)
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as error:
        payload = {
            "schema_version": 1,
            "status": "failed",
            "checkpoint_root": str(args.checkpoint_root.resolve()),
            "expected_steps": list(expected_steps),
            "error": str(error),
        }
    _write_json(args.output, payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    if payload["status"] != "complete":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
