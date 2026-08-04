#!/usr/bin/env python3
"""Prove that a one-step Qwen3.5 LoRA smoke run actually updated weights."""

from __future__ import annotations

import argparse
import json
import math
import tempfile
from pathlib import Path
from typing import Any, Mapping

import torch
from safetensors import safe_open


def _positive_finite(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a finite positive number")
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a finite positive number") from error
    if not math.isfinite(number) or number <= 0:
        raise ValueError(f"{name} must be a finite positive number, got {value!r}")
    return number


def _load_one_step_metrics(output_dir: Path) -> tuple[float, float, Path]:
    matches: list[tuple[float, float, Path]] = []
    for path in sorted(output_dir.rglob("trainer_state.json")):
        state = json.loads(path.read_text(encoding="utf-8"))
        if int(state.get("global_step") or 0) != 1:
            continue
        for record in state.get("log_history") or []:
            if int(record.get("step") or 0) != 1:
                continue
            if "loss" not in record or "grad_norm" not in record:
                continue
            matches.append(
                (
                    _positive_finite(record["loss"], "step-1 loss"),
                    _positive_finite(record["grad_norm"], "step-1 grad_norm"),
                    path,
                )
            )
    if not matches:
        raise ValueError(
            "no trainer_state.json contains step-1 finite positive loss and grad_norm"
        )
    if len(matches) > 1:
        unique = {(loss, grad_norm) for loss, grad_norm, _ in matches}
        if len(unique) != 1:
            raise ValueError("conflicting step-1 loss/grad_norm records")
    return matches[0]


def _adapter_directory(output_dir: Path) -> Path:
    candidates = sorted(
        path.parent
        for path in output_dir.rglob("adapter_model.safetensors")
        if (path.parent / "adapter_config.json").is_file()
    )
    checkpoint_one = [path for path in candidates if path.name == "checkpoint-1"]
    if len(checkpoint_one) == 1:
        return checkpoint_one[0]
    if len(candidates) == 1:
        return candidates[0]
    if not candidates:
        raise ValueError("no PEFT safetensors adapter/config pair was saved")
    raise ValueError(f"ambiguous saved adapters: {[str(path) for path in candidates]}")


def _verify_lora_b_weights(adapter_path: Path) -> tuple[list[str], int]:
    lora_b_keys: list[str] = []
    nonzero_values = 0
    with safe_open(adapter_path, framework="pt", device="cpu") as checkpoint:
        for key in checkpoint.keys():
            if "lora_B" not in key:
                continue
            tensor = checkpoint.get_tensor(key)
            lora_b_keys.append(key)
            if not bool(torch.isfinite(tensor).all()):
                raise ValueError(f"LoRA B tensor contains NaN or Inf: {key}")
            nonzero_values += int(torch.count_nonzero(tensor).item())
    if not lora_b_keys:
        raise ValueError("saved adapter contains no LoRA B tensors")
    if nonzero_values == 0:
        raise ValueError("all saved LoRA B tensors are still zero after one step")
    return lora_b_keys, nonzero_values


def _reload_with_peft(adapter_dir: Path) -> int:
    try:
        from peft import PeftConfig
        from peft.utils.save_and_load import load_peft_weights
    except ImportError as error:
        raise ValueError("PEFT is unavailable, so the saved adapter cannot be reloaded") from error

    config = PeftConfig.from_pretrained(str(adapter_dir), local_files_only=True)
    weights: Mapping[str, torch.Tensor] = load_peft_weights(
        str(adapter_dir), device="cpu"
    )
    if not getattr(config, "peft_type", None):
        raise ValueError("PEFT reloaded an invalid adapter config")
    if not weights:
        raise ValueError("PEFT reloaded no adapter weights")
    return len(weights)


def verify_smoke_run(output_dir: Path) -> dict[str, Any]:
    if not output_dir.is_dir():
        raise ValueError(f"smoke output directory not found: {output_dir}")
    loss, grad_norm, trainer_state = _load_one_step_metrics(output_dir)
    adapter_dir = _adapter_directory(output_dir)
    adapter_path = adapter_dir / "adapter_model.safetensors"
    lora_b_keys, nonzero_values = _verify_lora_b_weights(adapter_path)
    peft_tensor_count = _reload_with_peft(adapter_dir)
    return {
        "schema_version": 1,
        "status": "passed",
        "global_step": 1,
        "loss": loss,
        "grad_norm": grad_norm,
        "trainer_state": str(trainer_state.resolve()),
        "adapter_dir": str(adapter_dir.resolve()),
        "lora_b_tensor_count": len(lora_b_keys),
        "lora_b_nonzero_values": nonzero_values,
        "peft_reloaded_tensor_count": peft_tensor_count,
    }


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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Verify one-step loss, gradients, and a nonzero reloadable LoRA adapter."
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    try:
        payload = verify_smoke_run(args.output_dir)
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as error:
        payload = {"schema_version": 1, "status": "failed", "error": str(error)}
    _write_json(args.report, payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    if payload["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
