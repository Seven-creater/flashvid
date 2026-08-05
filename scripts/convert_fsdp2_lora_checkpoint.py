#!/usr/bin/env python3
"""Consolidate ms-swift FSDP2 DCP LoRA shards into a deployable PEFT adapter."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import tempfile
from typing import Mapping

import torch
from peft import LoraConfig, TaskType
from safetensors.torch import save_file
from torch.distributed.checkpoint.format_utils import dcp_to_torch_save


def normalize_adapter_state(state: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    normalized: dict[str, torch.Tensor] = {}
    for original_name, tensor in state.items():
        name = original_name.removeprefix("model.")
        if not name.startswith("base_model.model."):
            raise ValueError(f"unexpected FSDP2 adapter key: {original_name}")
        if not (name.endswith(".lora_A.weight") or name.endswith(".lora_B.weight")):
            raise ValueError(f"non-LoRA tensor in FSDP2 adapter state: {original_name}")
        if name in normalized:
            raise ValueError(f"duplicate normalized adapter key: {name}")
        normalized[name] = tensor.detach().cpu().contiguous()
    if not normalized:
        raise ValueError("FSDP2 adapter state is empty")
    a_count = sum(name.endswith(".lora_A.weight") for name in normalized)
    b_tensors = [tensor for name, tensor in normalized.items() if name.endswith(".lora_B.weight")]
    if a_count != len(b_tensors) or a_count == 0:
        raise ValueError(f"unpaired LoRA tensors: A={a_count} B={len(b_tensors)}")
    if not any(torch.count_nonzero(tensor).item() for tensor in b_tensors):
        raise ValueError("all consolidated LoRA-B tensors are zero")
    return normalized


def target_module_pattern(state: Mapping[str, torch.Tensor]) -> str:
    modules: set[str] = set()
    for name in state:
        module_path = name.removeprefix("base_model.model.").rsplit(".lora_", 1)[0]
        modules.add(module_path.rsplit(".", 1)[-1])
    alternatives = "|".join(re.escape(value) for value in sorted(modules))
    return rf"^(model\.language_model(?=\.).*\.({alternatives}))$"


def convert_checkpoint(checkpoint: Path, base_model: Path) -> dict[str, object]:
    dcp = checkpoint / "pytorch_model_fsdp_0"
    adapter_model = checkpoint / "adapter_model.safetensors"
    adapter_config = checkpoint / "adapter_config.json"
    report_path = checkpoint / "fsdp2_adapter_conversion.json"
    if not dcp.is_dir():
        raise FileNotFoundError(f"FSDP2 model shard directory is missing: {dcp}")
    if report_path.exists():
        if not (adapter_model.is_file() and adapter_config.is_file()):
            raise RuntimeError(f"completed conversion marker has missing files in {checkpoint}")
        return json.loads(report_path.read_text(encoding="utf-8"))
    # Without the final report marker, any adapter/config file is an incomplete
    # product of this converter. Rebuild it deterministically from immutable DCP
    # shards and atomically replace the model file.

    temporary_torch: str | None = None
    temporary_safe: str | None = None
    try:
        descriptor, temporary_torch = tempfile.mkstemp(
            prefix="fsdp2_lora_", suffix=".pt", dir=checkpoint
        )
        os.close(descriptor)
        os.unlink(temporary_torch)
        dcp_to_torch_save(str(dcp), temporary_torch)
        payload = torch.load(temporary_torch, map_location="cpu", weights_only=True)
        if not isinstance(payload, dict) or not isinstance(payload.get("model"), dict):
            raise ValueError("unexpected FSDP2 DCP payload")
        state = normalize_adapter_state(payload["model"])

        descriptor, temporary_safe = tempfile.mkstemp(
            prefix="adapter_model_", suffix=".safetensors", dir=checkpoint
        )
        os.close(descriptor)
        save_file(state, temporary_safe, metadata={"format": "pt"})
        os.replace(temporary_safe, adapter_model)
        temporary_safe = None

        config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            base_model_name_or_path=str(base_model.resolve()),
            inference_mode=True,
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            target_modules=target_module_pattern(state),
            bias="none",
        )
        config.save_pretrained(checkpoint)
        report = {
            "schema_version": 1,
            "status": "passed",
            "checkpoint": str(checkpoint.resolve()),
            "tensor_count": len(state),
            "lora_b_tensor_count": sum(
                name.endswith(".lora_B.weight") for name in state
            ),
            "lora_b_nonzero_values": sum(
                torch.count_nonzero(tensor).item()
                for name, tensor in state.items()
                if name.endswith(".lora_B.weight")
            ),
        }
        report_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        return report
    finally:
        for temporary in (temporary_torch, temporary_safe):
            if temporary and os.path.exists(temporary):
                os.unlink(temporary)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--base-model", type=Path, required=True)
    args = parser.parse_args()
    checkpoints = sorted(
        path for path in args.checkpoint_root.glob("checkpoint-*") if path.is_dir()
    )
    if not checkpoints:
        raise SystemExit(f"no checkpoint-* directories in {args.checkpoint_root}")
    reports = [convert_checkpoint(path, args.base_model) for path in checkpoints]
    print(json.dumps(reports, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
