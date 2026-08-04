#!/usr/bin/env python3
"""Fail-fast checks for the isolated Qwen3.5-9B LoRA training runtime."""

from __future__ import annotations

import argparse
import gc
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import tempfile
from typing import Any


REQUIRED_VERSIONS = {
    "ms-swift": "4.4.2",
    "transformers": "5.9.0",
    "qwen-vl-utils": "0.0.14",
    "peft": "0.19.1",
    "datasets": "4.8.4",
    "accelerate": "1.14.0",
    "trl": "0.29.1",
    "flash-linear-attention": "0.4.2",
    "causal-conv1d": "1.6.2.post1",
    "flash-attn": "2.8.3",
}
MIN_GPU_MEMORY_BYTES = 42 * 1024**3


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
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


def _check_python() -> None:
    if sys.version_info[:2] != (3, 12):
        raise RuntimeError(
            f"Qwen3.5 SFT requires Python 3.12, found {sys.version.split()[0]}"
        )
    if sys.prefix == sys.base_prefix:
        raise RuntimeError("training must run inside an independent virtual environment")


def _check_versions() -> dict[str, str]:
    installed: dict[str, str] = {}
    for distribution, expected in REQUIRED_VERSIONS.items():
        actual = importlib.metadata.version(distribution)
        if actual != expected:
            raise RuntimeError(
                f"expected {distribution}=={expected}, found {distribution}=={actual}"
            )
        installed[distribution] = actual
    return installed


def _check_cuda(expected_gpu_count: int) -> tuple[Any, dict[str, Any]]:
    import torch

    torch_version = str(torch.__version__)
    if not torch_version.startswith("2.5.1+cu121"):
        raise RuntimeError(f"expected torch 2.5.1+cu121, found {torch_version}")
    if torch.version.cuda != "12.1":
        raise RuntimeError(f"expected CUDA 12.1 torch runtime, found {torch.version.cuda}")
    if not torch.cuda.is_available():
        raise RuntimeError("torch.cuda.is_available() is false")
    count = torch.cuda.device_count()
    if count != expected_gpu_count:
        raise RuntimeError(
            f"expected {expected_gpu_count} visible GPUs, torch reports {count}"
        )
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("selected CUDA runtime does not report BF16 support")

    devices: list[dict[str, Any]] = []
    for index in range(count):
        properties = torch.cuda.get_device_properties(index)
        if properties.total_memory < MIN_GPU_MEMORY_BYTES:
            gib = properties.total_memory / 1024**3
            raise RuntimeError(f"visible GPU {index} has only {gib:.1f} GiB; 42 GiB required")
        devices.append(
            {
                "logical_index": index,
                "name": properties.name,
                "total_memory_bytes": properties.total_memory,
            }
        )
    return torch, {
        "torch": torch_version,
        "torch_cuda": torch.version.cuda,
        "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "device_count": count,
        "devices": devices,
        "bf16_supported": True,
    }


def _check_transformers_model(
    model_path: Path,
    *,
    load_weights: bool,
    torch_module: Any,
) -> dict[str, Any]:
    if not model_path.is_dir():
        raise FileNotFoundError(f"local Qwen3.5-9B model directory not found: {model_path}")

    from transformers import AutoConfig, AutoProcessor

    config = AutoConfig.from_pretrained(
        str(model_path), trust_remote_code=False, local_files_only=True
    )
    architectures = list(getattr(config, "architectures", []) or [])
    if getattr(config, "model_type", None) != "qwen3_5":
        raise RuntimeError(f"expected model_type=qwen3_5, found {config.model_type!r}")
    if "Qwen3_5ForConditionalGeneration" not in architectures:
        raise RuntimeError(f"unexpected Qwen3.5 architectures: {architectures}")
    processor = AutoProcessor.from_pretrained(
        str(model_path), trust_remote_code=False, local_files_only=True
    )
    report: dict[str, Any] = {
        "model_type": config.model_type,
        "architectures": architectures,
        "processor_class": type(processor).__name__,
        "weights_loaded": False,
    }

    if load_weights:
        from transformers import AutoModelForImageTextToText

        model = AutoModelForImageTextToText.from_pretrained(
            str(model_path),
            dtype=torch_module.bfloat16,
            device_map="auto",
            local_files_only=True,
            low_cpu_mem_usage=True,
            trust_remote_code=False,
        )
        if type(model).__name__ != "Qwen3_5ForConditionalGeneration":
            raise RuntimeError(f"unexpected loaded model class: {type(model).__name__}")
        report.update(
            {
                "weights_loaded": True,
                "model_class": type(model).__name__,
                "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
            }
        )
        del model
        gc.collect()
        torch_module.cuda.empty_cache()
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Verify CUDA, pinned dependencies and local Qwen3.5 Transformers loading."
    )
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--expected-gpu-count", required=True, type=int, choices=(4, 8))
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--load-weights",
        action="store_true",
        help="Load the complete 9B model through Transformers before the one-step smoke.",
    )
    args = parser.parse_args()

    _check_python()
    versions = _check_versions()
    torch_module, cuda = _check_cuda(args.expected_gpu_count)
    model = _check_transformers_model(
        args.model,
        load_weights=args.load_weights,
        torch_module=torch_module,
    )
    payload = {
        "status": "passed",
        "python": sys.version.split()[0],
        "executable": sys.executable,
        "versions": versions,
        "cuda": cuda,
        "model": model,
    }
    _atomic_write_json(args.output, payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
