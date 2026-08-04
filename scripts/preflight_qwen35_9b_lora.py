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
    "fla-core": "0.4.2",
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


def _check_attention_backends() -> dict[str, Any]:
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule

    if not callable(chunk_gated_delta_rule):
        raise RuntimeError("flash-linear-attention gated-delta kernel is unavailable")

    optional: dict[str, str | None] = {}
    for distribution in ("causal-conv1d", "flash-attn"):
        try:
            optional[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            optional[distribution] = None
    return {
        "linear_attention": "flash-linear-attention",
        "causal_conv1d": (
            "extension" if optional["causal-conv1d"] is not None else "transformers_torch_fallback"
        ),
        "full_attention": "sdpa",
        "optional_extensions": optional,
    }


def _check_cuda(expected_gpu_count: int) -> tuple[Any, dict[str, Any]]:
    import torch

    torch_version = str(torch.__version__)
    if not torch_version.startswith("2.6.0+cu124"):
        raise RuntimeError(f"expected torch 2.6.0+cu124, found {torch_version}")
    if torch.version.cuda != "12.4":
        raise RuntimeError(f"expected CUDA 12.4 torch runtime, found {torch.version.cuda}")
    triton_version = importlib.metadata.version("triton")
    if triton_version != "3.2.0":
        raise RuntimeError(f"expected triton 3.2.0, found {triton_version}")
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
        "triton": triton_version,
        "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "device_count": count,
        "devices": devices,
        "bf16_supported": True,
    }


def _relative_l1(torch_module: Any, actual: Any, expected: Any) -> float:
    actual_float = actual.detach().float()
    expected_float = expected.detach().float()
    denominator = expected_float.abs().mean().clamp_min(1e-6)
    return float(((actual_float - expected_float).abs().mean() / denominator).item())


def _check_fla_numerics(torch_module: Any) -> dict[str, Any]:
    """Compare the fused Qwen3.5 GDN path with FLA's PyTorch reference."""
    import torch.nn.functional as functional
    from fla.ops.gated_delta_rule import (
        chunk_gated_delta_rule,
        naive_recurrent_gated_delta_rule,
    )

    device = torch_module.device("cuda", 0)
    generator = torch_module.Generator(device=device).manual_seed(20260804)
    shape_qk = (1, 64, 2, 16)
    shape_v = (1, 64, 2, 16)
    shape_gate = (1, 64, 2)

    q_source = torch_module.randn(
        shape_qk, generator=generator, device=device, dtype=torch_module.float32
    ) * 0.1
    k_source = functional.normalize(
        torch_module.randn(
            shape_qk, generator=generator, device=device, dtype=torch_module.float32
        ),
        p=2,
        dim=-1,
    )
    v_source = torch_module.randn(
        shape_v, generator=generator, device=device, dtype=torch_module.float32
    ) * 0.1
    g_source = functional.logsigmoid(
        torch_module.randn(
            shape_gate, generator=generator, device=device, dtype=torch_module.float32
        )
        * 0.1
    )
    beta_source = torch_module.sigmoid(
        torch_module.randn(
            shape_gate, generator=generator, device=device, dtype=torch_module.float32
        )
    )

    sources = (q_source, k_source, v_source, g_source, beta_source)
    fused_inputs = tuple(
        value.to(torch_module.bfloat16).detach().requires_grad_(True)
        for value in sources
    )
    reference_inputs = tuple(
        value.detach().float().requires_grad_(True) for value in fused_inputs
    )

    fused_output, fused_state = chunk_gated_delta_rule(
        *fused_inputs,
        output_final_state=True,
    )
    reference_output, reference_state = naive_recurrent_gated_delta_rule(
        reference_inputs[0],
        reference_inputs[1],
        reference_inputs[2],
        reference_inputs[4],
        reference_inputs[3],
        output_final_state=True,
    )
    if fused_state is None or reference_state is None:
        raise RuntimeError("FLA numerical smoke did not return final states")

    output_weight = torch_module.randn(
        fused_output.shape,
        generator=generator,
        device=device,
        dtype=torch_module.float32,
    )
    fused_loss = (fused_output.float() * output_weight).mean()
    fused_loss = fused_loss + 0.01 * fused_state.float().square().mean()
    reference_loss = (reference_output * output_weight).mean()
    reference_loss = reference_loss + 0.01 * reference_state.square().mean()
    fused_loss.backward()
    reference_loss.backward()

    tensors = {
        "output": (fused_output, reference_output),
        "state": (fused_state, reference_state),
        "q_grad": (fused_inputs[0].grad, reference_inputs[0].grad),
        "k_grad": (fused_inputs[1].grad, reference_inputs[1].grad),
        "v_grad": (fused_inputs[2].grad, reference_inputs[2].grad),
        "g_grad": (fused_inputs[3].grad, reference_inputs[3].grad),
        "beta_grad": (fused_inputs[4].grad, reference_inputs[4].grad),
    }
    limits = {
        "output": 0.08,
        "state": 0.08,
        "q_grad": 0.15,
        "k_grad": 0.20,
        "v_grad": 0.15,
        "g_grad": 0.25,
        "beta_grad": 0.20,
    }
    errors: dict[str, float] = {}
    for name, (actual, expected) in tensors.items():
        if actual is None or expected is None:
            raise RuntimeError(f"FLA numerical smoke produced no {name}")
        if not bool(torch_module.isfinite(actual).all()) or not bool(
            torch_module.isfinite(expected).all()
        ):
            raise RuntimeError(f"FLA numerical smoke produced NaN or Inf in {name}")
        error = _relative_l1(torch_module, actual, expected)
        errors[name] = error
        if error > limits[name]:
            raise RuntimeError(
                f"FLA numerical smoke {name} relative L1 {error:.6f} "
                f"exceeds {limits[name]:.6f}"
            )

    del fused_output, fused_state, reference_output, reference_state
    del fused_inputs, reference_inputs, sources
    gc.collect()
    torch_module.cuda.empty_cache()
    return {
        "reference": "naive_recurrent_gated_delta_rule",
        "dtype": "bfloat16",
        "shape": [1, 64, 2, 16],
        "relative_l1": errors,
        "limits": limits,
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
    attention_backends = _check_attention_backends()
    torch_module, cuda = _check_cuda(args.expected_gpu_count)
    fla_numerics = _check_fla_numerics(torch_module)
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
        "attention_backends": attention_backends,
        "fla_numerics": fla_numerics,
        "cuda": cuda,
        "model": model,
    }
    _atomic_write_json(args.output, payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
