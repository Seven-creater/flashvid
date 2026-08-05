"""Keep LoRA parameters BF16 so PyTorch FSDP2 can shard Qwen3.5 uniformly.

PEFT intentionally promotes FP16/BF16 adapter parameters to FP32 by default.
That is normally harmless, but PyTorch FSDP2 FULL_SHARD requires every
original parameter in a wrapped module to have the same dtype.  The base model
is loaded as BF16, so this narrowly scoped ms-swift plugin casts only trainable
LoRA parameters back to BF16 after adapter injection and before FSDP wrapping.
"""

from __future__ import annotations

import logging

import torch
from swift.pipelines.train.tuner import TunerMixin


LOGGER = logging.getLogger("flashvid.swift_fsdp2_bf16_lora")
_ORIGINAL_PREPARE_MODEL = TunerMixin.prepare_model


def _is_bfloat16(value: object) -> bool:
    return value is torch.bfloat16 or str(value).lower() in {
        "bf16",
        "bfloat16",
        "torch.bfloat16",
    }


def _uses_fsdp2(args: object) -> bool:
    config = getattr(args, "fsdp_config", None)
    if isinstance(config, dict) and int(config.get("fsdp_version", 0)) == 2:
        return True
    value = getattr(args, "fsdp", None)
    if isinstance(value, (list, tuple, set)):
        return any("fsdp2" in str(item).lower() for item in value)
    return "fsdp2" in str(value).lower()


def _cast_trainable_lora_to_bfloat16(model: torch.nn.Module) -> tuple[int, int]:
    frozen_dtypes = {
        parameter.dtype
        for parameter in model.parameters()
        if not parameter.requires_grad and parameter.is_floating_point()
    }
    if frozen_dtypes != {torch.bfloat16}:
        raise RuntimeError(
            "FSDP2 BF16 LoRA patch requires a uniformly BF16 frozen base; "
            f"found {sorted(map(str, frozen_dtypes))}"
        )

    converted_parameters = 0
    converted_elements = 0
    trainable_elements = 0
    for parameter in model.parameters():
        if not parameter.requires_grad:
            continue
        if not parameter.is_floating_point():
            raise RuntimeError(
                "FSDP2 BF16 LoRA patch found a non-floating trainable parameter"
            )
        trainable_elements += parameter.numel()
        if parameter.dtype != torch.bfloat16:
            parameter.data = parameter.data.to(dtype=torch.bfloat16)
            converted_parameters += 1
            converted_elements += parameter.numel()

    if trainable_elements == 0:
        raise RuntimeError("FSDP2 BF16 LoRA patch found no trainable parameters")

    floating_dtypes = {
        parameter.dtype
        for parameter in model.parameters()
        if parameter.is_floating_point()
    }
    if floating_dtypes != {torch.bfloat16}:
        raise RuntimeError(
            "FSDP2 still has mixed floating parameter dtypes after LoRA cast: "
            f"{sorted(map(str, floating_dtypes))}"
        )
    return converted_parameters, converted_elements


@classmethod
def _prepare_model_with_uniform_bfloat16_lora(
    cls,
    args,
    model,
    *,
    template=None,
    train_dataset=None,
    task_type=None,
):
    model = _ORIGINAL_PREPARE_MODEL(
        args,
        model,
        template=template,
        train_dataset=train_dataset,
        task_type=task_type,
    )
    if (
        _uses_fsdp2(args)
        and getattr(args, "tuner_type", None) == "lora"
        and _is_bfloat16(getattr(args, "lora_dtype", None))
    ):
        count, elements = _cast_trainable_lora_to_bfloat16(model)
        LOGGER.warning(
            "FSDP2 compatibility: converted %d LoRA tensors (%d parameters) "
            "to BF16 before FULL_SHARD wrapping",
            count,
            elements,
        )
    return model


if not getattr(TunerMixin, "_flashvid_fsdp2_bf16_lora_patch", False):
    TunerMixin.prepare_model = _prepare_model_with_uniform_bfloat16_lora
    TunerMixin._flashvid_fsdp2_bf16_lora_patch = True
