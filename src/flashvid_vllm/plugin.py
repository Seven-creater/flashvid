from __future__ import annotations

from . import ARCHITECTURE


def register() -> None:
    """Register lazily so plugin discovery never initializes CUDA."""
    from vllm import ModelRegistry

    if ARCHITECTURE not in ModelRegistry.get_supported_archs():
        ModelRegistry.register_model(
            ARCHITECTURE,
            "flashvid_vllm.model:FlashVIDQwen3_5ForConditionalGeneration",
        )

