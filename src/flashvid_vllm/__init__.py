"""vLLM integration for FlashVID."""

ARCHITECTURE = "FlashVIDQwen3_5ForConditionalGeneration"


def register() -> None:
    from .plugin import register as register_plugin

    register_plugin()


__all__ = ["ARCHITECTURE", "register"]

