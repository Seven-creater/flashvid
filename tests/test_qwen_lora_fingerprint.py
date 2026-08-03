from __future__ import annotations

from pathlib import Path

from scripts.fingerprint_qwen_lora_stack import fingerprint_stack


def test_lora_stack_fingerprint_binds_base_and_adapter(tmp_path: Path) -> None:
    adapter = tmp_path / "checkpoint-1"
    adapter.mkdir()
    (adapter / "adapter_model.safetensors").write_bytes(b"adapter-v1")
    (adapter / "adapter_config.json").write_text("{}", encoding="utf-8")
    first = fingerprint_stack("a" * 64, adapter)
    assert len(first["served_stack_sha256"]) == 64
    assert first["served_stack_sha256"] != fingerprint_stack("b" * 64, adapter)[
        "served_stack_sha256"
    ]
    (adapter / "adapter_model.safetensors").write_bytes(b"adapter-v2")
    assert first["served_stack_sha256"] != fingerprint_stack("a" * 64, adapter)[
        "served_stack_sha256"
    ]
