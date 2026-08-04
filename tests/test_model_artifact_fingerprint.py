from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest


SCRIPT = Path(__file__).parents[1] / "scripts" / "fingerprint_model_artifact.py"
SPEC = importlib.util.spec_from_file_location("fingerprint_model_artifact", SCRIPT)
assert SPEC and SPEC.loader
module = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = module
SPEC.loader.exec_module(module)


def test_model_artifact_fingerprint_binds_weights_and_ignores_runtime_files(
    tmp_path: Path,
) -> None:
    root = tmp_path / "model"
    root.mkdir()
    weight = root / "model.safetensors"
    weight.write_bytes(b"weights-v1")
    (root / "config.json").write_text("{}", encoding="utf-8")
    (root / "READY.txt").write_text("volatile marker", encoding="utf-8")
    cache = root / ".cache" / "huggingface" / "download"
    cache.mkdir(parents=True)
    (cache / "model.safetensors.lock").write_text("", encoding="utf-8")
    first = module.fingerprint_model(root)
    (root / "READY.txt").write_text("changed", encoding="utf-8")
    (cache / "model.safetensors.lock").write_text("runtime change", encoding="utf-8")
    assert module.fingerprint_model(root)["artifact_sha256"] == first["artifact_sha256"]
    weight.write_bytes(b"weights-v2")
    assert module.fingerprint_model(root)["artifact_sha256"] != first["artifact_sha256"]


def test_model_artifact_fingerprint_requires_weights(tmp_path: Path) -> None:
    root = tmp_path / "model"
    root.mkdir()
    (root / "config.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="no safetensors"):
        module.fingerprint_model(root)
