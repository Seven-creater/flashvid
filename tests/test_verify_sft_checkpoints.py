from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from scripts.verify_sft_checkpoints import verify_checkpoints


def _checkpoint(root: Path, step: int, epoch: float, *, finite: bool = True) -> None:
    path = root / f"checkpoint-{step}"
    path.mkdir(parents=True)
    tensor = torch.ones((2, 2))
    if not finite:
        tensor[0, 0] = float("nan")
    save_file({"adapter.weight": tensor}, path / "adapter_model.safetensors")
    (path / "adapter_config.json").write_text(
        json.dumps({"r": 16, "task_type": "CAUSAL_LM"}), encoding="utf-8"
    )
    (path / "trainer_state.json").write_text(
        json.dumps({"global_step": step, "epoch": epoch}), encoding="utf-8"
    )


def test_verify_completed_sft_checkpoints(tmp_path: Path) -> None:
    for epoch, step in enumerate((13, 26, 39), start=1):
        _checkpoint(tmp_path, step, float(epoch))
    report = verify_checkpoints(tmp_path, (13, 26, 39))
    assert report["status"] == "complete"
    assert [row["global_step"] for row in report["checkpoints"]] == [13, 26, 39]
    assert all(row["tensor_count"] == 1 for row in report["checkpoints"])


def test_verify_rejects_nonfinite_adapter(tmp_path: Path) -> None:
    _checkpoint(tmp_path, 13, 1.0)
    _checkpoint(tmp_path, 26, 2.0, finite=False)
    _checkpoint(tmp_path, 39, 3.0)
    with pytest.raises(ValueError, match="NaN or Inf"):
        verify_checkpoints(tmp_path, (13, 26, 39))


def test_verify_rejects_missing_checkpoint(tmp_path: Path) -> None:
    _checkpoint(tmp_path, 13, 1.0)
    _checkpoint(tmp_path, 39, 3.0)
    with pytest.raises(ValueError, match="missing checkpoint"):
        verify_checkpoints(tmp_path, (13, 26, 39))
