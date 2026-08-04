from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from scripts.verify_sft_checkpoints import verify_checkpoints


def _checkpoint(
    root: Path,
    step: int,
    epoch: float,
    *,
    finite: bool = True,
    directory_name: str | None = None,
    global_step: int | None = None,
) -> None:
    path = root / (directory_name or f"checkpoint-{step}")
    path.mkdir(parents=True)
    tensor = torch.ones((2, 2))
    if not finite:
        tensor[0, 0] = float("nan")
    save_file({"adapter.weight": tensor}, path / "adapter_model.safetensors")
    (path / "adapter_config.json").write_text(
        json.dumps({"r": 16, "task_type": "CAUSAL_LM"}), encoding="utf-8"
    )
    (path / "trainer_state.json").write_text(
        json.dumps(
            {"global_step": step if global_step is None else global_step, "epoch": epoch}
        ),
        encoding="utf-8",
    )


def test_verify_completed_sft_checkpoints(tmp_path: Path) -> None:
    for epoch, step in enumerate((13, 26, 39), start=1):
        _checkpoint(tmp_path, step, float(epoch))
    report = verify_checkpoints(tmp_path, (13, 26, 39))
    assert report["status"] == "complete"
    assert [row["global_step"] for row in report["checkpoints"]] == [13, 26, 39]
    assert all(row["tensor_count"] == 1 for row in report["checkpoints"])


def test_verify_discovers_three_checkpoints_in_numeric_step_order(tmp_path: Path) -> None:
    _checkpoint(tmp_path, 30, 3.0)
    _checkpoint(tmp_path, 2, 1.0)
    _checkpoint(tmp_path, 10, 2.0)

    report = verify_checkpoints(tmp_path)

    assert report["expected_steps"] == [2, 10, 30]
    assert [row["name"] for row in report["checkpoints"]] == [
        "checkpoint-2",
        "checkpoint-10",
        "checkpoint-30",
    ]
    assert [row["epoch"] for row in report["checkpoints"]] == [1.0, 2.0, 3.0]


@pytest.mark.parametrize("count", [2, 4])
def test_verify_discovery_requires_exactly_three_checkpoints(
    tmp_path: Path, count: int
) -> None:
    for ordinal in range(1, count + 1):
        _checkpoint(tmp_path, ordinal * 10, float(ordinal))

    with pytest.raises(ValueError, match="expected exactly 3 checkpoint-N directories"):
        verify_checkpoints(tmp_path)


def test_verify_discovery_rejects_directory_and_global_step_mismatch(
    tmp_path: Path,
) -> None:
    _checkpoint(tmp_path, 10, 1.0, global_step=11)
    _checkpoint(tmp_path, 20, 2.0)
    _checkpoint(tmp_path, 30, 3.0)

    with pytest.raises(ValueError, match="expected global_step 10, got 11"):
        verify_checkpoints(tmp_path)


def test_verify_discovery_rejects_duplicate_numeric_steps(tmp_path: Path) -> None:
    _checkpoint(tmp_path, 1, 1.0)
    _checkpoint(tmp_path, 1, 2.0, directory_name="checkpoint-01")
    _checkpoint(tmp_path, 2, 3.0)

    with pytest.raises(ValueError, match="checkpoint steps must be strictly increasing"):
        verify_checkpoints(tmp_path)


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
