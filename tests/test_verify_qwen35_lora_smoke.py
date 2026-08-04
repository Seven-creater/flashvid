from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from scripts.verify_qwen35_lora_smoke import verify_smoke_run

peft = pytest.importorskip("peft")
LoraConfig = peft.LoraConfig


def _smoke_output(
    root: Path,
    *,
    loss: float = 1.25,
    grad_norm: float = 0.5,
    b_weight: float = 0.125,
) -> None:
    checkpoint = root / "checkpoint-1"
    checkpoint.mkdir(parents=True)
    (checkpoint / "trainer_state.json").write_text(
        json.dumps(
            {
                "global_step": 1,
                "log_history": [
                    {"step": 1, "loss": loss, "grad_norm": grad_norm}
                ],
            }
        ),
        encoding="utf-8",
    )
    LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj"],
        task_type="CAUSAL_LM",
    ).save_pretrained(checkpoint)
    save_file(
        {
            "base_model.model.layer.q_proj.lora_A.weight": torch.ones((2, 2)),
            "base_model.model.layer.q_proj.lora_B.weight": torch.full(
                (2, 2), b_weight
            ),
        },
        checkpoint / "adapter_model.safetensors",
    )


def test_verify_smoke_proves_positive_metrics_and_updated_reloadable_adapter(
    tmp_path: Path,
) -> None:
    _smoke_output(tmp_path)
    report = verify_smoke_run(tmp_path)
    assert report["status"] == "passed"
    assert report["global_step"] == 1
    assert report["loss"] == 1.25
    assert report["grad_norm"] == 0.5
    assert report["lora_b_tensor_count"] == 1
    assert report["lora_b_nonzero_values"] == 4
    assert report["peft_reloaded_tensor_count"] == 2


@pytest.mark.parametrize(
    ("loss", "grad_norm", "message"),
    [
        (0.0, 0.5, "loss"),
        (float("nan"), 0.5, "loss"),
        (1.0, 0.0, "grad_norm"),
        (1.0, float("inf"), "grad_norm"),
    ],
)
def test_verify_smoke_rejects_invalid_training_metrics(
    tmp_path: Path, loss: float, grad_norm: float, message: str
) -> None:
    _smoke_output(tmp_path, loss=loss, grad_norm=grad_norm)
    with pytest.raises(ValueError, match=message):
        verify_smoke_run(tmp_path)


def test_verify_smoke_rejects_unchanged_or_nonfinite_lora_b(tmp_path: Path) -> None:
    _smoke_output(tmp_path, b_weight=0.0)
    with pytest.raises(ValueError, match="still zero"):
        verify_smoke_run(tmp_path)

    adapter = tmp_path / "checkpoint-1" / "adapter_model.safetensors"
    save_file(
        {
            "base_model.model.layer.q_proj.lora_A.weight": torch.ones((2, 2)),
            "base_model.model.layer.q_proj.lora_B.weight": torch.tensor(
                [[float("nan"), 0.0], [0.0, 0.0]]
            ),
        },
        adapter,
    )
    with pytest.raises(ValueError, match="NaN or Inf"):
        verify_smoke_run(tmp_path)
