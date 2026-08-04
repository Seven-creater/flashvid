from __future__ import annotations

from pathlib import Path
import shutil
import subprocess

import pytest


def _text(path: str) -> str:
    return Path(path).read_text(encoding="utf-8")


def test_training_lock_pins_cuda121_and_qwen35_official_stack() -> None:
    lock = _text("configs/training/qwen35_9b_sft_cuda121.lock.txt")
    expected = {
        "torch==2.5.1+cu121",
        "torchvision==0.20.1+cu121",
        "torchaudio==2.5.1+cu121",
        "ms-swift==4.4.2",
        "transformers==5.9.0",
        "qwen-vl-utils==0.0.14",
        "ninja==1.13.0",
        "packaging==25.0",
        "flash-linear-attention==0.4.2",
        "causal-conv1d==1.6.2.post1",
        "flash-attn==2.8.3",
    }
    assert expected <= set(lock.splitlines())


def test_environment_installer_is_python312_and_venv_isolated() -> None:
    text = _text("scripts/install_ms_swift_442.sh")
    assert 'PYTHON312="${PYTHON312:-python3.12}"' in text
    assert ".venv-qwen35-sft-cu121" in text
    assert "--system-site-packages" not in text
    assert "include-system-site-packages = false" in text
    assert 'TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://mirror.sjtu.edu.cn/pytorch-wheels/cu121}"' in text
    assert '--index-url "$TORCH_INDEX_URL"' in text
    assert 'HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"' in text
    assert '"torch==2.5.1"' in text
    assert "--no-build-isolation -r \"$LOCK_FILE\"" in text
    assert '"$SWIFT_PYTHON" -m pip check' in text


def test_training_launcher_has_only_registered_gpu_layouts() -> None:
    text = _text("scripts/train_qwen_agent_9b_lora.sh")
    assert 'MIN_GPU_MEMORY_MIB="${MIN_GPU_MEMORY_MIB:-43008}"' in text
    assert 'eight_gpu_set="0,1,2,3,4,5,6,7"' in text
    assert 'four_gpu_set="4,5,6,7"' in text
    assert "gradient_accumulation_steps=4" in text
    assert "gradient_accumulation_steps=8" in text
    assert '--per_device_train_batch_size 1' in text
    assert '--gradient_accumulation_steps "$gradient_accumulation_steps"' in text
    assert "--query-gpu=memory.free" in text
    assert "at least 42 GiB free per GPU" in text
    assert "never signals or otherwise manages foreign PIDs" in text


def test_training_launcher_preserves_lora_and_loss_contract() -> None:
    text = _text("scripts/train_qwen_agent_9b_lora.sh")
    for expected in (
        "--max_length 16384",
        "--freeze_vit true",
        "--freeze_aligner true",
        "--lora_rank 16",
        "--lora_alpha 32",
        "--lora_dropout 0.05",
        "verify_swift_loss_mask.py",
        "preflight_qwen35_9b_lora.py",
        "smoke_one_sample.jsonl",
        "--max_steps 1",
    ):
        assert expected in text


def test_service_release_is_allowlisted_and_owned() -> None:
    stop = _text("scripts/stop_qwen_agent.sh")
    serve = _text("scripts/serve_qwen_agent.sh")
    launcher = _text("scripts/train_qwen_agent_9b_lora.sh")
    assert "8200|8201" in stop
    assert 'process_uid=$(awk' in stop
    assert "FLASHVID_QWEN_OWNER_DIR=$PROJECT_DIR" in stop
    assert 'export FLASHVID_QWEN_OWNER_DIR="$PROJECT_DIR"' in serve
    assert 'stop_qwen_agent.sh" 8200' in launcher
    assert 'stop_qwen_agent.sh" 8201' in launcher
    assert "pkill" not in stop
    assert "kill -9" not in stop


def test_full_preflight_requests_weight_load_and_one_step() -> None:
    wrapper = _text("scripts/preflight_qwen35_9b_lora.sh")
    python = _text("scripts/preflight_qwen35_9b_lora.py")
    assert "--release-project-services" in wrapper
    assert "--load-weights-preflight" in wrapper
    assert "--smoke" in wrapper
    assert "torch.cuda.is_available()" in python
    assert "AutoModelForImageTextToText.from_pretrained" in python
    assert 'device_map="auto"' in python


@pytest.mark.skipif(shutil.which("bash") is None, reason="bash is unavailable")
def test_training_shells_parse() -> None:
    for path in (
        "scripts/install_ms_swift_442.sh",
        "scripts/preflight_qwen35_9b_lora.sh",
        "scripts/train_qwen_agent_9b_lora.sh",
        "scripts/serve_qwen_agent.sh",
        "scripts/stop_qwen_agent.sh",
    ):
        subprocess.run(["bash", "-n", path], check=True)
