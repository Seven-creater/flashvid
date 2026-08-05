from __future__ import annotations

from pathlib import Path
import shutil
import subprocess

import pytest


def _text(path: str) -> str:
    return Path(path).read_text(encoding="utf-8")


def test_training_lock_pins_cuda124_and_qwen35_official_stack() -> None:
    lock = _text("configs/training/qwen35_9b_sft_cuda124.lock.txt")
    expected = {
        "torch==2.6.0+cu124",
        "torchvision==0.21.0+cu124",
        "torchaudio==2.6.0+cu124",
        "triton==3.2.0",
        "ms-swift==4.4.2",
        "transformers==5.9.0",
        "qwen-vl-utils==0.0.14",
        "ninja==1.13.0",
        "packaging==25.0",
        "flash-linear-attention==0.4.2",
        "fla-core==0.4.2",
    }
    assert expected <= set(lock.splitlines())
    assert "causal-conv1d==" not in lock
    assert "flash-attn==" not in lock


def test_environment_installer_is_python312_and_venv_isolated() -> None:
    text = _text("scripts/install_ms_swift_442.sh")
    assert 'PYTHON312="${PYTHON312:-python3.12}"' in text
    assert ".venv-qwen35-sft-cu124" in text
    assert "--system-site-packages" not in text
    assert "include-system-site-packages = false" in text
    assert "https://mirrors.aliyun.com/pytorch-wheels/cu124" in text
    assert "https://pypi.tuna.tsinghua.edu.cn/packages/60/ee/" in text
    assert 'PYPI_INDEX_URL="${PYPI_INDEX_URL:-https://pypi.tuna.tsinghua.edu.cn/simple}"' in text
    assert '[[ "$HF_ENDPOINT" == "https://hf-mirror.com" ]]' in text
    assert '[[ "$PYPI_INDEX_URL" == "https://pypi.tuna.tsinghua.edu.cn/simple" ]]' in text
    assert "export PIP_CONFIG_FILE=/dev/null" in text
    assert "unset PIP_EXTRA_INDEX_URL PIP_FIND_LINKS PIP_NO_INDEX" in text
    assert "NoRedirect" in text
    assert '"$1" == "--audit-only"' in text
    assert "domestic mirror audit passed; no packages were downloaded" in text
    assert 'allowed_hosts = {"mirrors.aliyun.com", "pypi.tuna.tsinghua.edu.cn"}' in text
    assert "a393b506844035c0dac2f30ea8478c343b8e95a429f06f3b3cadfc7f53adb597" in text
    assert "expected triton 3.2.0" in text
    assert 'HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"' in text
    assert "c08be006ce4dbe1be81f54938ee8e6fc7968cfba397c8d06c7669e97b8c44c0d" in text
    assert '"${FLA_WHEEL_URL}#sha256=${FLA_WHEEL_SHA256}"' in text
    assert "--no-build-isolation" not in text
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


def test_training_launcher_supports_official_eight_gpu_fsdp2() -> None:
    text = _text("scripts/train_qwen_agent_9b_lora.sh")
    assert 'USE_FSDP2="${USE_FSDP2:-0}"' in text
    assert 'USE_FSDP2=1 requires CUDA_VISIBLE_DEVICES=$eight_gpu_set' in text
    assert '--fsdp fsdp2' in text
    assert '--lora_dtype float32' in text
    assert '--fp16 false' in text
    assert '--bf16 true' in text
    assert "model_load_dtype=float32" in text
    assert '--torch_dtype "$model_load_dtype"' in text
    assert '"${distributed_args[@]}"' in text


def test_training_launcher_preserves_lora_and_loss_contract() -> None:
    text = _text("scripts/train_qwen_agent_9b_lora.sh")
    for expected in (
        "--max_length 16384",
        "--freeze_vit true",
        "--freeze_aligner true",
        "--lora_rank 16",
        "--lora_alpha 32",
        "--lora_dropout 0.05",
        "--attn_impl sdpa",
        "verify_swift_loss_mask.py",
        "preflight_qwen35_9b_lora.py",
        "smoke_one_sample.jsonl",
        "--max_steps 1",
        "--logging_steps 1",
        "--save_steps 1",
        "verify_qwen35_lora_smoke.py",
        "training_update.json",
        "--formal-output-dir",
        "--smoke-report",
        "qwen_sft_smoke_gate.py",
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
    assert "naive_recurrent_gated_delta_rule" in python
    assert "FLA numerical smoke" in python
    assert '"relative_l1": errors' in python
    assert "AutoModelForImageTextToText.from_pretrained" in python
    assert 'device_map="auto"' in python


def test_smoke_and_formal_runs_are_explicitly_bound() -> None:
    launcher = _text("scripts/train_qwen_agent_9b_lora.sh")
    assert "--smoke requires --formal-output-dir" in launcher
    assert "formal training requires --smoke-report" in launcher
    assert 'qwen_sft_smoke_gate.py" bind' in launcher
    assert 'qwen_sft_smoke_gate.py" check' in launcher


def test_train_eval_can_send_final_epoch_directly_to_test300() -> None:
    launcher = _text("scripts/run_fast_hybrid_train_eval.sh")
    assert 'DIRECT_TEST_AFTER_TRAIN="${DIRECT_TEST_AFTER_TRAIN:-0}"' in launcher
    assert "select_final_epoch_for_direct_test" in launcher
    assert "final_epoch_direct_test_no_dev_selection" in launcher
    assert 'expected_epoch = max(' in launcher
    assert 'CURRENT_STAGE=checkpoint_dev' in launcher


def test_fsdp2_one_shot_uses_all_gpus_then_test300() -> None:
    text = _text("scripts/run_fast_hybrid_fsdp2_train_test.sh")
    assert "export USE_FSDP2=1" in text
    assert "export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7" in text
    assert "export DIRECT_TEST_AFTER_TRAIN=1" in text
    assert "fsdp2_smoke_metadata.json" in text
    assert "qwen_sft_smoke_gate.py check" in text
    assert "run_fast_hybrid_train_eval.sh" in text


@pytest.mark.skipif(shutil.which("bash") is None, reason="bash is unavailable")
def test_training_shells_parse() -> None:
    for path in (
        "scripts/install_ms_swift_442.sh",
        "scripts/preflight_qwen35_9b_lora.sh",
        "scripts/train_qwen_agent_9b_lora.sh",
        "scripts/run_fast_hybrid_fsdp2_train_test.sh",
        "scripts/serve_qwen_agent.sh",
        "scripts/stop_qwen_agent.sh",
    ):
        subprocess.run(["bash", "-n", path], check=True)
