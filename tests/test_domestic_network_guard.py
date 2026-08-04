from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess

import pytest

from scripts import benchmark_openai, smoke_openai


def _text(path: str) -> str:
    return Path(path).read_text(encoding="utf-8")


@pytest.mark.parametrize("module", [smoke_openai, benchmark_openai])
def test_openai_clients_reject_remote_media_by_default(module: object) -> None:
    with pytest.raises(ValueError, match=r"remote HTTP\(S\) media is disabled"):
        module.validate_media_reference(
            "https://example.invalid/video.mp4",
            allow_remote=False,
        )
    module.validate_media_reference(
        "file:///data02/videos/example.mp4",
        allow_remote=False,
    )
    module.validate_media_reference(
        "https://example.invalid/video.mp4",
        allow_remote=True,
    )


def test_installers_pin_domestic_sources_and_ignore_hidden_pip_sources() -> None:
    for path in (
        "scripts/setup_server.sh",
        "scripts/install_flash_attn.sh",
        "scripts/download_model.sh",
    ):
        text = _text(path)
        assert "https://pypi.tuna.tsinghua.edu.cn/simple" in text
        assert "PIP_CONFIG_FILE=/dev/null" in text
        assert "unset PIP_EXTRA_INDEX_URL PIP_FIND_LINKS PIP_NO_INDEX" in text
    setup = _text("scripts/setup_server.sh")
    assert "https://mirrors.tuna.tsinghua.edu.cn/anaconda/" in setup
    assert "--override-channels" in setup
    assert "anaconda::" not in setup
    assert 'MODELSCOPE_DOMAIN="www.modelscope.cn"' in _text(
        "scripts/download_model.sh"
    )


@pytest.mark.parametrize(
    "path",
    [
        "scripts/serve_qwen35_9b.sh",
        "scripts/serve_dp8.sh",
        "scripts/serve_qwen_agent.sh",
    ],
)
def test_model_servers_are_offline_and_require_local_model_files(path: str) -> None:
    text = _text(path)
    assert "HF_HUB_OFFLINE=1" in text
    assert "TRANSFORMERS_OFFLINE=1" in text
    assert 'HF_ENDPOINT="https://hf-mirror.com"' in text
    assert "existing local model directory with config.json" in text
    assert "Hugging Face IDs are forbidden" in text


@pytest.mark.skipif(shutil.which("bash") is None, reason="bash is unavailable")
@pytest.mark.parametrize(
    ("path", "args", "environment"),
    [
        (
            "scripts/serve_qwen35_9b.sh",
            ["Qwen/nonexistent-domestic-guard-test"],
            {},
        ),
        (
            "scripts/serve_dp8.sh",
            [],
            {"MODEL_DIR": "Qwen/nonexistent-domestic-guard-test"},
        ),
        (
            "scripts/serve_qwen_agent.sh",
            ["Qwen/nonexistent-domestic-guard-test", "test-model"],
            {},
        ),
    ],
)
def test_model_servers_reject_hugging_face_ids_before_launch(
    path: str,
    args: list[str],
    environment: dict[str, str],
) -> None:
    result = subprocess.run(
        ["bash", path, *args],
        env={**os.environ, **environment},
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 2
    assert "Hugging Face IDs are forbidden" in result.stderr
