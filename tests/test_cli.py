import json

import pytest

from flashvid_vllm.cli import build_vllm_command


def test_cli_maps_retention_to_vllm_pruning():
    command, environment = build_vllm_command(
        [
            "/models/qwen",
            "--vision-retention-ratio",
            "0.1",
            "--data-parallel-size",
            "8",
        ]
    )
    assert command[0].endswith(("vllm", "vllm.exe"))
    assert command[1:3] == ["serve", "/models/qwen"]
    rate_index = command.index("--video-pruning-rate") + 1
    assert float(command[rate_index]) == pytest.approx(0.9)
    override_index = command.index("--hf-overrides") + 1
    assert json.loads(command[override_index])["architectures"] == [
        "FlashVIDQwen3_5ForConditionalGeneration"
    ]
    assert environment["FLASHVID_VISION_RETENTION_RATIO"] == "0.1"
    assert command[-2:] == ["--data-parallel-size", "8"]


@pytest.mark.parametrize("ratio", ["0", "-0.1", "1.1"])
def test_cli_rejects_invalid_ratio(ratio):
    with pytest.raises(SystemExit):
        build_vllm_command(["model", "--vision-retention-ratio", ratio])


def test_cli_rejects_conflicting_internal_flags():
    with pytest.raises(SystemExit):
        build_vllm_command(
            ["model", "--vision-retention-ratio", "0.5", "--video-pruning-rate", "0.2"]
        )
