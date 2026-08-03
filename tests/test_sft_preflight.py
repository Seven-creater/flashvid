from __future__ import annotations

from copy import deepcopy
import json
import os
from pathlib import Path
import shutil
import subprocess

import pytest

from scripts.check_sft_data import validate_sft_data
from scripts.verify_swift_loss_mask import (
    verify_all_record_encodings,
    verify_records_with_template,
)


def _tool_message(ratio: float) -> str:
    payload = {
        "tool": "frame_select",
        "arguments": {
            "start_time": 0,
            "end_time": 10,
            "nframes": 8,
            "resize": 0.5,
            "retention_ratio": ratio,
            "evidence_request": "Inspect the visible event.",
        },
    }
    return f"<tool_call>{json.dumps(payload, separators=(',', ':'))}</tool_call>"


def _messages(ratio: float, answer: str = "A") -> list[dict[str, object]]:
    return [
        {"role": "system", "content": "Use observations only."},
        {"role": "user", "content": "Question and choices"},
        {"role": "assistant", "content": _tool_message(ratio), "loss": True},
        {"role": "tool", "content": '{"observed_facts":["event"]}'},
        {"role": "assistant", "content": f"Answer: {answer}", "loss": True},
    ]


def _fixture_records(
    ratios: list[float],
) -> tuple[list[dict], list[dict], list[dict]]:
    sft: list[dict] = []
    selected: list[dict] = []
    answers: list[dict] = []
    for index, ratio in enumerate(ratios):
        dataset = "demo"
        sample_id = f"s{index}"
        trajectory_id = f"{sample_id}:0"
        messages = _messages(ratio)
        sft.append(
            {
                "messages": messages,
                "metadata": {
                    "dataset": dataset,
                    "sample_id": sample_id,
                    "trajectory_id": trajectory_id,
                    "selection_role": "primary",
                    "retained_visual_tokens": 100,
                },
            }
        )
        selected.append(
            {
                "dataset": dataset,
                "sample_id": sample_id,
                "trajectory_id": trajectory_id,
                "_selection_role": "primary",
                "prediction": "A",
                "final_prediction": "A",
                "trajectory_valid": True,
                "annotation_leak_check": "passed",
                "retained_visual_tokens": 100,
                "budget_sequence": [ratio],
                "training_messages": messages,
            }
        )
        answers.append(
            {
                "dataset": dataset,
                "sample_id": sample_id,
                "answer": "A",
            }
        )
    return sft, selected, answers


def test_sft_preflight_cross_checks_positive_traces_and_budget_mix() -> None:
    sft, selected, answers = _fixture_records([0.1, 0.25, 0.5, 1.0])
    report = validate_sft_data(
        sft,
        selected,
        answers,
        minimum_positive_trajectories=4,
        minimum_budget_levels=3,
        maximum_budget_share=0.70,
    )
    assert report["status"] == "passed"
    assert report["counts"]["verified_positive_trajectories"] == 4
    assert report["budget_distribution"]["tool_call_counts"] == {
        "0.10": 1,
        "0.25": 1,
        "0.50": 1,
        "1.00": 1,
    }


def test_sft_positive_gate_counts_unique_primary_samples_not_secondary_traces() -> None:
    sft, selected, answers = _fixture_records([0.1])
    secondary_sft = deepcopy(sft[0])
    secondary_sft["metadata"].update(
        {
            "trajectory_id": "s0:1",
            "selection_role": "secondary_changed_candidate",
        }
    )
    secondary_selected = deepcopy(selected[0])
    secondary_selected.update(
        {
            "trajectory_id": "s0:1",
            "_selection_role": "secondary_changed_candidate",
        }
    )
    sft.append(secondary_sft)
    selected.append(secondary_selected)

    report = validate_sft_data(
        sft,
        selected,
        answers,
        minimum_positive_trajectories=1,
        minimum_budget_levels=1,
        maximum_budget_share=1.0,
    )
    assert report["status"] == "passed"
    assert report["counts"]["selected_trace_count"] == 2
    assert report["counts"]["verified_positive_selected_traces"] == 2
    assert report["counts"]["verified_positive_primary_samples"] == 1
    assert report["counts"]["verified_positive_trajectories"] == 1

    stricter = validate_sft_data(
        sft,
        selected,
        answers,
        minimum_positive_trajectories=2,
        minimum_budget_levels=1,
        maximum_budget_share=1.0,
    )
    assert stricter["status"] == "failed"
    assert not stricter["constraints"]["positive_trajectories"]["passed"]


def test_sft_preflight_rejects_wrong_selected_trajectory() -> None:
    sft, selected, answers = _fixture_records([0.1, 0.25, 0.5])
    selected[0]["final_prediction"] = "B"
    report = validate_sft_data(
        sft,
        selected,
        answers,
        minimum_positive_trajectories=3,
        minimum_budget_levels=3,
    )
    assert report["status"] == "failed"
    assert any("not positive" in error for error in report["errors"])


def test_sft_preflight_rejects_dominant_budget_and_private_field() -> None:
    sft, selected, answers = _fixture_records([0.1, 0.1, 0.1, 0.25, 0.5])
    sft[0]["metadata"]["time_range"] = [1, 2]
    report = validate_sft_data(
        sft,
        selected,
        answers,
        minimum_positive_trajectories=5,
        minimum_budget_levels=3,
        maximum_budget_share=0.50,
    )
    assert report["status"] == "failed"
    assert not report["constraints"]["maximum_budget_share"]["passed"]
    assert not report["constraints"]["private_fields"]["passed"]


def test_sft_preflight_requires_tool_and_strict_final_targets() -> None:
    sft, selected, answers = _fixture_records([0.1, 0.25, 0.5])
    sft[0]["messages"][-1]["content"] = "I think option A is correct."
    report = validate_sft_data(
        sft,
        selected,
        answers,
        minimum_positive_trajectories=3,
        minimum_budget_levels=3,
    )
    assert report["status"] == "failed"
    assert any("strict final" in error for error in report["errors"])


class _FakeTemplate:
    """Character tokenizer with role-aware labels for differential probes."""

    template_meta = type("Meta", (), {"template_type": "fake-qwen3.5"})()

    def __init__(self, *, leak_tool: bool = False, mask_assistant: bool = False):
        self.leak_tool = leak_tool
        self.mask_assistant = mask_assistant
        self.mode = ""

    def set_mode(self, mode: str) -> None:
        self.mode = mode

    def encode(self, value: dict) -> dict[str, list[int]]:
        input_ids: list[int] = []
        labels: list[int] = []
        for message in value["messages"]:
            role = message["role"]
            token_ids = [ord(char) for char in message["content"]]
            masked = role in {"system", "user", "tool"}
            if role == "tool" and self.leak_tool:
                masked = False
            if role == "assistant" and self.mask_assistant:
                masked = True
            if role == "assistant" and message.get("loss") is False:
                masked = True
            input_ids.extend(token_ids)
            labels.extend([-100] * len(token_ids) if masked else token_ids)
            input_ids.append(10)
            labels.append(-100 if masked else 10)
        return {"input_ids": input_ids, "labels": labels}


def test_real_template_probe_logic_accepts_expected_masks() -> None:
    records = [{"messages": _messages(0.25)}]
    report = verify_records_with_template(records, _FakeTemplate(), sample_count=1)
    assert report["checked_records"] == 1
    assert report["masked_role_probes"] == 3
    assert report["assistant_loss_probes"] == 2
    encoded = verify_all_record_encodings(records, _FakeTemplate())
    assert encoded["encoded_records"] == 1
    assert encoded["maximum_encoded_tokens"] > 0


def test_real_template_probe_honors_explicit_false_assistant_loss() -> None:
    messages = _messages(0.25)
    messages[2]["loss"] = False
    report = verify_records_with_template(
        [{"messages": messages}],
        _FakeTemplate(),
        sample_count=1,
    )
    assert report["assistant_loss_probes"] == 1
    assert report["masked_role_probes"] == 4


@pytest.mark.parametrize(
    "template, expected",
    [
        (_FakeTemplate(leak_tool=True), "role=tool"),
        (_FakeTemplate(mask_assistant=True), "role=assistant"),
    ],
)
def test_real_template_probe_logic_hard_fails_unverified_masks(
    template: _FakeTemplate, expected: str
) -> None:
    with pytest.raises(RuntimeError, match=expected):
        verify_records_with_template(
            [{"messages": _messages(0.5)}],
            template,
            sample_count=1,
        )


def test_training_launcher_uses_stable_checkpoint_root_and_official_tuner_flag() -> None:
    text = Path("scripts/train_flashvid_budget_sft.sh").read_text(encoding="utf-8")
    assert "--tuner_type lora" in text
    assert "--train_type lora" not in text
    assert "--add_version false" in text
    assert "--split_dataset_ratio 0" in text
    assert "--check_model false" in text
    assert "--packing false" in text


def test_selected_sft_controller_serves_base_and_adapter_without_force_kill() -> None:
    text = Path("scripts/selected_sft_controller.sh").read_text(encoding="utf-8")
    assert "Qwen3.5-4B-Agent-SFT-base" in text
    assert "Qwen3.5-4B-Agent-SFT" in text
    assert "--enable-lora" in text
    assert "--max-lora-rank 16" in text
    assert "owned_process" in text
    assert "CHECKPOINT_STATE" in text
    assert "read_checkpoint_state" in text
    assert "write_checkpoint_state" in text
    assert "cleanup_start_failure" in text
    assert 'if [[ "$action" == "start" ]]' in text
    assert 'if [[ "$start_in_progress" -eq 1 ]]' in text
    assert "SIGKILL" in text
    assert "kill -9" not in text

    bash = shutil.which("bash")
    if bash:
        subprocess.run(
            [bash, "-n", "scripts/selected_sft_controller.sh"],
            check=True,
        )


@pytest.mark.skipif(os.name == "nt", reason="requires POSIX process semantics")
def test_selected_sft_controller_stop_does_not_require_selection_or_venv(
    tmp_path: Path,
) -> None:
    bash = shutil.which("bash")
    if not bash:
        pytest.skip("bash is unavailable")
    result = subprocess.run(
        [bash, str(Path("scripts/selected_sft_controller.sh").resolve()), "stop"],
        env={
            **os.environ,
            "STATE_DIR": str(tmp_path / "state"),
            "LOG_DIR": str(tmp_path / "logs"),
            "CHECKPOINT_SELECTION": str(tmp_path / "missing-selection.json"),
            "PYTHON_BIN": str(tmp_path / "missing-python"),
            "VLLM_BIN": str(tmp_path / "missing-vllm"),
            "BASE_MODEL": str(tmp_path / "missing-model"),
        },
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
