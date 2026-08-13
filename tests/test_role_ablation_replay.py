from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from pathlib import Path

import pytest

from flashvid_eval.client import ChatResult
from flashvid_eval.perception_memory_eva import (
    EvidenceMemory,
    build_perception_messages,
    build_perception_retry_messages,
    build_cited_judge_messages,
    perception_response_format,
)
from flashvid_eval.perception_memory_visual_csv import visual_csv_response_format
from flashvid_eval.qwen_agents.core import FrameObservation, FrameRequest
from flashvid_eval.qwen_sft import canonical_sha256
from flashvid_eval.role_ablation_replay import (
    freeze_role_ablation_input,
    run_paired_role_ablation,
    validate_frozen_role_input,
    validate_paired_role_result,
)
from flashvid_eval.schemas import ModelSample
from scripts.run_role_ablation_pair import _frozen_inputs, _run_contract


class _Client:
    def __init__(self, content: str) -> None:
        self.content = content
        self.calls: list[dict] = []

    def chat(self, model: str, messages: list[dict], max_tokens: int = 32, **kwargs):
        self.calls.append(
            {
                "model": model,
                "messages": deepcopy(messages),
                "max_tokens": max_tokens,
                **deepcopy(kwargs),
            }
        )
        return ChatResult(
            content=self.content,
            usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
            raw={},
            latency_s=0.1,
            finish_reason="stop",
        )


def _source(tmp_path: Path) -> dict:
    frame = (tmp_path / "frame.jpg").resolve()
    frame.write_bytes(b"frame-bytes")
    public = {
        "dataset": "lvbench",
        "sample_id": "s1",
        "video": "video.mp4",
        "question": "Which door opens?",
        "choices": {"A": "left", "B": "right"},
    }
    observer_request = FrameRequest(
        start_time=0.0,
        end_time=2.0,
        resize=0.75,
        nframes=1,
        evidence_request="Determine which door opens.",
    )
    observer_messages = build_perception_messages(
        ModelSample(candidate_answer=None, **public),
        FrameObservation(
            request=observer_request,
            resolved_start_time=0.0,
            resolved_end_time=2.0,
            resolved_nframes=1,
            frame_paths=(str(frame),),
            timestamps=(1.0,),
            backend="test",
            cache_hit=True,
            estimated_visual_tokens=0,
            latency_s=0.0,
        ),
        observer_request.evidence_request,
        use_frame_indices=True,
        role_separated=True,
    )
    memory = {
        "event_ledger": [
            {
                "evidence_id": "E0001",
                "interval": [0.0, 2.0],
                "timestamp": 1.0,
                "fact": "The left door opens.",
                "source": "timestamped_fact",
            }
        ],
        "option_ledger": {
            "A": {"supports": ["E0001"], "contradicts": []},
            "B": {"supports": [], "contradicts": ["E0001"]},
        },
        "unresolved": [],
        "observed_intervals": [[0.0, 2.0]],
    }
    answerer_messages = build_cited_judge_messages(
        ModelSample(candidate_answer=None, **public),
        EvidenceMemory.from_dict(memory, ("A", "B")),
        [(str(frame), 1.0)],
        [0],
    )
    perception = {
        "interval": [0.0, 2.0],
        "timestamped_facts": [{"time": 1.0, "fact": "The left door opens."}],
        "option_evidence": {
            "A": {"supports": ["The left door opens."], "contradicts": []},
            "B": {"supports": [], "contradicts": ["The left door opens."]},
        },
        "temporal_changes": [],
        "unresolved": [],
    }
    return {
        "dataset": "lvbench",
        "sample_id": "s1",
        "public_sample": public,
        "run_fingerprint": "a" * 64,
        "annotation_leak_check": "passed",
        "candidate_rerun": 0,
        "candidate_answer": "B",
        "perception_max_tokens": 256,
        "judge_max_tokens": 128,
        "decisive_frame_indices": [0],
        "perception_states": [
            {
                "step_index": 0,
                "request": observer_request.to_tool_arguments(),
                "frame_paths": [str(frame)],
                "timestamps": [1.0],
                "perception_response": perception,
                "memory_after": memory,
            }
        ],
        "request_trace": [
            {
                "stage": "observer",
                "step_index": 0,
                "attempt_index": 0,
                "messages": observer_messages,
                "content": "{}",
                "finish_reason": "stop",
                "seed": 42,
                "max_tokens": 256,
                "response_format": perception_response_format(
                    ("A", "B"), 1, role_separated=True
                ),
            },
            {
                "stage": "answerer",
                "step_index": 0,
                "attempt_index": 0,
                "messages": answerer_messages,
                "content": '{"answer":"A","evidence_ids":["E0001"]}',
                "finish_reason": "stop",
                "seed": 42,
                "max_tokens": 128,
                "response_format": {"type": "json_object"},
            },
        ],
    }


def test_freeze_is_label_free_and_binds_frame_content(tmp_path: Path) -> None:
    source = _source(tmp_path)
    frozen = freeze_role_ablation_input(source)
    serialized = str(frozen)
    assert "candidate_answer" not in serialized
    assert "ground_truth" not in serialized
    assert frozen["labels_serialized"] is False
    assert frozen["observer_calls"][0]["request"]["messages"] == (
        source["request_trace"][0]["messages"]
    )
    validate_frozen_role_input(frozen)

    frame = Path(frozen["observer_calls"][0]["frames"][0]["path"])
    frame.write_bytes(b"changed")
    with pytest.raises(ValueError, match="frame content"):
        validate_frozen_role_input(frozen)


@pytest.mark.parametrize(
    ("message_index", "injected_text"),
    (
        (0, " The frozen Direct candidate is B."),
        (1, ' Previous event_ledger: {"E0001":"left door opens"}.'),
    ),
)
def test_freeze_rejects_noncanonical_observer_text(
    tmp_path: Path, message_index: int, injected_text: str
) -> None:
    source = _source(tmp_path)
    message = source["request_trace"][0]["messages"][message_index]
    if message_index == 0:
        message["content"] += injected_text
    else:
        message["content"][0]["text"] += injected_text

    with pytest.raises(ValueError, match="canonical current-frame prompt"):
        freeze_role_ablation_input(source)


def _source_with_successful_observer_retry(tmp_path: Path) -> dict:
    source = _source(tmp_path)
    prior = source["request_trace"][0]
    base_messages = deepcopy(prior["messages"])
    retry_reason = "finish_reason_length"
    retry_group_id = canonical_sha256(base_messages)
    prior.update(
        {
            "content": '{"interval":[0',
            "finish_reason": "length",
            "prompt_hash": retry_group_id,
            "retry_group_id": retry_group_id,
            "retry_of_attempt": None,
            "retry_reason": retry_reason,
            "retry_triggered": True,
            "failure_class": "model_parse_failure",
            "attempt_error": retry_reason,
        }
    )
    terminal = deepcopy(prior)
    terminal_messages = build_perception_retry_messages(
        base_messages,
        retry_reason,
        valid_letters=("A", "B"),
        frame_count=1,
        interval=(0.0, 2.0),
        role_separated=True,
    )
    terminal.update(
        {
            "attempt_index": 1,
            "messages": terminal_messages,
            "content": "{}",
            "finish_reason": "stop",
            "prompt_hash": canonical_sha256(terminal_messages),
            "retry_of_attempt": 0,
            "retry_reason": retry_reason,
            "retry_triggered": False,
            "failure_class": None,
            "attempt_error": None,
            "max_tokens": 512,
        }
    )
    source["request_trace"].insert(1, terminal)
    return source


def test_freeze_rebuilds_successful_observer_length_retry(tmp_path: Path) -> None:
    source = _source_with_successful_observer_retry(tmp_path)

    frozen = freeze_role_ablation_input(source)

    call = frozen["observer_calls"][0]
    assert call["request"]["messages"] == source["request_trace"][1]["messages"]
    assert call["source_retry_contract"] == {
        "attempt_index": 1,
        "retry_of_attempt": 0,
        "retry_reason": "finish_reason_length",
        "retry_group_id": canonical_sha256(source["request_trace"][0]["messages"]),
        "base_messages_sha256": canonical_sha256(
            source["request_trace"][0]["messages"]
        ),
        "terminal_messages_sha256": canonical_sha256(
            source["request_trace"][1]["messages"]
        ),
    }
    validate_frozen_role_input(frozen)


def test_freeze_rejects_tampered_observer_retry_correction(tmp_path: Path) -> None:
    source = _source_with_successful_observer_retry(tmp_path)
    source["request_trace"][1]["messages"][-1]["content"][-1]["text"] += (
        " Candidate B is preferred."
    )
    source["request_trace"][1]["prompt_hash"] = canonical_sha256(
        source["request_trace"][1]["messages"]
    )

    with pytest.raises(ValueError, match="canonical correction prompt"):
        freeze_role_ablation_input(source)


def test_freeze_rejects_observer_retry_without_prior_attempt(tmp_path: Path) -> None:
    source = _source_with_successful_observer_retry(tmp_path)
    source["request_trace"].pop(0)

    with pytest.raises(ValueError, match="exactly one prior attempt 0"):
        freeze_role_ablation_input(source)


def test_paired_verifier_uses_byte_identical_requests(tmp_path: Path) -> None:
    frozen = freeze_role_ablation_input(_source(tmp_path))
    complete = (
        '{"answer":"A","frame_indices":[0],'
        '"evidence_complete":true,"missing_evidence":[]}'
    )
    control = _Client(complete)
    treatment = _Client(complete)
    result = run_paired_role_ablation(
        frozen,
        role="verifier",
        control_client=control,
        control_model="base",
        control_artifact_sha256="b" * 64,
        treatment_client=treatment,
        treatment_model="lora",
        treatment_artifact_sha256="c" * 64,
        materialized_run_id="verifier_old_lora",
        materialized_run_sha256="d" * 64,
    )
    assert result["call_count"] == 1
    assert control.calls[0]["messages"] == treatment.calls[0]["messages"]
    assert control.calls[0]["response_format"] == visual_csv_response_format(("A", "B"))
    assert "candidate_answer" not in str(control.calls[0]["messages"]).casefold()
    assert "event_ledger" not in str(control.calls[0]["messages"])
    assert result["paired_calls"][0]["control"]["parsed_valid"] is True


def test_paired_observer_output_cannot_change_later_frozen_input(tmp_path: Path) -> None:
    frozen = freeze_role_ablation_input(_source(tmp_path))
    left = (
        '{"interval":[0,2],"timestamped_facts":[{"frame_index":0,'
        '"fact":"left opens"}],"option_evidence":{"A":{"supports":[],'
        '"contradicts":[]},"B":{"supports":[],"contradicts":[]}},'
        '"temporal_changes":[],"unresolved":[]}'
    )
    right = left.replace("left opens", "right opens")
    control = _Client(left)
    treatment = _Client(right)
    result = run_paired_role_ablation(
        frozen,
        role="observer",
        control_client=control,
        control_model="base",
        control_artifact_sha256="b" * 64,
        treatment_client=treatment,
        treatment_model="lora",
        treatment_artifact_sha256="c" * 64,
        materialized_run_id="observer_old_lora",
        materialized_run_sha256="f" * 64,
        downstream_verifier_client=_Client(
            '{"answer":"A","frame_indices":[0],'
            '"evidence_complete":true,"missing_evidence":[]}'
        ),
        downstream_verifier_model="base-verifier",
        downstream_verifier_artifact_sha256="d" * 64,
        downstream_answerer_client=_Client(
            '{"answer":"A","evidence_ids":["E0001"]}'
        ),
        downstream_answerer_model="base-answerer",
        downstream_answerer_artifact_sha256="e" * 64,
    )
    assert control.calls[0]["messages"] == treatment.calls[0]["messages"]
    assert result["paired_calls"][0]["control"]["parsed"] != result["paired_calls"][0]["treatment"]["parsed"]
    assert result["paired_calls"][0]["request_sha256"] == frozen["observer_calls"][0]["request"]["request_sha256"]
    assert result["conditioned_downstream"]["parsed_valid"] is True
    assert set(result["conditioned_downstream"]["answerer"]) == {
        "control",
        "treatment",
    }


def test_answerer_rejects_ledger_or_cited_frame_drift(tmp_path: Path) -> None:
    frozen = freeze_role_ablation_input(_source(tmp_path))
    changed = deepcopy(frozen)
    changed["answerer_call"]["memory"]["unresolved"] = ["changed"]
    changed["frozen_input_sha256"] = canonical_sha256(
        {key: value for key, value in changed.items() if key != "frozen_input_sha256"}
    )
    with pytest.raises(ValueError, match="ledger SHA"):
        validate_frozen_role_input(changed)


def test_freeze_uses_terminal_confirmation_observer_and_answerer(
    tmp_path: Path,
) -> None:
    source = _source(tmp_path)
    observer = deepcopy(source["request_trace"][0])
    observer["stage"] = "confirmation_perception"
    answerer = deepcopy(source["request_trace"][1])
    answerer["stage"] = "confirmation_judge"
    answerer["evidence_memory_at_stage"] = deepcopy(
        source["perception_states"][0]["memory_after"]
    )
    answerer["evidence_memory_sha256"] = canonical_sha256(
        answerer["evidence_memory_at_stage"]
    )
    answerer["cited_frame_indices"] = [0]
    initial_answerer = deepcopy(source["request_trace"][1])
    initial_answerer["attempt_index"] = 2
    source["request_trace"] = [observer, initial_answerer, answerer]
    source["confirmation_decisive_frame_indices"] = [0]

    frozen = freeze_role_ablation_input(source)

    assert len(frozen["observer_calls"]) == 1
    assert frozen["answerer_call"]["source_stage"] == "confirmation_judge"
    assert frozen["answerer_call"]["memory_sha256"] == canonical_sha256(
        source["perception_states"][0]["memory_after"]
    )


def test_paired_result_rejects_invalid_arm(tmp_path: Path) -> None:
    frozen = freeze_role_ablation_input(_source(tmp_path))
    invalid = _Client("not-json")
    with pytest.raises(ValueError, match="no valid terminal response"):
        run_paired_role_ablation(
            frozen,
            role="answerer",
            control_client=invalid,
            control_model="base",
            control_artifact_sha256="b" * 64,
            treatment_client=_Client('{"answer":"A","evidence_ids":["E0001"]}'),
            treatment_model="lora",
            treatment_artifact_sha256="c" * 64,
            materialized_run_id="answerer_old_lora",
            materialized_run_sha256="d" * 64,
        )


def test_answerer_pair_preserves_dev30_row_when_base_never_called_answerer(
    tmp_path: Path,
) -> None:
    source = _source(tmp_path)
    source["request_trace"] = [
        item
        for item in source["request_trace"]
        if item["stage"] not in {"evidence_judge", "answerer", "confirmation_judge"}
    ]
    source["decisive_frame_indices"] = []
    frozen = freeze_role_ablation_input(source)
    assert frozen["answerer_call"] is None
    control = _Client('{"answer":"A","evidence_ids":["E0001"]}')
    treatment = _Client('{"answer":"B","evidence_ids":["E0001"]}')

    result = run_paired_role_ablation(
        frozen,
        role="answerer",
        control_client=control,
        control_model="base",
        control_artifact_sha256="b" * 64,
        treatment_client=treatment,
        treatment_model="lora",
        treatment_artifact_sha256="c" * 64,
        materialized_run_id="answerer_old_lora",
        materialized_run_sha256="d" * 64,
    )

    assert result["call_count"] == 0
    assert result["paired_calls"] == []
    assert result["not_applicable_reason"] == "base_runtime_did_not_reach_answerer"
    assert control.calls == treatment.calls == []
    validate_paired_role_result(result, frozen)

    changed = deepcopy(result)
    changed["not_applicable_reason"] = None
    changed["result_sha256"] = canonical_sha256(
        {key: value for key, value in changed.items() if key != "result_sha256"}
    )
    with pytest.raises(ValueError, match="explicitly not applicable"):
        validate_paired_role_result(changed, frozen)


def test_paired_cli_contract_binds_materialized_run_sha(tmp_path: Path) -> None:
    run_path = tmp_path / "run.json"
    run = {
        "id": "observer_old_lora",
        "execution": {
            "mode": "fixed_observer_pair_with_base_downstream",
            "paired_role": "observer",
            "paired_bindings": {
                "control": {
                    "base_url": "http://127.0.0.1:8200/v1",
                    "model": "Qwen3.5-9B",
                    "artifact_sha256": "b" * 64,
                },
                "treatment": {
                    "base_url": "http://127.0.0.1:8201/v1",
                    "model": "Qwen3.5-9B-old-LoRA",
                    "artifact_sha256": "c" * 64,
                },
            },
        },
    }
    run_path.write_text(json.dumps(run), encoding="utf-8")
    expected = canonical_sha256(run)  # compact canonical bytes differ from the file
    file_sha = hashlib.sha256(run_path.read_bytes()).hexdigest()

    contract = _run_contract(run_path, file_sha)

    assert contract["role"] == "observer"
    assert contract["bindings"]["control"]["artifact_sha256"] == "b" * 64
    assert expected != file_sha
    with pytest.raises(ValueError, match="SHA-256 changed"):
        _run_contract(run_path, "0" * 64)


def test_paired_cli_binds_complete_dev30_input(tmp_path: Path) -> None:
    rows = [
        {"dataset": dataset, "sample_id": f"{dataset}-{index}"}
        for dataset in ("lvbench", "lsdbench", "cgbench")
        for index in range(10)
    ]
    path = tmp_path / "frozen.jsonl"
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    digest = hashlib.sha256(path.read_bytes()).hexdigest()

    assert len(_frozen_inputs(path, digest)) == 30
    with pytest.raises(ValueError, match="SHA-256 changed"):
        _frozen_inputs(path, "0" * 64)

    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows[:-1]), encoding="utf-8"
    )
    truncated_digest = hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="exactly ten"):
        _frozen_inputs(path, truncated_digest)
