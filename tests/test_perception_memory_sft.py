from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import pytest

import flashvid_eval.perception_memory_sft as perception_memory_sft_module
from flashvid_eval.client import ChatResult
from flashvid_eval.perception_memory_eva import (
    EvidenceEvent,
    EvidenceMemory,
    OptionLedger,
    PERCEPTION_NORMALIZATION_VERSION,
    PerceptionMemoryEvaEvaluator,
    build_role_separated_controller_messages,
)
from flashvid_eval.perception_memory_sft import (
    QUALITY_CONTRACT_VERSION,
    VISUAL_PATH_CLASSIFIER_VERSION,
    VISUAL_PATH_FAMILIES,
    build_perception_memory_sft_records,
    classify_visual_path,
    enforce_perception_memory_selection_gate,
    summarize_perception_memory_sft,
)
from flashvid_eval.qwen_agents.core import FrameObservation, FrameRequest
from flashvid_eval.schemas import ModelSample


def _json(value: dict) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _observation(interval: tuple[float, float], *, sufficient: bool) -> dict:
    start, end = interval
    return {
        "interval": [start, end],
        "timestamped_facts": [
            {"time": start + 1.0, "fact": "A person opens the left door."}
        ],
        "option_evidence": {
            "A": {"supports": [], "contradicts": ["The right door stays closed."]},
            "B": {"supports": ["The left door visibly opens."], "contradicts": []},
        },
        "temporal_changes": ["closed -> open"],
        "unresolved": [] if sufficient else ["What happens after the door opens?"],
        "evidence_sufficient": sufficient,
        "next_evidence_needed": "" if sufficient else "Inspect the following interval.",
    }


def _messages(text: str, frame: Path | None = None) -> list[dict]:
    content: str | list[dict]
    if frame is None:
        content = text
    else:
        content = [
            {"type": "text", "text": text},
            {"type": "image_url", "image_url": {"url": frame.as_uri()}},
        ]
    return [
        {"role": "system", "content": "Follow the frozen role protocol."},
        {"role": "user", "content": content},
    ]


def _request(
    stage: str,
    step_index: int,
    messages: list[dict],
    content: str,
    marker: str,
    *,
    prefix_index: int | None = None,
) -> dict:
    request = {
        "stage": stage,
        "step_index": step_index,
        "prefix_index": step_index if prefix_index is None else prefix_index,
        "prompt_hash": marker * 64,
        "seed": 17,
        "attempt_index": 0,
        "messages": messages,
        "content": content,
        "finish_reason": "stop",
        "usage": {"total_tokens": 10},
    }
    if stage in {"controller", "planner", "confirmation_controller"}:
        request["action_accepted"] = True
    return request


def _trajectory(tmp_path: Path) -> dict:
    first = (tmp_path / "first.jpg").resolve()
    second = (tmp_path / "second.jpg").resolve()
    first.write_bytes(b"first")
    second.write_bytes(b"second")
    observation_0 = _observation((0.0, 10.0), sufficient=False)
    observation_1 = _observation((10.0, 20.0), sufficient=True)
    tool = (
        '<tool_call>{"tool":"frame_select","arguments":'
        '{"start_time":10,"end_time":20,"nframes":8,"resize":0.75,'
        '"evidence_request":"Observe what happens after the door opens."}}'
        "</tool_call>"
    )
    initial_tool = tool.replace(
        '"start_time":10,"end_time":20', '"start_time":0,"end_time":10'
    )
    states = [
        {
            "step_index": 0,
            "request": {
                "start_time": 0.0,
                "end_time": 10.0,
                "nframes": 8,
                "resize": 0.75,
                "evidence_request": "Observe what happens after the door opens.",
            },
            "frame_paths": [str(first)],
            "timestamps": [1.0],
            "perception_response": observation_0,
            "memory_after": {
                "event_ledger": [
                    {
                        "id": "e0",
                        "interval": [0.0, 10.0],
                        "timestamp": 1.0,
                        "fact": "The left door opens.",
                        "source": "step-0",
                    }
                ],
                "option_ledger": {
                    "A": {"supports": [], "contradicts": ["e0"]},
                    "B": {"supports": ["e0"], "contradicts": []},
                },
                "unresolved": ["What happens next?"],
            },
            "evidence_complete": False,
        },
        {
            "step_index": 1,
            "request": {
                "start_time": 10.0,
                "end_time": 20.0,
                "nframes": 8,
                "resize": 0.75,
                "evidence_request": "Observe what happens after the door opens.",
            },
            "frame_paths": [str(second)],
            "timestamps": [11.0],
            "perception_response": observation_1,
            "memory_after": {
                "event_ledger": [
                    {
                        "id": "e0",
                        "interval": [0.0, 10.0],
                        "timestamp": 1.0,
                        "fact": "The left door opens.",
                        "source": "step-0",
                    },
                    {
                        "id": "e1",
                        "interval": [10.0, 20.0],
                        "timestamp": 11.0,
                        "fact": "The person enters through the left door.",
                        "source": "step-1",
                    },
                ],
                "option_ledger": {
                    "A": {"supports": [], "contradicts": ["e0", "e1"]},
                    "B": {"supports": ["e0", "e1"], "contradicts": []},
                },
                "unresolved": [],
            },
            "evidence_complete": True,
            "judge_confirmations": [
                {
                    "judge_seed": seed,
                    "prediction": "B",
                    "evidence_ids": ["e0", "e1"],
                    "evidence_complete": True,
                    "annotation_leak_check": "passed",
                    "request_messages": _messages(
                        "Choose from the options using only the evidence ledger."
                    ),
                    "raw_response": ('{"answer":"B","evidence_ids":["e0","e1"]}'),
                    "finish_reason": "stop",
                    "usage": {"total_tokens": 10},
                }
                for seed in (17, 42, 73)
            ],
        },
    ]
    return {
        "dataset": "lvbench",
        "sample_id": "sample-1",
        "trajectory_id": "lvbench:sample-1:family:0",
        "family_id": "family",
        "perception_normalization_version": PERCEPTION_NORMALIZATION_VERSION,
        "public_sample": {
            "dataset": "lvbench",
            "sample_id": "sample-1",
            "video": "video.mp4",
            "question": "What does the person do?",
            "choices": {"A": "waits", "B": "opens the door"},
        },
        "manifest_sha256": "a" * 64,
        "train600_manifest_sha256": "a" * 64,
        "dataset_manifest_sha256": "b" * 64,
        "config_sha256": "c" * 64,
        "_selection_stable": True,
        "annotation_leak_check": "passed",
        "candidate_rerun": 0,
        "candidate_answer": "A",
        "prediction": "B",
        "final_prediction": "B",
        "total_tokens": 100,
        "visual_tokens": 60,
        "tool_steps": [
            {
                "request": deepcopy(states[0]["request"]),
                "start_time": 0.0,
                "end_time": 10.0,
                "nframes": 1,
                "resize": 1.0,
                "actual_timestamps": [1.0],
                "frame_paths": [str(first)],
            },
            {
                "request": deepcopy(states[1]["request"]),
                "start_time": 10.0,
                "end_time": 20.0,
                "nframes": 1,
                "resize": 1.0,
                "actual_timestamps": [11.0],
                "frame_paths": [str(second)],
            },
        ],
        "perception_states": states,
        "request_trace": [
            _request(
                "controller",
                0,
                _messages("The evidence ledger is empty; choose the first interval."),
                initial_tool,
                "3",
                prefix_index=-1,
            ),
            _request(
                "perception",
                0,
                _messages("Observe only this interval.", first),
                _json(observation_0),
                "d",
            ),
            _request(
                "controller",
                1,
                _messages("Use the current text evidence memory."),
                tool,
                "e",
                prefix_index=0,
            ),
            _request(
                "perception",
                1,
                _messages("Observe only this interval.", second),
                _json(observation_1),
                "f",
            ),
            _request(
                "controller",
                2,
                _messages("Use the current text evidence memory."),
                '{"action":"stop"}',
                "4",
                prefix_index=1,
            ),
            _request(
                "completeness",
                2,
                _messages("Check whether the evidence is complete."),
                '{"evidence_complete":true,"missing_evidence":[]}',
                "1",
                prefix_index=1,
            ),
            _request(
                "evidence_judge",
                2,
                _messages("Choose from the options using only the evidence ledger."),
                '{"answer":"B","evidence_ids":["e0","e1"]}',
                "2",
                prefix_index=1,
            ),
        ],
    }


def _add_visual_csv_confirmations(trajectory: dict) -> None:
    prediction = str(trajectory["final_prediction"])
    for state in trajectory["perception_states"]:
        complete = bool(state["evidence_complete"])
        state["visual_csv_confirmations"] = [
            {
                "judge_seed": seed,
                "prediction": prediction if complete else "A",
                "frame_indices": [0],
                "candidate_blind": True,
                "parsed_valid": True,
                "annotation_leak_check": "passed",
                "error": None,
            }
            for seed in (17, 42, 73)
        ]


def _bind_visual_csv_labels(trajectory: dict) -> None:
    _add_visual_csv_confirmations(trajectory)
    trajectory["offline_label_join"] = (
        "ground_truth_used_for_boolean_only_not_serialized"
    )
    for state in trajectory["perception_states"]:
        for field in ("evidence_sufficient", "next_evidence_needed"):
            state["perception_response"].pop(field, None)
        prefix_index = state["step_index"]
        perception_request = next(
            request
            for request in trajectory["request_trace"]
            if request["stage"] == "perception"
            and request["prefix_index"] == prefix_index
        )
        perception_payload = json.loads(perception_request["content"])
        for field in ("evidence_sufficient", "next_evidence_needed"):
            perception_payload.pop(field, None)
        perception_request["content"] = _json(perception_payload)
        state["completion_gate_kind"] = "visual_csv_3of3_offline_label"
        state["visual_csv_source_sha256"] = "a" * 64
        state["visual_csv_config_sha256"] = "b" * 64
    # Rebuild the exact explicit-role runtime snapshots: accepted Planner
    # user/action pairs followed by the latest full-ledger user snapshot.
    controller_requests = [
        request
        for request in trajectory["request_trace"]
        if request["stage"] == "controller"
    ]
    public = trajectory["public_sample"]
    sample = ModelSample(
        dataset=public["dataset"],
        sample_id=public["sample_id"],
        video=public["video"],
        question=public["question"],
        choices=public["choices"],
        candidate_answer=None,
    )
    memory = EvidenceMemory(sample.option_letters)
    accepted_history: list[tuple[dict, str]] = []
    for index, request in enumerate(controller_requests):
        if index:
            raw_memory = trajectory["perception_states"][index - 1]["memory_after"]
            memory = EvidenceMemory(
                sample.option_letters,
                event_ledger=[
                    EvidenceEvent(
                        item.get("evidence_id", item.get("id")),
                        tuple(item["interval"]),
                        item["timestamp"],
                        item["fact"],
                        item["source"],
                    )
                    for item in raw_memory["event_ledger"]
                ],
                option_ledger={
                    letter: OptionLedger(
                        tuple(raw_memory["option_ledger"][letter]["supports"]),
                        tuple(raw_memory["option_ledger"][letter]["contradicts"]),
                    )
                    for letter in sample.option_letters
                },
                unresolved=list(raw_memory["unresolved"]),
                observed_intervals=[
                    tuple(item["interval"])
                    for item in raw_memory["event_ledger"]
                ],
            )
        request["messages"] = build_role_separated_controller_messages(
            sample,
            memory,
            {"duration": 20.0, "width": 0, "height": 0},
            accepted_history,
        )
        is_tool = "<tool_call>" in request["content"]
        if (
            request.get("action_accepted") is True
            and is_tool
        ):
            canonical_action = _json(json.loads(request["content"][11:-12]))
            canonical_action = f"<tool_call>{canonical_action}</tool_call>"
            accepted_history.append(
                (deepcopy(request["messages"][-1]), canonical_action)
            )


def test_process_export_builds_role_specific_complete_episodes(
    tmp_path: Path,
) -> None:
    trajectory = _trajectory(tmp_path)
    records = build_perception_memory_sft_records(
        trajectory, include_observer=True
    )

    assert [record["metadata"]["process_role"] for record in records] == [
        "planner",
        "observer",
        "observer",
    ]
    planner, observer_0, observer_1 = records
    assert planner["metadata"]["assistant_target_types"] == [
        "tool",
        "tool",
        "stop",
    ]
    assert observer_0["metadata"]["assistant_target_types"] == ["memory"]
    assert observer_1["metadata"]["assistant_target_types"] == ["memory"]
    assert "images" not in planner
    assert observer_0["images"] == [trajectory["tool_steps"][0]["frame_paths"][0]]
    assert observer_1["images"] == [trajectory["tool_steps"][1]["frame_paths"][0]]
    assert all(
        record["metadata"]["episode_schema"] == "observer_current_frame_episode_v1"
        for record in (observer_0, observer_1)
    )
    assert sum(message["role"] == "system" for message in planner["messages"]) == 1
    assert all(
        sum(message["role"] == "system" for message in record["messages"]) == 1
        for record in (observer_0, observer_1)
    )
    assert planner["messages"][-1] == {
        "role": "assistant",
        "content": '{"action":"stop"}',
        "loss": True,
    }
    assert all(
        message.get("loss") is True
        for record in records
        for message in record["messages"]
        if message["role"] == "assistant"
    )


def test_visual_csv_controls_continue_stop_and_masks_observer_sufficiency(
    tmp_path: Path,
) -> None:
    trajectory = _trajectory(tmp_path)
    _bind_visual_csv_labels(trajectory)
    # The role-separated Observer has no stopping fields; only offline visual
    # CSV assigns CONTINUE/STOP.
    assert all(
        "evidence_sufficient" not in state["perception_response"]
        and "next_evidence_needed" not in state["perception_response"]
        for state in trajectory["perception_states"]
    )
    # The recorded stop may be truncated; a 3/3 complete prefix still creates
    # the stop target from a fresh full-ledger Controller snapshot.
    stop_request = next(
        request
        for request in trajectory["request_trace"]
        if request["stage"] == "controller" and request["prefix_index"] == 1
    )
    stop_request["finish_reason"] = "length"

    planner, *observers = build_perception_memory_sft_records(
        trajectory,
        include_observer=True,
        completion_gate_kind="visual_csv",
    )

    assert planner["metadata"]["completion_gate_kind"] == "visual_csv"
    assert planner["metadata"]["assistant_target_types"] == ["tool", "tool", "stop"]
    memory_targets = [
        json.loads(message["content"])
        for observer in observers
        for message in observer["messages"]
        if message["role"] == "assistant"
    ]
    assert all("evidence_sufficient" not in target for target in memory_targets)
    assert all("next_evidence_needed" not in target for target in memory_targets)
    assert all(
        observer["metadata"]["observer_sufficiency_fields_masked"] is True
        for observer in observers
    )
    assert all(
        observer["metadata"]["episode_turns"][0]["target_origin"]
        == "accepted_observer_trace_sufficiency_masked"
        for observer in observers
    )
    assert planner["metadata"]["episode_turns"][-1]["target_origin"] == (
        "offline_visual_csv_stop"
    )
    assert all(
        len(turn["prompt_hash"]) == 64 and turn["target_origin"]
        for record in (planner, *observers)
        for turn in record["metadata"]["episode_turns"]
    )


def test_visual_csv_rejects_non_visual_confirmation(tmp_path: Path) -> None:
    trajectory = _trajectory(tmp_path)
    _bind_visual_csv_labels(trajectory)
    del trajectory["perception_states"][0]["visual_csv_confirmations"][0][
        "frame_indices"
    ]

    with pytest.raises(ValueError, match="candidate-blind frame_indices"):
        build_perception_memory_sft_records(
            trajectory, completion_gate_kind="visual_csv"
        )


def test_visual_csv_rejects_unbound_offline_labels(tmp_path: Path) -> None:
    trajectory = _trajectory(tmp_path)
    _add_visual_csv_confirmations(trajectory)
    for state in trajectory["perception_states"]:
        for field in ("evidence_sufficient", "next_evidence_needed"):
            state["perception_response"].pop(field, None)

    with pytest.raises(ValueError, match="offline label join"):
        build_perception_memory_sft_records(
            trajectory, completion_gate_kind="visual_csv"
        )


def test_incomplete_prefix_cannot_train_stop_or_final(tmp_path: Path) -> None:
    trajectory = _trajectory(tmp_path)
    trajectory["request_trace"][2]["content"] = '{"answer":"B"}'

    with pytest.raises(
        ValueError,
        match="accepted Planner frame_select count differs|requires one next tool/plan target",
    ):
        build_perception_memory_sft_records(trajectory)


def test_rejected_stop_is_not_exported_from_incomplete_prefix(tmp_path: Path) -> None:
    trajectory = _trajectory(tmp_path)
    rejected = _request(
        "controller",
        1,
        _messages("Use the current text evidence memory."),
        '{"action":"stop"}',
        "5",
        prefix_index=0,
    )
    rejected["action_accepted"] = False
    rejected["rejection_reason"] = "missing visual evidence"
    trajectory["request_trace"].insert(2, rejected)

    records = build_perception_memory_sft_records(trajectory)

    assert records[0]["metadata"]["assistant_target_types"].count("stop") == 1
    assert records[0]["metadata"]["episode_turns"][-1]["prefix_index"] == 1


def test_complete_prefix_requires_three_unique_complete_judges(tmp_path: Path) -> None:
    trajectory = _trajectory(tmp_path)
    trajectory["perception_states"][-1]["judge_confirmations"].pop()

    with pytest.raises(ValueError, match="exactly 3 completion confirmations"):
        build_perception_memory_sft_records(trajectory)


def test_only_resolved_interval_may_differ_from_raw_perception(tmp_path: Path) -> None:
    trajectory = _trajectory(tmp_path)
    raw = json.loads(trajectory["request_trace"][1]["content"])
    raw["interval"] = [0.25, 9.75]
    trajectory["request_trace"][1]["content"] = _json(raw)
    build_perception_memory_sft_records(trajectory)

    raw["timestamped_facts"][0]["fact"] = "A different unsupported fact."
    trajectory["request_trace"][1]["content"] = _json(raw)
    with pytest.raises(ValueError, match="differs from the actual model response"):
        build_perception_memory_sft_records(trajectory)


def test_perception_export_strictly_binds_public_choices_and_tool_step(
    tmp_path: Path,
) -> None:
    trajectory = _trajectory(tmp_path)
    trajectory["request_trace"][1]["content"] = (
        "explanation\n" + trajectory["request_trace"][1]["content"]
    )
    with pytest.raises(ValueError, match="failed the frozen parser"):
        build_perception_memory_sft_records(trajectory)

    trajectory = _trajectory(tmp_path)
    trajectory["public_sample"]["choices"]["C"] = "leaves"
    with pytest.raises(ValueError, match="failed the frozen parser"):
        build_perception_memory_sft_records(trajectory)

    trajectory = _trajectory(tmp_path)
    trajectory["tool_steps"][0]["actual_timestamps"] = [2.0]
    with pytest.raises(ValueError, match="state/tool timestamps differ"):
        build_perception_memory_sft_records(trajectory)

    trajectory = _trajectory(tmp_path)
    trajectory["tool_steps"][0]["frame_paths"] = [
        trajectory["tool_steps"][1]["frame_paths"][0]
    ]
    with pytest.raises(ValueError, match="state/tool frame paths differ"):
        build_perception_memory_sft_records(trajectory)


@pytest.mark.parametrize(
    ("request_source", "field", "value"),
    (
        ("controller", "nframes", 4),
        ("state", "resize", 0.5),
        ("tool", "evidence_request", "A different visual objective."),
    ),
)
def test_frame_select_request_binding_is_exact_and_three_way(
    tmp_path: Path,
    request_source: str,
    field: str,
    value: object,
) -> None:
    trajectory = _trajectory(tmp_path)
    # The unmodified fixture is a runtime-shaped accepted request whose
    # Controller, state, and tool request values are exactly identical.
    build_perception_memory_sft_records(trajectory)

    if request_source == "controller":
        payload = json.loads(trajectory["request_trace"][0]["content"][11:-12])
        payload["arguments"][field] = value
        trajectory["request_trace"][0]["content"] = (
            f"<tool_call>{_json(payload)}</tool_call>"
        )
    elif request_source == "state":
        trajectory["perception_states"][0]["request"][field] = value
    else:
        trajectory["tool_steps"][0]["request"][field] = value

    with pytest.raises(
        ValueError,
        match=r"frame_select\[0\] controller/state/tool request binding differs",
    ):
        build_perception_memory_sft_records(trajectory)


@pytest.mark.parametrize(
    ("question", "intervals", "expected"),
    (
        (
            "What does the person do?",
            ((0.0, 10.0),),
            "single_frame_select",
        ),
        (
            "What happens at 00:05?",
            ((0.0, 10.0),),
            "timestamp_grounded_select",
        ),
        # An explicit timestamp does not override multi-step structure.
        (
            "What happens at 00:05?",
            ((0.0, 30.0), (2.0, 12.0), (4.0, 8.0)),
            "hierarchical_refinement",
        ),
        (
            "What happens at 00:02?",
            ((0.0, 5.0), (10.0, 15.0), (20.0, 25.0)),
            "multi_interval_exploration",
        ),
    ),
)
def test_visual_path_classifier_has_frozen_deterministic_precedence(
    question: str,
    intervals: tuple[tuple[float, float], ...],
    expected: str,
) -> None:
    trajectory = {
        "trajectory_id": "classifier-fixture",
        "public_sample": {"question": question},
        "tool_steps": [
            {"resolved_start_time": start, "resolved_end_time": end}
            for start, end in intervals
        ],
        "perception_states": [
            {"evidence_complete": index == len(intervals) - 1}
            for index in range(len(intervals))
        ],
    }

    assert classify_visual_path(trajectory) == expected


def test_verifier_judge_is_not_a_trainable_role_episode(
    tmp_path: Path,
) -> None:
    trajectory = _trajectory(tmp_path)
    trajectory["request_trace"][-1]["content"] = (
        '{"answer":"B","evidence_ids":["e0","e1"]}'
    )

    records = build_perception_memory_sft_records(trajectory)

    assert len(records) == 1
    assert records[0]["metadata"]["process_role"] == "planner"
    assert "final" not in records[0]["metadata"]["assistant_target_types"]
    assert not any(
        '"answer":"B"' in message["content"]
        for message in records[0]["messages"]
        if isinstance(message["content"], str)
    )

    trajectory["perception_states"][-1]["judge_confirmations"][0]["evidence_ids"] = [
        "UNKNOWN"
    ]
    with pytest.raises(ValueError, match="invalid evidence IDs"):
        build_perception_memory_sft_records(trajectory)


def test_first_three_of_three_complete_prefix_stops_without_redundant_tool(
    tmp_path: Path,
) -> None:
    trajectory = _trajectory(tmp_path)
    first = trajectory["perception_states"][0]
    first["evidence_complete"] = True
    first["judge_confirmations"] = [
        {
            "judge_seed": seed,
            "prediction": "B",
            "evidence_ids": ["e0"],
            "evidence_complete": True,
            "annotation_leak_check": "passed",
            "request_messages": _messages("Judge the first evidence prefix."),
            "raw_response": '{"answer":"B","evidence_ids":["e0"]}',
            "finish_reason": "stop",
            "usage": {"total_tokens": 10},
        }
        for seed in (17, 42, 73)
    ]

    records = build_perception_memory_sft_records(trajectory)

    assert len(records) == 1
    assert records[0]["metadata"]["assistant_target_types"] == ["tool", "stop"]
    assert records[0]["metadata"]["terminal_prefix_index"] == 0


def test_complete_prefix_synthesizes_stop_without_using_confirmation_controller(
    tmp_path: Path,
) -> None:
    trajectory = _trajectory(tmp_path)
    first = trajectory["perception_states"][0]
    first["evidence_complete"] = True
    first["judge_confirmations"] = [
        {
            "judge_seed": seed,
            "prediction": "B",
            "evidence_ids": ["e0"],
            "evidence_complete": True,
            "annotation_leak_check": "passed",
            "request_messages": _messages("Judge the first evidence prefix."),
            "raw_response": '{"answer":"B","evidence_ids":["e0"]}',
            "finish_reason": "stop",
            "usage": {"total_tokens": 10},
        }
        for seed in (17, 42, 73)
    ]
    trajectory["request_trace"] = [
        request
        for request in trajectory["request_trace"]
        if not (request["stage"] == "controller" and request["prefix_index"] == 0)
    ]
    trajectory["request_trace"].append(
        _request(
            "confirmation_controller",
            1,
            _messages("Select a confirmation interval; a tool call is mandatory."),
            '<tool_call>{"tool":"frame_select","arguments":'
            '{"start_time":10,"end_time":20,"nframes":8,"resize":1.0}}'
            "</tool_call>",
            "9",
            prefix_index=0,
        )
    )

    records = build_perception_memory_sft_records(trajectory)

    assert len(records) == 1
    assert records[0]["metadata"]["assistant_target_types"] == ["tool", "stop"]
    assert all(
        turn["stage"] != "confirmation_controller"
        for turn in records[0]["metadata"]["episode_turns"]
    )


class _RuntimeClient:
    def __init__(self, outputs: list[str]) -> None:
        self.outputs = outputs

    def chat(self, model: str, messages: list[dict], **kwargs: object) -> ChatResult:
        del model, kwargs
        content = self.outputs.pop(0)
        has_media = any(
            isinstance(message.get("content"), list) for message in messages
        )
        return ChatResult(
            content=content,
            usage={
                "prompt_tokens": 20,
                "completion_tokens": 5,
                "total_tokens": 25,
                "prompt_tokens_details": {"multimodal_tokens": 10 if has_media else 0},
            },
            raw={},
            latency_s=0.01,
            finish_reason="stop",
        )


class _RuntimeFrameTool:
    def __init__(self, video: Path, frame: Path) -> None:
        self.video = video
        self.frame = frame

    def open_session(self, video: Path, session_id: str) -> object:
        assert video == self.video and session_id.startswith("pm-")
        frame = self.frame

        class _Session:
            metadata = {"duration": 30.0, "width": 640, "height": 360}

            def select(self, request: FrameRequest) -> FrameObservation:
                return FrameObservation(
                    request=request,
                    resolved_start_time=0.0,
                    resolved_end_time=10.0,
                    resolved_nframes=1,
                    frame_paths=(str(frame),),
                    timestamps=(1.0,),
                    backend="official_eva_select_frame_fallback",
                    cache_hit=False,
                    estimated_visual_tokens=10,
                    latency_s=0.01,
                )

        return _Session()


def test_persisted_runtime_trace_exports_without_schema_translation(
    tmp_path: Path,
) -> None:
    video = (tmp_path / "video.mp4").resolve()
    frame = (tmp_path / "runtime-frame.jpg").resolve()
    video.write_bytes(b"video")
    frame.write_bytes(b"frame")
    observation = _observation((0.0, 10.0), sufficient=True)
    observation["timestamped_facts"] = [
        {"frame_index": 0, "fact": "A person opens the left door."}
    ]
    tool = (
        '<tool_call>{"tool":"frame_select","arguments":'
        '{"start_time":0,"end_time":10,"nframes":1,"resize":1.0,'
        '"evidence_request":"observe the door"}}</tool_call>'
    )
    client = _RuntimeClient(
        [
            tool,
            _json(observation),
            '{"action":"stop"}',
            '{"evidence_complete":true,"missing_evidence":[]}',
            '{"answer":"A","evidence_ids":["E0001"]}',
        ]
    )
    evaluator = PerceptionMemoryEvaEvaluator(
        client,  # type: ignore[arg-type]
        "Qwen3.5-9B",
        tmp_path,
        tmp_path / "frames",
        frame_tool=_RuntimeFrameTool(video, frame),  # type: ignore[arg-type]
        max_turns=2,
    )
    result = evaluator.run(
        ModelSample(
            dataset="lvbench",
            sample_id="runtime-1",
            video=video.name,
            question="Which door opens?",
            choices={"A": "left", "B": "right"},
            candidate_answer="A",
        )
    )
    trajectory = json.loads(json.dumps(result))
    trajectory.update(
        {
            "trajectory_id": "lvbench:runtime-1:family:0",
            "family_id": "family",
            "manifest_sha256": "a" * 64,
            "train600_manifest_sha256": "a" * 64,
            "dataset_manifest_sha256": "b" * 64,
            "config_sha256": "c" * 64,
            "_selection_stable": True,
        }
    )
    state = trajectory["perception_states"][-1]
    state["judge_confirmations"][0]["evidence_ids"] = ["E0001"]
    state["judge_confirmations"][0]["request_messages"] = deepcopy(
        trajectory["request_trace"][-1]["messages"]
    )
    state["judge_confirmations"][0]["raw_response"] = (
        '{"answer":"A","evidence_ids":["E0001"]}'
    )
    state["judge_confirmations"][0]["finish_reason"] = "stop"
    state["judge_confirmations"][0]["usage"] = {"total_tokens": 10}
    for seed in (42, 73):
        state["judge_confirmations"].append(
            {
                "seed": seed,
                "prediction": "A",
                "evidence_ids": ["E0001"],
                "evidence_complete": True,
                "annotation_leak_check": "passed",
                "request_messages": deepcopy(
                    trajectory["request_trace"][-1]["messages"]
                ),
                "raw_response": ('{"answer":"A","evidence_ids":["E0001"]}'),
                "finish_reason": "stop",
                "usage": {"total_tokens": 10},
                "error": None,
            }
        )
        request = deepcopy(trajectory["request_trace"][-1])
        request["seed"] = seed
        request["prompt_hash"] = str(seed % 10) * 64
        trajectory["request_trace"].append(request)

    records = build_perception_memory_sft_records(
        trajectory, include_observer=True
    )

    assert [record["metadata"]["process_role"] for record in records] == [
        "planner",
        "observer",
    ]
    assert records[0]["metadata"]["assistant_target_types"] == ["tool", "stop"]
    observer = records[1]
    assert json.loads(observer["messages"][-1]["content"])[
        "timestamped_facts"
    ] == [{"fact": "A person opens the left door.", "frame_index": 0}]


def test_export_rejects_private_annotations_and_candidate_leak(tmp_path: Path) -> None:
    trajectory = _trajectory(tmp_path)
    trajectory["request_trace"][2]["messages"][1]["time_range"] = [1, 2]
    with pytest.raises(ValueError, match="private annotation"):
        build_perception_memory_sft_records(trajectory)

    trajectory = _trajectory(tmp_path)
    trajectory["request_trace"][1]["messages"][0]["content"] += " Direct candidate: A"
    with pytest.raises(ValueError, match="candidate leaked"):
        build_perception_memory_sft_records(trajectory)


def test_selection_summary_and_corpus_gate(tmp_path: Path) -> None:
    rows = []
    all_records = []
    for index, dataset in enumerate(("lvbench", "lsdbench", "cgbench")):
        row = deepcopy(_trajectory(tmp_path))
        row["dataset"] = dataset
        row["sample_id"] = f"sample-{index}"
        row["trajectory_id"] = f"{dataset}:sample-{index}:family:0"
        rows.append(row)
        all_records.extend(
            build_perception_memory_sft_records(row, include_observer=True)
        )

    summary = enforce_perception_memory_selection_gate(rows, all_records)

    assert summary["selected_by_dataset"] == {
        "cgbench": 1,
        "lsdbench": 1,
        "lvbench": 1,
    }
    assert summary["candidate_fixes"] == 3
    assert summary["candidate_training_strata"] == {
        "candidate_correct": 0,
        "candidate_wrong": 3,
    }
    assert summary["visual_path_distribution"] == {
        "single_frame_select": 0,
        "timestamp_grounded_select": 0,
        "hierarchical_refinement": 0,
        "multi_interval_exploration": 3,
    }
    assert summary["prefixes"] == {
        "complete": 3,
        "incomplete": 3,
        "unobserved": 3,
    }
    assert summary["assistant_targets"] == {
        "memory": 6,
        "stop": 3,
        "tool": 6,
    }
    assert summary["role_episodes"] == {"observer": 6, "planner": 3}
    assert summary["planner_decisions"] == {
        "observed_incomplete_continue": 3,
        "stop": 3,
    }
    assert summarize_perception_memory_sft(rows)["sft_records"] == 0
    with pytest.raises(ValueError, match="Observer current-frame episodes"):
        summarize_perception_memory_sft(rows, all_records[:-2])

    rows[0]["candidate_answer"] = "B"
    advisory = enforce_perception_memory_selection_gate(rows)
    assert advisory["candidate_fixes"] == 2


def test_selection_summary_rejects_duplicate_samples(tmp_path: Path) -> None:
    row = _trajectory(tmp_path)
    with pytest.raises(ValueError, match="duplicate selected sample"):
        summarize_perception_memory_sft([row, deepcopy(row)])


def _quality_summary() -> dict:
    return {
        "quality_contract_version": QUALITY_CONTRACT_VERSION,
        "visual_path_classifier_version": VISUAL_PATH_CLASSIFIER_VERSION,
        "candidate_training_strata": {
            "candidate_correct": 4,
            "candidate_wrong": 4,
        },
        "planner_decisions": {
            "observed_incomplete_continue": 8,
            "stop": 8,
        },
        "visual_path_distribution": {family: 2 for family in VISUAL_PATH_FAMILIES},
    }


def test_visual_csv_quality_contract_accepts_only_balanced_quality_strata() -> None:
    summary = _quality_summary()
    perception_memory_sft_module._enforce_visual_csv_quality(summary)

    summary = _quality_summary()
    summary["visual_path_classifier_version"] = "visual_path_classifier_v2"
    with pytest.raises(ValueError, match="visual path classifier version drifted"):
        perception_memory_sft_module._enforce_visual_csv_quality(summary)

    summary = _quality_summary()
    summary["candidate_training_strata"] = {
        "candidate_correct": 5,
        "candidate_wrong": 6,
    }
    with pytest.raises(ValueError, match="candidate training strata ratio"):
        perception_memory_sft_module._enforce_visual_csv_quality(summary)

    summary = _quality_summary()
    summary["planner_decisions"]["observed_incomplete_continue"] = 7
    with pytest.raises(ValueError, match="observed CONTINUE/STOP ratio"):
        perception_memory_sft_module._enforce_visual_csv_quality(summary)

    summary = _quality_summary()
    summary["visual_path_distribution"]["single_frame_select"] = 0
    with pytest.raises(ValueError, match="all four non-empty families"):
        perception_memory_sft_module._enforce_visual_csv_quality(summary)

    summary = _quality_summary()
    summary["visual_path_distribution"]["single_frame_select"] = 3
    with pytest.raises(ValueError, match="visual path max/min ratio"):
        perception_memory_sft_module._enforce_visual_csv_quality(summary)


def test_sanitized_candidate_training_stratum_survives_candidate_removal(
    tmp_path: Path,
) -> None:
    trajectory = _trajectory(tmp_path)
    del trajectory["candidate_answer"]
    trajectory["candidate_training_stratum"] = "candidate_wrong"

    summary = summarize_perception_memory_sft([trajectory])

    assert summary["candidate_training_strata"] == {
        "candidate_correct": 0,
        "candidate_wrong": 1,
    }
