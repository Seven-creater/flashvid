from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import pytest

from flashvid_eval.client import ChatResult
from flashvid_eval.perception_memory_eva import (
    PERCEPTION_NORMALIZATION_VERSION,
    PerceptionMemoryEvaEvaluator,
)
from flashvid_eval.perception_memory_sft import (
    build_perception_memory_sft_records,
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
        '{"start_time":10,"end_time":20,"nframes":8,"resize":1.0}}'
        "</tool_call>"
    )
    initial_tool = tool.replace(
        '"start_time":10,"end_time":20', '"start_time":0,"end_time":10'
    )
    states = [
        {
            "step_index": 0,
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
                "start_time": 0.0,
                "end_time": 10.0,
                "nframes": 1,
                "resize": 1.0,
                "actual_timestamps": [1.0],
                "frame_paths": [str(first)],
            },
            {
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


def test_process_export_trains_each_observation_and_only_complete_final(
    tmp_path: Path,
) -> None:
    trajectory = _trajectory(tmp_path)
    records = build_perception_memory_sft_records(trajectory)

    assert [record["metadata"]["episode_target_type"] for record in records] == [
        "tool",
        "memory",
        "tool",
        "memory",
        "stop",
        "final",
    ]
    assert [record["metadata"]["prefix_complete"] for record in records] == [
        False,
        False,
        False,
        True,
        True,
        True,
    ]
    assert records[1]["images"] == [trajectory["tool_steps"][0]["frame_paths"][0]]
    assert records[3]["images"] == [trajectory["tool_steps"][1]["frame_paths"][0]]
    assert all("images" not in records[index] for index in (0, 2, 4, 5))
    assert records[-1]["messages"][-1] == {
        "role": "assistant",
        "content": '{"answer":"B","evidence_ids":["e0","e1"]}',
        "loss": True,
    }
    assert all(
        message.get("loss") is not True
        for record in records
        for message in record["messages"][:-1]
    )


def test_incomplete_prefix_cannot_train_stop_or_final(tmp_path: Path) -> None:
    trajectory = _trajectory(tmp_path)
    trajectory["request_trace"][2]["content"] = '{"answer":"B"}'

    with pytest.raises(ValueError, match="requires one next tool/plan target"):
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

    assert [record["metadata"]["episode_target_type"] for record in records].count(
        "stop"
    ) == 1
    assert all(
        record["metadata"]["prefix_index"] != 0
        for record in records
        if record["metadata"]["episode_target_type"] == "stop"
    )


def test_complete_prefix_requires_three_unique_complete_judges(tmp_path: Path) -> None:
    trajectory = _trajectory(tmp_path)
    trajectory["perception_states"][-1]["judge_confirmations"].pop()

    with pytest.raises(ValueError, match="exactly 3 Judge confirmations"):
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


def test_runtime_judge_with_evidence_ids_preserves_runtime_protocol(
    tmp_path: Path,
) -> None:
    trajectory = _trajectory(tmp_path)
    trajectory["request_trace"][-1]["content"] = (
        '{"answer":"B","evidence_ids":["e0","e1"]}'
    )

    records = build_perception_memory_sft_records(trajectory)

    assert records[-1]["messages"][-1]["content"] == (
        '{"answer":"B","evidence_ids":["e0","e1"]}'
    )

    trajectory["perception_states"][-1]["judge_confirmations"][0]["raw_response"] = (
        '{"answer":"B","evidence_ids":["UNKNOWN"]}'
    )
    with pytest.raises(ValueError, match="runtime-compatible final target"):
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

    assert [record["metadata"]["episode_target_type"] for record in records] == [
        "tool",
        "memory",
        "stop",
        "final",
    ]
    assert all(record["metadata"]["prefix_index"] <= 0 for record in records)


def test_complete_prefix_does_not_synthesize_stop_from_confirmation_controller(
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

    assert [record["metadata"]["episode_target_type"] for record in records] == [
        "tool",
        "memory",
        "final",
    ]


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

    records = build_perception_memory_sft_records(trajectory)

    assert [record["metadata"]["episode_target_type"] for record in records] == [
        "tool",
        "memory",
        "stop",
        "final",
    ]
    assert records[-1]["messages"][-1]["content"] == (
        '{"answer":"A","evidence_ids":["E0001"]}'
    )
    memory_record = next(
        record
        for record in records
        if record["metadata"]["episode_target_type"] == "memory"
    )
    assert json.loads(memory_record["messages"][-1]["content"])[
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
        all_records.extend(build_perception_memory_sft_records(row))

    summary = enforce_perception_memory_selection_gate(
        rows,
        all_records,
        minimum_total=3,
        minimum_per_dataset=1,
        minimum_candidate_fixes=3,
        minimum_candidate_fixes_per_dataset=1,
    )

    assert summary["selected_by_dataset"] == {
        "cgbench": 1,
        "lsdbench": 1,
        "lvbench": 1,
    }
    assert summary["candidate_fixes"] == 3
    assert summary["prefixes"] == {
        "complete": 3,
        "incomplete": 3,
        "unobserved": 3,
    }
    assert summary["assistant_targets"] == {
        "final": 3,
        "memory": 6,
        "stop": 3,
        "tool": 6,
    }
    assert summarize_perception_memory_sft(rows)["sft_records"] == 0
    with pytest.raises(ValueError, match="record coverage differ"):
        summarize_perception_memory_sft(rows, all_records[:-6])

    rows[0]["candidate_answer"] = "B"
    with pytest.raises(ValueError, match="candidate_fixes<3"):
        enforce_perception_memory_selection_gate(
            rows,
            minimum_total=3,
            minimum_per_dataset=1,
            minimum_candidate_fixes=3,
            minimum_candidate_fixes_per_dataset=0,
        )


def test_selection_summary_rejects_duplicate_samples(tmp_path: Path) -> None:
    row = _trajectory(tmp_path)
    with pytest.raises(ValueError, match="duplicate selected sample"):
        summarize_perception_memory_sft([row, deepcopy(row)])
