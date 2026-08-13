from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

from flashvid_eval.client import ChatResult
from flashvid_eval.perception_memory_eva import PerceptionMemoryEvaEvaluator
from flashvid_eval.qwen_agents.core import FrameObservation, FrameRequest
from flashvid_eval.schemas import ModelSample


def _sample(candidate: str = "B") -> ModelSample:
    return ModelSample(
        dataset="lvbench",
        sample_id="runtime-retry",
        video="video.mp4",
        question="What happens after the door opens?",
        choices={"A": "Walks outside", "B": "Sits down"},
        candidate_answer=candidate,
    )


def _state_json(
    *,
    support: str = "A",
    contradict: str = "B",
    interval: tuple[float, float] = (10.0, 20.0),
) -> str:
    fact = "The person walks outside."
    return json.dumps(
        {
            "interval": list(interval),
            "timestamped_facts": [{"frame_index": 0, "fact": fact}],
            "option_evidence": {
                letter: {
                    "supports": [fact] if letter == support else [],
                    "contradicts": [fact] if letter == contradict else [],
                }
                for letter in ("A", "B")
            },
            "temporal_changes": [],
            "unresolved": [],
            "evidence_sufficient": True,
            "next_evidence_needed": "",
        }
    )


def _tool_call() -> str:
    return (
        '<tool_call>{"tool":"frame_select","arguments":'
        '{"start_time":10,"end_time":20,"nframes":1,"resize":0.75,'
        '"evidence_request":"check the action"}}</tool_call>'
    )


def _confirmation_tool_call() -> str:
    return (
        '<tool_call>{"tool":"frame_select","arguments":'
        '{"start_time":30,"end_time":40,"nframes":1,"resize":0.75,'
        '"evidence_request":"check independent confirmation evidence"}}</tool_call>'
    )


class _Session:
    def __init__(self, observation: FrameObservation) -> None:
        self.observation = observation
        self.metadata = {"duration": 100.0, "width": 1920, "height": 1080}

    def select(self, request: FrameRequest) -> FrameObservation:
        if request == self.observation.request:
            return self.observation
        return replace(
            self.observation,
            request=request,
            resolved_start_time=request.start_time,
            resolved_end_time=request.end_time,
            timestamps=((request.start_time + request.end_time) / 2.0,),
        )


class _FrameTool:
    def __init__(self, observation: FrameObservation) -> None:
        self.session = _Session(observation)

    def open_session(self, video: Path, session_id: str) -> _Session:
        assert video.is_file()
        assert session_id.startswith("pm-")
        return self.session


class _Client:
    def __init__(self, outputs: list[Any]) -> None:
        self.outputs = list(outputs)
        self.calls: list[dict[str, Any]] = []

    def chat(
        self,
        model: str,
        messages: list[dict[str, Any]],
        max_tokens: int = 32,
        **kwargs: Any,
    ) -> ChatResult:
        self.calls.append(
            {
                "model": model,
                "messages": messages,
                "max_tokens": max_tokens,
                **kwargs,
            }
        )
        output = self.outputs.pop(0)
        if isinstance(output, Exception):
            raise output
        if isinstance(output, tuple):
            content, finish_reason, prompt_tokens = output
        else:
            content, finish_reason, prompt_tokens = output, "stop", 100
        return ChatResult(
            content=content,
            usage={
                "prompt_tokens": prompt_tokens,
                "completion_tokens": 10,
                "total_tokens": prompt_tokens + 10,
                "prompt_tokens_details": {"multimodal_tokens": 50},
            },
            raw={},
            latency_s=0.01,
            finish_reason=finish_reason,
        )


def _evaluator(
    tmp_path: Path,
    client: _Client,
    *,
    server_max_model_len: int = 4096,
    context_safety_tokens: int = 512,
) -> PerceptionMemoryEvaEvaluator:
    video = tmp_path / "video.mp4"
    video.write_bytes(b"placeholder")
    frame = tmp_path / "frame.jpg"
    frame.write_bytes(b"placeholder")
    request = FrameRequest(
        start_time=10.0,
        end_time=20.0,
        nframes=1,
        resize=0.75,
        evidence_request="check the action",
    )
    observation = FrameObservation(
        request=request,
        resolved_start_time=10.0,
        resolved_end_time=20.0,
        resolved_nframes=1,
        frame_paths=(str(frame),),
        timestamps=(15.0,),
        backend="official_eva_select_frame_fallback",
        cache_hit=False,
        estimated_visual_tokens=128,
        latency_s=0.01,
    )
    return PerceptionMemoryEvaEvaluator(
        client,
        "Qwen3.5-9B",
        tmp_path,
        tmp_path / "frames",
        frame_tool=_FrameTool(observation),  # type: ignore[arg-type]
        max_turns=2,
        server_max_model_len=server_max_model_len,
        context_safety_tokens=context_safety_tokens,
    )


def _perception_trace(result: dict[str, Any], stage: str = "perception") -> list[dict]:
    return [item for item in result["request_trace"] if item["stage"] == stage]


def test_runtime_retries_truncated_perception_once_with_same_frames_and_trace(
    tmp_path: Path,
) -> None:
    client = _Client(
        [
            _tool_call(),
            ('{"interval":[10', "length", 100),
            _state_json(),
            '{"action":"stop"}',
            '{"evidence_complete":true,"missing_evidence":[]}',
            '{"answer":"A","evidence_ids":["E0001"]}',
            _confirmation_tool_call(),
            "invalid confirmation",
            _state_json(interval=(30.0, 40.0)),
            '{"evidence_complete":true,"missing_evidence":[]}',
            '{"answer":"A","evidence_ids":["E0001"]}',
        ]
    )

    result = _evaluator(tmp_path, client).run(_sample())

    assert result["error"] is None
    attempts = _perception_trace(result)
    assert [item["attempt_index"] for item in attempts] == [0, 1]
    assert [item["max_tokens"] for item in attempts] == [1024, 2048]
    assert attempts[0]["content"] == '{"interval":[10'
    assert attempts[0]["retry_reason"] == "finish_reason_length"
    assert attempts[0]["retry_triggered"] is True
    assert attempts[1]["retry_of_attempt"] == 0
    assert attempts[0]["retry_group_id"] == attempts[1]["retry_group_id"]
    first_images = [
        item
        for item in attempts[0]["messages"][-1]["content"]
        if item.get("type") == "image_url"
    ]
    second_images = [
        item
        for item in attempts[1]["messages"][-1]["content"]
        if item.get("type") == "image_url"
    ]
    assert first_images == second_images
    state = result["perception_states"][0]
    assert state["perception_attempts"] == 2
    assert state["perception_retry_reason"] == "finish_reason_length"
    assert state["timestamp_reference_mode"] == "frame_index"
    assert json.loads(state["perception_model_target"])["timestamped_facts"] == [
        {"frame_index": 0, "fact": "The person walks outside."}
    ]
    confirmation_attempts = _perception_trace(result, "confirmation_perception")
    assert [item["attempt_index"] for item in confirmation_attempts] == [0, 1]
    assert confirmation_attempts[0]["retry_reason"] == "invalid_json_or_schema"
    assert result["perception_states"][1]["perception_attempts"] == 2
    assert (
        result["perception_states"][1]["perception_retry_reason"]
        == "invalid_json_or_schema"
    )


def test_runtime_caps_retry_to_service_context_headroom(tmp_path: Path) -> None:
    client = _Client(
        [
            _tool_call(),
            ("not json", "stop", 600),
            _state_json(),
            '{"action":"stop"}',
            '{"evidence_complete":true,"missing_evidence":[]}',
            '{"answer":"B","evidence_ids":["E0001"]}',
        ]
    )

    result = _evaluator(
        tmp_path,
        client,
        server_max_model_len=2300,
        context_safety_tokens=500,
    ).run(_sample())

    attempts = _perception_trace(result)
    assert [item["max_tokens"] for item in attempts] == [1024, 1200]
    assert attempts[0]["retry_reason"] == "invalid_json_or_schema"
    assert result["perception_states"][0]["perception_attempts"] == 2


def test_runtime_persists_both_invalid_attempts_as_model_failure(
    tmp_path: Path,
) -> None:
    client = _Client([_tool_call(), "invalid one", "invalid two"])

    result = _evaluator(tmp_path, client).run(_sample())

    assert result["prediction"] == "B"
    assert result["fallback_to_candidate"] is True
    assert result["failure_class"] == "model_parse_failure"
    assert result["model_parse_failure"] is True
    assert result["stop_reason"] == "model_parse_failure"
    attempts = _perception_trace(result)
    assert len(attempts) == 2
    assert [item["content"] for item in attempts] == ["invalid one", "invalid two"]
    assert all(item["failure_class"] == "model_parse_failure" for item in attempts)
    assert result["perception_states"] == []


def test_runtime_records_perception_api_failure_without_model_retry(
    tmp_path: Path,
) -> None:
    client = _Client([_tool_call(), RuntimeError("HTTP 500: unavailable")])

    result = _evaluator(tmp_path, client).run(_sample())

    assert result["prediction"] == "B"
    assert result["failure_class"] == "infrastructure_error"
    assert result["model_parse_failure"] is False
    attempts = _perception_trace(result)
    assert len(attempts) == 1
    assert attempts[0]["failure_class"] == "infrastructure_error"
    assert attempts[0]["attempt_index"] == 0
    assert "HTTP 500" in attempts[0]["attempt_error"]
