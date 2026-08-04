from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest

from flashvid_eval.client import ChatResult
from flashvid_eval.eva_official import frame_tool_identity
from flashvid_eval.qwen_agents import (
    AgentConfig,
    AgentStrategy,
    DuplicateFrameRequestError,
    FrameRequest,
    FrameTool,
    InferenceProtocol,
    build_strategy,
    parse_answer_json,
    parse_frame_tool_calls,
)
from flashvid_eval.qwen_agents.strategies import _parse_a2_local, _parse_a2_overview
from flashvid_eval.schemas import ModelSample


class QueueClient:
    def __init__(self, responses: list[str]) -> None:
        self.responses = list(responses)
        self.calls: list[dict[str, Any]] = []

    def chat(
        self,
        model: str,
        messages: list[dict[str, Any]],
        max_tokens: int = 32,
        *,
        temperature: float = 0.0,
        seed: int | None = None,
        response_format: dict[str, Any] | None = None,
        logprobs: bool = False,
        top_logprobs: int | None = None,
        chat_template_kwargs: dict[str, Any] | None = None,
        sampling_params: dict[str, Any] | None = None,
        mm_processor_kwargs: dict[str, Any] | None = None,
        media_io_kwargs: dict[str, Any] | None = None,
        extra_body: dict[str, Any] | None = None,
    ) -> ChatResult:
        if not self.responses:
            raise AssertionError("fake client has no queued response")
        content = self.responses.pop(0)
        call = {
            "model": model,
            "messages": deepcopy(messages),
            "max_tokens": max_tokens,
            "temperature": temperature,
            "seed": seed,
            "response_format": deepcopy(response_format),
            "chat_template_kwargs": deepcopy(chat_template_kwargs),
            "sampling_params": deepcopy(sampling_params),
            "mm_processor_kwargs": deepcopy(mm_processor_kwargs),
            "media_io_kwargs": deepcopy(media_io_kwargs),
            "extra_body": deepcopy(extra_body),
        }
        self.calls.append(call)
        serialized = json.dumps(messages, ensure_ascii=False)
        visual_kind: str | None = None
        visual: int | None = None
        if '"video_url"' in serialized:
            visual_kind, visual = "video", 77
        elif '"image_url"' in serialized:
            visual_kind, visual = "image", 55
        usage: dict[str, Any] = {
            "prompt_tokens": 10,
            "completion_tokens": 3,
            "total_tokens": 13,
            "completion_tokens_details": {"reasoning_tokens": 2},
        }
        if visual is not None:
            usage["prompt_tokens_details"] = {
                "multimodal_tokens": {str(visual_kind): visual}
            }
        raw = {
            "choices": [
                {
                    "message": {
                        "content": content,
                        "reasoning_content": "private fake reasoning",
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": usage,
        }
        return ChatResult(content=content, usage=usage, raw=raw, latency_s=0.01)


def _sample(candidate: str | None = "PRIVATE_CANDIDATE") -> ModelSample:
    return ModelSample(
        dataset="unit",
        sample_id="sample-1",
        video="video.mp4",
        question="What visible action occurs?",
        choices={"A": "opens the door", "B": "sits down", "C": "leaves"},
        candidate_answer=candidate,
    )


def _video_root(tmp_path: Path) -> Path:
    root = tmp_path / "videos"
    root.mkdir()
    (root / "video.mp4").write_bytes(b"fake")
    return root


def _fake_probe(_: Path) -> dict[str, float | int]:
    return {"duration": 240.0, "width": 280, "height": 280}


def _fake_selector_factory(calls: list[dict[str, Any]]):
    def select(
        video: Path,
        start_time: float,
        end_time: float,
        nframes: int,
        resize: float,
        output_dir: Path,
    ) -> tuple[list[Path], list[float], str]:
        calls.append(
            {
                "video": video,
                "start_time": start_time,
                "end_time": end_time,
                "nframes": nframes,
                "resize": resize,
            }
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        paths: list[Path] = []
        timestamps: list[float] = []
        for index in range(nframes):
            path = output_dir / f"frame_{index:04d}.jpg"
            path.write_bytes(b"frame")
            paths.append(path)
            timestamps.append(start_time + (index + 0.5) * (end_time - start_time) / nframes)
        return paths, timestamps, "fake-decord"

    return select


def _frame_tool(tmp_path: Path, max_frames: int = 128) -> tuple[FrameTool, list[dict[str, Any]]]:
    calls: list[dict[str, Any]] = []
    tool = FrameTool(
        tmp_path / "frames",
        max_frames_per_call=max_frames,
        selector=_fake_selector_factory(calls),
        probe=_fake_probe,
    )
    return tool, calls


def _build(
    tmp_path: Path,
    strategy: str,
    responses: list[str],
    **config: Any,
):
    root = _video_root(tmp_path)
    tool, selector_calls = _frame_tool(
        tmp_path,
        max_frames=int(config.get("max_frames_per_call", 128)),
    )
    client = QueueClient(responses)
    agent = build_strategy(
        AgentConfig(strategy=strategy, **config),
        client=client,
        model="Qwen3.5-9B",
        video_root=root,
        frame_root=tmp_path / "unused",
        frame_tool=tool,
    )
    assert isinstance(agent, AgentStrategy)
    return agent, client, selector_calls


def test_frame_request_requires_exactly_one_sampling_mode() -> None:
    with pytest.raises(ValueError, match="exactly one"):
        FrameRequest(0, 10)
    with pytest.raises(ValueError, match="exactly one"):
        FrameRequest(0, 10, nframes=4, fps=1.0)
    with pytest.raises(ValueError, match="greater"):
        FrameRequest(5, 5, nframes=4)
    assert FrameRequest(0, 10, fps=0.5).fps == 0.5


def test_agent_run_context_changes_resume_fingerprint(tmp_path: Path) -> None:
    root = _video_root(tmp_path)
    tool, _ = _frame_tool(tmp_path)
    common = dict(
        client=QueueClient([]),
        model="Qwen3.5-9B",
        video_root=root,
        frame_root=tmp_path / "unused",
        frame_tool=tool,
    )
    first = build_strategy(
        AgentConfig(strategy="a0_eva_clean"),
        protocol=InferenceProtocol(run_context_sha256="a" * 64),
        **common,
    )
    second = build_strategy(
        AgentConfig(strategy="a0_eva_clean"),
        protocol=InferenceProtocol(run_context_sha256="b" * 64),
        **common,
    )
    assert first.run_fingerprint() != second.run_fingerprint()
    with pytest.raises(ValueError, match="run_context_sha256"):
        InferenceProtocol(run_context_sha256="not-a-hash")


def test_frame_tool_caps_fps_caches_and_deduplicates(tmp_path: Path) -> None:
    root = _video_root(tmp_path)
    tool, selector_calls = _frame_tool(tmp_path, max_frames=5)
    request = FrameRequest(0, 20, fps=2.0, resize=0.5)

    first_session = tool.open_session(root / "video.mp4", "one")
    first = first_session.select(request)
    assert first.resolved_nframes == 5
    assert first.request.fps == 2.0
    assert not first.cache_hit
    assert len(first.timestamps) == 5
    with pytest.raises(DuplicateFrameRequestError):
        first_session.select(request)

    second = tool.open_session(root / "video.mp4", "two").select(request)
    assert second.cache_hit
    assert len(selector_calls) == 1
    assert selector_calls[0]["nframes"] == 5


def test_frame_cache_key_binds_the_actual_selector_identity(tmp_path: Path) -> None:
    root = _video_root(tmp_path)
    calls: list[str] = []

    def first_selector(*args: Any):
        calls.append("first")
        return _fake_selector_factory([])(*args)

    def second_selector(*args: Any):
        calls.append("second")
        return _fake_selector_factory([])(*args)

    request = FrameRequest(0, 10, nframes=2)
    first = FrameTool(
        tmp_path / "shared-cache",
        selector=first_selector,
        probe=_fake_probe,
    ).open_session(root / "video.mp4", "first").select(request)
    second = FrameTool(
        tmp_path / "shared-cache",
        selector=second_selector,
        probe=_fake_probe,
    ).open_session(root / "video.mp4", "second").select(request)

    assert calls == ["first", "second"]
    assert not first.cache_hit
    assert not second.cache_hit


def test_official_frame_identity_binds_resolved_script_sha(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    script = tmp_path / "select_frame_fallback.py"
    script.write_bytes(b"VERSION = 1\n")
    monkeypatch.setenv("FLASHVID_EVA_TOOL_PATH", str(script))

    first = frame_tool_identity()
    script.write_bytes(b"VERSION = 2\n")
    second = frame_tool_identity()

    assert first["path"] == str(script.resolve())
    assert first["sha256"] == hashlib.sha256(b"VERSION = 1\n").hexdigest()
    assert first["sha256"] != second["sha256"]


def test_strict_json_answer_and_official_tool_parser() -> None:
    assert parse_answer_json('{"answer":"B"}', ("A", "B")) == "B"
    assert parse_answer_json('```json\n{"answer":"A"}\n```', ("A", "B")) is None
    assert parse_answer_json('Answer: B', ("A", "B")) is None
    assert parse_answer_json('Explanation then {"answer":"B"}', ("A", "B")) is None
    assert parse_answer_json('{"answer":"B","reason":"x"}', ("A", "B")) is None
    assert parse_answer_json('{"answer":"Z"}', ("A", "B")) is None

    valid = (
        '<tool_call>{"tool":"frame_select","arguments":'
        '{"start_time":1,"end_time":5,"fps":2,"resize":0.5}}</tool_call>'
    )
    calls = parse_frame_tool_calls(valid)
    assert len(calls) == 1 and calls[0].fps == 2.0
    invalid = (
        '<tool_call>{"tool":"frame_select","arguments":'
        '{"start_time":1,"end_time":5,"fps":2,"nframes":8}}</tool_call>'
    )
    assert parse_frame_tool_calls(invalid) == []
    assert parse_frame_tool_calls('{"tool":"frame_select","arguments":{}}') == []


def test_a0_eva_clean_uses_official_tools_and_never_leaks_candidate(tmp_path: Path) -> None:
    tool_call = (
        '<tool_call>{"tool":"frame_select","arguments":'
        '{"start_time":10,"end_time":20,"nframes":4,"resize":0.5}}</tool_call>'
    )
    agent, client, _ = _build(tmp_path, "a0_eva_clean", [tool_call, '{"answer":"B"}'])
    trace = agent.run(_sample())
    result = trace.to_result_dict()

    assert result["prediction"] == result["final_prediction"] == "B"
    assert len(result["run_fingerprint"]) == 64
    assert len(result["tool_steps"]) == 1
    assert result["tool_steps"][0]["timestamps"]
    assert result["tool_steps"][0]["actual_timestamps"] == result["tool_steps"][0]["timestamps"]
    assert result["tool_steps"][0]["nframes"] == 4
    assert result["request_trace"][0]["finish_reason"] == "stop"
    assert result["reasoning_tokens"] == 4
    assert result["annotation_leak_check"] == "passed"
    serialized = json.dumps(client.calls, ensure_ascii=False)
    assert "PRIVATE_CANDIDATE" not in serialized
    for forbidden in ("time_range", "clue_intervals", "question_type"):
        assert forbidden not in serialized


def test_agent_emits_internal_api_and_frame_progress(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    events: list[str] = []
    monkeypatch.setattr(
        "flashvid_eval.qwen_agents.core.emit_progress",
        lambda event, **details: events.append(event) or True,
    )
    tool_call = (
        '<tool_call>{"tool":"frame_select","arguments":'
        '{"start_time":10,"end_time":20,"nframes":4,"resize":0.5}}</tool_call>'
    )
    agent, _, _ = _build(tmp_path, "a0_eva_clean", [tool_call, '{"answer":"B"}'])

    assert agent.run(_sample()).final_prediction == "B"
    assert events[0] == "agent_sample_started"
    assert events.count("api_request_started") == 2
    assert events.count("api_response") == 2
    assert "frame_select_started" in events
    assert "frame_select_complete" in events
    assert events[-1] == "agent_sample_complete"


def test_base_agent_classifies_annotation_leak_before_api_call(tmp_path: Path) -> None:
    agent, client, _ = _build(tmp_path, "a0_eva_clean", ['{"answer":"A"}'])
    sample = ModelSample(
        dataset="unit",
        sample_id="leak",
        video="video.mp4",
        question="SECRET_ANNOTATION_SENTINEL",
        choices={"A": "one", "B": "two"},
        candidate_answer=None,
    )

    result = agent.run(sample).to_result_dict()

    assert client.calls == []
    assert result["annotation_leak_check"] == "failed"
    assert result["failure_class"] == "annotation_leak"
    assert result["error_type"] == "AnnotationLeakError"


def test_a0_deterministic_overview_fallback_is_audited(tmp_path: Path) -> None:
    agent, _, calls = _build(
        tmp_path,
        "a0",
        ["I need evidence but emitted no tool.", '{"answer":"A"}'],
        overview_frames=4,
    )
    trace = agent.run(_sample())
    assert trace.final_prediction == "A"
    assert trace.fallback_used
    assert calls[0]["start_time"] == 0.0
    assert calls[0]["end_time"] == 240.0
    assert calls[0]["nframes"] == 4


def test_a0_rejects_duplicate_interval_without_crashing(tmp_path: Path) -> None:
    tool_call = (
        '<tool_call>{"tool":"frame_select","arguments":'
        '{"start_time":10,"end_time":20,"nframes":4}}</tool_call>'
    )
    agent, _, selector_calls = _build(
        tmp_path,
        "a0",
        [tool_call, tool_call, '{"answer":"C"}'],
        max_turns=3,
    )
    trace = agent.run(_sample())
    assert trace.final_prediction == "C"
    assert trace.fallback_used
    assert len(selector_calls) == 1


def test_a1_storyboard_zoom_records_overview_local_evidence_and_costs(tmp_path: Path) -> None:
    responses = [
        '{"intervals":[[10,20],[100,110]],"evidence":"two candidate events"}',
        '{"observed_facts":["door opens"],"supports":["A"]}',
        '{"observed_facts":["person sits"],"supports":["B"]}',
        '{"answer":"A"}',
    ]
    agent, client, _ = _build(
        tmp_path,
        "a1_storyboard_zoom",
        responses,
        overview_frames=4,
        max_intervals=2,
        local_fps=0.5,
        max_frames_per_call=16,
    )
    result = agent.run(_sample()).to_result_dict()

    assert result["final_prediction"] == "A"
    assert len(result["tool_steps"]) == 3
    assert [item["source"] for item in result["evidence_memory"]] == [
        "storyboard_overview",
        "local_zoom",
        "local_zoom",
    ]
    assert result["prompt_tokens"] == 40
    assert result["completion_tokens"] == 12
    assert result["total_tokens"] == 52
    assert result["branch_costs"]["evidence"]["tool_call_count"] == 3
    assert result["branch_costs"]["judge"]["request_count"] == 1
    assert client.calls[-1]["response_format"]["type"] == "json_schema"


def test_a2_multi_clue_prompt_requests_structured_memory(tmp_path: Path) -> None:
    responses = [
        '{"atomic_claims":["A opens door","B sits"],"unresolved":[],"intervals":[[20,30]]}',
        '{"observed_facts":["door opens"],"supports":["A"],"contradicts":["B"],"unresolved":[]}',
        '{"answer":"A"}',
    ]
    agent, client, _ = _build(
        tmp_path,
        "a2_multi_clue_memory",
        responses,
        overview_frames=4,
        max_intervals=1,
        local_fps=0.5,
    )
    trace = agent.run(_sample())
    assert trace.final_prediction == "A"
    prompt = json.dumps(client.calls[0]["messages"], ensure_ascii=False)
    assert "atomic_claims" in prompt
    assert "multiple disjoint clues" in prompt


def test_a2_schema_parsers_are_strict_and_canonical() -> None:
    overview = _parse_a2_overview(
        '{"atomic_claims":[" A "],"unresolved":[],"intervals":[[20,30],[0,10]]}',
        40.0,
        2,
    )
    assert overview == (
        {
            "atomic_claims": ["A"],
            "unresolved": [],
            "intervals": [[0.0, 10.0], [20.0, 30.0]],
        },
        [(0.0, 10.0), (20.0, 30.0)],
    )
    assert _parse_a2_overview(
        'prefix {"atomic_claims":["A"],"unresolved":[],"intervals":[[0,10]]}',
        40.0,
        1,
    ) is None
    assert _parse_a2_overview(
        '{"atomic_claims":["A"],"unresolved":[],"intervals":[[0,20],[10,30]]}',
        40.0,
        2,
    ) is None

    local = _parse_a2_local(
        '{"observed_facts":["door opens"],"supports":["a"],"contradicts":["B"],"unresolved":[]}',
        ("A", "B"),
    )
    assert local is not None
    assert local["supports"] == ["A"]
    assert _parse_a2_local(
        '{"observed_facts":["door opens"],"supports":["A"],"contradicts":["A"],"unresolved":[]}',
        ("A", "B"),
    ) is None
    assert _parse_a2_local(
        '{"observed_facts":["door opens"],"supports":["A"],"contradicts":[],"unresolved":[],"answer":"A"}',
        ("A", "B"),
    ) is None


def test_a2_invalid_overview_uses_deterministic_intervals_without_leaking_raw_text(
    tmp_path: Path,
) -> None:
    responses = [
        "not-json PRIVATE_INVALID_OVERVIEW",
        '{"observed_facts":["door opens"],"supports":["A"],"contradicts":["B"],"unresolved":[]}',
        '{"answer":"A"}',
    ]
    agent, client, calls = _build(
        tmp_path,
        "a2_multi_clue_memory",
        responses,
        overview_frames=4,
        max_intervals=1,
        local_fps=0.5,
    )
    trace = agent.run(_sample())
    assert trace.final_prediction == "A"
    assert trace.fallback_used is True
    assert len(calls) == 2
    assert [item.source for item in trace.evidence_memory] == ["local_zoom"]
    judge_messages = json.dumps(client.calls[-1]["messages"], ensure_ascii=False)
    assert "PRIVATE_INVALID_OVERVIEW" not in judge_messages


def test_a2_all_invalid_local_evidence_stops_before_judge(tmp_path: Path) -> None:
    responses = [
        '{"atomic_claims":["A opens door"],"unresolved":[],"intervals":[[20,30]]}',
        '{"observed_facts":[],"supports":["A"],"contradicts":[],"unresolved":[]}',
    ]
    agent, client, _ = _build(
        tmp_path,
        "a2_multi_clue_memory",
        responses,
        overview_frames=4,
        max_intervals=1,
        local_fps=0.5,
    )
    trace = agent.run(_sample())
    assert trace.final_prediction is None
    assert trace.error_type == "EvidenceValidationError"
    assert trace.failure_class == "model_parse_failure"
    assert len(client.calls) == 2
    assert [item.source for item in trace.evidence_memory] == ["storyboard_overview"]


def test_a3_hierarchical_search_narrows_interval_then_observes_leaf(tmp_path: Path) -> None:
    responses = [
        '{"selected_node":2,"node_summaries":["..."]}',
        '{"selected_node":1,"node_summaries":["..."]}',
        '{"observed_facts":["person leaves"]}',
        '{"answer":"C"}',
    ]
    agent, _, calls = _build(
        tmp_path,
        "a3_hierarchical_search",
        responses,
        hierarchy_nodes=12,
        hierarchy_depth=9,
        max_turns=4,
        local_window_s=1.0,
        local_fps=1.0,
        max_frames_per_call=16,
    )
    result = agent.run(_sample()).to_result_dict()

    assert result["final_prediction"] == "C"
    assert len(calls) == 3
    first = result["tool_steps"][0]
    second = result["tool_steps"][1]
    leaf = result["tool_steps"][2]
    assert (first["resolved_start_time"], first["resolved_end_time"]) == (0.0, 240.0)
    assert first["request"]["nframes"] == 8
    assert (second["resolved_start_time"], second["resolved_end_time"]) == (60.0, 90.0)
    assert second["request"]["nframes"] == 2
    assert (leaf["resolved_start_time"], leaf["resolved_end_time"]) == (75.0, 76.0)
    assert len(result["request_trace"]) == 4
    assert len(result["tool_steps"]) == 3


def test_a4_keeps_branches_independent_and_accounts_confirmation(tmp_path: Path) -> None:
    responses = [
        '{"answer":"A"}',  # independent full-video Direct
        '{"selected_node":1,"node_summaries":["event in node 1"]}',
        '{"observed_facts":["person sits"]}',
        '{"answer":"B"}',  # evidence branch
        (
            '<tool_call>{"tool":"frame_select","arguments":'
            '{"start_time":40,"end_time":45,"nframes":4,"resize":1.0}}</tool_call>'
        ),
        '{"answer":"B"}',  # arbitration after visual confirmation
    ]
    agent, client, _ = _build(
        tmp_path,
        "a4_independent_arbitration",
        responses,
        evidence_strategy="a3_hierarchical_search",
        hierarchy_nodes=4,
        hierarchy_depth=1,
        local_window_s=10.0,
        local_fps=0.5,
        max_frames_per_call=16,
    )
    result = agent.run(_sample()).to_result_dict()

    assert result["final_prediction"] == "B"
    assert {"direct", "evidence/evidence", "evidence/judge", "arbiter"} <= set(
        result["branch_costs"]
    )
    assert result["branch_costs"]["direct"]["visual_tokens"] == 77
    assert client.calls[0]["mm_processor_kwargs"] == {
        "do_sample_frames": False,
    }
    assert client.calls[0]["media_io_kwargs"] == {
        "video": {
            "num_frames": 64,
            "fps": -1,
        }
    }
    assert client.calls[0]["response_format"]["type"] == "json_schema"
    assert client.calls[-1]["response_format"]["type"] == "json_schema"
    assert result["branch_costs"]["arbiter"]["tool_call_count"] == 1
    assert any(item["source"] == "arbitration_confirmation" for item in result["evidence_memory"])

    evidence_calls = [
        item for item in result["request_trace"] if item["branch"].startswith("evidence/")
    ]
    evidence_payload = json.dumps(evidence_calls, ensure_ascii=False)
    assert "Full-video branch answer" not in evidence_payload
    assert "PRIVATE_CANDIDATE" not in json.dumps(client.calls, ensure_ascii=False)


def test_a4_missing_direct_visual_usage_is_not_reported_as_zero(tmp_path: Path) -> None:
    class NoDetailClient(QueueClient):
        def chat(self, *args: Any, **kwargs: Any) -> ChatResult:
            result = super().chat(*args, **kwargs)
            usage = dict(result.usage)
            usage.pop("prompt_tokens_details", None)
            raw = deepcopy(result.raw)
            raw["usage"] = usage
            return ChatResult(result.content, usage, raw, result.latency_s)

    root = _video_root(tmp_path)
    tool, _ = _frame_tool(tmp_path, max_frames=16)
    client = NoDetailClient(
        [
            '{"answer":"A"}',
            '{"selected_node":1}',
            '{"observed_facts":["door opens"]}',
            '{"answer":"A"}',
        ]
    )
    agent = build_strategy(
        AgentConfig(
            strategy="a4",
            hierarchy_nodes=4,
            hierarchy_depth=1,
            local_window_s=10,
            local_fps=0.5,
            max_frames_per_call=16,
        ),
        client=client,
        model="Qwen3.5-9B",
        video_root=root,
        frame_root=tmp_path / "unused",
        frame_tool=tool,
    )
    result = agent.run(_sample()).to_result_dict()
    assert result["final_prediction"] == "A"
    assert result["visual_tokens"] is None
    assert not result["visual_token_accounting_complete"]
    assert result["branch_costs"]["direct"]["visual_tokens"] is None


def test_inference_protocol_is_forwarded_and_reasoning_is_audited(tmp_path: Path) -> None:
    root = _video_root(tmp_path)
    tool, _ = _frame_tool(tmp_path)
    client = QueueClient(['{"answer":"A"}'])
    agent = build_strategy(
        {"strategy": "a0", "max_turns": 1},
        client=client,
        model="Qwen3.5-9B",
        video_root=root,
        frame_root=tmp_path / "unused",
        frame_tool=tool,
        protocol=InferenceProtocol(
            enable_thinking=True,
            temperature=1.0,
            top_p=0.95,
            top_k=20,
            min_p=0.0,
            presence_penalty=1.5,
            repetition_penalty=1.0,
            planner_max_tokens=8192,
            seed=17,
        ),
    )
    result = agent.run(_sample()).to_result_dict()
    assert client.calls[0]["max_tokens"] == 8192
    assert client.calls[0]["seed"] == 17
    assert client.calls[0]["chat_template_kwargs"] == {"enable_thinking": True}
    assert client.calls[0]["sampling_params"] == {
        "top_p": 0.95,
        "top_k": 20,
        "min_p": 0.0,
        "presence_penalty": 1.5,
        "repetition_penalty": 1.0,
    }
    assert client.calls[0]["extra_body"] == {"return_token_ids": True}
    assert result["request_trace"][0]["reasoning_content"] == "private fake reasoning"
    assert result["request_trace"][0]["enable_thinking"] is True


def test_length_retry_is_counted_and_respects_context_headroom(tmp_path: Path) -> None:
    class LengthClient(QueueClient):
        def chat(self, *args: Any, **kwargs: Any) -> ChatResult:
            result = super().chat(*args, **kwargs)
            attempt = len(self.calls)
            finish = "length" if attempt == 1 else "stop"
            usage = dict(result.usage)
            usage["prompt_tokens"] = 100
            usage["total_tokens"] = 103
            raw = deepcopy(result.raw)
            raw["usage"] = usage
            raw["choices"][0]["finish_reason"] = finish
            return ChatResult(
                content=result.content,
                usage=usage,
                raw=raw,
                latency_s=result.latency_s,
                reasoning_content="private fake reasoning",
                finish_reason=finish,
            )

    root = _video_root(tmp_path)
    tool, _ = _frame_tool(tmp_path)
    client = LengthClient(["truncated", '{"answer":"A"}'])
    agent = build_strategy(
        {"strategy": "a0", "max_turns": 1},
        client=client,
        model="Qwen3.5-9B",
        video_root=root,
        frame_root=tmp_path / "unused",
        frame_tool=tool,
        protocol=InferenceProtocol(
            planner_max_tokens=64,
            length_retry_max_tokens=512,
            server_max_model_len=400,
            context_safety_tokens=50,
        ),
    )
    result = agent.run(_sample()).to_result_dict()
    assert [call["max_tokens"] for call in client.calls] == [64, 250]
    assert result["total_tokens"] == 206
    assert [item["retry_of_length"] for item in result["request_trace"]] == [False, True]
    assert result["final_prediction"] == "A"


def test_invalid_answer_is_an_auditable_error_not_a_guessed_letter(tmp_path: Path) -> None:
    agent, _, _ = _build(
        tmp_path,
        "a1",
        [
            '{"intervals":[[10,11]]}',
            '{"observed_facts":[]}',
            "I think option A is likely.",
        ],
        overview_frames=4,
        max_intervals=1,
        local_fps=1.0,
    )
    result = agent.run(_sample()).to_result_dict()
    assert result["prediction"] is None
    assert result["error_type"] == "AnswerParseError"


def test_config_rejects_unknown_fields_and_recursive_arbitration() -> None:
    with pytest.raises(ValueError, match="unknown agent config"):
        AgentConfig.from_mapping({"strategy": "a0", "typo": 1})
    with pytest.raises(ValueError, match="recursively"):
        AgentConfig(
            strategy="a4_independent_arbitration",
            evidence_strategy="a4_independent_arbitration",
        )
