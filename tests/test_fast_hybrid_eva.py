from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import MethodType, SimpleNamespace

import pytest

import flashvid_eval.fast_hybrid_eva as fast_module
from flashvid_eval.client import ChatResult
from flashvid_eval.fast_hybrid_eva import (
    OFFICIAL_EVA_COMMIT,
    FastHybridEvaEvaluator,
    _budgeted_frame_select,
    _load_official_module,
    _validated_trajectory_context,
)
from flashvid_eval.qwen_protocol import QWEN_PROTOCOLS
from flashvid_eval.schemas import ModelSample, Sample
from scripts.summarize_fast_hybrid import _metric


def _run_record(
    prediction: str | None, *, tool: bool = True, stage: str = "verification"
) -> dict:
    assistant = f"Answer: {prediction}" if prediction else ""
    return {
        "prediction": prediction,
        "raw_response": assistant,
        "finish_reason": "stop",
        "rounds": 2,
        "usage": {
            "prompt_tokens": 100,
            "completion_tokens": 10,
            "total_tokens": 110,
        },
        "latency_s": 0.5,
        "visual_tokens": 1000 if tool else 0,
        "visual_usage_complete": True,
        "visual_budget_estimate": 1000 if tool else 0,
        "agent_total_tokens_complete": True,
        "tool_calls": (
            [
                {
                    "stage": stage,
                    "start_time": 0.0,
                    "end_time": 10.0,
                    "nframes": 8,
                    "resize": 1.0,
                    "timestamps": [0, 2, 4, 6, 8],
                    "frame_paths": ["/frames/frame_0001.png"],
                    "estimated_visual_tokens": 1000,
                    "backend": "official_eva_select_frame_fallback",
                }
            ]
            if tool
            else []
        ),
        "request_trace": [
            {
                "stage": stage,
                "branch": stage,
                "request_kind": "eva_agent_turn",
                "model": "Qwen3.5-9B",
                "messages": [{"role": "user", "content": "Question"}],
                "content": assistant,
                "assistant_content": assistant,
                "assistant_tool_calls": [],
                "reasoning_content": "",
                "finish_reason": "stop",
                "usage": {
                    "prompt_tokens": 100,
                    "completion_tokens": 10,
                    "total_tokens": 110,
                },
                "latency_s": 0.5,
                "max_tokens": 2048,
                "temperature": 0.0,
                "enable_thinking": False,
                "attempt_index": 1,
                "prompt_hash": "d" * 64,
            }
        ],
        "messages": [
            {"role": "user", "content": "Question"},
            {"role": "assistant", "content": assistant},
        ],
        "prompt_sha256": "e" * 64,
        "stage": stage,
        "call_records": [],
        "stop_reason": "answer_found",
        "error": None,
    }


def _evaluator(records: list[dict]) -> tuple[FastHybridEvaEvaluator, list[str]]:
    evaluator = FastHybridEvaEvaluator.__new__(FastHybridEvaEvaluator)
    evaluator.version = "fast_hybrid_v1"
    evaluator.model = "Qwen3.5-9B"
    evaluator.max_turns = 6
    evaluator.max_call_visual_tokens = 12000
    evaluator.max_total_visual_tokens = 24000
    evaluator.candidate_results_sha256 = "a" * 64
    evaluator.teacher_model_sha256 = "b" * 64
    evaluator.experiment_config_sha256 = "c" * 64
    evaluator.scoring_deferred = True
    evaluator.teacher_temperature = 0.0
    evaluator.generation_seed = 0
    evaluator.trajectory_context = {
        "experiment_config_sha256": "c" * 64,
        "model_artifact_sha256": "b" * 64,
        "manifest_sha256": "1" * 64,
        "trajectory_schedule_id": "teacher-search-v1",
        "trajectory_variant_id": "base",
        "trajectory_replica_id": 0,
    }
    prompts: list[str] = []

    forced_calls: list[list[dict] | None] = []

    def fake_run(
        self,
        sample,
        system_prompt,
        visual_budget,
        stage,
        forced_tool_calls=None,
        **kwargs,
    ):
        del self, sample, visual_budget, kwargs
        prompts.append(system_prompt)
        forced_calls.append(forced_tool_calls)
        record = records.pop(0)
        record["stage"] = stage
        record["prompt_sha256"] = "e" * 64 if stage == "verification" else "f" * 64
        for request in record.get("request_trace", []):
            request["stage"] = stage
            request["branch"] = stage
        return record

    evaluator._official_run = MethodType(fake_run, evaluator)
    evaluator._forced_calls = forced_calls
    return evaluator, prompts


def test_fast_hybrid_uses_pinned_official_eva_single() -> None:
    module = _load_official_module()
    assert module.single.__module__ == "flashvid_official_eva_eval"
    assert OFFICIAL_EVA_COMMIT == "758ad8d3dcb84a8086e5d70c9afb9a6f278e8f5a"


def test_clean_direct_protocol_is_greedy_no_think() -> None:
    protocol = QWEN_PROTOCOLS["no_think_greedy"]
    assert protocol.enable_thinking is False
    assert protocol.max_tokens == 512
    assert protocol.temperature == 0.0


def test_fast_hybrid_fingerprint_freezes_generation_and_trajectory_context() -> None:
    evaluator, _ = _evaluator([_run_record("A")])
    original = evaluator.run_fingerprint()
    evaluator.generation_seed = 17
    assert evaluator.run_fingerprint() != original
    evaluator.generation_seed = 0
    evaluator.teacher_temperature = 0.2
    assert evaluator.run_fingerprint() != original
    evaluator.teacher_temperature = 0.0
    evaluator.trajectory_context = {
        **evaluator.trajectory_context,
        "trajectory_variant_id": "changed",
    }
    assert evaluator.run_fingerprint() != original


def test_fast_hybrid_allows_only_confirmed_visual_change() -> None:
    evaluator, prompts = _evaluator(
        [_run_record("C"), _run_record("C", stage="change_confirmation")]
    )
    sample = Sample(
        "lsdbench",
        "one",
        "one.mp4",
        "What happens?",
        {"A": "first", "B": "second", "C": "third"},
        "C",
        metadata={"time_range": "SECRET_TIME", "clue_intervals": "SECRET_CLUE"},
    )

    result = evaluator.fast_hybrid_eva(sample, "B")

    assert result["prediction"] == "C"
    assert result["candidate_rerun"] == 0
    assert result["candidate_changed"] is True
    assert result["final_decision_source"] == "confirmed_visual_change"
    assert result["visual_tokens"] == 2000
    assert result["total_tokens"] == 220
    assert len(prompts) == 2
    assert "Direct candidate: B" in prompts[0]
    assert "neither is privileged" in prompts[1]
    assert "SECRET" not in json.dumps(prompts)


def test_fast_hybrid_confirmation_failure_is_aggregated_and_rejected() -> None:
    first = _run_record("C")
    confirmation = _run_record(None, tool=False, stage="change_confirmation")
    confirmation["stop_reason"] = "no_answer_no_tool_call"
    confirmation["error"] = "confirmation API failed"
    evaluator, _ = _evaluator([first, confirmation])
    sample = Sample(
        "lsdbench",
        "confirmation-failure",
        "one.mp4",
        "What happens?",
        {"A": "first", "B": "second", "C": "third"},
        "B",
    )

    result = evaluator.fast_hybrid_eva(sample, "B")

    assert result["prediction"] == "B"
    assert result["fallback_to_candidate"] is True
    assert result["change_rejection_reason"] == "confirmation_failed"
    assert result["run_stop_reasons"] == {
        "verification": "answer_found",
        "change_confirmation": "no_answer_no_tool_call",
    }
    assert result["run_errors"] == {
        "change_confirmation": "confirmation API failed"
    }
    assert "change_confirmation" in result["error"]
    assert result["error_type"] == "required_run_failure"


def test_fast_hybrid_preserves_unscored_trace_and_provenance() -> None:
    evaluator, _ = _evaluator(
        [_run_record("C"), _run_record("C", stage="change_confirmation")]
    )
    sample = Sample(
        "lsdbench",
        "trace",
        "trace.mp4",
        "What happens?",
        {"A": "first", "B": "second", "C": "third"},
        "C",
        metadata={
            "time_range": "PRIVATE_TIME_SENTINEL",
            "clue_intervals": "PRIVATE_CLUE_SENTINEL",
            "question_type": "PRIVATE_TYPE_SENTINEL",
        },
    )

    result = evaluator.fast_hybrid_eva(sample, "B")

    assert result["scoring_deferred"] is True
    assert result["candidate_results_sha256"] == "a" * 64
    assert result["teacher_model_sha256"] == "b" * 64
    assert result["model_artifact_sha256"] == "b" * 64
    assert result["experiment_config_sha256"] == "c" * 64
    assert result["manifest_sha256"] == "1" * 64
    assert result["trajectory_schedule_id"] == "teacher-search-v1"
    assert result["trajectory_variant_id"] == "base"
    assert result["trajectory_replica_id"] == 0
    assert len(result["prompt_sha256"]) == 64
    assert result["prompt_hashes"] == {
        "verification": "e" * 64,
        "change_confirmation": "f" * 64,
    }
    assert len(result["request_trace"]) == 2
    assert {item["stage"] for item in result["request_trace"]} == {
        "verification",
        "change_confirmation",
    }
    assert len(result["conversation_traces"]) == 2
    assert all("stage" in message for message in result["messages"])
    assert "answer" not in result
    assert "correct" not in result
    serialized = json.dumps(
        {
            "request_trace": result["request_trace"],
            "conversation_traces": result["conversation_traces"],
            "messages": result["messages"],
        },
        ensure_ascii=False,
    )
    assert "PRIVATE_TIME_SENTINEL" not in serialized
    assert "PRIVATE_CLUE_SENTINEL" not in serialized
    assert "PRIVATE_TYPE_SENTINEL" not in serialized


def test_official_run_captures_every_request_and_tool_message(
    tmp_path: Path, monkeypatch
) -> None:
    video = tmp_path / "video.mp4"
    video.write_bytes(b"video")
    frame = (tmp_path / "frame.png").resolve()
    frame.write_bytes(b"frame")

    class _Index:
        def resolve(self, _video: str) -> Path:
            return video

    class _Client:
        def __init__(self) -> None:
            self.requests = []
            self.responses = [
                (
                    '<tool_call>{"tool":"frame_select","arguments":'
                    '{"start_time":0,"end_time":10,"nframes":1,"resize":1.0}}'
                    "</tool_call>"
                ),
                "Answer: B",
            ]

        def chat(self, model, messages, **kwargs):
            self.requests.append(
                {"model": model, "messages": messages, "kwargs": kwargs}
            )
            content = self.responses.pop(0)
            usage = {
                "prompt_tokens": 10,
                "completion_tokens": 2,
                "total_tokens": 12,
            }
            if any(
                isinstance(message.get("content"), list)
                for message in messages
                if isinstance(message, dict)
            ):
                usage["prompt_tokens_details"] = {
                    "multimodal_tokens": {"image": 7}
                }
            return ChatResult(
                content=content,
                usage=usage,
                raw={},
                latency_s=0.1,
                finish_reason="stop",
            )

    async def fake_single(
        index,
        item,
        model,
        dataset_cfg,
        tokenizer,
        max_turns,
        timestamp_fmt,
        max_visual_tokens,
        maxp,
        fallback,
    ):
        del index, dataset_cfg, max_turns, timestamp_fmt, max_visual_tokens, maxp, fallback
        messages = list(item["prompt"])
        completions = fast_module._OfficialAsyncCompletions()
        first = await completions.create(
            model=model, messages=messages, temperature=0.0, max_tokens=2048
        )
        messages.append({"role": "assistant", "content": first.choices[0].message.content})
        messages.append(
            {
                "role": "tool",
                "content": [
                    {"type": "text", "text": "<tool_response>"},
                    {
                        "type": "image_url",
                        "image_url": {"url": frame.as_uri()},
                    },
                    {"type": "text", "text": "</tool_response>"},
                ],
            }
        )
        second = await completions.create(
            model=model, messages=messages, temperature=0.0, max_tokens=2048
        )
        messages.append({"role": "assistant", "content": second.choices[0].message.content})
        return {
            "answer": "B",
            "num_rounds": 2,
            "messages": tokenizer.apply_chat_template(messages, tokenize=False),
            "stop_reason": "answer_found",
            "error": None,
        }

    monkeypatch.setattr(
        fast_module,
        "probe_video",
        lambda _path: {"duration": 10.0, "height": 224, "width": 224},
    )
    evaluator = FastHybridEvaEvaluator.__new__(FastHybridEvaEvaluator)
    evaluator.client = _Client()
    evaluator.model = "Qwen3.5-9B"
    evaluator.index = _Index()
    evaluator.max_turns = 6
    evaluator.max_call_visual_tokens = 12000
    evaluator.teacher_temperature = 0.2
    evaluator.generation_seed = 73
    evaluator.official = SimpleNamespace(single=fake_single)
    sample = ModelSample(
        "lsdbench",
        "safe",
        "video.mp4",
        "What happens?",
        {"A": "first", "B": "second"},
        "A",
    )

    result = evaluator._official_run(
        sample, "Verifier without private annotations", 24000, "verification"
    )

    assert len(result["request_trace"]) == 2
    assert all(item["stage"] == "verification" for item in result["request_trace"])
    assert result["request_trace"][0]["assistant_tool_calls"][0].startswith(
        "<tool_call>"
    )
    second_messages = result["request_trace"][1]["messages"]
    tool_message = next(item for item in second_messages if item["role"] == "tool")
    assert tool_message["content"][0]["text"] == "<tool_response>"
    assert tool_message["content"][1]["image_url"]["url"] == frame.as_uri()
    assert any(
        item["role"] == "assistant" and "<tool_call>" in item["content"]
        for item in result["messages"]
    )
    assert len(result["prompt_sha256"]) == 64
    assert all(
        request["kwargs"]["temperature"] == 0.2
        and request["kwargs"]["seed"] == 73
        for request in evaluator.client.requests
    )
    assert all(item["temperature"] == 0.2 for item in result["request_trace"])
    assert all(item["seed"] == 73 for item in result["request_trace"])
    assert result["visual_tokens"] == 7
    assert result["visual_usage_complete"] is True
    assert result["visual_budget_estimate"] == 0
    assert result["agent_total_tokens_complete"] is True


def test_official_adapter_replays_forced_tool_queue_before_model() -> None:
    class _Client:
        def __init__(self) -> None:
            self.calls = 0

        def chat(self, model, messages, **kwargs):
            del model, messages, kwargs
            self.calls += 1
            return ChatResult(
                content="Answer: B",
                usage={"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5},
                raw={},
                latency_s=0.1,
                finish_reason="stop",
            )

    client = _Client()
    log: list[dict] = []
    tokens = [
        (fast_module._ACTIVE_CLIENT, fast_module._ACTIVE_CLIENT.set(client)),
        (fast_module._ACTIVE_CALLS, fast_module._ACTIVE_CALLS.set(log)),
        (
            fast_module._ACTIVE_GENERATION,
            fast_module._ACTIVE_GENERATION.set({"temperature": 0.2, "seed": 17}),
        ),
        (
            fast_module._ACTIVE_FORCED_TOOL_CALLS,
            fast_module._ACTIVE_FORCED_TOOL_CALLS.set(
                {
                    "calls": [
                        {"tool": "frame_select", "arguments": {"start_time": 0, "end_time": 1, "nframes": 1, "resize": 1}},
                        {"tool": "frame_select", "arguments": {"start_time": 1, "end_time": 2, "nframes": 1, "resize": 1}},
                    ],
                    "index": 0,
                    "source": "test_replay",
                }
            ),
        ),
    ]
    try:
        completions = fast_module._OfficialAsyncCompletions()
        messages = [{"role": "user", "content": "Question"}]
        outputs = [
            asyncio.run(
                completions.create(
                    model="Qwen3.5-9B", messages=messages, max_tokens=2048
                )
            ).choices[0].message.content
            for _ in range(3)
        ]
    finally:
        for variable, token in reversed(tokens):
            variable.reset(token)

    assert all("<tool_call>" in content for content in outputs[:2])
    assert outputs[2] == "Answer: B"
    assert client.calls == 1
    assert [record["source"] for record in log] == [
        "test_replay",
        "test_replay",
        "model",
    ]


def test_actual_visual_usage_fails_closed_when_media_details_are_missing() -> None:
    media_call = {
        "source": "model",
        "messages": [
            {
                "role": "tool",
                "content": [{"type": "image_url", "image_url": {"url": "x"}}],
            }
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 1, "total_tokens": 11},
    }
    assert fast_module._actual_visual_usage([media_call]) == (None, False)
    media_call["usage"]["input_tokens_details"] = {
        "multimodal_tokens": {"image": 9, "video": 3}
    }
    assert fast_module._actual_visual_usage([media_call]) == (12, True)
    media_call["error"] = "timeout after submission"
    assert fast_module._actual_visual_usage([media_call]) == (None, False)
    assert fast_module._actual_total_usage_complete([media_call]) is False


def test_budgeted_tool_trace_saves_absolute_frame_paths(
    tmp_path: Path, monkeypatch
) -> None:
    frame = tmp_path / "frame.png"
    frame.write_bytes(b"frame")

    async def fake_select(*args, **kwargs):
        del args, kwargs
        return [str(frame)], [2]

    monkeypatch.setattr(fast_module, "_ORIGINAL_FRAME_SELECT", fake_select)
    state = {
        "remaining": 12000,
        "metadata": {"height": 224, "width": 224},
        "trace": [],
    }
    token = fast_module._ACTIVE_TOOL_STATE.set(state)
    try:
        paths, timestamps = asyncio.run(
            _budgeted_frame_select(
                "video.mp4",
                {
                    "start_time": 0,
                    "end_time": 4,
                    "nframes": 1,
                    "resize": 1.0,
                },
            )
        )
    finally:
        fast_module._ACTIVE_TOOL_STATE.reset(token)

    assert paths == [str(frame)]
    assert timestamps == [2]
    assert state["trace"][0]["frame_paths"] == [str(frame.resolve())]
    assert state["trace"][0]["timestamps"] == [2]
    assert state["trace"][0]["estimated_visual_tokens"] > 0


def test_trajectory_context_allows_only_public_provenance() -> None:
    context = _validated_trajectory_context(
        {
            "experiment_config_sha256": "A" * 64,
            "model_artifact_sha256": "b" * 64,
            "manifest_sha256": "c" * 64,
            "trajectory_schedule_id": "schedule",
            "trajectory_variant_id": "variant",
            "trajectory_replica_id": 2,
        }
    )
    assert context["experiment_config_sha256"] == "a" * 64
    assert context["trajectory_replica_id"] == 2

    try:
        _validated_trajectory_context({"answer": "A"})
    except ValueError as exc:
        assert "unsupported trajectory_context keys" in str(exc)
    else:
        raise AssertionError("private scoring keys must be rejected")

    try:
        _validated_trajectory_context({"manifest_sha256": "not-a-digest"})
    except ValueError as exc:
        assert "must be a SHA-256 digest" in str(exc)
    else:
        raise AssertionError("invalid provenance digests must be rejected")


def test_fast_hybrid_rejects_unconfirmed_change() -> None:
    evaluator, _ = _evaluator([_run_record("C"), _run_record("A")])
    sample = Sample(
        "cgbench",
        "two",
        "two.mp4",
        "What happens?",
        {"A": "first", "B": "second", "C": "third"},
        "B",
    )

    result = evaluator.fast_hybrid_eva(sample, "B")

    assert result["prediction"] == "B"
    assert result["candidate_changed"] is False
    assert result["change_gate_triggered"] is True
    assert result["change_rejection_reason"] == "independent_confirmation_disagreed"


def test_fast_hybrid_v2_forces_dense_official_confirmation_call() -> None:
    evaluator, _ = _evaluator([_run_record("C"), _run_record("C")])
    evaluator.version = "fast_hybrid_v2"
    sample = Sample(
        "lsdbench",
        "timed",
        "timed.mp4",
        "What happens from 04:40-04:46?",
        {"A": "first", "B": "second", "C": "third"},
        "C",
    )

    result = evaluator.fast_hybrid_eva(sample, "B")

    forced = evaluator._forced_calls[1]
    assert result["prediction"] == "C"
    assert forced is not None and len(forced) == 1
    assert forced[0]["tool"] == "frame_select"
    assert forced[0]["arguments"] == {
        "start_time": 279.0,
        "end_time": 287.0,
        "nframes": 32,
        "resize": 1.0,
    }


def test_fixed_evidence_replay_uses_only_frozen_calls() -> None:
    evaluator, prompts = _evaluator([_run_record("B")])
    sample = Sample(
        "lsdbench",
        "replay",
        "replay.mp4",
        "What happens?",
        {"A": "first", "B": "second"},
        "B",
        metadata={"time_range": "PRIVATE_TIME"},
    )
    planned = [
        {
            "start_time": 10.0,
            "end_time": 20.0,
            "nframes": 8,
            "resize": 0.5,
            "source_actual_timestamps": [10.0, 20.0],
        }
    ]

    result = evaluator.replay_fixed_evidence(sample, "A", planned)

    assert result["prediction"] == "B"
    assert result["fallback_to_candidate"] is False
    assert result["planned_calls_completed"] is True
    assert result["agent_total_tokens_complete"] is False
    assert evaluator._forced_calls == [
        [
            {
                "tool": "frame_select",
                "arguments": {
                    "start_time": 10.0,
                    "end_time": 20.0,
                    "nframes": 8,
                    "resize": 0.5,
                },
            }
        ]
    ]
    assert "Direct candidate: A" in prompts[0]
    assert "PRIVATE_TIME" not in json.dumps(prompts)


def test_fixed_evidence_replay_rejects_unapproved_tool_fields() -> None:
    evaluator, _ = _evaluator([])
    sample = Sample(
        "lsdbench",
        "replay",
        "replay.mp4",
        "What happens?",
        {"A": "first", "B": "second"},
        "B",
    )
    with pytest.raises(ValueError, match="unsupported fields"):
        evaluator.replay_fixed_evidence(
            sample,
            "A",
            [
                {
                    "start_time": 0,
                    "end_time": 1,
                    "nframes": 1,
                    "resize": 1.0,
                    "answer": "B",
                }
            ],
        )


def test_fast_hybrid_falls_back_when_verifier_has_no_answer() -> None:
    evaluator, _ = _evaluator([_run_record(None, tool=False)])
    sample = Sample(
        "lvbench",
        "three",
        "three.mp4",
        "What happens?",
        {"A": "first", "B": "second"},
        "A",
    )

    result = evaluator.fast_hybrid_eva(sample, "A")

    assert result["prediction"] == "A"
    assert result["fallback_to_candidate"] is True
    assert result["candidate_rerun"] == 0
    assert result["annotation_leak_check"] == "passed"


def test_fast_hybrid_summary_separates_missing_source_from_engineering_error(
    tmp_path: Path,
) -> None:
    direct_path = tmp_path / "direct.jsonl"
    agent_path = tmp_path / "agent.jsonl"
    direct_path.write_text(
        "\n".join(
            json.dumps(row)
            for row in (
                {"sample_id": "ok", "prediction": "A", "answer": "B", "correct": False},
                {
                    "sample_id": "missing",
                    "prediction": None,
                    "answer": "A",
                    "correct": False,
                    "error": "missing",
                    "data_unavailable": True,
                },
            )
        )
        + "\n",
        encoding="utf-8",
    )
    agent_path.write_text(
        "\n".join(
            json.dumps(row)
            for row in (
                {
                    "sample_id": "ok",
                    "prediction": "B",
                    "answer": "B",
                    "correct": True,
                    "annotation_leak_check": "passed",
                },
                {
                    "sample_id": "missing",
                    "prediction": None,
                    "answer": "A",
                    "correct": False,
                    "error": "missing",
                    "data_unavailable": True,
                    "annotation_leak_check": "passed",
                },
            )
        )
        + "\n",
        encoding="utf-8",
    )

    summary, _ = _metric("lvbench", direct_path, agent_path)

    assert summary["source_data_unavailable"] == 1
    assert summary["engineering_errors"] == 0
    assert summary["common_valid_samples"] == 1
    assert summary["gain"] == 1
    assert summary["passed"] is True
