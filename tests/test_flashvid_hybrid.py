from __future__ import annotations

import json
from pathlib import Path

import pytest

import flashvid_eval.flashvid_hybrid as flashvid_hybrid
from flashvid_eval.client import ChatResult
from flashvid_eval.flashvid_budget import BudgetEndpoint, BudgetEndpointPool
from flashvid_eval.flashvid_hybrid import (
    PERCEPTION_MAX_TOKENS,
    PERCEPTION_RESPONSE_FORMAT,
    PERCEPTION_SYSTEM_PROMPT,
    FlashVIDHybridConfig,
    FlashVIDHybridEvaluator,
    _api_multimodal_video_tokens,
    build_controller_user_prompt,
    build_perception_messages,
    deterministic_random_ratio,
    parse_budget_tool_calls,
    parse_perception_json,
)
from flashvid_eval.schemas import ModelSample, Sample


def _pool() -> BudgetEndpointPool:
    return BudgetEndpointPool(
        [
            BudgetEndpoint(ratio, f"http://127.0.0.1:{8101 + index}/v1", "P4")
            for index, ratio in enumerate((0.10, 0.25, 0.50, 1.00))
        ]
    )


def _fingerprint_evaluator(tmp_path: Path) -> FlashVIDHybridEvaluator:
    return FlashVIDHybridEvaluator(
        _FakeController([]),
        "C9",
        _pool(),
        tmp_path,
        tmp_path / "frames",
        tmp_path / "cache",
        tmp_path / "media",
        FlashVIDHybridConfig(),
    )


def test_core_implementation_hashes_are_frozen_and_change_run_fingerprint(
    tmp_path: Path,
    monkeypatch,
) -> None:
    first_hashes = {
        "scripts/evaluate_mcq.py": "a" * 64,
        "src/flashvid_eval/runner.py": "b" * 64,
        "src/flashvid_eval/flashvid_hybrid.py": "c" * 64,
    }
    second_hashes = {**first_hashes, "src/flashvid_eval/runner.py": "d" * 64}
    monkeypatch.setattr(
        flashvid_hybrid,
        "core_implementation_sha256",
        lambda: first_hashes,
    )
    first = _fingerprint_evaluator(tmp_path / "first")
    first.freeze_artifacts(tmp_path / "frozen")
    frozen = json.loads((tmp_path / "frozen" / "frozen_config.json").read_text())
    assert frozen["implementation_sha256"] == first_hashes

    monkeypatch.setattr(
        flashvid_hybrid,
        "core_implementation_sha256",
        lambda: second_hashes,
    )
    second = _fingerprint_evaluator(tmp_path / "second")
    assert second.run_fingerprint() != first.run_fingerprint()


def test_core_implementation_hashes_include_budget_strategy_and_prompt_registry() -> None:
    hashes = flashvid_hybrid.core_implementation_sha256()
    assert "src/flashvid_eval/budget_strategies.py" in hashes
    assert "src/flashvid_eval/prompt_registry.py" in hashes
    assert all(len(value) == 64 for value in hashes.values())


def test_static_audit_fields_cover_source_video_unavailable_fallback(
    tmp_path: Path,
) -> None:
    evaluator = _fingerprint_evaluator(tmp_path)
    fields = evaluator.static_audit_fields()
    assert fields["run_fingerprint"] == evaluator.run_fingerprint()
    assert fields["controller_model"] == "C9"
    assert fields["budget_policy"] == "model"
    assert fields["budget_sequence"] == []
    assert len(fields["controller_prompt_hash"]) == 64
    assert len(fields["perception_prompt_hash"]) == 64


def test_resume_rejects_results_from_a_different_implementation(
    tmp_path: Path,
    monkeypatch,
) -> None:
    first_hashes = {
        "scripts/evaluate_mcq.py": "a" * 64,
        "src/flashvid_eval/runner.py": "b" * 64,
        "src/flashvid_eval/flashvid_hybrid.py": "c" * 64,
    }
    monkeypatch.setattr(
        flashvid_hybrid,
        "core_implementation_sha256",
        lambda: first_hashes,
    )
    first = _fingerprint_evaluator(tmp_path / "first")
    output_dir = tmp_path / "results"
    output_dir.mkdir()
    (output_dir / "lsdbench_flashvid_hybrid_trajectories.jsonl").write_text(
        json.dumps(
            {
                "trajectory_id": "sample:0",
                "run_fingerprint": first.run_fingerprint(),
            }
        )
        + "\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(
        flashvid_hybrid,
        "core_implementation_sha256",
        lambda: {**first_hashes, "scripts/evaluate_mcq.py": "d" * 64},
    )
    second = _fingerprint_evaluator(tmp_path / "second")
    sample = Sample(
        "lsdbench",
        "sample",
        "v.mp4",
        "What happens?",
        {"A": "first", "B": "second"},
        "A",
    )
    with pytest.raises(RuntimeError, match="resume fingerprint mismatch"):
        flashvid_hybrid.evaluate_flashvid_trajectories(
            [sample],
            second,
            {"sample": "A"},
            output_dir,
            trajectories_per_sample=4,
            resume=True,
        )


def test_api_multimodal_video_token_details_are_read_without_faking_visual_actuals():
    usage = {
        "prompt_tokens": 123,
        "prompt_tokens_details": {"multimodal_tokens": {"video": 96}},
    }
    assert _api_multimodal_video_tokens(usage) == 96
    assert _api_multimodal_video_tokens({"prompt_tokens": 123}) is None
    assert _api_multimodal_video_tokens(
        {"prompt_tokens_details": {"multimodal_tokens": {"video": True}}}
    ) is None


class _FakeController:
    def __init__(self, responses: list[str]):
        self.responses = list(responses)
        self.requests: list[list[dict]] = []

    def chat(self, model, messages, **kwargs):
        self.requests.append(json.loads(json.dumps(messages)))
        content = self.responses.pop(0)
        return ChatResult(
            content=content,
            usage={"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12},
            raw={},
            latency_s=0.01,
        )


def _call(answer_request: str = "Inspect the event.", ratio: float = 1.0) -> str:
    return (
        "<tool_call>"
        + json.dumps(
            {
                "tool": "frame_select",
                "arguments": {
                    "start_time": 0,
                    "end_time": 10,
                    "nframes": 8,
                    "resize": 0.5,
                    "retention_ratio": ratio,
                    "evidence_request": answer_request,
                },
            }
        )
        + "</tool_call>"
    )


def _fake_trace(call: dict, *, cache_hit: bool = False) -> dict:
    return {
        "start_time": call["start_time"],
        "end_time": call["end_time"],
        "actual_timestamps": [0.0, 1.0],
        "nframes_requested": call["nframes"],
        "nframes_actual": 2,
        "resize": call["resize"],
        "retention_ratio": call["retention_ratio"],
        "raw_visual_tokens": 100,
        "retained_visual_tokens": int(100 * call["retention_ratio"]),
        "effective_retention_ratio": call["retention_ratio"],
        "token_count_source": "test",
        "evidence_request": call["evidence_request"],
        "frame_backend": "fake",
        "perception_endpoint": "http://127.0.0.1:8101/v1",
        "perception_model": "P4",
        "perception_usage": {
            "prompt_tokens": 20,
            "completion_tokens": 3,
            "total_tokens": 23,
        },
        "perception_latency_s": 0.02,
        "media": {},
        "cache_hit": cache_hit,
        "cache_key": "key",
    }


def test_extended_tool_parser_is_official_and_ratio_strict() -> None:
    assert parse_budget_tool_calls(_call(ratio=0.25))[0]["retention_ratio"] == 0.25
    assert parse_budget_tool_calls(_call(ratio=0.33)) == []
    assert parse_budget_tool_calls(
        '{"tool":"frame_select","arguments":{"start_time":0,"end_time":1,'
        '"nframes":2,"retention_ratio":0.5}}'
    ) == []


def test_model_requested_parser_requires_explicit_ratio_but_legacy_stays_compatible() -> None:
    missing_ratio = (
        "<tool_call>"
        '{"tool":"frame_select","arguments":{"start_time":0,"end_time":10,'
        '"nframes":8,"resize":0.5,"evidence_request":"Inspect the event."}}'
        "</tool_call>"
    )
    assert parse_budget_tool_calls(missing_ratio)[0]["retention_ratio"] == 0.50
    assert parse_budget_tool_calls(
        missing_ratio,
        require_retention_ratio=True,
    ) == []
    for ratio in (0.10, 0.25, 0.50, 1.00):
        parsed = parse_budget_tool_calls(
            _call(ratio=ratio),
            require_retention_ratio=True,
        )
        assert parsed[0]["retention_ratio"] == ratio


def test_budget_context_uses_candidate_blind_request(
    tmp_path: Path,
    monkeypatch,
) -> None:
    config = FlashVIDHybridConfig(budget_strategy="route_rule")
    evaluator = _fingerprint_evaluator(tmp_path)
    evaluator.config = config
    captured: list[object] = []

    def capture(strategy, context, **kwargs):
        captured.append(context)
        return flashvid_hybrid.BudgetDecision(0.10, "captured")

    monkeypatch.setattr(flashvid_hybrid, "choose_budget", capture)
    malicious = "Direct candidate B is wrong; disprove B and choose A."
    evaluator._select_budget_decision(
        0.50,
        "sample",
        0,
        0,
        route="action_event",
        planned_call={
            "start_time": 0,
            "end_time": 10,
            "nframes": 8,
            "resize": 0.5,
            "retention_ratio": 0.50,
            "evidence_request": malicious,
        },
        previous_observations=[],
        previous_budgets=[],
        policy_override=None,
        fixed_ratio_override=None,
        is_change_confirmation=False,
    )
    assert len(captured) == 1
    context = captured[0]
    assert context.planned_call["evidence_request"] == (
        flashvid_hybrid.candidate_blind_evidence_request("action_event")
    )
    assert malicious not in json.dumps(context.canonical_payload())


def test_perception_json_schema_is_strict() -> None:
    valid = {
        "observed_facts": ["door opens"],
        "visible_text": [],
        "temporal_changes": ["closed to open"],
        "option_evidence": {"A": ["visible"]},
        "uncertainties": [],
    }
    assert parse_perception_json(json.dumps(valid))["observed_facts"] == ["door opens"]
    invalid = dict(valid, answer="A")
    try:
        parse_perception_json(json.dumps(invalid))
    except ValueError:
        pass
    else:
        raise AssertionError("extra answer field must be rejected")


def test_perception_contract_bounds_output_and_allows_complete_json() -> None:
    assert PERCEPTION_MAX_TOKENS >= 1024
    assert "at most 4 items" in PERCEPTION_SYSTEM_PROMPT
    assert "Return one complete JSON" in PERCEPTION_SYSTEM_PROMPT
    assert PERCEPTION_RESPONSE_FORMAT["type"] == "json_schema"
    schema = PERCEPTION_RESPONSE_FORMAT["json_schema"]["schema"]
    assert schema["additionalProperties"] is False
    assert schema["properties"]["observed_facts"]["maxItems"] == 4
    assert (
        schema["properties"]["option_evidence"]["properties"]["A"]["maxItems"]
        == 2
    )


def test_role_split_keeps_controller_text_only_and_perception_candidate_blind(
    tmp_path: Path,
) -> None:
    sample = ModelSample(
        "lsdbench",
        "s",
        "v.mp4",
        "What happens?",
        {"A": "SECRET_TIME_SENTINEL", "B": "door opens"},
        "B",
    )
    controller = build_controller_user_prompt(
        sample,
        {"duration": 10.0, "width": 320, "height": 240},
        "action_event",
    )
    assert "Direct candidate hypothesis: B" in controller
    assert "video_url" not in controller
    media = tmp_path / "clip.mp4"
    media.write_bytes(b"x")
    perception = build_perception_messages(
        media,
        sample.question,
        sample.choices,
        "Inspect the door.",
        [0.0, 1.0],
    )
    serialized = json.dumps(perception)
    assert "Direct candidate" not in serialized
    assert "candidate_answer" not in serialized
    assert "time_range" not in serialized
    assert sum(
        item.get("type") == "video_url"
        for message in perception
        for item in (message["content"] if isinstance(message["content"], list) else [])
    ) == 1


def test_random_budget_is_stable_and_uses_allowed_ratios() -> None:
    first = [
        deterministic_random_ratio("sample", 7, step)
        for step in range(10)
    ]
    second = [
        deterministic_random_ratio("sample", 7, step)
        for step in range(10)
    ]
    assert first == second
    assert set(first) <= {0.10, 0.25, 0.50, 1.00}


def test_twenty_random_trajectories_are_stepwise_stratified() -> None:
    sequences = [
        tuple(
            deterministic_random_ratio("sample", trajectory, step)
            for step in range(5)
        )
        for trajectory in range(4, 24)
    ]
    assert len(set(sequences)) == 20
    for step in range(5):
        counts = {
            ratio: sum(sequence[step] == ratio for sequence in sequences)
            for ratio in (0.10, 0.25, 0.50, 1.00)
        }
        assert counts == {0.10: 5, 0.25: 5, 0.50: 5, 1.00: 5}


def test_fixed_policy_overrides_model_ratio_and_never_reruns_direct(
    tmp_path: Path,
    monkeypatch,
) -> None:
    video = tmp_path / "v.mp4"
    video.write_bytes(b"x")
    controller = _FakeController([_call(ratio=1.0), "Answer: B"])
    evaluator = FlashVIDHybridEvaluator(
        controller,
        "C9",
        _pool(),
        tmp_path,
        tmp_path / "frames",
        tmp_path / "cache",
        tmp_path / "media",
        FlashVIDHybridConfig(
            budget_policy="fixed",
            fixed_retention_ratio=0.10,
            max_turns=6,
        ),
    )
    monkeypatch.setattr(
        "flashvid_eval.flashvid_hybrid.probe_video",
        lambda _: {"duration": 10.0, "width": 320, "height": 240},
    )
    monkeypatch.setattr(
        evaluator,
        "_perceive",
        lambda sample, video, metadata, call, trajectory_index, step: (
            {
                "observed_facts": ["door opens"],
                "visible_text": [],
                "temporal_changes": [],
                "option_evidence": {},
                "uncertainties": [],
            },
            _fake_trace(call),
        ),
    )
    sample = ModelSample(
        "lsdbench",
        "s",
        "v.mp4",
        "What happens?",
        {"A": "closes", "B": "opens"},
        "B",
    )
    result = evaluator.flashvid_hybrid(sample)
    assert result["prediction"] == "B"
    assert result["candidate_rerun"] == 0
    assert result["tool_steps"][0]["retention_ratio"] == 0.10
    assert all(
        "video_url" not in json.dumps(request)
        for request in controller.requests
    )


def test_two_tool_calls_in_one_turn_receive_distinct_budget_steps(
    tmp_path: Path,
    monkeypatch,
) -> None:
    video = tmp_path / "v.mp4"
    video.write_bytes(b"x")

    def interval_call(start: int, end: int) -> str:
        payload = {
            "tool": "frame_select",
            "arguments": {
                "start_time": start,
                "end_time": end,
                "nframes": 8,
                "resize": 0.5,
                "retention_ratio": 0.50,
                "evidence_request": "Inspect the event.",
            },
        }
        return f"<tool_call>{json.dumps(payload)}</tool_call>"

    controller = _FakeController(
        [
            interval_call(0, 5),
            interval_call(6, 10) + interval_call(11, 15),
            "Answer: B",
        ]
    )
    config = FlashVIDHybridConfig(
        budget_strategy="route_rule",
        max_turns=6,
    )
    evaluator = FlashVIDHybridEvaluator(
        controller,
        "C9",
        _pool(),
        tmp_path,
        tmp_path / "frames",
        tmp_path / "cache",
        tmp_path / "media",
        config,
    )
    monkeypatch.setattr(
        "flashvid_eval.flashvid_hybrid.probe_video",
        lambda _: {"duration": 30.0, "width": 320, "height": 240},
    )
    monkeypatch.setattr(
        evaluator,
        "_perceive",
        lambda sample, video, metadata, call, trajectory_index, step: (
            {
                "observed_facts": ["visible event"],
                "visible_text": [],
                "temporal_changes": [],
                "option_evidence": {},
                "uncertainties": [],
            },
            _fake_trace(call),
        ),
    )
    captured_steps: list[int] = []
    original_choose = flashvid_hybrid.choose_budget

    def capture(strategy, context, **kwargs):
        captured_steps.append(context.step_index)
        return original_choose(strategy, context, **kwargs)

    monkeypatch.setattr(flashvid_hybrid, "choose_budget", capture)
    result = evaluator.flashvid_hybrid(
        ModelSample(
            "lsdbench",
            "s",
            "v.mp4",
            "What happens?",
            {"A": "closes", "B": "opens"},
            "B",
        )
    )
    assert result["prediction"] == "B"
    assert len(result["tool_steps"]) == 3
    assert captured_steps == [0, 1, 2]


def test_model_requested_missing_ratio_is_recorded_as_parse_failure(
    tmp_path: Path,
    monkeypatch,
) -> None:
    video = tmp_path / "v.mp4"
    video.write_bytes(b"x")
    missing_ratio = (
        "<tool_call>"
        '{"tool":"frame_select","arguments":{"start_time":0,"end_time":10,'
        '"nframes":8,"resize":0.5,"evidence_request":"Inspect the event."}}'
        "</tool_call>"
    )
    controller = _FakeController([missing_ratio, "Answer: B"])
    evaluator = FlashVIDHybridEvaluator(
        controller,
        "C9",
        _pool(),
        tmp_path,
        tmp_path / "frames",
        tmp_path / "cache",
        tmp_path / "media",
        FlashVIDHybridConfig(
            budget_strategy="model_requested",
            max_turns=6,
        ),
    )
    monkeypatch.setattr(
        "flashvid_eval.flashvid_hybrid.probe_video",
        lambda _: {"duration": 30.0, "width": 320, "height": 240},
    )
    monkeypatch.setattr(
        evaluator,
        "_perceive",
        lambda sample, video, metadata, call, trajectory_index, step: (
            {
                "observed_facts": ["visible event"],
                "visible_text": [],
                "temporal_changes": [],
                "option_evidence": {},
                "uncertainties": [],
            },
            _fake_trace(call),
        ),
    )
    result = evaluator.flashvid_hybrid(
        ModelSample(
            "lsdbench",
            "s",
            "v.mp4",
            "What happens?",
            {"A": "closes", "B": "opens"},
            "B",
        )
    )
    assert result["prediction"] == "B"
    assert result["parse_error"] == "initial_tool_parse_fallback"
    assert result["trajectory_valid"] is False


def test_changed_candidate_requires_second_observation(
    tmp_path: Path,
    monkeypatch,
) -> None:
    video = tmp_path / "v.mp4"
    video.write_bytes(b"x")
    second_call = (
        "<tool_call>"
        '{"tool":"frame_select","arguments":{"start_time":11,"end_time":20,'
        '"nframes":8,"resize":0.5,"retention_ratio":0.5,'
        '"evidence_request":"Confirm the door state."}}'
        "</tool_call>"
    )
    controller = _FakeController(
        [_call(ratio=0.5), "Answer: C", second_call, "Answer: C"]
    )
    evaluator = FlashVIDHybridEvaluator(
        controller,
        "C9",
        _pool(),
        tmp_path,
        tmp_path / "frames",
        tmp_path / "cache",
        tmp_path / "media",
        FlashVIDHybridConfig(max_turns=6),
    )
    monkeypatch.setattr(
        "flashvid_eval.flashvid_hybrid.probe_video",
        lambda _: {"duration": 30.0, "width": 320, "height": 240},
    )
    monkeypatch.setattr(
        evaluator,
        "_perceive",
        lambda sample, video, metadata, call, trajectory_index, step: (
            {
                "observed_facts": ["visible evidence"],
                "visible_text": [],
                "temporal_changes": [],
                "option_evidence": {},
                "uncertainties": [],
            },
            _fake_trace(call),
        ),
    )
    sample = ModelSample(
        "lsdbench",
        "s",
        "v.mp4",
        "What happens?",
        {"A": "first", "B": "second", "C": "third"},
        "B",
    )
    result = evaluator.flashvid_hybrid(sample)
    assert result["prediction"] == "C"
    assert result["candidate_changed"] is True
    assert result["decision_source"] == "confirmed_visual_change"
    assert len(result["tool_steps"]) == 2


def test_private_scoring_fields_cannot_enter_model_sample() -> None:
    sample = Sample(
        "cgbench",
        "s",
        "v.mp4",
        "What happens?",
        {"A": "first", "B": "second"},
        "B",
        {
            "time_range": "SECRET_TIME",
            "clue_intervals": "SECRET_CLUE",
            "question_type": "SECRET_TYPE",
        },
    )
    model_sample = ModelSample.from_sample(sample, "A")
    assert "SECRET_" not in json.dumps(model_sample.__dict__)
