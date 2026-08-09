from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.evaluate_mcq import (
    _candidate_superset_allowed,
    _validate_perception_memory_variant,
)
from flashvid_eval.client import ChatResult
from flashvid_eval.perception_memory_eva import (
    EvidenceDecision,
    EvidenceMemory,
    PerceptionMemoryEvaEvaluator,
    RESCUE_TRAJECTORY_VARIANTS,
    apply_candidate_gate,
    build_completeness_messages,
    build_confirmation_controller_messages,
    build_controller_messages,
    build_judge_messages,
    build_perception_messages,
    duplicate_interval,
    interval_iou,
    messages_have_media,
    parse_completeness,
    parse_controller_action,
    parse_perception_state,
    rescue_frame_request,
    validate_perception_state_observation,
)
from flashvid_eval.perception_memory_sft import build_perception_memory_sft_records
from flashvid_eval.privacy import assert_annotation_free_request
from flashvid_eval.qwen_agents.core import FrameObservation, FrameRequest
from flashvid_eval.schemas import ModelSample


def _sample(candidate: str | None = "PRIVATE_CANDIDATE_SENTINEL") -> ModelSample:
    return ModelSample(
        dataset="lvbench",
        sample_id="sample-1",
        video="video.mp4",
        question="What does the person do after opening the door?",
        choices={"A": "Sits down", "B": "Walks outside"},
        candidate_answer=candidate,
    )


def _observation(tmp_path: Path) -> FrameObservation:
    frame = tmp_path / "frame.png"
    frame.write_bytes(b"not-decoded-by-unit-test")
    request = FrameRequest(
        start_time=10.0,
        end_time=20.0,
        nframes=1,
        resize=0.75,
        evidence_request="Observe what happens after the door opens.",
    )
    return FrameObservation(
        request=request,
        resolved_start_time=10.0,
        resolved_end_time=20.0,
        resolved_nframes=1,
        frame_paths=(str(frame),),
        timestamps=(15.0,),
        backend="official_eva_select_frame_fallback",
        cache_hit=False,
        estimated_visual_tokens=128,
        latency_s=0.1,
    )


def _state_json(
    *,
    interval: tuple[float, float] = (10.0, 20.0),
    fact: str = "The person walks outside.",
    support_b: bool = True,
    contradict_a: bool = True,
    sufficient: bool = True,
    fact_time: float = 15.0,
) -> str:
    return json.dumps(
        {
            "interval": list(interval),
            "timestamped_facts": [{"time": fact_time, "fact": fact}],
            "option_evidence": {
                "A": {
                    "supports": [],
                    "contradicts": [fact] if contradict_a else [],
                },
                "B": {
                    "supports": [fact] if support_b else [],
                    "contradicts": [],
                },
            },
            "temporal_changes": ["The door changes from closed to open."],
            "unresolved": [] if sufficient else ["What happens next is not visible."],
            "evidence_sufficient": sufficient,
            "next_evidence_needed": "" if sufficient else "Observe later frames.",
        }
    )


def test_controller_is_candidate_blind_and_text_only() -> None:
    sample = _sample()
    memory = EvidenceMemory(sample.option_letters)
    messages = build_controller_messages(
        sample,
        memory,
        {"duration": 100.0, "width": 1920, "height": 1080},
    )

    serialized = json.dumps(messages, ensure_ascii=False)
    assert "PRIVATE_CANDIDATE_SENTINEL" not in serialized
    assert not messages_have_media(messages)
    assert_annotation_free_request({"messages": messages})


def test_perception_sees_only_current_frames_and_no_candidate_or_private_fields(
    tmp_path: Path,
) -> None:
    messages = build_perception_messages(
        _sample(),
        _observation(tmp_path),
        "Observe what happens after the door opens.",
    )

    serialized = json.dumps(messages, ensure_ascii=False)
    assert "PRIVATE_CANDIDATE_SENTINEL" not in serialized
    assert "time_range" not in serialized
    assert "clue_intervals" not in serialized
    assert "question_type" not in serialized
    assert messages_have_media(messages)
    assert serialized.count("image_url") == 2  # type plus payload key
    content = messages[-1]["content"]
    assert not content[0]["text"].startswith("<tool_response>")
    assert all(item.get("text") != "</tool_response>" for item in content)
    assert "at most 12 timestamped_facts" in messages[0]["content"]
    assert "at most one support and one contradiction per option" in messages[0]["content"]
    assert_annotation_free_request({"messages": messages})


def test_official_controller_parser_is_strict() -> None:
    call = (
        '<tool_call>{"tool":"frame_select","arguments":'
        '{"start_time":1,"end_time":4,"nframes":8,"resize":0.5,'
        '"evidence_request":"check the action"}}</tool_call>'
    )
    parsed = parse_controller_action(call)
    assert parsed is not None and parsed.action == "observe"
    assert parsed.request is not None and parsed.request.nframes == 8
    assert parse_controller_action('{"action":"stop"}').action == "stop"  # type: ignore[union-attr]
    assert parse_controller_action(call + " trailing prose") is None
    assert parse_controller_action(call.replace("<tool_call>", "")) is None


def test_memory_merge_persists_and_deduplicates_evidence() -> None:
    first = parse_perception_state(_state_json(), ("A", "B"))
    second = parse_perception_state(
        _state_json(
            interval=(30.0, 40.0),
            fact="The person picks up a bag.",
            support_b=False,
            contradict_a=False,
            sufficient=False,
        ),
        ("A", "B"),
    )
    assert first is not None and second is not None
    memory = EvidenceMemory(("A", "B"))
    memory.merge(first)
    first_ids = memory.evidence_ids
    memory.merge(second)

    assert first_ids < memory.evidence_ids
    assert memory.observed_intervals == [(10.0, 20.0), (30.0, 40.0)]
    assert memory.option_ledger["B"].supports
    assert any(item.fact == "The person walks outside." for item in memory.event_ledger)
    before = len(memory.event_ledger)
    memory.merge(first)
    assert len(memory.event_ledger) == before


def test_memory_keeps_repeated_actions_at_distinct_times_intervals_and_sources() -> None:
    repeated = json.loads(
        _state_json(fact="The person places a plate.", fact_time=12.0)
    )
    repeated["timestamped_facts"].append(
        {"time": 18.0, "fact": "The person places a plate."}
    )
    later = json.loads(
        _state_json(
            interval=(30.0, 40.0),
            fact="The person places a plate.",
            fact_time=35.0,
        )
    )
    first = parse_perception_state(json.dumps(repeated), ("A", "B"))
    second = parse_perception_state(json.dumps(later), ("A", "B"))
    assert first is not None and second is not None
    memory = EvidenceMemory(("A", "B"))

    memory.merge(first)
    memory.merge(second)

    timestamped = [
        item
        for item in memory.event_ledger
        if item.fact == "The person places a plate."
        and item.source == "timestamped_fact"
    ]
    assert [(item.interval, item.timestamp) for item in timestamped] == [
        ((10.0, 20.0), 12.0),
        ((10.0, 20.0), 18.0),
        ((30.0, 40.0), 35.0),
    ]
    assert {
        item.source
        for item in memory.event_ledger
        if item.fact == "The person places a plate."
    } == {"timestamped_fact"}
    timestamped_ids = tuple(item.evidence_id for item in timestamped)
    assert memory.option_ledger["A"].contradicts == timestamped_ids
    assert memory.option_ledger["B"].supports == timestamped_ids
    before = len(memory.event_ledger)
    memory.merge(second)
    assert len(memory.event_ledger) == before


def test_perception_timestamp_binds_to_actual_sampled_frame(tmp_path: Path) -> None:
    state = parse_perception_state(
        _state_json(fact_time=15.0004), ("A", "B")
    )
    assert state is not None

    validated = validate_perception_state_observation(state, _observation(tmp_path))

    assert validated.interval == (10.0, 20.0)
    assert validated.timestamped_facts[0].time == 15.0


def test_interval_iou_and_duplicate_rejection() -> None:
    assert interval_iou((0.0, 10.0), (0.0, 10.0)) == 1.0
    assert interval_iou((0.0, 10.0), (10.0, 20.0)) == 0.0
    assert duplicate_interval((0.5, 10.5), [(0.0, 10.0)], threshold=0.85)
    assert not duplicate_interval((20.0, 30.0), [(0.0, 10.0)])


def test_candidate_gate_requires_complete_supported_and_confirmed_change() -> None:
    state = parse_perception_state(_state_json(), ("A", "B"))
    assert state is not None
    memory = EvidenceMemory(("A", "B"))
    memory.merge(state)
    support = memory.option_ledger["B"].supports[0]
    contradiction = memory.option_ledger["A"].contradicts[0]
    evidence = EvidenceDecision("B", (support, contradiction))

    incomplete = apply_candidate_gate(
        candidate="A",
        first=evidence,
        confirmation=evidence,
        complete=False,
        memory=memory,
    )
    assert incomplete.prediction == "A"
    assert incomplete.fallback_to_candidate

    unconfirmed = apply_candidate_gate(
        candidate="A",
        first=evidence,
        confirmation=EvidenceDecision("A", (support,)),
        complete=True,
        memory=memory,
    )
    assert unconfirmed.prediction == "A"

    changed = apply_candidate_gate(
        candidate="A",
        first=evidence,
        confirmation=evidence,
        complete=True,
        memory=memory,
    )
    assert changed.prediction == "B"
    assert changed.source == "confirmed_visual_change"


def test_completeness_and_judge_prompts_remain_candidate_blind() -> None:
    sample = _sample()
    state = parse_perception_state(_state_json(), sample.option_letters)
    assert state is not None
    memory = EvidenceMemory(sample.option_letters)
    memory.merge(state)

    messages = build_completeness_messages(sample, memory) + build_judge_messages(
        sample, memory
    )
    assert "PRIVATE_CANDIDATE_SENTINEL" not in json.dumps(messages)
    assert not messages_have_media(messages)
    assert parse_completeness(
        '{"evidence_complete":false,"missing_evidence":["later action"]}'
    ).evidence_complete is False  # type: ignore[union-attr]


def test_confirmation_controller_hides_direct_branch_identity() -> None:
    sample = _sample("A")
    state = parse_perception_state(_state_json(), sample.option_letters)
    assert state is not None
    memory = EvidenceMemory(sample.option_letters)
    memory.merge(state)
    messages = build_confirmation_controller_messages(
        sample,
        memory,
        {"duration": 100.0, "width": 1920, "height": 1080},
        ("B", "A"),
    )
    serialized = json.dumps(messages)
    assert "Direct" not in serialized
    assert "candidate" not in serialized.lower()
    assert "Hypotheses (unordered): A, B" in serialized
    assert not messages_have_media(messages)


class _FakeSession:
    def __init__(self, observation: FrameObservation) -> None:
        self.metadata = {"duration": 100.0, "width": 1920, "height": 1080}
        self.observation = observation

    def select(self, request: FrameRequest) -> FrameObservation:
        assert request.evidence_request
        return self.observation


class _FakeFrameTool:
    def __init__(self, observation: FrameObservation) -> None:
        self.session = _FakeSession(observation)

    def open_session(self, video: Path, session_id: str) -> _FakeSession:
        assert video.is_file()
        assert session_id.startswith("pm-")
        return self.session


class _FakeClient:
    def __init__(self, outputs: list[str]) -> None:
        self.outputs = outputs
        self.messages: list[list[dict]] = []

    def chat(self, model: str, messages: list[dict], **kwargs: object) -> ChatResult:
        del model, kwargs
        self.messages.append(messages)
        content = self.outputs.pop(0)
        has_media = messages_have_media(messages)
        return ChatResult(
            content=content,
            usage={
                "prompt_tokens": 100,
                "completion_tokens": 10,
                "total_tokens": 110,
                "prompt_tokens_details": {"multimodal_tokens": 50 if has_media else 0},
            },
            raw={},
            latency_s=0.01,
            finish_reason="stop",
        )


class _DurationOnlySession:
    def __init__(self, root: Path, duration: float = 100.0) -> None:
        self.root = root
        self.metadata = {
            "duration": duration,
            "width": 1920,
            "height": 1080,
            "answer": "SECRET_ANNOTATION_SENTINEL",
            "time_range": [999.0, 1000.0],
            "clue_intervals": [[999.0, 1000.0]],
            "question_type": "SECRET_ANNOTATION_SENTINEL",
        }
        self.requests: list[FrameRequest] = []

    def select(self, request: FrameRequest) -> FrameObservation:
        self.requests.append(request)
        count = int(request.nframes or 1)
        if count == 1:
            timestamps = ((request.start_time + request.end_time) / 2.0,)
        else:
            span = request.end_time - request.start_time
            timestamps = tuple(
                request.start_time + span * index / (count - 1)
                for index in range(count)
            )
        paths = []
        for index in range(count):
            path = self.root / f"rescue-{len(self.requests)}-{index}.jpg"
            path.write_bytes(b"cached test frame")
            paths.append(str(path.resolve()))
        return FrameObservation(
            request=request,
            resolved_start_time=request.start_time,
            resolved_end_time=request.end_time,
            resolved_nframes=count,
            frame_paths=tuple(paths),
            timestamps=timestamps,
            backend="official_eva_select_frame_fallback",
            cache_hit=False,
            estimated_visual_tokens=count * 128,
            latency_s=0.01,
        )


class _DurationOnlyFrameTool:
    def __init__(self, session: _DurationOnlySession) -> None:
        self.session = session

    def open_session(self, video: Path, session_id: str) -> _DurationOnlySession:
        assert video.is_file()
        assert session_id.startswith("pm-")
        return self.session


@pytest.mark.parametrize(
    ("variant", "expected"),
    [
        ("rescue_global32", (0.0, 100.0, 32)),
        ("rescue_global64", (0.0, 100.0, 64)),
        ("rescue_first_half64", (0.0, 50.0, 64)),
        ("rescue_second_half64", (50.0, 100.0, 64)),
    ],
)
def test_rescue_frame_request_is_duration_only_and_pre_registered(
    variant: str, expected: tuple[float, float, int]
) -> None:
    request = rescue_frame_request(variant, 100.0)
    assert request is not None
    assert (request.start_time, request.end_time, request.nframes) == expected
    assert request.resize == 0.75
    assert set(RESCUE_TRAJECTORY_VARIANTS) == {
        "rescue_global32",
        "rescue_global64",
        "rescue_first_half64",
        "rescue_second_half64",
    }
    assert rescue_frame_request("base", 100.0) is None


def test_rescue_first_action_is_private_free_then_controller_resumes_and_is_sft_ready(
    tmp_path: Path,
) -> None:
    video = tmp_path / "video.mp4"
    video.write_bytes(b"placeholder")
    session = _DurationOnlySession(tmp_path)
    client = _FakeClient(
        [
            _state_json(interval=(0.0, 50.0), fact_time=0.0),
            '{"action":"stop"}',
            '{"evidence_complete":true,"missing_evidence":[]}',
            '{"answer":"A","evidence_ids":["E0001"]}',
        ]
    )
    evaluator = PerceptionMemoryEvaEvaluator(
        client,
        "Qwen3.5-9B",
        tmp_path,
        tmp_path / "frames",
        frame_tool=_DurationOnlyFrameTool(session),  # type: ignore[arg-type]
        max_turns=2,
        scoring_deferred=True,
        train600_manifest_sha256="3" * 64,
        trajectory_schedule_id="rescue-schedule-v1",
        trajectory_variant_id="rescue_first_half64",
    )

    result = evaluator.run(_sample("A"))

    assert result["error"] is None
    assert len(session.requests) == 1
    first_request = session.requests[0]
    assert (first_request.start_time, first_request.end_time, first_request.nframes) == (
        0.0,
        50.0,
        64,
    )
    controller_trace = [
        item for item in result["request_trace"] if item["stage"] == "controller"
    ]
    assert controller_trace[0]["source_cached_action"] is True
    assert controller_trace[0]["trajectory_variant_id"] == "rescue_first_half64"
    assert controller_trace[0]["action_accepted"] is True
    assert controller_trace[0]["content"].startswith("<tool_call>")
    assert "source_cached_action" not in controller_trace[1]
    assert len(client.messages) == 4
    assert messages_have_media(client.messages[0])
    assert not messages_have_media(client.messages[1])
    serialized = json.dumps(
        {"requests": client.messages, "trace": result["request_trace"]},
        ensure_ascii=False,
    )
    assert "PRIVATE_CANDIDATE_SENTINEL" not in serialized
    assert "SECRET_ANNOTATION_SENTINEL" not in serialized
    assert "time_range" not in serialized
    assert "clue_intervals" not in serialized
    assert "question_type" not in serialized

    # Offline prefix judging normally supplies these three confirmations.  Once
    # present, the synthetic official tool call is directly exportable as the
    # first controller target rather than becoming an unsupervised side effect.
    judge_trace = next(
        item for item in result["request_trace"] if item["stage"] == "evidence_judge"
    )
    result["perception_states"][0]["judge_confirmations"] = [
        {
            "judge_seed": seed,
            "prediction": "A",
            "evidence_ids": ["E0001"],
            "request_messages": judge_trace["messages"],
            "raw_response": judge_trace["content"],
            "evidence_complete": True,
            "annotation_leak_check": "passed",
            "error": None,
        }
        for seed in (17, 42, 73)
    ]
    result["_selection_stable"] = True
    records = build_perception_memory_sft_records(result)
    assert records[0]["metadata"]["episode_target_type"] == "tool"
    assert "frame_select" in records[0]["messages"][-1]["content"]


def test_rescue_variant_is_fail_closed_and_changes_run_fingerprint(
    tmp_path: Path,
) -> None:
    common = {
        "client": _FakeClient([]),
        "model": "Qwen3.5-9B",
        "video_root": tmp_path,
        "frame_root": tmp_path / "frames",
        "frame_tool": _FakeFrameTool(_observation(tmp_path)),
        "scoring_deferred": True,
        "train600_manifest_sha256": "3" * 64,
        "trajectory_schedule_id": "rescue-schedule-v1",
    }
    base = PerceptionMemoryEvaEvaluator(
        **common, trajectory_variant_id="base"  # type: ignore[arg-type]
    )
    rescue = PerceptionMemoryEvaEvaluator(
        **common, trajectory_variant_id="rescue_global32"  # type: ignore[arg-type]
    )
    assert base.run_fingerprint() != rescue.run_fingerprint()
    with pytest.raises(ValueError, match="unsupported"):
        PerceptionMemoryEvaEvaluator(
            **common, trajectory_variant_id="rescue_custom"  # type: ignore[arg-type]
        )
    with pytest.raises(ValueError, match="require scoring_deferred"):
        PerceptionMemoryEvaEvaluator(
            _FakeClient([]),
            "Qwen3.5-9B",
            tmp_path,
            tmp_path / "frames",
            frame_tool=_FakeFrameTool(_observation(tmp_path)),  # type: ignore[arg-type]
            trajectory_variant_id="rescue_global32",
        )
    with pytest.raises(ValueError, match="unsupported"):
        rescue_frame_request("rescue_custom", 100.0)
    assert _validate_perception_memory_variant("rescue_global64") == (
        "rescue_global64"
    )
    with pytest.raises(ValueError, match="trajectory-variant-id"):
        _validate_perception_memory_variant("rescue_custom")
    assert _candidate_superset_allowed(
        "perception_memory_eva", True, "rescue_global64"
    )
    assert not _candidate_superset_allowed(
        "perception_memory_eva", False, "rescue_global64"
    )
    assert not _candidate_superset_allowed("fast_hybrid_eva", True, "rescue_global64")


def test_evaluator_keeps_images_out_of_later_controller_and_judge(
    tmp_path: Path,
) -> None:
    video = tmp_path / "video.mp4"
    video.write_bytes(b"placeholder")
    tool_call = (
        '<tool_call>{"tool":"frame_select","arguments":'
        '{"start_time":10,"end_time":20,"nframes":1,"resize":0.75,'
        '"evidence_request":"check the action"}}</tool_call>'
    )
    client = _FakeClient(
        [
            tool_call,
            _state_json(),
            '{"action":"stop"}',
            '{"evidence_complete":true,"missing_evidence":[]}',
            '{"answer":"A","evidence_ids":["E0001"]}',
        ]
    )
    evaluator = PerceptionMemoryEvaEvaluator(
        client,
        "Qwen3.5-9B",
        tmp_path,
        tmp_path / "frames",
        frame_tool=_FakeFrameTool(_observation(tmp_path)),  # type: ignore[arg-type]
        max_turns=2,
    )
    result = evaluator.run(_sample("A"))

    assert result["final_prediction"] == "A"
    assert result["error"] is None
    assert result["visual_tokens"] == 50
    media_by_stage = {
        item["stage"]: messages_have_media(item["messages"])
        for item in result["request_trace"]
    }
    assert media_by_stage == {
        "controller": False,
        "perception": True,
        "completeness": False,
        "evidence_judge": False,
    }
    assert [item["prefix_index"] for item in result["request_trace"]] == [
        -1,
        0,
        0,
        0,
        0,
    ]
    controller_actions = [
        item for item in result["request_trace"] if item["stage"] == "controller"
    ]
    assert [item["action_accepted"] for item in controller_actions] == [True, True]
    assert result["perception_states"][0]["frame_paths"]
    assert result["candidate_rerun"] == 0
    assert set(result["public_sample"]) == {
        "dataset",
        "sample_id",
        "video",
        "question",
        "choices",
    }
    assert "PRIVATE_CANDIDATE_SENTINEL" not in json.dumps(result["public_sample"])
    assert len(result["run_fingerprint"]) == 64
    audit = evaluator.static_audit_fields()
    assert audit["backend"] == "perception_memory_eva"
    assert len(audit["implementation_sha256"]) == 64
    assert len(audit["implementation_bundle_sha256"]) == 64


@pytest.mark.parametrize(
    ("fact_time", "expected_error"),
    [
        (9999.0, "outside the resolved perception interval"),
        (16.0, "does not match an actual sampled frame timestamp"),
    ],
)
def test_evaluator_rejects_unbound_perception_timestamps_and_falls_back(
    tmp_path: Path,
    fact_time: float,
    expected_error: str,
) -> None:
    video = tmp_path / "video.mp4"
    video.write_bytes(b"placeholder")
    tool_call = (
        '<tool_call>{"tool":"frame_select","arguments":'
        '{"start_time":10,"end_time":20,"nframes":1,"resize":0.75,'
        '"evidence_request":"check the action"}}</tool_call>'
    )
    client = _FakeClient([tool_call, _state_json(fact_time=fact_time)])
    evaluator = PerceptionMemoryEvaEvaluator(
        client,
        "Qwen3.5-9B",
        tmp_path,
        tmp_path / "frames",
        frame_tool=_FakeFrameTool(_observation(tmp_path)),  # type: ignore[arg-type]
        max_turns=2,
    )

    result = evaluator.run(_sample("A"))

    assert result["final_prediction"] == "A"
    assert result["fallback_to_candidate"] is True
    assert result["error_type"] == "ValueError"
    assert expected_error in result["error"]
    assert result["event_ledger"] == []
    assert result["perception_states"] == []
    perception_request = result["request_trace"][-1]
    assert perception_request["stage"] == "perception"
    assert expected_error in perception_request["state_validation_error"]


def test_deferred_trajectory_provenance_is_part_of_fingerprint(
    tmp_path: Path,
) -> None:
    evaluator = PerceptionMemoryEvaEvaluator(
        _FakeClient([]),
        "Qwen3.5-9B",
        tmp_path,
        tmp_path / "frames",
        frame_tool=_FakeFrameTool(_observation(tmp_path)),  # type: ignore[arg-type]
        scoring_deferred=True,
        train600_manifest_sha256="3" * 64,
        trajectory_schedule_id="budget-24000-seed-42",
        trajectory_variant_id="base",
        trajectory_replica_id=2,
    )
    audit = evaluator.static_audit_fields()
    assert audit["scoring_deferred"] is True
    assert audit["train600_manifest_sha256"] == "3" * 64
    assert audit["trajectory_replica_id"] == 2
    assert len(evaluator.run_fingerprint()) == 64


def test_rejected_stop_does_not_create_a_gap_in_perception_prefixes(
    tmp_path: Path,
) -> None:
    video = tmp_path / "video.mp4"
    video.write_bytes(b"placeholder")
    tool_call = (
        '<tool_call>{"tool":"frame_select","arguments":'
        '{"start_time":10,"end_time":20,"nframes":1,"resize":0.75,'
        '"evidence_request":"check the action"}}</tool_call>'
    )
    client = _FakeClient(
        [
            '{"action":"stop"}',
            '{"evidence_complete":false,"missing_evidence":["visual action"]}',
            tool_call,
            _state_json(),
            '{"action":"stop"}',
            '{"evidence_complete":true,"missing_evidence":[]}',
            '{"answer":"A","evidence_ids":["E0001"]}',
        ]
    )
    evaluator = PerceptionMemoryEvaEvaluator(
        client,
        "Qwen3.5-9B",
        tmp_path,
        tmp_path / "frames",
        frame_tool=_FakeFrameTool(_observation(tmp_path)),  # type: ignore[arg-type]
        max_turns=3,
    )
    result = evaluator.run(_sample("A"))
    assert [state["step_index"] for state in result["perception_states"]] == [0]
    controllers = [
        item for item in result["request_trace"] if item["stage"] == "controller"
    ]
    assert [item["prefix_index"] for item in controllers] == [-1, -1, 0]
    assert [item["action_accepted"] for item in controllers] == [False, True, True]
