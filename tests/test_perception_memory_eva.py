from __future__ import annotations

import json
import math
import re
from dataclasses import replace
from pathlib import Path

import pytest

from scripts.evaluate_mcq import (
    _candidate_superset_allowed,
    _load_perception_memory_role_config,
    _validate_perception_memory_variant,
)
from flashvid_eval.client import ChatResult
from flashvid_eval.perception_memory_eva import (
    DURATION_RESCUE_TRAJECTORY_VARIANTS,
    EvidenceDecision,
    EvidenceMemory,
    PERCEPTION_NORMALIZATION_VERSION,
    PERCEPTION_MEMORY_ROLE_NAMES,
    PerceptionMemoryEvaEvaluator,
    PerceptionMemoryRoleBinding,
    REPAIR_ONLY_TRAJECTORY_VARIANTS,
    RESCUE_TRAJECTORY_VARIANTS,
    apply_candidate_gate,
    bind_perception_state,
    build_completeness_messages,
    build_confirmation_controller_messages,
    build_controller_messages,
    build_judge_messages,
    build_cited_judge_messages,
    build_perception_messages,
    build_role_separated_controller_messages,
    evidence_request_addresses_unresolved,
    build_runtime_visual_csv_messages,
    controller_structured_outputs,
    duplicate_interval,
    explicit_time_rescue_request,
    interval_iou,
    messages_have_media,
    parse_completeness,
    parse_controller_action,
    parse_perception_state,
    perception_model_target,
    perception_memory_role_for_stage,
    perception_response_format,
    rescue_frame_request,
    validate_perception_state_observation,
    validate_perception_memory_role_config,
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


def test_evidence_request_must_name_one_unresolved_item() -> None:
    unresolved = ("what happens after the door opens", "which person returns")
    assert evidence_request_addresses_unresolved(
        "Observe what happens after the door opens in the next interval", unresolved
    )
    assert not evidence_request_addresses_unresolved(
        "Inspect more relevant visual evidence", unresolved
    )
    assert evidence_request_addresses_unresolved("anything", ())


def _role_config_payload() -> dict[str, dict[str, str]]:
    return {
        role: {
            "base_url": f"http://127.0.0.1:{8300 + index}/v1",
            "model": f"qwen-{role}",
            "artifact_sha256": f"{index + 1:x}" * 64,
        }
        for index, role in enumerate(PERCEPTION_MEMORY_ROLE_NAMES)
    }


def test_perception_memory_role_config_is_exact_and_normalized() -> None:
    validated = validate_perception_memory_role_config(_role_config_payload())

    assert tuple(validated) == PERCEPTION_MEMORY_ROLE_NAMES
    assert validated["planner"]["base_url"] == "http://127.0.0.1:8300/v1"
    assert validated["answerer"]["artifact_sha256"] == "4" * 64

    missing = _role_config_payload()
    missing.pop("verifier")
    with pytest.raises(ValueError, match="exactly"):
        validate_perception_memory_role_config(missing)

    extra_field = _role_config_payload()
    extra_field["planner"]["unexpected"] = "x"
    with pytest.raises(ValueError, match="exactly"):
        validate_perception_memory_role_config(extra_field)

    invalid_url = _role_config_payload()
    invalid_url["observer"]["base_url"] = "file:///tmp/model"
    with pytest.raises(ValueError, match="HTTP"):
        validate_perception_memory_role_config(invalid_url)

    invalid_hash = _role_config_payload()
    invalid_hash["answerer"]["artifact_sha256"] = "not-a-sha"
    with pytest.raises(ValueError, match="SHA-256"):
        validate_perception_memory_role_config(invalid_hash)


def test_role_config_loader_freezes_file_and_enables_only_media_roles(
    tmp_path: Path,
) -> None:
    path = tmp_path / "roles.json"
    path.write_text(json.dumps(_role_config_payload()), encoding="utf-8")

    config_sha256, bindings, audit = _load_perception_memory_role_config(
        path,
        api_key="test-key",
        timeout=12.0,
        local_media_paths=True,
    )

    assert len(config_sha256) == 64
    assert tuple(bindings) == PERCEPTION_MEMORY_ROLE_NAMES
    assert audit["verifier"]["model"] == "qwen-verifier"
    for role, binding in bindings.items():
        assert binding.client.local_file_urls_as_paths is (role != "planner")
        assert binding.artifact_sha256 == audit[role]["artifact_sha256"]


@pytest.mark.parametrize(
    ("stage", "role"),
    (
        ("controller", "planner"),
        ("confirmation_controller", "planner"),
        ("perception", "observer"),
        ("confirmation_perception", "observer"),
        ("completeness", "verifier"),
        ("evidence_judge", "answerer"),
        ("confirmation_judge", "answerer"),
    ),
)
def test_perception_memory_stage_role_mapping(stage: str, role: str) -> None:
    assert perception_memory_role_for_stage(stage) == role


def test_perception_memory_unknown_stage_fails_closed() -> None:
    with pytest.raises(ValueError, match="unsupported"):
        perception_memory_role_for_stage("mystery")


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


def _two_frame_observation(tmp_path: Path) -> FrameObservation:
    frames = (tmp_path / "first.png", tmp_path / "last.png")
    for frame in frames:
        frame.write_bytes(b"not-decoded-by-unit-test")
    request = FrameRequest(
        start_time=10.0,
        end_time=20.0,
        nframes=2,
        resize=0.75,
        evidence_request="Observe the full action sequence.",
    )
    return FrameObservation(
        request=request,
        resolved_start_time=10.0,
        resolved_end_time=20.0,
        resolved_nframes=2,
        frame_paths=tuple(str(frame) for frame in frames),
        timestamps=(10.123456, 19.987654),
        backend="official_eva_select_frame_fallback",
        cache_hit=False,
        estimated_visual_tokens=256,
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


def _indexed_state_json(
    *,
    interval: tuple[float, float] = (10.0, 20.0),
    fact: str = "The person walks outside.",
    frame_indices: tuple[int, ...] = (0,),
    support_b: bool = True,
    contradict_a: bool = True,
    sufficient: bool = True,
) -> str:
    payload = json.loads(
        _state_json(
            interval=interval,
            fact=fact,
            support_b=support_b,
            contradict_a=contradict_a,
            sufficient=sufficient,
        )
    )
    payload["timestamped_facts"] = [
        {"frame_index": frame_index, "fact": fact} for frame_index in frame_indices
    ]
    return json.dumps(payload)


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


def test_role_separated_planner_history_keeps_only_exact_accepted_pairs() -> None:
    sample = _sample()
    memory = EvidenceMemory(sample.option_letters)
    accepted = (
        (
            {"role": "user", "content": "Exact prior ledger snapshot."},
            '<tool_call>{"tool":"frame_select","arguments":'
            '{"start_time":0,"end_time":10,"nframes":8,"resize":0.75,'
            '"evidence_request":"inspect the first action"}}</tool_call>',
        ),
    )

    messages = build_role_separated_controller_messages(
        sample,
        memory,
        {"duration": 100.0, "width": 1920, "height": 1080},
        accepted,
        feedback="Current retry only.",
    )

    assert [item["role"] for item in messages] == [
        "system",
        "user",
        "assistant",
        "user",
    ]
    assert messages[1] == accepted[0][0]
    assert messages[2]["content"] == accepted[0][1]
    assert "Evidence memory:" in messages[-1]["content"]
    assert "Current retry only." in messages[-1]["content"]
    assert not messages_have_media(messages)


def test_role_separated_visual_csv_is_ledger_free_and_answerer_caps_citations(
    tmp_path: Path,
) -> None:
    sample = _sample()
    memory = EvidenceMemory(sample.option_letters)
    state = parse_perception_state(_state_json(), sample.option_letters)
    assert state is not None
    memory.merge(state)
    frames = []
    for index in range(20):
        path = tmp_path / f"frame-{index}.jpg"
        path.write_bytes(b"frame")
        frames.append((str(path.resolve()), float(index)))

    verifier = build_runtime_visual_csv_messages(sample, frames, prefix_index=0)
    serialized_verifier = json.dumps(verifier, ensure_ascii=False)
    assert "Evidence memory" not in serialized_verifier
    assert "PRIVATE_CANDIDATE_SENTINEL" not in serialized_verifier
    assert serialized_verifier.count('"type": "image_url"') == 20

    answerer = build_cited_judge_messages(
        sample, memory, frames, tuple(range(20))
    )
    serialized_answerer = json.dumps(answerer, ensure_ascii=False)
    assert "Evidence memory:" in serialized_answerer
    assert "PRIVATE_CANDIDATE_SENTINEL" not in serialized_answerer
    assert serialized_answerer.count('"type": "image_url"') == 16


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
    assert (
        "at most one support and one contradiction per option" in messages[0]["content"]
    )
    assert_annotation_free_request({"messages": messages})


def test_perception_response_schema_binds_frame_indices_and_option_shape() -> None:
    response_format = perception_response_format(("A", "B"), 32)
    assert response_format["type"] == "json_schema"
    envelope = response_format["json_schema"]
    assert envelope["strict"] is True
    schema = envelope["schema"]
    assert schema["additionalProperties"] is False
    facts = schema["properties"]["timestamped_facts"]
    assert facts["maxItems"] == 12
    index = facts["items"]["properties"]["frame_index"]
    assert (index["minimum"], index["maximum"]) == (0, 31)
    options = schema["properties"]["option_evidence"]
    assert options["required"] == ["A", "B"]
    assert options["additionalProperties"] is False
    for letter in ("A", "B"):
        assert options["properties"][letter]["required"] == [
            "contradicts",
            "supports",
        ]

    with pytest.raises(ValueError, match="positive frame count"):
        perception_response_format(("A", "B"), 0)


def test_perception_parser_accepts_only_an_exact_json_fence() -> None:
    fenced = f"```json\n{_state_json()}\n```"
    assert parse_perception_state(fenced, ("A", "B")) is not None
    assert parse_perception_state(f"explanation\n{fenced}", ("A", "B")) is None
    assert parse_perception_state(f"{fenced}\nextra", ("A", "B")) is None


def test_frame_index_protocol_binds_first_and_last_exact_timestamps() -> None:
    payload = json.loads(_state_json())
    payload["timestamped_facts"] = [
        {"frame_index": 0, "fact": "The door is closed."},
        {"frame_index": 2, "fact": "The person walks outside."},
    ]
    timestamps = (10.123456, 15.0, 19.987654)

    state, mode = bind_perception_state(
        json.dumps(payload),
        ("A", "B"),
        timestamps,
        allow_timestamp_schema=False,
    )

    assert state is not None
    assert mode == "frame_index"
    assert [fact.time for fact in state.timestamped_facts] == [
        timestamps[0],
        timestamps[-1],
    ]
    target = json.loads(perception_model_target(state, timestamps))
    assert target["timestamped_facts"] == payload["timestamped_facts"]


def test_runtime_frame_index_protocol_explicitly_rejects_legacy_time_schema() -> None:
    legacy = _state_json(fact_time=15.0)

    compatible, compatible_mode = bind_perception_state(legacy, ("A", "B"), (15.0,))
    strict, strict_mode = bind_perception_state(
        legacy,
        ("A", "B"),
        (15.0,),
        allow_timestamp_schema=False,
    )

    assert compatible is not None and compatible_mode == "timestamp"
    assert strict is None and strict_mode == "timestamp"


def test_perception_validation_compacts_long_states_deterministically() -> None:
    payload = json.loads(_state_json())
    payload["interval"] = [0.0, 30.0]
    payload["timestamped_facts"] = [
        {"time": float(index), "fact": f"visible event number {index}"}
        for index in range(20)
    ]
    payload["option_evidence"]["A"]["supports"] = [
        "visible event number 11",
        "second",
    ]
    payload["temporal_changes"] = [
        " ".join([f"change-{index}"] * 30) for index in range(8)
    ]
    payload["unresolved"] = ["缺" * 300 for _index in range(6)]
    payload["next_evidence_needed"] = " ".join(["later"] * 30)

    parsed = parse_perception_state(json.dumps(payload), ("A", "B"))
    assert parsed is not None
    request = FrameRequest(
        start_time=0.0,
        end_time=30.0,
        nframes=20,
        resize=1.0,
        evidence_request="Observe the full interval.",
    )
    observation = FrameObservation(
        request=request,
        resolved_start_time=0.0,
        resolved_end_time=30.0,
        resolved_nframes=20,
        frame_paths=tuple(f"frame-{index}" for index in range(20)),
        timestamps=tuple(float(index) for index in range(20)),
        backend="test",
        cache_hit=False,
        estimated_visual_tokens=0,
        latency_s=0.0,
    )

    state = validate_perception_state_observation(parsed, observation)

    assert len(state.timestamped_facts) == 12
    assert [item.time for item in state.timestamped_facts] == [
        0.0,
        2.0,
        4.0,
        5.0,
        7.0,
        9.0,
        11.0,
        12.0,
        14.0,
        15.0,
        17.0,
        19.0,
    ]
    assert state.option_evidence["A"].supports == ("visible event number 11",)
    assert len(state.temporal_changes) == 4
    assert len(state.unresolved) == 3
    assert all(len(item.split()) == 20 for item in state.temporal_changes)
    assert all(len(item) == 240 for item in state.unresolved)
    assert len(state.next_evidence_needed.split()) == 20

    payload["timestamped_facts"].reverse()
    reparsed = parse_perception_state(json.dumps(payload), ("A", "B"))
    assert reparsed is not None
    assert validate_perception_state_observation(reparsed, observation) == state


def test_perception_validation_checks_every_fact_before_compaction(
    tmp_path: Path,
) -> None:
    payload = json.loads(_state_json())
    payload["timestamped_facts"] = [
        {"time": 15.0, "fact": f"visible fact {index}"} for index in range(20)
    ]
    payload["timestamped_facts"][10]["time"] = 9999.0
    parsed = parse_perception_state(json.dumps(payload), ("A", "B"))
    assert parsed is not None

    with pytest.raises(ValueError, match="outside the resolved perception interval"):
        validate_perception_state_observation(parsed, _observation(tmp_path))


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


def test_memory_keeps_repeated_actions_at_distinct_times_intervals_and_sources() -> (
    None
):
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
    state = parse_perception_state(_state_json(fact_time=15.0004), ("A", "B"))
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
        first_valid_evidence_ids=memory.evidence_ids,
        confirmation_valid_evidence_ids=evidence.evidence_ids,
    )
    assert incomplete.prediction == "A"
    assert incomplete.fallback_to_candidate

    unconfirmed = apply_candidate_gate(
        candidate="A",
        first=evidence,
        confirmation=EvidenceDecision("A", (support,)),
        complete=True,
        memory=memory,
        first_valid_evidence_ids=memory.evidence_ids,
        confirmation_valid_evidence_ids=evidence.evidence_ids,
    )
    assert unconfirmed.prediction == "A"

    changed = apply_candidate_gate(
        candidate="A",
        first=evidence,
        confirmation=evidence,
        complete=True,
        memory=memory,
        first_valid_evidence_ids=memory.evidence_ids,
        confirmation_valid_evidence_ids=evidence.evidence_ids,
    )
    assert changed.prediction == "B"
    assert changed.source == "confirmed_visual_change"


def test_candidate_gate_binds_each_decision_to_its_evidence_stage() -> None:
    state = parse_perception_state(_state_json(), ("A", "B"))
    assert state is not None
    memory = EvidenceMemory(("A", "B"))
    memory.merge(state)
    support = memory.option_ledger["B"].supports[0]
    temporal_change = next(
        item.evidence_id
        for item in memory.event_ledger
        if item.source == "temporal_change"
    )
    first = EvidenceDecision("B", (support, temporal_change))

    unbound_first = apply_candidate_gate(
        candidate="A",
        first=first,
        confirmation=first,
        complete=True,
        memory=memory,
        first_valid_evidence_ids=(support,),
        confirmation_valid_evidence_ids=(support, temporal_change),
    )
    assert unbound_first.prediction == "A"
    assert unbound_first.fallback_to_candidate

    stale_confirmation = apply_candidate_gate(
        candidate="A",
        first=first,
        confirmation=first,
        complete=True,
        memory=memory,
        first_valid_evidence_ids=(support, temporal_change),
        confirmation_valid_evidence_ids=(),
    )
    assert stale_confirmation.prediction == "A"
    assert stale_confirmation.fallback_to_candidate


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
    assert (
        parse_completeness(
            '{"evidence_complete":false,"missing_evidence":["later action"]}'
        ).evidence_complete
        is False
    )  # type: ignore[union-attr]


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
        return replace(self.observation, request=request)


class _FakeFrameTool:
    def __init__(self, observation: FrameObservation) -> None:
        self.session = _FakeSession(observation)

    def open_session(self, video: Path, session_id: str) -> _FakeSession:
        assert video.is_file()
        assert session_id.startswith("pm-")
        return self.session


class _FakeClient:
    def __init__(self, outputs: list[str | tuple[str, str]]) -> None:
        self.outputs = outputs
        self.messages: list[list[dict]] = []
        self.models: list[str] = []
        self.request_kwargs: list[dict[str, object]] = []

    def chat(self, model: str, messages: list[dict], **kwargs: object) -> ChatResult:
        self.models.append(model)
        self.messages.append(messages)
        self.request_kwargs.append(dict(kwargs))
        scripted = self.outputs.pop(0)
        content, finish_reason = (
            scripted if isinstance(scripted, tuple) else (scripted, "stop")
        )
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
            finish_reason=finish_reason,
        )


def test_evaluator_routes_every_stage_to_its_frozen_role(
    tmp_path: Path,
) -> None:
    clients = {
        role: _FakeClient(["{}"] * (2 if role != "verifier" else 1))
        for role in PERCEPTION_MEMORY_ROLE_NAMES
    }
    bindings = {
        role: PerceptionMemoryRoleBinding(
            client=clients[role],
            model=f"model-{role}",
            artifact_sha256=f"{index + 1:x}" * 64,
        )
        for index, role in enumerate(PERCEPTION_MEMORY_ROLE_NAMES)
    }
    evaluator = PerceptionMemoryEvaEvaluator(
        clients["planner"],
        "legacy-model",
        tmp_path,
        tmp_path / "frames",
        role_bindings=bindings,
        role_config_sha256="a" * 64,
    )
    trace: list[dict] = []
    stages = (
        "controller",
        "confirmation_controller",
        "perception",
        "confirmation_perception",
        "completeness",
        "evidence_judge",
        "confirmation_judge",
    )
    for index, stage in enumerate(stages):
        evaluator._chat(
            trace,
            [
                {"role": "system", "content": "Public role instruction."},
                {"role": "user", "content": "Public request."},
            ],
            stage=stage,
            max_tokens=32,
            seed_offset=index,
            json_mode=True,
        )

    assert [item["role_name"] for item in trace] == [
        "planner",
        "planner",
        "observer",
        "observer",
        "verifier",
        "answerer",
        "answerer",
    ]
    for role, client in clients.items():
        assert client.models == [f"model-{role}"] * len(client.models)
    audit = evaluator.static_audit_fields()
    assert audit["role_config_sha256"] == "a" * 64
    assert audit["role_models"] == {
        role: f"model-{role}" for role in PERCEPTION_MEMORY_ROLE_NAMES
    }
    assert audit["role_artifact_sha256s"]["observer"] == "2" * 64


def test_explicit_roles_use_visual_csv_cited_answer_and_accepted_planner_history(
    tmp_path: Path,
) -> None:
    video = tmp_path / "video.mp4"
    video.write_bytes(b"placeholder")
    tool_call = (
        '<tool_call>{"tool":"frame_select","arguments":'
        '{"start_time":10,"end_time":20,"nframes":1,"resize":0.75,'
        '"evidence_request":"check the action"}}</tool_call>'
    )
    observer_payload = json.loads(_indexed_state_json())
    observer_payload.pop("evidence_sufficient")
    observer_payload.pop("next_evidence_needed")
    clients = {
        "planner": _FakeClient([tool_call, '{"action":"stop"}']),
        "observer": _FakeClient([json.dumps(observer_payload)]),
        "verifier": _FakeClient(
            [
                json.dumps(
                    {
                        "answer": "A",
                        "frame_indices": [0],
                        "evidence_complete": True,
                        "missing_evidence": [],
                    }
                )
            ]
        ),
        "answerer": _FakeClient(
            ['{"answer":"A","evidence_ids":["E0001"]}']
        ),
    }
    bindings = {
        role: PerceptionMemoryRoleBinding(
            client=clients[role],
            model=f"model-{role}",
            artifact_sha256=f"{index + 1:x}" * 64,
        )
        for index, role in enumerate(PERCEPTION_MEMORY_ROLE_NAMES)
    }
    evaluator = PerceptionMemoryEvaEvaluator(
        clients["planner"],
        "legacy-model",
        tmp_path,
        tmp_path / "frames",
        frame_tool=_FakeFrameTool(_observation(tmp_path)),  # type: ignore[arg-type]
        max_turns=2,
        role_bindings=bindings,
        role_config_sha256="a" * 64,
    )

    result = evaluator.run(_sample("A"))

    assert result["error"] is None
    assert result["final_prediction"] == "A"
    assert len(result["frame_inventory"]) == 1
    assert result["decisive_frame_indices"] == [0]
    assert result["accepted_planner_actions"] == [tool_call]
    planner_messages = clients["planner"].messages
    assert [item["role"] for item in planner_messages[1]] == [
        "system",
        "user",
        "assistant",
        "user",
    ]
    assert planner_messages[1][1] == planner_messages[0][-1]
    assert planner_messages[1][2]["content"] == tool_call
    assert all(not messages_have_media(messages) for messages in planner_messages)
    verifier_messages = clients["verifier"].messages[0]
    assert messages_have_media(verifier_messages)
    serialized_verifier = json.dumps(verifier_messages, ensure_ascii=False)
    assert "Evidence memory" not in serialized_verifier
    assert "PRIVATE_CANDIDATE_SENTINEL" not in serialized_verifier
    answerer_messages = clients["answerer"].messages[0]
    assert messages_have_media(answerer_messages)
    serialized_answerer = json.dumps(answerer_messages, ensure_ascii=False)
    assert "Evidence memory:" in serialized_answerer
    assert "PRIVATE_CANDIDATE_SENTINEL" not in serialized_answerer
    persisted = result["perception_states"][0]["perception_response"]
    assert "evidence_sufficient" not in persisted
    assert "next_evidence_needed" not in persisted
    assert clients["verifier"].request_kwargs[0]["extra_body"]["tool_choice"] == "none"
    assert clients["answerer"].request_kwargs[0]["extra_body"]["tool_choice"] == "none"


def test_role_binding_or_hash_drift_changes_fingerprint(tmp_path: Path) -> None:
    client = _FakeClient([])

    def build(config_hash: str, answerer_hash: str) -> PerceptionMemoryEvaEvaluator:
        bindings = {
            role: PerceptionMemoryRoleBinding(
                client=client,
                model=f"model-{role}",
                artifact_sha256=(
                    answerer_hash if role == "answerer" else f"{index + 1:x}" * 64
                ),
            )
            for index, role in enumerate(PERCEPTION_MEMORY_ROLE_NAMES)
        }
        return PerceptionMemoryEvaEvaluator(
            client,
            "legacy-model",
            tmp_path,
            tmp_path / "frames",
            role_bindings=bindings,
            role_config_sha256=config_hash,
        )

    baseline = build("a" * 64, "4" * 64)
    assert baseline.run_fingerprint() != build("b" * 64, "4" * 64).run_fingerprint()
    assert baseline.run_fingerprint() != build("a" * 64, "f" * 64).run_fingerprint()

    incomplete = {
        role: PerceptionMemoryRoleBinding(client, role, "1" * 64)
        for role in PERCEPTION_MEMORY_ROLE_NAMES[:-1]
    }
    with pytest.raises(ValueError, match="exactly"):
        PerceptionMemoryEvaEvaluator(
            client,
            "legacy-model",
            tmp_path,
            tmp_path / "frames",
            role_bindings=incomplete,
            role_config_sha256="a" * 64,
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
        count = int(
            request.nframes
            or math.ceil(
                (request.end_time - request.start_time) * float(request.fps or 1.0)
            )
        )
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
    assert set(DURATION_RESCUE_TRAJECTORY_VARIANTS) == {
        "rescue_global32",
        "rescue_global64",
        "rescue_first_half64",
        "rescue_second_half64",
    }
    assert REPAIR_ONLY_TRAJECTORY_VARIANTS == {"rescue_explicit_time"}
    assert RESCUE_TRAJECTORY_VARIANTS == {
        *DURATION_RESCUE_TRAJECTORY_VARIANTS,
        *REPAIR_ONLY_TRAJECTORY_VARIANTS,
    }
    assert rescue_frame_request("base", 100.0) is None


def test_explicit_time_rescue_uses_public_question_seconds_and_proves_mismatch() -> (
    None
):
    request, audit = explicit_time_rescue_request(
        "What happens at 65:02?",
        4000.0,
        [[65.02, 65.29]],
    )

    assert request.to_tool_arguments() == {
        "start_time": 3901.0,
        "end_time": 3903.0,
        "resize": 1.0,
        "fps": 4.0,
        "evidence_request": request.evidence_request,
    }
    assert audit == {
        "mode": "rescue_explicit_time",
        "parsed_time_source": "public_question",
        "parsed_time_range": [3901.0, 3903.0],
        "selected_interval": [3901.0, 3903.0],
        "source_requested_intervals": [[65.02, 65.29]],
        "source_interval_mismatch": True,
        "sampling": {"fps": 4.0},
        "max_frames": 96,
    }


def test_explicit_time_rescue_caps_long_public_interval_at_96_frames() -> None:
    request, audit = explicit_time_rescue_request(
        "What happens from 65:02 to 66:02?",
        4100.0,
        [[65.02, 66.02]],
    )

    assert (request.start_time, request.end_time) == (3901.0, 3963.0)
    assert request.nframes == 96
    assert request.fps is None
    assert audit["sampling"] == {"nframes": 96}


def test_explicit_time_rescue_fails_closed_without_repair_mismatch() -> None:
    with pytest.raises(ValueError, match="source interval mismatch"):
        explicit_time_rescue_request(
            "What happens at 65:02?",
            4000.0,
            [[3901.0, 3903.0]],
        )
    with pytest.raises(ValueError, match="repair source requested intervals"):
        explicit_time_rescue_request("What happens at 65:02?", 4000.0, None)


def test_explicit_time_rescue_is_private_free_official_and_controller_resumes(
    tmp_path: Path,
) -> None:
    video = tmp_path / "video.mp4"
    video.write_bytes(b"placeholder")
    sample = ModelSample(
        dataset="lvbench",
        sample_id="explicit-65m",
        video="video.mp4",
        question="What happens at 65:02?",
        choices={"A": "Sits down", "B": "Walks outside"},
        candidate_answer="B",
    )
    session = _DurationOnlySession(tmp_path, duration=4000.0)
    client = _FakeClient(
        [
            _indexed_state_json(
                interval=(3901.0, 3903.0),
            ),
            '{"action":"stop"}',
            '{"evidence_complete":true,"missing_evidence":[]}',
            '{"answer":"B","evidence_ids":["E0001"]}',
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
        trajectory_schedule_id="repair-explicit-time-v1",
        trajectory_variant_id="rescue_explicit_time",
    )

    result = evaluator.run(
        sample,
        rescue_source_requested_intervals=[[65.02, 65.29]],
    )

    assert result["error"] is None
    assert result["explicit_time_rescue_audit"]["source_interval_mismatch"] is True
    assert len(session.requests) == 1
    request = session.requests[0]
    assert (request.start_time, request.end_time, request.fps) == (
        3901.0,
        3903.0,
        4.0,
    )
    assert request.nframes is None
    controller_trace = [
        item for item in result["request_trace"] if item["stage"] == "controller"
    ]
    assert controller_trace[0]["deterministic_rescue_action"] is True
    assert controller_trace[0]["source_cached_action"] is False
    parsed_action = parse_controller_action(controller_trace[0]["content"])
    assert parsed_action is not None and parsed_action.request == request
    assert "deterministic_rescue_action" not in controller_trace[1]
    assert len(result["tool_steps"][0]["actual_timestamps"]) == 8
    assert result["tool_steps"][0]["actual_timestamps"] == tuple(
        sorted(result["tool_steps"][0]["actual_timestamps"])
    )
    assert all(
        3901.0 <= timestamp <= 3903.0
        for timestamp in result["tool_steps"][0]["actual_timestamps"]
    )
    serialized_requests = json.dumps(client.messages, ensure_ascii=False)
    assert "PRIVATE_CANDIDATE_SENTINEL" not in serialized_requests
    assert "SECRET_ANNOTATION_SENTINEL" not in serialized_requests
    assert "time_range" not in serialized_requests
    assert "clue_intervals" not in serialized_requests
    assert "question_type" not in serialized_requests
    assert messages_have_media(client.messages[0])
    assert not messages_have_media(client.messages[1])


def test_rescue_first_action_is_private_free_then_controller_resumes_and_is_sft_ready(
    tmp_path: Path,
) -> None:
    video = tmp_path / "video.mp4"
    video.write_bytes(b"placeholder")
    session = _DurationOnlySession(tmp_path)
    client = _FakeClient(
        [
            _indexed_state_json(interval=(0.0, 50.0)),
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
    assert (
        first_request.start_time,
        first_request.end_time,
        first_request.nframes,
    ) == (
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
    assert records[0]["metadata"]["process_role"] == "planner"
    assert records[0]["metadata"]["assistant_target_types"] == ["tool", "stop"]
    assert "frame_select" in records[0]["messages"][2]["content"]


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
        **common,
        trajectory_variant_id="base",  # type: ignore[arg-type]
    )
    rescue = PerceptionMemoryEvaEvaluator(
        **common,
        trajectory_variant_id="rescue_global32",  # type: ignore[arg-type]
    )
    assert base.run_fingerprint() != rescue.run_fingerprint()
    with pytest.raises(ValueError, match="unsupported"):
        PerceptionMemoryEvaEvaluator(
            **common,
            trajectory_variant_id="rescue_custom",  # type: ignore[arg-type]
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
    assert _validate_perception_memory_variant("rescue_global64") == ("rescue_global64")
    with pytest.raises(ValueError, match="trajectory-variant-id"):
        _validate_perception_memory_variant("rescue_custom")
    assert _candidate_superset_allowed("perception_memory_eva", True, "rescue_global64")
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
            _indexed_state_json(),
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


def test_runtime_binds_indexed_perception_to_first_and_last_frame(
    tmp_path: Path,
) -> None:
    video = tmp_path / "video.mp4"
    video.write_bytes(b"placeholder")
    tool_call = (
        '<tool_call>{"tool":"frame_select","arguments":'
        '{"start_time":10,"end_time":20,"nframes":2,"resize":0.75,'
        '"evidence_request":"check the action"}}</tool_call>'
    )
    perception = _indexed_state_json(frame_indices=(0, 1))
    client = _FakeClient(
        [
            tool_call,
            perception,
            '{"action":"stop"}',
            '{"evidence_complete":true,"missing_evidence":[]}',
            '{"answer":"A","evidence_ids":["E0001"]}',
        ]
    )
    observation = _two_frame_observation(tmp_path)
    evaluator = PerceptionMemoryEvaEvaluator(
        client,
        "Qwen3.5-9B",
        tmp_path,
        tmp_path / "frames",
        frame_tool=_FakeFrameTool(observation),  # type: ignore[arg-type]
        max_turns=2,
    )

    result = evaluator.run(_sample("A"))

    assert result["error"] is None
    state = result["perception_states"][0]
    assert [
        item["time"] for item in state["perception_response"]["timestamped_facts"]
    ] == [
        observation.timestamps[0],
        observation.timestamps[-1],
    ]
    assert state["timestamp_reference_mode"] == "frame_index"
    assert json.loads(state["perception_model_target"])["timestamped_facts"] == [
        {"fact": "The person walks outside.", "frame_index": 0},
        {"fact": "The person walks outside.", "frame_index": 1},
    ]
    perception_request = next(
        item for item in result["request_trace"] if item["stage"] == "perception"
    )
    assert perception_request["timestamp_reference_mode"] == "frame_index"
    serialized = json.dumps(perception_request["messages"], ensure_ascii=False)
    assert "frame_index must be a zero-based integer" in serialized
    assert "Frame index 0, timestamp 10.123 seconds" in serialized
    assert "Frame index 1, timestamp 19.988 seconds" in serialized


def test_confirmation_runtime_uses_the_same_indexed_perception_protocol(
    tmp_path: Path,
) -> None:
    video = tmp_path / "video.mp4"
    video.write_bytes(b"placeholder")
    tool_call = (
        '<tool_call>{"tool":"frame_select","arguments":'
        '{"start_time":10,"end_time":20,"nframes":2,"resize":0.75,'
        '"evidence_request":"check the action"}}</tool_call>'
    )
    indexed = _indexed_state_json(frame_indices=(0, 1))
    client = _FakeClient(
        [
            tool_call,
            indexed,
            '{"action":"stop"}',
            '{"evidence_complete":true,"missing_evidence":[]}',
            '{"answer":"B","evidence_ids":["E0001"]}',
            tool_call,
            indexed,
            '{"evidence_complete":true,"missing_evidence":[]}',
            '{"answer":"B","evidence_ids":["E0001"]}',
        ]
    )
    observation = _two_frame_observation(tmp_path)
    evaluator = PerceptionMemoryEvaEvaluator(
        client,
        "Qwen3.5-9B",
        tmp_path,
        tmp_path / "frames",
        frame_tool=_FakeFrameTool(observation),  # type: ignore[arg-type]
        max_turns=2,
    )

    result = evaluator.run(_sample("A"))

    assert result["error"] is None
    assert result["final_prediction"] == "B"
    confirmation = result["perception_states"][1]
    assert confirmation["stage"] == "change_confirmation"
    assert confirmation["timestamp_reference_mode"] == "frame_index"
    assert [
        item["time"]
        for item in confirmation["perception_response"]["timestamped_facts"]
    ] == [observation.timestamps[0], observation.timestamps[-1]]
    confirmation_request = next(
        item
        for item in result["request_trace"]
        if item["stage"] == "confirmation_perception"
    )
    assert confirmation_request["timestamp_reference_mode"] == "frame_index"
    assert "frame_index" in json.dumps(
        confirmation_request["messages"], ensure_ascii=False
    )


def test_runtime_compacts_raw_perception_and_remains_sft_exportable(
    tmp_path: Path,
) -> None:
    video = tmp_path / "video.mp4"
    video.write_bytes(b"placeholder")
    tool_call = (
        '<tool_call>{"tool":"frame_select","arguments":'
        '{"start_time":10,"end_time":20,"nframes":1,"resize":0.75,'
        '"evidence_request":"check the action"}}</tool_call>'
    )
    payload = json.loads(_state_json())
    payload["timestamped_facts"] = [
        {"frame_index": 0, "fact": f"visible fact {index}"} for index in range(20)
    ]
    payload["option_evidence"]["B"]["supports"] = [
        "visible fact 19",
        "redundant support",
    ]
    payload["option_evidence"]["A"]["contradicts"] = [
        "visible fact 19",
        "redundant contradiction",
    ]
    client = _FakeClient(
        [
            tool_call,
            json.dumps(payload),
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

    assert result["error"] is None
    assert (
        result["perception_normalization_version"] == PERCEPTION_NORMALIZATION_VERSION
    )
    state = result["perception_states"][0]["perception_response"]
    assert len(state["timestamped_facts"]) == 12
    assert any(item["fact"] == "visible fact 19" for item in state["timestamped_facts"])
    raw = json.loads(
        next(
            item["content"]
            for item in result["request_trace"]
            if item["stage"] == "perception"
        )
    )
    assert len(raw["timestamped_facts"]) == 20

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
    records = build_perception_memory_sft_records(result, include_observer=True)
    memory_target = next(
        record
        for record in records
        if record["metadata"]["process_role"] == "observer"
    )
    assert (
        len(json.loads(memory_target["messages"][-1]["content"])["timestamped_facts"])
        == 12
    )


def test_runtime_accepts_exact_legacy_time_schema_and_canonicalizes_target(
    tmp_path: Path,
) -> None:
    video = tmp_path / "video.mp4"
    video.write_bytes(b"placeholder")
    tool_call = (
        '<tool_call>{"tool":"frame_select","arguments":'
        '{"start_time":10,"end_time":20,"nframes":1,"resize":0.75,'
        '"evidence_request":"check the action"}}</tool_call>'
    )
    legacy = _state_json(fact_time=15.0)
    client = _FakeClient(
        [
            tool_call,
            legacy,
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
    assert result["failure_class"] is None
    assert result["event_ledger"][0]["timestamp"] == 15.0
    assert len(result["perception_states"]) == 1
    perception_requests = [
        item for item in result["request_trace"] if item["stage"] == "perception"
    ]
    assert len(perception_requests) == 1
    assert perception_requests[0]["timestamp_reference_mode"] == "timestamp"
    assert perception_requests[0]["retry_triggered"] is False
    target = json.loads(result["perception_states"][0]["perception_model_target"])
    assert target["timestamped_facts"][0]["frame_index"] == 0
    assert "time" not in target["timestamped_facts"][0]
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
    records = build_perception_memory_sft_records(result, include_observer=True)
    memory_target = next(
        record
        for record in records
        if record["metadata"]["process_role"] == "observer"
    )
    exported = json.loads(memory_target["messages"][-1]["content"])
    assert exported["timestamped_facts"][0]["frame_index"] == 0


def test_runtime_rejects_unbound_legacy_timestamp_after_one_retry(
    tmp_path: Path,
) -> None:
    video = tmp_path / "video.mp4"
    video.write_bytes(b"placeholder")
    tool_call = (
        '<tool_call>{"tool":"frame_select","arguments":'
        '{"start_time":10,"end_time":20,"nframes":1,"resize":0.75,'
        '"evidence_request":"check the action"}}</tool_call>'
    )
    unbound = _state_json(fact_time=15.5)
    client = _FakeClient([tool_call, unbound, unbound])
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
    assert result["error_type"] == "PerceptionModelFailure"
    assert result["failure_class"] == "model_parse_failure"
    assert "invalid_observation_binding" in result["error"]
    assert result["event_ledger"] == []
    attempts = [
        item for item in result["request_trace"] if item["stage"] == "perception"
    ]
    assert len(attempts) == 2
    assert all(item["timestamp_reference_mode"] == "timestamp" for item in attempts)
    assert all("does not match" in item["state_validation_error"] for item in attempts)


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
            _indexed_state_json(),
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


def test_duplicate_controller_action_does_not_consume_an_evidence_step(
    tmp_path: Path,
) -> None:
    video = tmp_path / "video.mp4"
    video.write_bytes(b"placeholder")
    first_call = (
        '<tool_call>{"tool":"frame_select","arguments":'
        '{"start_time":10,"end_time":20,"nframes":1,"resize":0.75,'
        '"evidence_request":"check the first interval"}}</tool_call>'
    )
    second_call = (
        '<tool_call>{"tool":"frame_select","arguments":'
        '{"start_time":30,"end_time":40,"nframes":1,"resize":0.75,'
        '"evidence_request":"check a different interval"}}</tool_call>'
    )
    client = _FakeClient(
        [
            first_call,
            _indexed_state_json(interval=(10.0, 20.0), sufficient=False),
            first_call,
            '{"evidence_complete":false,"missing_evidence":["later action"]}',
            second_call,
            _indexed_state_json(interval=(30.0, 40.0)),
            '{"evidence_complete":true,"missing_evidence":[]}',
            '{"answer":"A","evidence_ids":["E0001"]}',
        ]
    )
    session = _DurationOnlySession(tmp_path)
    evaluator = PerceptionMemoryEvaEvaluator(
        client,
        "Qwen3.5-9B",
        tmp_path,
        tmp_path / "frames",
        frame_tool=_DurationOnlyFrameTool(session),  # type: ignore[arg-type]
        max_turns=2,
    )

    result = evaluator.run(_sample("A"))

    controllers = [
        item for item in result["request_trace"] if item["stage"] == "controller"
    ]
    assert result["error"] is None
    assert result["accepted_evidence_steps"] == 2
    assert result["controller_attempts"] == 3
    assert len(result["perception_states"]) == 2
    assert len(session.requests) == 2
    assert [item["step_index"] for item in controllers] == [0, 1, 1]
    assert [item["action_accepted"] for item in controllers] == [True, False, True]
    assert controllers[1]["action_rejection_reason"] == "duplicate_interval"
    assert controllers[2]["retry_reason"] == "duplicate_interval"
    assert "duplicated already observed evidence" in json.dumps(
        controllers[2]["messages"], ensure_ascii=False
    )


def test_two_duplicate_actions_stop_as_audited_incomplete_fallback(
    tmp_path: Path,
) -> None:
    video = tmp_path / "video.mp4"
    video.write_bytes(b"placeholder")
    first_call = (
        '<tool_call>{"tool":"frame_select","arguments":'
        '{"start_time":10,"end_time":20,"nframes":1,"resize":0.75,'
        '"evidence_request":"check the first interval"}}</tool_call>'
    )
    client = _FakeClient(
        [
            first_call,
            _indexed_state_json(interval=(10.0, 20.0), sufficient=False),
            first_call,
            '{"evidence_complete":false,"missing_evidence":["later action"]}',
            first_call,
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

    assert result["error"] is None
    assert result["failure_class"] is None
    assert result["fallback_to_candidate"] is True
    assert result["stop_reason"] == "evidence_incomplete_no_novel_action"
    assert result["controller_attempt_limit_reached"] is False
    assert result["accepted_evidence_steps"] == 1
    assert result["no_novel_action_rejections"] == 2
    assert result["no_novel_action_reason"] == "duplicate_interval"


def test_two_incomplete_stop_requests_become_audited_safe_fallback(
    tmp_path: Path,
) -> None:
    video = tmp_path / "video.mp4"
    video.write_bytes(b"placeholder")
    client = _FakeClient(
        [
            '{"action":"stop"}',
            '{"evidence_complete":false,"missing_evidence":["visual action"]}',
            '{"action":"stop"}',
            '{"evidence_complete":false,"missing_evidence":["visual action"]}',
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

    assert result["error"] is None
    assert result["failure_class"] is None
    assert result["fallback_to_candidate"] is True
    assert result["stop_reason"] == "evidence_incomplete_no_novel_action"
    assert result["accepted_evidence_steps"] == 0
    assert result["controller_attempt_limit_reached"] is False
    assert result["no_novel_action_rejections"] == 2
    assert result["no_novel_action_reason"] == "incomplete_evidence"


def test_duplicate_stall_runs_blind_completeness_before_answering(
    tmp_path: Path,
) -> None:
    video = tmp_path / "video.mp4"
    video.write_bytes(b"placeholder")
    first_call = (
        '<tool_call>{"tool":"frame_select","arguments":'
        '{"start_time":10,"end_time":20,"nframes":1,"resize":0.75,'
        '"evidence_request":"check the first interval"}}</tool_call>'
    )
    client = _FakeClient(
        [
            first_call,
            _indexed_state_json(interval=(10.0, 20.0), sufficient=True),
            first_call,
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

    assert result["error"] is None
    assert result["stop_reason"] == "evidence_complete_after_duplicate_stall"
    assert result["fallback_to_candidate"] is False
    assert result["perception_states"][0]["evidence_complete"] is True
    completeness = [
        item for item in result["request_trace"] if item["stage"] == "completeness"
    ]
    assert len(completeness) == 1
    assert not messages_have_media(completeness[0]["messages"])


def _controller_reference(messages: list[dict[str, object]], marker: str) -> str:
    content = str(messages[-1]["content"])
    return content.split(marker, 1)[1].split("\nController feedback:", 1)[0]


def test_controller_prompt_has_dynamic_valid_official_reference() -> None:
    memory = EvidenceMemory(("A", "B"))
    messages = build_controller_messages(
        _sample(None), memory, {"duration": 100.0, "width": 1, "height": 1}
    )
    prompt = json.dumps(messages, ensure_ascii=False)
    reference = _controller_reference(
        messages, "Exact currently-valid output reference: "
    )
    action = parse_controller_action(reference)
    assert action is not None and action.request is not None
    assert action.request.start_time == 0.0
    assert action.request.end_time == 100.0
    assert action.request.nframes == 32
    assert action.request.resize == 0.75
    assert reference.startswith("<tool_call>")
    assert reference.endswith("</tool_call>")
    assert '"arguments"' in reference
    assert '"args"' not in reference
    assert "PRIVATE_CANDIDATE_SENTINEL" not in prompt


def test_controller_structured_output_accepts_only_canonical_eva_actions() -> None:
    observe = (
        '<tool_call>{"arguments":{"end_time":100.0,"evidence_request":'
        '"Observe the action.","nframes":32,"resize":0.75,"start_time":0.0},'
        '"tool":"frame_select"}</tool_call>'
    )
    stop = '{"action":"stop"}'
    with_stop = controller_structured_outputs(allow_stop=True)["regex"]
    observe_only = controller_structured_outputs(allow_stop=False)["regex"]
    assert re.fullmatch(with_stop, observe)
    assert re.fullmatch(with_stop, stop)
    assert re.fullmatch(observe_only, observe)
    assert not re.fullmatch(observe_only, stop)
    assert not re.fullmatch(with_stop, observe.replace('"arguments"', '"args"'))
    assert not re.fullmatch(with_stop, observe.replace("0.75", '"fit"'))


def test_controller_reference_uses_public_explicit_time_and_avoids_observed() -> None:
    sample = ModelSample(
        dataset="lvbench",
        sample_id="long-time",
        video="video.mp4",
        question="What happens from 65:02 to 65:08?",
        choices={"A": "First", "B": "Second"},
        candidate_answer="PRIVATE_CANDIDATE_SENTINEL",
    )
    memory = EvidenceMemory(("A", "B"))
    messages = build_controller_messages(
        sample, memory, {"duration": 4000.0, "width": 1, "height": 1}
    )
    reference = _controller_reference(
        messages, "Exact currently-valid output reference: "
    )
    action = parse_controller_action(reference)
    assert action is not None and action.request is not None
    assert (action.request.start_time, action.request.end_time) == (3901.0, 3909.0)

    memory.observed_intervals.append((3901.0, 3909.0))
    retry_messages = build_controller_messages(
        sample, memory, {"duration": 4000.0, "width": 1, "height": 1}
    )
    retry_reference = _controller_reference(
        retry_messages, "Exact currently-valid output reference: "
    )
    retry_action = parse_controller_action(retry_reference)
    assert retry_action is not None and retry_action.request is not None
    assert not duplicate_interval(
        (retry_action.request.start_time, retry_action.request.end_time),
        memory.observed_intervals,
    )
    assert "PRIVATE_CANDIDATE_SENTINEL" not in json.dumps(
        retry_messages, ensure_ascii=False
    )


def test_confirmation_controller_contains_parseable_official_reference() -> None:
    messages = build_confirmation_controller_messages(
        _sample(None),
        EvidenceMemory(("A", "B")),
        {"duration": 100.0, "width": 1, "height": 1},
        ("A", "B"),
    )
    reference = _controller_reference(messages, "Exact official output reference: ")
    action = parse_controller_action(reference)
    assert action is not None and action.request is not None
    assert reference.startswith("<tool_call>")
    assert reference.endswith("</tool_call>")


def test_invalid_controller_interval_is_retried_without_rewriting_it(
    tmp_path: Path,
) -> None:
    video = tmp_path / "video.mp4"
    video.write_bytes(b"placeholder")
    invalid_call = (
        '<tool_call>{"tool":"frame_select","arguments":'
        '{"start_time":10,"end_time":10,"nframes":1,"resize":0.75,'
        '"evidence_request":"check the action"}}</tool_call>'
    )
    valid_call = (
        '<tool_call>{"tool":"frame_select","arguments":'
        '{"start_time":10,"end_time":20,"nframes":1,"resize":0.75,'
        '"evidence_request":"check the action"}}</tool_call>'
    )
    client = _FakeClient(
        [
            invalid_call,
            valid_call,
            _indexed_state_json(),
            '{"evidence_complete":true,"missing_evidence":[]}',
            '{"answer":"A","evidence_ids":["E0001"]}',
        ]
    )
    session = _DurationOnlySession(tmp_path)
    evaluator = PerceptionMemoryEvaEvaluator(
        client,
        "Qwen3.5-9B",
        tmp_path,
        tmp_path / "frames",
        frame_tool=_DurationOnlyFrameTool(session),  # type: ignore[arg-type]
        max_turns=1,
    )

    result = evaluator.run(_sample("A"))

    controllers = [
        item for item in result["request_trace"] if item["stage"] == "controller"
    ]
    assert result["error"] is None
    assert result["accepted_evidence_steps"] == 1
    assert result["controller_attempts"] == 2
    assert [item["step_index"] for item in controllers] == [0, 0]
    assert controllers[0]["content"] == invalid_call
    assert controllers[0]["action_accepted"] is False
    assert controllers[0]["action_rejection_reason"] == "invalid_interval"
    assert controllers[1]["action_accepted"] is True
    assert "0 <= start_time < end_time <= 100.000" in json.dumps(
        controllers[1]["messages"], ensure_ascii=False
    )
    assert len(session.requests) == 1
    assert session.requests[0].to_tool_arguments() == {
        "start_time": 10.0,
        "end_time": 20.0,
        "resize": 0.75,
        "nframes": 1,
        "evidence_request": "check the action",
    }


def test_truncated_controller_action_gets_one_bounded_retry(tmp_path: Path) -> None:
    video = tmp_path / "video.mp4"
    video.write_bytes(b"placeholder")
    valid_call = (
        '<tool_call>{"tool":"frame_select","arguments":'
        '{"start_time":10,"end_time":20,"nframes":1,"resize":0.75,'
        '"evidence_request":"check the action"}}</tool_call>'
    )
    client = _FakeClient(
        [
            ("<tool_call>{", "length"),
            valid_call,
            _indexed_state_json(),
            '{"evidence_complete":true,"missing_evidence":[]}',
            '{"answer":"A","evidence_ids":["E0001"]}',
        ]
    )
    session = _DurationOnlySession(tmp_path)
    evaluator = PerceptionMemoryEvaEvaluator(
        client,
        "Qwen3.5-9B",
        tmp_path,
        tmp_path / "frames",
        frame_tool=_DurationOnlyFrameTool(session),  # type: ignore[arg-type]
        max_turns=1,
    )

    result = evaluator.run(_sample("A"))

    controllers = [
        item for item in result["request_trace"] if item["stage"] == "controller"
    ]
    assert result["error"] is None
    assert result["controller_attempts"] == 2
    assert [item["action_accepted"] for item in controllers] == [False, True]
    assert controllers[0]["action_rejection_reason"] == (
        "controller_response_truncated"
    )
    assert controllers[1]["retry_reason"] == "controller_response_truncated"
    assert [item["attempt_index"] for item in controllers] == [0, 1]
    assert len({item["retry_group_id"] for item in controllers}) == 1
    assert len({item["seed"] for item in controllers}) == 1
    assert len(session.requests) == 1

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
    assert records[0]["metadata"]["process_role"] == "planner"
    assert records[0]["metadata"]["assistant_target_types"] == ["tool", "stop"]


def test_first_controller_failure_on_new_step_does_not_link_previous_step(
    tmp_path: Path,
) -> None:
    video = tmp_path / "video.mp4"
    video.write_bytes(b"placeholder")
    first_call = (
        '<tool_call>{"tool":"frame_select","arguments":'
        '{"start_time":10,"end_time":20,"nframes":1,"resize":0.75,'
        '"evidence_request":"check the first interval"}}</tool_call>'
    )
    second_call = (
        '<tool_call>{"tool":"frame_select","arguments":'
        '{"start_time":30,"end_time":40,"nframes":1,"resize":0.75,'
        '"evidence_request":"check the second interval"}}</tool_call>'
    )
    client = _FakeClient(
        [
            first_call,
            _indexed_state_json(interval=(10.0, 20.0), sufficient=False),
            ("<tool_call>{", "length"),
            second_call,
            _indexed_state_json(interval=(30.0, 40.0)),
            '{"evidence_complete":true,"missing_evidence":[]}',
            '{"answer":"A","evidence_ids":["E0001"]}',
        ]
    )
    session = _DurationOnlySession(tmp_path)
    evaluator = PerceptionMemoryEvaEvaluator(
        client,
        "Qwen3.5-9B",
        tmp_path,
        tmp_path / "frames",
        frame_tool=_DurationOnlyFrameTool(session),  # type: ignore[arg-type]
        max_turns=2,
    )

    result = evaluator.run(_sample("A"))

    controllers = [
        item for item in result["request_trace"] if item["stage"] == "controller"
    ]
    assert result["error"] is None
    assert [item["step_index"] for item in controllers] == [0, 1, 1]
    assert [item["attempt_index"] for item in controllers] == [0, 0, 1]
    assert [item["retry_of_attempt"] for item in controllers] == [None, None, 0]
    assert controllers[0]["retry_group_id"] != controllers[1]["retry_group_id"]
    assert controllers[1]["retry_group_id"] == controllers[2]["retry_group_id"]


def test_controller_attempt_limit_is_an_explicit_policy_failure(tmp_path: Path) -> None:
    video = tmp_path / "video.mp4"
    video.write_bytes(b"placeholder")
    invalid_call = (
        '<tool_call>{"tool":"frame_select","arguments":'
        '{"start_time":10,"end_time":10,"nframes":1,"resize":0.75,'
        '"evidence_request":"check the action"}}</tool_call>'
    )
    client = _FakeClient([invalid_call, invalid_call])
    session = _DurationOnlySession(tmp_path)
    evaluator = PerceptionMemoryEvaEvaluator(
        client,
        "Qwen3.5-9B",
        tmp_path,
        tmp_path / "frames",
        frame_tool=_DurationOnlyFrameTool(session),  # type: ignore[arg-type]
        max_turns=1,
    )

    result = evaluator.run(_sample("A"))

    assert result["final_prediction"] == "A"
    assert result["fallback_to_candidate"] is True
    assert result["controller_attempt_limit_reached"] is True
    assert result["controller_attempts"] == 2
    assert result["accepted_evidence_steps"] == 0
    assert result["failure_class"] == "agent_policy_failure"
    assert result["stop_reason"] == "controller_attempt_limit"
    assert result["controller_attempt_limit_reason"] == "invalid_interval"
    assert result["error_type"] == "ControllerAttemptLimit"
    assert result["evidence_complete"] is False
    assert session.requests == []


def test_malformed_controller_attempt_limit_is_a_parse_failure(tmp_path: Path) -> None:
    video = tmp_path / "video.mp4"
    video.write_bytes(b"placeholder")
    client = _FakeClient(["not a tool call", "still not a tool call"])
    session = _DurationOnlySession(tmp_path)
    evaluator = PerceptionMemoryEvaEvaluator(
        client,
        "Qwen3.5-9B",
        tmp_path,
        tmp_path / "frames",
        frame_tool=_DurationOnlyFrameTool(session),  # type: ignore[arg-type]
        max_turns=1,
    )

    result = evaluator.run(_sample("A"))

    assert result["final_prediction"] == "A"
    assert result["fallback_to_candidate"] is True
    assert result["failure_class"] == "model_parse_failure"
    assert result["model_parse_failure"] is True
    assert result["stop_reason"] == "controller_attempt_limit"
    assert result["controller_attempt_limit_reason"] == "invalid_controller_action"
    assert session.requests == []


@pytest.mark.parametrize(
    ("bad_confirmation", "rejection_reason", "feedback_text"),
    [
        (
            "I cannot choose an interval.",
            "invalid_confirmation_action",
            "official EVA action schema",
        ),
        (
            ('{"arguments":{', "length"),
            "controller_response_truncated",
            "previous response was truncated",
        ),
    ],
    ids=("bare_error", "truncated"),
)
def test_bad_confirmation_retries_then_runs_second_perception_and_judge(
    tmp_path: Path,
    bad_confirmation: str | tuple[str, str],
    rejection_reason: str,
    feedback_text: str,
) -> None:
    video = tmp_path / "video.mp4"
    video.write_bytes(b"placeholder")
    first_call = (
        '<tool_call>{"tool":"frame_select","arguments":'
        '{"start_time":10,"end_time":20,"nframes":1,"resize":0.75,'
        '"evidence_request":"check the action"}}</tool_call>'
    )
    confirmation_call = (
        '<tool_call>{"tool":"frame_select","arguments":'
        '{"start_time":30,"end_time":40,"nframes":1,"resize":0.75,'
        '"evidence_request":"symmetrically distinguish A and B"}}</tool_call>'
    )
    client = _FakeClient(
        [
            first_call,
            _indexed_state_json(),
            '{"evidence_complete":true,"missing_evidence":[]}',
            '{"answer":"B","evidence_ids":["E0001"]}',
            bad_confirmation,
            confirmation_call,
            _indexed_state_json(
                interval=(30.0, 40.0),
                fact="The person walks outside after checking the doorway.",
            ),
            '{"evidence_complete":true,"missing_evidence":[]}',
            '{"answer":"B","evidence_ids":["E0003"]}',
        ]
    )
    session = _DurationOnlySession(tmp_path)
    evaluator = PerceptionMemoryEvaEvaluator(
        client,
        "Qwen3.5-9B",
        tmp_path,
        tmp_path / "frames",
        frame_tool=_DurationOnlyFrameTool(session),  # type: ignore[arg-type]
        max_turns=1,
    )

    result = evaluator.run(_sample("A"))

    confirmations = [
        item
        for item in result["request_trace"]
        if item["stage"] == "confirmation_controller"
    ]
    assert result["error"] is None
    assert result["final_prediction"] == "B"
    assert result["decision_source"] == "confirmed_visual_change"
    assert len(confirmations) == 2
    assert [item["action_accepted"] for item in confirmations] == [False, True]
    assert [item["attempt_index"] for item in confirmations] == [0, 1]
    assert len({item["retry_group_id"] for item in confirmations}) == 1
    assert len({item["seed"] for item in confirmations}) == 1
    assert confirmations[0]["action_rejection_reason"] == rejection_reason
    assert confirmations[1]["retry_reason"] == rejection_reason
    retry_prompt = json.dumps(confirmations[1]["messages"], ensure_ascii=False)
    assert feedback_text in retry_prompt
    assert "Direct" not in retry_prompt
    assert "candidate" not in retry_prompt.lower()
    assert len(result["perception_states"]) == 2
    assert result["perception_states"][1]["stage"] == "change_confirmation"
    assert len(result["judge_answers"]) == 2
    assert [request.start_time for request in session.requests] == [10.0, 30.0]
