from __future__ import annotations

import json
from pathlib import Path

from flashvid_eval.answers import extract_answer_letter, extract_strict_answer_letter
from flashvid_eval.datasets import VideoIndex, load_samples
from flashvid_eval.media import estimate_visual_tokens
from flashvid_eval.runner import (
    Evaluator,
    _agent_system_prompt,
    _final_answer_prompt,
    _change_gate_reason,
    _fit_tool_budget,
    _initial_tool_call,
    _hybrid_change_confirmation_prompt,
    _hybrid_v2_confirmation_tool_prompt,
    _hybrid_v3_confirmation_tool_prompt,
    _merge_hybrid_result,
    _minimum_tool_rounds,
    _parse_tool_calls,
    _ranges_overlap,
    _ranges_redundant,
    available_samples,
    evaluate,
    format_question,
    parse_question_time_range,
    _requires_change_confirmation,
    _question_route,
    select_manifest,
)
from flashvid_eval.schemas import ModelSample, Sample


def test_answer_parser_prefers_explicit_marker() -> None:
    assert extract_answer_letter("The visual evidence suggests (B). Answer: C", ["A", "B", "C"]) == "C"
    assert extract_answer_letter("I choose (H)", list("ABCDEFGH")) == "H"
    assert extract_answer_letter("No usable answer", ["A", "B"]) is None


def test_strict_answer_parser_rejects_explanatory_letters() -> None:
    assert extract_strict_answer_letter("A. The words written on the screen", ["A", "B"]) is None
    assert extract_strict_answer_letter("The evidence points to B\nAnswer: C", ["A", "B", "C"]) == "C"
    assert extract_strict_answer_letter("B", ["A", "B", "C"]) == "B"
    assert extract_strict_answer_letter("Answer is C", ["A", "B", "C"]) is None


def test_dataset_adapters(tmp_path: Path) -> None:
    cg = tmp_path / "cg.json"
    cg.write_text(
        json.dumps([{"qid": 7, "video_uid": "v1", "question": "What?", "choices": ["x", "y", "z"], "right_answer": "B", "clue_intervals": [[1, 2]]}]),
        encoding="utf-8",
    )
    lsd = tmp_path / "lsd.json"
    lsd.write_text(
        json.dumps([{"video_id": "v2", "question": "Why?", "options": {"A": "a", "B": "b", "C": "c", "D": "d"}, "correct_answer": "D", "time_range": {"start": "0", "end": "1"}}]),
        encoding="utf-8",
    )
    lv = tmp_path / "lv.jsonl"
    lv.write_text(
        json.dumps({"id": "v3-1", "videos": ["v3.mp4"], "prompt": [{"role": "user", "content": "What?\n(A) x\n(B) y\n(C) z"}], "reward_model": {"ground_truth": "A"}}) + "\n",
        encoding="utf-8",
    )
    assert load_samples("cgbench", cg)[0].answer == "B"
    assert load_samples("lsdbench", lsd)[0].metadata["time_range"]["end"] == "1"
    assert load_samples("lvbench", lv)[0].choices == {"A": "x", "B": "y", "C": "z"}


def test_lvbench_literal_newline_prompt(tmp_path: Path) -> None:
    path = tmp_path / "lv_literal.jsonl"
    path.write_text(
        json.dumps(
            {
                "id": "literal-newlines",
                "videos": ["v.mp4"],
                "prompt": [
                    {
                        "role": "user",
                        "content": "What year?\\n(A) 1636\\n(B) 1366\\n(C) 1363\\n(D) 1633",
                    }
                ],
                "reward_model": {"ground_truth": "D"},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    sample = load_samples("lvbench", path)[0]
    assert sample.question == "What year?"
    assert sample.choices == {"A": "1636", "B": "1366", "C": "1363", "D": "1633"}
    assert sample.answer == "D"


def test_manifest_is_deterministic(tmp_path: Path) -> None:
    cg = tmp_path / "cg.json"
    cg.write_text(
        json.dumps([
            {"qid": i, "video_uid": f"v{i}", "question": "Q", "choices": ["x", "y"], "right_answer": "A"}
            for i in range(5)
        ]),
        encoding="utf-8",
    )
    samples = load_samples("cgbench", cg)
    assert [s.sample_id for s in select_manifest(samples, 3, 42)] == ["0", "2", "4"]


def test_video_index_resolves_nested_files(tmp_path: Path) -> None:
    nested = tmp_path / "nested"
    nested.mkdir()
    (nested / "clip.mp4").write_bytes(b"")
    assert VideoIndex(tmp_path).resolve("clip.mp4") == (nested / "clip.mp4").resolve()


def test_available_samples_filters_partial_download(tmp_path: Path) -> None:
    (tmp_path / "v2.mp4").write_bytes(b"")
    samples = [
        Sample("x", "1", "v1.mp4", "Q", {"A": "x", "B": "y"}, "A"),
        Sample("x", "2", "v2.mp4", "Q", {"A": "x", "B": "y"}, "B"),
    ]
    assert [sample.sample_id for sample in available_samples(samples, tmp_path)] == ["2"]


def test_agent_time_parser_uses_question_only() -> None:
    assert parse_question_time_range("How is the mood from 44:18-44:21?") == (2658.0, 2661.0)
    assert parse_question_time_range("What happens at 04:40?") == (279.0, 281.0)
    assert parse_question_time_range("When does the first game start? (A) 01:23 (B) 05:15") is None


def test_evidence_model_sample_physically_strips_private_scoring_fields() -> None:
    sample = Sample(
        "lsdbench",
        "sentinel",
        "clip.mp4",
        "What happens?",
        {"A": "first", "B": "second"},
        "B",
        metadata={
            "time_range": "SECRET_TIME_SENTINEL",
            "clue_intervals": "SECRET_CLUE_SENTINEL",
            "question_type": "SECRET_TYPE_SENTINEL",
        },
    )
    model_sample = ModelSample.from_sample(sample, "A")
    serialized = json.dumps(model_sample.__dict__, ensure_ascii=False)
    assert model_sample.candidate_answer == "A"
    assert not hasattr(model_sample, "answer")
    assert not hasattr(model_sample, "metadata")
    assert "SECRET_" not in serialized


def test_frozen_hybrid_reuses_candidate_without_direct_or_metadata(monkeypatch, tmp_path: Path) -> None:
    client = object()
    evaluator = Evaluator.__new__(Evaluator)
    evaluator.client = client
    evaluator.model = "model"
    evaluator.agent_version = "hybrid_v3c"
    seen: dict[str, object] = {}

    def fake_verify(sample: Sample, candidate: str | None) -> dict:
        seen["metadata"] = sample.metadata
        seen["candidate"] = candidate
        return {
            "prediction": candidate,
            "raw_response": f"Answer: {candidate}",
            "usage": {},
            "latency_s": 0.0,
            "turn_count": 1,
        }

    monkeypatch.setattr(evaluator, "_agent_verify", fake_verify)
    monkeypatch.setattr(
        evaluator,
        "direct",
        lambda sample: (_ for _ in ()).throw(AssertionError("Direct must not run")),
    )
    sample = Sample(
        "lsdbench",
        "frozen",
        "v.mp4",
        "What happens?",
        {"A": "one", "B": "two"},
        "B",
        {"time_range": "SECRET"},
    )
    result = evaluator.hybrid_frozen(sample, "B")
    assert result["prediction"] == "B"
    assert result["candidate_rerun"] == 0
    assert result["candidate_raw_response"] == ""
    assert seen == {"metadata": {}, "candidate": "B"}


def test_agent_official_tool_calls_only_and_batch_support() -> None:
    assert _parse_tool_calls(
        '```json\n{"tool":"frame_select","arguments":{"start_time":1,"end_time":4,"nframes":8,"resize":0.5}}\n```'
    ) == []
    official = _parse_tool_calls(
        '<tool_call>{"tool":"frame_select","arguments":{"start_time":2,"end_time":5,"nframes":4,"resize":1.0}}'
        '{"tool":"frame_select","arguments":{"start_time":7,"end_time":9,"nframes":6,"resize":0.5}}</tool_call>'
    )
    assert official == [
        {"start_time": 2.0, "end_time": 5.0, "nframes": 4, "resize": 1.0},
        {"start_time": 7.0, "end_time": 9.0, "nframes": 6, "resize": 0.5},
    ]


def test_agent_budget_reduces_resolution_then_frames() -> None:
    metadata = {"width": 1920, "height": 1080, "duration": 60.0}
    fitted = _fit_tool_budget(
        metadata,
        {"start_time": 0.0, "end_time": 10.0, "nframes": 8, "resize": 1.0},
        4000,
    )
    assert fitted is not None
    call, estimated = fitted
    assert estimated <= 4000
    assert call["resize"] < 1.0
    assert call["nframes"] <= 8
    assert estimate_visual_tokens(metadata, call["nframes"], call["resize"]) == estimated


def test_agent_rejects_overlapping_intervals() -> None:
    assert _ranges_overlap((0.0, 10.0), (9.0, 12.0))
    assert not _ranges_overlap((0.0, 10.0), (10.0, 12.0))
    assert _ranges_redundant((0.0, 10.0), (0.0, 10.0))
    assert not _ranges_redundant((0.0, 100.0), (40.0, 50.0))


def test_agent_prompt_has_no_fixed_early_video_anchor_or_annotations() -> None:
    metadata = {"width": 640, "height": 360, "duration": 180.0}
    initial, route = _initial_tool_call("What happens in the video?", 180.0, "v2a")
    prompt = _agent_system_prompt(metadata, initial, route, "v2a", 4000, 8000)
    assert "end_time\\\":60" not in prompt
    assert "time_range" not in prompt
    assert "clue_intervals" not in prompt
    assert "question_type" not in prompt
    assert "Answer the video multiple-choice question" in format_question(
        Sample("x", "1", "v.mp4", "What?", {"A": "yes", "B": "no"}, "A")
    )


def test_agent_resume_and_variant_output_isolation(tmp_path: Path) -> None:
    class FakeEvaluator:
        calls = 0

        def agent(self, sample: Sample) -> dict[str, object]:
            self.calls += 1
            return {
                "prediction": "A",
                "raw_response": "Answer: A",
                "correct": True,
                "rounds": 2,
                "visual_tokens": 100,
                "usage": {"total_tokens": 200},
                "prompt_tokens": 150,
                "completion_tokens": 50,
                "total_tokens": 200,
                "latency_s": 0.1,
                "fallback_used": False,
                "tool_calls": [],
            }

    sample = Sample("x", "1", "v.mp4", "Q", {"A": "yes", "B": "no"}, "A")
    evaluator = FakeEvaluator()
    evaluate([sample], evaluator, "agent", tmp_path / "v2a", concurrency=1)
    evaluate([sample], evaluator, "agent", tmp_path / "v2a", concurrency=1, resume=True)
    assert evaluator.calls == 1
    evaluate([sample], evaluator, "agent", tmp_path / "v2b", concurrency=1, resume=True)
    assert evaluator.calls == 2
    assert (tmp_path / "v2a" / "x_agent.jsonl").is_file()
    assert (tmp_path / "v2b" / "x_agent.jsonl").is_file()


def test_retry_errors_retries_candidate_fallback_parse_failure(tmp_path: Path) -> None:
    class FakeEvaluator:
        calls = 0

        def run_fingerprint(self) -> str:
            return "retry-test"

        def flashvid_hybrid(self, sample: ModelSample) -> dict[str, object]:
            self.calls += 1
            return {
                "prediction": "A",
                "candidate_answer": sample.candidate_answer,
                "candidate_rerun": 0,
                "failure_stage": None,
                "parse_error": None,
                "trajectory_valid": True,
            }

    sample = Sample("x", "retry-1", "v.mp4", "Q", {"A": "yes", "B": "no"}, "A")
    output_dir = tmp_path / "retry"
    output_dir.mkdir()
    output_path = output_dir / "x_flashvid_hybrid.jsonl"
    output_path.write_text(
        json.dumps(
            {
                "dataset": "x",
                "sample_id": "retry-1",
                "prediction": "A",
                "answer": "A",
                "correct": True,
                "failure_stage": "controller_parse",
                "parse_error": "invalid_or_repeated_tool_call",
                "trajectory_valid": False,
                "run_fingerprint": "retry-test",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    evaluator = FakeEvaluator()

    evaluate(
        [sample],
        evaluator,
        "flashvid_hybrid",
        output_dir,
        resume=True,
        retry_errors=True,
        candidate_answers={"retry-1": "A"},
    )

    assert evaluator.calls == 1
    record = json.loads(output_path.read_text(encoding="utf-8"))
    assert record["failure_stage"] is None
    assert record["parse_error"] is None
    assert record["trajectory_valid"] is True


def _hybrid_sample(answer: str = "C") -> Sample:
    return Sample(
        "lvbench",
        "hybrid-1",
        "clip.mp4",
        "Which option is supported by the observed evidence?",
        {"A": "first", "B": "second", "C": "third", "D": "fourth"},
        answer,
    )


def test_hybrid_prompt_passes_only_candidate_hypothesis() -> None:
    metadata = {"width": 1280, "height": 720, "duration": 90.0}
    initial, route = _initial_tool_call("What happens?", 90.0, "hybrid_v1")
    prompt = _agent_system_prompt(
        metadata,
        initial,
        route,
        "hybrid_v1",
        12000,
        24000,
        candidate_answer="B",
    )
    assert "Direct candidate hypothesis: B" in prompt
    assert "not ground truth" in prompt
    # The candidate verifier receives the letter only, never the candidate call's
    # explanation or any answer/annotation fields.
    assert "private direct rationale" not in prompt
    assert "candidate_raw_response" not in prompt
    assert "time_range" not in prompt
    assert "clue_intervals" not in prompt
    assert "question_type" not in prompt


def test_hybrid_v2_prompt_and_confirmation_helpers_are_conservative() -> None:
    metadata = {"width": 1280, "height": 720, "duration": 90.0}
    initial, route = _initial_tool_call("What happens?", 90.0, "hybrid_v2")
    prompt = _agent_system_prompt(
        metadata,
        initial,
        route,
        "hybrid_v2",
        12000,
        24000,
        candidate_answer="B",
    )
    assert "Treat the Direct candidate as the default answer" in prompt
    assert "Final answers must be one line only: Answer: X" in prompt
    assert "time_range" not in prompt
    confirm = _hybrid_v2_confirmation_tool_prompt("B", "C")
    assert "non-overlapping frame_select interval" in confirm
    assert _change_gate_reason("It seems unclear and possibly implied.") == "uncertain_language"
    assert _change_gate_reason("Answer: C") is None
    sample = Sample(
        "lvbench",
        "s1",
        "clip.mp4",
        "What happens after the second scene?",
        {"A": "a", "B": "b"},
        "A",
        metadata={"question_type": "['event understanding']"},
    )
    assert _requires_change_confirmation(sample, "global_overview") is True
    assert _requires_change_confirmation(sample, "explicit_question_time") is False


def test_hybrid_v3_routes_questions_without_annotation_leakage() -> None:
    ocr_call, ocr_route = _initial_tool_call(
        "What word is written on the sign?",
        120.0,
        "hybrid_v3b",
        {"question_type": "global overview"},
    )
    assert ocr_route == "ocr_detail"
    assert ocr_call["nframes"] == 8
    assert ocr_call["resize"] == 0.85

    # Frozen versions keep their original overview controller.
    _, old_route = _initial_tool_call("What word is written on the sign?", 120.0, "hybrid_v2")
    assert old_route == "global_overview"

    # Dataset annotations must not alter routing or enter the verifier prompt.
    assert _question_route("What is the overall mood?", {"question_type": "OCR"}) == "global_overview"
    local_call, local_route = _initial_tool_call("What happens from 04:40-04:46?", 300.0, "hybrid_v3b")
    assert local_route == "explicit_question_time"
    assert local_call["start_time"] == 279.0
    assert local_call["end_time"] == 287.0


def test_hybrid_v3_prompt_and_confirmation_variants() -> None:
    metadata = {"width": 1280, "height": 720, "duration": 90.0}
    initial, route = _initial_tool_call("What does the person do next?", 90.0, "hybrid_v3d")
    prompt = _agent_system_prompt(
        metadata,
        initial,
        route,
        "hybrid_v3d",
        12000,
        24000,
        candidate_answer="B",
    )
    assert "up to two non-overlapping frame_select intervals" in prompt
    assert "Direct candidate hypothesis: B" in prompt
    assert "time_range" not in prompt
    assert "clue_intervals" not in prompt
    assert "question_type" not in prompt

    confirm = _hybrid_v3_confirmation_tool_prompt("B", "C", allow_two_intervals=True)
    assert "up to two non-overlapping frame_select intervals" in confirm
    final_prompt = _final_answer_prompt("hybrid_v3c", "B")
    assert "Direct candidate is B" in final_prompt
    assert "exactly one line" in final_prompt

    overview_call, overview_route = _initial_tool_call("What is the overall mood?", 90.0, "hybrid_v3e")
    temporal_call, temporal_route = _initial_tool_call("What happens after the person leaves?", 90.0, "hybrid_v3e")
    assert overview_route == "global_overview"
    assert overview_call["nframes"] == 16
    assert overview_call["resize"] == 0.75
    assert temporal_route == "temporal_event"
    assert temporal_call["nframes"] == 12
    assert temporal_call["resize"] == 0.42

    arbitration_call, arbitration_route = _initial_tool_call(
        "What happens after the person leaves?", 90.0, "hybrid_v3f"
    )
    arbitration_prompt = _agent_system_prompt(
        metadata,
        arbitration_call,
        arbitration_route,
        "hybrid_v3f",
        12000,
        24000,
        candidate_answer="B",
    )
    assert "uncertain first change proposal" in arbitration_prompt
    assert "same strict answer again" in arbitration_prompt


def test_hybrid_final_prompt_prefers_candidate_when_evidence_is_ambiguous() -> None:
    prompt = _final_answer_prompt("hybrid_v1", "B")
    assert "Direct candidate is B" in prompt
    assert "if evidence is ambiguous or missing" in prompt
    assert "Answer: X" in prompt

    hybrid_v2_prompt = _final_answer_prompt("hybrid_v2", "B")
    assert "Direct candidate is B and remains the default" in hybrid_v2_prompt
    assert "Output exactly one line: Answer: X" in hybrid_v2_prompt


def test_hybrid_change_confirmation_prompt_requires_visual_contradiction() -> None:
    prompt = _hybrid_change_confirmation_prompt("B", "C")
    assert "Direct candidate B" in prompt
    assert "Answer: C" in prompt
    assert "candidate_raw_response" not in prompt
    assert "time_range" not in prompt


def test_hybrid_without_candidate_accepts_first_valid_answer() -> None:
    assert _minimum_tool_rounds("hybrid_v1", "global_overview", "B") == 2
    assert _minimum_tool_rounds("hybrid_v1", "global_overview", None) == 1
    assert _minimum_tool_rounds("v2d", "global_overview", None) == 2
    assert _minimum_tool_rounds("hybrid_v1", "explicit_question_time", "B") == 1
    assert _minimum_tool_rounds("hybrid_v3a", "ocr_detail", "B") == 1
    assert _minimum_tool_rounds("hybrid_v3b", "ocr_detail", "B") == 2
    assert _minimum_tool_rounds("hybrid_v3d", "global_overview", "B") == 2


def test_hybrid_passes_candidate_letter_not_direct_reasoning_to_verifier() -> None:
    class FakeHybrid:
        verifier_input: str | None = None

        def direct(self, sample: Sample) -> dict[str, object]:
            return {
                "prediction": "B",
                "raw_response": "SECRET DIRECT REASONING",
                "usage": {},
                "latency_s": 0.0,
            }

        def _agent_verify(self, sample: Sample, candidate_answer: str | None) -> dict[str, object]:
            self.verifier_input = candidate_answer
            return {"prediction": "B", "usage": {}, "latency_s": 0.0}

    fake = FakeHybrid()
    result = Evaluator.hybrid(fake, _hybrid_sample(answer="B"))  # type: ignore[arg-type]
    assert fake.verifier_input == "B"
    assert "SECRET DIRECT REASONING" not in str(fake.verifier_input)
    assert result["candidate_raw_response"] == "SECRET DIRECT REASONING"


def test_hybrid_v3g_delegates_by_question_text_and_strips_annotations() -> None:
    class FakeHybridV3G:
        agent_version = "hybrid_v3g"
        observations: list[tuple[str, dict[str, object], str | None]] = []

        def direct(self, sample: Sample) -> dict[str, object]:
            return {"prediction": "B", "raw_response": "Answer: B", "usage": {}, "latency_s": 0.0}

        def _agent_verify(self, sample: Sample, candidate_answer: str | None) -> dict[str, object]:
            self.observations.append((self.agent_version, sample.metadata, candidate_answer))
            return {"prediction": "B", "usage": {}, "latency_s": 0.0, "rounds": 1}

    sample = Sample(
        "lvbench",
        "s-route",
        "clip.mp4",
        "What word is written on the sign?",
        {"A": "one", "B": "two"},
        "B",
        metadata={"question_type": "event understanding", "clue_intervals": [[1, 2]]},
    )
    fake = FakeHybridV3G()
    result = Evaluator.hybrid(fake, sample)  # type: ignore[arg-type]
    assert fake.observations == [("hybrid_v2", {}, "B")]
    assert result["delegated_version"] == "hybrid_v2"
    assert result["prompt_version"] == "hybrid_v3g"
    assert result["question_route"] == "ocr_detail"
    assert result["question_type"] == ["event understanding"]


def test_hybrid_verifier_no_answer_falls_back_to_candidate() -> None:
    sample = _hybrid_sample(answer="B")
    candidate = {
        "prediction": "B",
        "raw_response": "Answer: B (private direct rationale)",
        "usage": {"prompt_tokens": 10, "completion_tokens": 4, "total_tokens": 14},
        "latency_s": 1.25,
    }
    verifier = {
        "prediction": None,
        "error": "no_answer",
        "usage": {"prompt_tokens": 20, "completion_tokens": 6, "total_tokens": 26},
        "latency_s": 2.5,
        "visual_tokens": 1234,
    }
    result = _merge_hybrid_result(sample, candidate, verifier)
    assert result["candidate_answer"] == "B"
    assert result["prediction"] == "B"
    assert result["final_prediction"] == "B"
    assert result["fallback_to_candidate"] is True
    assert result["candidate_changed"] is False
    assert result["correct"] is True
    assert result["usage"] == {"prompt_tokens": 30, "completion_tokens": 10, "total_tokens": 40}
    assert result["latency_s"] == 3.75
    # The raw candidate is retained for audit output, but is not part of the
    # verifier record (the merge function only combines result dictionaries).
    assert result["candidate_raw_response"].endswith("private direct rationale)")
    assert "error" not in result
    assert result["verifier_error"] == "no_answer"


def test_hybrid_verifier_can_change_candidate_and_records_it() -> None:
    sample = _hybrid_sample(answer="C")
    candidate = {
        "prediction": "B",
        "raw_response": "Answer: B",
        "usage": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5},
        "latency_s": 0.5,
    }
    verifier = {
        "prediction": "C",
        "raw_response": "Answer: C",
        "usage": {"prompt_tokens": 7, "completion_tokens": 3, "total_tokens": 10},
        "latency_s": 1.5,
        "tool_calls": [{"start_time": 12.0, "end_time": 16.0}],
    }
    result = _merge_hybrid_result(sample, candidate, verifier)
    assert result["candidate_answer"] == "B"
    assert result["final_prediction"] == "C"
    assert result["prediction"] == "C"
    assert result["candidate_changed"] is True
    assert result["fallback_to_candidate"] is False
    assert result["correct"] is True
    assert result["tool_calls"] == verifier["tool_calls"]


def test_hybrid_v2_merge_records_gate_and_fallback_source() -> None:
    sample = _hybrid_sample(answer="B")
    candidate = {
        "prediction": "B",
        "raw_response": "Answer: B",
        "usage": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5},
        "latency_s": 0.5,
    }
    verifier = {
        "prediction": None,
        "strict_verifier_answer": "C",
        "change_gate_triggered": True,
        "change_rejection_reason": "uncertain_language",
        "change_confirmation_requested": True,
        "change_confirmation_observed": False,
        "candidate_change_reviewed": True,
        "candidate_change_rejected": True,
        "turn_count": 3,
        "question_route": "action_event",
        "question_type": ["event understanding"],
        "usage": {"prompt_tokens": 7, "completion_tokens": 3, "total_tokens": 10},
        "latency_s": 1.5,
    }
    result = _merge_hybrid_result(sample, candidate, verifier)
    assert result["prediction"] == "B"
    assert result["final_prediction"] == "B"
    assert result["fallback_to_candidate"] is True
    assert result["final_decision_source"] == "candidate_gate"
    assert result["strict_verifier_answer"] == "C"
    assert result["change_gate_triggered"] is True
    assert result["change_rejection_reason"] == "uncertain_language"
    assert result["gate_reason"] == "uncertain_language"
    assert result["change_confirmation_requested"] is True
    assert result["candidate_change_reviewed"] is True
    assert result["turn_count"] == 3
    assert result["question_route"] == "action_event"


def test_hybrid_invalid_candidate_does_not_trigger_fallback() -> None:
    sample = _hybrid_sample(answer="C")
    invalid_candidate = {
        "prediction": "Z",
        "raw_response": "Answer: Z",
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        "latency_s": 0.1,
    }
    verifier_without_answer = {"prediction": None, "error": "no_answer", "usage": {}, "latency_s": 0.2}
    result = _merge_hybrid_result(sample, invalid_candidate, verifier_without_answer)
    assert result["candidate_answer"] is None
    assert result["prediction"] is None
    assert result["final_prediction"] is None
    assert result["fallback_to_candidate"] is False
    assert result["candidate_changed"] is False
    assert result["correct"] is False
    assert result["error"] == "no_answer"

    # An invalid candidate must not prevent a valid verifier decision.
    verifier_answer = {"prediction": "C", "usage": {}, "latency_s": 0.3}
    valid_result = _merge_hybrid_result(sample, invalid_candidate, verifier_answer)
    assert valid_result["prediction"] == "C"
    assert valid_result["fallback_to_candidate"] is False
