from __future__ import annotations

import json
from pathlib import Path

import pytest

from flashvid_eval.client import ChatResult
from flashvid_eval.qwen_evaluation import (
    QwenBaselineConfig,
    QwenBaselineRunner,
    _visual_tokens,
    direct_sampling_spec,
    evaluate_qwen_runner,
)
from flashvid_eval.qwen_protocol import NO_THINK_PROTOCOL, THINK_PROTOCOL
from flashvid_eval.schemas import ModelSample, Sample


class FakeClient:
    def __init__(self, results: list[ChatResult]) -> None:
        self.results = list(results)
        self.calls: list[dict] = []

    def chat(self, model, messages, **kwargs):
        self.calls.append({"model": model, "messages": messages, **kwargs})
        return self.results.pop(0)


def _sample() -> Sample:
    return Sample(
        dataset="demo",
        sample_id="s1",
        video="v.mp4",
        question="What happens?",
        choices={"A": "alpha", "B": "beta", "C": "gamma", "D": "delta"},
        answer="B",
        metadata={"time_range": "SECRET", "question_type": "SECRET_TYPE"},
    )


def test_text_baseline_is_annotation_free_and_strict_json(tmp_path: Path) -> None:
    client = FakeClient([ChatResult('{"answer":"B"}', {"total_tokens": 9}, {}, 0.1)])
    runner = QwenBaselineRunner(
        client,
        "Qwen",
        tmp_path,
        QwenBaselineConfig("question_choices", NO_THINK_PROTOCOL),
    )
    result = runner.run(ModelSample.from_sample(_sample(), None))
    serialized = json.dumps(client.calls[0], ensure_ascii=False)
    assert "SECRET" not in serialized
    assert "video_url" not in serialized
    assert result["prediction"] == "B"
    assert result["visual_tokens"] == 0
    assert result["annotation_leak_check"] == "passed"


def test_choices_only_runner_sends_no_question_or_video(tmp_path: Path) -> None:
    client = FakeClient([ChatResult('{"answer":"A"}', {}, {}, 0.0)])
    runner = QwenBaselineRunner(
        client,
        "Qwen",
        tmp_path,
        QwenBaselineConfig("choices_only", NO_THINK_PROTOCOL),
    )
    result = runner.run(ModelSample.from_sample(_sample(), None))
    serialized = json.dumps(client.calls[0], ensure_ascii=False)
    assert "What happens?" not in serialized
    assert "video_url" not in serialized
    assert result["prediction"] == "A"
    assert result["visual_tokens"] == 0


def test_permuted_choices_runner_maps_displayed_answer_back(tmp_path: Path) -> None:
    client = FakeClient([ChatResult('{"answer":"C"}', {}, {}, 0.0)])
    runner = QwenBaselineRunner(
        client,
        "Qwen",
        tmp_path,
        QwenBaselineConfig(
            "permuted_choices",
            NO_THINK_PROTOCOL,
            option_permutation_seed=17,
        ),
    )
    result = runner.run(ModelSample.from_sample(_sample(), None))
    prompt = client.calls[0]["messages"][0]["content"]
    assert "B. gamma" in prompt
    assert "C. beta" in prompt
    assert result["displayed_prediction"] == "C"
    assert result["prediction"] == "B"


def test_thinking_length_retry_records_both_attempts(tmp_path: Path) -> None:
    client = FakeClient(
        [
            ChatResult("", {"completion_tokens": 8192}, {}, 1.0, finish_reason="length"),
            ChatResult('{"answer":"A"}', {"completion_tokens": 2}, {}, 2.0, finish_reason="stop"),
        ]
    )
    runner = QwenBaselineRunner(
        client,
        "Qwen",
        tmp_path,
        QwenBaselineConfig("question_choices", THINK_PROTOCOL),
    )
    result = runner.run(ModelSample.from_sample(_sample(), None))
    assert [call["max_tokens"] for call in client.calls] == [8192, 32768]
    assert all("response_format" not in call for call in client.calls)
    assert result["length_retry_used"] is True
    assert result["latency_s"] == 3
    assert result["completion_tokens"] == 8194
    assert result["total_tokens"] == 8194
    assert result["reasoning_tokens"] is None


def test_retry_usage_is_cumulative_and_nested_multimodal_tokens_are_parsed(
    tmp_path: Path,
    monkeypatch,
) -> None:
    (tmp_path / "v.mp4").write_bytes(b"fake")
    monkeypatch.setattr(
        "flashvid_eval.qwen_evaluation.probe_video",
        lambda path: {"duration": 10.0, "width": 640, "height": 360},
    )
    first_usage = {
        "prompt_tokens": 100,
        "completion_tokens": 8192,
        "total_tokens": 8292,
        "completion_tokens_details": {"reasoning_tokens": 8000},
        "prompt_tokens_details": {"multimodal_tokens": {"video": 500}},
    }
    second_usage = {
        "prompt_tokens": 100,
        "completion_tokens": 2,
        "total_tokens": 102,
        "completion_tokens_details": {"reasoning_tokens": 0},
        "prompt_tokens_details": {"multimodal_tokens": {"video": 400}},
    }
    client = FakeClient(
        [
            ChatResult("", first_usage, {}, 1.0, finish_reason="length"),
            ChatResult('{"answer":"B"}', second_usage, {}, 2.0, finish_reason="stop"),
        ]
    )
    runner = QwenBaselineRunner(
        client,
        "Qwen",
        tmp_path,
        QwenBaselineConfig(
            "direct",
            THINK_PROTOCOL,
            direct_sampling=direct_sampling_spec("uniform32"),
            max_model_len=131_072,
        ),
    )
    result = runner.run(ModelSample.from_sample(_sample(), None))
    assert result["prompt_tokens"] == 200
    assert result["completion_tokens"] == 8194
    assert result["reasoning_tokens"] == 8000
    assert result["visual_tokens"] == 900
    assert result["visual_token_breakdown"] == {"video": 900}
    assert result["total_tokens"] == 8394
    assert result["final_total_tokens"] == 102
    assert result["sampled_frames_source"] == "estimated_from_request"
    assert result["sampled_frames_actual"] is None
    assert client.calls[0]["mm_processor_kwargs"] == {"do_sample_frames": False}
    assert client.calls[0]["media_io_kwargs"] == {
        "video": {
            "num_frames": -1,
            "fps": 3.2,
            "min_frames": 32,
            "max_frames": 32,
        }
    }
    assert _visual_tokens(first_usage) == 500


def test_retry_respects_context_headroom(tmp_path: Path) -> None:
    client = FakeClient(
        [
            ChatResult(
                "",
                {"prompt_tokens": 123_000, "completion_tokens": 8192},
                {},
                1.0,
                finish_reason="length",
            )
        ]
    )
    runner = QwenBaselineRunner(
        client,
        "Qwen",
        tmp_path,
        QwenBaselineConfig(
            "question_choices",
            THINK_PROTOCOL,
            max_model_len=131_072,
        ),
    )
    result = runner.run(ModelSample.from_sample(_sample(), None))
    assert len(client.calls) == 1
    assert result["length_retry_blocked_by_headroom"] is True
    assert result["parse_error"] == "strict_json_answer_missing"
    assert result["model_parse_failure"] is True


def test_generation_seed_is_stable_per_sample_and_in_fingerprint(tmp_path: Path) -> None:
    sample_two = Sample(
        dataset="demo",
        sample_id="s2",
        video="v2.mp4",
        question="What happens?",
        choices={"A": "alpha", "B": "beta"},
        answer="A",
    )
    client = FakeClient(
        [
            ChatResult('{"answer":"B"}', {}, {}, 0.0),
            ChatResult('{"answer":"A"}', {}, {}, 0.0),
            ChatResult('{"answer":"B"}', {}, {}, 0.0),
        ]
    )
    config = QwenBaselineConfig(
        "question_choices",
        NO_THINK_PROTOCOL,
        generation_seed=73,
        run_context={"manifest_sha256": "a" * 64},
    )
    runner = QwenBaselineRunner(client, "Qwen", tmp_path, config)
    first = runner.run(ModelSample.from_sample(_sample(), None))
    second = runner.run(ModelSample.from_sample(sample_two, None))
    repeated = runner.run(ModelSample.from_sample(_sample(), None))
    assert first["generation_seed"] == repeated["generation_seed"]
    assert first["generation_seed"] != second["generation_seed"]
    assert [call["seed"] for call in client.calls] == [
        first["generation_seed"],
        second["generation_seed"],
        repeated["generation_seed"],
    ]
    changed = QwenBaselineRunner(
        FakeClient([]),
        "Qwen",
        tmp_path,
        QwenBaselineConfig(
            "question_choices",
            NO_THINK_PROTOCOL,
            generation_seed=74,
            run_context={"manifest_sha256": "a" * 64},
        ),
    )
    assert runner.run_fingerprint() != changed.run_fingerprint()
    changed_artifact = QwenBaselineRunner(
        FakeClient([]),
        "Qwen",
        tmp_path,
        QwenBaselineConfig(
            "question_choices",
            NO_THINK_PROTOCOL,
            generation_seed=73,
            run_context={
                "manifest_sha256": "a" * 64,
                "model_artifact_sha256": "b" * 64,
            },
        ),
    )
    assert runner.run_fingerprint() != changed_artifact.run_fingerprint()
    assert first["run_context"] == {"manifest_sha256": "a" * 64}
    assert first["content"] == '{"answer":"B"}'


def test_baseline_fingerprint_uses_v3_sampling_schema(
    tmp_path: Path,
    monkeypatch,
) -> None:
    captured: dict = {}

    def capture(payload):
        captured.update(payload)
        return "fingerprint"

    monkeypatch.setattr("flashvid_eval.qwen_evaluation._canonical_hash", capture)
    runner = QwenBaselineRunner(
        FakeClient([]),
        "Qwen",
        tmp_path,
        QwenBaselineConfig("question_choices", NO_THINK_PROTOCOL),
    )
    assert runner.run_fingerprint() == "fingerprint"
    assert captured["runner"] == "qwen_baseline_v3"


def test_evaluation_joins_private_answer_only_after_runner_returns(tmp_path: Path) -> None:
    class Runner:
        def run_fingerprint(self):
            return "fingerprint"

        def run(self, sample):
            assert isinstance(sample, ModelSample)
            assert not hasattr(sample, "answer")
            return {
                "prediction": "B",
                "annotation_leak_check": "passed",
                "run_fingerprint": "fingerprint",
            }

    summary = evaluate_qwen_runner(
        [_sample()],
        Runner(),
        "unit",
        tmp_path,
        concurrency=2,
    )
    assert summary["correct"] == 1
    row = json.loads((tmp_path / "demo_unit.jsonl").read_text(encoding="utf-8"))
    assert row["answer"] == "B"
    assert row["candidate_rerun"] == 0
    assert row["sample_fingerprint"]


def test_deferred_scoring_never_serializes_private_labels(tmp_path: Path) -> None:
    class Runner:
        def run_fingerprint(self):
            return "trajectory-fingerprint"

        def run(self, sample):
            assert isinstance(sample, ModelSample)
            assert not hasattr(sample, "answer")
            return {
                "prediction": "B",
                "annotation_leak_check": "passed",
            }

    summary = evaluate_qwen_runner(
        [_sample()],
        Runner(),
        "trajectory",
        tmp_path,
        defer_scoring=True,
    )
    row = json.loads(
        (tmp_path / "demo_trajectory.jsonl").read_text(encoding="utf-8")
    )
    assert "answer" not in row
    assert "correct" not in row
    assert row["scoring_deferred"] is True
    assert summary["scoring_deferred"] is True
    assert summary["correct"] is None
    assert summary["accuracy"] is None
    assert summary["common_valid"]["denominator"] == 1
    assert summary["common_valid"]["correct"] is None


def test_deferred_scoring_rejects_private_runner_fields_and_resume_rows(
    tmp_path: Path,
) -> None:
    class LeakyRunner:
        def run_fingerprint(self):
            return "leaky"

        def run(self, sample):
            return {"prediction": "B", "answer": "PRIVATE"}

    summary = evaluate_qwen_runner(
        [_sample()],
        LeakyRunner(),
        "leaky",
        tmp_path,
        defer_scoring=True,
    )
    row = json.loads((tmp_path / "demo_leaky.jsonl").read_text(encoding="utf-8"))
    assert "answer" not in row
    assert row["failure_class"] == "annotation_leak"
    assert summary["failure_class_counts"]["annotation_leak"] == 1

    row["correct"] = True
    (tmp_path / "demo_leaky.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")
    with pytest.raises(Exception, match="deferred-scoring key"):
        evaluate_qwen_runner(
            [_sample()],
            LeakyRunner(),
            "leaky",
            tmp_path,
            defer_scoring=True,
            resume=True,
        )


def test_strict_parse_failure_and_summary_denominators(tmp_path: Path) -> None:
    client = FakeClient([ChatResult("Answer: B", {}, {}, 0.0)])
    runner = QwenBaselineRunner(
        client,
        "Qwen",
        tmp_path,
        QwenBaselineConfig("question_choices", NO_THINK_PROTOCOL),
    )
    summary = evaluate_qwen_runner([_sample()], runner, "parse", tmp_path)
    row = json.loads((tmp_path / "demo_parse.jsonl").read_text(encoding="utf-8"))
    assert row["model_parse_failure"] is True
    assert row["failure_class"] == "model_parse_failure"
    assert row["reasoning_tokens"] is None
    assert summary["nominal"]["denominator"] == 1
    assert summary["accessible"]["denominator"] == 1
    assert summary["common_valid"]["denominator"] == 0
    assert summary["failure_class_counts"]["model_parse_failure"] == 1


def test_resume_retry_replaces_stale_or_duplicate_rows_atomically(tmp_path: Path) -> None:
    class Runner:
        calls = 0

        def run_fingerprint(self):
            return "fingerprint"

        def run(self, sample):
            self.calls += 1
            return {
                "prediction": "B",
                "annotation_leak_check": "passed",
                "run_fingerprint": "fingerprint",
            }

    output = tmp_path / "demo_retry.jsonl"
    stale = {
        "sample_id": "s1",
        "run_fingerprint": "fingerprint",
        "error": "RuntimeError: interrupted",
    }
    output.write_text(
        json.dumps(stale) + "\n" + json.dumps(stale) + "\n",
        encoding="utf-8",
    )
    runner = Runner()
    evaluate_qwen_runner(
        [_sample()],
        runner,
        "retry",
        tmp_path,
        resume=True,
        retry_errors=True,
    )
    rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert runner.calls == 1
    assert len(rows) == 1
    assert rows[0]["prediction"] == "B"


def test_resume_discards_only_an_interrupted_final_json_row(tmp_path: Path) -> None:
    class Runner:
        def run_fingerprint(self):
            return "fingerprint"

        def run(self, sample):
            return {
                "prediction": "B",
                "annotation_leak_check": "passed",
                "run_fingerprint": "fingerprint",
            }

    output = tmp_path / "demo_partial.jsonl"
    output.write_text('{"sample_id":"s1"', encoding="utf-8")
    evaluate_qwen_runner(
        [_sample()], Runner(), "partial", tmp_path, resume=True
    )
    rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 1
    assert rows[0]["prediction"] == "B"


def test_unavailable_mismatch_is_control_unavailable_not_model_error(
    tmp_path: Path,
) -> None:
    runner = QwenBaselineRunner(
        FakeClient([]),
        "Qwen",
        tmp_path,
        QwenBaselineConfig(
            "mismatched_video",
            NO_THINK_PROTOCOL,
            direct_sampling=direct_sampling_spec("uniform32"),
            mismatched_videos={},
        ),
    )
    summary = evaluate_qwen_runner([_sample()], runner, "mismatch", tmp_path)
    row = json.loads((tmp_path / "demo_mismatch.jsonl").read_text(encoding="utf-8"))
    assert row["control_unavailable"] is True
    assert row["failure_class"] == "control_unavailable"
    assert row["model_parse_failure"] is False
    assert summary["accessible"]["denominator"] == 0
    assert summary["failure_class_counts"]["control_unavailable"] == 1


def test_runtime_annotation_guard_blocks_api_call(tmp_path: Path) -> None:
    sample = Sample(
        dataset="demo",
        sample_id="leak",
        video="v.mp4",
        question="runtime-secret-42",
        choices={"A": "alpha", "B": "beta"},
        answer="A",
    )
    client = FakeClient([ChatResult('{"answer":"A"}', {}, {}, 0.0)])
    runner = QwenBaselineRunner(
        client,
        "Qwen",
        tmp_path,
        QwenBaselineConfig(
            "question_choices",
            NO_THINK_PROTOCOL,
            annotation_leak_sentinels=("runtime-secret-42",),
        ),
    )
    summary = evaluate_qwen_runner([sample], runner, "leak", tmp_path)
    row = json.loads((tmp_path / "demo_leak.jsonl").read_text(encoding="utf-8"))
    assert client.calls == []
    assert row["annotation_leak_check"] == "failed"
    assert row["failure_class"] == "annotation_leak"
    assert summary["failure_class_counts"]["annotation_leak"] == 1


def test_baseline_config_rejects_silent_sampling_fallback() -> None:
    with pytest.raises(ValueError, match="Direct sampling"):
        QwenBaselineConfig("direct", NO_THINK_PROTOCOL)
