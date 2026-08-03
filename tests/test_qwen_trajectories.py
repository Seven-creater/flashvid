from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from flashvid_eval.client import ChatResult
from flashvid_eval.qwen_agents import (
    AgentConfig,
    AgentTrace,
    EvidenceEntry,
    FrameRequest,
    FrameTool,
    InferenceProtocol,
)
from flashvid_eval.qwen_trajectories import (
    FixedEvidenceReplayStrategy,
    QwenTrajectoryRunner,
    TrajectoryGenerationConfig,
)
from flashvid_eval.schemas import ModelSample


@dataclass
class FakeStrategy:
    trace: AgentTrace

    strategy_id: str = "fake"

    def run(self, sample: ModelSample) -> AgentTrace:
        assert sample.candidate_answer is None
        return self.trace

    def run_fingerprint(self) -> str:
        return "f" * 64


class FakeClient:
    def __init__(self, answers: list[str]) -> None:
        self.answers = list(answers)
        self.calls: list[dict[str, Any]] = []

    def chat(self, model: str, messages: list[dict[str, Any]], **kwargs: Any) -> ChatResult:
        self.calls.append({"model": model, "messages": messages, **kwargs})
        answer = self.answers.pop(0)
        return ChatResult(
            content=f'{{"answer":"{answer}"}}',
            usage={"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12},
            raw={},
            latency_s=0.1,
            reasoning_content="private reasoning",
            finish_reason="stop",
        )


def _runner(trace: AgentTrace, client: FakeClient) -> QwenTrajectoryRunner:
    return QwenTrajectoryRunner(
        strategy=FakeStrategy(trace),
        client=client,
        model="Qwen3.5-9B",
        protocol=InferenceProtocol(enable_thinking=True, top_p=0.95),
        config=TrajectoryGenerationConfig(
            schedule_id="schedule-00",
            dataset_manifest_sha256="a" * 64,
            train600_manifest_sha256="9" * 64,
            experiment_config_sha256="b" * 64,
            agent_config_sha256="c" * 64,
            model_artifact_sha256="d" * 64,
        ),
    )


def _sample(question: str = "What happens?") -> ModelSample:
    return ModelSample(
        dataset="lvbench",
        sample_id="sample-1",
        video="video.mp4",
        question=question,
        choices={"A": "sits", "B": "opens the door"},
        candidate_answer=None,
    )


def test_trajectory_runner_adds_three_blind_confirmations_and_identity() -> None:
    trace = AgentTrace(
        strategy="fake",
        model="Qwen3.5-9B",
        dataset="lvbench",
        sample_id="sample-1",
        prediction="B",
        final_prediction="B",
        annotation_leak_check="passed",
    )
    trace.evidence_memory.append(
        EvidenceEntry("zoom", (1.0, 2.0), (1.0, 2.0), "The door opens.")
    )
    client = FakeClient(["B", "B", "B"])
    row = _runner(trace, client).run(_sample())
    assert row["trajectory_id"] == "lvbench:sample-1:schedule-00:0"
    assert [item["judge_seed"] for item in row["judge_confirmations"]] == [17, 42, 73]
    assert all(item["prediction"] == "B" for item in row["judge_confirmations"])
    assert row["confirmation_status"] == "passed_engineering"
    assert all(call["chat_template_kwargs"] == {"enable_thinking": True} for call in client.calls)
    serialized = str(client.calls)
    assert "candidate_answer" not in serialized
    assert "ground_truth" not in serialized


def test_trajectory_skips_all_judges_after_base_annotation_leak() -> None:
    trace = AgentTrace(
        strategy="fake",
        model="Qwen3.5-9B",
        dataset="lvbench",
        sample_id="sample-1",
        error="annotation sentinel found",
        error_type="AnnotationLeakError",
        failure_class="annotation_leak",
        annotation_leak_check="failed",
    )
    client = FakeClient(["A"])

    row = _runner(trace, client).run(_sample())

    assert client.calls == []
    assert row["judge_confirmations"] == []
    assert row["confirmation_status"] == "skipped_annotation_leak"
    assert row["failure_class"] == "annotation_leak"


def test_confirmation_annotation_leak_stops_remaining_judges() -> None:
    trace = AgentTrace(
        strategy="fake",
        model="Qwen3.5-9B",
        dataset="lvbench",
        sample_id="sample-1",
        prediction="B",
        final_prediction="B",
        annotation_leak_check="passed",
    )
    client = FakeClient(["B", "B", "B"])

    row = _runner(trace, client).run(_sample("SECRET_ANNOTATION_SENTINEL"))

    assert client.calls == []
    assert len(row["judge_confirmations"]) == 1
    assert row["judge_confirmations"][0]["failure_class"] == "annotation_leak"
    assert row["annotation_leak_check"] == "failed"
    assert row["failure_class"] == "annotation_leak"


def test_confirmation_parse_failure_is_visible_on_trajectory_row() -> None:
    trace = AgentTrace(
        strategy="fake",
        model="Qwen3.5-9B",
        dataset="lvbench",
        sample_id="sample-1",
        prediction="B",
        final_prediction="B",
        annotation_leak_check="passed",
    )
    client = FakeClient(["not-json", "B", "B"])

    row = _runner(trace, client).run(_sample())

    assert row["confirmation_status"] == "failed"
    assert row["confirmation_failure_counts"]["model_parse_failure"] == 1
    assert row["failure_class"] == "model_parse_failure"
    assert row["model_parse_failure"] is True
    assert row["parse_error"] == "trajectory_confirmation_strict_json_answer_missing"


def test_trajectory_config_rejects_duplicate_judge_seeds() -> None:
    try:
        TrajectoryGenerationConfig(
            schedule_id="schedule",
            dataset_manifest_sha256="a" * 64,
            train600_manifest_sha256="9" * 64,
            experiment_config_sha256="b" * 64,
            agent_config_sha256="c" * 64,
            model_artifact_sha256="d" * 64,
            judge_seeds=(17, 17),
        )
    except ValueError as error:
        assert "unique" in str(error)
    else:
        raise AssertionError("duplicate judge seeds were accepted")


def test_fixed_evidence_replay_changes_only_the_frozen_frame_plan(
    tmp_path,
) -> None:
    video_root = tmp_path / "videos"
    video_root.mkdir()
    (video_root / "video.mp4").write_bytes(b"video")
    selector_calls: list[tuple[float, float, int, float]] = []

    def selector(video, start, end, nframes, resize, output_dir):
        assert video.name == "video.mp4"
        output_dir.mkdir(parents=True, exist_ok=True)
        paths = []
        timestamps = []
        for index in range(nframes):
            path = output_dir / f"{index}.jpg"
            path.write_bytes(b"frame")
            paths.append(path)
            timestamps.append(start + (index + 0.5) * (end - start) / nframes)
        selector_calls.append((start, end, nframes, resize))
        return paths, timestamps, "fake-decord"

    tool = FrameTool(
        tmp_path / "frames",
        max_frames_per_call=16,
        selector=selector,
        probe=lambda _path: {"duration": 30.0, "width": 320, "height": 240},
    )
    client = FakeClient(["A", "B"])
    strategy = FixedEvidenceReplayStrategy(
        client=client,
        model="Qwen3.5-9B",
        video_root=video_root,
        frame_tool=tool,
        config=AgentConfig(
            strategy="counterfactual_fixed_evidence",
            max_frames_per_call=16,
        ),
        protocol=InferenceProtocol(),
        planned_calls=(
            FrameRequest(10.0, 20.0, nframes=2, resize=0.5),
        ),
        source_fingerprint="e" * 64,
    )
    sample = ModelSample(
        dataset="lvbench",
        sample_id="sample-1",
        video="video.mp4",
        question="What happens?",
        choices={"A": "sits", "B": "opens the door"},
        candidate_answer=None,
    )
    trace = strategy.run(sample)
    assert trace.final_prediction == "B"
    assert selector_calls == [(10.0, 20.0, 2, 0.5)]
    assert len(trace.tool_steps) == 1
    assert [call["seed"] for call in client.calls] == [42, 43]
    assert len(strategy.run_fingerprint()) == 64
