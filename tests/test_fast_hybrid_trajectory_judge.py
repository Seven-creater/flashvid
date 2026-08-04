from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from typing import Any

import pytest

from flashvid_eval.client import ChatResult
from flashvid_eval.fast_hybrid_trajectory_judge import (
    FastHybridTrajectoryJudge,
    TrajectoryJudgeConfig,
    _question_and_choices_prompt,
    bind_trajectory_jobs,
)
from flashvid_eval.privacy import AnnotationLeakError


_SCRIPT_PATH = (
    Path(__file__).resolve().parents[1] / "scripts" / "judge_fast_hybrid_trajectories.py"
)
_SCRIPT_SPEC = importlib.util.spec_from_file_location(
    "judge_fast_hybrid_trajectories", _SCRIPT_PATH
)
assert _SCRIPT_SPEC is not None and _SCRIPT_SPEC.loader is not None
judge_script = importlib.util.module_from_spec(_SCRIPT_SPEC)
_SCRIPT_SPEC.loader.exec_module(judge_script)


SHA_A = "a" * 64
SHA_B = "b" * 64
SHA_C = "c" * 64


class FakeClient:
    def __init__(self, answers: dict[int, str] | None = None) -> None:
        self.answers = answers or {17: "B", 42: "B", 73: "B"}
        self.calls: list[dict[str, Any]] = []

    def chat(
        self,
        model: str,
        messages: list[dict[str, Any]],
        max_tokens: int = 32,
        **kwargs: Any,
    ) -> ChatResult:
        seed = int(kwargs["seed"])
        self.calls.append(
            {
                "model": model,
                "messages": messages,
                "max_tokens": max_tokens,
                **kwargs,
            }
        )
        content = self.answers[seed]
        return ChatResult(
            content=content,
            usage={
                "prompt_tokens": 100,
                "completion_tokens": 5,
                "total_tokens": 105,
                "prompt_tokens_details": {
                    "multimodal_tokens": {"image": 77}
                },
            },
            raw={},
            latency_s=0.25,
            reasoning_content="hidden",
            finish_reason="stop",
        )


def _spec() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "phase": "base",
        "dataset": "lvbench",
        "sample_id": "sample-1",
        "schedule_id": "budget_006000_seed_17",
        "variant_id": "base",
        "family_id": "budget_006000_seed_17",
        "replica_id": "0",
        "trajectory_id": "lvbench:sample-1:budget_006000_seed_17:0",
        "planner_seed": 17,
        "max_total_visual_tokens": 6000,
        "max_call_visual_tokens": 6000,
        "max_turns": 6,
        "required_judge_seeds": [17, 42, 73],
        "manifest_sha256": SHA_A,
        "train600_manifest_sha256": SHA_A,
        "dataset_manifest_sha256": SHA_B,
        "config_sha256": SHA_C,
        "controller_fingerprint": SHA_A,
        "run_spec_fingerprint": SHA_B,
    }


def _trajectory(frame_paths: list[Path]) -> dict[str, Any]:
    question = (
        "Video Length: 694 seconds. Original video resolution: 480p. "
        "Question: What colour is the car?\nA: red\nB: blue\nC: green"
    )
    return {
        "dataset": "lvbench",
        "sample_id": "sample-1",
        "scoring_deferred": True,
        "candidate_rerun": 0,
        "annotation_leak_check": "passed",
        "candidate_answer": "A",
        "final_prediction": "B",
        "trajectory_schedule_id": "budget_006000_seed_17",
        "trajectory_variant_id": "base",
        "trajectory_replica_id": 0,
        "generation_seed": 17,
        "manifest_sha256": SHA_A,
        "train600_manifest_sha256": SHA_A,
        "experiment_config_sha256": SHA_C,
        "request_trace": [
            {
                "stage": "verification",
                "messages": [
                    {
                        "role": "system",
                        "content": (
                            "Direct candidate: A. PRIVATE_CANDIDATE_EXPLANATION"
                        ),
                    },
                    {"role": "user", "content": question},
                ],
                "content": "Answer: B",
            }
        ],
        "conversation_traces": [
            {
                "stage": "verification",
                "messages": [
                    {"role": "system", "content": "Direct candidate: A"},
                    {"role": "user", "content": question},
                ],
            }
        ],
        "tool_calls": [
            {
                "stage": "verification",
                "start_time": 10.0,
                "end_time": 20.0,
                "nframes": len(frame_paths),
                "resize": 1.0,
                "timestamps": [10.0 + index for index in range(len(frame_paths))],
                "frame_paths": [str(path.resolve()) for path in frame_paths],
                "estimated_visual_tokens": 123,
            }
        ],
    }


def _frames(tmp_path: Path, count: int = 2) -> list[Path]:
    paths: list[Path] = []
    for index in range(count):
        path = tmp_path / f"frame_{index}.jpg"
        path.write_bytes(b"not-decoded-by-unit-test")
        paths.append(path)
    return paths


def _job(tmp_path: Path):
    jobs = bind_trajectory_jobs(
        [_spec()],
        [_trajectory(_frames(tmp_path))],
        schedule_id="budget_006000_seed_17",
    )
    assert len(jobs) == 1
    return jobs[0]


def test_judges_same_frames_candidate_blind_and_merges_provenance(tmp_path: Path) -> None:
    client = FakeClient(
        {17: '{"answer":"B"}', 42: '{"answer":"B"}', 73: '{"answer":"B"}'}
    )
    result = FastHybridTrajectoryJudge(client).judge(_job(tmp_path))

    assert result["trajectory_id"] == _spec()["trajectory_id"]
    assert result["controller_fingerprint"] == SHA_A
    assert result["run_spec_fingerprint"] == SHA_B
    assert result["phase"] == "base"
    assert result["family_id"] == "budget_006000_seed_17"
    assert result["judge_status"] == "complete"
    assert [row["prediction"] for row in result["judge_confirmations"]] == [
        "B",
        "B",
        "B",
    ]
    assert result["judge_visual_tokens"] == 231
    assert result["judge_visual_tokens_complete"] is True
    assert [call["seed"] for call in client.calls] == [17, 42, 73]
    assert all(call["temperature"] == pytest.approx(0.2) for call in client.calls)

    serialized = json.dumps(client.calls, ensure_ascii=False)
    assert "Video Length" not in serialized
    assert "Original video resolution" not in serialized
    assert "Direct candidate" not in serialized
    assert "PRIVATE_CANDIDATE_EXPLANATION" not in serialized
    assert "final_prediction" not in serialized
    assert "file:///" in serialized
    assert all(
        call["response_format"]["json_schema"]["schema"]["additionalProperties"]
        is False
        for call in client.calls
    )


def test_question_recovery_rejects_unsafe_prefix(tmp_path: Path) -> None:
    trajectory = _trajectory(_frames(tmp_path))
    unsafe = (
        "Direct candidate: A. "
        "Question: What colour is the car?\nA: red\nB: blue\nC: green"
    )
    trajectory["request_trace"][0]["messages"][1]["content"] = unsafe
    trajectory["conversation_traces"][0]["messages"][1]["content"] = unsafe

    with pytest.raises(
        ValueError, match="deferred trajectory has no public question/choices prompt"
    ):
        _question_and_choices_prompt(trajectory)


def test_strict_json_parse_failure_is_not_fallback(tmp_path: Path) -> None:
    client = FakeClient(
        {17: '{"answer":"B"}', 42: "Answer: B", 73: '{"answer":"B"}'}
    )
    result = FastHybridTrajectoryJudge(client).judge(_job(tmp_path))
    failed = next(row for row in result["judge_confirmations"] if row["judge_seed"] == 42)
    assert failed["prediction"] is None
    assert failed["parse_error"] == "strict_json_answer_missing"
    assert failed["failure_class"] == "model_parse_failure"
    assert failed["fallback_used"] is False
    assert result["judge_status"] == "complete_with_failures"


def test_resume_runs_only_missing_judge_seed(tmp_path: Path) -> None:
    first_client = FakeClient(
        {17: '{"answer":"B"}', 42: '{"answer":"B"}', 73: '{"answer":"B"}'}
    )
    judge = FastHybridTrajectoryJudge(first_client)
    job = _job(tmp_path)
    existing = judge.judge(job)
    existing["judge_confirmations"] = existing["judge_confirmations"][:2]

    second_client = FakeClient({73: '{"answer":"C"}'})
    updates: list[dict[str, Any]] = []
    resumed = FastHybridTrajectoryJudge(second_client).judge(
        job,
        existing=existing,
        on_update=updates.append,
    )
    assert len(second_client.calls) == 1
    assert second_client.calls[0]["seed"] == 73
    assert len(updates) == 1
    assert [row["judge_seed"] for row in resumed["judge_confirmations"]] == [17, 42, 73]


def test_private_label_is_rejected_before_any_judge_request(tmp_path: Path) -> None:
    trajectory = _trajectory(_frames(tmp_path))
    trajectory["answer"] = "B"
    with pytest.raises(AnnotationLeakError):
        bind_trajectory_jobs(
            [_spec()],
            [trajectory],
            schedule_id="budget_006000_seed_17",
        )


def test_missing_existing_frame_is_saved_as_frame_failure(tmp_path: Path) -> None:
    missing = tmp_path / "missing.jpg"
    job = bind_trajectory_jobs(
        [_spec()],
        [_trajectory([missing])],
        schedule_id="budget_006000_seed_17",
    )[0]
    client = FakeClient()
    result = FastHybridTrajectoryJudge(client).judge(job)
    assert client.calls == []
    assert result["judge_status"] == "complete_with_failures"
    assert len(result["judge_confirmations"]) == 3
    assert all(row["frame_error"] for row in result["judge_confirmations"])


def test_cli_atomic_resume_does_not_repeat_completed_judges(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    spec_path = tmp_path / "specs.jsonl"
    trajectory_path = tmp_path / "trajectories.jsonl"
    output_path = tmp_path / "judged.jsonl"
    spec_path.write_text(json.dumps(_spec()) + "\n", encoding="utf-8")
    trajectory_path.write_text(
        json.dumps(_trajectory(_frames(tmp_path)), ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    clients: list[FakeClient] = []

    def client_factory(*args: Any, **kwargs: Any) -> FakeClient:
        del args, kwargs
        client = FakeClient(
            {17: '{"answer":"B"}', 42: '{"answer":"B"}', 73: '{"answer":"B"}'}
        )
        clients.append(client)
        return client

    monkeypatch.setattr(judge_script, "OpenAICompatibleClient", client_factory)
    args = argparse.Namespace(
        specs=spec_path,
        trajectories=[trajectory_path],
        schedule_id="budget_006000_seed_17",
        output=output_path,
        base_url="http://unused/v1",
        api_key="no",
        model="Qwen3.5-9B",
        judge_seed=[17, 42, 73],
        max_tokens=512,
        temperature=0.2,
        timeout=80.0,
        concurrency=2,
        resume=False,
    )
    first = judge_script.run(args)
    assert first["completed"] == 1
    assert len(clients[0].calls) == 3

    args.resume = True
    second = judge_script.run(args)
    assert second["completed"] == 1
    assert clients[1].calls == []
