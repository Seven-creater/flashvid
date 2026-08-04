from __future__ import annotations

import json
from pathlib import Path

import pytest

from flashvid_eval.privacy import AnnotationLeakError
from flashvid_eval.runner import evaluate, frozen_candidate_costs
from flashvid_eval.schemas import Sample


class _DeferredEvaluator:
    def run_fingerprint(self) -> str:
        return "f" * 64

    def fast_hybrid_eva(self, sample: Sample, candidate: str | None) -> dict:
        return {
            "prediction": candidate,
            "candidate_answer": candidate,
            "candidate_rerun": 0,
            "annotation_leak_check": "passed",
            "usage": {
                "prompt_tokens": 2,
                "completion_tokens": 1,
                "total_tokens": 3,
            },
            "visual_tokens": 0,
            "visual_usage_complete": True,
            "agent_total_tokens_complete": True,
            "request_trace": [{"content": "Answer: A", "usage": {"total_tokens": 3}}],
        }


class _LeakingEvaluator(_DeferredEvaluator):
    def fast_hybrid_eva(self, sample: Sample, candidate: str | None) -> dict:
        result = super().fast_hybrid_eva(sample, candidate)
        result["time_range"] = [1, 2]
        return result


class _MissingVisualUsageEvaluator(_DeferredEvaluator):
    def fast_hybrid_eva(self, sample: Sample, candidate: str | None) -> dict:
        result = super().fast_hybrid_eva(sample, candidate)
        result["visual_tokens"] = None
        result["visual_usage_complete"] = False
        return result


class _TimeoutEvaluator(_DeferredEvaluator):
    def static_audit_fields(self) -> dict:
        return {
            "experiment_config_sha256": "e" * 64,
            "model_artifact_sha256": "m" * 64,
            "manifest_sha256": "d" * 64,
            "train600_manifest_sha256": "t" * 64,
            "trajectory_schedule_id": "budget_006000_seed_17",
            "trajectory_variant_id": "base",
            "trajectory_replica_id": 0,
            "generation_seed": 17,
            "scoring_deferred": True,
        }

    def fast_hybrid_eva(self, sample: Sample, candidate: str | None) -> dict:
        raise TimeoutError("timed out")


def _sample() -> Sample:
    return Sample(
        dataset="lsdbench",
        sample_id="sample-1",
        video="video.mp4",
        question="What happens?",
        choices={"A": "one", "B": "two"},
        answer="A",
        metadata={"time_range": [1, 2], "question_type": "secret"},
    )


def test_deferred_fast_hybrid_result_never_joins_scoring_fields(tmp_path: Path) -> None:
    summary = evaluate(
        [_sample()],
        _DeferredEvaluator(),
        "fast_hybrid_eva",
        tmp_path,
        candidate_answers={"sample-1": "A"},
        candidate_records={
            "sample-1": {
                "executed_usage": {
                    "prompt_tokens": 11,
                    "completion_tokens": 2,
                    "total_tokens": 13,
                },
                "usage": {
                    "prompt_tokens": 5,
                    "completion_tokens": 1,
                    "total_tokens": 6,
                },
                "visual_tokens": 7,
                "visual_usage_complete": True,
                "latency_s": 0.5,
            }
        },
        defer_scoring=True,
    )

    row = json.loads((tmp_path / "lsdbench_fast_hybrid_eva.jsonl").read_text())
    assert row["scoring_deferred"] is True
    assert row["candidate_answer"] == "A"
    for private_key in ("answer", "correct", "time_range", "question_type", "metadata"):
        assert private_key not in row
    assert summary["correct"] is None
    assert summary["accuracy"] is None
    assert row["candidate_usage"]["total_tokens"] == 13
    assert row["agent_usage"]["total_tokens"] == 3
    assert row["end_to_end_total_tokens"] == 16
    assert row["end_to_end_visual_tokens"] == 7
    assert row["end_to_end_latency_s"] >= 0.5
    assert row["candidate_cost_complete"] is True
    assert row["end_to_end_total_tokens_complete"] is True
    assert row["end_to_end_visual_tokens_complete"] is True


def test_candidate_cost_fails_closed_when_visual_accounting_is_incomplete() -> None:
    cost = frozen_candidate_costs(
        {
            "executed_usage": {
                "prompt_tokens": 11,
                "completion_tokens": 2,
                "total_tokens": 13,
            },
            "usage": {
                "prompt_tokens": 5,
                "completion_tokens": 1,
                "total_tokens": 6,
            },
            "visual_tokens": None,
            "visual_usage_complete": False,
        }
    )
    assert cost["usage"]["total_tokens"] == 13
    assert cost["visual_tokens"] is None
    assert cost["usage_complete"] is True
    assert cost["visual_complete"] is False
    assert cost["complete"] is False


def test_end_to_end_visual_cost_is_not_zero_filled_when_agent_usage_is_missing(
    tmp_path: Path,
) -> None:
    evaluate(
        [_sample()],
        _MissingVisualUsageEvaluator(),
        "fast_hybrid_eva",
        tmp_path,
        candidate_answers={"sample-1": "A"},
        candidate_records={
            "sample-1": {
                "executed_usage": {
                    "prompt_tokens": 11,
                    "completion_tokens": 2,
                    "total_tokens": 13,
                },
                "visual_tokens": 7,
                "visual_usage_complete": True,
            }
        },
        defer_scoring=True,
    )
    row = json.loads((tmp_path / "lsdbench_fast_hybrid_eva.jsonl").read_text())
    assert row["agent_visual_tokens"] is None
    assert row["end_to_end_visual_tokens"] is None
    assert row["end_to_end_visual_tokens_complete"] is False


def test_deferred_fast_hybrid_fails_closed_on_backend_annotation(tmp_path: Path) -> None:
    with pytest.raises(AnnotationLeakError, match="time_range"):
        evaluate(
            [_sample()],
            _LeakingEvaluator(),
            "fast_hybrid_eva",
            tmp_path,
            candidate_answers={"sample-1": "A"},
            defer_scoring=True,
        )


def test_exception_row_keeps_frozen_trajectory_provenance(tmp_path: Path) -> None:
    evaluate(
        [_sample()],
        _TimeoutEvaluator(),
        "fast_hybrid_eva",
        tmp_path,
        candidate_answers={"sample-1": "A"},
        defer_scoring=True,
    )

    row = json.loads((tmp_path / "lsdbench_fast_hybrid_eva.jsonl").read_text())
    assert row["error"] == "TimeoutError: timed out"
    assert row["prediction"] == "A"
    assert row["annotation_leak_check"] == "not_run"
    assert row["trajectory_schedule_id"] == "budget_006000_seed_17"
    assert row["generation_seed"] == 17
    assert row["manifest_sha256"] == "d" * 64
    assert row["train600_manifest_sha256"] == "t" * 64
    assert row["candidate_rerun"] == 0
