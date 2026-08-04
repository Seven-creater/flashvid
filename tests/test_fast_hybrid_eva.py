from __future__ import annotations

import json
from pathlib import Path
from types import MethodType

from flashvid_eval.fast_hybrid_eva import (
    OFFICIAL_EVA_COMMIT,
    FastHybridEvaEvaluator,
    _load_official_module,
)
from flashvid_eval.qwen_protocol import QWEN_PROTOCOLS
from flashvid_eval.schemas import Sample
from scripts.summarize_fast_hybrid import _metric


def _run_record(prediction: str | None, *, tool: bool = True) -> dict:
    return {
        "prediction": prediction,
        "raw_response": f"Answer: {prediction}" if prediction else "",
        "finish_reason": "stop",
        "rounds": 2,
        "usage": {
            "prompt_tokens": 100,
            "completion_tokens": 10,
            "total_tokens": 110,
        },
        "latency_s": 0.5,
        "visual_tokens": 1000 if tool else 0,
        "tool_calls": (
            [
                {
                    "stage": "verification",
                    "start_time": 0.0,
                    "end_time": 10.0,
                    "nframes": 8,
                    "resize": 1.0,
                    "timestamps": [0, 2, 4, 6, 8],
                    "estimated_visual_tokens": 1000,
                    "backend": "official_eva_select_frame_fallback",
                }
            ]
            if tool
            else []
        ),
        "call_records": [],
        "stop_reason": "answer_found",
        "error": None,
    }


def _evaluator(records: list[dict]) -> tuple[FastHybridEvaEvaluator, list[str]]:
    evaluator = FastHybridEvaEvaluator.__new__(FastHybridEvaEvaluator)
    evaluator.version = "fast_hybrid_v1"
    evaluator.max_total_visual_tokens = 24000
    evaluator.candidate_results_sha256 = "a" * 64
    prompts: list[str] = []

    forced_calls: list[dict | None] = []

    def fake_run(
        self,
        sample,
        system_prompt,
        visual_budget,
        stage,
        forced_tool_call=None,
    ):
        del self, sample, visual_budget, stage
        prompts.append(system_prompt)
        forced_calls.append(forced_tool_call)
        return records.pop(0)

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


def test_fast_hybrid_allows_only_confirmed_visual_change() -> None:
    evaluator, prompts = _evaluator([_run_record("C"), _run_record("C")])
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
    assert result["change_rejection_reason"] == "independent_confirmation_failed"


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
    assert forced["tool"] == "frame_select"
    assert forced["arguments"] == {
        "start_time": 279.0,
        "end_time": 287.0,
        "nframes": 32,
        "resize": 1.0,
    }


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
