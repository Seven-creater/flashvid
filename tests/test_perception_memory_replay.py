from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

import flashvid_eval.perception_memory_replay as replay_module
from flashvid_eval.client import ChatResult
from flashvid_eval.perception_memory_replay import (
    PerceptionMemoryReplay,
    ReplayConfig,
    cached_frame_steps,
    public_model_sample,
    replay_implementation_dependency_hashes,
    replay_jsonl,
    validate_badcase_audit_summary,
)
from flashvid_eval.perception_memory_sft import build_perception_memory_sft_records


def _audit(path: Path) -> Path:
    funnel = {
        "target_hit": {"known": 0, "passed": 0},
        "evidence_state_valid": {"known": 0, "passed": 0},
        "evidence_complete": {"known": 0, "passed": 0},
        "judge_correct": {"known": 0, "passed": 0},
    }
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "status": "passed",
                "scope_passed": True,
                "samples": 300,
                "paired_samples": 300,
                "datasets": {
                    "lvbench": {"samples": 100, "funnel": funnel},
                    "lsdbench": {"samples": 100, "funnel": funnel},
                    "cgbench": {"samples": 100, "funnel": funnel},
                },
                "flip_totals": {
                    "untrained_correct_sft_wrong": 0,
                    "untrained_wrong_sft_correct": 0,
                    "both_correct": 300,
                    "both_wrong": 0,
                },
                "failure_mode_totals": {
                    "localization": 0,
                    "visual_fact_extraction": 0,
                    "cross_interval_memory": 0,
                    "incomplete_evidence_early_stop": 0,
                    "judging": 0,
                    "candidate_gate": 0,
                    "engineering": 0,
                },
                "taxonomy_required": 0,
                "taxonomy_classified": 0,
                "taxonomy_coverage_passed": True,
            }
        ),
        encoding="utf-8",
    )
    return path


def _state(interval: tuple[float, float], fact: str, *, sufficient: bool) -> str:
    return json.dumps(
        {
            "interval": list(interval),
            "timestamped_facts": [{"time": interval[0] + 1.0, "fact": fact}],
            "option_evidence": {
                "A": {"supports": [], "contradicts": [fact]},
                "B": {"supports": [fact], "contradicts": []},
            },
            "temporal_changes": [],
            "unresolved": [] if sufficient else ["what happens next"],
            "evidence_sufficient": sufficient,
            "next_evidence_needed": "" if sufficient else "observe later",
        }
    )


class FakeClient:
    def __init__(self, outputs: list[str]) -> None:
        self.outputs = list(outputs)
        self.calls: list[dict[str, Any]] = []

    def chat(self, model, messages, max_tokens=32, **kwargs):
        self.calls.append(
            {
                "model": model,
                "messages": messages,
                "max_tokens": max_tokens,
                **kwargs,
            }
        )
        return ChatResult(
            content=self.outputs.pop(0),
            usage={
                "prompt_tokens": 100,
                "completion_tokens": 20,
                "total_tokens": 120,
            },
            raw={},
            latency_s=0.1,
            finish_reason="stop",
        )


def _source(tmp_path: Path, trajectory_id: str = "lvbench:s1:teacher:0") -> dict:
    frames: list[Path] = []
    for index in range(2):
        path = (tmp_path / f"frame-{index}.jpg").resolve()
        path.write_bytes(b"cached frame")
        frames.append(path)
    return {
        "dataset": "lvbench",
        "sample_id": "s1",
        "video": "videos/s1.mp4",
        "trajectory_id": trajectory_id,
        "manifest_sha256": "a" * 64,
        "train600_manifest_sha256": "a" * 64,
        "dataset_manifest_sha256": "b" * 64,
        "candidate_results_sha256": "c" * 64,
        "model_artifact_sha256": "d" * 64,
        "candidate_answer": "A",
        # Private scoring values may exist in a legacy result, but the replay
        # implementation must never traverse/copy them into a model request.
        "answer": "PRIVATE_GROUND_TRUTH_SENTINEL",
        "time_range": [999, 1000],
        "public_sample": {
            "dataset": "lvbench",
            "sample_id": "s1",
            "video": "videos/s1.mp4",
            "question": "What does the person do?",
            "choices": {"A": "waits", "B": "opens the door"},
        },
        "tool_steps": [
            {
                "start_time": 0.0,
                "end_time": 10.0,
                "nframes": 1,
                "resize": 0.75,
                "evidence_request": "Observe the first action.",
                "frame_paths": [str(frames[0])],
                "actual_timestamps": [1.0],
            },
            {
                "start_time": 20.0,
                "end_time": 30.0,
                "nframes": 1,
                "resize": 0.75,
                "evidence_request": "Observe what happens later.",
                "frame_paths": [str(frames[1])],
                "actual_timestamps": [21.0],
            },
        ],
    }


def test_replay_uses_only_current_cached_frames_and_merges_memory(tmp_path: Path) -> None:
    source = _source(tmp_path)
    client = FakeClient(
        [
            _state((0.0, 10.0), "The person approaches the door.", sufficient=False),
            _state((20.0, 30.0), "The person opens the door.", sufficient=True),
        ]
    )
    result = PerceptionMemoryReplay(client).replay(
        source,
        source_file_sha256="a" * 64,
        audit_summary_sha256="b" * 64,
    )

    assert result["scoring_deferred"] is True
    assert result["trajectory_id"].endswith(":perception_memory_replay_v1")
    assert result["candidate_answer"] == "A"
    assert result["diagnostics_gate_sha256"] == "b" * 64
    assert result["experiment_config_sha256"] == result["config_sha256"]
    assert result["candidate_rerun"] == 0
    assert [state["step_index"] for state in result["perception_states"]] == [0, 1]
    assert len(result["event_ledger"]) == 2
    assert [
        (item["stage"], item["prefix_index"], item.get("action_accepted"))
        for item in result["request_trace"]
    ] == [
        ("controller", -1, True),
        ("perception", 0, None),
        ("controller", 0, True),
        ("perception", 1, None),
    ]
    assert result["perception_states"][0]["evidence_complete"] is False
    assert result["perception_states"][1]["evidence_complete"] is False
    assert len(client.calls) == 2
    first_serialized = json.dumps(client.calls[0]["messages"], ensure_ascii=False)
    second_serialized = json.dumps(client.calls[1]["messages"], ensure_ascii=False)
    assert "candidate" not in first_serialized.casefold()
    assert "PRIVATE_GROUND_TRUTH_SENTINEL" not in first_serialized
    assert "999" not in first_serialized
    assert "frame-0.jpg" in first_serialized and "frame-1.jpg" not in first_serialized
    assert "frame-1.jpg" in second_serialized and "frame-0.jpg" not in second_serialized


def test_replay_prefix_schema_becomes_exportable_after_offline_three_seed_gate(
    tmp_path: Path,
) -> None:
    client = FakeClient(
        [
            _state((0.0, 10.0), "The person approaches the door.", sufficient=False),
            _state((20.0, 30.0), "The person opens the door.", sufficient=True),
        ]
    )
    result = PerceptionMemoryReplay(client).replay(
        _source(tmp_path),
        source_file_sha256="a" * 64,
        audit_summary_sha256="b" * 64,
    )
    result["_selection_stable"] = True
    result["prediction"] = result["final_prediction"] = "B"
    result["perception_states"][-1]["evidence_complete"] = True
    judge_messages = [
        {"role": "system", "content": "Judge only the supplied evidence."},
        {"role": "user", "content": "Evidence IDs E0001 and E0002."},
    ]
    result["perception_states"][-1]["judge_confirmations"] = [
        {
            "judge_seed": seed,
            "prediction": "B",
            "evidence_ids": ["E0002"],
            "raw_response": '{"answer":"B","evidence_ids":["E0002"]}',
            "request_messages": judge_messages,
            "evidence_complete": True,
            "annotation_leak_check": "passed",
            "error": None,
        }
        for seed in (17, 42, 73)
    ]
    result["request_trace"].append(
        {
            "stage": "evidence_judge",
            "model": "Qwen3.5-9B",
            "messages": [
                *judge_messages,
            ],
            "content": '{"answer":"B","evidence_ids":["E0002"]}',
            "reasoning_content": "",
            "finish_reason": "stop",
            "usage": {},
            "latency_s": 0.0,
            "seed": 17,
            "step_index": 1,
            "prefix_index": 1,
            "prompt_hash": "f" * 64,
        }
    )
    records = build_perception_memory_sft_records(result)
    assert [record["metadata"]["episode_target_type"] for record in records] == [
        "tool",
        "memory",
        "tool",
        "memory",
        "final",
    ]


def test_legacy_problem_is_read_strictly_from_public_user_prompt(tmp_path: Path) -> None:
    source = _source(tmp_path)
    source.pop("public_sample")
    source["request_trace"] = [
        {
            "messages": [
                {"role": "system", "content": "Direct candidate: A"},
                {
                    "role": "user",
                    "content": (
                        "Question: What does the person do?\n"
                        "A: waits\nB: opens the door"
                    ),
                },
            ]
        }
    ]
    sample = public_model_sample(source)
    assert sample.question == "What does the person do?"
    assert sample.choices == {"A": "waits", "B": "opens the door"}


def test_cached_replay_fails_on_missing_or_mismatched_frames(tmp_path: Path) -> None:
    source = _source(tmp_path)
    source["tool_steps"][0]["actual_timestamps"] = [1.0, 2.0]
    with pytest.raises(ValueError, match="count differs"):
        cached_frame_steps(source)
    source = _source(tmp_path)
    source["tool_steps"][0]["frame_paths"] = [str(tmp_path / "missing.jpg")]
    with pytest.raises(FileNotFoundError, match="cached frame does not exist"):
        cached_frame_steps(source)


def test_replay_fingerprint_includes_semantic_source_dependencies(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dependencies = replay_implementation_dependency_hashes()
    assert set(dependencies) == {
        "perception_memory_replay",
        "perception_memory_eva",
        "privacy",
        "qwen_agent_core",
    }
    assert all(len(value) == 64 for value in dependencies.values())

    baseline = ReplayConfig().fingerprint()
    monkeypatch.setattr(
        replay_module,
        "replay_implementation_dependency_hashes",
        lambda: {**dependencies, "perception_memory_eva": "0" * 64},
    )
    assert ReplayConfig().fingerprint() != baseline


def test_audit_gate_requires_exact_passed_test300_scope(tmp_path: Path) -> None:
    audit = _audit(tmp_path / "summary.json")
    payload, digest = validate_badcase_audit_summary(audit)
    assert payload["samples"] == 300 and len(digest) == 64

    payload["datasets"]["cgbench"]["samples"] = 99
    audit.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="cgbench scope"):
        validate_badcase_audit_summary(audit)
    payload["datasets"]["cgbench"]["samples"] = 100
    payload["duplicate_count"] = 1
    audit.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="integrity failure"):
        validate_badcase_audit_summary(audit)


def test_jsonl_replay_is_atomic_resumable_and_fingerprint_locked(tmp_path: Path) -> None:
    source_path = tmp_path / "source.jsonl"
    output_path = tmp_path / "output.jsonl"
    audit = _audit(tmp_path / "audit.json")
    source = _source(tmp_path)
    source_path.write_text(json.dumps(source) + "\n", encoding="utf-8")
    client = FakeClient(
        [
            _state((0.0, 10.0), "The person approaches the door.", sufficient=False),
            _state((20.0, 30.0), "The person opens the door.", sufficient=True),
        ]
    )
    replayer = PerceptionMemoryReplay(client, ReplayConfig())
    summary = replay_jsonl(
        input_path=source_path,
        output_path=output_path,
        audit_summary_path=audit,
        replayer=replayer,
        concurrency=2,
    )
    assert summary["completed"] == 1 and summary["failed"] == 0
    assert len(output_path.read_text(encoding="utf-8").splitlines()) == 1
    replay_jsonl(
        input_path=source_path,
        output_path=output_path,
        audit_summary_path=audit,
        replayer=PerceptionMemoryReplay(FakeClient([])),
        resume=True,
    )

    changed = dict(source)
    changed["candidate_answer"] = "B"
    source_path.write_text(json.dumps(changed) + "\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="fingerprint mismatch"):
        replay_jsonl(
            input_path=source_path,
            output_path=output_path,
            audit_summary_path=audit,
            replayer=PerceptionMemoryReplay(FakeClient([])),
            resume=True,
        )


def test_bad_cached_sample_is_persisted_without_model_call(tmp_path: Path) -> None:
    source_path = tmp_path / "source.jsonl"
    output_path = tmp_path / "output.jsonl"
    audit = _audit(tmp_path / "audit.json")
    source = _source(tmp_path)
    source["tool_steps"][0]["frame_paths"] = [str(tmp_path / "missing.jpg")]
    source_path.write_text(json.dumps(source) + "\n", encoding="utf-8")
    client = FakeClient([])
    summary = replay_jsonl(
        input_path=source_path,
        output_path=output_path,
        audit_summary_path=audit,
        replayer=PerceptionMemoryReplay(client),
    )
    row = json.loads(output_path.read_text(encoding="utf-8"))
    assert summary["failed"] == 1
    assert row["error_type"] == "FileNotFoundError"
    assert row["scoring_deferred"] is True
    assert client.calls == []
