import argparse
import hashlib
import json
from pathlib import Path

import pytest

from flashvid_eval.client import ChatResult
from flashvid_eval.perception_memory_prefix_judge import bind_prefix_jobs
from flashvid_eval.perception_memory_eva import PERCEPTION_NORMALIZATION_VERSION
from flashvid_eval.perception_memory_replay import PerceptionMemoryReplay
from flashvid_eval.perception_memory_selection import (
    label_and_select_trajectories,
    load_frozen_answers,
)
from flashvid_eval.perception_memory_sft import build_perception_memory_sft_records
from scripts.select_perception_memory_trajectories import (
    _load_bound_train600_answers,
    run as run_selection,
)


def _memory(index: int) -> dict:
    events = [
        {
            "evidence_id": f"E{number:04d}",
            "interval": [float(number - 1) * 10.0, float(number) * 10.0],
            "timestamp": float(number) * 10.0 - 5.0,
            "fact": f"Visible fact {number}.",
            "source": "timestamped_fact",
        }
        for number in range(1, index + 2)
    ]
    ids = [item["evidence_id"] for item in events]
    return {
        "event_ledger": events,
        "option_ledger": {
            "A": {"supports": [], "contradicts": ids},
            "B": {"supports": ids, "contradicts": []},
        },
        "unresolved": [] if index else ["Need the later action."],
        "observed_intervals": [item["interval"] for item in events],
    }


def _trajectory(
    trajectory_id: str,
    *,
    total_tokens: int,
    dataset: str = "lvbench",
    sample_id: str = "s1",
) -> dict:
    return {
        "dataset": dataset,
        "sample_id": sample_id,
        "trajectory_id": trajectory_id,
        "config_sha256": "c" * 64,
        "run_fingerprint": "r" * 64,
        "scoring_deferred": True,
        "annotation_leak_check": "passed",
        "candidate_rerun": 0,
        "perception_normalization_version": PERCEPTION_NORMALIZATION_VERSION,
        "candidate_answer": "A",
        "prediction": None,
        "final_prediction": None,
        "fallback_used": False,
        "fallback_to_candidate": False,
        "error": None,
        "total_tokens": total_tokens,
        "visual_tokens": total_tokens // 2,
        "latency_s": float(total_tokens),
        "public_sample": {
            "dataset": dataset,
            "sample_id": sample_id,
            "video": f"/data/{dataset}/{sample_id}.mp4",
            "question": "Which option is supported?",
            "choices": {"A": "right", "B": "left"},
        },
        "model": "Qwen3.5-9B",
        "tool_steps": [
            {
                "start_time": 0.0,
                "end_time": 10.0,
                "visual_tokens": total_tokens // 4,
                "latency_s": 0.0,
            },
            {
                "start_time": 10.0,
                "end_time": 20.0,
                "visual_tokens": total_tokens // 4,
                "latency_s": 0.0,
            },
        ],
        "perception_states": [
            {"step_index": 0, "memory_after": _memory(0)},
            {"step_index": 1, "memory_after": _memory(1)},
        ],
        "request_trace": [
            {
                "stage": "controller",
                "step_index": 0,
                "prefix_index": -1,
                "usage": {"total_tokens": total_tokens // 4},
                "latency_s": 0.0,
            },
            {
                "stage": "perception",
                "step_index": 0,
                "prefix_index": 0,
                "usage": {"total_tokens": total_tokens // 4},
                "latency_s": 0.0,
            },
            {
                "stage": "controller",
                "step_index": 1,
                "prefix_index": 0,
                "usage": {"total_tokens": total_tokens // 4},
                "latency_s": 0.0,
            },
            {
                "stage": "perception",
                "step_index": 1,
                "prefix_index": 1,
                "usage": {"total_tokens": total_tokens // 4},
                "latency_s": 0.0,
            },
        ],
    }


def _judgments(trajectories: list[dict]) -> list[dict]:
    rows = []
    for job in bind_prefix_jobs(trajectories):
        correct = job.prefix_index == 1
        prediction = "B" if correct else "A"
        rows.append(
            {
                "dataset": job.dataset,
                "sample_id": job.sample_id,
                "trajectory_id": job.trajectory_id,
                "prefix_id": job.prefix_id,
                "prefix_index": job.prefix_index,
                "source_sha256": job.source_sha256,
                "prefix_judge_config_sha256": "j" * 64,
                "scoring_deferred": True,
                "candidate_blind": True,
                "tools_disabled": True,
                "media_count": 0,
                "annotation_leak_check": "passed",
                "judge_status": "complete",
                "judge_confirmations": [
                    {
                        "judge_seed": seed,
                        "prediction": prediction,
                        "evidence_ids": ["E0001"],
                        "parsed_valid": True,
                        "candidate_blind": True,
                        "tools_disabled": True,
                        "media_count": 0,
                        "annotation_leak_check": "passed",
                        "request_messages": [
                            {"role": "system", "content": "Judge evidence only."},
                            {"role": "user", "content": "Public evidence ledger."},
                        ],
                        "request_kwargs": {"tool_choice": "none"},
                        "raw_response": json.dumps(
                            {"answer": prediction, "evidence_ids": ["E0001"]}
                        ),
                        "reasoning_content": "",
                        "finish_reason": "stop",
                        "usage": {"total_tokens": 10},
                        "latency_s": 0.0,
                        "request_prompt_sha256": str(seed % 10) * 64,
                        "error": None,
                        "error_type": None,
                    }
                    for seed in (17, 42, 73)
                ],
            }
        )
    return rows


def test_offline_join_preserves_incomplete_prefix_and_selects_lowest_cost() -> None:
    expensive = _trajectory("lvbench:s1:family:expensive", total_tokens=200)
    cheap = _trajectory("lvbench:s1:family:cheap", total_tokens=100)
    trajectories = [expensive, cheap]
    labeled, selected, summary = label_and_select_trajectories(
        trajectories,
        _judgments(trajectories),
        {("lvbench", "s1"): "B"},
    )

    assert len(labeled) == 2
    assert all(
        [state["evidence_complete"] for state in row["perception_states"]]
        == [False, True]
        for row in labeled
    )
    assert all(row["_selection_stable"] is False for row in labeled)
    assert [row["trajectory_id"] for row in selected] == [
        "lvbench:s1:family:cheap"
    ]
    winner = selected[0]
    assert winner["_selection_stable"] is True
    assert winner["prediction"] == "B"
    assert winner["final_prediction"] == "B"
    assert winner["evidence_answer"] == "B"
    assert winner["earliest_complete_prefix_index"] == 1
    assert winner["retained_total_tokens"] == 110
    assert winner["retained_visual_tokens"] == 50
    assert winner["retained_tool_steps"] == 2
    assert len(winner["request_trace"]) == 5
    assert winner["perception_states"][0]["judge_confirmations"][0][
        "evidence_complete"
    ] is False
    assert winner["perception_states"][1]["judge_confirmations"][0][
        "evidence_complete"
    ] is True
    assert "request_kwargs" not in winner["perception_states"][1][
        "judge_confirmations"
    ][0]
    assert "answer" not in winner
    assert summary["stable_trajectories"] == 2
    assert summary["selected"] == 1
    assert summary["no_stable"] == 0
    assert summary["no_stable_sample_ids"] == {"lvbench": []}
    assert summary["candidate_fixes"] == 1


def test_selection_cost_uses_earliest_complete_prefix_not_redundant_tail() -> None:
    early = _trajectory("lvbench:s1:family:early", total_tokens=1000)
    late = _trajectory("lvbench:s1:family:late", total_tokens=100)
    for request, cost in zip(early["request_trace"], (10, 10, 490, 490)):
        request["usage"]["total_tokens"] = cost
    early["tool_steps"][0]["visual_tokens"] = 10
    early["tool_steps"][1]["visual_tokens"] = 490
    judgments = _judgments([early, late])
    for row in judgments:
        if row["trajectory_id"] == early["trajectory_id"]:
            for confirmation in row["judge_confirmations"]:
                confirmation["prediction"] = "B"
                confirmation["raw_response"] = (
                    '{"answer":"B","evidence_ids":["E0001"]}'
                )

    _labeled, selected, _summary = label_and_select_trajectories(
        [early, late], judgments, {("lvbench", "s1"): "B"}
    )

    assert selected[0]["trajectory_id"] == early["trajectory_id"]
    assert selected[0]["earliest_complete_prefix_index"] == 0
    assert selected[0]["retained_total_tokens"] == 30
    assert selected[0]["retained_visual_tokens"] == 10
    assert selected[0]["retained_tool_steps"] == 1
    assert len(selected[0]["perception_states"]) == 1
    assert len(selected[0]["tool_steps"]) == 1


def test_prefix_complete_requires_three_of_three_correct_valid_answers() -> None:
    trajectory = _trajectory("lvbench:s1:family:0", total_tokens=100)
    judgments = _judgments([trajectory])
    final = next(row for row in judgments if row["prefix_index"] == 1)
    final["judge_confirmations"][2]["prediction"] = "A"
    _labeled, selected, summary = label_and_select_trajectories(
        [trajectory], judgments, {("lvbench", "s1"): "B"}
    )
    assert selected == ()
    assert summary["prefix_status"] == {"incomplete": 2}

    judgments = _judgments([trajectory])
    final = next(row for row in judgments if row["prefix_index"] == 1)
    final["judge_confirmations"][2]["parsed_valid"] = False
    _labeled, selected, _summary = label_and_select_trajectories(
        [trajectory], judgments, {("lvbench", "s1"): "B"}
    )
    assert selected == ()


def test_offline_scoring_refuses_missing_model_calls_or_prefixes() -> None:
    trajectory = _trajectory("lvbench:s1:family:0", total_tokens=100)
    judgments = _judgments([trajectory])
    judgments[-1]["judge_confirmations"].pop()
    with pytest.raises(ValueError, match="all three"):
        label_and_select_trajectories(
            [trajectory], judgments, {("lvbench", "s1"): "B"}
        )

    judgments = _judgments([trajectory])[:-1]
    with pytest.raises(ValueError, match="incomplete"):
        label_and_select_trajectories(
            [trajectory], judgments, {("lvbench", "s1"): "B"}
        )


def test_frozen_answers_are_unique_and_never_serialized_into_selection() -> None:
    answers = load_frozen_answers(
        [{"dataset": "lvbench", "sample_id": "s1", "answer": "B"}]
    )
    assert answers == {("lvbench", "s1"): "B"}
    with pytest.raises(ValueError, match="duplicate"):
        load_frozen_answers(
            [
                {"dataset": "lvbench", "sample_id": "s1", "answer": "B"},
                {"dataset": "lvbench", "sample_id": "s1", "answer": "B"},
            ]
        )

    trajectory = _trajectory("lvbench:s1:family:0", total_tokens=100)
    _labeled, selected, summary = label_and_select_trajectories(
        [trajectory], _judgments([trajectory]), answers
    )
    assert selected[0].get("answer") is None
    assert summary["answers_serialized_into_selected"] == 0


def test_fallback_trajectory_is_not_stable_training_data() -> None:
    trajectory = _trajectory("lvbench:s1:family:0", total_tokens=100)
    trajectory["fallback_to_candidate"] = True
    _labeled, selected, _summary = label_and_select_trajectories(
        [trajectory], _judgments([trajectory]), {("lvbench", "s1"): "B"}
    )
    assert selected == ()


class _ReplayClient:
    def __init__(self, outputs: list[str]) -> None:
        self.outputs = outputs

    def chat(self, model, messages, max_tokens=32, **kwargs):
        del model, messages, max_tokens, kwargs
        return ChatResult(
            content=self.outputs.pop(0),
            usage={"prompt_tokens": 80, "completion_tokens": 20, "total_tokens": 100},
            raw={},
            latency_s=0.1,
            finish_reason="stop",
        )


def _replay_state(interval: tuple[float, float], timestamp: float, fact: str) -> str:
    return json.dumps(
        {
            "interval": list(interval),
            "timestamped_facts": [{"time": timestamp, "fact": fact}],
            "option_evidence": {
                "A": {"supports": [], "contradicts": [fact]},
                "B": {"supports": [fact], "contradicts": []},
            },
            "temporal_changes": [],
            "unresolved": [] if interval[0] else ["Need the later action."],
            "evidence_sufficient": interval[0] > 0,
            "next_evidence_needed": "" if interval[0] else "Observe later.",
        }
    )


def _replay_source(tmp_path: Path) -> dict:
    frames = []
    for index in range(2):
        frame = (tmp_path / f"replay-{index}.jpg").resolve()
        frame.write_bytes(b"cached")
        frames.append(frame)
    return {
        "dataset": "lvbench",
        "sample_id": "s1",
        "video": "videos/s1.mp4",
        "trajectory_id": "lvbench:s1:teacher:0",
        "manifest_sha256": "a" * 64,
        "train600_manifest_sha256": "a" * 64,
        "dataset_manifest_sha256": "b" * 64,
        "candidate_results_sha256": "c" * 64,
        "model_artifact_sha256": "d" * 64,
        "candidate_answer": "A",
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
                "estimated_visual_tokens": 50,
            },
            {
                "start_time": 20.0,
                "end_time": 30.0,
                "nframes": 1,
                "resize": 0.75,
                "evidence_request": "Observe the later action.",
                "frame_paths": [str(frames[1])],
                "actual_timestamps": [21.0],
                "estimated_visual_tokens": 50,
            },
        ],
    }


def test_replay_prefix_selection_materializes_evidence_answer_and_sft(
    tmp_path: Path,
) -> None:
    replayed = PerceptionMemoryReplay(
        _ReplayClient(
            [
                _replay_state((0.0, 10.0), 1.0, "The person approaches."),
                _replay_state((20.0, 30.0), 21.0, "The person opens the door."),
            ]
        )
    ).replay(
        _replay_source(tmp_path),
        source_file_sha256="e" * 64,
        audit_summary_sha256="f" * 64,
    )
    assert replayed["final_prediction"] is None
    judgments = _judgments([replayed])

    _labeled, selected, summary = label_and_select_trajectories(
        [replayed], judgments, {("lvbench", "s1"): "B"}
    )

    assert summary["selected"] == 1
    assert selected[0]["final_prediction"] == "B"
    assert selected[0]["evidence_answer"] == "B"
    assert selected[0].get("answer") is None
    assert selected[0]["retained_total_tokens"] == 210
    assert selected[0]["retained_visual_tokens"] == 100
    assert selected[0]["retained_tool_steps"] == 2
    records = build_perception_memory_sft_records(selected[0])
    assert len(records) == 1
    assert records[0]["metadata"]["process_role"] == "planner"
    assert records[0]["metadata"]["assistant_target_types"] == [
        "tool",
        "tool",
        "stop",
    ]
    assert records[0]["messages"][-1]["content"] == '{"action":"stop"}'
    assert all(
        target != "final"
        for target in records[0]["metadata"]["assistant_target_types"]
    )


def test_replay_prefix_selection_rejects_non_unanimous_evidence_answer(
    tmp_path: Path,
) -> None:
    replayed = PerceptionMemoryReplay(
        _ReplayClient(
            [
                _replay_state((0.0, 10.0), 1.0, "The person approaches."),
                _replay_state((20.0, 30.0), 21.0, "The person opens the door."),
            ]
        )
    ).replay(
        _replay_source(tmp_path),
        source_file_sha256="e" * 64,
        audit_summary_sha256="f" * 64,
    )
    judgments = _judgments([replayed])
    final = next(row for row in judgments if row["prefix_index"] == 1)
    final["judge_confirmations"][2]["prediction"] = "A"

    _labeled, selected, summary = label_and_select_trajectories(
        [replayed], judgments, {("lvbench", "s1"): "B"}
    )

    assert selected == ()
    assert summary["selected"] == 0
    assert summary["no_stable"] == 1
    assert summary["no_stable_by_dataset"] == {"lvbench": 1}
    assert summary["no_stable_sample_ids"] == {"lvbench": ["s1"]}


def test_train600_answer_join_is_hash_and_scope_bound(tmp_path: Path) -> None:
    path = tmp_path / "train600.jsonl"
    rows = [
        {"dataset": dataset, "sample_id": f"{dataset}-{index}", "answer": "A"}
        for dataset in ("lvbench", "lsdbench", "cgbench")
        for index in range(200)
    ]
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    answers, observed = _load_bound_train600_answers(path, digest)
    assert observed == digest
    assert len(answers) == 600
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        _load_bound_train600_answers(path, "0" * 64)

    path.write_text(path.read_text(encoding="utf-8").splitlines()[0] + "\n")
    short_digest = hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="600 rows/200 per dataset"):
        _load_bound_train600_answers(path, short_digest)


def _train600_source_rows() -> list[dict]:
    return [
        {
            "dataset": dataset,
            "sample_id": f"{dataset}-{index:03d}",
            "video": f"videos/{dataset}-{index:03d}.mp4",
            "question": f"Question {dataset} {index}",
            "choices": {"A": "left", "B": "right"},
            "answer": "B",
            "private_marker": f"private-{dataset}-{index:03d}",
        }
        for dataset in ("lvbench", "lsdbench", "cgbench")
        for index in range(200)
    ]


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def _rescue_fixture(tmp_path: Path) -> tuple[argparse.Namespace, list[dict], str]:
    answers_path = tmp_path / "private_train600.jsonl"
    source_rows = _train600_source_rows()
    _write_jsonl(answers_path, source_rows)
    answers_sha256 = hashlib.sha256(answers_path.read_bytes()).hexdigest()

    trajectories: list[dict] = []
    for dataset in ("lvbench", "lsdbench", "cgbench"):
        for index in range(200):
            sample_id = f"{dataset}-{index:03d}"
            trajectory = _trajectory(
                f"{dataset}:{sample_id}:family:0",
                total_tokens=100,
                dataset=dataset,
                sample_id=sample_id,
            )
            trajectory["train600_manifest_sha256"] = answers_sha256
            trajectories.append(trajectory)
    judgments = _judgments(trajectories)
    for row in judgments:
        if str(row["sample_id"]).endswith("-001"):
            for confirmation in row["judge_confirmations"]:
                confirmation["prediction"] = "A"
                confirmation["raw_response"] = json.dumps(
                    {"answer": "A", "evidence_ids": ["E0001"]}
                )

    trajectories_path = tmp_path / "trajectories.jsonl"
    judgments_path = tmp_path / "judgments.jsonl"
    _write_jsonl(trajectories_path, trajectories)
    _write_jsonl(judgments_path, judgments)
    args = argparse.Namespace(
        trajectories=[trajectories_path],
        prefix_judgments=[judgments_path],
        answers=answers_path,
        expected_answers_sha256=answers_sha256,
        labeled_output=tmp_path / "labeled.jsonl",
        selected_output=tmp_path / "selected.jsonl",
        summary=tmp_path / "summary.json",
        rescue_manifest_dir=tmp_path / "rescue",
        overwrite=False,
    )
    return args, source_rows, answers_sha256


def test_rescue_manifests_contain_only_bound_no_stable_samples(
    tmp_path: Path,
) -> None:
    args, source_rows, _answers_sha256 = _rescue_fixture(tmp_path)
    report = run_selection(args)

    assert report["no_stable"] == 3
    assert report["no_stable_by_dataset"] == {
        "cgbench": 1,
        "lsdbench": 1,
        "lvbench": 1,
    }
    expected_source = {
        (row["dataset"], row["sample_id"]): row for row in source_rows
    }
    observed: set[tuple[str, str]] = set()
    for dataset in ("lvbench", "lsdbench", "cgbench"):
        metadata = report["rescue_manifests"][dataset]
        path = Path(metadata["path"])
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        assert metadata["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
        assert metadata["samples"] == 1
        assert metadata["sample_ids"] == [f"{dataset}-001"]
        assert rows == [expected_source[(dataset, f"{dataset}-001")]]
        identity = (rows[0]["dataset"], rows[0]["sample_id"])
        assert identity not in observed
        observed.add(identity)
    assert len(observed) == 3
    persisted = json.loads(args.summary.read_text(encoding="utf-8"))
    assert persisted["rescue_manifests"] == report["rescue_manifests"]


def test_existing_rescue_manifest_requires_overwrite(tmp_path: Path) -> None:
    args, _source_rows, _answers_sha256 = _rescue_fixture(tmp_path)
    run_selection(args)
    second = argparse.Namespace(
        **{
            **vars(args),
            "labeled_output": tmp_path / "second-labeled.jsonl",
            "selected_output": tmp_path / "second-selected.jsonl",
            "summary": tmp_path / "second-summary.json",
        }
    )
    with pytest.raises(RuntimeError, match="pass --overwrite"):
        run_selection(second)
    second.overwrite = True
    report = run_selection(second)
    assert report["rescue_manifests"]["lvbench"]["samples"] == 1


def test_rescue_cannot_be_generated_from_unbound_train600_source(
    tmp_path: Path,
) -> None:
    args, _source_rows, _answers_sha256 = _rescue_fixture(tmp_path)
    rows = [json.loads(line) for line in args.trajectories[0].read_text().splitlines()]
    for row in rows:
        row["train600_manifest_sha256"] = "0" * 64
    _write_jsonl(args.trajectories[0], rows)

    with pytest.raises(ValueError, match="not bound to the supplied Train600"):
        run_selection(args)
    assert not args.labeled_output.exists()
    assert not args.selected_output.exists()
    assert not args.summary.exists()
    assert not args.rescue_manifest_dir.exists()
