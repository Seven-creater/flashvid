from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path

import pytest

from flashvid_eval.qwen_sft import (
    ExpectedTrajectoryProvenance,
    TrainingManifest,
    build_sft_record,
    build_sft_records,
    checkpoint_gate,
    composite_trajectory_id,
    generate_counterfactual_specs,
    load_training_manifest,
    materialize_trajectory_identity,
    parse_composite_trajectory_id,
    select_stable_correct_trajectories,
    sha256_file,
    source_fingerprint,
    validate_exported_sft_record,
)
from flashvid_eval.qwen_agents.core import (
    AgentTrace,
    EvidenceEntry,
    RequestTrace,
    ToolStep,
)
from scripts.build_qwen_agent_sft import build
from scripts.check_qwen_agent_sft import validate_data


MANIFEST_HASH = "b" * 64
DATASET_MANIFEST_HASH = "c" * 64
CONFIG_HASH = "a" * 64
MODEL_HASH = "d" * 64
AGENT_HASH = "e" * 64
RUNNER_HASH = "f" * 64


def _provenance() -> ExpectedTrajectoryProvenance:
    return ExpectedTrajectoryProvenance(
        model="Qwen3.5-9B",
        model_artifact_sha256=MODEL_HASH,
        dataset_manifest_sha256s=frozenset({DATASET_MANIFEST_HASH}),
        agent_config_sha256s=frozenset({AGENT_HASH}),
        runner_fingerprints=frozenset({RUNNER_HASH}),
    )


def _messages() -> list[dict[str, object]]:
    return [
        {"role": "system", "content": "Use frame evidence."},
        {"role": "user", "content": "Question and choices"},
        {"role": "assistant", "content": '{"route":"event"}', "target_type": "plan"},
        {
            "role": "assistant",
            "content": '<tool_call>{"start_time":0,"end_time":10}</tool_call>',
            "target_type": "tool",
        },
        {"role": "tool", "content": "Frames at 0s and 10s"},
        {
            "role": "assistant",
            "content": '{"supports":["A"]}',
            "target_type": "memory",
        },
        {"role": "assistant", "content": "safe commentary", "target_type": "note"},
        {"role": "assistant", "content": '{"answer":"A"}', "target_type": "final"},
    ]


def _steps() -> list[dict[str, object]]:
    return [
        {
            "start_time": 0.0,
            "end_time": 10.0,
            "actual_timestamps": [0.0, 5.0, 10.0],
            "nframes": 4,
            "resize": 0.5,
            "visual_tokens": 40,
        },
        {
            "start_time": 20.0,
            "end_time": 40.0,
            "actual_timestamps": [20.0, 30.0, 40.0],
            "nframes": 8,
            "resize": 0.75,
            "visual_tokens": 80,
        },
    ]


def _trajectory(
    schedule: str,
    replica: int,
    *,
    prediction: str = "A",
    total_tokens: int = 100,
    variant: str = "base",
    error: str | None = None,
    fallback: bool = False,
) -> dict[str, object]:
    family = schedule if variant == "base" else f"{schedule}~{variant}"
    return {
        "dataset": "lvbench",
        "sample_id": "video:question/1",
        "schedule_id": schedule,
        "variant_id": variant,
        "family_id": family,
        "replica_id": replica,
        "trajectory_id": composite_trajectory_id(
            "lvbench", "video:question/1", family, replica
        ),
        "manifest_sha256": MANIFEST_HASH,
        "train600_manifest_sha256": MANIFEST_HASH,
        "dataset_manifest_sha256": DATASET_MANIFEST_HASH,
        "config_sha256": CONFIG_HASH,
        "model": "Qwen3.5-9B",
        "model_artifact_sha256": MODEL_HASH,
        "agent_config_sha256": AGENT_HASH,
        "trajectory_runner_fingerprint": RUNNER_HASH,
        "scoring_deferred": True,
        "prediction": prediction,
        "training_messages": _messages(),
        "request_trace": [{"usage": {"total_tokens": total_tokens}}],
        "tool_steps": _steps(),
        "prompt_tokens": 70,
        "completion_tokens": 20,
        "reasoning_tokens": 10,
        "visual_tokens": 120,
        "total_tokens": total_tokens,
        "latency_s": 1.0,
        "fallback_used": fallback,
        "annotation_leak_check": "passed",
        "error": error,
        "error_type": None,
        "judge_seed": (17, 42, 73)[replica],
    }


def _manifest() -> TrainingManifest:
    return TrainingManifest(
        answers={("lvbench", "video:question/1"): "A"},
        sha256=MANIFEST_HASH,
        counts={"lvbench": 1},
    )


def test_composite_trajectory_id_round_trips_delimiters() -> None:
    identity = ("lvbench", "a:b/c", "schedule:0~frames_050", "2")
    encoded = composite_trajectory_id(*identity)
    assert parse_composite_trajectory_id(encoded) == identity
    assert encoded.count(":") == 3


def test_agent_trace_materializer_adds_provenance_and_tool_aliases() -> None:
    raw = _trajectory("schedule-0", 0)
    raw.pop("schedule_id")
    raw.pop("variant_id")
    raw.pop("family_id")
    raw.pop("replica_id")
    raw.pop("trajectory_id")
    raw["tool_steps"] = [
        {
            "request": {"start_time": 1, "end_time": 9, "nframes": 4, "resize": 0.5},
            "resolved_start_time": 1.0,
            "resolved_end_time": 9.0,
            "resolved_nframes": 4,
            "timestamps": [1.0, 3.0, 6.0, 9.0],
            "visual_tokens": 40,
        }
    ]
    raw["visual_tokens"] = 40
    materialized = materialize_trajectory_identity(
        raw,
        schedule_id="schedule-7",
        replica_id=2,
        judge_seed=73,
        manifest_sha256=MANIFEST_HASH,
        dataset_manifest_sha256=DATASET_MANIFEST_HASH,
        config_sha256=CONFIG_HASH,
    )
    assert materialized["trajectory_id"] == composite_trajectory_id(
        "lvbench", "video:question/1", "schedule-7", 2
    )
    assert materialized["tool_steps"][0]["actual_timestamps"] == [1.0, 3.0, 6.0, 9.0]
    assert materialized["tool_steps"][0]["nframes"] == 4


def test_manifest_is_an_offline_ground_truth_join_and_enforces_counts(
    tmp_path: Path,
) -> None:
    path = tmp_path / "train.jsonl"
    rows = [
        {"dataset": dataset, "sample_id": f"{dataset}-1", "answer": "A"}
        for dataset in ("lvbench", "lsdbench", "cgbench")
    ]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    manifest = load_training_manifest(
        path,
        expected_counts={"lvbench": 1, "lsdbench": 1, "cgbench": 1},
    )
    assert manifest.answers[("lsdbench", "lsdbench-1")] == "A"
    assert len(manifest.sha256) == 64
    with pytest.raises(ValueError, match="Train600 counts"):
        load_training_manifest(path)


def test_stable_selection_requires_all_three_judges_and_uses_median_cost() -> None:
    rows = [
        _trajectory("schedule-0", replica, total_tokens=cost)
        for replica, cost in enumerate((90, 100, 120))
    ]
    rows.extend(
        _trajectory("schedule-1", replica, prediction="B", total_tokens=10)
        for replica in range(3)
    )
    result = select_stable_correct_trajectories(
        rows,
        _manifest(),
        config_sha256=CONFIG_HASH,
        expected_schedules=2,
    )
    assert len(result.selected) == 1
    assert result.selected[0]["total_tokens"] == 100
    assert result.selected[0]["_selection_median_total_tokens"] == 100
    assert result.selected[0]["_selection_confirmation_count"] == 3
    assert result.rejected_families == {"incorrect": 1}


def test_nested_three_judge_confirmations_are_supported() -> None:
    row = _trajectory("schedule-0", 0)
    row["judge_confirmations"] = [
        {"judge_seed": seed, "prediction": "A", "annotation_leak_check": "passed"}
        for seed in (17, 42, 73)
    ]
    result = select_stable_correct_trajectories(
        [row],
        _manifest(),
        config_sha256=CONFIG_HASH,
        expected_schedules=1,
    )
    assert len(result.selected) == 1
    row["judge_confirmations"][2]["prediction"] = "B"
    result = select_stable_correct_trajectories(
        [row],
        _manifest(),
        config_sha256=CONFIG_HASH,
        expected_schedules=1,
    )
    assert result.rejected_families == {"unstable": 1}


@pytest.mark.parametrize(
    ("mutation", "reason"),
    [
        (lambda row: row.update(error="api timeout"), "infrastructure"),
        (lambda row: row.update(error_type="parse_error"), "parser"),
        (lambda row: row.update(fallback_used=True), "fallback"),
        (lambda row: row.update(annotation_leak_check="failed"), "annotation_leak"),
        (lambda row: row["training_messages"][1].update(time_range=[1, 2]), "annotation_leak"),
    ],
)
def test_selection_rejects_invalid_families(mutation, reason: str) -> None:
    rows = [_trajectory("schedule-0", replica) for replica in range(3)]
    for row in rows:
        mutation(row)
    result = select_stable_correct_trajectories(
        rows,
        _manifest(),
        config_sha256=CONFIG_HASH,
        expected_schedules=1,
    )
    assert result.selected == ()
    assert result.rejected_families == {reason: 1}


def test_bad_accounting_rejects_only_its_family_but_hashes_still_fail_hard() -> None:
    stable = [_trajectory("schedule-0", replica) for replica in range(3)]
    missing_tools = [_trajectory("schedule-1", replica) for replica in range(3)]
    for row in missing_tools:
        row["tool_steps"] = []
    missing_tokens = [_trajectory("schedule-2", replica) for replica in range(3)]
    for row in missing_tokens:
        row.pop("total_tokens")
        row["request_trace"] = []

    result = select_stable_correct_trajectories(
        [*stable, *missing_tools, *missing_tokens],
        _manifest(),
        config_sha256=CONFIG_HASH,
        expected_schedules=3,
    )

    assert len(result.selected) == 1
    assert result.selected[0]["schedule_id"] == "schedule-0"
    assert result.rejected_families == {
        "invalid_token_accounting": 1,
        "invalid_tool_trace": 1,
    }

    missing_tools[0]["manifest_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="manifest_sha256 mismatch"):
        select_stable_correct_trajectories(
            [*stable, *missing_tools, *missing_tokens],
            _manifest(),
            config_sha256=CONFIG_HASH,
            expected_schedules=3,
        )


def test_expected_q9_provenance_is_a_hard_gate() -> None:
    row = _trajectory("schedule-0", 0)
    row["judge_confirmations"] = [
        {"judge_seed": seed, "prediction": "A", "annotation_leak_check": "passed"}
        for seed in (17, 42, 73)
    ]
    result = select_stable_correct_trajectories(
        [row],
        _manifest(),
        config_sha256=CONFIG_HASH,
        expected_schedules=1,
        expected_provenance=_provenance(),
    )
    assert len(result.selected) == 1

    for field, value, message in (
        ("scoring_deferred", False, "scoring_deferred"),
        ("model", "Qwen3.5-4B", "Qwen3.5-9B"),
        ("model_artifact_sha256", "0" * 64, "model_artifact"),
        ("agent_config_sha256", "1" * 64, "agent_config"),
        ("trajectory_runner_fingerprint", "2" * 64, "runner_fingerprint"),
    ):
        changed = deepcopy(row)
        changed[field] = value
        with pytest.raises(ValueError, match=message):
            select_stable_correct_trajectories(
                [changed],
                _manifest(),
                config_sha256=CONFIG_HASH,
                expected_schedules=1,
                expected_provenance=_provenance(),
            )


def test_selection_enforces_twelve_base_schedules_per_train_sample() -> None:
    rows = [
        _trajectory(f"schedule-{schedule}", replica)
        for schedule in range(11)
        for replica in range(3)
    ]
    with pytest.raises(ValueError, match="12 base schedules"):
        select_stable_correct_trajectories(
            rows,
            _manifest(),
            config_sha256=CONFIG_HASH,
        )


def test_counterfactuals_preserve_intervals_and_reduce_only_prefix_or_frames() -> None:
    source = _trajectory("schedule-0", 0)
    specs = generate_counterfactual_specs(source)
    prefix = next(spec for spec in specs if spec["variant_id"] == "prefix_1")
    assert len(prefix["planned_calls"]) == 1
    assert prefix["planned_calls"][0]["start_time"] == 0
    assert prefix["planned_calls"][0]["end_time"] == 10
    assert prefix["planned_calls"][0]["resize"] == 0.5
    reduced = next(
        spec for spec in specs if spec["variant_id"] == "prefix_2_frames_025"
    )
    assert [(call["start_time"], call["end_time"]) for call in reduced["planned_calls"]] == [
        (0, 10),
        (20, 40),
    ]
    assert [call["nframes"] for call in reduced["planned_calls"]] == [1, 2]
    assert len(reduced["replica_trajectory_ids"]) == 3


def test_counterfactual_build_covers_every_stable_correct_base_family(
    tmp_path: Path,
) -> None:
    manifest_path = tmp_path / "train.jsonl"
    manifest_path.write_text(
        json.dumps(
            {"dataset": "lvbench", "sample_id": "video:question/1", "answer": "A"}
        )
        + "\n",
        encoding="utf-8",
    )
    manifest_hash = sha256_file(manifest_path)
    rows = []
    for schedule, cost in (("schedule-0", 80), ("schedule-1", 120)):
        row = _trajectory(schedule, 0, total_tokens=cost)
        row["manifest_sha256"] = manifest_hash
        row["train600_manifest_sha256"] = manifest_hash
        row["judge_confirmations"] = [
            {
                "judge_seed": seed,
                "prediction": "A",
                "annotation_leak_check": "passed",
            }
            for seed in (17, 42, 73)
        ]
        rows.append(row)
    trajectory_path = tmp_path / "base.jsonl"
    trajectory_path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )

    summary = build(
        phase="counterfactuals",
        train_manifest_path=manifest_path,
        trajectory_paths=[trajectory_path],
        output_dir=tmp_path / "counterfactuals",
        config_sha256=CONFIG_HASH,
        expected_provenance=_provenance(),
        resume=False,
        expected_counts={"lvbench": 1},
        expected_schedules=2,
    )

    sources = [
        json.loads(line)
        for line in (tmp_path / "counterfactuals/counterfactual_sources.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert summary["outputs"]["stable_source_trajectories"] == 2
    assert {row["schedule_id"] for row in sources} == {"schedule-0", "schedule-1"}


def test_sft_export_masks_only_approved_targets_and_has_no_ground_truth() -> None:
    trajectory = _trajectory("schedule-0", 0)
    trajectory.update(
        {
            "_selection_family_id": "schedule-0",
            "_selection_stable": True,
            "_selection_confirmation_count": 3,
        }
    )
    record = build_sft_record(trajectory)
    validate_exported_sft_record(record)
    assistant = [message for message in record["messages"] if message["role"] == "assistant"]
    assert [message["loss"] for message in assistant] == [True, True, True, False, True]
    assert record["metadata"]["assistant_target_types"] == [
        "plan",
        "tool",
        "memory",
        "final",
    ]
    serialized = json.dumps(record)
    assert "ground_truth" not in serialized
    assert '"answer"' not in serialized


def test_sft_export_rejects_hidden_reasoning_even_when_masked() -> None:
    trajectory = _trajectory("schedule-0", 0)
    trajectory["training_messages"][2]["reasoning_content"] = "private thought"
    with pytest.raises(ValueError, match="hidden reasoning"):
        build_sft_record(trajectory)


def test_source_fingerprint_binds_manifest_config_files_and_phase() -> None:
    first = source_fingerprint(
        manifest_sha256=MANIFEST_HASH,
        config_sha256=CONFIG_HASH,
        trajectory_files={"a.jsonl": "c" * 64},
        phase="select",
    )
    second = source_fingerprint(
        manifest_sha256=MANIFEST_HASH,
        config_sha256=CONFIG_HASH,
        trajectory_files={"a.jsonl": "d" * 64},
        phase="select",
    )
    assert first != second


def test_checkpoint_gate_requires_strict_accuracy_and_both_30pct_reductions() -> None:
    teacher = {"correct": 100, "total_tokens": 1000, "visual_tokens": 800}
    passing = checkpoint_gate(
        teacher,
        {"correct": 101, "total_tokens": 700, "visual_tokens": 560},
    )
    assert passing["passed"] is True
    same_accuracy = checkpoint_gate(
        teacher,
        {"correct": 100, "total_tokens": 600, "visual_tokens": 400},
    )
    assert same_accuracy["passed"] is False
    expensive = checkpoint_gate(
        teacher,
        {"correct": 101, "total_tokens": 701, "visual_tokens": 560},
    )
    assert expensive["passed"] is False


def test_stable_replicas_must_share_the_exact_tool_trace() -> None:
    rows = [_trajectory("schedule-0", replica) for replica in range(3)]
    changed = deepcopy(rows[2]["tool_steps"])
    changed[0]["actual_timestamps"] = [0.0, 4.0, 10.0]
    rows[2]["tool_steps"] = changed
    result = select_stable_correct_trajectories(
        rows,
        _manifest(),
        config_sha256=CONFIG_HASH,
        expected_schedules=1,
    )
    assert result.rejected_families == {"unstable_tool_trace": 1}


def test_offline_build_resume_is_content_addressed_and_refuses_changed_inputs(
    tmp_path: Path,
) -> None:
    manifest_path = tmp_path / "train.jsonl"
    manifest_path.write_text(
        json.dumps(
            {"dataset": "lvbench", "sample_id": "video:question/1", "answer": "A"}
        )
        + "\n",
        encoding="utf-8",
    )
    manifest_hash = sha256_file(manifest_path)
    row = _trajectory("schedule-0", 0)
    row["manifest_sha256"] = manifest_hash
    row["train600_manifest_sha256"] = manifest_hash
    row["judge_confirmations"] = [
        {"judge_seed": seed, "prediction": "A", "annotation_leak_check": "passed"}
        for seed in (17, 42, 73)
    ]
    trajectory_path = tmp_path / "traces.jsonl"
    trajectory_path.write_text(json.dumps(row) + "\n", encoding="utf-8")
    output = tmp_path / "out"
    first = build(
        phase="select",
        train_manifest_path=manifest_path,
        trajectory_paths=[trajectory_path],
        output_dir=output,
        config_sha256=CONFIG_HASH,
        expected_provenance=_provenance(),
        resume=False,
        expected_counts={"lvbench": 1},
        expected_schedules=1,
    )
    assert first["outputs"]["sft_records"] == 1
    resumed = build(
        phase="select",
        train_manifest_path=manifest_path,
        trajectory_paths=[trajectory_path],
        output_dir=output,
        config_sha256=CONFIG_HASH,
        expected_provenance=_provenance(),
        resume=True,
        expected_counts={"lvbench": 1},
        expected_schedules=1,
    )
    assert resumed["resumed_without_changes"] is True
    trajectory_path.write_text(
        trajectory_path.read_text(encoding="utf-8") + "\n",
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="fingerprint changed"):
        build(
            phase="select",
            train_manifest_path=manifest_path,
            trajectory_paths=[trajectory_path],
            output_dir=output,
            config_sha256=CONFIG_HASH,
            expected_provenance=_provenance(),
            resume=True,
            expected_counts={"lvbench": 1},
            expected_schedules=1,
        )


def _request_trace(
    *,
    branch: str,
    request_kind: str,
    system: str,
    user_content: object,
    content: str,
    prompt_hash: str,
    finish_reason: str = "stop",
    attempt_index: int = 0,
    reasoning_content: str = "SECRET_HIDDEN_REASONING",
) -> RequestTrace:
    return RequestTrace(
        branch=branch,
        request_kind=request_kind,
        model="Qwen3.5-9B",
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user_content},
        ],
        content=content,
        reasoning_content=reasoning_content,
        finish_reason=finish_reason,
        usage={"prompt_tokens": 10, "completion_tokens": 3, "total_tokens": 13},
        latency_s=0.1,
        max_tokens=512,
        temperature=0,
        seed=42,
        enable_thinking=True,
        sampling_params={},
        mm_processor_kwargs=None,
        media_io_kwargs=None,
        attempt_index=attempt_index,
        retry_of_length=attempt_index > 0,
        prompt_hash=prompt_hash,
    )


def test_real_agent_trace_exports_independent_request_level_episodes(
    tmp_path: Path,
) -> None:
    frame = tmp_path / "frame.jpg"
    frame.write_bytes(b"frame")
    observer = _request_trace(
        branch="evidence",
        request_kind="observer",
        system="observer-system",
        user_content=[
            {"type": "text", "text": "Inspect the frame"},
            {"type": "image_url", "image_url": {"url": frame.resolve().as_uri()}},
        ],
        content='{"observed_facts":["door opens"]}',
        prompt_hash="1" * 64,
    )
    judge = _request_trace(
        branch="judge",
        request_kind="judge",
        system="judge-system",
        user_content="Question, choices, and timestamped evidence",
        content='{"answer":"A"}',
        prompt_hash="2" * 64,
    )
    trace = AgentTrace(
        strategy="a1_storyboard_zoom",
        model="Qwen3.5-9B",
        dataset="lvbench",
        sample_id="video:question/1",
        prediction="A",
        final_prediction="A",
        request_trace=[observer, judge],
        tool_steps=[
            ToolStep(
                branch="evidence",
                request={"start_time": 0, "end_time": 10, "nframes": 1, "resize": 0.5},
                start_time=0,
                end_time=10,
                nframes=1,
                resize=0.5,
                actual_timestamps=(5.0,),
                resolved_start_time=0,
                resolved_end_time=10,
                resolved_nframes=1,
                frame_paths=(str(frame),),
                timestamps=(5.0,),
                backend="fake-decord",
                cache_hit=False,
                visual_tokens=50,
                latency_s=0.01,
            )
        ],
        evidence_memory=[
            EvidenceEntry(
                source="storyboard_overview",
                interval=(0, 10),
                timestamps=(5.0,),
                content=observer.content,
            )
        ],
    )
    raw = trace.to_result_dict()
    materialized = materialize_trajectory_identity(
        raw,
        schedule_id="schedule-0",
        replica_id=0,
        judge_seed=17,
        manifest_sha256=MANIFEST_HASH,
        dataset_manifest_sha256=DATASET_MANIFEST_HASH,
        config_sha256=CONFIG_HASH,
    )
    records = build_sft_records(materialized)
    assert len(records) == 2
    assert [record["metadata"]["episode_target_type"] for record in records] == [
        "memory",
        "final",
    ]
    assert [record["metadata"]["turn_index"] for record in records] == [0, 1]
    assert records[0]["metadata"]["tool_step_indices"] == [0]
    assert records[0]["metadata"]["evidence_memory_indices"] == [0]
    assert records[0]["images"] == [str(frame.resolve())]
    assert "<image>" in records[0]["messages"][1]["content"]
    assert "images" not in records[1]
    assert records[0]["messages"][0]["content"] == "observer-system"
    assert records[1]["messages"][0]["content"] == "judge-system"
    assert all(
        sum(message.get("loss") is True for message in record["messages"]) == 1
        for record in records
    )
    assert "SECRET_HIDDEN_REASONING" not in json.dumps(records)
    assert all(len([m for m in record["messages"] if m["role"] == "system"]) == 1 for record in records)
    for record in records:
        validate_exported_sft_record(record)


@pytest.mark.parametrize(
    ("strategy", "observer_contents", "expected_targets"),
    [
        (
            "a1_storyboard_zoom",
            [
                '{"intervals":[[10,20],[100,110]],"evidence":"two events"}',
                '{"observed_facts":["the door opens"],"supports":["A"]}',
            ],
            ["plan", "memory", "final"],
        ),
        (
            "a3_hierarchical_search",
            [
                '{"selected_node":2,"node_summaries":["first split"]}',
                '{"selected_node":1,"node_summaries":["second split"]}',
                '{"observed_facts":["the person leaves"]}',
            ],
            ["plan", "plan", "memory", "final"],
        ),
    ],
)
def test_real_a1_a3_observer_requests_distinguish_plans_from_memory(
    strategy: str,
    observer_contents: list[str],
    expected_targets: list[str],
) -> None:
    requests = [
        _request_trace(
            branch="evidence",
            request_kind="observer",
            system="observer-system",
            user_content=f"Observation step {index}",
            content=content,
            prompt_hash=f"{index + 1:x}" * 64,
        )
        for index, content in enumerate(observer_contents)
    ]
    requests.append(
        _request_trace(
            branch="judge",
            request_kind="judge",
            system="judge-system",
            user_content="Question, choices, and timestamped evidence",
            content='{"answer":"A"}',
            prompt_hash="f" * 64,
        )
    )
    trace = AgentTrace(
        strategy=strategy,
        model="Qwen3.5-9B",
        dataset="lvbench",
        sample_id="video:question/1",
        prediction="A",
        final_prediction="A",
        request_trace=requests,
        evidence_memory=[
            EvidenceEntry(
                source=f"observation_{index}",
                interval=(float(index), float(index + 1)),
                timestamps=(float(index) + 0.5,),
                content=content,
            )
            for index, content in enumerate(observer_contents)
        ],
    )
    materialized = materialize_trajectory_identity(
        trace.to_result_dict(),
        schedule_id="schedule-0",
        replica_id=0,
        judge_seed=17,
        manifest_sha256=MANIFEST_HASH,
        dataset_manifest_sha256=DATASET_MANIFEST_HASH,
        config_sha256=CONFIG_HASH,
    )

    records = build_sft_records(materialized)

    assert [
        record["metadata"]["episode_target_type"] for record in records
    ] == expected_targets


def test_request_episode_normalizes_video_and_rejects_unsafe_media(
    tmp_path: Path,
) -> None:
    video = tmp_path / "video.mp4"
    video.write_bytes(b"video")
    trajectory = _trajectory("schedule-0", 0)
    trajectory.pop("training_messages")
    trajectory["evidence_memory"] = []
    direct = asdict(
        _request_trace(
            branch="direct",
            request_kind="direct",
            system="direct-system",
            user_content=[
                {"type": "video_url", "video_url": {"url": video.resolve().as_uri()}},
                {"type": "text", "text": "Question and choices"},
            ],
            content='{"answer":"A"}',
            prompt_hash="7" * 64,
        )
    )
    trajectory["request_trace"] = [direct]
    record = build_sft_records(trajectory)[0]
    assert record["videos"] == [str(video.resolve())]
    assert record["messages"][1]["content"].splitlines() == [
        "<video>",
        "Question and choices",
    ]
    assert "images" not in record
    validate_exported_sft_record(record)

    direct["messages"][1]["content"][0]["video_url"]["url"] = "https://example.com/video.mp4"
    trajectory["request_trace"] = [direct]
    with pytest.raises(ValueError, match="only accepts local file"):
        build_sft_records(trajectory)

    direct["messages"][1]["content"][0]["video_url"]["url"] = (
        tmp_path / "missing.mp4"
    ).resolve().as_uri()
    with pytest.raises(ValueError, match="does not exist"):
        build_sft_records(trajectory)


def test_request_episode_keeps_history_masked_and_refuses_invalid_history() -> None:
    trajectory = _trajectory("schedule-0", 0)
    trajectory.pop("training_messages")
    trajectory["evidence_memory"] = []
    request = asdict(
        _request_trace(
            branch="arbiter",
            request_kind="judge",
            system="arbiter-system",
            user_content="Compare two hypotheses",
            content='{"answer":"A"}',
            prompt_hash="3" * 64,
        )
    )
    request["messages"].extend(
        [
            {
                "role": "assistant",
                "content": '<tool_call>{"tool":"frame_select"}</tool_call>',
            },
            {"role": "tool", "content": "visible confirmation"},
        ]
    )
    trajectory["request_trace"] = [request]
    record = build_sft_records(trajectory)[0]
    assistant = [message for message in record["messages"] if message["role"] == "assistant"]
    assert [message["loss"] for message in assistant] == [False, True]

    request["messages"].insert(1, {"role": "system", "content": "second system"})
    with pytest.raises(ValueError, match="system message may only appear first"):
        build_sft_records(trajectory)


def test_request_episode_uses_only_terminal_retry_and_never_guesses_judge_type() -> None:
    trajectory = _trajectory("schedule-0", 0)
    trajectory.pop("training_messages")
    trajectory["evidence_memory"] = []
    truncated = asdict(
        _request_trace(
            branch="judge",
            request_kind="judge",
            system="judge-system",
            user_content="Question and evidence",
            content='{"ans',
            prompt_hash="4" * 64,
            finish_reason="length",
        )
    )
    completed = asdict(
        _request_trace(
            branch="judge",
            request_kind="judge",
            system="judge-system",
            user_content="Question and evidence",
            content='{"answer":"A"}',
            prompt_hash="4" * 64,
            attempt_index=1,
        )
    )
    trajectory["request_trace"] = [truncated, completed]
    records = build_sft_records(trajectory)
    assert len(records) == 1
    assert records[0]["metadata"]["turn_index"] == 1
    assert records[0]["messages"][-1]["content"] == '{"answer":"A"}'

    completed["content"] = "Option A seems likely."
    trajectory["request_trace"] = [completed]
    with pytest.raises(ValueError, match="neither strict answer nor tool"):
        build_sft_records(trajectory)


def test_request_episode_requires_a_valid_official_frame_tool_call() -> None:
    trajectory = _trajectory("schedule-0", 0)
    trajectory.pop("training_messages")
    trajectory["evidence_memory"] = []
    planner = asdict(
        _request_trace(
            branch="evidence",
            request_kind="planner",
            system="planner-system",
            user_content="Plan the next evidence request.",
            content=(
                '<tool_call>{"tool":"frame_select","arguments":'
                '{"start_time":0,"end_time":10,"nframes":8,"resize":0.5}}'
                "</tool_call>"
            ),
            prompt_hash="a" * 64,
        )
    )
    judge = asdict(
        _request_trace(
            branch="judge",
            request_kind="judge",
            system="judge-system",
            user_content="Answer from the evidence.",
            content='{"answer":"A"}',
            prompt_hash="b" * 64,
        )
    )
    trajectory["request_trace"] = [planner, judge]
    assert [
        record["metadata"]["episode_target_type"]
        for record in build_sft_records(trajectory)
    ] == ["tool", "final"]

    planner["content"] = '<tool_call>{"tool":"frame_select"}</tool_call>'
    trajectory["request_trace"] = [planner, judge]
    with pytest.raises(ValueError, match="arguments must be an object"):
        build_sft_records(trajectory)


def test_a4_exports_only_terminal_arbiter_answer_not_branch_hypotheses() -> None:
    trajectory = _trajectory("schedule-0", 0)
    trajectory.pop("training_messages")
    trajectory["evidence_memory"] = []
    requests = [
        _request_trace(
            branch="direct",
            request_kind="direct",
            system="direct-system",
            user_content="Answer from the complete video.",
            content='{"answer":"B"}',
            prompt_hash="8" * 64,
        ),
        _request_trace(
            branch="evidence",
            request_kind="observer",
            system="observer-system",
            user_content="Summarize the selected frames.",
            content='{"observed_facts":["the door opens"]}',
            prompt_hash="9" * 64,
        ),
        _request_trace(
            branch="evidence",
            request_kind="judge",
            system="evidence-judge-system",
            user_content="Answer from the collected evidence.",
            content='{"answer":"C"}',
            prompt_hash="c" * 64,
        ),
        _request_trace(
            branch="arbiter",
            request_kind="judge",
            system="arbiter-system",
            user_content="Resolve the independent branch disagreement.",
            content='{"answer":"A"}',
            prompt_hash="d" * 64,
        ),
    ]
    trajectory["request_trace"] = [asdict(request) for request in requests]

    records = build_sft_records(trajectory)

    assert [record["metadata"]["episode_target_type"] for record in records] == [
        "memory",
        "final",
    ]
    trainable_outputs = [
        message["content"]
        for record in records
        for message in record["messages"]
        if message.get("loss") is True
    ]
    assert trainable_outputs == [
        '{"observed_facts":["the door opens"]}',
        '{"answer":"A"}',
    ]
    assert '{"answer":"B"}' not in trainable_outputs
    assert '{"answer":"C"}' not in trainable_outputs


def test_build_and_check_consume_evaluate_mcq_agenttrace_jsonl_directly(
    tmp_path: Path,
) -> None:
    manifest_path = tmp_path / "train.jsonl"
    manifest_path.write_text(
        json.dumps(
            {"dataset": "lvbench", "sample_id": "video:question/1", "answer": "A"}
        )
        + "\n",
        encoding="utf-8",
    )
    trajectory = _trajectory("schedule-0", 0)
    trajectory.pop("training_messages")
    trajectory["manifest_sha256"] = sha256_file(manifest_path)
    trajectory["train600_manifest_sha256"] = sha256_file(manifest_path)
    observer = asdict(
        _request_trace(
            branch="evidence",
            request_kind="observer",
            system="observer-system",
            user_content="Frame observation request",
            content='{"observed_facts":["door opens"]}',
            prompt_hash="5" * 64,
        )
    )
    judge = asdict(
        _request_trace(
            branch="judge",
            request_kind="judge",
            system="judge-system",
            user_content="Question and evidence",
            content='{"answer":"A"}',
            prompt_hash="6" * 64,
        )
    )
    trajectory["request_trace"] = [observer, judge]
    trajectory["total_tokens"] = 26
    trajectory["evidence_memory"] = [
        {
            "source": "storyboard_overview",
            "interval": [0, 10],
            "timestamps": [5],
            "content": observer["content"],
        }
    ]
    trajectory["judge_confirmations"] = [
        {"judge_seed": seed, "prediction": "A", "annotation_leak_check": "passed"}
        for seed in (17, 42, 73)
    ]
    trace_path = tmp_path / "evaluate_mcq_agent.jsonl"
    trace_path.write_text(json.dumps(trajectory) + "\n", encoding="utf-8")
    output = tmp_path / "sft"
    summary = build(
        phase="select",
        train_manifest_path=manifest_path,
        trajectory_paths=[trace_path],
        output_dir=output,
        config_sha256=CONFIG_HASH,
        expected_provenance=_provenance(),
        resume=False,
        expected_counts={"lvbench": 1},
        expected_schedules=1,
    )
    assert summary["outputs"]["selected_trajectories"] == 1
    assert summary["outputs"]["sft_records"] == 2
    rows = [json.loads(line) for line in (output / "sft.jsonl").read_text().splitlines()]
    assert [row["metadata"]["episode_target_type"] for row in rows] == [
        "memory",
        "final",
    ]
    report = validate_data(
        train_manifest_path=manifest_path,
        selected_path=output / "selected.jsonl",
        sft_path=output / "sft.jsonl",
        config_sha256=CONFIG_HASH,
        state_path=output / "select_state.json",
        expected_counts={"lvbench": 1},
    )
    assert report["status"] == "passed"
    assert report["sft_count"] == 2
    assert report["episode_target_counts"] == {"final": 1, "memory": 1}

    memory_only = tmp_path / "memory_only.jsonl"
    memory_only.write_text(json.dumps(rows[0]) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="exactly one final target"):
        validate_data(
            train_manifest_path=manifest_path,
            selected_path=output / "selected.jsonl",
            sft_path=memory_only,
            config_sha256=CONFIG_HASH,
            expected_counts={"lvbench": 1},
        )


def test_qwen9b_training_launcher_hard_gates_model_and_real_loss_mask() -> None:
    text = Path("scripts/train_qwen_agent_9b_lora.sh").read_text(encoding="utf-8")
    assert "EXPECTED_MODEL_ARTIFACT_SHA256" in text
    assert "fingerprint_model_artifact.py" in text
    assert "verify_swift_loss_mask.py" in text
    assert "--max_length 16384" in text
    assert "CUDA_VISIBLE_DEVICES:-4,5,6,7" in text
