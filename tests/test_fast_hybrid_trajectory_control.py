from __future__ import annotations

from collections import Counter

import pytest

from flashvid_eval.fast_hybrid_trajectory_control import (
    build_base_run_specs,
    build_rescue_run_specs,
    compression_execution_state,
    controller_fingerprint,
    generate_compression_replay_specs,
    pending_run_specs,
    positive_rejection_reason,
    prepare_prejudge_candidates,
    select_lowest_cost_positives,
    sft_start_gate,
    validate_prejudge_coverage,
)


SHA_A = "a" * 64
SHA_B = "b" * 64
DATASET_HASHES = {
    "lvbench": "c" * 64,
    "lsdbench": "d" * 64,
    "cgbench": "e" * 64,
}


def _manifest_rows() -> list[dict]:
    return [
        {"dataset": "lvbench", "sample_id": "l1"},
        {"dataset": "lsdbench", "sample_id": "s1"},
    ]


def _confirmations(answer: str) -> list[dict]:
    return [
        {
            "judge_seed": seed,
            "prediction": answer,
            "annotation_leak_check": "passed",
            "fallback_used": False,
            "error": None,
        }
        for seed in (17, 42, 73)
    ]


def _trajectory(
    *,
    dataset: str = "lvbench",
    sample_id: str = "l1",
    trajectory_id: str = "lvbench:l1:budget:0",
    prediction: str = "B",
    candidate: str = "A",
    total_tokens: int = 1000,
    visual_tokens: int = 600,
) -> dict:
    return {
        "schema_version": 1,
        "phase": "base",
        "dataset": dataset,
        "sample_id": sample_id,
        "schedule_id": "budget",
        "variant_id": "base",
        "family_id": "budget",
        "replica_id": "0",
        "trajectory_id": trajectory_id,
        "manifest_sha256": SHA_A,
        "train600_manifest_sha256": SHA_A,
        "dataset_manifest_sha256": DATASET_HASHES.get(dataset, SHA_A),
        "config_sha256": SHA_B,
        "controller_fingerprint": SHA_A,
        "run_spec_fingerprint": SHA_B,
        "prediction": prediction,
        "final_prediction": prediction,
        "candidate_answer": candidate,
        "candidate_rerun": 0,
        "fallback_to_candidate": False,
        "annotation_leak_check": "passed",
        "error": None,
        "total_tokens": total_tokens,
        "visual_tokens": visual_tokens,
        "latency_s": 1.0,
        "end_to_end_total_tokens": total_tokens,
        "end_to_end_visual_tokens": visual_tokens,
        "end_to_end_latency_s": 1.0,
        "tool_steps": [
            {
                "start_time": 0.0,
                "end_time": 10.0,
                "nframes": 8,
                "resize": 1.0,
                "actual_timestamps": [0.0, 5.0, 9.9],
            },
            {
                "start_time": 20.0,
                "end_time": 30.0,
                "nframes": 4,
                "resize": 0.8,
                "actual_timestamps": [20.0, 25.0, 29.9],
            },
        ],
        "judge_confirmations": _confirmations(prediction),
    }


def test_base_plan_is_four_budgets_by_three_seeds_and_resume_bound() -> None:
    specs = build_base_run_specs(
        _manifest_rows(),
        manifest_sha256=SHA_A,
        config_sha256=SHA_B,
        dataset_manifest_sha256s=DATASET_HASHES,
    )
    assert len(specs) == 24
    assert Counter(spec["sample_id"] for spec in specs) == {"l1": 12, "s1": 12}
    assert len({spec["trajectory_id"] for spec in specs}) == 24
    assert {spec["max_total_visual_tokens"] for spec in specs} == {
        6000,
        12000,
        18000,
        24000,
    }
    assert {spec["planner_seed"] for spec in specs} == {17, 42, 73}
    assert all(spec["required_judge_seeds"] == [17, 42, 73] for spec in specs)

    fingerprint = specs[0]["controller_fingerprint"]
    completed = [{**specs[0]}]
    pending = pending_run_specs(
        specs, completed, expected_controller_fingerprint=fingerprint
    )
    assert len(pending) == 23
    assert specs[0]["trajectory_id"] not in {row["trajectory_id"] for row in pending}

    tampered = [{**specs[0], "run_spec_fingerprint": "0" * 64}]
    with pytest.raises(RuntimeError, match="run-spec fingerprint mismatch"):
        pending_run_specs(specs, tampered, expected_controller_fingerprint=fingerprint)


def test_positive_requires_visual_evidence_and_all_three_judges() -> None:
    row = _trajectory()
    assert positive_rejection_reason(row, "B") is None

    no_tool = {**row, "tool_steps": []}
    assert positive_rejection_reason(no_tool, "B") == "no_frame_select"

    unstable = {**row, "judge_confirmations": _confirmations("B")[:2]}
    assert positive_rejection_reason(unstable, "B") == "incomplete_judge_confirmation"

    wrong_candidate_fallback = {**row, "fallback_to_candidate": True}
    assert (
        positive_rejection_reason(wrong_candidate_fallback, "B")
        == "wrong_candidate_fallback"
    )

    correct_candidate_fallback = {
        **row,
        "candidate_answer": "B",
        "fallback_to_candidate": True,
    }
    assert positive_rejection_reason(correct_candidate_fallback, "B") is None


def test_prejudge_filters_offline_without_serializing_labels() -> None:
    spec = build_base_run_specs(
        [{"dataset": "lvbench", "sample_id": "l1"}],
        manifest_sha256=SHA_A,
        config_sha256=SHA_B,
        dataset_manifest_sha256s=DATASET_HASHES,
    )[0]
    raw = _trajectory()
    for private in ("answer", "correct", "question_type"):
        raw.pop(private, None)
    raw.update(
        {
            "scoring_deferred": True,
            "trajectory_schedule_id": spec["schedule_id"],
            "generation_seed": spec["planner_seed"],
            "candidate_cost_complete": True,
        }
    )
    raw.pop("judge_confirmations")

    outcome = prepare_prejudge_candidates(
        [spec], raw_trajectories=[raw], answers={("lvbench", "l1"): "B"}
    )

    assert len(outcome.eligible_specs) == 1
    assert len(outcome.eligible_trajectories) == 1
    eligible = outcome.eligible_trajectories[0]
    assert eligible["trajectory_id"] == spec["trajectory_id"]
    assert eligible["tool_steps"]
    assert "answer" not in eligible and "correct" not in eligible
    assert outcome.completion_index[0]["prejudge_status"] == "eligible"
    validate_prejudge_coverage(
        [spec], outcome.completion_index, [eligible]
    )


def test_prejudge_rejects_incorrect_before_judge() -> None:
    spec = build_base_run_specs(
        [{"dataset": "lvbench", "sample_id": "l1"}],
        manifest_sha256=SHA_A,
        config_sha256=SHA_B,
        dataset_manifest_sha256s=DATASET_HASHES,
    )[0]
    raw = _trajectory(prediction="A")
    raw.update(
        {
            "scoring_deferred": True,
            "trajectory_schedule_id": spec["schedule_id"],
            "generation_seed": spec["planner_seed"],
            "candidate_cost_complete": True,
        }
    )
    raw.pop("judge_confirmations")
    outcome = prepare_prejudge_candidates(
        [spec], raw_trajectories=[raw], answers={("lvbench", "l1"): "B"}
    )
    assert outcome.eligible_specs == ()
    assert outcome.rejected == {"incorrect": 1}
    assert outcome.completion_index[0]["prejudge_status"] == "rejected"
    validate_prejudge_coverage([spec], outcome.completion_index, [])


def test_prejudge_rejects_required_confirmation_failure() -> None:
    spec = build_base_run_specs(
        [{"dataset": "lvbench", "sample_id": "l1"}],
        manifest_sha256=SHA_A,
        config_sha256=SHA_B,
        dataset_manifest_sha256s=DATASET_HASHES,
    )[0]
    raw = _trajectory(prediction="B", candidate="B")
    raw.update(
        {
            "scoring_deferred": True,
            "trajectory_schedule_id": spec["schedule_id"],
            "generation_seed": spec["planner_seed"],
            "candidate_cost_complete": True,
            "error": "change_confirmation: no valid final answer",
            "error_type": "required_run_failure",
        }
    )
    raw.pop("judge_confirmations")

    outcome = prepare_prejudge_candidates(
        [spec], raw_trajectories=[raw], answers={("lvbench", "l1"): "B"}
    )

    assert outcome.eligible_specs == ()
    assert outcome.rejected == {"engineering_or_parse_error": 1}
    validate_prejudge_coverage([spec], outcome.completion_index, [])


def test_offline_join_selects_lowest_cost_and_removes_labels() -> None:
    # Agent-only accounting is intentionally inverted: selection must include
    # the frozen Direct candidate and rank by the complete-system cost.
    expensive = _trajectory(trajectory_id="lvbench:l1:expensive:0", total_tokens=100)
    expensive["end_to_end_total_tokens"] = 2000
    expensive["answer"] = "B"
    expensive["question_type"] = "private"
    cheap = _trajectory(trajectory_id="lvbench:l1:cheap:0", total_tokens=900)
    outcome = select_lowest_cost_positives([expensive, cheap], {("lvbench", "l1"): "B"})
    assert len(outcome.selected) == 1
    assert outcome.selected[0]["trajectory_id"] == "lvbench:l1:cheap:0"
    assert outcome.selected[0]["scoring_deferred"] is True
    assert outcome.selected[0]["_selection_stable"] is True
    assert outcome.selected[0]["_selection_cost_basis"] == "end_to_end"
    assert "answer" not in outcome.selected[0]
    assert "question_type" not in outcome.selected[0]
    assert outcome.gate["passed"] is False


def test_no_positive_sample_gets_exactly_four_deterministic_rescues() -> None:
    fingerprint = controller_fingerprint(
        manifest_sha256=SHA_A,
        config_sha256=SHA_B,
        dataset_manifest_sha256s=DATASET_HASHES,
    )
    first = build_rescue_run_specs(
        _manifest_rows(),
        no_positive_sample_ids=["lsdbench/s1"],
        manifest_sha256=SHA_A,
        config_sha256=SHA_B,
        controller_sha256=fingerprint,
        dataset_manifest_sha256s=DATASET_HASHES,
    )
    second = build_rescue_run_specs(
        _manifest_rows(),
        no_positive_sample_ids=["lsdbench/s1"],
        manifest_sha256=SHA_A,
        config_sha256=SHA_B,
        controller_sha256=fingerprint,
        dataset_manifest_sha256s=DATASET_HASHES,
    )
    assert first == second
    assert len(first) == 4
    assert {row["phase"] for row in first} == {"rescue"}
    assert len({row["trajectory_id"] for row in first}) == 4
    assert {(row["max_total_visual_tokens"], row["planner_seed"]) for row in first} == {
        (32000, 101),
        (32000, 211),
        (48000, 101),
        (48000, 211),
    }
    assert {row["max_call_visual_tokens"] for row in first} == {12000}
    assert all(row["max_turns"] == 8 for row in first)


def test_compression_specs_cover_tail_early_stop_and_three_scales() -> None:
    source = _trajectory()
    specs = generate_compression_replay_specs(source)
    by_variant = {row["variant_id"]: row for row in specs}

    assert len(specs) == 9
    tail = by_variant["tail_drop_1"]
    assert tail["stage"] == "tail_delete"
    assert tail["stage_order"] == 1
    assert tail["stop_policy"] == "replay_then_judge"

    # If the first deletion fails, early-stop runs from the untouched source.
    early_root = by_variant["early_stop_after_tail_0"]
    assert early_root["stage"] == "early_stop"
    assert early_root["stage_order"] == 2
    assert early_root["parent_fingerprint"] == tail["parent_fingerprint"]
    assert early_root["requires_failure_of"] == tail["counterfactual_fingerprint"]

    # If deletion succeeds, the early-stop node depends on that successful tail.
    early_tail = by_variant["early_stop_after_tail_1"]
    assert early_tail["parent_fingerprint"] == tail["counterfactual_fingerprint"]
    assert "requires_failure_of" not in early_tail

    # Evidence compression is a joint nframes+resize success chain, not an
    # unordered matrix of independent replay jobs.
    scale_075 = by_variant["early_stop_after_tail_1_scale_075"]
    scale_050 = by_variant["early_stop_after_tail_1_scale_050"]
    scale_025 = by_variant["early_stop_after_tail_1_scale_025"]
    assert scale_075["parent_fingerprint"] == early_tail["counterfactual_fingerprint"]
    assert scale_050["parent_fingerprint"] == scale_075["counterfactual_fingerprint"]
    assert scale_025["parent_fingerprint"] == scale_050["counterfactual_fingerprint"]
    assert scale_075["planned_calls"][0]["nframes"] == 6
    assert scale_075["planned_calls"][0]["resize"] == pytest.approx(0.75)
    assert scale_050["planned_calls"][0]["nframes"] == 4
    assert scale_050["planned_calls"][0]["resize"] == pytest.approx(0.5)
    assert scale_025["planned_calls"][0]["nframes"] == 2
    assert scale_025["planned_calls"][0]["resize"] == pytest.approx(0.25)
    assert all(row["on_dependency_failure"] == "skip_branch" for row in specs)
    assert all(
        row["execution_policy"] == "sequential_dependency_gated" for row in specs
    )
    assert all(row["parallel_safe"] is False for row in specs)
    assert [row["execution_order"] for row in specs] == list(range(1, 10))
    assert all(row["required_judge_confirmations"] == 3 for row in specs)
    assert all(row["required_judge_seeds"] == [17, 42, 73] for row in specs)
    assert len({row["counterfactual_fingerprint"] for row in specs}) == len(specs)


def test_compression_executor_releases_one_node_and_skips_failed_branch() -> None:
    specs = generate_compression_replay_specs(_trajectory())
    by_variant = {row["variant_id"]: row for row in specs}

    initial = compression_execution_state(specs, {})
    assert [row["variant_id"] for row in initial.ready] == ["tail_drop_1"]

    tail_fingerprint = by_variant["tail_drop_1"]["counterfactual_fingerprint"]
    after_tail = compression_execution_state(specs, {tail_fingerprint: "passed"})
    assert [row["variant_id"] for row in after_tail.ready] == [
        "early_stop_after_tail_1"
    ]
    assert (
        by_variant["early_stop_after_tail_0"]["counterfactual_fingerprint"]
        in after_tail.skipped_fingerprints
    )

    early_fingerprint = by_variant["early_stop_after_tail_1"][
        "counterfactual_fingerprint"
    ]
    after_early = compression_execution_state(
        specs,
        {tail_fingerprint: True, early_fingerprint: True},
    )
    assert [row["variant_id"] for row in after_early.ready] == [
        "early_stop_after_tail_1_scale_075"
    ]

    scale_fingerprint = by_variant["early_stop_after_tail_1_scale_075"][
        "counterfactual_fingerprint"
    ]
    stopped = compression_execution_state(
        specs,
        {
            tail_fingerprint: True,
            early_fingerprint: True,
            scale_fingerprint: False,
        },
    )
    assert stopped.ready == ()
    assert stopped.complete is True
    assert (
        by_variant["early_stop_after_tail_1_scale_050"]["counterfactual_fingerprint"]
        in stopped.skipped_fingerprints
    )
    assert (
        by_variant["early_stop_after_tail_1_scale_025"]["counterfactual_fingerprint"]
        in stopped.skipped_fingerprints
    )


def test_compression_executor_uses_root_branch_when_tail_delete_fails() -> None:
    specs = generate_compression_replay_specs(_trajectory())
    by_variant = {row["variant_id"]: row for row in specs}
    tail_fingerprint = by_variant["tail_drop_1"]["counterfactual_fingerprint"]

    state = compression_execution_state(specs, {tail_fingerprint: "failed"})
    assert [row["variant_id"] for row in state.ready] == ["early_stop_after_tail_0"]
    assert (
        by_variant["early_stop_after_tail_1"]["counterfactual_fingerprint"]
        in state.skipped_fingerprints
    )


def test_sft_gate_requires_300_total_and_80_per_dataset() -> None:
    passing = [
        {"dataset": dataset, "sample_id": f"{dataset}-{index}"}
        for dataset in ("lvbench", "lsdbench", "cgbench")
        for index in range(100)
    ]
    assert sft_start_gate(passing)["passed"] is True

    failing = [
        {"dataset": dataset, "sample_id": f"{dataset}-{index}"}
        for dataset, count in (("lvbench", 111), ("lsdbench", 110), ("cgbench", 79))
        for index in range(count)
    ]
    gate = sft_start_gate(failing)
    assert gate["selected_total"] == 300
    assert gate["passed"] is False
    assert gate["conditions"]["each_dataset_at_least_80"] is False
