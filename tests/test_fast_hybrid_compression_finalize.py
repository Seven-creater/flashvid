from __future__ import annotations

from copy import deepcopy
import importlib.util
import json
from pathlib import Path

import pytest

from flashvid_eval.fast_hybrid_trajectory_control import (
    analyze_compression_replays,
    finalize_compression_dags,
    generate_compression_replay_specs,
)


ANSWER = {("lsdbench", "one"): "B"}

SCRIPT = Path(__file__).parents[1] / "scripts" / "finalize_fast_hybrid_compression.py"
SPEC = importlib.util.spec_from_file_location("finalize_fast_hybrid_compression", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
finalizer_script = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(finalizer_script)


def _base(*, tool_count: int = 1) -> dict:
    steps = [
        {
            "start_time": float(index * 20),
            "end_time": float(index * 20 + 10),
            "nframes": 8,
            "resize": 1.0,
            "actual_timestamps": [float(index * 20 + offset) for offset in range(8)],
            "frame_paths": [f"/frames/base-{index}-{offset}.png" for offset in range(8)],
        }
        for index in range(tool_count)
    ]
    return {
        "dataset": "lsdbench",
        "sample_id": "one",
        "schedule_id": "budget_006000_seed_17",
        "variant_id": "base",
        "family_id": "budget_006000_seed_17",
        "trajectory_id": "lsdbench:one:budget_006000_seed_17:0",
        "manifest_sha256": "a" * 64,
        "train600_manifest_sha256": "a" * 64,
        "dataset_manifest_sha256": "b" * 64,
        "config_sha256": "c" * 64,
        "scoring_deferred": True,
        "annotation_leak_check": "passed",
        "candidate_rerun": 0,
        "candidate_answer": "A",
        "prediction": "B",
        "final_prediction": "B",
        "fallback_to_candidate": False,
        "tool_steps": steps,
        "end_to_end_total_tokens": 1_000,
        "end_to_end_total_tokens_complete": True,
        "end_to_end_visual_tokens": 800,
        "end_to_end_visual_tokens_complete": True,
        "end_to_end_latency_s": 10.0,
        "_selection_stable": True,
        "_selection_confirmation_count": 3,
    }


def _replicas(spec: dict, *, costs: tuple[int, int, int] = (110, 100, 120)) -> list[dict]:
    calls = []
    for call_index, planned in enumerate(spec["planned_calls"]):
        nframes = int(planned["nframes"])
        calls.append(
            {
                "start_time": planned["start_time"],
                "end_time": planned["end_time"],
                "nframes": nframes,
                "resize": planned["resize"],
                "timestamps": [float(call_index * 100 + i) for i in range(nframes)],
                "actual_timestamps": [
                    float(call_index * 100 + i) for i in range(nframes)
                ],
                "frame_paths": [
                    f"/frames/{spec['variant_id']}-{call_index}-{i}.png"
                    for i in range(nframes)
                ],
            }
        )
    rows = []
    for replica, (seed, cost) in enumerate(zip((17, 42, 73), costs)):
        rows.append(
            {
                "dataset": spec["dataset"],
                "sample_id": spec["sample_id"],
                "schedule_id": spec["schedule_id"],
                "variant_id": spec["variant_id"],
                "family_id": spec["family_id"],
                "trajectory_id": spec["replica_trajectory_ids"][replica],
                "base_trajectory_id": spec["base_trajectory_id"],
                "counterfactual_fingerprint": spec["counterfactual_fingerprint"],
                "replica_id": str(replica),
                "judge_seed": seed,
                "scoring_deferred": True,
                "annotation_leak_check": "passed",
                "candidate_rerun": 0,
                "candidate_cost_complete": True,
                "agent_total_tokens_complete": False,
                "agent_visual_tokens_complete": True,
                "candidate_answer": "A",
                "prediction": "B",
                "final_prediction": "B",
                "fallback_to_candidate": False,
                "planned_calls": deepcopy(spec["planned_calls"]),
                "planned_calls_completed": True,
                "strict_replay_violation": None,
                "tool_steps": deepcopy(calls),
                "request_trace": [{"content": "Answer: B"}],
                "error": None,
                "end_to_end_total_tokens": cost,
                "end_to_end_total_tokens_complete": False,
                "end_to_end_visual_tokens": cost - 10,
                "end_to_end_visual_tokens_complete": True,
                "end_to_end_latency_s": float(replica + 1),
            }
        )
    return rows


def test_replay_gate_requires_exact_three_and_chooses_median_replica() -> None:
    spec = generate_compression_replay_specs(_base())[0]
    rows = _replicas(spec)

    partial = analyze_compression_replays([spec], rows[:2], ANSWER)
    assert partial.outcomes == {}
    assert partial.incomplete_fingerprints == (spec["counterfactual_fingerprint"],)

    complete = analyze_compression_replays([spec], rows, ANSWER)
    assert complete.outcomes == {spec["counterfactual_fingerprint"]: "passed"}
    representative = complete.representatives[0]
    assert representative["trajectory_id"] == rows[0]["trajectory_id"]
    assert representative["_selection_median_end_to_end_visual_tokens"] == 100
    assert representative["_selection_cost_basis"] == (
        "actual_end_to_end_visual_tokens_then_tool_calls_latency"
    )
    assert representative["_selection_stable"] is True


def test_replay_gate_rejects_an_observed_schedule_different_from_plan() -> None:
    spec = generate_compression_replay_specs(_base())[0]
    rows = _replicas(spec)
    rows[0]["tool_steps"][0]["start_time"] += 1.0
    outcome = analyze_compression_replays([spec], rows, ANSWER)
    fingerprint = spec["counterfactual_fingerprint"]
    assert outcome.outcomes[fingerprint] == "failed"
    assert outcome.rejection_reasons[fingerprint] == "schedule_mismatch"


def test_replay_representative_ignores_incomplete_total_token_lower_bound() -> None:
    spec = generate_compression_replay_specs(_base())[0]
    rows = _replicas(spec)
    for row, total, visual in zip(rows, (1, 10_000, 5_000), (300, 100, 200)):
        row["end_to_end_total_tokens"] = total
        row["end_to_end_visual_tokens"] = visual
    outcome = analyze_compression_replays([spec], rows, ANSWER)
    representative = outcome.representatives[0]
    assert representative["trajectory_id"] == rows[2]["trajectory_id"]
    assert representative["_selection_median_end_to_end_visual_tokens"] == 200


@pytest.mark.parametrize(
    ("mutation", "reason"),
    [
        (lambda row: row.update(prediction="A", final_prediction="A"), "incorrect"),
        (lambda row: row.update(fallback_to_candidate=True), "fallback"),
        (lambda row: row.update(candidate_rerun=1), "candidate_rerun"),
        (lambda row: row.update(error="API failed"), "engineering_or_parse_error"),
        (
            lambda row: row.update(end_to_end_total_tokens_complete=True),
            "forced_replay_total_cost_not_marked_incomplete",
        ),
        (
            lambda row: row.update(end_to_end_visual_tokens_complete=False),
            "visual_cost_incomplete",
        ),
        (lambda row: row.update(tool_steps=[]), "invalid_tool_trace"),
        (
            lambda row: row["tool_steps"][0].update(nframes=1),
            "invalid_tool_trace",
        ),
        (lambda row: row.update(answer="B"), "annotation_leak"),
    ],
)
def test_replay_gate_fails_closed(mutation, reason: str) -> None:
    spec = generate_compression_replay_specs(_base())[0]
    rows = _replicas(spec)
    mutation(rows[1])
    outcome = analyze_compression_replays([spec], rows, ANSWER)
    fingerprint = spec["counterfactual_fingerprint"]
    assert outcome.outcomes[fingerprint] == "failed"
    assert outcome.rejection_reasons[fingerprint] == reason
    assert outcome.representatives == ()


def test_finalizer_releases_at_most_one_node_per_sample() -> None:
    base = _base(tool_count=2)
    specs = generate_compression_replay_specs(base)
    initial = finalize_compression_dags([base], specs, [], ANSWER)
    assert initial.complete is False
    assert len(initial.ready) == 1
    assert initial.ready[0]["variant_id"] == "tail_drop_1"
    assert initial.selected_pruned == ()
    assert initial.gate["conditions"]["compression_dags_complete"] is False

    rows = _replicas(initial.ready[0])
    after_tail = finalize_compression_dags([base], specs, rows, ANSWER)
    assert len(after_tail.ready) == 1
    assert after_tail.ready[0]["variant_id"] == "early_stop_after_tail_1"


def test_completed_dag_selects_cheapest_measured_visual_cost() -> None:
    base = _base(tool_count=1)
    specs = generate_compression_replay_specs(base)
    rows: list[dict] = []
    for spec in specs:
        middle = 900 - int(spec["execution_order"]) * 100
        rows.extend(_replicas(spec, costs=(middle, middle - 10, middle + 10)))

    final = finalize_compression_dags([base], specs, rows, ANSWER)
    assert final.complete is True
    assert final.ready == ()
    assert len(final.selected_pruned) == 1
    winner = final.selected_pruned[0]
    assert winner["variant_id"].endswith("scale_025")
    assert winner["_selection_stable"] is True
    assert winner["_compression_finalized"] is True
    assert winner["_selection_cost_basis"] == (
        "actual_end_to_end_visual_tokens_then_tool_calls_latency"
    )


def test_compression_ledger_requires_monotonic_immutable_result_files(
    tmp_path: Path,
) -> None:
    first = tmp_path / "first.jsonl"
    second = tmp_path / "second.jsonl"
    first.write_text(json.dumps({"row": 1}) + "\n", encoding="utf-8")
    second.write_text(json.dumps({"row": 2}) + "\n", encoding="utf-8")
    first_hash = finalizer_script.sha256_file(first)
    second_hash = finalizer_script.sha256_file(second)
    fingerprint = "d" * 64
    previous = {
        "train600_sha256": "a" * 64,
        "base_selected_sha256": "b" * 64,
        "compression_specs_sha256": "c" * 64,
        "replay_result_sha256s": {str(first.resolve()): first_hash},
        "outcomes": {fingerprint: "failed"},
        "rejection_reasons": {fingerprint: "incorrect"},
    }
    common = {
        "train600_sha256": "a" * 64,
        "base_selected_sha256": "b" * 64,
        "compression_specs_sha256": "c" * 64,
        "outcomes": {fingerprint: "failed"},
        "rejection_reasons": {fingerprint: "incorrect"},
    }
    finalizer_script._validate_monotonic_ledger(
        previous,
        replay_result_sha256s={
            str(first.resolve()): first_hash,
            str(second.resolve()): second_hash,
        },
        **common,
    )
    with pytest.raises(RuntimeError, match="monotonic superset"):
        finalizer_script._validate_monotonic_ledger(
            previous,
            replay_result_sha256s={str(second.resolve()): second_hash},
            **common,
        )
    with pytest.raises(RuntimeError, match="outcome is not monotonic"):
        finalizer_script._validate_monotonic_ledger(
            previous,
            replay_result_sha256s={str(first.resolve()): first_hash},
            **{**common, "outcomes": {fingerprint: "passed"}},
        )
    with pytest.raises(RuntimeError, match="previous replay result changed"):
        finalizer_script._validate_monotonic_ledger(
            previous,
            replay_result_sha256s={str(first.resolve()): "e" * 64},
            **common,
        )
