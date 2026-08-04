from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "control_fast_hybrid_trajectories.py"
SPEC = importlib.util.spec_from_file_location(
    "control_fast_hybrid_trajectories_script", SCRIPT
)
assert SPEC is not None and SPEC.loader is not None
control = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = control
SPEC.loader.exec_module(control)


def _confirmations(answer: str) -> list[dict[str, object]]:
    return [
        {
            "judge_seed": seed,
            "prediction": answer,
            "annotation_leak_check": "passed",
            "fallback_used": False,
            "fallback_to_candidate": False,
            "error": None,
            "api_error": None,
            "frame_error": None,
            "parse_error": None,
        }
        for seed in (17, 42, 73)
    ]


def _row(trajectory_id: str, *, complete_cost: bool = True) -> dict[str, object]:
    return {
        "trajectory_id": trajectory_id,
        "dataset": "lvbench",
        "sample_id": "sample-1",
        "prediction": "B",
        "final_prediction": "B",
        "candidate_answer": "A",
        "candidate_rerun": 0,
        "fallback_to_candidate": False,
        "annotation_leak_check": "passed",
        "error": None,
        "end_to_end_total_tokens": 1000,
        "end_to_end_total_tokens_complete": complete_cost,
        "end_to_end_visual_tokens": 600,
        "end_to_end_visual_tokens_complete": complete_cost,
        "end_to_end_latency_s": 1.0,
        "tool_steps": [
            {
                "start_time": 0.0,
                "end_time": 10.0,
                "nframes": 2,
                "resize": 1.0,
                "actual_timestamps": [0.0, 9.9],
            }
        ],
        "judge_confirmations": _confirmations("B"),
    }


def test_cost_filter_rejects_incomplete_positive_without_aborting() -> None:
    incomplete = _row("lvbench:sample-1:incomplete", complete_cost=False)
    complete = _row("lvbench:sample-1:complete")

    outcome = control._select_complete_cost_positives(
        [incomplete, complete], {("lvbench", "sample-1"): "B"}
    )

    assert [row["trajectory_id"] for row in outcome.selected] == [
        "lvbench:sample-1:complete"
    ]
    assert outcome.rejected == {"end_to_end_total_cost_incomplete": 1}


def test_cost_filter_still_rejects_duplicate_ids() -> None:
    row = _row("lvbench:sample-1:duplicate", complete_cost=False)

    try:
        control._select_complete_cost_positives(
            [row, row], {("lvbench", "sample-1"): "B"}
        )
    except ValueError as error:
        assert "duplicate trajectory_id" in str(error)
    else:
        raise AssertionError("duplicate trajectory ID was accepted")
