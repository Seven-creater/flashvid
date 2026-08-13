from __future__ import annotations

import pytest

from scripts.filter_swift_sft_length import select_length_safe_trajectories


def _row(trajectory: str, target: str, *, sample: str = "1") -> dict:
    return {
        "metadata": {
            "trajectory_id": trajectory,
            "episode_id": f"{trajectory}#{target}",
            "episode_target_type": target,
            "dataset": "demo",
            "sample_id": sample,
        }
    }


def _planner_row(trajectory: str, *, sample: str = "1") -> dict:
    return {
        "metadata": {
            "trajectory_id": trajectory,
            "episode_id": f"{trajectory}#role-planner",
            "episode_schema": "planner_complete_episode_v1",
            "process_role": "planner",
            "assistant_target_types": ["tool", "plan", "tool", "stop"],
            "dataset": "demo",
            "sample_id": sample,
        }
    }


def _observer_row(trajectory: str, *, sample: str = "1") -> dict:
    return {
        "metadata": {
            "trajectory_id": trajectory,
            "episode_id": f"{trajectory}#role-observer#prefix-000",
            "episode_schema": "observer_current_frame_episode_v1",
            "process_role": "observer",
            "assistant_target_types": ["memory"],
            "dataset": "demo",
            "sample_id": sample,
        }
    }


def test_rejects_entire_trajectory_when_one_episode_is_too_long() -> None:
    rows = [
        _row("keep", "tool"),
        _row("keep", "final"),
        _row("reject", "tool", sample="2"),
        _row("reject", "final", sample="2"),
    ]

    retained, report = select_length_safe_trajectories(
        rows, [100, 200, 300, 16_385], max_length=16_384
    )

    assert retained == [0, 1]
    assert report["retained_trajectories"] == 1
    assert report["rejected_trajectories"] == 1
    assert report["rejections"][0]["trajectory_id"] == "reject"
    assert report["rejections"][0]["over_length_episodes"] == [
        {
            "row_index": 3,
            "episode_id": "reject#final",
            "encoded_tokens": 16_385,
        }
    ]


def test_requires_complete_targets_in_retained_trajectory() -> None:
    with pytest.raises(ValueError, match="at least one tool target"):
        select_length_safe_trajectories([_row("broken", "final")], [100])


def test_length_boundary_is_retained() -> None:
    rows = [_row("boundary", "tool"), _row("boundary", "final")]
    retained, report = select_length_safe_trajectories(
        rows, [16_384, 16_384], max_length=16_384
    )

    assert retained == [0, 1]
    assert report["maximum_retained_tokens"] == 16_384


def test_accepts_one_complete_role_separated_planner_episode() -> None:
    retained, report = select_length_safe_trajectories(
        [_planner_row("planner")], [1_000], max_length=16_384
    )

    assert retained == [0]
    assert report["retained_trajectories"] == 1


def test_accepts_planner_with_image_bearing_observer_probe() -> None:
    rows = [_planner_row("probe"), _observer_row("probe")]
    retained, report = select_length_safe_trajectories(
        rows, [1_000, 2_000], max_length=16_384
    )

    assert retained == [0, 1]
    assert report["retained_rows"] == 2


@pytest.mark.parametrize(
    "targets",
    (["tool"], ["stop"], ["tool", "stop", "plan"], ["tool", "final", "stop"]),
)
def test_rejects_invalid_role_separated_planner_targets(targets: list[str]) -> None:
    row = _planner_row("broken")
    row["metadata"]["assistant_target_types"] = targets

    with pytest.raises(ValueError, match="tool target.*stop target"):
        select_length_safe_trajectories([row], [100])
