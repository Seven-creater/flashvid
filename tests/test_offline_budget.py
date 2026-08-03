from __future__ import annotations

import json
from pathlib import Path

import pytest

from flashvid_eval.offline_budget import (
    build_sft_record,
    normalize_candidate,
    retained_visual_tokens,
    select_training_trajectories,
    split_samples_by_video,
    video_group_key,
    write_sft_jsonl,
)
from flashvid_eval.schemas import Sample


def _sample(sample_id: str, video: str, question: str = "What happens?") -> Sample:
    return Sample(
        dataset="demo",
        sample_id=sample_id,
        video=video,
        question=question,
        choices={"A": "first", "B": "second"},
        answer="A",
    )


def _messages(answer: str = "A") -> list[dict[str, object]]:
    return [
        {"role": "system", "content": "Use visual observations only."},
        {"role": "user", "content": "Question: What happens?\nA: first\nB: second"},
        {"role": "assistant", "content": "<tool_call>{\"start_time\":0}</tool_call>"},
        {"role": "tool", "content": "{\"observed_facts\":[\"first\"]}"},
        {"role": "assistant", "content": f"Answer: {answer}"},
    ]


def _trajectory(
    sample_id: str,
    trajectory_id: str,
    *,
    prediction: str = "A",
    candidate: str = "B",
    cost: float = 100,
    ratio: float = 0.1,
    calls: int = 1,
    latency: float = 1.0,
    error: str | None = None,
) -> dict[str, object]:
    return {
        "dataset": "demo",
        "sample_id": sample_id,
        "trajectory_id": trajectory_id,
        "final_prediction": prediction,
        "candidate_answer": candidate,
        "retained_visual_tokens": cost,
        "budget_sequence": [ratio],
        "tool_steps": [
            {"retention_ratio": ratio, "retained_visual_tokens": cost / calls}
            for _ in range(calls)
        ],
        "latency_s": latency,
        "messages": _messages(prediction),
        "annotation_leak_check": "passed",
        "error": error,
    }


def test_candidate_normalization_is_conservative_and_deterministic() -> None:
    choices = {"A": "The red door opens.", "B": "The blue door closes."}
    assert normalize_candidate("A", "", choices).answer == "A"
    strict = normalize_candidate(None, "Explanation.\nAnswer: B", choices)
    assert (strict.answer, strict.source) == ("B", "parsed")
    matched = normalize_candidate(None, "The red door opens.", choices)
    assert (matched.answer, matched.source) == ("A", "normalized")
    assert normalize_candidate(None, "I considered the red door.", choices).answer is None
    assert normalize_candidate(None, "same", {"A": "same", "B": "same"}).answer is None


def test_video_split_is_exact_deterministic_and_video_disjoint() -> None:
    candidates = [
        _sample(f"{video}-{index}", f"{video}.mp4", "What is written on screen?" if index else "What happens after this?")
        for video in ("test", "v1", "v2", "v3", "v4", "v5")
        for index in range(2)
    ]
    frozen = [_sample("frozen", "nested/test.mkv")]
    first = split_samples_by_video(candidates, frozen, train_count=4, dev_count=3, seed=42)
    second = split_samples_by_video(list(reversed(candidates)), frozen, train_count=4, dev_count=3, seed=42)

    assert [sample.sample_id for sample in first.train] == [sample.sample_id for sample in second.train]
    assert [sample.sample_id for sample in first.dev] == [sample.sample_id for sample in second.dev]
    assert len(first.train) == 4
    assert len(first.dev) == 3
    train_videos = {video_group_key(sample) for sample in first.train}
    dev_videos = {video_group_key(sample) for sample in first.dev}
    assert train_videos.isdisjoint(dev_videos)
    assert all("test" not in video_group_key(sample) for sample in (*first.train, *first.dev))
    assert set(first.excluded_test_sample_ids) == {"test-0", "test-1"}


def test_video_split_rejects_impossible_group_partition() -> None:
    samples = [
        *[_sample(f"a-{index}", "a.mp4") for index in range(4)],
        _sample("b-0", "b.mp4"),
    ]
    with pytest.raises(ValueError, match="video grouping"):
        split_samples_by_video(samples, [], train_count=3, dev_count=2)


def test_trajectory_selection_uses_cost_tie_breaks_and_optional_second_trace() -> None:
    trajectories = [
        _trajectory("s1", "error", cost=1, error="api failed"),
        _trajectory("s1", "two-calls", cost=100, ratio=0.1, calls=2, latency=1),
        _trajectory("s1", "primary", cost=100, ratio=0.1, calls=1, latency=3),
        _trajectory("s1", "secondary", cost=110, ratio=0.5, calls=1, latency=1),
        _trajectory("s1", "too-expensive", cost=121, ratio=1.0),
        _trajectory("s2", "wrong", prediction="B", candidate="B", cost=10),
        _trajectory("s3", "candidate-correct", candidate="A", cost=50),
        _trajectory("unknown", "unknown", cost=1),
    ]
    result = select_training_trajectories(
        trajectories,
        {"s1": "A", "s2": "A", "s3": "A"},
    )
    selected_ids = [
        (record["trajectory_id"], record["_selection_role"])
        for record in result.selected
    ]
    assert selected_ids == [
        ("primary", "primary"),
        ("secondary", "secondary_changed_candidate"),
        ("candidate-correct", "primary"),
    ]
    assert result.no_positive_sample_ids == ("s2",)
    assert result.unknown_sample_ids == ("unknown",)


def test_retained_visual_tokens_can_be_summed_from_steps() -> None:
    assert retained_visual_tokens(
        {
            "tool_steps": [
                {"retained_visual_tokens": 12},
                {"retained_visual_tokens": 8.5},
            ]
        }
    ) == 20.5


def test_sft_export_keeps_messages_but_drops_labels_and_hidden_reasoning(tmp_path: Path) -> None:
    trajectory = _trajectory("s1", "primary")
    trajectory["training_messages"] = trajectory.pop("messages")
    trajectory.update(
        {
            "answer": "A",
            "time_range": [1, 2],
            "_selection_role": "primary",
        }
    )
    trajectory["training_messages"][2]["reasoning_content"] = "private chain of thought"
    record = build_sft_record(trajectory)
    serialized = json.dumps(record)
    assert "time_range" not in serialized
    assert "reasoning_content" not in serialized
    assert record["metadata"] == {
        "dataset": "demo",
        "sample_id": "s1",
        "trajectory_id": "primary",
        "selection_role": "primary",
        "retained_visual_tokens": 100.0,
    }
    assert record["messages"][-1]["content"] == "Answer: A"

    output = tmp_path / "sft.jsonl"
    assert write_sft_jsonl([trajectory], output) == 1
    assert json.loads(output.read_text(encoding="utf-8")) == record


@pytest.mark.parametrize(
    "content",
    [
        '{"time_range":[1,2]}',
        "ground_truth: B",
        {"clue_intervals": [[1, 2]]},
    ],
)
def test_sft_export_rejects_private_fields_in_model_inputs(content: object) -> None:
    trajectory = _trajectory("s1", "leak")
    trajectory["messages"][3]["content"] = content
    with pytest.raises(ValueError, match="private benchmark"):
        build_sft_record(trajectory)
