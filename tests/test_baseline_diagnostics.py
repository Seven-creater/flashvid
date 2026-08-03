from __future__ import annotations

import pytest

from flashvid_eval.baseline_diagnostics import (
    DEFAULT_DURATION_BUCKET_EDGES_S,
    DIRECT_SAMPLING_SPECS,
    OPTION_PERMUTATION_SEEDS,
    DirectSamplingSpec,
    VideoDiagnosticItem,
    assign_mismatched_videos,
    duration_bucket,
    format_choices_only,
    permute_choices,
)


def test_choices_only_prompt_is_stable_and_contains_no_question_or_media() -> None:
    prompt = format_choices_only({"B": "second", "A": "first"})
    assert "A. first\nB. second" in prompt
    assert "question text" not in prompt
    assert "video_url" not in prompt
    assert prompt.endswith('{"answer":"X"}')


def test_fixed_option_permutations_are_deterministic_and_remappable() -> None:
    choices = {"A": "alpha", "B": "beta", "C": "gamma", "D": "delta"}
    assert OPTION_PERMUTATION_SEEDS == (17, 42, 73)
    for seed in OPTION_PERMUTATION_SEEDS:
        first = permute_choices(choices, seed)
        second = permute_choices(choices, seed)
        assert first == second
        assert first.choices != choices
        assert set(first.choices.values()) == set(choices.values())
        for displayed, original in first.displayed_to_original.items():
            assert first.remap_prediction(displayed) == original
    assert permute_choices(choices, 17) != permute_choices(choices, 42)


def test_mismatched_videos_stay_in_dataset_and_duration_bucket() -> None:
    assert DEFAULT_DURATION_BUCKET_EDGES_S == (
        0.0,
        300.0,
        900.0,
        1800.0,
        3600.0,
        7200.0,
        float("inf"),
    )
    items = [
        VideoDiagnosticItem("lvbench", "1", "a.mp4", 400.0),
        VideoDiagnosticItem("lvbench", "2", "b.mp4", 500.0),
        VideoDiagnosticItem("lvbench", "3", "c.mp4", 5000.0),
        VideoDiagnosticItem("lsdbench", "4", "d.mp4", 450.0),
        VideoDiagnosticItem("lsdbench", "5", "e.mp4", 550.0),
    ]
    assignments = assign_mismatched_videos(items, seed=42)
    by_id = {(item.dataset, item.sample_id): item for item in items}

    assert {(item.dataset, item.sample_id) for item in assignments} == {
        ("lvbench", "1"),
        ("lvbench", "2"),
        ("lsdbench", "4"),
        ("lsdbench", "5"),
    }
    for assignment in assignments:
        source = by_id[(assignment.dataset, assignment.sample_id)]
        target = next(
            item
            for item in items
            if item.dataset == assignment.dataset
            and item.video == assignment.target_video
        )
        assert assignment.source_video != assignment.target_video
        assert duration_bucket(source.duration_s) == duration_bucket(target.duration_s)


def test_direct_sampling_specs_are_mutually_exclusive() -> None:
    assert DIRECT_SAMPLING_SPECS["uniform32"].mm_processor_kwargs() == {
        "do_sample_frames": False,
    }
    assert DIRECT_SAMPLING_SPECS["uniform64"].media_io_kwargs(100.0) == {
        "video": {
            "num_frames": -1,
            "fps": 0.64,
            "min_frames": 64,
            "max_frames": 64,
        }
    }
    assert DIRECT_SAMPLING_SPECS["uniform128"].media_io_kwargs(64.0)["video"][
        "fps"
    ] == 2.0
    assert DIRECT_SAMPLING_SPECS["fps2"].media_io_kwargs(100.0) == {
        "video": {
            "num_frames": -1,
            "fps": 2.0,
            "min_frames": 4,
            "max_frames": 768,
        }
    }
    with pytest.raises(ValueError, match="duration"):
        DIRECT_SAMPLING_SPECS["uniform32"].media_io_kwargs(0)
    with pytest.raises(ValueError, match="exactly one"):
        DirectSamplingSpec("invalid")
    with pytest.raises(ValueError, match="exactly one"):
        DirectSamplingSpec("invalid", num_frames=32, fps=2.0)
    with pytest.raises(ValueError, match="only valid for fps"):
        DirectSamplingSpec("invalid", num_frames=32, max_frames=64)
