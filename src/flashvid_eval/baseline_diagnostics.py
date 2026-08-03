from __future__ import annotations

import hashlib
import math
import random
from dataclasses import dataclass
from typing import Iterable, Mapping


OPTION_PERMUTATION_SEEDS = (17, 42, 73)
DEFAULT_DURATION_BUCKET_EDGES_S = (
    0.0,
    300.0,
    900.0,
    1800.0,
    3600.0,
    7200.0,
    math.inf,
)


def _ordered_choices(choices: Mapping[str, str]) -> tuple[tuple[str, str], ...]:
    return tuple(sorted((str(key).upper(), str(value)) for key, value in choices.items()))


def format_choices_only(choices: Mapping[str, str]) -> str:
    """Build the deterministic hard-guess control without a question or media."""

    options = "\n".join(f"{letter}. {text}" for letter, text in _ordered_choices(choices))
    return (
        "The question and video are intentionally unavailable. Guess the most "
        "likely answer using only the option texts below.\n"
        f"{options}\n"
        'Return exactly one JSON object: {"answer":"X"}'
    )


@dataclass(frozen=True)
class OptionPermutation:
    seed: int
    choices: dict[str, str]
    displayed_to_original: dict[str, str]

    def remap_prediction(self, prediction: str | None) -> str | None:
        if prediction is None:
            return None
        return self.displayed_to_original.get(str(prediction).strip().upper())


def permute_choices(
    choices: Mapping[str, str],
    seed: int,
) -> OptionPermutation:
    """Relabel option texts using a fixed, deterministic permutation."""

    ordered = _ordered_choices(choices)
    displayed_letters = [letter for letter, _ in ordered]
    original_letters = displayed_letters.copy()
    shuffled_originals = original_letters.copy()
    random.Random(seed).shuffle(shuffled_originals)
    if len(shuffled_originals) > 1 and shuffled_originals == original_letters:
        shuffled_originals = shuffled_originals[1:] + shuffled_originals[:1]
    displayed_to_original = dict(zip(displayed_letters, shuffled_originals))
    choice_text = dict(ordered)
    return OptionPermutation(
        seed=seed,
        choices={
            displayed: choice_text[original]
            for displayed, original in displayed_to_original.items()
        },
        displayed_to_original=displayed_to_original,
    )


@dataclass(frozen=True)
class VideoDiagnosticItem:
    dataset: str
    sample_id: str
    video: str
    duration_s: float


@dataclass(frozen=True)
class MismatchedVideoAssignment:
    dataset: str
    sample_id: str
    source_video: str
    target_video: str
    duration_bucket: int


def duration_bucket(
    duration_s: float,
    edges_s: tuple[float, ...] = DEFAULT_DURATION_BUCKET_EDGES_S,
) -> int:
    if not math.isfinite(duration_s) or duration_s < 0:
        raise ValueError("duration_s must be finite and non-negative")
    if len(edges_s) < 2 or edges_s[0] != 0.0 or edges_s[-1] != math.inf:
        raise ValueError("duration bucket edges must start at 0 and end at infinity")
    if any(left >= right for left, right in zip(edges_s, edges_s[1:])):
        raise ValueError("duration bucket edges must be strictly increasing")
    for index, (left, right) in enumerate(zip(edges_s, edges_s[1:])):
        if left <= duration_s < right:
            return index
    raise AssertionError("duration was not assigned to a bucket")


def assign_mismatched_videos(
    items: Iterable[VideoDiagnosticItem],
    *,
    seed: int = 42,
    edges_s: tuple[float, ...] = DEFAULT_DURATION_BUCKET_EDGES_S,
) -> tuple[MismatchedVideoAssignment, ...]:
    """Assign another video from the same dataset and duration bucket.

    Buckets with fewer than two distinct videos are deliberately omitted rather
    than silently borrowing a less comparable video from another bucket.
    """

    groups: dict[tuple[str, int], list[VideoDiagnosticItem]] = {}
    for item in items:
        bucket = duration_bucket(float(item.duration_s), edges_s)
        groups.setdefault((item.dataset, bucket), []).append(item)

    assignments: list[MismatchedVideoAssignment] = []
    for (dataset, bucket), group in sorted(groups.items()):
        videos = sorted({item.video for item in group})
        if len(videos) < 2:
            continue
        group_seed = hashlib.sha256(
            f"{seed}\0{dataset}\0{bucket}".encode("utf-8")
        ).digest()
        random.Random(int.from_bytes(group_seed[:8], "big")).shuffle(videos)
        target_by_source = {
            source: videos[(index + 1) % len(videos)]
            for index, source in enumerate(videos)
        }
        for item in sorted(group, key=lambda value: value.sample_id):
            assignments.append(
                MismatchedVideoAssignment(
                    dataset=dataset,
                    sample_id=item.sample_id,
                    source_video=item.video,
                    target_video=target_by_source[item.video],
                    duration_bucket=bucket,
                )
            )
    return tuple(assignments)


@dataclass(frozen=True)
class DirectSamplingSpec:
    sampling_id: str
    num_frames: int | None = None
    fps: float | None = None
    max_frames: int | None = None

    def __post_init__(self) -> None:
        if (self.num_frames is None) == (self.fps is None):
            raise ValueError("exactly one of num_frames or fps must be set")
        if self.num_frames is not None and self.num_frames <= 0:
            raise ValueError("num_frames must be positive")
        if self.fps is not None and self.fps <= 0:
            raise ValueError("fps must be positive")
        if self.max_frames is not None and self.max_frames <= 0:
            raise ValueError("max_frames must be positive")
        if self.num_frames is not None and self.max_frames is not None:
            raise ValueError("max_frames is only valid for fps sampling")

    def mm_processor_kwargs(self) -> dict[str, int | float | bool]:
        values: dict[str, int | float | bool] = {"do_sample_frames": True}
        if self.num_frames is not None:
            values["num_frames"] = self.num_frames
        else:
            values["fps"] = float(self.fps)
            if self.max_frames is not None:
                values["max_frames"] = self.max_frames
        return values


DIRECT_SAMPLING_SPECS = {
    "uniform32": DirectSamplingSpec("uniform32", num_frames=32),
    "uniform64": DirectSamplingSpec("uniform64", num_frames=64),
    "uniform128": DirectSamplingSpec("uniform128", num_frames=128),
    "fps2": DirectSamplingSpec("fps2", fps=2.0, max_frames=768),
}
