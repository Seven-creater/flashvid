"""Public-question timestamp parsing shared by agent and repair runtimes."""

from __future__ import annotations

import math
import re


_TIMESTAMP = re.compile(
    r"(?<![\d:])(?:"
    r"(?P<hours>\d+):(?P<hour_minutes>[0-5]\d):(?P<hour_seconds>[0-5]\d)"
    r"|(?P<minutes>\d+):(?P<seconds>[0-5]\d)"
    r")(?![:\d])"
)


def _timestamp_seconds(match: re.Match[str]) -> float:
    if match.group("hours") is not None:
        hours = int(match.group("hours"))
        minutes = int(match.group("hour_minutes"))
        seconds = int(match.group("hour_seconds"))
    else:
        hours = 0
        minutes = int(match.group("minutes"))
        seconds = int(match.group("seconds"))
    return float(hours * 3600 + minutes * 60 + seconds)


def parse_question_time_range(
    question: str,
    *,
    padding_s: float | None = None,
) -> tuple[float, float] | None:
    """Parse timestamps present in public question text, never dataset metadata."""

    if padding_s is not None and (
        not math.isfinite(padding_s) or padding_s < 0
    ):
        raise ValueError("padding_s must be finite and non-negative")
    matches = list(_TIMESTAMP.finditer(question))
    if not matches:
        return None
    first = _timestamp_seconds(matches[0])
    if len(matches) == 1:
        point_padding = 1.0 if padding_s is None else padding_s
        return max(0.0, first - point_padding), first + point_padding
    between = question[matches[0].end() : matches[1].start()]
    if not re.search(
        r"(?:-|\u2013|\u2014|~|to|through|until|from)",
        between,
        re.IGNORECASE,
    ):
        return None
    second = _timestamp_seconds(matches[1])
    range_padding = 0.0 if padding_s is None else padding_s
    return (
        max(0.0, min(first, second) - range_padding),
        max(first, second) + range_padding,
    )


__all__ = ["parse_question_time_range"]
