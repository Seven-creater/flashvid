from __future__ import annotations

import hashlib
import json
import math
import re
import unicodedata
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .answers import extract_strict_answer_letter
from .schemas import Sample


_OPTION_LETTER_RE = re.compile(r"^[A-H]$")
_ANSWER_PREFIX_RE = re.compile(
    r"(?i)^(?:final\s+)?(?:answer|choice|option)\s*(?:is|:|=)?\s*"
)
_PRIVATE_INPUT_KEYS = {
    "answer",
    "correct_answer",
    "ground_truth",
    "right_answer",
    "time_range",
    "clue_intervals",
    "question_type",
}
_TRAJECTORY_ERROR_FIELDS = {
    "error",
    "verifier_error",
    "api_error",
    "frame_error",
    "transcode_error",
    "parse_error",
}
_PRIVATE_TEXT_MARKER_RE = re.compile(
    r"(?i)(?:^|[\s{\"'])"
    r"(?:correct[_ ]answer|ground[_ ]truth|right[_ ]answer|time[_ ]range|"
    r"clue[_ ]intervals?|question[_ ]type)"
    r"\s*[:=]"
)
_TIMESTAMP_RE = re.compile(r"(?<!\d)(?:\d{1,2}:)?[0-5]?\d:[0-5]\d(?!\d)")
_OCR_RE = re.compile(
    r"(?is)\b(text|written|writing|word|words|read|reads|says|sign|subtitle|"
    r"caption|label|logo|screen|display|banner|poster|letter|number|symbol)\b"
)
_ACTION_RE = re.compile(
    r"(?is)\b(what did .+ do|what does .+ do|what action|what activity|how did|"
    r"how does|gesture|movement|move|turn|pick up|put down|open|close|enter|"
    r"leave|change in|changes in|reaction|interact)\b"
)
_TEMPORAL_RE = re.compile(
    r"(?is)\b(before|after|while|during|then|first|last|next|earlier|later|"
    r"sequence|order|timeline|at the end|at the beginning|finally)\b"
)


@dataclass(frozen=True)
class CandidateNormalization:
    answer: str | None
    source: str
    reason: str


@dataclass(frozen=True)
class VideoSplit:
    train: tuple[Sample, ...]
    dev: tuple[Sample, ...]
    excluded_test_sample_ids: tuple[str, ...]
    unused_sample_ids: tuple[str, ...]


@dataclass(frozen=True)
class TrajectorySelection:
    selected: tuple[dict[str, Any], ...]
    no_positive_sample_ids: tuple[str, ...]
    unknown_sample_ids: tuple[str, ...]


def _stable_rank(seed: int, namespace: str, *values: str) -> str:
    payload = "\0".join((str(seed), namespace, *values)).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _normalized_text(value: Any) -> str:
    text = unicodedata.normalize("NFKC", str(value or "")).casefold()
    text = re.sub(r"[\s\W_]+", " ", text, flags=re.UNICODE)
    return " ".join(text.split())


def _valid_choices(choices: Mapping[str, Any]) -> dict[str, str]:
    normalized: dict[str, str] = {}
    for raw_letter, text in choices.items():
        letter = str(raw_letter).strip().upper()
        if not _OPTION_LETTER_RE.fullmatch(letter):
            raise ValueError(f"invalid option letter: {raw_letter!r}")
        normalized[letter] = str(text)
    if not normalized:
        raise ValueError("choices cannot be empty")
    return normalized


def _match_choice_text(text: Any, choices: Mapping[str, str]) -> str | None:
    candidate = str(text or "").strip().strip("`*_\"'")
    candidate = _ANSWER_PREFIX_RE.sub("", candidate, count=1)
    candidate = candidate.strip().strip("`*_\"'").rstrip(".!?")
    normalized = _normalized_text(candidate)
    if not normalized:
        return None
    matches = [
        letter
        for letter, choice_text in choices.items()
        if _normalized_text(choice_text) == normalized
    ]
    return matches[0] if len(matches) == 1 else None


def normalize_candidate(
    prediction: Any,
    raw_response: Any,
    choices: Mapping[str, Any],
) -> CandidateNormalization:
    """Normalize a frozen Direct candidate without making a model request.

    The function first accepts an already parsed option or a strict final-line
    answer. It then performs only an exact, normalized match against one unique
    option text. Explanatory mentions of an option are intentionally rejected.
    """

    normalized_choices = _valid_choices(choices)
    valid_letters = tuple(normalized_choices)
    prediction_text = str(prediction or "").strip()
    direct_letter = prediction_text.upper()
    if direct_letter in normalized_choices:
        return CandidateNormalization(direct_letter, "parsed", "prediction_letter")

    for value, reason in (
        (prediction_text, "strict_prediction"),
        (str(raw_response or ""), "strict_raw_response"),
    ):
        parsed = extract_strict_answer_letter(value, valid_letters)
        if parsed is not None:
            return CandidateNormalization(parsed, "parsed", reason)

    raw_text = str(raw_response or "")
    final_line = next(
        (line.strip() for line in reversed(raw_text.splitlines()) if line.strip()),
        "",
    )
    for value, reason in (
        (prediction_text, "prediction_choice_text"),
        (final_line, "final_line_choice_text"),
        (raw_text, "raw_response_choice_text"),
    ):
        matched = _match_choice_text(value, normalized_choices)
        if matched is not None:
            return CandidateNormalization(matched, "normalized", reason)
    return CandidateNormalization(None, "none", "no_unambiguous_match")


def infer_question_route(question: str) -> str:
    """Infer the v3-style route from question text, never annotation metadata."""

    if _TIMESTAMP_RE.search(question):
        return "explicit_question_time"
    if _OCR_RE.search(question):
        return "ocr_detail"
    if _ACTION_RE.search(question):
        return "action_event"
    if _TEMPORAL_RE.search(question):
        return "temporal_event"
    return "global_overview"


def video_group_key(sample: Sample) -> str:
    normalized_path = sample.video.replace("\\", "/")
    video_uid = Path(normalized_path).stem.casefold()
    return f"{sample.dataset.casefold()}:{video_uid}"


def _route_quotas(samples_by_route: Mapping[str, Sequence[Sample]], count: int) -> dict[str, int]:
    nonempty = {route: len(items) for route, items in samples_by_route.items() if items}
    if not nonempty or count <= 0:
        return {route: 0 for route in nonempty}
    quotas = {route: 0 for route in nonempty}
    remaining = count

    if count >= len(nonempty):
        for route in nonempty:
            quotas[route] = 1
        remaining -= len(nonempty)

    if remaining:
        capacity = {route: nonempty[route] - quotas[route] for route in nonempty}
        capacity_total = sum(capacity.values())
        exact = {
            route: (remaining * capacity[route] / capacity_total if capacity_total else 0.0)
            for route in nonempty
        }
        for route in nonempty:
            addition = min(capacity[route], math.floor(exact[route]))
            quotas[route] += addition
        left = count - sum(quotas.values())
        order = sorted(
            nonempty,
            key=lambda route: (-(exact[route] - math.floor(exact[route])), route),
        )
        while left:
            progressed = False
            for route in order:
                if quotas[route] >= nonempty[route]:
                    continue
                quotas[route] += 1
                left -= 1
                progressed = True
                if not left:
                    break
            if not progressed:
                raise RuntimeError("unable to allocate route quotas")
    return quotas


def _take_stratified(
    samples: Sequence[Sample],
    count: int,
    seed: int,
    namespace: str,
) -> tuple[Sample, ...]:
    if count > len(samples):
        raise ValueError(f"requested {count} samples but only {len(samples)} are available")
    buckets: dict[str, list[Sample]] = defaultdict(list)
    for sample in samples:
        buckets[infer_question_route(sample.question)].append(sample)
    for route, items in buckets.items():
        items.sort(
            key=lambda sample: (
                _stable_rank(
                    seed,
                    f"{namespace}:{route}",
                    video_group_key(sample),
                    sample.sample_id,
                ),
                sample.sample_id,
            )
        )
    quotas = _route_quotas(buckets, count)
    selected = [
        sample
        for route in sorted(buckets)
        for sample in buckets[route][: quotas[route]]
    ]
    return tuple(
        sorted(
            selected,
            key=lambda sample: (sample.dataset, sample.sample_id),
        )
    )


def split_samples_by_video(
    samples: Sequence[Sample],
    frozen_test_samples: Sequence[Sample],
    train_count: int,
    dev_count: int,
    seed: int = 42,
) -> VideoSplit:
    """Create deterministic exact-size splits with disjoint video groups."""

    if train_count < 0 or dev_count < 0:
        raise ValueError("train_count and dev_count must be non-negative")
    identities = [(sample.dataset, sample.sample_id) for sample in samples]
    if len(identities) != len(set(identities)):
        raise ValueError("candidate samples contain duplicate dataset/sample_id pairs")

    frozen_videos = {video_group_key(sample) for sample in frozen_test_samples}
    eligible = [sample for sample in samples if video_group_key(sample) not in frozen_videos]
    excluded = [sample.sample_id for sample in samples if video_group_key(sample) in frozen_videos]
    groups: dict[str, list[Sample]] = defaultdict(list)
    for sample in eligible:
        groups[video_group_key(sample)].append(sample)

    ordered_groups = sorted(
        groups,
        key=lambda key: (_stable_rank(seed, "dev-video-reservation", key), key),
    )
    total_capacity = sum(len(groups[key]) for key in ordered_groups)
    if total_capacity < train_count + dev_count:
        raise ValueError(
            f"need {train_count + dev_count} non-test samples, found {total_capacity}"
        )

    dev_groups: set[str] = set()
    dev_capacity = 0
    remaining_capacity = total_capacity
    for key in ordered_groups:
        if dev_capacity >= dev_count:
            break
        group_size = len(groups[key])
        if remaining_capacity - group_size < train_count:
            continue
        dev_groups.add(key)
        dev_capacity += group_size
        remaining_capacity -= group_size
    if dev_capacity < dev_count:
        raise ValueError(
            "video grouping cannot satisfy the requested train/dev sizes without overlap"
        )

    dev_pool = [sample for key in dev_groups for sample in groups[key]]
    train_pool = [
        sample
        for key in ordered_groups
        if key not in dev_groups
        for sample in groups[key]
    ]
    train = _take_stratified(train_pool, train_count, seed, "train")
    dev = _take_stratified(dev_pool, dev_count, seed, "dev")

    train_videos = {video_group_key(sample) for sample in train}
    dev_videos = {video_group_key(sample) for sample in dev}
    if train_videos & dev_videos or train_videos & frozen_videos or dev_videos & frozen_videos:
        raise AssertionError("video leakage detected after split construction")
    selected_ids = {(sample.dataset, sample.sample_id) for sample in (*train, *dev)}
    unused = [
        sample.sample_id
        for sample in eligible
        if (sample.dataset, sample.sample_id) not in selected_ids
    ]
    return VideoSplit(
        train=train,
        dev=dev,
        excluded_test_sample_ids=tuple(sorted(excluded)),
        unused_sample_ids=tuple(sorted(unused)),
    )


def retained_visual_tokens(trajectory: Mapping[str, Any]) -> float:
    direct = trajectory.get("retained_visual_tokens")
    if isinstance(direct, (int, float)) and not isinstance(direct, bool) and direct >= 0:
        return float(direct)
    steps = trajectory.get("tool_steps")
    if not isinstance(steps, list):
        raise ValueError("trajectory has no retained_visual_tokens or tool_steps")
    values = [step.get("retained_visual_tokens") for step in steps if isinstance(step, Mapping)]
    if not values or any(
        not isinstance(value, (int, float)) or isinstance(value, bool) or value < 0
        for value in values
    ):
        raise ValueError("tool_steps contain invalid retained_visual_tokens")
    return float(sum(values))


def _tool_call_count(trajectory: Mapping[str, Any]) -> int:
    explicit = trajectory.get("tool_call_count")
    if isinstance(explicit, int) and not isinstance(explicit, bool) and explicit >= 0:
        return explicit
    for key in ("tool_steps", "tool_calls"):
        calls = trajectory.get(key)
        if isinstance(calls, list):
            return len(calls)
    return 0


def _output_length(trajectory: Mapping[str, Any]) -> int:
    completion_tokens = trajectory.get("completion_tokens")
    if isinstance(completion_tokens, int) and not isinstance(completion_tokens, bool):
        return completion_tokens
    messages = trajectory.get("training_messages", trajectory.get("messages"))
    if isinstance(messages, list):
        return sum(
            len(str(message.get("content") or ""))
            for message in messages
            if isinstance(message, Mapping) and message.get("role") == "assistant"
        )
    return len(str(trajectory.get("raw_response") or ""))


def _budget_sequence(trajectory: Mapping[str, Any]) -> tuple[float, ...]:
    direct = trajectory.get("budget_sequence")
    if isinstance(direct, list):
        return tuple(float(value) for value in direct)
    steps = trajectory.get("tool_steps")
    if isinstance(steps, list):
        return tuple(
            float(step["retention_ratio"])
            for step in steps
            if isinstance(step, Mapping)
            and isinstance(step.get("retention_ratio"), (int, float))
            and not isinstance(step.get("retention_ratio"), bool)
        )
    return ()


def _trajectory_id(trajectory: Mapping[str, Any]) -> str:
    for key in ("trajectory_id", "trace_id", "trajectory_index"):
        if trajectory.get(key) is not None:
            return str(trajectory[key])
    canonical = json.dumps(trajectory, ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _trajectory_sort_key(trajectory: Mapping[str, Any]) -> tuple[float, int, float, int, str]:
    latency = trajectory.get("latency_s")
    normalized_latency = (
        float(latency)
        if isinstance(latency, (int, float)) and not isinstance(latency, bool) and latency >= 0
        else math.inf
    )
    return (
        retained_visual_tokens(trajectory),
        _tool_call_count(trajectory),
        normalized_latency,
        _output_length(trajectory),
        _trajectory_id(trajectory),
    )


def _prediction(trajectory: Mapping[str, Any]) -> str | None:
    value = trajectory.get("final_prediction", trajectory.get("prediction"))
    candidate = str(value or "").strip().upper()
    return candidate if _OPTION_LETTER_RE.fullmatch(candidate) else None


def _trajectory_usable(trajectory: Mapping[str, Any]) -> bool:
    if trajectory.get("trajectory_valid") is False:
        return False
    if trajectory.get("failure_stage"):
        return False
    if any(trajectory.get(key) for key in _TRAJECTORY_ERROR_FIELDS):
        return False
    leak_status = trajectory.get("annotation_leak_check")
    if leak_status != "passed":
        return False
    try:
        retained_visual_tokens(trajectory)
    except (TypeError, ValueError):
        return False
    messages = trajectory.get("training_messages", trajectory.get("messages"))
    return isinstance(messages, list) and bool(messages)


def select_training_trajectories(
    trajectories: Iterable[Mapping[str, Any]],
    correct_answers: Mapping[str, str],
    second_trace_limit: float | None = 1.2,
) -> TrajectorySelection:
    """Select the cheapest correct trace and an optional diverse second trace."""

    if second_trace_limit is not None and second_trace_limit < 1.0:
        raise ValueError("second_trace_limit must be at least 1.0")
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    unknown: set[str] = set()
    for source in trajectories:
        trajectory = dict(source)
        sample_id = str(trajectory.get("sample_id") or "")
        if not sample_id or sample_id not in correct_answers:
            if sample_id:
                unknown.add(sample_id)
            continue
        answer = str(correct_answers[sample_id]).strip().upper()
        if _prediction(trajectory) == answer and _trajectory_usable(trajectory):
            grouped[sample_id].append(trajectory)

    selected: list[dict[str, Any]] = []
    no_positive: list[str] = []
    for sample_id in sorted(correct_answers):
        positives = sorted(grouped.get(sample_id, ()), key=_trajectory_sort_key)
        if not positives:
            no_positive.append(sample_id)
            continue
        primary = dict(positives[0])
        primary["_selection_role"] = "primary"
        primary["_selection_cost"] = retained_visual_tokens(primary)
        selected.append(primary)

        expected = str(correct_answers[sample_id]).strip().upper()
        candidate = str(primary.get("candidate_answer") or "").strip().upper()
        if (
            second_trace_limit is None
            or not candidate
            or candidate == expected
        ):
            continue
        primary_sequence = _budget_sequence(primary)
        cost_limit = retained_visual_tokens(primary) * second_trace_limit
        alternatives = [
            trajectory
            for trajectory in positives[1:]
            if retained_visual_tokens(trajectory) <= cost_limit
            and _budget_sequence(trajectory) != primary_sequence
        ]
        if alternatives:
            secondary = dict(min(alternatives, key=_trajectory_sort_key))
            secondary["_selection_role"] = "secondary_changed_candidate"
            secondary["_selection_cost"] = retained_visual_tokens(secondary)
            selected.append(secondary)

    return TrajectorySelection(
        selected=tuple(selected),
        no_positive_sample_ids=tuple(no_positive),
        unknown_sample_ids=tuple(sorted(unknown)),
    )


def _contains_private_key(value: Any) -> bool:
    if isinstance(value, Mapping):
        for key, item in value.items():
            normalized_key = str(key).strip().casefold().replace(" ", "_")
            if normalized_key in _PRIVATE_INPUT_KEYS or _contains_private_key(item):
                return True
        return False
    if isinstance(value, list):
        return any(_contains_private_key(item) for item in value)
    return False


def _validate_input_content(content: Any) -> None:
    if _contains_private_key(content):
        raise ValueError("private benchmark field found in an SFT input message")
    if not isinstance(content, str):
        return
    stripped = content.strip()
    if stripped.startswith(("{", "[")):
        try:
            parsed = json.loads(stripped)
        except json.JSONDecodeError:
            parsed = None
        if parsed is not None and _contains_private_key(parsed):
            raise ValueError("private benchmark field found in JSON SFT input")
    if _PRIVATE_TEXT_MARKER_RE.search(content):
        raise ValueError("private benchmark marker found in an SFT input message")


def build_sft_record(trajectory: Mapping[str, Any]) -> dict[str, Any]:
    """Build one standard messages record while excluding labels and reasoning."""

    source_messages = trajectory.get("training_messages", trajectory.get("messages"))
    if not isinstance(source_messages, list) or not source_messages:
        raise ValueError("trajectory messages must be a non-empty list")
    messages: list[dict[str, Any]] = []
    assistant_count = 0
    allowed_optional = ("name", "tool_call_id", "tool_calls", "function_call")
    for source in source_messages:
        if not isinstance(source, Mapping):
            raise ValueError("each trajectory message must be an object")
        role = str(source.get("role") or "")
        if role not in {"system", "user", "assistant", "tool"}:
            raise ValueError(f"unsupported message role: {role!r}")
        content = source.get("content", "")
        _validate_input_content(content)
        message: dict[str, Any] = {"role": role, "content": content}
        for key in allowed_optional:
            if key in source:
                value = source[key]
                if _contains_private_key(value):
                    raise ValueError("private benchmark field found in message metadata")
                message[key] = value
        if role == "assistant":
            assistant_count += 1
        messages.append(message)
    if not assistant_count:
        raise ValueError("trajectory has no assistant training target")

    metadata: dict[str, Any] = {}
    for key in ("dataset", "sample_id", "trajectory_id", "_selection_role"):
        if trajectory.get(key) is not None:
            output_key = "selection_role" if key == "_selection_role" else key
            metadata[output_key] = trajectory[key]
    metadata["retained_visual_tokens"] = retained_visual_tokens(trajectory)
    return {"messages": messages, "metadata": metadata}


def write_sft_jsonl(
    trajectories: Iterable[Mapping[str, Any]],
    output: str | Path,
) -> int:
    records = [build_sft_record(trajectory) for trajectory in trajectories]
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )
    temporary.replace(path)
    return len(records)


def route_counts(samples: Iterable[Sample]) -> dict[str, int]:
    return dict(sorted(Counter(infer_question_route(sample.question) for sample in samples).items()))
