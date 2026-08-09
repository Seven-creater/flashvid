"""Candidate-blind perception-memory runtime backed by EVA frame selection.

The controller never receives media or the frozen Direct candidate.  A separate
perception turn sees only the frames selected in the current step and emits a
small, timestamped evidence state.  Subsequent controller/judge turns consume
the deterministic text ledger instead of re-sending old images.

This module is intentionally independent from ``fast_hybrid_eva.py``.  It
reuses only the public Qwen client/frame-tool interfaces already used by the
other agent strategies.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import re
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .datasets import VideoIndex
from .privacy import AnnotationLeakError, assert_annotation_free_request
from .qwen_agents.core import (
    ChatClient,
    FrameObservation,
    FrameRequest,
    FrameTool,
    ToolStep,
    observation_content,
)
from .schemas import ModelSample


_TOOL_CALL_RE = re.compile(r"^\s*<tool_call>\s*(\{.*\})\s*</tool_call>\s*$", re.DOTALL)
_JSON_FENCE_RE = re.compile(
    r"^\s*```(?:json)?\s*\n(\{.*\})\n```\s*$",
    re.DOTALL | re.IGNORECASE,
)
_SPACE_RE = re.compile(r"\s+")
RESCUE_TRAJECTORY_VARIANTS = frozenset(
    {
        "rescue_global32",
        "rescue_global64",
        "rescue_first_half64",
        "rescue_second_half64",
    }
)
_ALLOWED_TRAJECTORY_VARIANTS = frozenset({"base", *RESCUE_TRAJECTORY_VARIANTS})


def _clean_text(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")
    return _SPACE_RE.sub(" ", value.strip())


def _string_list(value: Any, field_name: str) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise ValueError(f"{field_name} must be an array")
    return tuple(_clean_text(item, field_name) for item in value)


def _normal_key(value: str) -> str:
    return _SPACE_RE.sub(" ", value.strip()).casefold()


def _finite_number(value: Any, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field_name} must be a number")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{field_name} must be finite")
    return number


def _question_text(sample: ModelSample) -> str:
    options = "\n".join(f"{key}: {value}" for key, value in sample.choices.items())
    return f"Question: {sample.question}\nChoices:\n{options}"


def rescue_frame_request(variant: str, video_duration: float) -> FrameRequest | None:
    """Return one annotation-free, duration-only rescue coverage request."""

    name = str(variant).strip()
    if name == "base":
        return None
    if name not in RESCUE_TRAJECTORY_VARIANTS:
        raise ValueError(f"unsupported Perception-Memory trajectory variant: {name}")
    duration = _finite_number(video_duration, "video_duration")
    if duration <= 0:
        raise ValueError("video_duration must be positive")
    start, end = 0.0, duration
    nframes = 32 if name == "rescue_global32" else 64
    if name == "rescue_first_half64":
        end = duration / 2.0
    elif name == "rescue_second_half64":
        start = duration / 2.0
    return FrameRequest(
        start_time=start,
        end_time=end,
        nframes=nframes,
        resize=0.75,
        evidence_request=(
            "Record only directly visible actions, state changes, and option-"
            "discriminative facts across this fixed coverage interval."
        ),
    )


def messages_have_media(messages: Sequence[Mapping[str, Any]]) -> bool:
    """Return whether an OpenAI message list contains image/video content."""

    for message in messages:
        content = message.get("content")
        items = content if isinstance(content, list) else (content,)
        for item in items:
            if not isinstance(item, Mapping):
                continue
            item_type = str(item.get("type", "")).lower()
            if item_type in {"image", "image_url", "video", "video_url"}:
                return True
            if "image_url" in item or "video_url" in item:
                return True
    return False


@dataclass(frozen=True)
class ControllerAction:
    action: str
    request: FrameRequest | None = None

    def __post_init__(self) -> None:
        if self.action not in {"observe", "stop"}:
            raise ValueError("controller action must be observe or stop")
        if (self.action == "observe") != (self.request is not None):
            raise ValueError("observe requires a request and stop forbids one")


def parse_controller_action(text: str) -> ControllerAction | None:
    """Parse one official EVA tool call or the exact stop object."""

    candidate = (text or "").strip()
    try:
        stop = json.loads(candidate)
    except json.JSONDecodeError:
        stop = None
    if stop == {"action": "stop"}:
        return ControllerAction("stop")

    match = _TOOL_CALL_RE.fullmatch(candidate)
    if match is None:
        return None
    try:
        payload = json.loads(match.group(1))
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict) or set(payload) != {"tool", "arguments"}:
        return None
    if payload.get("tool") != "frame_select":
        return None
    arguments = payload.get("arguments")
    if not isinstance(arguments, dict):
        return None
    allowed = {
        "start_time",
        "end_time",
        "nframes",
        "fps",
        "resize",
        "evidence_request",
    }
    if set(arguments) - allowed:
        return None
    if ("nframes" in arguments) == ("fps" in arguments):
        return None
    try:
        request = FrameRequest(
            start_time=float(arguments["start_time"]),
            end_time=float(arguments["end_time"]),
            nframes=(int(arguments["nframes"]) if "nframes" in arguments else None),
            fps=(float(arguments["fps"]) if "fps" in arguments else None),
            resize=float(arguments.get("resize", 1.0)),
            evidence_request=_clean_text(
                arguments.get("evidence_request"), "evidence_request"
            ),
        )
    except (KeyError, TypeError, ValueError):
        return None
    return ControllerAction("observe", request)


def _controller_tool_call(request: FrameRequest) -> str:
    payload = {"tool": "frame_select", "arguments": request.to_tool_arguments()}
    return (
        "<tool_call>"
        + json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "</tool_call>"
    )


def _messages_sha256(messages: Sequence[Mapping[str, Any]]) -> str:
    return hashlib.sha256(
        json.dumps(
            messages,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        ).encode("utf-8")
    ).hexdigest()


@dataclass(frozen=True)
class TimestampedFact:
    time: float
    fact: str


@dataclass(frozen=True)
class OptionObservation:
    supports: tuple[str, ...] = ()
    contradicts: tuple[str, ...] = ()


@dataclass(frozen=True)
class PerceptionState:
    interval: tuple[float, float]
    timestamped_facts: tuple[TimestampedFact, ...]
    option_evidence: dict[str, OptionObservation]
    temporal_changes: tuple[str, ...]
    unresolved: tuple[str, ...]
    evidence_sufficient: bool
    next_evidence_needed: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "interval": list(self.interval),
            "timestamped_facts": [asdict(item) for item in self.timestamped_facts],
            "option_evidence": {
                letter: {
                    "supports": list(item.supports),
                    "contradicts": list(item.contradicts),
                }
                for letter, item in self.option_evidence.items()
            },
            "temporal_changes": list(self.temporal_changes),
            "unresolved": list(self.unresolved),
            "evidence_sufficient": self.evidence_sufficient,
            "next_evidence_needed": self.next_evidence_needed,
        }


_PERCEPTION_TIMESTAMP_TOLERANCE_S = 0.001
_MAX_PERCEPTION_FACTS = 12
_MAX_OPTION_EVIDENCE_ITEMS = 1
_MAX_TEMPORAL_CHANGES = 4
_MAX_UNRESOLVED_ITEMS = 3
_MAX_OBSERVATION_WORDS = 20
_MAX_OBSERVATION_CHARS = 240
PERCEPTION_NORMALIZATION_VERSION = "perception_state_normalization_v2"


def _bounded_observation_text(value: str, *, max_words: int) -> str:
    words = _clean_text(value, "perception observation").split()
    return " ".join(words[:max_words])[:_MAX_OBSERVATION_CHARS].strip()


def _evenly_spaced_positions(length: int, limit: int) -> tuple[int, ...]:
    if length < 0 or limit <= 0:
        raise ValueError("length must be non-negative and limit must be positive")
    if length <= limit:
        return tuple(range(length))
    denominator = limit - 1
    if denominator <= 0:
        return (0,)
    return tuple(
        (index * (length - 1) + denominator // 2) // denominator
        for index in range(limit)
    )


def _compact_timestamped_facts(
    facts: Sequence[TimestampedFact],
    option_evidence: Mapping[str, OptionObservation],
) -> tuple[TimestampedFact, ...]:
    ordered = sorted(enumerate(facts), key=lambda item: (item[1].time, item[0]))
    referenced = {
        _normal_key(text)
        for observation in option_evidence.values()
        for text in (*observation.supports, *observation.contradicts)
    }
    priority = [item for item in ordered if _normal_key(item[1].fact) in referenced]
    ordinary = [item for item in ordered if _normal_key(item[1].fact) not in referenced]
    selected: list[tuple[int, TimestampedFact]] = []
    priority_limit = min(_MAX_PERCEPTION_FACTS, len(priority))
    if priority_limit:
        selected.extend(
            priority[index]
            for index in _evenly_spaced_positions(len(priority), priority_limit)
        )
    remaining = _MAX_PERCEPTION_FACTS - len(selected)
    if remaining:
        selected.extend(
            ordinary[index]
            for index in _evenly_spaced_positions(len(ordinary), remaining)
        )
    selected.sort(key=lambda item: (item[1].time, item[0]))
    return tuple(
        TimestampedFact(
            item.time,
            _bounded_observation_text(item.fact, max_words=_MAX_OBSERVATION_WORDS),
        )
        for _original_index, item in selected
    )


def parse_perception_state(
    text: str, valid_letters: Iterable[str]
) -> PerceptionState | None:
    """Strictly parse one perception observation without guessing fields."""

    letters = tuple(str(item).strip().upper() for item in valid_letters)
    raw_text = text or ""
    fenced = _JSON_FENCE_RE.fullmatch(raw_text)
    if fenced is not None:
        raw_text = fenced.group(1)
    try:
        payload = json.loads(raw_text.strip())
        required = {
            "interval",
            "timestamped_facts",
            "option_evidence",
            "temporal_changes",
            "unresolved",
            "evidence_sufficient",
            "next_evidence_needed",
        }
        if not isinstance(payload, dict) or set(payload) != required:
            return None
        interval = payload["interval"]
        if not isinstance(interval, list) or len(interval) != 2:
            return None
        start = _finite_number(interval[0], "interval[0]")
        end = _finite_number(interval[1], "interval[1]")
        if end <= start:
            return None

        facts_raw = payload["timestamped_facts"]
        if not isinstance(facts_raw, list):
            return None
        facts: list[TimestampedFact] = []
        for item in facts_raw:
            if not isinstance(item, dict) or set(item) != {"time", "fact"}:
                return None
            facts.append(
                TimestampedFact(
                    _finite_number(item["time"], "timestamped_facts.time"),
                    _clean_text(item["fact"], "timestamped_facts.fact"),
                )
            )

        option_raw = payload["option_evidence"]
        if not isinstance(option_raw, dict) or set(option_raw) != set(letters):
            return None
        options: dict[str, OptionObservation] = {}
        for letter in letters:
            item = option_raw[letter]
            if not isinstance(item, dict) or set(item) != {"supports", "contradicts"}:
                return None
            options[letter] = OptionObservation(
                _string_list(item["supports"], f"option_evidence.{letter}.supports"),
                _string_list(
                    item["contradicts"], f"option_evidence.{letter}.contradicts"
                ),
            )
        if not isinstance(payload["evidence_sufficient"], bool):
            return None
        next_needed = payload["next_evidence_needed"]
        if not isinstance(next_needed, str):
            return None
        return PerceptionState(
            interval=(start, end),
            timestamped_facts=tuple(facts),
            option_evidence=options,
            temporal_changes=_string_list(
                payload["temporal_changes"], "temporal_changes"
            ),
            unresolved=_string_list(payload["unresolved"], "unresolved"),
            evidence_sufficient=payload["evidence_sufficient"],
            next_evidence_needed=_SPACE_RE.sub(" ", next_needed.strip()),
        )
    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
        return None


def normalize_perception_state(
    state: PerceptionState,
    *,
    resolved_start_time: float,
    resolved_end_time: float,
    actual_timestamps: Sequence[float],
    timestamp_tolerance_s: float = _PERCEPTION_TIMESTAMP_TOLERANCE_S,
) -> PerceptionState:
    """Validate raw state, bind real timestamps, then compact deterministically.

    The perception prompt prints timestamps to millisecond precision, so a model
    timestamp may differ from the selector value by at most one millisecond.  A
    reported fact outside the resolved interval or not attached to an actual
    frame is rejected instead of entering the persistent evidence ledger.
    """

    if (
        isinstance(timestamp_tolerance_s, bool)
        or not isinstance(timestamp_tolerance_s, (int, float))
        or not math.isfinite(float(timestamp_tolerance_s))
        or timestamp_tolerance_s < 0
    ):
        raise ValueError("timestamp_tolerance_s must be a finite non-negative number")
    tolerance = float(timestamp_tolerance_s)
    start = _finite_number(resolved_start_time, "resolved_start_time")
    end = _finite_number(resolved_end_time, "resolved_end_time")
    if end <= start:
        raise ValueError("resolved perception interval must be increasing")
    actual_timestamps = tuple(
        _finite_number(item, "actual frame timestamp") for item in actual_timestamps
    )
    if not actual_timestamps:
        raise ValueError("perception observation contains no actual frame timestamps")
    for timestamp in actual_timestamps:
        if timestamp < start - tolerance or timestamp > end + tolerance:
            raise ValueError(
                "actual frame timestamp lies outside the resolved perception interval"
            )

    bound_facts: list[TimestampedFact] = []
    for item in state.timestamped_facts:
        if item.time < start - tolerance or item.time > end + tolerance:
            raise ValueError(
                "perception fact timestamp lies outside the resolved perception interval"
            )
        actual = min(actual_timestamps, key=lambda value: abs(value - item.time))
        if abs(actual - item.time) > tolerance + 1e-12:
            raise ValueError(
                "perception fact timestamp does not match an actual sampled frame timestamp"
            )
        bound_facts.append(TimestampedFact(actual, item.fact))
    compact_options = {
        letter: OptionObservation(
            tuple(
                _bounded_observation_text(value, max_words=_MAX_OBSERVATION_WORDS)
                for value in observation.supports[:_MAX_OPTION_EVIDENCE_ITEMS]
            ),
            tuple(
                _bounded_observation_text(value, max_words=_MAX_OBSERVATION_WORDS)
                for value in observation.contradicts[:_MAX_OPTION_EVIDENCE_ITEMS]
            ),
        )
        for letter, observation in state.option_evidence.items()
    }
    return replace(
        state,
        interval=(start, end),
        timestamped_facts=_compact_timestamped_facts(
            bound_facts, state.option_evidence
        ),
        option_evidence=compact_options,
        temporal_changes=tuple(
            _bounded_observation_text(value, max_words=_MAX_OBSERVATION_WORDS)
            for value in state.temporal_changes[:_MAX_TEMPORAL_CHANGES]
        ),
        unresolved=tuple(
            _bounded_observation_text(value, max_words=_MAX_OBSERVATION_WORDS)
            for value in state.unresolved[:_MAX_UNRESOLVED_ITEMS]
        ),
        next_evidence_needed=(
            _bounded_observation_text(
                state.next_evidence_needed, max_words=_MAX_OBSERVATION_WORDS
            )
            if state.next_evidence_needed
            else ""
        ),
    )


def validate_perception_state_observation(
    state: PerceptionState,
    observation: FrameObservation,
    *,
    timestamp_tolerance_s: float = _PERCEPTION_TIMESTAMP_TOLERANCE_S,
) -> PerceptionState:
    """Normalize one state against the exact frames returned by the tool."""

    return normalize_perception_state(
        state,
        resolved_start_time=observation.resolved_start_time,
        resolved_end_time=observation.resolved_end_time,
        actual_timestamps=observation.timestamps,
        timestamp_tolerance_s=timestamp_tolerance_s,
    )


@dataclass(frozen=True)
class EvidenceEvent:
    evidence_id: str
    interval: tuple[float, float]
    timestamp: float | None
    fact: str
    source: str


@dataclass(frozen=True)
class OptionLedger:
    supports: tuple[str, ...] = ()
    contradicts: tuple[str, ...] = ()


@dataclass
class EvidenceMemory:
    """Deterministic text memory shared by controller and evidence judges."""

    option_letters: tuple[str, ...]
    event_ledger: list[EvidenceEvent] = field(default_factory=list)
    option_ledger: dict[str, OptionLedger] = field(default_factory=dict)
    unresolved: list[str] = field(default_factory=list)
    observed_intervals: list[tuple[float, float]] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.option_ledger:
            self.option_ledger = {
                letter: OptionLedger() for letter in self.option_letters
            }

    def _evidence_id(
        self,
        fact: str,
        interval: tuple[float, float],
        timestamp: float | None,
        source: str,
    ) -> str:
        key = (
            _normal_key(fact),
            interval,
            timestamp,
            _normal_key(source),
        )
        for event in self.event_ledger:
            event_key = (
                _normal_key(event.fact),
                event.interval,
                event.timestamp,
                _normal_key(event.source),
            )
            if event_key == key:
                return event.evidence_id
        evidence_id = f"E{len(self.event_ledger) + 1:04d}"
        self.event_ledger.append(
            EvidenceEvent(evidence_id, interval, timestamp, fact, source)
        )
        return evidence_id

    def _option_evidence_ids(
        self,
        fact: str,
        interval: tuple[float, float],
        source: str,
    ) -> tuple[str, ...]:
        """Link option claims to existing facts before creating a new event.

        Perception commonly repeats the same visible fact in ``timestamped_facts``
        and in one or more option support/contradiction arrays.  Those arrays are
        relations, not additional observed events.  When the fact and interval
        already exist, every matching timestamped occurrence is referenced so
        counting and ordering evidence remains intact without tripling the ledger.
        """

        key = _normal_key(fact)
        existing = tuple(
            event.evidence_id
            for event in self.event_ledger
            if _normal_key(event.fact) == key and event.interval == interval
        )
        if existing:
            return existing
        return (self._evidence_id(fact, interval, None, source),)

    def merge(self, state: PerceptionState) -> None:
        """Merge one observation in stable input order and preserve old evidence."""

        self.observed_intervals.append(state.interval)
        for item in state.timestamped_facts:
            self._evidence_id(item.fact, state.interval, item.time, "timestamped_fact")
        for change in state.temporal_changes:
            self._evidence_id(change, state.interval, None, "temporal_change")
        for letter in self.option_letters:
            observation = state.option_evidence[letter]
            current = self.option_ledger[letter]
            supports = list(current.supports)
            contradicts = list(current.contradicts)
            for fact in observation.supports:
                for evidence_id in self._option_evidence_ids(
                    fact, state.interval, f"option_{letter}_support"
                ):
                    if evidence_id not in supports:
                        supports.append(evidence_id)
            for fact in observation.contradicts:
                for evidence_id in self._option_evidence_ids(
                    fact, state.interval, f"option_{letter}_contradiction"
                ):
                    if evidence_id not in contradicts:
                        contradicts.append(evidence_id)
            self.option_ledger[letter] = OptionLedger(
                tuple(supports), tuple(contradicts)
            )
        self.unresolved = list(dict.fromkeys(state.unresolved))

    @property
    def evidence_ids(self) -> frozenset[str]:
        return frozenset(item.evidence_id for item in self.event_ledger)

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_ledger": [
                {
                    "evidence_id": item.evidence_id,
                    "interval": list(item.interval),
                    "timestamp": item.timestamp,
                    "fact": item.fact,
                    "source": item.source,
                }
                for item in self.event_ledger
            ],
            "option_ledger": {
                letter: {
                    "supports": list(self.option_ledger[letter].supports),
                    "contradicts": list(self.option_ledger[letter].contradicts),
                }
                for letter in self.option_letters
            },
            "unresolved": list(self.unresolved),
            "observed_intervals": [list(item) for item in self.observed_intervals],
        }


def interval_iou(first: tuple[float, float], second: tuple[float, float]) -> float:
    """Return temporal intersection-over-union for two valid intervals."""

    if first[1] <= first[0] or second[1] <= second[0]:
        raise ValueError("interval end must exceed start")
    intersection = max(0.0, min(first[1], second[1]) - max(first[0], second[0]))
    union = max(first[1], second[1]) - min(first[0], second[0])
    return intersection / union if union else 0.0


def duplicate_interval(
    candidate: tuple[float, float],
    observed: Iterable[tuple[float, float]],
    *,
    threshold: float = 0.85,
) -> bool:
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be in [0, 1]")
    return any(interval_iou(candidate, item) >= threshold for item in observed)


def _memory_json(memory: EvidenceMemory) -> str:
    return json.dumps(
        memory.to_dict(), ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )


def build_controller_messages(
    sample: ModelSample,
    memory: EvidenceMemory,
    video_metadata: Mapping[str, float | int],
    *,
    feedback: str = "",
) -> list[dict[str, Any]]:
    """Build the candidate-blind, text-only controller request."""

    duration = float(video_metadata["duration"])
    width = int(video_metadata.get("width", 0))
    height = int(video_metadata.get("height", 0))
    system = (
        "You are the text-only controller of a long-video evidence agent. Decide "
        "what visual evidence is still missing from the timestamped text ledger. "
        "You cannot see images. Never infer an action that is absent from the ledger. "
        "To observe, output only one official EVA call: "
        '<tool_call>{"tool":"frame_select","arguments":{"start_time":0.0,'
        '"end_time":30.0,"nframes":16,"resize":0.75,'
        '"evidence_request":"a precise visual question"}}</tool_call>. '
        "Use exactly one of nframes or fps and do not repeat an observed interval. "
        "Only when the ledger is ready for an independent completeness check, output "
        'exactly {"action":"stop"}.'
    )
    user = (
        f"{_question_text(sample)}\nVideo duration: {duration:.3f} seconds; "
        f"resolution: {width}x{height}.\nEvidence memory: {_memory_json(memory)}"
    )
    if feedback:
        user += f"\nController feedback: {_SPACE_RE.sub(' ', feedback.strip())}"
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    if messages_have_media(messages):
        raise AssertionError("controller messages must be text-only")
    return messages


def build_perception_messages(
    sample: ModelSample,
    observation: FrameObservation,
    evidence_request: str,
    *,
    use_frame_indices: bool = False,
) -> list[dict[str, Any]]:
    """Build a candidate-blind request containing only the current-step frames."""

    letters = ", ".join(sample.option_letters)
    fact_schema = "{frame_index,fact}" if use_frame_indices else "{time,fact}"
    fact_reference = (
        f"frame_index must be a zero-based integer from 0 to "
        f"{len(observation.timestamps) - 1}; do not copy or estimate timestamps"
        if use_frame_indices
        else "time must be the numeric timestamp printed beside an observed frame"
    )
    system = (
        "You are the visual perception role. Report only facts directly visible in "
        "the current timestamped frames. Do not guess missing actions. Return one JSON "
        "object with exactly: interval, timestamped_facts, option_evidence, "
        "temporal_changes, unresolved, evidence_sufficient, next_evidence_needed. "
        f"timestamped_facts is an array of {fact_schema}; {fact_reference}. "
        "option_evidence must contain "
        f"exactly these option labels: {letters}; each has supports and contradicts "
        "string arrays. interval is two numbers; each timestamped fact contains its "
        "required frame reference and a string fact; temporal_changes and unresolved "
        "are string arrays; "
        "evidence_sufficient is boolean; next_evidence_needed is one string, empty only "
        "when no further evidence is needed. evidence_sufficient describes only the "
        "current accumulated visual question, not benchmark correctness. Keep the state "
        "short: at most 12 timestamped_facts (the most discriminative facts, at most 20 "
        "words each), at most one support and one contradiction per option, at most four "
        "temporal_changes, at most three unresolved items, and at most 20 words in "
        "next_evidence_needed. Return raw JSON only, with no markdown or code fences."
    )
    instruction = (
        f"{_question_text(sample)}\nVisual evidence request: {evidence_request}\n"
        "Describe what these frames establish and what remains unobserved."
    )
    content = observation_content(observation, instruction)
    opening = "<tool_response>"
    if not content or not str(content[0].get("text", "")).startswith(opening):
        raise ValueError("EVA observation is missing the tool-response wrapper")
    if content[-1].get("text") != "</tool_response>":
        raise ValueError("EVA observation has an invalid tool-response wrapper")
    content[0] = dict(content[0])
    content[0]["text"] = str(content[0]["text"])[len(opening) :]
    content.pop()
    return [
        {"role": "system", "content": system},
        {
            "role": "user",
            "content": content,
        },
    ]


def build_completeness_messages(
    sample: ModelSample, memory: EvidenceMemory
) -> list[dict[str, Any]]:
    """Build a candidate-blind, tool-disabled evidence completeness request."""

    return [
        {
            "role": "system",
            "content": (
                "Judge only whether the supplied timestamped visual evidence is enough "
                "to distinguish the options. You have no tools and cannot assume unseen "
                "events. Return exactly one JSON object with evidence_complete (boolean) "
                "and missing_evidence (array of short strings)."
            ),
        },
        {
            "role": "user",
            "content": f"{_question_text(sample)}\nEvidence memory: {_memory_json(memory)}",
        },
    ]


def build_confirmation_controller_messages(
    sample: ModelSample,
    memory: EvidenceMemory,
    video_metadata: Mapping[str, float | int],
    hypotheses: tuple[str, str],
) -> list[dict[str, Any]]:
    """Request fresh visual evidence without identifying the Direct branch."""

    first, second = sorted(str(item).strip().upper() for item in hypotheses)
    duration = float(video_metadata["duration"])
    return [
        {
            "role": "system",
            "content": (
                "Two equally unprivileged hypotheses disagree. Select one decisive visual "
                "interval that can distinguish them. Re-observing a prior area is allowed "
                "only with denser frames or wider before/after context. Return only one "
                "official EVA frame_select tool call with start_time, end_time, exactly one "
                "of nframes or fps, resize, and a symmetric evidence_request that tests "
                "both hypotheses."
            ),
        },
        {
            "role": "user",
            "content": (
                f"{_question_text(sample)}\nHypotheses (unordered): {first}, {second}. "
                f"Video duration: {duration:.3f} seconds.\nEvidence memory: "
                f"{_memory_json(memory)}"
            ),
        },
    ]


def build_judge_messages(
    sample: ModelSample, memory: EvidenceMemory
) -> list[dict[str, Any]]:
    """Build a candidate-blind evidence-only MCQ request."""

    return [
        {
            "role": "system",
            "content": (
                "Answer only from the timestamped evidence ledger. Every choice must be "
                "supported by cited evidence IDs; never fill in an unobserved action. "
                'Return exactly {"answer":"X","evidence_ids":["E0001"]}.'
            ),
        },
        {
            "role": "user",
            "content": f"{_question_text(sample)}\nEvidence memory: {_memory_json(memory)}",
        },
    ]


@dataclass(frozen=True)
class CompletenessDecision:
    evidence_complete: bool
    missing_evidence: tuple[str, ...]


def parse_completeness(text: str) -> CompletenessDecision | None:
    try:
        payload = json.loads((text or "").strip())
        if not isinstance(payload, dict) or set(payload) != {
            "evidence_complete",
            "missing_evidence",
        }:
            return None
        if not isinstance(payload["evidence_complete"], bool):
            return None
        return CompletenessDecision(
            payload["evidence_complete"],
            _string_list(payload["missing_evidence"], "missing_evidence"),
        )
    except (TypeError, ValueError, json.JSONDecodeError):
        return None


@dataclass(frozen=True)
class EvidenceDecision:
    answer: str
    evidence_ids: tuple[str, ...]


def parse_evidence_decision(
    text: str,
    valid_letters: Iterable[str],
    valid_evidence_ids: Iterable[str],
) -> EvidenceDecision | None:
    letters = {str(item).strip().upper() for item in valid_letters}
    evidence = set(valid_evidence_ids)
    try:
        payload = json.loads((text or "").strip())
        if not isinstance(payload, dict) or set(payload) != {"answer", "evidence_ids"}:
            return None
        answer = str(payload["answer"]).strip().upper()
        ids = _string_list(payload["evidence_ids"], "evidence_ids")
        if (
            answer not in letters
            or not ids
            or any(item not in evidence for item in ids)
        ):
            return None
        return EvidenceDecision(answer, tuple(dict.fromkeys(ids)))
    except (TypeError, ValueError, json.JSONDecodeError):
        return None


@dataclass(frozen=True)
class FinalDecision:
    prediction: str | None
    source: str
    fallback_to_candidate: bool


def apply_candidate_gate(
    *,
    candidate: str | None,
    first: EvidenceDecision | None,
    confirmation: EvidenceDecision | None,
    complete: bool,
    memory: EvidenceMemory,
) -> FinalDecision:
    """Apply the conservative Direct-candidate gate without lexical heuristics."""

    if not complete or first is None:
        return FinalDecision(candidate, "candidate_fallback", candidate is not None)
    if candidate is None:
        return FinalDecision(first.answer, "evidence_only", False)
    if first.answer == candidate:
        return FinalDecision(candidate, "evidence_agrees_with_candidate", False)
    if confirmation is None or confirmation.answer != first.answer:
        return FinalDecision(candidate, "candidate_fallback", True)
    new_ledger = memory.option_ledger.get(first.answer, OptionLedger())
    old_ledger = memory.option_ledger.get(candidate, OptionLedger())
    if not new_ledger.supports or not old_ledger.contradicts:
        return FinalDecision(candidate, "candidate_fallback", True)
    cited = set(first.evidence_ids) | set(confirmation.evidence_ids)
    if not cited.intersection(new_ledger.supports) or not cited.intersection(
        old_ledger.contradicts
    ):
        return FinalDecision(candidate, "candidate_fallback", True)
    return FinalDecision(first.answer, "confirmed_visual_change", False)


def _visual_tokens(usage: Mapping[str, Any]) -> int | None:
    details = usage.get("prompt_tokens_details")
    if isinstance(details, Mapping):
        value = details.get("multimodal_tokens")
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return int(value)
        if isinstance(value, Mapping):
            numbers = [
                item for item in value.values() if isinstance(item, (int, float))
            ]
            if len(numbers) == len(value):
                return int(sum(numbers))
    value = usage.get("visual_tokens")
    return (
        int(value)
        if isinstance(value, (int, float)) and not isinstance(value, bool)
        else None
    )


class PerceptionMemoryEvaEvaluator:
    """Synchronous perception-memory evaluator using one Qwen model in all roles."""

    backend = "perception_memory_eva"
    version = "perception_memory_v1"

    def __init__(
        self,
        client: ChatClient,
        model: str,
        video_root: Path,
        frame_root: Path,
        *,
        frame_tool: FrameTool | None = None,
        max_turns: int = 6,
        max_frames_per_call: int = 128,
        controller_max_tokens: int = 512,
        perception_max_tokens: int = 1024,
        judge_max_tokens: int = 512,
        seed: int = 42,
        candidate_results_sha256: str | None = None,
        model_artifact_sha256: str | None = None,
        manifest_sha256: str | None = None,
        experiment_config_sha256: str | None = None,
        diagnostics_gate_sha256: str | None = None,
        scoring_deferred: bool = False,
        train600_manifest_sha256: str | None = None,
        trajectory_schedule_id: str | None = None,
        trajectory_variant_id: str = "base",
        trajectory_replica_id: int = 0,
    ) -> None:
        if max_turns <= 0:
            raise ValueError("max_turns must be positive")
        self.client = client
        self.local_media_transport = (
            "path"
            if bool(getattr(client, "local_file_urls_as_paths", False))
            else "file_url"
        )
        self.model = model
        self.index = VideoIndex(video_root)
        self.frame_tool = frame_tool or FrameTool(
            frame_root, max_frames_per_call=max_frames_per_call
        )
        self.max_turns = max_turns
        self.controller_max_tokens = controller_max_tokens
        self.perception_max_tokens = perception_max_tokens
        self.judge_max_tokens = judge_max_tokens
        self.seed = seed
        self.candidate_results_sha256 = candidate_results_sha256
        self.model_artifact_sha256 = model_artifact_sha256
        self.manifest_sha256 = manifest_sha256
        self.experiment_config_sha256 = experiment_config_sha256
        self.diagnostics_gate_sha256 = diagnostics_gate_sha256
        self.scoring_deferred = bool(scoring_deferred)
        self.train600_manifest_sha256 = train600_manifest_sha256
        self.trajectory_schedule_id = trajectory_schedule_id
        self.trajectory_variant_id = str(trajectory_variant_id)
        self.trajectory_replica_id = int(trajectory_replica_id)
        if self.scoring_deferred and not self.train600_manifest_sha256:
            raise ValueError("deferred trajectories require train600_manifest_sha256")
        if self.scoring_deferred and not self.trajectory_schedule_id:
            raise ValueError("deferred trajectories require trajectory_schedule_id")
        if self.trajectory_replica_id < 0:
            raise ValueError("trajectory_replica_id must be non-negative")
        if not self.trajectory_variant_id.strip():
            raise ValueError("trajectory_variant_id cannot be empty")
        if self.trajectory_variant_id not in _ALLOWED_TRAJECTORY_VARIANTS:
            raise ValueError(
                "unsupported Perception-Memory trajectory variant: "
                f"{self.trajectory_variant_id}"
            )
        if (
            self.trajectory_variant_id in RESCUE_TRAJECTORY_VARIANTS
            and not self.scoring_deferred
        ):
            raise ValueError("rescue trajectory variants require scoring_deferred=True")

    def static_audit_fields(self) -> dict[str, Any]:
        """Return immutable fields needed by resume and result auditing."""

        source = Path(__file__).resolve()
        package_root = source.parent
        implementation_files = (
            source,
            package_root / "privacy.py",
            package_root / "eva_official.py",
            package_root / "qwen_agents" / "core.py",
        )
        implementation_hashes = {
            path.relative_to(package_root).as_posix(): hashlib.sha256(
                path.read_bytes()
            ).hexdigest()
            for path in implementation_files
        }
        implementation_bundle_sha256 = hashlib.sha256(
            json.dumps(
                implementation_hashes,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        selector = getattr(self.frame_tool, "selector_identity", None)
        if selector is None:
            selector = {
                "kind": "injected_frame_tool",
                "type": type(self.frame_tool).__qualname__,
            }
        return {
            "backend": self.backend,
            "agent_version": self.version,
            "perception_normalization_version": PERCEPTION_NORMALIZATION_VERSION,
            "model": self.model,
            "implementation_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            "implementation_bundle_sha256": implementation_bundle_sha256,
            "frame_tool_identity": copy.deepcopy(selector),
            "max_turns": self.max_turns,
            "max_frames_per_call": getattr(
                self.frame_tool, "max_frames_per_call", None
            ),
            "controller_max_tokens": self.controller_max_tokens,
            "perception_max_tokens": self.perception_max_tokens,
            "judge_max_tokens": self.judge_max_tokens,
            "seed": self.seed,
            "candidate_results_sha256": self.candidate_results_sha256,
            "model_artifact_sha256": self.model_artifact_sha256,
            "manifest_sha256": self.manifest_sha256,
            "experiment_config_sha256": self.experiment_config_sha256,
            "diagnostics_gate_sha256": self.diagnostics_gate_sha256,
            "local_media_transport": self.local_media_transport,
            "scoring_deferred": self.scoring_deferred,
            "train600_manifest_sha256": self.train600_manifest_sha256,
            "trajectory_schedule_id": self.trajectory_schedule_id,
            "trajectory_variant_id": self.trajectory_variant_id,
            "trajectory_replica_id": self.trajectory_replica_id,
        }

    def run_fingerprint(self) -> str:
        payload = json.dumps(
            self.static_audit_fields(),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def _chat(
        self,
        trace: list[dict[str, Any]],
        messages: list[dict[str, Any]],
        *,
        stage: str,
        max_tokens: int,
        seed_offset: int,
        json_mode: bool,
        step_index: int | None = None,
        prefix_index: int | None = None,
    ) -> str:
        assert_annotation_free_request({"messages": messages})
        if stage in {"controller", "confirmation_controller"} and messages_have_media(
            messages
        ):
            raise AssertionError("controller request contains media")
        prompt_hash = hashlib.sha256(
            json.dumps(
                messages,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                default=str,
            ).encode("utf-8")
        ).hexdigest()
        result = self.client.chat(
            self.model,
            messages,
            max_tokens=max_tokens,
            temperature=0.0,
            seed=self.seed + seed_offset,
            response_format={"type": "json_object"} if json_mode else None,
            chat_template_kwargs={"enable_thinking": False},
            extra_body={"return_token_ids": True},
        )
        trace.append(
            {
                "stage": stage,
                "model": self.model,
                "messages": copy.deepcopy(messages),
                "content": result.content,
                "reasoning_content": result.reasoning_content,
                "finish_reason": result.finish_reason,
                "usage": copy.deepcopy(result.usage),
                "latency_s": result.latency_s,
                "seed": self.seed + seed_offset,
                "step_index": step_index,
                "prefix_index": prefix_index,
                "prompt_hash": prompt_hash,
            }
        )
        if result.finish_reason == "length":
            raise RuntimeError(f"{stage} response was truncated")
        return result.content

    def _completeness(
        self,
        sample: ModelSample,
        memory: EvidenceMemory,
        trace: list[dict[str, Any]],
        seed_offset: int,
        step_index: int | None = None,
        prefix_index: int | None = None,
    ) -> CompletenessDecision | None:
        content = self._chat(
            trace,
            build_completeness_messages(sample, memory),
            stage="completeness",
            max_tokens=self.judge_max_tokens,
            seed_offset=seed_offset,
            json_mode=True,
            step_index=step_index,
            prefix_index=prefix_index,
        )
        return parse_completeness(content)

    def _judge(
        self,
        sample: ModelSample,
        memory: EvidenceMemory,
        trace: list[dict[str, Any]],
        seed_offset: int,
        stage: str,
        step_index: int | None = None,
        prefix_index: int | None = None,
    ) -> EvidenceDecision | None:
        content = self._chat(
            trace,
            build_judge_messages(sample, memory),
            stage=stage,
            max_tokens=self.judge_max_tokens,
            seed_offset=seed_offset,
            json_mode=True,
            step_index=step_index,
            prefix_index=prefix_index,
        )
        return parse_evidence_decision(
            content, sample.option_letters, memory.evidence_ids
        )

    def run(self, sample: ModelSample) -> dict[str, Any]:
        request_trace: list[dict[str, Any]] = []
        tool_steps: list[dict[str, Any]] = []
        perception_states: list[dict[str, Any]] = []
        memory = EvidenceMemory(sample.option_letters)
        candidate = (
            sample.candidate_answer
            if sample.candidate_answer in sample.option_letters
            else None
        )
        final = FinalDecision(candidate, "candidate_fallback", candidate is not None)
        complete = False
        stop_reason = "error"
        error: str | None = None
        annotation_check = "passed"
        tool_latency = 0.0

        try:
            video = self.index.resolve(sample.video)
            session = self.frame_tool.open_session(
                video, f"pm-{sample.dataset}-{sample.sample_id}"
            )
            feedback = ""
            rescue_request = (
                rescue_frame_request(
                    self.trajectory_variant_id, float(session.metadata["duration"])
                )
                if self.scoring_deferred
                and self.trajectory_variant_id in RESCUE_TRAJECTORY_VARIANTS
                else None
            )
            for turn in range(self.max_turns):
                if turn == 0 and rescue_request is not None:
                    controller_messages = build_controller_messages(
                        sample, memory, session.metadata, feedback=feedback
                    )
                    assert_annotation_free_request({"messages": controller_messages})
                    controller_text = _controller_tool_call(rescue_request)
                    request_trace.append(
                        {
                            "stage": "controller",
                            "model": self.model,
                            "messages": copy.deepcopy(controller_messages),
                            "content": controller_text,
                            "reasoning_content": "",
                            "finish_reason": "source_cached_action",
                            "usage": {
                                "prompt_tokens": 0,
                                "completion_tokens": 0,
                                "total_tokens": 0,
                            },
                            "latency_s": 0.0,
                            "seed": self.seed,
                            "step_index": turn,
                            "prefix_index": -1,
                            "prompt_hash": _messages_sha256(controller_messages),
                            "source_cached_action": True,
                            "trajectory_variant_id": self.trajectory_variant_id,
                        }
                    )
                else:
                    controller_text = self._chat(
                        request_trace,
                        build_controller_messages(
                            sample, memory, session.metadata, feedback=feedback
                        ),
                        stage="controller",
                        max_tokens=self.controller_max_tokens,
                        seed_offset=turn * 10,
                        json_mode=False,
                        step_index=turn,
                        prefix_index=len(perception_states) - 1,
                    )
                action = parse_controller_action(controller_text)
                if action is None:
                    request_trace[-1]["action_accepted"] = False
                    request_trace[-1]["action_rejection_reason"] = (
                        "invalid_controller_action"
                    )
                    raise ValueError("invalid controller action")
                if action.action == "stop":
                    decision = self._completeness(
                        sample,
                        memory,
                        request_trace,
                        turn * 10 + 1,
                        step_index=turn,
                        prefix_index=len(perception_states) - 1,
                    )
                    if (
                        decision is not None
                        and decision.evidence_complete
                        and memory.event_ledger
                    ):
                        request_trace[-2]["action_accepted"] = True
                        complete = True
                        stop_reason = "evidence_complete"
                        if perception_states:
                            perception_states[-1]["evidence_complete"] = True
                        break
                    request_trace[-2]["action_accepted"] = False
                    request_trace[-2]["action_rejection_reason"] = "incomplete_evidence"
                    missing = (
                        decision.missing_evidence
                        if decision
                        else ("invalid completeness response",)
                    )
                    feedback = "Stop rejected; missing evidence: " + "; ".join(missing)
                    continue

                assert action.request is not None
                requested_interval = (
                    action.request.start_time,
                    action.request.end_time,
                )
                if duplicate_interval(requested_interval, memory.observed_intervals):
                    request_trace[-1]["action_accepted"] = False
                    request_trace[-1]["action_rejection_reason"] = "duplicate_interval"
                    feedback = "Requested interval duplicates prior evidence; choose a different interval."
                    continue
                observation = session.select(action.request)
                request_trace[-1]["action_accepted"] = True
                tool_step = ToolStep.from_observation("perception", observation)
                tool_steps.append(asdict(tool_step))
                tool_latency += observation.latency_s
                state_index = len(perception_states)
                perception_text = self._chat(
                    request_trace,
                    build_perception_messages(
                        sample, observation, action.request.evidence_request
                    ),
                    stage="perception",
                    max_tokens=self.perception_max_tokens,
                    seed_offset=turn * 10 + 2,
                    json_mode=True,
                    step_index=state_index,
                    prefix_index=state_index,
                )
                state = parse_perception_state(perception_text, sample.option_letters)
                if state is None:
                    request_trace[-1]["state_validation_error"] = (
                        "invalid perception state schema"
                    )
                    raise ValueError("invalid perception state")
                try:
                    state = validate_perception_state_observation(state, observation)
                except ValueError as exc:
                    request_trace[-1]["state_validation_error"] = str(exc)
                    raise
                memory.merge(state)
                perception_states.append(
                    {
                        "step_index": state_index,
                        "turn_index": turn,
                        "request": action.request.to_tool_arguments(),
                        "frame_paths": list(observation.frame_paths),
                        "timestamps": list(observation.timestamps),
                        "perception": state.to_dict(),
                        "perception_response": state.to_dict(),
                        "memory_after": memory.to_dict(),
                        "evidence_complete": False,
                        "judge_confirmations": [],
                    }
                )
                feedback = ""
            else:
                decision = self._completeness(
                    sample,
                    memory,
                    request_trace,
                    self.max_turns * 10 + 1,
                    step_index=(
                        len(perception_states) - 1 if perception_states else None
                    ),
                    prefix_index=len(perception_states) - 1,
                )
                complete = bool(
                    decision and decision.evidence_complete and memory.event_ledger
                )
                if complete and perception_states:
                    perception_states[-1]["evidence_complete"] = True
                stop_reason = (
                    "max_turns_complete" if complete else "max_turns_incomplete"
                )

            if complete:
                first = self._judge(
                    sample,
                    memory,
                    request_trace,
                    1001,
                    "evidence_judge",
                    step_index=(
                        len(perception_states) - 1 if perception_states else None
                    ),
                    prefix_index=len(perception_states) - 1,
                )
                if perception_states:
                    perception_states[-1]["judge_confirmations"].append(
                        {
                            "seed": self.seed + 1001,
                            "prediction": first.answer if first else None,
                            "evidence_complete": True,
                            "annotation_leak_check": "passed",
                            "error": None
                            if first
                            else "invalid evidence judge response",
                        }
                    )
                confirmation = None
                if (
                    first is not None
                    and candidate is not None
                    and first.answer != candidate
                ):
                    confirmation_text = self._chat(
                        request_trace,
                        build_confirmation_controller_messages(
                            sample,
                            memory,
                            session.metadata,
                            (candidate, first.answer),
                        ),
                        stage="confirmation_controller",
                        max_tokens=self.controller_max_tokens,
                        seed_offset=2001,
                        json_mode=False,
                        step_index=len(perception_states),
                        prefix_index=len(perception_states) - 1,
                    )
                    confirmation_action = parse_controller_action(confirmation_text)
                    if (
                        confirmation_action is not None
                        and confirmation_action.action == "observe"
                        and confirmation_action.request is not None
                    ):
                        confirmation_observation = session.select(
                            confirmation_action.request
                        )
                        request_trace[-1]["action_accepted"] = True
                        confirmation_tool = ToolStep.from_observation(
                            "confirmation_perception", confirmation_observation
                        )
                        tool_steps.append(asdict(confirmation_tool))
                        tool_latency += confirmation_observation.latency_s
                        confirmation_perception_text = self._chat(
                            request_trace,
                            build_perception_messages(
                                sample,
                                confirmation_observation,
                                confirmation_action.request.evidence_request,
                            ),
                            stage="confirmation_perception",
                            max_tokens=self.perception_max_tokens,
                            seed_offset=2002,
                            json_mode=True,
                            step_index=len(perception_states),
                            prefix_index=len(perception_states),
                        )
                        confirmation_state = parse_perception_state(
                            confirmation_perception_text, sample.option_letters
                        )
                        if confirmation_state is None:
                            request_trace[-1]["state_validation_error"] = (
                                "invalid confirmation perception state schema"
                            )
                            raise ValueError("invalid confirmation perception state")
                        try:
                            confirmation_state = validate_perception_state_observation(
                                confirmation_state, confirmation_observation
                            )
                        except ValueError as exc:
                            request_trace[-1]["state_validation_error"] = str(exc)
                            raise
                        memory.merge(confirmation_state)
                        perception_states.append(
                            {
                                "step_index": len(perception_states),
                                "turn_index": len(perception_states),
                                "request": confirmation_action.request.to_tool_arguments(),
                                "frame_paths": list(
                                    confirmation_observation.frame_paths
                                ),
                                "timestamps": list(confirmation_observation.timestamps),
                                "perception": confirmation_state.to_dict(),
                                "perception_response": confirmation_state.to_dict(),
                                "memory_after": memory.to_dict(),
                                "evidence_complete": False,
                                "judge_confirmations": [],
                                "stage": "change_confirmation",
                            }
                        )
                        confirmed_complete = self._completeness(
                            sample,
                            memory,
                            request_trace,
                            2003,
                            step_index=len(perception_states) - 1,
                            prefix_index=len(perception_states) - 1,
                        )
                        if confirmed_complete and confirmed_complete.evidence_complete:
                            perception_states[-1]["evidence_complete"] = True
                            confirmation = self._judge(
                                sample,
                                memory,
                                request_trace,
                                2004,
                                "confirmation_judge",
                                step_index=len(perception_states) - 1,
                                prefix_index=len(perception_states) - 1,
                            )
                    if (
                        confirmation_action is None
                        or confirmation_action.action != "observe"
                        or confirmation_action.request is None
                    ):
                        request_trace[-1]["action_accepted"] = False
                        request_trace[-1]["action_rejection_reason"] = (
                            "invalid_confirmation_action"
                        )
                    if perception_states:
                        perception_states[-1]["judge_confirmations"].append(
                            {
                                "seed": self.seed + 2004,
                                "prediction": confirmation.answer
                                if confirmation
                                else None,
                                "evidence_complete": confirmation is not None,
                                "annotation_leak_check": "passed",
                                "error": None
                                if confirmation
                                else "visual confirmation failed",
                            }
                        )
                final = apply_candidate_gate(
                    candidate=candidate,
                    first=first,
                    confirmation=confirmation,
                    complete=complete,
                    memory=memory,
                )
            else:
                stop_reason = stop_reason or "incomplete_evidence"
        except AnnotationLeakError as exc:
            annotation_check = "failed"
            error = f"{type(exc).__name__}: {exc}"
            stop_reason = "annotation_leak"
        except Exception as exc:  # sample-local failure always falls back
            error = f"{type(exc).__name__}: {exc}"
            stop_reason = "runtime_error"

        prompt_tokens = sum(
            int(item["usage"].get("prompt_tokens", 0) or 0) for item in request_trace
        )
        completion_tokens = sum(
            int(item["usage"].get("completion_tokens", 0) or 0)
            for item in request_trace
        )
        total_tokens = sum(
            int(item["usage"].get("total_tokens", 0) or 0) for item in request_trace
        )
        visual_values = [
            (
                _visual_tokens(item["usage"])
                if messages_have_media(item["messages"])
                else 0
            )
            for item in request_trace
        ]
        visual_complete = all(item is not None for item in visual_values)
        total_usage_complete = all(
            all(
                isinstance(item["usage"].get(key), (int, float))
                and not isinstance(item["usage"].get(key), bool)
                for key in ("prompt_tokens", "completion_tokens", "total_tokens")
            )
            for item in request_trace
        )
        return {
            **self.static_audit_fields(),
            "run_fingerprint": self.run_fingerprint(),
            "public_sample": {
                "dataset": sample.dataset,
                "sample_id": sample.sample_id,
                "video": sample.video,
                "question": sample.question,
                "choices": dict(sample.choices),
            },
            "trajectory_id": (
                f"{sample.dataset}:{sample.sample_id}:"
                f"{self.trajectory_schedule_id}:{self.trajectory_variant_id}:"
                f"{self.trajectory_replica_id}"
                if self.trajectory_schedule_id
                else None
            ),
            "manifest_sha256": self.manifest_sha256,
            "dataset_manifest_sha256": self.manifest_sha256,
            "train600_manifest_sha256": self.train600_manifest_sha256,
            "config_sha256": self.experiment_config_sha256 or self.run_fingerprint(),
            "scoring_deferred": self.scoring_deferred,
            "dataset": sample.dataset,
            "sample_id": sample.sample_id,
            "candidate_answer": candidate,
            "prediction": final.prediction,
            "final_prediction": final.prediction,
            "decision_source": final.source,
            "candidate_changed": bool(
                candidate is not None
                and final.prediction is not None
                and final.prediction != candidate
            ),
            "fallback_to_candidate": final.fallback_to_candidate,
            "fallback_used": final.fallback_to_candidate,
            "candidate_rerun": 0,
            "annotation_leak_check": annotation_check,
            "event_ledger": memory.to_dict()["event_ledger"],
            "option_ledger": memory.to_dict()["option_ledger"],
            "unresolved": list(memory.unresolved),
            "observed_intervals": [list(item) for item in memory.observed_intervals],
            "evidence_complete": complete,
            "stop_reason": stop_reason,
            "tool_steps": tool_steps,
            "perception_states": perception_states,
            "judge_answers": [
                item["content"]
                for item in request_trace
                if item["stage"] in {"evidence_judge", "confirmation_judge"}
            ],
            "request_trace": request_trace,
            "rounds": len(perception_states),
            "turn_count": len(request_trace),
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "visual_tokens": (
                sum(int(item) for item in visual_values if item is not None)
                if visual_complete
                else None
            ),
            "visual_token_accounting_complete": visual_complete,
            "visual_usage_complete": visual_complete,
            "total_tokens": total_tokens,
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens,
            },
            "agent_total_tokens_complete": total_usage_complete,
            "latency_s": sum(float(item["latency_s"]) for item in request_trace)
            + tool_latency,
            "error": error,
            "error_type": error.split(":", 1)[0] if error else None,
        }


__all__ = [
    "CompletenessDecision",
    "ControllerAction",
    "EvidenceDecision",
    "EvidenceMemory",
    "FinalDecision",
    "OptionLedger",
    "OptionObservation",
    "PerceptionMemoryEvaEvaluator",
    "PerceptionState",
    "PERCEPTION_NORMALIZATION_VERSION",
    "RESCUE_TRAJECTORY_VARIANTS",
    "TimestampedFact",
    "apply_candidate_gate",
    "build_completeness_messages",
    "build_confirmation_controller_messages",
    "build_controller_messages",
    "build_judge_messages",
    "build_perception_messages",
    "duplicate_interval",
    "interval_iou",
    "messages_have_media",
    "parse_completeness",
    "parse_controller_action",
    "parse_evidence_decision",
    "parse_perception_state",
    "normalize_perception_state",
    "rescue_frame_request",
    "validate_perception_state_observation",
]
