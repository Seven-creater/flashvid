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
import inspect
import json
import math
import re
import urllib.parse
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .datasets import VideoIndex
from .perception_memory_visual_csv import (
    VISUAL_CSV_SCHEMA_VERSION,
    VisualCsvJob,
    build_visual_csv_messages,
    frame_content_sha256s,
    parse_visual_csv_response,
    visual_csv_response_format,
)
from .privacy import AnnotationLeakError, assert_annotation_free_request
from .qwen_agents.core import (
    ChatClient,
    DuplicateFrameRequestError,
    FrameObservation,
    FrameRequest,
    FrameTool,
    ToolStep,
    observation_content,
)
from .question_time import parse_question_time_range
from .schemas import ModelSample


_TOOL_CALL_RE = re.compile(r"^\s*<tool_call>\s*(\{.*\})\s*</tool_call>\s*$", re.DOTALL)
_JSON_FENCE_RE = re.compile(
    r"^\s*```(?:json)?\s*\n(\{.*\})\n```\s*$",
    re.DOTALL | re.IGNORECASE,
)
_SPACE_RE = re.compile(r"\s+")
REPAIR_ONLY_TRAJECTORY_VARIANTS = frozenset({"rescue_explicit_time"})
DURATION_RESCUE_TRAJECTORY_VARIANTS = frozenset(
    {
        "rescue_global32",
        "rescue_global64",
        "rescue_first_half64",
        "rescue_second_half64",
    }
)
RESCUE_TRAJECTORY_VARIANTS = frozenset(
    {*DURATION_RESCUE_TRAJECTORY_VARIANTS, *REPAIR_ONLY_TRAJECTORY_VARIANTS}
)
_ALLOWED_TRAJECTORY_VARIANTS = frozenset({"base", *RESCUE_TRAJECTORY_VARIANTS})
_MAX_NO_NOVEL_ACTION_REJECTIONS = 2
_MAX_ANSWERER_CITED_FRAMES = 16
RUNTIME_VERIFIER_MAX_FRAMES = 16
RUNTIME_VERIFIER_FRAME_SELECTION_POLICY = "uniform_full_inventory_endpoints_v1"
ROLE_SEPARATED_RUNTIME_VERSION = "role_separated_visual_csv_v1"
PERCEPTION_RESPONSE_SCHEMA_VERSION = "perception_state_json_schema_v1"
CONTROLLER_OUTPUT_CONSTRAINT_VERSION = "eva_tool_call_regex_v1"
EVIDENCE_REQUEST_CONTENT_SIGNATURE_VERSION = "evidence_request_content_signature_v1"
_CONTROLLER_FRAME_COUNTS = (8, 16, 32, 64, 128)
_EVIDENCE_REQUEST_IGNORED_WORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "been",
        "being",
        "by",
        "confirmed",
        "definitively",
        "directly",
        "determined",
        "e",
        "eg",
        "evidence",
        "for",
        "from",
        "g",
        "identified",
        "in",
        "is",
        "it",
        "not",
        "observed",
        "of",
        "on",
        "or",
        "seen",
        "shown",
        "the",
        "to",
        "unclear",
        "unknown",
        "unresolved",
        "visible",
        "was",
        "were",
        "whether",
        "with",
    }
)
_MAX_EVIDENCE_REQUEST_CHARS = 160
PERCEPTION_MEMORY_ROLE_NAMES = (
    "planner",
    "observer",
    "verifier",
    "answerer",
)
_PERCEPTION_MEMORY_STAGE_ROLES = {
    "controller": "planner",
    "confirmation_controller": "planner",
    "perception": "observer",
    "confirmation_perception": "observer",
    "completeness": "verifier",
    "evidence_judge": "answerer",
    "confirmation_judge": "answerer",
}


def role_prompt_schema_bundle_sha256() -> str:
    """Fingerprint only the frozen four-role prompt/schema surface."""

    builders = (
        controller_structured_outputs,
        parse_controller_action,
        build_role_separated_controller_messages,
        build_role_separated_confirmation_controller_messages,
        build_perception_messages,
        build_perception_retry_messages,
        perception_response_format,
        bind_perception_state,
        select_runtime_verifier_frames,
        map_runtime_verifier_frame_indices,
        build_runtime_visual_csv_messages,
        parse_visual_csv_response,
        build_cited_judge_messages,
        parse_evidence_decision,
    )
    payload = {
        "role_separated_runtime_version": ROLE_SEPARATED_RUNTIME_VERSION,
        "perception_response_schema_version": PERCEPTION_RESPONSE_SCHEMA_VERSION,
        "controller_output_constraint_version": CONTROLLER_OUTPUT_CONSTRAINT_VERSION,
        "evidence_request_content_signature_version": (
            EVIDENCE_REQUEST_CONTENT_SIGNATURE_VERSION
        ),
        "visual_csv_schema_version": VISUAL_CSV_SCHEMA_VERSION,
        "runtime_verifier_max_frames": RUNTIME_VERIFIER_MAX_FRAMES,
        "runtime_verifier_frame_selection_policy": (
            RUNTIME_VERIFIER_FRAME_SELECTION_POLICY
        ),
        "contracts": {
            function.__name__: inspect.getsource(function) for function in builders
        },
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


@dataclass(frozen=True)
class PerceptionMemoryRoleBinding:
    """One immutable model/client binding used by a runtime role."""

    client: ChatClient
    model: str
    artifact_sha256: str | None

    def __post_init__(self) -> None:
        if not callable(getattr(self.client, "chat", None)):
            raise ValueError("role binding client must implement chat")
        if not isinstance(self.model, str) or not self.model.strip():
            raise ValueError("role binding model must be non-empty text")
        if self.artifact_sha256 is not None and (
            len(self.artifact_sha256) != 64
            or any(
                character not in "0123456789abcdefABCDEF"
                for character in self.artifact_sha256
            )
        ):
            raise ValueError("role binding artifact_sha256 must be a SHA-256")
        object.__setattr__(self, "model", self.model.strip())
        if self.artifact_sha256 is not None:
            object.__setattr__(
                self, "artifact_sha256", self.artifact_sha256.lower()
            )


def perception_memory_role_for_stage(stage: str) -> str:
    """Map every model-call stage to exactly one frozen runtime role."""

    normalized = str(stage).strip().casefold()
    role = _PERCEPTION_MEMORY_STAGE_ROLES.get(normalized)
    if role is None:
        raise ValueError(f"unsupported Perception-Memory model stage: {stage!r}")
    return role


def validate_perception_memory_role_config(
    payload: Any,
) -> dict[str, dict[str, str]]:
    """Validate the exact public JSON schema accepted by ``--pm-role-config``."""

    if not isinstance(payload, Mapping) or set(payload) != set(
        PERCEPTION_MEMORY_ROLE_NAMES
    ):
        raise ValueError(
            "Perception-Memory role config must contain exactly planner, observer, "
            "verifier, and answerer"
        )
    validated: dict[str, dict[str, str]] = {}
    required = {"base_url", "model", "artifact_sha256"}
    for role in PERCEPTION_MEMORY_ROLE_NAMES:
        raw = payload[role]
        if not isinstance(raw, Mapping) or set(raw) != required:
            raise ValueError(
                f"Perception-Memory role {role} must contain exactly "
                "base_url, model, and artifact_sha256"
            )
        base_url = str(raw["base_url"] or "").strip().rstrip("/")
        parsed = urllib.parse.urlparse(base_url)
        if (
            parsed.scheme not in {"http", "https"}
            or not parsed.netloc
            or parsed.query
            or parsed.fragment
        ):
            raise ValueError(f"Perception-Memory role {role} requires an HTTP base_url")
        model = str(raw["model"] or "").strip()
        if not model:
            raise ValueError(f"Perception-Memory role {role} model cannot be empty")
        artifact = str(raw["artifact_sha256"] or "").strip().lower()
        if len(artifact) != 64 or any(
            character not in "0123456789abcdef" for character in artifact
        ):
            raise ValueError(
                f"Perception-Memory role {role} artifact_sha256 must be a SHA-256"
            )
        validated[role] = {
            "base_url": base_url,
            "model": model,
            "artifact_sha256": artifact,
        }
    return validated


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


def _source_requested_intervals(
    value: Sequence[Sequence[float]] | None,
) -> tuple[tuple[float, float], ...]:
    if not value:
        raise ValueError(
            "rescue_explicit_time requires repair source requested intervals"
        )
    intervals: list[tuple[float, float]] = []
    for index, raw in enumerate(value):
        if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
            raise ValueError(f"source requested interval {index} must be an array")
        if len(raw) != 2:
            raise ValueError(f"source requested interval {index} must have two values")
        start = _finite_number(raw[0], f"source requested interval {index} start")
        end = _finite_number(raw[1], f"source requested interval {index} end")
        if end <= start:
            raise ValueError(f"source requested interval {index} is invalid")
        intervals.append((start, end))
    return tuple(intervals)


def explicit_time_rescue_request(
    question: str,
    video_duration: float,
    source_requested_intervals: Sequence[Sequence[float]] | None,
) -> tuple[FrameRequest, dict[str, Any]]:
    """Build one dense, question-time rescue request with source-mismatch proof."""

    duration = _finite_number(video_duration, "video_duration")
    if duration <= 0:
        raise ValueError("video_duration must be positive")
    parsed = parse_question_time_range(question)
    selected = parse_question_time_range(question, padding_s=1.0)
    if parsed is None or selected is None:
        raise ValueError("rescue_explicit_time requires a public question timestamp")
    source_intervals = _source_requested_intervals(source_requested_intervals)
    source_mismatch = all(
        max(parsed[0], source[0]) >= min(parsed[1], source[1])
        for source in source_intervals
    )
    if not source_mismatch:
        raise ValueError("rescue_explicit_time requires a source interval mismatch")

    start = max(0.0, selected[0])
    end = min(duration, selected[1])
    if start >= duration or end <= start:
        raise ValueError("question timestamp is outside the video duration")
    requested_frames = max(1, int(math.ceil((end - start) * 4.0)))
    sampling = {"fps": 4.0} if requested_frames <= 96 else {"nframes": 96}
    request = FrameRequest(
        start_time=start,
        end_time=end,
        resize=1.0,
        evidence_request=(
            "Record the directly visible action, state change, ordering, count, "
            "or text needed to distinguish the answer options at the explicit "
            "time stated in the public question."
        ),
        **sampling,
    )
    return request, {
        "mode": "rescue_explicit_time",
        "parsed_time_source": "public_question",
        "parsed_time_range": list(parsed),
        "selected_interval": [start, end],
        "source_requested_intervals": [list(item) for item in source_intervals],
        "source_interval_mismatch": True,
        "sampling": sampling,
        "max_frames": 96,
    }


def rescue_frame_request(
    variant: str,
    video_duration: float,
    *,
    question: str | None = None,
    source_requested_intervals: Sequence[Sequence[float]] | None = None,
) -> FrameRequest | None:
    """Return one annotation-free, duration-only rescue coverage request."""

    name = str(variant).strip()
    if name == "base":
        return None
    if name not in RESCUE_TRAJECTORY_VARIANTS:
        raise ValueError(f"unsupported Perception-Memory trajectory variant: {name}")
    if name == "rescue_explicit_time":
        if question is None:
            raise ValueError("rescue_explicit_time requires the public question")
        request, _audit = explicit_time_rescue_request(
            question, video_duration, source_requested_intervals
        )
        return request
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


def _controller_rejection_reason(text: str, video_duration: float) -> str:
    """Distinguish malformed calls from model-proposed invalid intervals."""

    match = _TOOL_CALL_RE.fullmatch((text or "").strip())
    if match is None:
        return "invalid_controller_action"
    try:
        payload = json.loads(match.group(1))
        arguments = payload["arguments"]
        start = _finite_number(arguments["start_time"], "controller start_time")
        end = _finite_number(arguments["end_time"], "controller end_time")
    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
        return "invalid_controller_action"
    if start < 0.0 or end <= start or end > video_duration:
        return "invalid_interval"
    return "invalid_controller_action"


def _request_interval_rejection_reason(
    request: FrameRequest, video_duration: float
) -> str | None:
    if (
        request.start_time < 0.0
        or request.end_time <= request.start_time
        or request.end_time > video_duration
    ):
        return "invalid_interval"
    return None


def _controller_retry_feedback(
    reason: str,
    video_duration: float,
    *,
    allow_stop: bool = True,
    requested_interval: tuple[float, float] | None = None,
    observed_intervals: Sequence[tuple[float, float]] = (),
) -> str:
    """Return candidate-blind, format-strict feedback for one controller retry."""

    if reason == "duplicate_interval":
        detail = (
            "The previous interval duplicated already observed evidence. Select a "
            "different, non-overlapping interval needed by the unresolved evidence."
        )
        if requested_interval is not None:
            detail += (
                " The rejected interval was "
                f"[{requested_interval[0]:.3f}, {requested_interval[1]:.3f}]."
            )
        if observed_intervals:
            intervals = ", ".join(
                f"[{start:.3f}, {end:.3f}]" for start, end in observed_intervals
            )
            detail += f" Already observed intervals: {intervals}."
    elif reason == "invalid_interval":
        detail = (
            "The previous interval was invalid. It must satisfy "
            f"0 <= start_time < end_time <= {video_duration:.3f}."
        )
    elif reason == "controller_response_truncated":
        detail = "The previous response was truncated; return one compact action."
    else:
        detail = "The previous response did not match the official EVA action schema."
    stop_action = (
        "If the text ledger is ready for an independent completeness check, return "
        'exactly {"action":"stop"}. Otherwise, '
        if allow_stop
        else ""
    )
    return (
        f"{detail} Retry without prose or markdown. {stop_action}return exactly one "
        "official EVA <tool_call> JSON object whose tool is frame_select. Its "
        "arguments must contain numeric start_time and end_time; resize exactly "
        "0.75; a short evidence_request without quotation marks; and nframes equal "
        "to one of 8, 16, 32, 64, or 128. Choose the "
        "interval from the question and current evidence gap; do not copy an "
        "illustrative interval."
    )


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


def _evidence_content_signature(text: str) -> tuple[str, ...]:
    """Extract deterministic content words from an evidence-gap description."""

    return tuple(
        dict.fromkeys(
            token
            for token in re.findall(r"[a-z0-9]+", str(text).casefold())
            if len(token) > 1 and token not in _EVIDENCE_REQUEST_IGNORED_WORDS
        )
    )


def evidence_request_addresses_unresolved(
    evidence_request: str, unresolved: Sequence[str]
) -> bool:
    """Require explicit-role plans to name the unresolved evidence they pursue."""

    if not unresolved:
        return True
    request_tokens = set(_evidence_content_signature(evidence_request))
    if not request_tokens:
        return False
    for item in unresolved:
        needed = set(_evidence_content_signature(str(item)))
        if needed and needed.issubset(request_tokens):
            return True
    return False


def controller_structured_outputs(*, allow_stop: bool) -> dict[str, Any]:
    """Constrain vLLM to one canonical official EVA action surface form."""

    number = r"(?:0|[1-9][0-9]*)(?:\.[0-9]+)?"
    tool_call = (
        r'<tool_call>\{"arguments":\{"end_time":'
        + number
        + r',"evidence_request":"[^"\\]{1,160}","nframes":'
        + r'(?:8|16|32|64|128),"resize":0\.75,"start_time":'
        + number
        + r'\},"tool":"frame_select"\}</tool_call>'
    )
    regex = (
        rf'(?:{tool_call}|\{{"action":"stop"\}})'
        if allow_stop
        else tool_call
    )
    return {"regex": regex}


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
    text: str,
    valid_letters: Iterable[str],
    *,
    allow_role_separated_schema: bool = False,
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
        role_separated_required = required - {
            "evidence_sufficient",
            "next_evidence_needed",
        }
        expected = role_separated_required if allow_role_separated_schema else required
        if not isinstance(payload, dict) or set(payload) != expected:
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
        if "evidence_sufficient" in payload and not isinstance(
            payload["evidence_sufficient"], bool
        ):
            return None
        next_needed = payload.get("next_evidence_needed", "")
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
            evidence_sufficient=bool(payload.get("evidence_sufficient", False)),
            next_evidence_needed=_SPACE_RE.sub(" ", next_needed.strip()),
        )
    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
        return None


def bind_perception_state(
    text: str,
    valid_letters: Iterable[str],
    actual_timestamps: Sequence[float],
    *,
    allow_timestamp_schema: bool = True,
    allow_role_separated_schema: bool = False,
) -> tuple[PerceptionState | None, str]:
    """Bind zero-based frame references to exact sampled-frame timestamps.

    The shared binder keeps replay/SFT backward-compatible with legacy
    ``{time,fact}`` observations.  A runtime using the indexed prompt can set
    ``allow_timestamp_schema=False`` so its accepted outputs match the SFT
    target protocol exactly.
    """

    candidate = (text or "").strip()
    fenced = _JSON_FENCE_RE.fullmatch(candidate)
    if fenced is not None:
        candidate = fenced.group(1)
    try:
        payload = json.loads(candidate)
    except json.JSONDecodeError:
        return None, "invalid"
    if not isinstance(payload, dict):
        return None, "invalid"
    facts = payload.get("timestamped_facts")
    if not isinstance(facts, list):
        return None, "invalid"
    if not facts:
        return (
            parse_perception_state(
                text,
                valid_letters,
                allow_role_separated_schema=allow_role_separated_schema,
            ),
            "none",
        )
    if all(
        isinstance(item, dict) and set(item) == {"frame_index", "fact"}
        for item in facts
    ):
        try:
            timestamps = tuple(
                _finite_number(value, "actual frame timestamp")
                for value in actual_timestamps
            )
        except (TypeError, ValueError):
            return None, "invalid"
        rebound: list[dict[str, Any]] = []
        for item in facts:
            frame_index = item["frame_index"]
            if (
                isinstance(frame_index, bool)
                or not isinstance(frame_index, int)
                or not 0 <= frame_index < len(timestamps)
            ):
                return None, "invalid"
            rebound.append({"time": timestamps[frame_index], "fact": item["fact"]})
        payload = dict(payload)
        payload["timestamped_facts"] = rebound
        return (
            parse_perception_state(
                json.dumps(payload, ensure_ascii=False),
                valid_letters,
                allow_role_separated_schema=allow_role_separated_schema,
            ),
            "frame_index",
        )
    parsed = parse_perception_state(
        text,
        valid_letters,
        allow_role_separated_schema=allow_role_separated_schema,
    )
    if parsed is None or not allow_timestamp_schema:
        return None, "timestamp" if parsed is not None else "invalid"
    return parsed, "timestamp"


def perception_model_target(
    state: PerceptionState,
    actual_timestamps: Sequence[float],
    *,
    role_separated: bool = False,
) -> str:
    """Serialize a validated perception state using zero-based frame indices."""

    timestamps = tuple(
        _finite_number(value, "actual frame timestamp") for value in actual_timestamps
    )
    payload = state.to_dict()
    indexed_facts: list[dict[str, Any]] = []
    for fact in state.timestamped_facts:
        matches = [
            index
            for index, timestamp in enumerate(timestamps)
            if timestamp == fact.time
        ]
        if not matches:
            raise ValueError("perception fact is not bound to an actual frame")
        indexed_facts.append({"frame_index": matches[0], "fact": fact.fact})
    payload["timestamped_facts"] = indexed_facts
    if role_separated:
        payload.pop("evidence_sufficient", None)
        payload.pop("next_evidence_needed", None)
    return json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )


def perception_state_payload(
    state: PerceptionState, *, role_separated: bool = False
) -> dict[str, Any]:
    """Serialize one normalized observation without assigning stop responsibility."""

    payload = state.to_dict()
    if role_separated:
        payload.pop("evidence_sufficient", None)
        payload.pop("next_evidence_needed", None)
    return payload


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

    @classmethod
    def from_dict(
        cls, value: Mapping[str, Any], option_letters: Sequence[str]
    ) -> "EvidenceMemory":
        """Rehydrate one canonical public ledger without weakening its schema."""

        required = {
            "event_ledger",
            "option_ledger",
            "unresolved",
            "observed_intervals",
        }
        if not isinstance(value, Mapping) or set(value) != required:
            raise ValueError("evidence memory must contain exactly the ledger fields")
        letters = tuple(str(letter).strip().upper() for letter in option_letters)
        if (
            not letters
            or len(letters) != len(set(letters))
            or any(len(letter) != 1 or not "A" <= letter <= "H" for letter in letters)
        ):
            raise ValueError("evidence memory option labels must be unique A-H letters")

        raw_events = value["event_ledger"]
        if not isinstance(raw_events, list):
            raise ValueError("event_ledger must be an array")
        events: list[EvidenceEvent] = []
        evidence_ids: set[str] = set()
        for index, raw in enumerate(raw_events):
            if not isinstance(raw, Mapping) or set(raw) != {
                "evidence_id",
                "interval",
                "timestamp",
                "fact",
                "source",
            }:
                raise ValueError(f"event_ledger[{index}] has an invalid schema")
            evidence_id = _clean_text(
                raw["evidence_id"], f"event_ledger[{index}].evidence_id"
            )
            if not re.fullmatch(r"E\d{4}", evidence_id) or evidence_id in evidence_ids:
                raise ValueError("event_ledger evidence IDs must be unique E#### values")
            evidence_ids.add(evidence_id)
            interval = raw["interval"]
            if (
                not isinstance(interval, list)
                or len(interval) != 2
                or isinstance(interval[0], bool)
                or isinstance(interval[1], bool)
            ):
                raise ValueError(f"event_ledger[{index}].interval is invalid")
            start = _finite_number(interval[0], f"event_ledger[{index}].interval[0]")
            end = _finite_number(interval[1], f"event_ledger[{index}].interval[1]")
            if end <= start:
                raise ValueError(f"event_ledger[{index}].interval is invalid")
            timestamp = raw["timestamp"]
            parsed_timestamp = (
                None
                if timestamp is None
                else _finite_number(timestamp, f"event_ledger[{index}].timestamp")
            )
            events.append(
                EvidenceEvent(
                    evidence_id=evidence_id,
                    interval=(start, end),
                    timestamp=parsed_timestamp,
                    fact=_clean_text(raw["fact"], f"event_ledger[{index}].fact"),
                    source=_clean_text(raw["source"], f"event_ledger[{index}].source"),
                )
            )

        raw_options = value["option_ledger"]
        if not isinstance(raw_options, Mapping) or set(raw_options) != set(letters):
            raise ValueError("option_ledger must exactly match the public choices")
        options: dict[str, OptionLedger] = {}
        for letter in letters:
            raw = raw_options[letter]
            if not isinstance(raw, Mapping) or set(raw) != {
                "supports",
                "contradicts",
            }:
                raise ValueError(f"option_ledger.{letter} has an invalid schema")
            supports = _string_list(raw["supports"], f"option_ledger.{letter}.supports")
            contradicts = _string_list(
                raw["contradicts"], f"option_ledger.{letter}.contradicts"
            )
            if any(item not in evidence_ids for item in (*supports, *contradicts)):
                raise ValueError(f"option_ledger.{letter} cites unknown evidence")
            options[letter] = OptionLedger(
                tuple(dict.fromkeys(supports)), tuple(dict.fromkeys(contradicts))
            )

        raw_unresolved = value["unresolved"]
        unresolved = list(_string_list(raw_unresolved, "unresolved"))
        raw_intervals = value["observed_intervals"]
        if not isinstance(raw_intervals, list):
            raise ValueError("observed_intervals must be an array")
        observed_intervals: list[tuple[float, float]] = []
        for index, raw in enumerate(raw_intervals):
            if not isinstance(raw, list) or len(raw) != 2:
                raise ValueError(f"observed_intervals[{index}] is invalid")
            start = _finite_number(raw[0], f"observed_intervals[{index}][0]")
            end = _finite_number(raw[1], f"observed_intervals[{index}][1]")
            if end <= start:
                raise ValueError(f"observed_intervals[{index}] is invalid")
            observed_intervals.append((start, end))
        return cls(
            option_letters=letters,
            event_ledger=events,
            option_ledger=options,
            unresolved=list(dict.fromkeys(unresolved)),
            observed_intervals=observed_intervals,
        )

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

    def merge(self, state: PerceptionState) -> tuple[str, ...]:
        """Merge one observation and return the evidence IDs bound to this step.

        The returned IDs include exact, deterministically reused events.  This lets
        the confirmation gate distinguish evidence actually observed again in the
        confirmation step from unrelated IDs that merely exist in the final ledger.
        """

        self.observed_intervals.append(state.interval)
        bound_ids: list[str] = []

        def bind(evidence_id: str) -> None:
            if evidence_id not in bound_ids:
                bound_ids.append(evidence_id)

        for item in state.timestamped_facts:
            bind(
                self._evidence_id(
                    item.fact, state.interval, item.time, "timestamped_fact"
                )
            )
        for change in state.temporal_changes:
            bind(self._evidence_id(change, state.interval, None, "temporal_change"))
        for letter in self.option_letters:
            observation = state.option_evidence[letter]
            current = self.option_ledger[letter]
            supports = list(current.supports)
            contradicts = list(current.contradicts)
            for fact in observation.supports:
                for evidence_id in self._option_evidence_ids(
                    fact, state.interval, f"option_{letter}_support"
                ):
                    bind(evidence_id)
                    if evidence_id not in supports:
                        supports.append(evidence_id)
            for fact in observation.contradicts:
                for evidence_id in self._option_evidence_ids(
                    fact, state.interval, f"option_{letter}_contradiction"
                ):
                    bind(evidence_id)
                    if evidence_id not in contradicts:
                        contradicts.append(evidence_id)
            self.option_ledger[letter] = OptionLedger(
                tuple(supports), tuple(contradicts)
            )
        self.unresolved = list(dict.fromkeys(state.unresolved))
        return tuple(bound_ids)

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


def _controller_reference_request(
    sample: ModelSample,
    memory: EvidenceMemory,
    video_duration: float,
) -> FrameRequest | None:
    """Build one public-input-only action that is valid for the current state.

    This is a format reference, but it is deliberately executable: models that
    copy it still request a useful question-timestamp or coverage interval.  It
    never uses the Direct candidate or private annotation fields.
    """

    duration = _finite_number(video_duration, "video_duration")
    if duration <= 0.0:
        raise ValueError("video_duration must be positive")

    candidates: list[tuple[float, float, str]] = []
    explicit = parse_question_time_range(sample.question, padding_s=1.0)
    if explicit is not None:
        start = max(0.0, explicit[0])
        end = min(duration, explicit[1])
        if end > start:
            candidates.append(
                (
                    start,
                    end,
                    "Inspect the public question's stated time and record the "
                    "directly visible option-discriminative event.",
                )
            )

    if not memory.observed_intervals:
        candidates.append(
            (
                0.0,
                duration,
                "Build a uniform visual overview and record directly visible facts "
                "that distinguish the answer choices.",
            )
        )
    else:
        clipped = sorted(
            (
                max(0.0, float(start)),
                min(duration, float(end)),
            )
            for start, end in memory.observed_intervals
            if min(duration, float(end)) > max(0.0, float(start))
        )
        merged: list[list[float]] = []
        for start, end in clipped:
            if not merged or start > merged[-1][1]:
                merged.append([start, end])
            else:
                merged[-1][1] = max(merged[-1][1], end)
        cursor = 0.0
        gaps: list[tuple[float, float]] = []
        for start, end in merged:
            if start > cursor:
                gaps.append((cursor, start))
            cursor = max(cursor, end)
        if cursor < duration:
            gaps.append((cursor, duration))
        for start, end in sorted(
            gaps, key=lambda item: item[1] - item[0], reverse=True
        ):
            candidates.append(
                (
                    start,
                    end,
                    "Observe the largest remaining time gap and resolve missing "
                    "option-discriminative evidence.",
                )
            )

        # A whole-video overview can cover every timestamp while still leaving a
        # need for denser local inspection.  These public duration-relative windows
        # remain non-duplicates under the runtime IoU rule and provide valid format
        # references without a fixed 0-30 second anchor.
        candidates.extend(
            (
                (
                    0.0,
                    duration / 2.0,
                    "Inspect the first half for unresolved evidence.",
                ),
                (
                    duration / 2.0,
                    duration,
                    "Inspect the second half for unresolved evidence.",
                ),
                (
                    duration / 4.0,
                    3.0 * duration / 4.0,
                    "Inspect the middle half for unresolved evidence.",
                ),
            )
        )

    if memory.unresolved:
        signature = _evidence_content_signature(memory.unresolved[0])
        if signature:
            focused_request = "Inspect for " + " ".join(signature)
            focused_request = focused_request[:_MAX_EVIDENCE_REQUEST_CHARS]
            candidates = [
                (start, end, focused_request)
                for start, end, _evidence_request in candidates
            ]

    for start, end, evidence_request in candidates:
        if end - start < 0.001:
            continue
        interval = (start, end)
        if duplicate_interval(interval, memory.observed_intervals):
            continue
        span = end - start
        desired_frames = min(64, max(8, int(math.ceil(span * 2.0))))
        if explicit is None or interval != candidates[0][:2]:
            desired_frames = min(desired_frames, 32)
        nframes = next(
            count for count in _CONTROLLER_FRAME_COUNTS if count >= desired_frames
        )
        return FrameRequest(
            start_time=round(start, 3),
            end_time=round(end, 3),
            nframes=nframes,
            resize=0.75,
            evidence_request=evidence_request,
        )
    return None


def _controller_reference_output(
    sample: ModelSample,
    memory: EvidenceMemory,
    video_duration: float,
    *,
    allow_stop: bool,
) -> str:
    request = _controller_reference_request(sample, memory, video_duration)
    if request is not None:
        return _controller_tool_call(request)
    if allow_stop:
        return '{"action":"stop"}'
    return _controller_tool_call(
        FrameRequest(
            start_time=0.0,
            end_time=video_duration,
            nframes=64,
            resize=0.75,
            evidence_request=(
                "Re-check the decisive visual evidence symmetrically for both "
                "hypotheses."
            ),
        )
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
    reference = _controller_reference_output(sample, memory, duration, allow_stop=True)
    system = (
        "You are the text-only controller of a long-video evidence agent. Decide "
        "what visual evidence is still missing from the timestamped text ledger. "
        "You cannot see images. Never infer an action that is absent from the ledger. "
        "To observe, output only one official EVA <tool_call> JSON object whose tool "
        "is frame_select. Its arguments must contain numeric start_time, end_time, "
        "resize exactly 0.75, a precise evidence_request without quotation marks, "
        "and nframes equal to one of 8, 16, 32, 64, or 128. Do not output fps. The "
        "user message contains one currently valid, public-input-only "
        "reference action. Copy its wrapper and JSON key structure exactly; use the "
        "reference action itself when it addresses the missing evidence, otherwise "
        "change only its argument values. "
        "Use exactly one of nframes or fps and do not repeat an observed interval. "
        "Only when the ledger is ready for an independent completeness check, output "
        'exactly {"action":"stop"}.'
    )
    user = (
        f"{_question_text(sample)}\nVideo duration: {duration:.3f} seconds; "
        f"resolution: {width}x{height}.\nEvidence memory: {_memory_json(memory)}\n"
        f"Exact currently-valid output reference: {reference}"
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


def _with_accepted_planner_history(
    current_messages: Sequence[Mapping[str, Any]],
    accepted_history: Sequence[tuple[Mapping[str, Any], str]],
) -> list[dict[str, Any]]:
    """Add only successfully executed planner actions before the latest snapshot."""

    if (
        len(current_messages) != 2
        or current_messages[0].get("role") != "system"
        or current_messages[1].get("role") != "user"
    ):
        raise ValueError("planner history requires one system and one current user")
    messages = [copy.deepcopy(dict(current_messages[0]))]
    for raw_user, raw_action in accepted_history:
        action = parse_controller_action(raw_action)
        if action is None or action.action != "observe" or action.request is None:
            raise ValueError("planner history accepts only valid frame_select actions")
        if raw_user.get("role") != "user" or set(raw_user) != {"role", "content"}:
            raise ValueError("planner history requires exact prior user snapshots")
        messages.extend(
            [
                copy.deepcopy(dict(raw_user)),
                {"role": "assistant", "content": raw_action.strip()},
            ]
        )
    messages.append(copy.deepcopy(dict(current_messages[1])))
    if sum(message.get("role") == "system" for message in messages) != 1:
        raise AssertionError("planner accepted history must contain one system message")
    if messages_have_media(messages):
        raise AssertionError("planner accepted history must remain text-only")
    return messages


def build_role_separated_controller_messages(
    sample: ModelSample,
    memory: EvidenceMemory,
    video_metadata: Mapping[str, float | int],
    accepted_history: Sequence[tuple[Mapping[str, Any], str]],
    *,
    feedback: str = "",
) -> list[dict[str, Any]]:
    """Build the explicit-role planner request with accepted-action history only."""

    messages = build_controller_messages(
        sample, memory, video_metadata, feedback=feedback
    )
    if memory.unresolved:
        messages[0] = copy.deepcopy(messages[0])
        messages[0]["content"] += (
            " When unresolved evidence is listed, an observation's evidence_request "
            "must explicitly include every word of one unresolved item so the action "
            "can be checked against its stated evidence gap."
        )
    return _with_accepted_planner_history(
        messages,
        accepted_history,
    )


def build_perception_messages(
    sample: ModelSample,
    observation: FrameObservation,
    evidence_request: str,
    *,
    use_frame_indices: bool = False,
    role_separated: bool = False,
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
    output_fields = (
        "interval, timestamped_facts, option_evidence, temporal_changes, unresolved"
        if role_separated
        else (
            "interval, timestamped_facts, option_evidence, temporal_changes, "
            "unresolved, evidence_sufficient, next_evidence_needed"
        )
    )
    sufficiency_instruction = (
        "Completeness and stopping are decided by a separate visual verifier; do "
        "not output a sufficiency or next-step decision. "
        if role_separated
        else (
            "evidence_sufficient is boolean; next_evidence_needed is one string, "
            "empty only when no further evidence is needed. evidence_sufficient "
            "describes only the current accumulated visual question, not benchmark "
            "correctness. "
        )
    )
    system = (
        "You are the visual perception role. Report only facts directly visible in "
        "the current timestamped frames. Do not guess missing actions. Return one JSON "
        f"object with exactly: {output_fields}. "
        f"timestamped_facts is an array of {fact_schema}; {fact_reference}. "
        "option_evidence must contain "
        f"exactly these option labels: {letters}; each has supports and contradicts "
        "string arrays. interval is two numbers; each timestamped fact contains its "
        "required frame reference and a string fact; temporal_changes and unresolved "
        "are string arrays; "
        f"{sufficiency_instruction}Keep the state "
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
    if use_frame_indices:
        expected_items = 1 + 2 * len(observation.timestamps)
        if len(content) != expected_items:
            raise ValueError("indexed EVA observation has an unexpected content layout")
        for frame_index, timestamp in enumerate(observation.timestamps):
            text_position = 1 + frame_index * 2
            image_position = text_position + 1
            if (
                content[text_position].get("type") != "text"
                or content[image_position].get("type") != "image_url"
            ):
                raise ValueError("indexed EVA observation has an invalid frame layout")
            content[text_position] = {
                "type": "text",
                "text": (
                    f"Frame index {frame_index}, timestamp {timestamp:.3f} seconds:"
                ),
            }
    return [
        {"role": "system", "content": system},
        {
            "role": "user",
            "content": content,
        },
    ]


def perception_response_format(
    valid_letters: Sequence[str],
    frame_count: int,
    *,
    role_separated: bool = False,
) -> dict[str, Any]:
    """Return the strict, per-request schema enforced by the serving runtime."""

    letters = tuple(str(letter).strip().upper() for letter in valid_letters)
    if not letters or len(set(letters)) != len(letters):
        raise ValueError("perception response schema requires unique option letters")
    if isinstance(frame_count, bool) or not isinstance(frame_count, int) or frame_count <= 0:
        raise ValueError("perception response schema requires a positive frame count")

    short_text = {"type": "string", "minLength": 1, "maxLength": 160}
    option_state = {
        "type": "object",
        "properties": {
            "supports": {
                "type": "array",
                "items": short_text,
                "maxItems": 1,
            },
            "contradicts": {
                "type": "array",
                "items": short_text,
                "maxItems": 1,
            },
        },
        "required": ["contradicts", "supports"],
        "additionalProperties": False,
    }
    schema = {
        "type": "object",
        "properties": {
            "interval": {
                "type": "array",
                "items": {"type": "number"},
                "minItems": 2,
                "maxItems": 2,
            },
            "timestamped_facts": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "frame_index": {
                            "type": "integer",
                            "minimum": 0,
                            "maximum": frame_count - 1,
                        },
                        "fact": short_text,
                    },
                    "required": ["fact", "frame_index"],
                    "additionalProperties": False,
                },
                "maxItems": 12,
            },
            "option_evidence": {
                "type": "object",
                "properties": {letter: option_state for letter in letters},
                "required": sorted(letters),
                "additionalProperties": False,
            },
            "temporal_changes": {
                "type": "array",
                "items": short_text,
                "maxItems": 4,
            },
            "unresolved": {
                "type": "array",
                "items": short_text,
                "maxItems": 3,
            },
            "evidence_sufficient": {"type": "boolean"},
            "next_evidence_needed": {"type": "string", "maxLength": 160},
        },
        "required": [
            "evidence_sufficient",
            "interval",
            "next_evidence_needed",
            "option_evidence",
            "temporal_changes",
            "timestamped_facts",
            "unresolved",
        ],
        "additionalProperties": False,
    }
    if role_separated:
        schema["properties"].pop("evidence_sufficient")
        schema["properties"].pop("next_evidence_needed")
        schema["required"] = [
            key
            for key in schema["required"]
            if key not in {"evidence_sufficient", "next_evidence_needed"}
        ]
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "perception_memory_state",
            "strict": True,
            "schema": schema,
        },
    }


def build_perception_retry_messages(
    messages: Sequence[Mapping[str, Any]],
    reason: str,
    *,
    valid_letters: Sequence[str],
    frame_count: int,
    interval: tuple[float, float],
    role_separated: bool = False,
) -> list[dict[str, Any]]:
    """Add one compact schema correction without changing the observed frames."""

    retry = copy.deepcopy(list(messages))
    if len(retry) != 2 or retry[-1].get("role") != "user":
        raise ValueError("perception retry requires the frozen two-message prompt")
    content = retry[-1].get("content")
    if not isinstance(content, list):
        raise ValueError("perception retry requires multimodal user content")
    letters = tuple(str(letter).strip().upper() for letter in valid_letters)
    if not letters or len(set(letters)) != len(letters):
        raise ValueError("perception retry requires unique valid option letters")
    if frame_count <= 0:
        raise ValueError("perception retry requires a positive frame count")
    start = _finite_number(interval[0], "perception retry interval start")
    end = _finite_number(interval[1], "perception retry interval end")
    if end <= start:
        raise ValueError("perception retry interval must be increasing")
    skeleton = {
        "interval": [start, end],
        "timestamped_facts": [],
        "option_evidence": {
            letter: {"supports": [], "contradicts": []} for letter in letters
        },
        "temporal_changes": [],
        "unresolved": [],
        "evidence_sufficient": False,
        "next_evidence_needed": "",
    }
    if role_separated:
        skeleton.pop("evidence_sufficient")
        skeleton.pop("next_evidence_needed")
    correction = (
        f"Correction after {reason or 'invalid_json_or_schema'}: return one compact "
        "raw JSON object only. The current observation contains exactly "
        f"{frame_count} frames, so every timestamped_facts item must be exactly "
        f'{{"frame_index": <integer 0 through {frame_count - 1}>, '
        '"fact": "<directly visible fact>"}}. do not use timestamp seconds as '
        f"frame_index. Use no more than 6 facts. Copy interval exactly as "
        f"[{start}, {end}]. option_evidence must contain exactly "
        f"{', '.join(letters)}; every option value must be an object with exactly "
        "supports and contradicts string arrays. A bare list as an option value is "
        "forbidden. Preserve every top-level key and its type. Use this exact JSON "
        "skeleton, replacing only its arrays and observation values: "
        + json.dumps(skeleton, ensure_ascii=False, separators=(",", ":"))
    )
    content.append({"type": "text", "text": correction})
    return retry


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
    *,
    feedback: str = "",
) -> list[dict[str, Any]]:
    """Request fresh visual evidence without identifying the Direct branch."""

    first, second = sorted(str(item).strip().upper() for item in hypotheses)
    duration = float(video_metadata["duration"])
    reference = _controller_reference_output(sample, memory, duration, allow_stop=False)
    user = (
        f"{_question_text(sample)}\nHypotheses (unordered): {first}, {second}. "
        f"Video duration: {duration:.3f} seconds.\nEvidence memory: "
        f"{_memory_json(memory)}\nExact official output reference: {reference}"
    )
    if feedback:
        user += f"\nController feedback: {_SPACE_RE.sub(' ', feedback.strip())}"
    return [
        {
            "role": "system",
            "content": (
                "Two equally unprivileged hypotheses disagree. Select one decisive visual "
                "interval that can distinguish them. Re-observing a prior area is allowed "
                "only with denser frames or wider before/after context. Return only one "
                "official EVA frame_select tool call with start_time, end_time, exactly one "
                "nframes value from 8, 16, 32, 64, or 128, resize exactly 0.75, and "
                "a symmetric evidence_request that tests both hypotheses. Copy the "
                "reference wrapper and JSON key structure "
                "exactly; never return bare JSON, args, markdown, or prose."
            ),
        },
        {
            "role": "user",
            "content": user,
        },
    ]


def build_role_separated_confirmation_controller_messages(
    sample: ModelSample,
    memory: EvidenceMemory,
    video_metadata: Mapping[str, float | int],
    hypotheses: tuple[str, str],
    accepted_history: Sequence[tuple[Mapping[str, Any], str]],
    *,
    feedback: str = "",
) -> list[dict[str, Any]]:
    """Build the explicit-role confirmation request with accepted history only."""

    return _with_accepted_planner_history(
        build_confirmation_controller_messages(
            sample,
            memory,
            video_metadata,
            hypotheses,
            feedback=feedback,
        ),
        accepted_history,
    )


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


def _deduplicated_frame_inventory(
    inventory: Sequence[tuple[str, float]],
) -> tuple[tuple[str, float], ...]:
    result: list[tuple[str, float]] = []
    seen: set[tuple[str, float]] = set()
    for raw_path, raw_timestamp in inventory:
        resolved = Path(raw_path).resolve()
        if not resolved.is_file():
            raise ValueError(f"observed frame is missing: {resolved}")
        path = str(resolved)
        timestamp = _finite_number(raw_timestamp, "frame inventory timestamp")
        item = (path, timestamp)
        if item not in seen:
            seen.add(item)
            result.append(item)
    return tuple(result)


def select_runtime_verifier_frames(
    frame_inventory: Sequence[tuple[str, float]],
) -> tuple[tuple[tuple[str, float], ...], tuple[int, ...]]:
    """Select at most 16 frames with deterministic full-inventory provenance."""

    frames = _deduplicated_frame_inventory(frame_inventory)
    frame_count = len(frames)
    if frame_count <= RUNTIME_VERIFIER_MAX_FRAMES:
        full_indices = tuple(range(frame_count))
    else:
        denominator = RUNTIME_VERIFIER_MAX_FRAMES - 1
        full_indices = tuple(
            (
                local_index * (frame_count - 1)
                + denominator // 2
            )
            // denominator
            for local_index in range(RUNTIME_VERIFIER_MAX_FRAMES)
        )
    return tuple(frames[index] for index in full_indices), full_indices


def map_runtime_verifier_frame_indices(
    local_indices: Sequence[int],
    selected_full_indices: Sequence[int],
) -> tuple[int, ...]:
    """Map verifier-local citations back to the complete frame inventory."""

    if any(
        isinstance(index, bool) or not isinstance(index, int) or index < 0
        for index in selected_full_indices
    ):
        raise ValueError("selected verifier frame indices must be non-negative integers")
    full_indices = tuple(selected_full_indices)
    result: list[int] = []
    seen: set[int] = set()
    for raw_index in local_indices:
        if isinstance(raw_index, bool) or not isinstance(raw_index, int):
            raise ValueError("runtime verifier citations must be integer local indices")
        local_index = raw_index
        if local_index < 0 or local_index >= len(full_indices):
            raise ValueError("runtime verifier cited an invalid local frame index")
        full_index = full_indices[local_index]
        if full_index not in seen:
            seen.add(full_index)
            result.append(full_index)
    return tuple(result)


def build_runtime_visual_csv_messages(
    sample: ModelSample,
    frame_inventory: Sequence[tuple[str, float]],
    *,
    prefix_index: int,
) -> list[dict[str, Any]]:
    """Reuse the offline visual-CSV contract for one candidate-blind runtime check."""

    frames, _full_indices = select_runtime_verifier_frames(frame_inventory)
    if not frames:
        raise ValueError("runtime visual CSV requires at least one observed frame")
    public_sample = replace(sample, candidate_answer=None)
    content_sha256s = frame_content_sha256s(tuple(item[0] for item in frames))
    source_sha256 = hashlib.sha256(
        json.dumps(
            {
                "dataset": sample.dataset,
                "sample_id": sample.sample_id,
                "prefix_index": prefix_index,
                "frames": frames,
                "frame_content_sha256s": content_sha256s,
            },
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    job = VisualCsvJob(
        trajectory_id=f"{sample.dataset}:{sample.sample_id}:runtime",
        prefix_id=f"{sample.dataset}:{sample.sample_id}:runtime#{prefix_index:03d}",
        prefix_index=prefix_index,
        dataset=sample.dataset,
        sample_id=sample.sample_id,
        sample=public_sample,
        frame_paths=tuple(item[0] for item in frames),
        timestamps=tuple(item[1] for item in frames),
        frame_content_sha256s=content_sha256s,
        source_sha256=source_sha256,
    )
    return build_visual_csv_messages(job)


def build_cited_judge_messages(
    sample: ModelSample,
    memory: EvidenceMemory,
    frame_inventory: Sequence[tuple[str, float]],
    cited_frame_indices: Sequence[int],
) -> list[dict[str, Any]]:
    """Attach at most 16 verifier-cited frames to the evidence-ledger answerer."""

    frames = _deduplicated_frame_inventory(frame_inventory)
    unique_indices: list[int] = []
    for raw_index in cited_frame_indices:
        if isinstance(raw_index, bool) or not isinstance(raw_index, int):
            raise ValueError("cited frame index must be an integer")
        if raw_index < 0 or raw_index >= len(frames):
            raise ValueError("cited frame index is outside the accumulated inventory")
        if raw_index not in unique_indices:
            unique_indices.append(raw_index)
        if len(unique_indices) == _MAX_ANSWERER_CITED_FRAMES:
            break
    if not unique_indices:
        raise ValueError("answerer requires at least one verifier-cited frame")
    content: list[dict[str, Any]] = [
        {
            "type": "text",
            "text": (
                f"{_question_text(sample)}\nEvidence memory: {_memory_json(memory)}\n"
                "The following decisive frames were cited by the candidate-blind "
                "visual verifier. Answer from the ledger and these frames only."
            ),
        }
    ]
    for index in unique_indices:
        path, timestamp = frames[index]
        content.extend(
            [
                {
                    "type": "text",
                    "text": f"Cited frame index {index}, timestamp {timestamp:.3f} seconds",
                },
                {
                    "type": "image_url",
                    "image_url": {"url": Path(path).resolve().as_uri()},
                },
            ]
        )
    messages = [
        {
            "role": "system",
            "content": (
                "Answer only from the timestamped evidence ledger and attached cited "
                "frames. Every choice must cite valid evidence IDs; never fill in an "
                'unobserved action. Return exactly {"answer":"X","evidence_ids":["E0001"]}.'
            ),
        },
        {"role": "user", "content": content},
    ]
    assert_annotation_free_request({"messages": messages})
    return messages


@dataclass(frozen=True)
class CompletenessDecision:
    evidence_complete: bool
    missing_evidence: tuple[str, ...]
    answer: str | None = None
    frame_indices: tuple[int, ...] = ()


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


class PerceptionModelFailure(ValueError):
    """A Perception response remained invalid after its one bounded retry."""

    failure_class = "model_parse_failure"

    def __init__(self, stage: str, reason: str, attempts: int) -> None:
        super().__init__(
            f"{stage} returned invalid evidence after {attempts} attempt(s): {reason}"
        )
        self.stage = stage
        self.reason = reason
        self.attempts = attempts


def apply_candidate_gate(
    *,
    candidate: str | None,
    first: EvidenceDecision | None,
    confirmation: EvidenceDecision | None,
    complete: bool,
    memory: EvidenceMemory,
    first_valid_evidence_ids: Iterable[str] | None = None,
    confirmation_valid_evidence_ids: Iterable[str] | None = None,
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
    first_allowed = (
        frozenset(first_valid_evidence_ids)
        if first_valid_evidence_ids is not None
        else frozenset()
    )
    confirmation_allowed = (
        frozenset(confirmation_valid_evidence_ids)
        if confirmation_valid_evidence_ids is not None
        else frozenset()
    )
    if (
        not first_allowed
        or not confirmation_allowed
        or not set(first.evidence_ids).issubset(first_allowed)
        or not set(confirmation.evidence_ids).issubset(confirmation_allowed)
    ):
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
        max_controller_attempts: int | None = None,
        max_frames_per_call: int = 128,
        controller_max_tokens: int = 512,
        perception_max_tokens: int = 1024,
        judge_max_tokens: int = 512,
        server_max_model_len: int = 131072,
        context_safety_tokens: int = 1024,
        seed: int = 42,
        candidate_results_sha256: str | None = None,
        model_artifact_sha256: str | None = None,
        role_bindings: Mapping[str, PerceptionMemoryRoleBinding] | None = None,
        role_config_sha256: str | None = None,
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
        if max_controller_attempts is None:
            max_controller_attempts = max_turns * 2
        if (
            isinstance(max_controller_attempts, bool)
            or not isinstance(max_controller_attempts, int)
            or max_controller_attempts <= 0
        ):
            raise ValueError("max_controller_attempts must be a positive integer")
        for name, value in (
            ("controller_max_tokens", controller_max_tokens),
            ("perception_max_tokens", perception_max_tokens),
            ("judge_max_tokens", judge_max_tokens),
            ("server_max_model_len", server_max_model_len),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if (
            isinstance(context_safety_tokens, bool)
            or not isinstance(context_safety_tokens, int)
            or context_safety_tokens < 0
        ):
            raise ValueError("context_safety_tokens must be a non-negative integer")
        if context_safety_tokens >= server_max_model_len:
            raise ValueError("context_safety_tokens must be below server_max_model_len")
        if role_bindings is None:
            if role_config_sha256 is not None:
                raise ValueError("role_config_sha256 requires explicit role_bindings")
            default_binding = PerceptionMemoryRoleBinding(
                client=client,
                model=model,
                artifact_sha256=model_artifact_sha256,
            )
            resolved_role_bindings = {
                role: default_binding for role in PERCEPTION_MEMORY_ROLE_NAMES
            }
        else:
            if set(role_bindings) != set(PERCEPTION_MEMORY_ROLE_NAMES):
                raise ValueError(
                    "role_bindings must contain exactly planner, observer, verifier, "
                    "and answerer"
                )
            if role_config_sha256 is None or len(role_config_sha256) != 64 or any(
                character not in "0123456789abcdefABCDEF"
                for character in role_config_sha256
            ):
                raise ValueError(
                    "explicit role_bindings require a 64-character role_config_sha256"
                )
            resolved_role_bindings = {}
            for role in PERCEPTION_MEMORY_ROLE_NAMES:
                binding = role_bindings[role]
                if not isinstance(binding, PerceptionMemoryRoleBinding):
                    raise ValueError(
                        f"role_bindings.{role} must be PerceptionMemoryRoleBinding"
                    )
                if binding.artifact_sha256 is None:
                    raise ValueError(
                        f"role_bindings.{role} requires artifact_sha256"
                    )
                resolved_role_bindings[role] = binding
        self.client = client
        self.role_bindings = resolved_role_bindings
        self.role_config_sha256 = (
            role_config_sha256.lower() if role_config_sha256 is not None else None
        )
        observer_client = self.role_bindings["observer"].client
        self.local_media_transport = (
            "path"
            if bool(getattr(observer_client, "local_file_urls_as_paths", False))
            else "file_url"
        )
        self.role_media_transports = {
            role: (
                "path"
                if bool(
                    getattr(
                        self.role_bindings[role].client,
                        "local_file_urls_as_paths",
                        False,
                    )
                )
                else "file_url"
            )
            for role in PERCEPTION_MEMORY_ROLE_NAMES
        }
        self.model = model
        self.index = VideoIndex(video_root)
        self.frame_tool = frame_tool or FrameTool(
            frame_root, max_frames_per_call=max_frames_per_call
        )
        self.max_turns = max_turns
        self.max_controller_attempts = max_controller_attempts
        self.controller_max_tokens = controller_max_tokens
        self.perception_max_tokens = perception_max_tokens
        self.judge_max_tokens = judge_max_tokens
        self.server_max_model_len = server_max_model_len
        self.context_safety_tokens = context_safety_tokens
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
            package_root / "question_time.py",
            package_root / "perception_memory_visual_csv.py",
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
            "perception_response_schema_version": PERCEPTION_RESPONSE_SCHEMA_VERSION,
            "controller_output_constraint_version": (
                CONTROLLER_OUTPUT_CONSTRAINT_VERSION
            ),
            "evidence_request_content_signature_version": (
                EVIDENCE_REQUEST_CONTENT_SIGNATURE_VERSION
            ),
            "runtime_verifier_max_frames": RUNTIME_VERIFIER_MAX_FRAMES,
            "runtime_verifier_frame_selection_policy": (
                RUNTIME_VERIFIER_FRAME_SELECTION_POLICY
            ),
            "model": self.model,
            "role_config_sha256": self.role_config_sha256,
            "role_models": {
                role: self.role_bindings[role].model
                for role in PERCEPTION_MEMORY_ROLE_NAMES
            },
            "role_artifact_sha256s": {
                role: self.role_bindings[role].artifact_sha256
                for role in PERCEPTION_MEMORY_ROLE_NAMES
            },
            "implementation_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            "implementation_bundle_sha256": implementation_bundle_sha256,
            "frame_tool_identity": copy.deepcopy(selector),
            "max_turns": self.max_turns,
            "max_controller_attempts": self.max_controller_attempts,
            "confirmation_controller_max_attempts": 2,
            "max_frames_per_call": getattr(
                self.frame_tool, "max_frames_per_call", None
            ),
            "controller_max_tokens": self.controller_max_tokens,
            "perception_max_tokens": self.perception_max_tokens,
            "perception_retry_max_tokens": min(
                self.perception_max_tokens * 2,
                self.server_max_model_len - self.context_safety_tokens,
            ),
            "judge_max_tokens": self.judge_max_tokens,
            "server_max_model_len": self.server_max_model_len,
            "context_safety_tokens": self.context_safety_tokens,
            "seed": self.seed,
            "candidate_results_sha256": self.candidate_results_sha256,
            "model_artifact_sha256": self.model_artifact_sha256,
            "manifest_sha256": self.manifest_sha256,
            "experiment_config_sha256": self.experiment_config_sha256,
            "diagnostics_gate_sha256": self.diagnostics_gate_sha256,
            "local_media_transport": self.local_media_transport,
            "role_media_transports": copy.deepcopy(self.role_media_transports),
            "role_separated_runtime_version": (
                ROLE_SEPARATED_RUNTIME_VERSION
                if self.role_config_sha256 is not None
                else None
            ),
            "role_prompt_schema_bundle_sha256": (
                role_prompt_schema_bundle_sha256()
                if self.role_config_sha256 is not None
                else None
            ),
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
        response_format: dict[str, Any] | None = None,
        structured_outputs: dict[str, Any] | None = None,
        tools_disabled: bool = False,
        step_index: int | None = None,
        prefix_index: int | None = None,
    ) -> str:
        assert_annotation_free_request({"messages": messages})
        role_name = perception_memory_role_for_stage(stage)
        role_binding = self.role_bindings[role_name]
        if messages_have_media(messages) and (
            role_name == "planner"
            or (self.role_config_sha256 is None and role_name != "observer")
        ):
            raise AssertionError(f"{role_name} request contains media")
        prompt_hash = hashlib.sha256(
            json.dumps(
                messages,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                default=str,
            ).encode("utf-8")
        ).hexdigest()
        effective_response_format = (
            copy.deepcopy(response_format)
            if response_format is not None
            else ({"type": "json_object"} if json_mode else None)
        )
        effective_structured_outputs = (
            copy.deepcopy(structured_outputs)
            if structured_outputs is not None
            else None
        )
        extra_body: dict[str, Any] = {"return_token_ids": True}
        if effective_structured_outputs is not None:
            extra_body["structured_outputs"] = effective_structured_outputs
        if tools_disabled:
            extra_body["tool_choice"] = "none"
        result = role_binding.client.chat(
            role_binding.model,
            messages,
            max_tokens=max_tokens,
            temperature=0.0,
            seed=self.seed + seed_offset,
            response_format=effective_response_format,
            chat_template_kwargs={"enable_thinking": False},
            extra_body=extra_body,
        )
        trace.append(
            {
                "stage": stage,
                "role_name": role_name,
                "model": role_binding.model,
                "model_artifact_sha256": role_binding.artifact_sha256,
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
                "response_format": copy.deepcopy(effective_response_format),
                "structured_outputs": copy.deepcopy(effective_structured_outputs),
                "tools_disabled": tools_disabled,
            }
        )
        if result.finish_reason == "length":
            raise RuntimeError(f"{stage} response was truncated")
        return result.content

    def _perception_state(
        self,
        sample: ModelSample,
        observation: FrameObservation,
        evidence_request: str,
        trace: list[dict[str, Any]],
        *,
        stage: str,
        seed_offset: int,
        step_index: int,
        prefix_index: int,
    ) -> tuple[PerceptionState, dict[str, Any]]:
        """Run one Perception turn with at most one same-frame model retry."""

        messages = build_perception_messages(
            sample,
            observation,
            evidence_request,
            use_frame_indices=True,
            role_separated=self.role_config_sha256 is not None,
        )
        retry_group_id = _messages_sha256(messages)
        retry_reason: str | None = None
        attempts = 0
        timestamp_reference_mode = "invalid"
        for attempt_index in range(2):
            attempt_messages = (
                messages
                if attempt_index == 0
                else build_perception_retry_messages(
                    messages,
                    retry_reason or "",
                    valid_letters=sample.option_letters,
                    frame_count=len(observation.timestamps),
                    interval=(
                        observation.resolved_start_time,
                        observation.resolved_end_time,
                    ),
                    role_separated=self.role_config_sha256 is not None,
                )
            )
            if attempt_index == 0:
                max_tokens = self.perception_max_tokens
            else:
                first_usage = trace[-1].get("usage", {}) if trace else {}
                prompt_tokens = first_usage.get("prompt_tokens")
                if (
                    isinstance(prompt_tokens, bool)
                    or not isinstance(prompt_tokens, (int, float))
                    or not math.isfinite(float(prompt_tokens))
                    or float(prompt_tokens) < 0
                ):
                    prompt_tokens = 0
                headroom = (
                    self.server_max_model_len
                    - int(prompt_tokens)
                    - self.context_safety_tokens
                )
                max_tokens = min(self.perception_max_tokens * 2, headroom)
                if max_tokens <= 0:
                    trace[-1]["retry_blocked_reason"] = (
                        "insufficient_service_context_headroom"
                    )
                    break

            trace_length = len(trace)
            try:
                content = self._chat(
                    trace,
                    attempt_messages,
                    stage=stage,
                    max_tokens=max_tokens,
                    seed_offset=seed_offset,
                    json_mode=True,
                    response_format=perception_response_format(
                        sample.option_letters,
                        len(observation.timestamps),
                        role_separated=self.role_config_sha256 is not None,
                    ),
                    step_index=step_index,
                    prefix_index=prefix_index,
                )
            except AnnotationLeakError:
                raise
            except Exception as exc:
                if len(trace) == trace_length:
                    role_name = perception_memory_role_for_stage(stage)
                    role_binding = self.role_bindings[role_name]
                    trace.append(
                        {
                            "stage": stage,
                            "role_name": role_name,
                            "model": role_binding.model,
                            "model_artifact_sha256": (
                                role_binding.artifact_sha256
                            ),
                            "messages": copy.deepcopy(attempt_messages),
                            "content": "",
                            "reasoning_content": "",
                            "finish_reason": None,
                            "usage": {},
                            "latency_s": 0.0,
                            "seed": self.seed + seed_offset,
                            "step_index": step_index,
                            "prefix_index": prefix_index,
                            "prompt_hash": _messages_sha256(attempt_messages),
                            "retry_group_id": retry_group_id,
                            "attempt_index": attempt_index,
                            "retry_of_attempt": 0 if attempt_index else None,
                            "retry_reason": retry_reason,
                            "retry_triggered": False,
                            "timestamp_reference_mode": "not_run",
                            "max_tokens": max_tokens,
                            "failure_class": "infrastructure_error",
                            "attempt_error": f"{type(exc).__name__}: {exc}",
                        }
                    )
                    raise
                attempt_trace = trace[-1]
                if (
                    not isinstance(exc, RuntimeError)
                    or attempt_trace.get("finish_reason") != "length"
                ):
                    raise
                content = str(attempt_trace.get("content") or "")
            attempts += 1
            attempt_trace = trace[-1]
            state, timestamp_reference_mode = bind_perception_state(
                content,
                sample.option_letters,
                observation.timestamps,
                # Some Qwen responses faithfully copy the printed frame timestamp
                # instead of emitting frame_index.  Accept that legacy surface form
                # only when the normalizer below can bind it to an actual sampled
                # frame within the frozen 1 ms tolerance.  The persisted SFT target
                # remains canonical frame_index via perception_model_target().
                allow_timestamp_schema=True,
                allow_role_separated_schema=self.role_config_sha256 is not None,
            )
            attempt_failure: str | None = None
            validation_error: str | None = None
            if attempt_trace.get("finish_reason") == "length":
                attempt_failure = "finish_reason_length"
            elif state is None:
                attempt_failure = (
                    "legacy_timestamp_schema"
                    if timestamp_reference_mode == "timestamp"
                    else "invalid_json_or_schema"
                )
            else:
                try:
                    state = validate_perception_state_observation(state, observation)
                except ValueError as exc:
                    validation_error = str(exc)
                    attempt_failure = "invalid_observation_binding"
                    state = None

            attempt_trace.update(
                {
                    "retry_group_id": retry_group_id,
                    "attempt_index": attempt_index,
                    "retry_of_attempt": 0 if attempt_index else None,
                    "retry_reason": attempt_failure or retry_reason,
                    "retry_triggered": attempt_index == 0
                    and attempt_failure is not None,
                    "timestamp_reference_mode": timestamp_reference_mode,
                    "max_tokens": max_tokens,
                    "failure_class": (
                        "model_parse_failure" if attempt_failure else None
                    ),
                    "attempt_error": attempt_failure,
                }
            )
            if validation_error is not None:
                attempt_trace["state_validation_error"] = validation_error
            if state is not None and attempt_failure is None:
                return state, {
                    "perception_attempts": attempts,
                    "perception_retry_reason": retry_reason,
                    "timestamp_reference_mode": timestamp_reference_mode,
                    "perception_model_target": perception_model_target(
                        state,
                        observation.timestamps,
                        role_separated=self.role_config_sha256 is not None,
                    ),
                }
            retry_reason = attempt_failure

        raise PerceptionModelFailure(
            stage,
            retry_reason or "insufficient_service_context_headroom",
            attempts,
        )

    def _completeness(
        self,
        sample: ModelSample,
        memory: EvidenceMemory,
        trace: list[dict[str, Any]],
        seed_offset: int,
        step_index: int | None = None,
        prefix_index: int | None = None,
        frame_inventory: Sequence[tuple[str, float]] = (),
    ) -> CompletenessDecision | None:
        if self.role_config_sha256 is not None:
            frames = _deduplicated_frame_inventory(frame_inventory)
            if not frames:
                return CompletenessDecision(False, ("no visual frames observed",))
            verifier_frames, selected_full_indices = select_runtime_verifier_frames(
                frames
            )
            effective_prefix = prefix_index if prefix_index is not None else -1
            messages = build_runtime_visual_csv_messages(
                sample, frames, prefix_index=effective_prefix
            )
            content = self._chat(
                trace,
                messages,
                stage="completeness",
                max_tokens=self.judge_max_tokens,
                seed_offset=seed_offset,
                json_mode=False,
                response_format=visual_csv_response_format(sample.option_letters),
                tools_disabled=True,
                step_index=step_index,
                prefix_index=prefix_index,
            )
            decision = parse_visual_csv_response(
                content, sample.option_letters, len(verifier_frames)
            )
            local_cited_indices = (
                decision.frame_indices if decision is not None else ()
            )
            cited_full_indices = map_runtime_verifier_frame_indices(
                local_cited_indices, selected_full_indices
            )
            trace[-1].update(
                {
                    "candidate_blind": True,
                    "ledger_blind": True,
                    "prior_reasoning_blind": True,
                    "frame_inventory_count": len(frames),
                    "verifier_frame_selection_policy": (
                        RUNTIME_VERIFIER_FRAME_SELECTION_POLICY
                    ),
                    "verifier_frame_selection_limit": RUNTIME_VERIFIER_MAX_FRAMES,
                    "verifier_selected_frame_count": len(verifier_frames),
                    "verifier_selected_full_frame_indices": list(
                        selected_full_indices
                    ),
                    "verifier_local_cited_frame_indices": list(
                        local_cited_indices
                    ),
                    "cited_frame_indices": list(cited_full_indices),
                }
            )
            if decision is None:
                return None
            return CompletenessDecision(
                evidence_complete=decision.evidence_complete,
                missing_evidence=decision.missing_evidence,
                answer=decision.answer,
                frame_indices=cited_full_indices,
            )
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
        frame_inventory: Sequence[tuple[str, float]] = (),
        cited_frame_indices: Sequence[int] = (),
    ) -> EvidenceDecision | None:
        messages = build_judge_messages(sample, memory)
        if self.role_config_sha256 is not None:
            messages = build_cited_judge_messages(
                sample, memory, frame_inventory, cited_frame_indices
            )
        content = self._chat(
            trace,
            messages,
            stage=stage,
            max_tokens=self.judge_max_tokens,
            seed_offset=seed_offset,
            json_mode=True,
            tools_disabled=self.role_config_sha256 is not None,
            step_index=step_index,
            prefix_index=prefix_index,
        )
        if self.role_config_sha256 is not None:
            trace[-1].update(
                {
                    "frame_inventory_count": len(
                        _deduplicated_frame_inventory(frame_inventory)
                    ),
                    "cited_frame_indices": list(cited_frame_indices)[
                        :_MAX_ANSWERER_CITED_FRAMES
                    ],
                    "evidence_memory_at_stage": memory.to_dict(),
                    "evidence_memory_sha256": hashlib.sha256(
                        json.dumps(
                            memory.to_dict(),
                            ensure_ascii=False,
                            sort_keys=True,
                            separators=(",", ":"),
                        ).encode("utf-8")
                    ).hexdigest(),
                }
            )
        return parse_evidence_decision(
            content, sample.option_letters, memory.evidence_ids
        )

    def run(
        self,
        sample: ModelSample,
        *,
        rescue_source_requested_intervals: Sequence[Sequence[float]] | None = None,
    ) -> dict[str, Any]:
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
        failure_class: str | None = None
        annotation_check = "passed"
        tool_latency = 0.0
        explicit_time_rescue_audit: dict[str, Any] | None = None
        evidence_steps = 0
        controller_attempts = 0
        controller_attempt_limit_reached = False
        controller_attempt_limit_reason: str | None = None
        no_novel_action_rejections = 0
        no_novel_action_reason: str | None = None
        stopped_without_novel_action = False
        accepted_planner_history: list[tuple[dict[str, Any], str]] = []
        frame_inventory: list[tuple[str, float]] = []
        decisive_frame_indices: tuple[int, ...] = ()
        confirmation_decisive_frame_indices: tuple[int, ...] = ()
        first_valid_evidence_ids: frozenset[str] = frozenset()
        confirmation_bound_evidence_ids: tuple[str, ...] = ()
        confirmation_new_evidence_ids: tuple[str, ...] = ()

        try:
            video = self.index.resolve(sample.video)
            session = self.frame_tool.open_session(
                video, f"pm-{sample.dataset}-{sample.sample_id}"
            )
            feedback = ""
            rescue_request = None
            if (
                self.scoring_deferred
                and self.trajectory_variant_id == "rescue_explicit_time"
            ):
                rescue_request, explicit_time_rescue_audit = (
                    explicit_time_rescue_request(
                        sample.question,
                        float(session.metadata["duration"]),
                        rescue_source_requested_intervals,
                    )
                )
            elif (
                self.scoring_deferred
                and self.trajectory_variant_id in DURATION_RESCUE_TRAJECTORY_VARIANTS
            ):
                rescue_request = rescue_frame_request(
                    self.trajectory_variant_id, float(session.metadata["duration"])
                )
            last_controller_rejection: str | None = None
            controller_step_attempts = 0

            def planner_messages(*, current_feedback: str = "") -> list[dict[str, Any]]:
                if self.role_config_sha256 is None:
                    return build_controller_messages(
                        sample,
                        memory,
                        session.metadata,
                        feedback=current_feedback,
                    )
                return build_role_separated_controller_messages(
                    sample,
                    memory,
                    session.metadata,
                    accepted_planner_history,
                    feedback=current_feedback,
                )

            while (
                evidence_steps < self.max_turns
                and controller_attempts < self.max_controller_attempts
            ):
                controller_attempt_index = controller_attempts
                controller_attempts += 1
                attempt_index = controller_step_attempts
                controller_step_attempts += 1
                retry_reason = last_controller_rejection
                controller_retry_group_id = _messages_sha256(
                    planner_messages()
                )
                if controller_attempt_index == 0 and rescue_request is not None:
                    controller_messages = planner_messages(current_feedback=feedback)
                    assert_annotation_free_request({"messages": controller_messages})
                    controller_text = _controller_tool_call(rescue_request)
                    request_trace.append(
                        {
                            "stage": "controller",
                            "role_name": "planner",
                            "model": self.role_bindings["planner"].model,
                            "model_artifact_sha256": self.role_bindings[
                                "planner"
                            ].artifact_sha256,
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
                            "step_index": evidence_steps,
                            "prefix_index": -1,
                            "controller_attempt_index": controller_attempt_index,
                            "retry_group_id": controller_retry_group_id,
                            "attempt_index": attempt_index,
                            "retry_of_attempt": None,
                            "retry_reason": None,
                            "prompt_hash": _messages_sha256(controller_messages),
                            "source_cached_action": (
                                self.trajectory_variant_id
                                in DURATION_RESCUE_TRAJECTORY_VARIANTS
                            ),
                            "deterministic_rescue_action": True,
                            "trajectory_variant_id": self.trajectory_variant_id,
                            "explicit_time_rescue_audit": copy.deepcopy(
                                explicit_time_rescue_audit
                            ),
                        }
                    )
                else:
                    controller_messages = planner_messages(current_feedback=feedback)
                    try:
                        controller_text = self._chat(
                            request_trace,
                            controller_messages,
                            stage="controller",
                            max_tokens=self.controller_max_tokens,
                            seed_offset=evidence_steps * 10,
                            json_mode=False,
                            structured_outputs=controller_structured_outputs(
                                allow_stop=True
                            ),
                            step_index=evidence_steps,
                            prefix_index=len(perception_states) - 1,
                        )
                    except RuntimeError:
                        current = request_trace[-1]
                        if (
                            current.get("stage") != "controller"
                            or current.get("finish_reason") != "length"
                        ):
                            raise
                        current.update(
                            {
                                "controller_attempt_index": controller_attempt_index,
                                "retry_group_id": controller_retry_group_id,
                                "attempt_index": attempt_index,
                                "retry_of_attempt": (
                                    attempt_index - 1 if retry_reason else None
                                ),
                                "retry_reason": retry_reason,
                                "action_accepted": False,
                                "action_rejection_reason": (
                                    "controller_response_truncated"
                                ),
                            }
                        )
                        feedback = _controller_retry_feedback(
                            "controller_response_truncated",
                            float(session.metadata["duration"]),
                        )
                        last_controller_rejection = "controller_response_truncated"
                        continue
                    request_trace[-1].update(
                        {
                            "controller_attempt_index": controller_attempt_index,
                            "retry_group_id": controller_retry_group_id,
                            "attempt_index": attempt_index,
                            "retry_of_attempt": (
                                attempt_index - 1 if retry_reason else None
                            ),
                            "retry_reason": retry_reason,
                        }
                    )
                action = parse_controller_action(controller_text)
                if action is None:
                    rejection_reason = _controller_rejection_reason(
                        controller_text, float(session.metadata["duration"])
                    )
                    request_trace[-1]["action_accepted"] = False
                    request_trace[-1]["action_rejection_reason"] = rejection_reason
                    feedback = _controller_retry_feedback(
                        rejection_reason, float(session.metadata["duration"])
                    )
                    last_controller_rejection = rejection_reason
                    continue
                if action.action == "stop":
                    controller_trace = request_trace[-1]
                    decision = self._completeness(
                        sample,
                        memory,
                        request_trace,
                        controller_attempt_index * 10 + 1,
                        step_index=evidence_steps,
                        prefix_index=len(perception_states) - 1,
                        frame_inventory=frame_inventory,
                    )
                    if (
                        decision is not None
                        and decision.evidence_complete
                        and memory.event_ledger
                    ):
                        controller_trace["action_accepted"] = True
                        decisive_frame_indices = decision.frame_indices
                        complete = True
                        stop_reason = "evidence_complete"
                        if perception_states:
                            perception_states[-1]["evidence_complete"] = True
                        break
                    controller_trace["action_accepted"] = False
                    controller_trace["action_rejection_reason"] = "incomplete_evidence"
                    missing = (
                        decision.missing_evidence
                        if decision
                        else ("invalid completeness response",)
                    )
                    feedback = "Stop rejected; missing evidence: " + "; ".join(missing)
                    last_controller_rejection = "incomplete_evidence"
                    no_novel_action_rejections += 1
                    if no_novel_action_rejections >= _MAX_NO_NOVEL_ACTION_REJECTIONS:
                        stopped_without_novel_action = True
                        no_novel_action_reason = "incomplete_evidence"
                        stop_reason = "evidence_incomplete_no_novel_action"
                        break
                    continue

                assert action.request is not None
                interval_rejection = _request_interval_rejection_reason(
                    action.request, float(session.metadata["duration"])
                )
                if interval_rejection is not None:
                    request_trace[-1]["action_accepted"] = False
                    request_trace[-1]["action_rejection_reason"] = interval_rejection
                    feedback = _controller_retry_feedback(
                        interval_rejection, float(session.metadata["duration"])
                    )
                    last_controller_rejection = interval_rejection
                    continue
                if (
                    self.role_config_sha256 is not None
                    and memory.unresolved
                    and not evidence_request_addresses_unresolved(
                        action.request.evidence_request, memory.unresolved
                    )
                ):
                    request_trace[-1]["action_accepted"] = False
                    request_trace[-1]["action_rejection_reason"] = (
                        "unrelated_evidence_request"
                    )
                    feedback = (
                        "The evidence_request did not explicitly name any current "
                        "unresolved item. Copy one unresolved item verbatim into the "
                        "next evidence_request, then select the interval needed to "
                        "resolve it."
                    )
                    last_controller_rejection = "unrelated_evidence_request"
                    no_novel_action_rejections += 1
                    if (
                        evidence_steps > 0
                        and no_novel_action_rejections
                        >= _MAX_NO_NOVEL_ACTION_REJECTIONS
                    ):
                        stopped_without_novel_action = True
                        no_novel_action_reason = "unrelated_evidence_request"
                        stop_reason = "evidence_incomplete_no_novel_action"
                        break
                    continue
                requested_interval = (
                    action.request.start_time,
                    action.request.end_time,
                )
                if duplicate_interval(requested_interval, memory.observed_intervals):
                    request_trace[-1]["action_accepted"] = False
                    request_trace[-1]["action_rejection_reason"] = "duplicate_interval"
                    no_novel_action_rejections += 1
                    feedback = _controller_retry_feedback(
                        "duplicate_interval",
                        float(session.metadata["duration"]),
                        requested_interval=requested_interval,
                        observed_intervals=tuple(memory.observed_intervals),
                    )
                    last_controller_rejection = "duplicate_interval"
                    if evidence_steps > 0 and no_novel_action_rejections == 1:
                        decision = self._completeness(
                            sample,
                            memory,
                            request_trace,
                            controller_attempt_index * 10 + 1,
                            step_index=evidence_steps - 1,
                            prefix_index=len(perception_states) - 1,
                            frame_inventory=frame_inventory,
                        )
                        if (
                            decision is not None
                            and decision.evidence_complete
                            and memory.event_ledger
                        ):
                            complete = True
                            decisive_frame_indices = decision.frame_indices
                            stop_reason = "evidence_complete_after_duplicate_stall"
                            perception_states[-1]["evidence_complete"] = True
                            break
                        missing = (
                            decision.missing_evidence
                            if decision
                            else ("valid evidence completeness decision",)
                        )
                        feedback += (
                            " Completeness check still requires: "
                            + "; ".join(missing)
                            + "."
                        )
                    if (
                        evidence_steps > 0
                        and no_novel_action_rejections
                        >= _MAX_NO_NOVEL_ACTION_REJECTIONS
                    ):
                        stopped_without_novel_action = True
                        no_novel_action_reason = "duplicate_interval"
                        stop_reason = "evidence_incomplete_no_novel_action"
                        break
                    continue
                try:
                    observation = session.select(action.request)
                except Exception:
                    request_trace[-1]["action_accepted"] = False
                    request_trace[-1]["action_rejection_reason"] = "frame_select_error"
                    raise
                request_trace[-1]["action_accepted"] = True
                tool_step = ToolStep.from_observation("perception", observation)
                tool_steps.append(asdict(tool_step))
                tool_latency += observation.latency_s
                state_index = len(perception_states)
                state, perception_audit = self._perception_state(
                    sample,
                    observation,
                    action.request.evidence_request,
                    request_trace,
                    stage="perception",
                    seed_offset=evidence_steps * 10 + 2,
                    step_index=state_index,
                    prefix_index=state_index,
                )
                memory.merge(state)
                frame_inventory.extend(
                    zip(observation.frame_paths, observation.timestamps, strict=True)
                )
                accepted_planner_history.append(
                    (copy.deepcopy(dict(controller_messages[-1])), controller_text.strip())
                )
                perception_payload = perception_state_payload(
                    state, role_separated=self.role_config_sha256 is not None
                )
                perception_states.append(
                    {
                        "step_index": state_index,
                        "turn_index": evidence_steps,
                        "request": action.request.to_tool_arguments(),
                        "frame_paths": list(observation.frame_paths),
                        "timestamps": list(observation.timestamps),
                        "perception": perception_payload,
                        "perception_response": copy.deepcopy(perception_payload),
                        **perception_audit,
                        "memory_after": memory.to_dict(),
                        "evidence_complete": False,
                        "judge_confirmations": [],
                    }
                )
                evidence_steps += 1
                feedback = ""
                last_controller_rejection = None
                controller_step_attempts = 0
                no_novel_action_rejections = 0
            if not complete:
                if stopped_without_novel_action:
                    controller_attempt_limit_reached = False
                    controller_attempt_limit_reason = None
                else:
                    controller_attempt_limit_reached = (
                        controller_attempts >= self.max_controller_attempts
                        and evidence_steps < self.max_turns
                    )
                if stopped_without_novel_action:
                    stop_reason = "evidence_incomplete_no_novel_action"
                elif controller_attempt_limit_reached:
                    controller_attempt_limit_reason = last_controller_rejection
                    error = (
                        "ControllerAttemptLimit: exhausted "
                        f"{controller_attempts} controller attempts after "
                        f"{evidence_steps} accepted evidence steps; final rejection: "
                        f"{controller_attempt_limit_reason or 'unknown'}"
                    )
                    failure_class = (
                        "model_parse_failure"
                        if controller_attempt_limit_reason
                        in {
                            "invalid_controller_action",
                            "controller_response_truncated",
                        }
                        else "agent_policy_failure"
                    )
                    stop_reason = "controller_attempt_limit"
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
                        frame_inventory=frame_inventory,
                    )
                    complete = bool(
                        decision and decision.evidence_complete and memory.event_ledger
                    )
                    if complete and perception_states:
                        assert decision is not None
                        decisive_frame_indices = decision.frame_indices
                        perception_states[-1]["evidence_complete"] = True
                    stop_reason = (
                        "max_turns_complete" if complete else "max_turns_incomplete"
                    )

            if complete:
                first_valid_evidence_ids = memory.evidence_ids
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
                    frame_inventory=frame_inventory,
                    cited_frame_indices=decisive_frame_indices,
                )
                request_trace[-1]["valid_evidence_ids_at_stage"] = sorted(
                    first_valid_evidence_ids
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
                    confirmation_action: ControllerAction | None = None
                    confirmation_observation: FrameObservation | None = None
                    confirmation_feedback = ""
                    confirmation_retry_reason: str | None = None

                    def confirmation_planner_messages(
                        *, current_feedback: str = ""
                    ) -> list[dict[str, Any]]:
                        if self.role_config_sha256 is None:
                            return build_confirmation_controller_messages(
                                sample,
                                memory,
                                session.metadata,
                                (candidate, first.answer),
                                feedback=current_feedback,
                            )
                        return build_role_separated_confirmation_controller_messages(
                            sample,
                            memory,
                            session.metadata,
                            (candidate, first.answer),
                            accepted_planner_history,
                            feedback=current_feedback,
                        )

                    confirmation_retry_group_id = _messages_sha256(
                        confirmation_planner_messages()
                    )
                    for confirmation_attempt in range(2):
                        confirmation_messages = confirmation_planner_messages(
                            current_feedback=confirmation_feedback
                        )
                        try:
                            confirmation_text = self._chat(
                                request_trace,
                                confirmation_messages,
                                stage="confirmation_controller",
                                max_tokens=self.controller_max_tokens,
                                seed_offset=2001,
                                json_mode=False,
                                structured_outputs=controller_structured_outputs(
                                    allow_stop=False
                                ),
                                step_index=len(perception_states),
                                prefix_index=len(perception_states) - 1,
                            )
                        except RuntimeError:
                            current = request_trace[-1]
                            if (
                                current.get("stage") != "confirmation_controller"
                                or current.get("finish_reason") != "length"
                            ):
                                raise
                            current.update(
                                {
                                    "controller_attempt_index": confirmation_attempt,
                                    "retry_group_id": confirmation_retry_group_id,
                                    "attempt_index": confirmation_attempt,
                                    "retry_of_attempt": (
                                        confirmation_attempt - 1
                                        if confirmation_attempt
                                        else None
                                    ),
                                    "retry_reason": confirmation_retry_reason,
                                    "action_accepted": False,
                                    "action_rejection_reason": (
                                        "controller_response_truncated"
                                    ),
                                }
                            )
                            confirmation_retry_reason = "controller_response_truncated"
                            confirmation_feedback = _controller_retry_feedback(
                                confirmation_retry_reason,
                                float(session.metadata["duration"]),
                                allow_stop=False,
                            )
                            continue
                        current = request_trace[-1]
                        current.update(
                            {
                                "controller_attempt_index": confirmation_attempt,
                                "retry_group_id": confirmation_retry_group_id,
                                "attempt_index": confirmation_attempt,
                                "retry_of_attempt": (
                                    confirmation_attempt - 1
                                    if confirmation_retry_reason
                                    else None
                                ),
                                "retry_reason": confirmation_retry_reason,
                            }
                        )
                        confirmation_action = parse_controller_action(confirmation_text)
                        rejection_reason = None
                        if (
                            confirmation_action is None
                            or confirmation_action.action != "observe"
                            or confirmation_action.request is None
                        ):
                            parsed_reason = _controller_rejection_reason(
                                confirmation_text,
                                float(session.metadata["duration"]),
                            )
                            rejection_reason = (
                                parsed_reason
                                if parsed_reason == "invalid_interval"
                                else "invalid_confirmation_action"
                            )
                        else:
                            rejection_reason = _request_interval_rejection_reason(
                                confirmation_action.request,
                                float(session.metadata["duration"]),
                            )
                        if rejection_reason is not None:
                            current["action_accepted"] = False
                            current["action_rejection_reason"] = rejection_reason
                            confirmation_retry_reason = rejection_reason
                            confirmation_feedback = _controller_retry_feedback(
                                rejection_reason,
                                float(session.metadata["duration"]),
                                allow_stop=False,
                            )
                            confirmation_action = None
                            continue
                        assert confirmation_action is not None
                        assert confirmation_action.request is not None
                        confirmation_interval = (
                            confirmation_action.request.start_time,
                            confirmation_action.request.end_time,
                        )
                        if duplicate_interval(
                            confirmation_interval, memory.observed_intervals
                        ):
                            current["action_accepted"] = False
                            current["action_rejection_reason"] = "duplicate_interval"
                            confirmation_retry_reason = "duplicate_interval"
                            confirmation_feedback = _controller_retry_feedback(
                                confirmation_retry_reason,
                                float(session.metadata["duration"]),
                                allow_stop=False,
                                requested_interval=confirmation_interval,
                                observed_intervals=tuple(memory.observed_intervals),
                            )
                            confirmation_action = None
                            continue
                        try:
                            confirmation_observation = session.select(
                                confirmation_action.request
                            )
                        except DuplicateFrameRequestError:
                            current["action_accepted"] = False
                            current["action_rejection_reason"] = "duplicate_interval"
                            confirmation_retry_reason = "duplicate_interval"
                            confirmation_feedback = _controller_retry_feedback(
                                confirmation_retry_reason,
                                float(session.metadata["duration"]),
                                allow_stop=False,
                                requested_interval=confirmation_interval,
                                observed_intervals=tuple(memory.observed_intervals),
                            )
                            confirmation_action = None
                            continue
                        except Exception:
                            current["action_accepted"] = False
                            current["action_rejection_reason"] = "frame_select_error"
                            raise
                        current["action_accepted"] = True
                        break

                    if (
                        confirmation_action is not None
                        and confirmation_action.request is not None
                        and confirmation_observation is not None
                    ):
                        confirmation_tool = ToolStep.from_observation(
                            "confirmation_perception", confirmation_observation
                        )
                        tool_steps.append(asdict(confirmation_tool))
                        tool_latency += confirmation_observation.latency_s
                        confirmation_state, confirmation_perception_audit = (
                            self._perception_state(
                                sample,
                                confirmation_observation,
                                confirmation_action.request.evidence_request,
                                request_trace,
                                stage="confirmation_perception",
                                seed_offset=2002,
                                step_index=len(perception_states),
                                prefix_index=len(perception_states),
                            )
                        )
                        confirmation_bound_evidence_ids = memory.merge(
                            confirmation_state
                        )
                        confirmation_new_evidence_ids = tuple(
                            evidence_id
                            for evidence_id in confirmation_bound_evidence_ids
                            if evidence_id not in first_valid_evidence_ids
                        )
                        frame_inventory.extend(
                            zip(
                                confirmation_observation.frame_paths,
                                confirmation_observation.timestamps,
                                strict=True,
                            )
                        )
                        accepted_planner_history.append(
                            (
                                copy.deepcopy(dict(confirmation_messages[-1])),
                                confirmation_text.strip(),
                            )
                        )
                        confirmation_payload = perception_state_payload(
                            confirmation_state,
                            role_separated=self.role_config_sha256 is not None,
                        )
                        perception_states.append(
                            {
                                "step_index": len(perception_states),
                                "turn_index": len(perception_states),
                                "request": confirmation_action.request.to_tool_arguments(),
                                "frame_paths": list(
                                    confirmation_observation.frame_paths
                                ),
                                "timestamps": list(confirmation_observation.timestamps),
                                "perception": confirmation_payload,
                                "perception_response": copy.deepcopy(
                                    confirmation_payload
                                ),
                                **confirmation_perception_audit,
                                "memory_after": memory.to_dict(),
                                "confirmation_bound_evidence_ids": list(
                                    confirmation_bound_evidence_ids
                                ),
                                "confirmation_new_evidence_ids": list(
                                    confirmation_new_evidence_ids
                                ),
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
                            frame_inventory=frame_inventory,
                        )
                        if confirmed_complete and confirmed_complete.evidence_complete:
                            confirmation_decisive_frame_indices = (
                                confirmed_complete.frame_indices
                            )
                            perception_states[-1]["evidence_complete"] = True
                            confirmation = self._judge(
                                sample,
                                memory,
                                request_trace,
                                2004,
                                "confirmation_judge",
                                step_index=len(perception_states) - 1,
                                prefix_index=len(perception_states) - 1,
                                frame_inventory=frame_inventory,
                                cited_frame_indices=(
                                    confirmation_decisive_frame_indices
                                ),
                            )
                            request_trace[-1]["valid_evidence_ids_at_stage"] = list(
                                confirmation_bound_evidence_ids
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
                    first_valid_evidence_ids=first_valid_evidence_ids,
                    confirmation_valid_evidence_ids=(
                        confirmation_bound_evidence_ids
                    ),
                )
            else:
                stop_reason = stop_reason or "incomplete_evidence"
        except AnnotationLeakError as exc:
            annotation_check = "failed"
            error = f"{type(exc).__name__}: {exc}"
            failure_class = "annotation_leak"
            stop_reason = "annotation_leak"
        except PerceptionModelFailure as exc:
            error = f"{type(exc).__name__}: {exc}"
            failure_class = exc.failure_class
            stop_reason = "model_parse_failure"
        except Exception as exc:  # sample-local failure always falls back
            error = f"{type(exc).__name__}: {exc}"
            failure_class = "infrastructure_error"
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
            "explicit_time_rescue_audit": explicit_time_rescue_audit,
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
            "accepted_evidence_steps": evidence_steps,
            "controller_attempts": controller_attempts,
            "controller_attempt_limit": self.max_controller_attempts,
            "controller_attempt_limit_reached": controller_attempt_limit_reached,
            "controller_attempt_limit_reason": controller_attempt_limit_reason,
            "no_novel_action_rejections": no_novel_action_rejections,
            "no_novel_action_reason": no_novel_action_reason,
            "frame_inventory": [
                {"path": path, "timestamp": timestamp}
                for path, timestamp in _deduplicated_frame_inventory(frame_inventory)
            ],
            "decisive_frame_indices": list(decisive_frame_indices),
            "confirmation_decisive_frame_indices": list(
                confirmation_decisive_frame_indices
            ),
            "first_valid_evidence_ids": sorted(first_valid_evidence_ids),
            "confirmation_bound_evidence_ids": list(
                confirmation_bound_evidence_ids
            ),
            "confirmation_new_evidence_ids": list(confirmation_new_evidence_ids),
            "accepted_planner_actions": [
                action for _user, action in accepted_planner_history
            ],
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
            "model_parse_failure": failure_class == "model_parse_failure",
            "failure_class": failure_class,
        }


__all__ = [
    "CONTROLLER_OUTPUT_CONSTRAINT_VERSION",
    "CompletenessDecision",
    "ControllerAction",
    "DURATION_RESCUE_TRAJECTORY_VARIANTS",
    "EvidenceDecision",
    "EvidenceMemory",
    "FinalDecision",
    "OptionLedger",
    "OptionObservation",
    "PERCEPTION_MEMORY_ROLE_NAMES",
    "PerceptionMemoryEvaEvaluator",
    "PerceptionModelFailure",
    "PerceptionMemoryRoleBinding",
    "PerceptionState",
    "PERCEPTION_NORMALIZATION_VERSION",
    "PERCEPTION_RESPONSE_SCHEMA_VERSION",
    "REPAIR_ONLY_TRAJECTORY_VARIANTS",
    "RESCUE_TRAJECTORY_VARIANTS",
    "RUNTIME_VERIFIER_FRAME_SELECTION_POLICY",
    "RUNTIME_VERIFIER_MAX_FRAMES",
    "TimestampedFact",
    "apply_candidate_gate",
    "bind_perception_state",
    "build_completeness_messages",
    "build_cited_judge_messages",
    "build_confirmation_controller_messages",
    "build_role_separated_confirmation_controller_messages",
    "build_role_separated_controller_messages",
    "build_runtime_visual_csv_messages",
    "build_controller_messages",
    "build_judge_messages",
    "build_perception_messages",
    "build_perception_retry_messages",
    "controller_structured_outputs",
    "duplicate_interval",
    "evidence_request_addresses_unresolved",
    "explicit_time_rescue_request",
    "interval_iou",
    "messages_have_media",
    "map_runtime_verifier_frame_indices",
    "parse_completeness",
    "parse_controller_action",
    "parse_evidence_decision",
    "parse_perception_state",
    "perception_model_target",
    "perception_state_payload",
    "perception_memory_role_for_stage",
    "role_prompt_schema_bundle_sha256",
    "perception_response_format",
    "normalize_perception_state",
    "rescue_frame_request",
    "select_runtime_verifier_frames",
    "validate_perception_state_observation",
    "validate_perception_memory_role_config",
]
