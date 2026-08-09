"""Replay cached Fast Hybrid frames into Perception-Memory observation states.

This module never selects or decodes new frames.  A source trajectory contributes
only its public question/options, frozen candidate, and local cached frame steps.
The injected Qwen client is called only in the candidate-blind Perception role;
ground-truth scoring and prefix-completeness judging happen in later stages.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import re
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .perception_memory_eva import (
    EvidenceMemory,
    PERCEPTION_NORMALIZATION_VERSION,
    PerceptionState,
    build_controller_messages,
    build_perception_messages,
    parse_perception_state,
    validate_perception_state_observation,
)
from .privacy import AnnotationLeakError, assert_annotation_free_request
from .qwen_agents.core import ChatClient, FrameObservation, FrameRequest, ToolStep
from .runner import parse_question_time_range
from .schemas import ModelSample


REPLAY_VERSION = "perception_memory_replay_v3"
FRAME_SUBSAMPLE_POLICY = "uniform_nearest"
FRAME_SUBSAMPLE_VERSION = "v1"
_DATASETS = ("lvbench", "lsdbench", "cgbench")
_CHOICE_LINE = re.compile(r"(?m)^([A-H]):\s*(.+?)\s*$")
_STRICT_JSON_FENCE = re.compile(
    r"\A\s*```(?:json)?[ \t]*\r?\n(?P<body>\{.*\})\r?\n```[ \t]*\s*\Z",
    re.DOTALL | re.IGNORECASE,
)
_FLIP_GROUPS = frozenset(
    {
        "untrained_correct_sft_wrong",
        "untrained_wrong_sft_correct",
        "both_correct",
        "both_wrong",
    }
)
_FAILURE_GROUPS = frozenset(
    {
        "localization",
        "visual_fact_extraction",
        "cross_interval_memory",
        "incomplete_evidence_early_stop",
        "judging",
        "candidate_gate",
        "engineering",
    }
)
_FUNNEL_STAGES = frozenset(
    {"target_hit", "evidence_state_valid", "evidence_complete", "judge_correct"}
)


def canonical_sha256(value: Any) -> str:
    serialized = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@lru_cache(maxsize=1)
def replay_implementation_dependency_hashes() -> dict[str, str]:
    """Hash every local implementation file that can change replay semantics."""

    source = Path(__file__).resolve()
    package_root = source.parent
    paths = {
        "perception_memory_replay": source,
        "perception_memory_eva": package_root / "perception_memory_eva.py",
        "privacy": package_root / "privacy.py",
        "qwen_agent_core": package_root / "qwen_agents" / "core.py",
        "runner": package_root / "runner.py",
    }
    missing = [name for name, path in paths.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "replay implementation dependency is missing: " + ", ".join(missing)
        )
    return {name: file_sha256(path) for name, path in sorted(paths.items())}


def _nonempty(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be non-empty text")
    return value.strip()


def _finite(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{field} must be finite")
    return result


def _sha256_text(value: Any, field: str) -> str:
    text = _nonempty(value, field).lower()
    if len(text) != 64 or any(
        character not in "0123456789abcdef" for character in text
    ):
        raise ValueError(f"{field} must be a SHA-256")
    return text


def _explicit_problem(value: Any) -> bool:
    if value in (None, False, 0, "", [], {}):
        return False
    if isinstance(value, Mapping):
        return any(_explicit_problem(item) for item in value.values())
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return any(_explicit_problem(item) for item in value)
    return True


def validate_badcase_audit_summary(path: Path) -> tuple[dict[str, Any], str]:
    """Fail closed unless the exact current Test300 paired audit passed."""

    if not path.is_file():
        raise FileNotFoundError(f"paired audit summary does not exist: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError("paired audit summary is not valid JSON") from error
    if not isinstance(payload, Mapping):
        raise ValueError("paired audit summary must be an object")

    if payload.get("schema_version") != 1:
        raise ValueError("paired audit has an unsupported schema")
    if payload.get("status") != "passed" or payload.get("scope_passed") is not True:
        raise ValueError("paired audit status/scope is not passed")
    paired = payload.get("paired_samples", payload.get("samples"))
    if paired != 300 or payload.get("samples") != 300:
        raise ValueError("paired audit must contain exactly 300 paired samples")
    datasets = payload.get("datasets")
    if not isinstance(datasets, Mapping) or set(datasets) != set(_DATASETS):
        raise ValueError("paired audit must contain exactly the three frozen datasets")
    for dataset in _DATASETS:
        value = datasets[dataset]
        if not isinstance(value, Mapping) or value.get("samples") != 100:
            raise ValueError(f"paired audit {dataset} scope must be exactly 100")
        if set(value.get("funnel") or {}) != _FUNNEL_STAGES:
            raise ValueError(
                f"paired audit {dataset} has an incomplete evidence funnel"
            )
    if set(payload.get("flip_totals") or {}) != _FLIP_GROUPS:
        raise ValueError("paired audit has incomplete flip groups")
    if set(payload.get("failure_mode_totals") or {}) != _FAILURE_GROUPS:
        raise ValueError("paired audit has incomplete failure taxonomy")
    if payload.get("taxonomy_coverage_passed") is not True or payload.get(
        "taxonomy_classified"
    ) != payload.get("taxonomy_required"):
        raise ValueError("paired audit has unclassified failures")
    for key in (
        "duplicate_ids",
        "duplicate_sample_ids",
        "duplicate_count",
        "duplicate_errors",
        "scope_errors",
        "scope_mismatches",
        "scope_error_count",
    ):
        if key in payload and _explicit_problem(payload[key]):
            raise ValueError(f"paired audit reports integrity failure: {key}")
    return dict(payload), file_sha256(path)


def _message_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    return "\n".join(
        str(item.get("text") or "")
        for item in content
        if isinstance(item, Mapping) and item.get("type") == "text"
    )


def _parse_question_choices(text: str) -> tuple[str, dict[str, str]] | None:
    matches = list(_CHOICE_LINE.finditer(text))
    if len(matches) < 2:
        return None
    choices: dict[str, str] = {}
    first_start: int | None = None
    for match in matches:
        letter = match.group(1).upper()
        if letter in choices:
            return None
        choices[letter] = " ".join(match.group(2).strip().split())
        first_start = match.start() if first_start is None else first_start
    if tuple(choices) != tuple(chr(ord("A") + index) for index in range(len(choices))):
        return None
    assert first_start is not None
    prefix = text[:first_start].strip()
    marker = prefix.rfind("Question:")
    if marker < 0:
        return None
    question = " ".join(prefix[marker + len("Question:") :].strip().split())
    if not question or any(not choice for choice in choices.values()):
        return None
    return question, choices


def _problem_from_trace(row: Mapping[str, Any]) -> tuple[str, dict[str, str]]:
    parsed: dict[str, tuple[str, dict[str, str]]] = {}
    trace = row.get("request_trace")
    if not isinstance(trace, list):
        raise ValueError(
            "source trajectory lacks public_sample/question and request_trace"
        )
    for request in trace:
        if not isinstance(request, Mapping):
            continue
        messages = request.get("messages", request.get("request_messages"))
        if not isinstance(messages, list):
            continue
        for message in messages:
            if not isinstance(message, Mapping) or message.get("role") != "user":
                continue
            result = _parse_question_choices(_message_text(message.get("content")))
            if result is not None:
                parsed[canonical_sha256(result)] = result
    if len(parsed) != 1:
        raise ValueError(
            "source request trace does not contain one unambiguous public problem"
        )
    return next(iter(parsed.values()))


def public_model_sample(row: Mapping[str, Any]) -> ModelSample:
    """Extract only model-public fields; private annotations are never copied."""

    public = row.get("public_sample")
    if public is not None:
        if not isinstance(public, Mapping) or set(public) != {
            "dataset",
            "sample_id",
            "video",
            "question",
            "choices",
        }:
            raise ValueError(
                "public_sample must contain exactly the public sample fields"
            )
        source: Mapping[str, Any] = public
        question = source.get("question")
        choices = source.get("choices")
    else:
        source = row
        question = row.get("question")
        choices = row.get("choices")
        if not isinstance(question, str) or not isinstance(choices, Mapping):
            question, choices = _problem_from_trace(row)

    normalized_choices: dict[str, str] = {}
    if not isinstance(choices, Mapping) or len(choices) < 2:
        raise ValueError("source choices must contain at least two options")
    for raw_letter, raw_choice in choices.items():
        letter = str(raw_letter).strip().upper()
        if len(letter) != 1 or not "A" <= letter <= "H" or letter in normalized_choices:
            raise ValueError("source choices must use unique A-H letters")
        normalized_choices[letter] = _nonempty(raw_choice, f"choices.{letter}")
    candidate_raw = row.get("candidate_answer")
    candidate = None if candidate_raw is None else str(candidate_raw).strip().upper()
    if candidate is not None and candidate not in normalized_choices:
        raise ValueError("frozen candidate is not one of the source options")
    sample = ModelSample(
        dataset=_nonempty(source.get("dataset"), "dataset"),
        sample_id=_nonempty(source.get("sample_id"), "sample_id"),
        video=_nonempty(source.get("video"), "video"),
        question=_nonempty(question, "question"),
        choices=normalized_choices,
        candidate_answer=candidate,
    )
    for field in ("dataset", "sample_id", "video"):
        top_level = row.get(field)
        if top_level is not None and str(top_level) != str(getattr(sample, field)):
            raise ValueError(f"public_sample {field} differs from source identity")
    assert_annotation_free_request(
        {
            "dataset": sample.dataset,
            "sample_id": sample.sample_id,
            "video": sample.video,
            "question": sample.question,
            "choices": sample.choices,
        }
    )
    return sample


@dataclass(frozen=True)
class CachedFrameStep:
    request: FrameRequest
    observation: FrameObservation
    source_frame_count: int
    subsample_indices: tuple[int, ...]
    subsample_policy: str = FRAME_SUBSAMPLE_POLICY
    subsample_version: str = FRAME_SUBSAMPLE_VERSION
    frame_cap_applied: bool = False


class SourceExplicitTimeParseMismatch(ValueError):
    """Reject cached evidence selected from a misparsed explicit question time."""

    failure_type = "source_explicit_time_parse_mismatch"

    def __init__(
        self,
        expected_interval: tuple[float, float],
        requested_intervals: Sequence[tuple[float, float]],
    ) -> None:
        super().__init__(
            "cached requests do not cover the correctly parsed explicit question time"
        )
        self.expected_interval = expected_interval
        self.requested_intervals = tuple(requested_intervals)


def _reject_misparsed_explicit_time_source(
    sample: ModelSample, steps: Sequence[CachedFrameStep]
) -> None:
    expected = parse_question_time_range(sample.question)
    if expected is None:
        return
    requested = tuple(
        (step.request.start_time, step.request.end_time) for step in steps
    )
    expected_start, expected_end = expected
    if any(start <= expected_end and end >= expected_start for start, end in requested):
        return
    raise SourceExplicitTimeParseMismatch(expected, requested)


def cached_frame_steps(row: Mapping[str, Any]) -> tuple[CachedFrameStep, ...]:
    raw_steps = row.get("tool_steps", row.get("tool_calls"))
    if not isinstance(raw_steps, list) or not raw_steps:
        raise ValueError("source trajectory has no cached frame tool steps")
    steps: list[CachedFrameStep] = []
    for index, raw in enumerate(raw_steps):
        if not isinstance(raw, Mapping):
            raise ValueError(f"tool step {index} must be an object")
        frame_paths = raw.get("frame_paths")
        timestamps = raw.get("actual_timestamps", raw.get("timestamps"))
        if not isinstance(frame_paths, list) or not frame_paths:
            raise ValueError(f"tool step {index} has no cached frame_paths")
        if not isinstance(timestamps, list) or len(timestamps) != len(frame_paths):
            raise ValueError(f"tool step {index} frame_paths/timestamps count differs")
        paths: list[str] = []
        for raw_path in frame_paths:
            path = Path(_nonempty(raw_path, f"tool step {index} frame path"))
            if not path.is_absolute() or not path.is_file():
                raise FileNotFoundError(f"cached frame does not exist: {path}")
            paths.append(str(path.resolve()))
        parsed_timestamps = tuple(
            _finite(value, f"tool step {index} timestamp") for value in timestamps
        )
        if any(
            later < earlier
            for earlier, later in zip(parsed_timestamps, parsed_timestamps[1:])
        ):
            raise ValueError(f"tool step {index} timestamps must be non-decreasing")
        start = _finite(raw.get("start_time"), f"tool step {index} start_time")
        end = _finite(raw.get("end_time"), f"tool step {index} end_time")
        if end <= start:
            raise ValueError(f"tool step {index} interval is invalid")
        has_nframes = raw.get("nframes") is not None
        has_fps = raw.get("fps") is not None
        if has_nframes == has_fps:
            raise ValueError(
                f"tool step {index} must contain exactly one of nframes/fps"
            )
        nframes = int(raw["nframes"]) if has_nframes else None
        fps = _finite(raw["fps"], f"tool step {index} fps") if has_fps else None
        if nframes is not None and nframes != len(paths):
            raise ValueError(f"tool step {index} nframes does not match cached frames")
        evidence_request = raw.get(
            "evidence_request", raw.get("controller_evidence_request")
        )
        if not isinstance(evidence_request, str) or not evidence_request.strip():
            evidence_request = (
                "Identify directly visible facts in this cached interval that "
                "distinguish the answer choices."
            )
        request = FrameRequest(
            start_time=start,
            end_time=end,
            nframes=nframes,
            fps=fps,
            resize=_finite(raw.get("resize", 1.0), f"tool step {index} resize"),
            evidence_request=" ".join(evidence_request.strip().split()),
        )
        resolved_start = _finite(
            raw.get("resolved_start_time", start),
            f"tool step {index} resolved_start_time",
        )
        resolved_end = _finite(
            raw.get("resolved_end_time", end),
            f"tool step {index} resolved_end_time",
        )
        if resolved_end <= resolved_start:
            raise ValueError(f"tool step {index} resolved interval is invalid")
        observation = FrameObservation(
            request=request,
            resolved_start_time=resolved_start,
            resolved_end_time=resolved_end,
            resolved_nframes=len(paths),
            frame_paths=tuple(paths),
            timestamps=parsed_timestamps,
            backend="cached_fast_hybrid_replay",
            cache_hit=True,
            estimated_visual_tokens=int(raw.get("estimated_visual_tokens") or 0),
            latency_s=0.0,
        )
        steps.append(
            CachedFrameStep(
                request,
                observation,
                source_frame_count=len(paths),
                subsample_indices=tuple(range(len(paths))),
            )
        )
    return tuple(steps)


def _uniform_subsample_indices(
    source_count: int, retained_count: int
) -> tuple[int, ...]:
    if source_count <= 0 or retained_count <= 0 or retained_count > source_count:
        raise ValueError("frame subsample counts are invalid")
    if retained_count == source_count:
        return tuple(range(source_count))
    if retained_count == 1:
        return ((source_count - 1) // 2,)
    return tuple(
        (index * (source_count - 1) + (retained_count - 1) // 2) // (retained_count - 1)
        for index in range(retained_count)
    )


def _cap_cached_frame_step(
    cached: CachedFrameStep, max_frames_per_call: int
) -> CachedFrameStep:
    source_count = len(cached.observation.frame_paths)
    if source_count <= max_frames_per_call:
        return cached
    indices = _uniform_subsample_indices(source_count, max_frames_per_call)
    request = FrameRequest(
        start_time=cached.request.start_time,
        end_time=cached.request.end_time,
        nframes=len(indices),
        resize=cached.request.resize,
        evidence_request=cached.request.evidence_request,
    )
    visual_tokens = cached.observation.estimated_visual_tokens
    if visual_tokens:
        visual_tokens = max(1, round(visual_tokens * len(indices) / source_count))
    observation = FrameObservation(
        request=request,
        resolved_start_time=cached.observation.resolved_start_time,
        resolved_end_time=cached.observation.resolved_end_time,
        resolved_nframes=len(indices),
        frame_paths=tuple(cached.observation.frame_paths[index] for index in indices),
        timestamps=tuple(cached.observation.timestamps[index] for index in indices),
        backend=cached.observation.backend,
        cache_hit=cached.observation.cache_hit,
        estimated_visual_tokens=visual_tokens,
        latency_s=cached.observation.latency_s,
    )
    return CachedFrameStep(
        request=request,
        observation=observation,
        source_frame_count=source_count,
        subsample_indices=indices,
        frame_cap_applied=True,
    )


def _tool_target(request: FrameRequest) -> str:
    payload = {"tool": "frame_select", "arguments": request.to_tool_arguments()}
    return (
        "<tool_call>"
        + json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "</tool_call>"
    )


def _prompt_hash(messages: Sequence[Mapping[str, Any]]) -> str:
    return canonical_sha256(messages)


def bind_replay_perception_state(
    text: str,
    valid_letters: Iterable[str],
    actual_timestamps: Sequence[float],
) -> tuple[PerceptionState | None, str]:
    """Bind zero-based frame references to immutable cached timestamps."""

    candidate = (text or "").strip()
    fenced = _STRICT_JSON_FENCE.fullmatch(candidate)
    if fenced is not None:
        candidate = fenced.group("body")
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
        return parse_perception_state(text, valid_letters), "none"
    if all(
        isinstance(item, dict) and set(item) == {"frame_index", "fact"}
        for item in facts
    ):
        timestamps = tuple(float(value) for value in actual_timestamps)
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
                json.dumps(payload, ensure_ascii=False), valid_letters
            ),
            "frame_index",
        )
    return parse_perception_state(text, valid_letters), "timestamp"


def _perception_model_target(
    state: PerceptionState, actual_timestamps: Sequence[float]
) -> str:
    timestamps = tuple(float(value) for value in actual_timestamps)
    payload = state.to_dict()
    indexed_facts: list[dict[str, Any]] = []
    for fact in state.timestamped_facts:
        matches = [
            index
            for index, timestamp in enumerate(timestamps)
            if timestamp == fact.time
        ]
        if not matches:
            raise ValueError(
                "normalized perception fact is not bound to a cached frame"
            )
        indexed_facts.append({"frame_index": matches[0], "fact": fact.fact})
    payload["timestamped_facts"] = indexed_facts
    return json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )


def _retry_perception_messages(
    messages: Sequence[Mapping[str, Any]],
    reason: str,
    *,
    valid_letters: Sequence[str],
    frame_count: int,
    interval: tuple[float, float],
) -> list[dict[str, Any]]:
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
    start = _finite(interval[0], "perception retry interval start")
    end = _finite(interval[1], "perception retry interval end")
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
        "skeleton, replacing only arrays, the boolean, and next_evidence_needed: "
        + json.dumps(skeleton, ensure_ascii=False, separators=(",", ":"))
    )
    content.append({"type": "text", "text": correction})
    return retry


class ReplayAttemptFailure(ValueError):
    """Carry failed model attempts into the durable JSONL failure row."""

    def __init__(
        self,
        message: str,
        *,
        failure_type: str,
        request_trace: Sequence[Mapping[str, Any]],
        perception_states: Sequence[Mapping[str, Any]],
        tool_steps: Sequence[Mapping[str, Any]],
    ) -> None:
        super().__init__(message)
        self.failure_type = failure_type
        self.request_trace = copy.deepcopy(list(request_trace))
        self.perception_states = copy.deepcopy(list(perception_states))
        self.tool_steps = copy.deepcopy(list(tool_steps))


@dataclass(frozen=True)
class ReplayConfig:
    model: str = "Qwen3.5-9B"
    seed: int = 42
    perception_max_tokens: int = 1024
    temperature: float = 0.0
    enable_thinking: bool = False
    local_media_transport: str = "file_url"
    max_frames_per_call: int = 128
    request_timeout_s: float = 300.0

    @property
    def perception_retry_max_tokens(self) -> int:
        return self.perception_max_tokens * 2

    def __post_init__(self) -> None:
        if not self.model.strip():
            raise ValueError("model cannot be empty")
        if self.perception_max_tokens <= 0:
            raise ValueError("perception_max_tokens must be positive")
        if self.max_frames_per_call <= 0:
            raise ValueError("max_frames_per_call must be positive")
        if (
            isinstance(self.request_timeout_s, bool)
            or not isinstance(self.request_timeout_s, (int, float))
            or not math.isfinite(float(self.request_timeout_s))
            or self.request_timeout_s <= 0
        ):
            raise ValueError("request_timeout_s must be positive and finite")
        if self.temperature != 0.0 or self.enable_thinking:
            raise ValueError(
                "cached Perception replay must be greedy with thinking disabled"
            )
        if self.local_media_transport not in {"file_url", "path"}:
            raise ValueError("local_media_transport must be file_url or path")

    def fingerprint(self) -> str:
        return canonical_sha256(
            {
                "version": REPLAY_VERSION,
                "perception_normalization_version": (PERCEPTION_NORMALIZATION_VERSION),
                "model": self.model,
                "seed": self.seed,
                "perception_max_tokens": self.perception_max_tokens,
                "perception_retry_max_tokens": self.perception_retry_max_tokens,
                "temperature": self.temperature,
                "enable_thinking": self.enable_thinking,
                "local_media_transport": self.local_media_transport,
                "max_frames_per_call": self.max_frames_per_call,
                "frame_subsample_policy": FRAME_SUBSAMPLE_POLICY,
                "frame_subsample_version": FRAME_SUBSAMPLE_VERSION,
                "request_timeout_s": float(self.request_timeout_s),
                "perception_prompt": "build_perception_messages_v1",
                "memory_merge": "EvidenceMemory.merge_v1",
                "implementation_dependencies": (
                    replay_implementation_dependency_hashes()
                ),
            }
        )


class PerceptionMemoryReplay:
    """Regenerate only visual observation states from immutable cached frames."""

    def __init__(
        self,
        client: ChatClient | Sequence[ChatClient],
        config: ReplayConfig | None = None,
    ) -> None:
        clients = tuple(client) if isinstance(client, Sequence) else (client,)
        if not clients or any(not isinstance(item, ChatClient) for item in clients):
            raise ValueError("replay requires one or more ChatClient instances")
        self.clients = clients
        # Preserve the old single-client attribute for callers that inspect it.
        self.client = clients[0]
        self.config = config or ReplayConfig()
        for item in clients:
            client_timeout = getattr(item, "timeout", None)
            if client_timeout is not None and not math.isclose(
                float(client_timeout), float(self.config.request_timeout_s)
            ):
                raise ValueError(
                    "client timeout differs from ReplayConfig.request_timeout_s"
                )

    @property
    def endpoint_count(self) -> int:
        return len(self.clients)

    def _client_for_source(self, source_trajectory_id: str) -> ChatClient:
        digest = hashlib.sha256(source_trajectory_id.encode("utf-8")).digest()
        index = int.from_bytes(digest[:8], "big") % self.endpoint_count
        return self.clients[index]

    def replay(
        self,
        source: Mapping[str, Any],
        *,
        source_file_sha256: str,
        audit_summary_sha256: str,
    ) -> dict[str, Any]:
        sample = public_model_sample(source)
        steps = cached_frame_steps(source)
        _reject_misparsed_explicit_time_source(sample, steps)
        source_trajectory_id = _nonempty(
            source.get("trajectory_id"), "source trajectory_id"
        )
        client = self._client_for_source(source_trajectory_id)
        manifest_sha256 = _sha256_text(
            source.get("manifest_sha256", source.get("train600_manifest_sha256")),
            "manifest_sha256",
        )
        train600_manifest_sha256 = _sha256_text(
            source.get("train600_manifest_sha256", source.get("manifest_sha256")),
            "train600_manifest_sha256",
        )
        dataset_manifest_sha256 = _sha256_text(
            source.get("dataset_manifest_sha256"), "dataset_manifest_sha256"
        )
        candidate_results_sha256 = _sha256_text(
            source.get("candidate_results_sha256"), "candidate_results_sha256"
        )
        model_artifact_sha256 = _sha256_text(
            source.get("model_artifact_sha256"), "model_artifact_sha256"
        )
        trajectory_id = f"{source_trajectory_id}:{REPLAY_VERSION}"
        source_row_sha256 = canonical_sha256(source)
        run_fingerprint = canonical_sha256(
            {
                "config_sha256": self.config.fingerprint(),
                "source_file_sha256": source_file_sha256,
                "diagnostics_gate_sha256": audit_summary_sha256,
            }
        )
        memory = EvidenceMemory(sample.option_letters)
        trace: list[dict[str, Any]] = []
        states: list[dict[str, Any]] = []
        tool_steps: list[dict[str, Any]] = []

        duration = max(item.request.end_time for item in steps)
        video_metadata = {"duration": duration, "width": 0, "height": 0}
        for step_index, source_cached in enumerate(steps):
            cached = _cap_cached_frame_step(
                source_cached, self.config.max_frames_per_call
            )
            controller_messages = build_controller_messages(
                sample, memory, video_metadata
            )
            assert_annotation_free_request({"messages": controller_messages})
            trace.append(
                {
                    "stage": "controller",
                    "model": self.config.model,
                    "messages": copy.deepcopy(controller_messages),
                    "content": _tool_target(cached.request),
                    "reasoning_content": "",
                    "finish_reason": "source_replay",
                    "usage": {},
                    "latency_s": 0.0,
                    "seed": self.config.seed + step_index * 10,
                    "step_index": step_index,
                    "prefix_index": step_index - 1,
                    "prompt_hash": _prompt_hash(controller_messages),
                    "action_accepted": True,
                    "source_cached_action": True,
                }
            )
            perception_messages = build_perception_messages(
                sample,
                cached.observation,
                cached.request.evidence_request,
                use_frame_indices=True,
            )
            assert_annotation_free_request({"messages": perception_messages})
            retry_group_id = _prompt_hash(perception_messages)
            state = None
            retry_reason: str | None = None
            timestamp_reference_mode = "invalid"
            for attempt_index in range(2):
                attempt_messages = (
                    perception_messages
                    if attempt_index == 0
                    else _retry_perception_messages(
                        perception_messages,
                        retry_reason or "",
                        valid_letters=sample.option_letters,
                        frame_count=len(cached.observation.timestamps),
                        interval=(
                            cached.observation.resolved_start_time,
                            cached.observation.resolved_end_time,
                        ),
                    )
                )
                assert_annotation_free_request({"messages": attempt_messages})
                max_tokens = (
                    self.config.perception_max_tokens
                    if attempt_index == 0
                    else self.config.perception_retry_max_tokens
                )
                result = client.chat(
                    self.config.model,
                    attempt_messages,
                    max_tokens=max_tokens,
                    temperature=0.0,
                    seed=self.config.seed + step_index * 10 + 2,
                    response_format={"type": "json_object"},
                    chat_template_kwargs={"enable_thinking": False},
                )
                parsed, timestamp_reference_mode = bind_replay_perception_state(
                    result.content,
                    sample.option_letters,
                    cached.observation.timestamps,
                )
                attempt_failure: str | None = None
                if result.finish_reason == "length":
                    attempt_failure = "finish_reason_length"
                elif parsed is None:
                    attempt_failure = "invalid_json_or_schema"
                else:
                    try:
                        state = validate_perception_state_observation(
                            parsed, cached.observation
                        )
                    except ValueError as error:
                        if "timestamp" not in str(error):
                            raise
                        attempt_failure = "invalid_frame_reference"
                trace.append(
                    {
                        "stage": "perception",
                        "model": self.config.model,
                        "messages": copy.deepcopy(attempt_messages),
                        "content": result.content,
                        "reasoning_content": result.reasoning_content,
                        "finish_reason": result.finish_reason,
                        "usage": copy.deepcopy(result.usage),
                        "latency_s": float(result.latency_s),
                        "seed": self.config.seed + step_index * 10 + 2,
                        "step_index": step_index,
                        "prefix_index": step_index,
                        "prompt_hash": _prompt_hash(attempt_messages),
                        "retry_group_id": retry_group_id,
                        "attempt_index": attempt_index,
                        "retry_of_attempt": 0 if attempt_index else None,
                        "retry_reason": attempt_failure or retry_reason,
                        "retry_triggered": attempt_index == 0
                        and attempt_failure is not None,
                        "timestamp_reference_mode": timestamp_reference_mode,
                        "max_tokens": max_tokens,
                    }
                )
                if attempt_failure is None:
                    break
                retry_reason = attempt_failure
                state = None
            if state is None:
                if retry_reason == "finish_reason_length":
                    failure_type = "RuntimeError"
                    message = f"perception step {step_index} response was truncated after retry"
                else:
                    failure_type = "ValueError"
                    message = (
                        f"perception step {step_index} returned invalid evidence after "
                        f"retry: {retry_reason}"
                    )
                raise ReplayAttemptFailure(
                    message,
                    failure_type=failure_type,
                    request_trace=trace,
                    perception_states=states,
                    tool_steps=tool_steps,
                )
            memory.merge(state)
            perception_model_target = _perception_model_target(
                state, cached.observation.timestamps
            )
            states.append(
                {
                    "step_index": step_index,
                    "turn_index": step_index,
                    "request": cached.request.to_tool_arguments(),
                    "frame_paths": list(cached.observation.frame_paths),
                    "timestamps": list(cached.observation.timestamps),
                    "source_frame_count": cached.source_frame_count,
                    "frame_cap_applied": cached.frame_cap_applied,
                    "subsample_indices": list(cached.subsample_indices),
                    "subsample_policy": cached.subsample_policy,
                    "subsample_version": cached.subsample_version,
                    "perception": state.to_dict(),
                    "perception_response": state.to_dict(),
                    "perception_model_target": perception_model_target,
                    "memory_after": memory.to_dict(),
                    # Perception's self-report is not a correctness label.  The
                    # later three-seed, ground-truth-deferred prefix gate owns this.
                    "evidence_complete": False,
                    "judge_confirmations": [],
                    "source_perception_evidence_sufficient": state.evidence_sufficient,
                    "perception_attempts": attempt_index + 1,
                    "perception_retry_reason": retry_reason,
                }
            )
            tool_step = asdict(
                ToolStep.from_observation("perception", cached.observation)
            )
            tool_step.update(
                {
                    "source_frame_count": cached.source_frame_count,
                    "frame_cap_applied": cached.frame_cap_applied,
                    "subsample_indices": list(cached.subsample_indices),
                    "subsample_policy": cached.subsample_policy,
                    "subsample_version": cached.subsample_version,
                }
            )
            tool_steps.append(tool_step)

        prompt_tokens = sum(
            int(item["usage"].get("prompt_tokens", 0) or 0) for item in trace
        )
        completion_tokens = sum(
            int(item["usage"].get("completion_tokens", 0) or 0) for item in trace
        )
        total_tokens = sum(
            int(item["usage"].get("total_tokens", 0) or 0) for item in trace
        )
        return {
            "schema_version": 1,
            "backend": "perception_memory_replay",
            "agent_version": REPLAY_VERSION,
            "perception_normalization_version": (PERCEPTION_NORMALIZATION_VERSION),
            "dataset": sample.dataset,
            "sample_id": sample.sample_id,
            "trajectory_id": trajectory_id,
            "source_trajectory_id": source_trajectory_id,
            "source_row_sha256": source_row_sha256,
            "source_file_sha256": source_file_sha256,
            "diagnostics_gate_sha256": audit_summary_sha256,
            "audit_summary_sha256": audit_summary_sha256,
            "manifest_sha256": manifest_sha256,
            "train600_manifest_sha256": train600_manifest_sha256,
            "dataset_manifest_sha256": dataset_manifest_sha256,
            "candidate_results_sha256": candidate_results_sha256,
            "model_artifact_sha256": model_artifact_sha256,
            "config_sha256": self.config.fingerprint(),
            "experiment_config_sha256": self.config.fingerprint(),
            "run_fingerprint": run_fingerprint,
            "scoring_deferred": True,
            "public_sample": {
                "dataset": sample.dataset,
                "sample_id": sample.sample_id,
                "video": sample.video,
                "question": sample.question,
                "choices": dict(sample.choices),
            },
            "model": self.config.model,
            "request_timeout_s": float(self.config.request_timeout_s),
            "endpoint_count": self.endpoint_count,
            "candidate_answer": sample.candidate_answer,
            "candidate_rerun": 0,
            "annotation_leak_check": "passed",
            "prediction": None,
            "final_prediction": None,
            "fallback_used": False,
            "fallback_to_candidate": False,
            "event_ledger": memory.to_dict()["event_ledger"],
            "option_ledger": memory.to_dict()["option_ledger"],
            "unresolved": list(memory.unresolved),
            "observed_intervals": [list(item) for item in memory.observed_intervals],
            "evidence_complete": False,
            "stop_reason": "prefix_judging_deferred",
            "tool_steps": tool_steps,
            "perception_states": states,
            "request_trace": trace,
            "rounds": len(states),
            "turn_count": len(trace),
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "visual_tokens": sum(int(item["visual_tokens"]) for item in tool_steps),
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens,
            },
            "latency_s": sum(float(item["latency_s"]) for item in trace),
            "error": None,
            "error_type": None,
        }


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"{path}:{line_number}: invalid JSON") from error
            if not isinstance(value, Mapping):
                raise ValueError(f"{path}:{line_number}: row must be an object")
            rows.append(dict(value))
    return rows


def _write_jsonl_atomic(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False
    ) as handle:
        temporary = Path(handle.name)
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _source_identity(row: Mapping[str, Any]) -> tuple[str, str, str]:
    dataset = _nonempty(row.get("dataset"), "dataset")
    sample_id = _nonempty(row.get("sample_id"), "sample_id")
    source_id = _nonempty(row.get("trajectory_id"), "trajectory_id")
    return dataset, sample_id, source_id


def _failure_row(
    source: Mapping[str, Any],
    error: Exception,
    *,
    source_file_sha256: str,
    audit_summary_sha256: str,
    config: ReplayConfig,
    endpoint_count: int,
) -> dict[str, Any]:
    dataset, sample_id, source_id = _source_identity(source)
    annotation = "failed" if isinstance(error, AnnotationLeakError) else "passed"
    error_type = str(getattr(error, "failure_type", type(error).__name__))
    row = {
        "schema_version": 1,
        "backend": "perception_memory_replay",
        "agent_version": REPLAY_VERSION,
        "perception_normalization_version": PERCEPTION_NORMALIZATION_VERSION,
        "dataset": dataset,
        "sample_id": sample_id,
        "trajectory_id": f"{source_id}:{REPLAY_VERSION}",
        "source_trajectory_id": source_id,
        "source_row_sha256": canonical_sha256(source),
        "source_file_sha256": source_file_sha256,
        "diagnostics_gate_sha256": audit_summary_sha256,
        "audit_summary_sha256": audit_summary_sha256,
        "config_sha256": config.fingerprint(),
        "request_timeout_s": float(config.request_timeout_s),
        "endpoint_count": endpoint_count,
        "run_fingerprint": canonical_sha256(
            {
                "config_sha256": config.fingerprint(),
                "source_file_sha256": source_file_sha256,
                "diagnostics_gate_sha256": audit_summary_sha256,
            }
        ),
        "scoring_deferred": True,
        "candidate_rerun": 0,
        "annotation_leak_check": annotation,
        "perception_states": copy.deepcopy(
            list(getattr(error, "perception_states", ()))
        ),
        "request_trace": copy.deepcopy(list(getattr(error, "request_trace", ()))),
        "tool_steps": copy.deepcopy(list(getattr(error, "tool_steps", ()))),
        "error": f"{error_type}: {error}",
        "error_type": error_type,
    }
    if isinstance(error, SourceExplicitTimeParseMismatch):
        row.update(
            {
                "source_explicit_time_parse_mismatch": True,
                "question_explicit_time_range": list(error.expected_interval),
                "source_requested_intervals": [
                    list(interval) for interval in error.requested_intervals
                ],
            }
        )
    return row


def replay_jsonl(
    *,
    input_path: Path,
    output_path: Path,
    audit_summary_path: Path,
    replayer: PerceptionMemoryReplay,
    concurrency: int = 1,
    resume: bool = False,
) -> dict[str, Any]:
    """Replay a JSONL with atomic durable updates and strict resume provenance."""

    if concurrency <= 0:
        raise ValueError("concurrency must be positive")
    if not input_path.is_file():
        raise FileNotFoundError(f"source JSONL does not exist: {input_path}")
    _audit, audit_sha = validate_badcase_audit_summary(audit_summary_path)
    source_sha = file_sha256(input_path)
    rows = _read_jsonl(input_path)
    identities = [_source_identity(row) for row in rows]
    source_ids = [item[2] for item in identities]
    if len(source_ids) != len(set(source_ids)):
        raise ValueError("source JSONL contains duplicate trajectory_id")
    expected_fingerprint = canonical_sha256(
        {
            "config_sha256": replayer.config.fingerprint(),
            "source_file_sha256": source_sha,
            "diagnostics_gate_sha256": audit_sha,
        }
    )

    existing: dict[str, dict[str, Any]] = {}
    if output_path.exists():
        if not resume:
            raise RuntimeError("output exists; pass --resume to continue")
        for row in _read_jsonl(output_path):
            source_id = _nonempty(
                row.get("source_trajectory_id"), "source_trajectory_id"
            )
            if source_id in existing:
                raise RuntimeError(
                    "resume output contains duplicate source_trajectory_id"
                )
            if row.get("run_fingerprint") != expected_fingerprint:
                raise RuntimeError(
                    "resume fingerprint mismatch; source/audit/config changed"
                )
            existing[source_id] = row
        extras = sorted(set(existing) - set(source_ids))
        if extras:
            raise RuntimeError(
                f"resume output contains trajectories outside source: {extras[:3]}"
            )
    elif resume:
        existing = {}

    source_by_id = {identity[2]: row for identity, row in zip(identities, rows)}
    order = {source_id: index for index, source_id in enumerate(source_ids)}
    results = dict(existing)

    def execute(source_id: str) -> dict[str, Any]:
        source = source_by_id[source_id]
        try:
            return replayer.replay(
                source,
                source_file_sha256=source_sha,
                audit_summary_sha256=audit_sha,
            )
        except Exception as error:  # one bad cached trajectory must not stop the matrix
            return _failure_row(
                source,
                error,
                source_file_sha256=source_sha,
                audit_summary_sha256=audit_sha,
                config=replayer.config,
                endpoint_count=replayer.endpoint_count,
            )

    pending = [source_id for source_id in source_ids if source_id not in results]
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = {pool.submit(execute, source_id): source_id for source_id in pending}
        for future in as_completed(futures):
            row = future.result()
            results[str(row["source_trajectory_id"])] = row
            _write_jsonl_atomic(
                output_path,
                (
                    results[source_id]
                    for source_id in sorted(results, key=lambda item: order[item])
                ),
            )
    if not pending and not output_path.exists():
        _write_jsonl_atomic(output_path, ())
    failed = sum(bool(row.get("error")) for row in results.values())
    return {
        "status": "passed" if failed == 0 else "failed",
        "source_rows": len(rows),
        "completed": len(results),
        "failed": failed,
        "source_file_sha256": source_sha,
        "diagnostics_gate_sha256": audit_sha,
        "audit_summary_sha256": audit_sha,
        "config_sha256": replayer.config.fingerprint(),
        "endpoint_count": replayer.endpoint_count,
        "run_fingerprint": expected_fingerprint,
        "output": str(output_path.resolve()),
    }


__all__ = [
    "REPLAY_VERSION",
    "CachedFrameStep",
    "PerceptionMemoryReplay",
    "ReplayConfig",
    "cached_frame_steps",
    "bind_replay_perception_state",
    "canonical_sha256",
    "file_sha256",
    "public_model_sample",
    "replay_implementation_dependency_hashes",
    "replay_jsonl",
    "validate_badcase_audit_summary",
]
