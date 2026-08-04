from __future__ import annotations

import hashlib
import inspect
import json
import math
import re
import threading
import time
from copy import deepcopy
from dataclasses import asdict, dataclass, field, fields, replace
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol, runtime_checkable

from flashvid_eval.client import ChatResult
from flashvid_eval.datasets import VideoIndex
from flashvid_eval.eva_official import frame_tool_identity
from flashvid_eval.eva_official import select_frames as official_select_frames
from flashvid_eval.media import estimate_visual_tokens, probe_video
from flashvid_eval.privacy import AnnotationLeakError, assert_annotation_free_request
from flashvid_eval.qwen_protocol import mcq_answer_response_format
from flashvid_eval.qwen_token_accounting import enrich_usage_with_qwen_prompt_tokens
from flashvid_eval.schemas import ModelSample


_TOOL_BLOCK_RE = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)
_SAFE_ID_RE = re.compile(r"[^A-Za-z0-9_.-]+")


@runtime_checkable
class ChatClient(Protocol):
    def chat(
        self,
        model: str,
        messages: list[dict[str, Any]],
        max_tokens: int = 32,
        *,
        temperature: float = 0.0,
        seed: int | None = None,
        response_format: dict[str, Any] | None = None,
        logprobs: bool = False,
        top_logprobs: int | None = None,
        chat_template_kwargs: dict[str, Any] | None = None,
        sampling_params: dict[str, Any] | None = None,
        mm_processor_kwargs: dict[str, Any] | None = None,
        media_io_kwargs: dict[str, Any] | None = None,
        extra_body: dict[str, Any] | None = None,
    ) -> ChatResult: ...


@runtime_checkable
class AgentStrategy(Protocol):
    strategy_id: str

    def run(self, sample: ModelSample) -> "AgentTrace": ...

    def run_fingerprint(self) -> str: ...


@dataclass(frozen=True)
class InferenceProtocol:
    enable_thinking: bool = False
    temperature: float = 0.0
    top_p: float | None = None
    top_k: int | None = None
    min_p: float | None = None
    presence_penalty: float | None = None
    repetition_penalty: float | None = None
    seed: int = 42
    planner_max_tokens: int = 512
    observer_max_tokens: int = 768
    judge_max_tokens: int = 512
    direct_max_tokens: int = 512
    length_retry_max_tokens: int | None = None
    server_max_model_len: int = 131072
    context_safety_tokens: int = 1024
    run_context_sha256: str | None = None

    def __post_init__(self) -> None:
        for name in (
            "planner_max_tokens",
            "observer_max_tokens",
            "judge_max_tokens",
            "direct_max_tokens",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.length_retry_max_tokens is not None and self.length_retry_max_tokens <= 0:
            raise ValueError("length_retry_max_tokens must be positive")
        if self.server_max_model_len <= 0 or self.context_safety_tokens < 0:
            raise ValueError("context limits must be valid")
        if self.run_context_sha256 is not None and (
            len(self.run_context_sha256) != 64
            or any(
                character not in "0123456789abcdef"
                for character in self.run_context_sha256.lower()
            )
        ):
            raise ValueError("run_context_sha256 must be a SHA-256")

    def max_tokens_for(self, request_kind: str) -> int:
        return {
            "planner": self.planner_max_tokens,
            "observer": self.observer_max_tokens,
            "judge": self.judge_max_tokens,
            "direct": self.direct_max_tokens,
        }.get(request_kind, self.judge_max_tokens)

    def sampling_params(self) -> dict[str, float | int]:
        values = {
            "top_p": self.top_p,
            "top_k": self.top_k,
            "min_p": self.min_p,
            "presence_penalty": self.presence_penalty,
            "repetition_penalty": self.repetition_penalty,
        }
        return {key: value for key, value in values.items() if value is not None}


@dataclass(frozen=True)
class AgentConfig:
    strategy: str
    overview_frames: int = 64
    local_fps: float = 1.0
    local_window_s: float = 120.0
    max_intervals: int = 3
    max_turns: int = 6
    max_frames_per_call: int = 128
    resize: float = 0.75
    hierarchy_nodes: int = 8
    hierarchy_depth: int = 3
    evidence_strategy: str = "a3_hierarchical_search"
    direct_sampling: str = "uniform64"

    def __post_init__(self) -> None:
        if self.overview_frames <= 0 or self.max_frames_per_call <= 0:
            raise ValueError("frame limits must be positive")
        if self.local_fps <= 0 or self.local_window_s <= 0:
            raise ValueError("local sampling values must be positive")
        if self.max_intervals <= 0 or self.max_turns <= 0:
            raise ValueError("agent limits must be positive")
        if not 0.05 <= self.resize <= 2.0:
            raise ValueError("resize must be in [0.05, 2.0]")
        if self.hierarchy_nodes < 2 or self.hierarchy_depth <= 0:
            raise ValueError("hierarchy settings must be positive")
        if self.strategy == "a4_independent_arbitration" and self.evidence_strategy.startswith("a4"):
            raise ValueError("A4 evidence_strategy cannot recursively select A4")
        if self.direct_sampling not in {"uniform32", "uniform64", "uniform128", "fps2"}:
            raise ValueError("direct_sampling must be uniform32, uniform64, uniform128, or fps2")

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "AgentConfig":
        known = {item.name for item in fields(cls)}
        unknown = sorted(set(payload) - known)
        if unknown:
            raise ValueError(f"unknown agent config fields: {unknown}")
        return cls(**dict(payload))


@dataclass(frozen=True)
class FrameRequest:
    start_time: float
    end_time: float
    resize: float = 1.0
    nframes: int | None = None
    fps: float | None = None
    evidence_request: str = ""

    def __post_init__(self) -> None:
        if not math.isfinite(self.start_time) or not math.isfinite(self.end_time):
            raise ValueError("frame interval must be finite")
        if self.end_time <= self.start_time:
            raise ValueError("end_time must be greater than start_time")
        if (self.nframes is None) == (self.fps is None):
            raise ValueError("exactly one of nframes or fps must be provided")
        if self.nframes is not None and self.nframes <= 0:
            raise ValueError("nframes must be positive")
        if self.fps is not None and (not math.isfinite(self.fps) or self.fps <= 0):
            raise ValueError("fps must be positive and finite")
        if not math.isfinite(self.resize) or not 0.05 <= self.resize <= 2.0:
            raise ValueError("resize must be in [0.05, 2.0]")

    def to_tool_arguments(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "start_time": self.start_time,
            "end_time": self.end_time,
            "resize": self.resize,
        }
        if self.nframes is not None:
            result["nframes"] = self.nframes
        else:
            result["fps"] = self.fps
        if self.evidence_request:
            result["evidence_request"] = self.evidence_request
        return result


@dataclass(frozen=True)
class FrameObservation:
    request: FrameRequest
    resolved_start_time: float
    resolved_end_time: float
    resolved_nframes: int
    frame_paths: tuple[str, ...]
    timestamps: tuple[float, ...]
    backend: str
    cache_hit: bool
    estimated_visual_tokens: int
    latency_s: float


@dataclass(frozen=True)
class RequestTrace:
    branch: str
    request_kind: str
    model: str
    messages: list[dict[str, Any]]
    content: str
    reasoning_content: str
    finish_reason: str | None
    usage: dict[str, Any]
    latency_s: float
    max_tokens: int
    temperature: float
    seed: int
    enable_thinking: bool
    sampling_params: dict[str, float | int]
    mm_processor_kwargs: dict[str, Any] | None
    media_io_kwargs: dict[str, Any] | None
    attempt_index: int
    retry_of_length: bool
    prompt_hash: str
    response_format: dict[str, Any] | None = None


@dataclass(frozen=True)
class ToolStep:
    branch: str
    request: dict[str, Any]
    start_time: float
    end_time: float
    nframes: int
    resize: float
    actual_timestamps: tuple[float, ...]
    resolved_start_time: float
    resolved_end_time: float
    resolved_nframes: int
    frame_paths: tuple[str, ...]
    timestamps: tuple[float, ...]
    backend: str
    cache_hit: bool
    visual_tokens: int
    latency_s: float

    @classmethod
    def from_observation(cls, branch: str, observation: FrameObservation) -> "ToolStep":
        return cls(
            branch=branch,
            request=observation.request.to_tool_arguments(),
            start_time=observation.resolved_start_time,
            end_time=observation.resolved_end_time,
            nframes=observation.resolved_nframes,
            resize=observation.request.resize,
            actual_timestamps=observation.timestamps,
            resolved_start_time=observation.resolved_start_time,
            resolved_end_time=observation.resolved_end_time,
            resolved_nframes=observation.resolved_nframes,
            frame_paths=observation.frame_paths,
            timestamps=observation.timestamps,
            backend=observation.backend,
            cache_hit=observation.cache_hit,
            visual_tokens=observation.estimated_visual_tokens,
            latency_s=observation.latency_s,
        )


@dataclass(frozen=True)
class EvidenceEntry:
    source: str
    interval: tuple[float, float] | None
    timestamps: tuple[float, ...]
    content: str


@dataclass
class AgentTrace:
    strategy: str
    model: str
    dataset: str
    sample_id: str
    prediction: str | None = None
    final_prediction: str | None = None
    raw_response: str = ""
    route: str | None = None
    request_trace: list[RequestTrace] = field(default_factory=list)
    tool_steps: list[ToolStep] = field(default_factory=list)
    evidence_memory: list[EvidenceEntry] = field(default_factory=list)
    branch_costs: dict[str, dict[str, Any]] = field(default_factory=dict)
    prompt_tokens: int = 0
    completion_tokens: int = 0
    reasoning_tokens: int | None = None
    visual_tokens: int | None = 0
    total_tokens: int = 0
    latency_s: float = 0.0
    turn_count: int = 0
    fallback_used: bool = False
    error: str | None = None
    error_type: str | None = None
    failure_class: str | None = None
    annotation_leak_check: str = "passed"
    candidate_rerun: int = 0
    visual_token_accounting_complete: bool = True
    run_fingerprint: str | None = None

    def finalize_costs(self) -> None:
        self.prompt_tokens = sum(_usage_number(item.usage, "prompt_tokens") for item in self.request_trace)
        self.completion_tokens = sum(
            _usage_number(item.usage, "completion_tokens") for item in self.request_trace
        )
        reasoning_values = [_reasoning_tokens(item.usage) for item in self.request_trace]
        self.reasoning_tokens = (
            sum(value for value in reasoning_values if value is not None)
            if all(value is not None for value in reasoning_values)
            else None
        )
        self.total_tokens = sum(_total_tokens(item.usage) for item in self.request_trace)
        self.latency_s = sum(item.latency_s for item in self.request_trace) + sum(
            item.latency_s for item in self.tool_steps
        )
        self.turn_count = len(self.request_trace)

        branches = {item.branch for item in self.request_trace} | {
            item.branch for item in self.tool_steps
        }
        branch_costs: dict[str, dict[str, Any]] = {}
        complete = True
        visual_total = 0
        for branch in sorted(branches):
            requests = [item for item in self.request_trace if item.branch == branch]
            tools = [item for item in self.tool_steps if item.branch == branch]
            branch_reasoning = [_reasoning_tokens(item.usage) for item in requests]
            visual_requests = [
                item for item in requests if _messages_have_visual_media(item.messages)
            ]
            actual_visual = [_api_visual_tokens(item.usage) for item in visual_requests]
            estimated = sum(item.visual_tokens for item in tools)
            branch_complete = all(value is not None for value in actual_visual)
            branch_visual: int | None
            if visual_requests:
                branch_visual = (
                    sum(value for value in actual_visual if value is not None)
                    if branch_complete
                    else None
                )
            elif tools:
                # A frame tool call without a corresponding multimodal model
                # request is not a complete end-to-end visual measurement.
                branch_complete = False
                branch_visual = None
            else:
                branch_visual = 0
            complete = complete and branch_complete
            if branch_visual is not None:
                visual_total += branch_visual
            branch_costs[branch] = {
                "request_count": len(requests),
                "tool_call_count": len(tools),
                "prompt_tokens": sum(_usage_number(item.usage, "prompt_tokens") for item in requests),
                "completion_tokens": sum(
                    _usage_number(item.usage, "completion_tokens") for item in requests
                ),
                "reasoning_tokens": (
                    sum(value for value in branch_reasoning if value is not None)
                    if all(value is not None for value in branch_reasoning)
                    else None
                ),
                "visual_tokens": branch_visual,
                "estimated_visual_tokens": estimated,
                "visual_token_accounting_complete": branch_complete,
                "total_tokens": sum(_total_tokens(item.usage) for item in requests),
                "latency_s": sum(item.latency_s for item in requests)
                + sum(item.latency_s for item in tools),
            }
        self.branch_costs = branch_costs
        self.visual_token_accounting_complete = complete
        self.visual_tokens = visual_total if complete else None

    def to_result_dict(self) -> dict[str, Any]:
        self.finalize_costs()
        return asdict(self)


class DuplicateFrameRequestError(ValueError):
    pass


FrameSelector = Callable[
    [Path, float, float, int, float, Path],
    tuple[list[Path], list[float], str],
]


def _selector_identity(selector: FrameSelector) -> dict[str, Any]:
    if selector is official_select_frames:
        return {"kind": "official_eva", **frame_tool_identity()}
    unwrapped = inspect.unwrap(selector)
    source = inspect.getsourcefile(unwrapped)
    source_path = Path(source).resolve() if source else None
    return {
        "kind": "python_callable",
        "module": getattr(unwrapped, "__module__", None),
        "qualname": getattr(unwrapped, "__qualname__", type(unwrapped).__qualname__),
        "source_path": str(source_path) if source_path is not None else None,
        "source_sha256": (
            _sha256_file(source_path)
            if source_path is not None and source_path.is_file()
            else None
        ),
    }


class FrameSession:
    def __init__(
        self,
        tool: "FrameTool",
        video: Path,
        metadata: dict[str, float | int],
        session_id: str,
    ) -> None:
        self._tool = tool
        self.video = video
        self.metadata = metadata
        self.session_id = session_id
        self._seen: set[tuple[float, float, int, float]] = set()

    def select(self, request: FrameRequest) -> FrameObservation:
        normalized, nframes = self._tool._normalize_request(request, self.metadata)
        signature = (
            round(normalized.start_time, 3),
            round(normalized.end_time, 3),
            nframes,
            round(normalized.resize, 4),
        )
        if signature in self._seen:
            raise DuplicateFrameRequestError(
                f"duplicate frame request in session {self.session_id}: {signature}"
            )
        observation = replace(
            self._tool._extract(self.video, normalized, nframes),
            request=request,
        )
        self._seen.add(signature)
        return observation


class FrameTool:
    """Official EVA frame extraction with request validation and disk caching."""

    def __init__(
        self,
        frame_root: Path,
        *,
        max_frames_per_call: int = 768,
        selector: FrameSelector = official_select_frames,
        probe: Callable[[Path], dict[str, float | int]] = probe_video,
    ) -> None:
        if max_frames_per_call <= 0:
            raise ValueError("max_frames_per_call must be positive")
        self.frame_root = frame_root.resolve()
        self.max_frames_per_call = max_frames_per_call
        self._selector = selector
        self._probe = probe
        self.selector_identity = _selector_identity(selector)
        self._guard = threading.Lock()
        self._locks: dict[str, threading.Lock] = {}

    def open_session(self, video: Path, session_id: str) -> FrameSession:
        resolved = video.resolve()
        metadata = self.probe(resolved)
        return FrameSession(self, resolved, metadata, session_id)

    def probe(self, video: Path) -> dict[str, float | int]:
        resolved = video.resolve()
        if not resolved.is_file():
            raise FileNotFoundError(resolved)
        metadata = self._probe(resolved)
        if float(metadata.get("duration", 0.0)) <= 0:
            raise ValueError(f"video has invalid duration: {resolved}")
        return dict(metadata)

    def _normalize_request(
        self,
        request: FrameRequest,
        metadata: Mapping[str, float | int],
    ) -> tuple[FrameRequest, int]:
        duration = float(metadata["duration"])
        start = max(0.0, min(float(request.start_time), max(0.0, duration - 0.001)))
        end = max(start + 0.001, min(float(request.end_time), duration))
        if request.nframes is not None:
            if request.nframes > self.max_frames_per_call:
                raise ValueError(
                    f"nframes {request.nframes} exceeds limit {self.max_frames_per_call}"
                )
            nframes = request.nframes
        else:
            assert request.fps is not None
            nframes = min(
                self.max_frames_per_call,
                max(1, int(math.ceil((end - start) * request.fps))),
            )
        return (
            FrameRequest(
                start_time=start,
                end_time=end,
                nframes=nframes,
                resize=request.resize,
                evidence_request=request.evidence_request,
            ),
            nframes,
        )

    def _extract(
        self,
        video: Path,
        request: FrameRequest,
        nframes: int,
    ) -> FrameObservation:
        fingerprint = {
            "video": str(video),
            "size": video.stat().st_size,
            "mtime_ns": video.stat().st_mtime_ns,
            "start_time": round(request.start_time, 6),
            "end_time": round(request.end_time, 6),
            "nframes": nframes,
            "resize": round(request.resize, 6),
            "selector": self.selector_identity,
        }
        key = hashlib.sha256(
            json.dumps(fingerprint, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        cache_dir = self.frame_root / key[:2] / key
        manifest_path = cache_dir / "manifest.json"
        with self._guard:
            lock = self._locks.setdefault(key, threading.Lock())
        started = time.perf_counter()
        with lock:
            cached = _read_frame_manifest(manifest_path)
            if cached is not None:
                paths, timestamps, backend = cached
                cache_hit = True
            else:
                cache_dir.mkdir(parents=True, exist_ok=True)
                paths, timestamps, backend = self._selector(
                    video,
                    request.start_time,
                    request.end_time,
                    nframes,
                    request.resize,
                    cache_dir / "frames",
                )
                paths = [Path(item).resolve() for item in paths]
                if not paths or len(paths) != len(timestamps):
                    raise RuntimeError("frame selector returned invalid paths/timestamps")
                manifest_path.write_text(
                    json.dumps(
                        {
                            "frame_paths": [str(item) for item in paths],
                            "timestamps": [float(item) for item in timestamps],
                            "backend": str(backend),
                        },
                        ensure_ascii=False,
                        indent=2,
                    ),
                    encoding="utf-8",
                )
                cache_hit = False
        latency = time.perf_counter() - started
        metadata = self._probe(video)
        visual_tokens = estimate_visual_tokens(metadata, len(paths), request.resize)
        return FrameObservation(
            request=request,
            resolved_start_time=request.start_time,
            resolved_end_time=request.end_time,
            resolved_nframes=nframes,
            frame_paths=tuple(str(item) for item in paths),
            timestamps=tuple(float(item) for item in timestamps),
            backend=str(backend),
            cache_hit=cache_hit,
            estimated_visual_tokens=visual_tokens,
            latency_s=latency,
        )


class BaseQwenAgent:
    strategy_id = "base"

    def __init__(
        self,
        *,
        client: ChatClient,
        model: str,
        video_root: Path,
        frame_tool: FrameTool,
        config: AgentConfig,
        protocol: InferenceProtocol | None = None,
    ) -> None:
        self.client = client
        self.model = model
        self.index = VideoIndex(video_root)
        self.frame_tool = frame_tool
        self.config = config
        self.protocol = protocol or InferenceProtocol()

    def run(self, sample: ModelSample) -> AgentTrace:
        trace = AgentTrace(
            strategy=self.strategy_id,
            model=self.model,
            dataset=sample.dataset,
            sample_id=sample.sample_id,
            route=self.strategy_id,
            run_fingerprint=self.run_fingerprint(),
        )
        try:
            self._run(sample, trace)
        except AnnotationLeakError as exc:
            trace.error = str(exc)
            trace.error_type = type(exc).__name__
            trace.failure_class = "annotation_leak"
            trace.annotation_leak_check = "failed"
        except Exception as exc:
            trace.error = str(exc)
            trace.error_type = type(exc).__name__
        trace.finalize_costs()
        return trace

    def run_fingerprint(self) -> str:
        implementation_files = {
            str(Path(__file__).name): _sha256_file(Path(__file__)),
            str(Path(__file__).with_name("strategies.py").name): _sha256_file(
                Path(__file__).with_name("strategies.py")
            ),
        }
        payload = {
            "runner": "qwen_agent_core_v1",
            "strategy": self.strategy_id,
            "model": self.model,
            "config": asdict(self.config),
            "protocol": asdict(self.protocol),
            "frame_tool": {
                "selector": self.frame_tool.selector_identity,
                "max_frames_per_call": self.frame_tool.max_frames_per_call,
            },
            "implementation_files": implementation_files,
        }
        serialized = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()

    def _run(self, sample: ModelSample, trace: AgentTrace) -> None:
        raise NotImplementedError

    def _resolve_video(self, sample: ModelSample) -> Path:
        return self.index.resolve(sample.video)

    def _chat(
        self,
        trace: AgentTrace,
        messages: list[dict[str, Any]],
        *,
        branch: str,
        request_kind: str,
        seed_offset: int = 0,
        response_format: dict[str, Any] | None = None,
        mm_processor_kwargs: dict[str, Any] | None = None,
        media_io_kwargs: dict[str, Any] | None = None,
    ) -> ChatResult:
        max_tokens = self.protocol.max_tokens_for(request_kind)
        seed = self.protocol.seed + seed_offset
        serialized = json.dumps(messages, ensure_ascii=False, sort_keys=True, default=str)
        prompt_hash = hashlib.sha256(serialized.encode("utf-8")).hexdigest()
        attempt_index = 0
        retry_of_length = False
        while True:
            assert_annotation_free_request(
                {
                    "messages": messages,
                    "mm_processor_kwargs": mm_processor_kwargs,
                    "media_io_kwargs": media_io_kwargs,
                    "request_kwargs": {"response_format": response_format},
                }
            )
            result = self.client.chat(
                self.model,
                messages,
                max_tokens=max_tokens,
                temperature=self.protocol.temperature,
                seed=seed,
                response_format=response_format,
                chat_template_kwargs={"enable_thinking": self.protocol.enable_thinking},
                sampling_params=self.protocol.sampling_params(),
                mm_processor_kwargs=mm_processor_kwargs,
                media_io_kwargs=media_io_kwargs,
                extra_body={"return_token_ids": True},
            )
            if isinstance(result.raw.get("prompt_token_ids"), list):
                result = replace(
                    result,
                    usage=enrich_usage_with_qwen_prompt_tokens(
                        result.usage,
                        result.raw,
                    ),
                )
            raw_choice = ((result.raw.get("choices") or [{}])[0]) if result.raw else {}
            raw_message = raw_choice.get("message") or {}
            reasoning = (
                getattr(result, "reasoning_content", "")
                or raw_message.get("reasoning")
                or raw_message.get("reasoning_content")
                or ""
            )
            finish_reason = getattr(result, "finish_reason", None) or raw_choice.get(
                "finish_reason"
            )
            trace.request_trace.append(
                RequestTrace(
                    branch=branch,
                    request_kind=request_kind,
                    model=self.model,
                    messages=deepcopy(messages),
                    content=result.content,
                    reasoning_content=str(reasoning),
                    finish_reason=str(finish_reason) if finish_reason is not None else None,
                    usage=deepcopy(result.usage),
                    latency_s=result.latency_s,
                    max_tokens=max_tokens,
                    temperature=self.protocol.temperature,
                    seed=seed,
                    enable_thinking=self.protocol.enable_thinking,
                    sampling_params=self.protocol.sampling_params(),
                    mm_processor_kwargs=deepcopy(mm_processor_kwargs),
                    media_io_kwargs=deepcopy(media_io_kwargs),
                    attempt_index=attempt_index,
                    retry_of_length=retry_of_length,
                    prompt_hash=prompt_hash,
                    response_format=deepcopy(response_format),
                )
            )
            retry_max = self.protocol.length_retry_max_tokens
            if finish_reason != "length" or retry_max is None or max_tokens >= retry_max:
                return result
            prompt_tokens = _usage_number(result.usage, "prompt_tokens")
            headroom = self.protocol.server_max_model_len - prompt_tokens - self.protocol.context_safety_tokens
            next_max = min(retry_max, headroom)
            if next_max <= max_tokens:
                return result
            max_tokens = next_max
            attempt_index += 1
            retry_of_length = True

    def _record_observation(
        self,
        trace: AgentTrace,
        branch: str,
        observation: FrameObservation,
        content: str,
        source: str,
    ) -> None:
        trace.tool_steps.append(ToolStep.from_observation(branch, observation))
        trace.evidence_memory.append(
            EvidenceEntry(
                source=source,
                interval=(observation.resolved_start_time, observation.resolved_end_time),
                timestamps=observation.timestamps,
                content=content,
            )
        )


def sample_question(sample: ModelSample) -> str:
    choices = "\n".join(f"{letter}: {text}" for letter, text in sample.choices.items())
    return f"Question: {sample.question}\nChoices:\n{choices}"


def parse_answer_json(text: str, valid_letters: tuple[str, ...]) -> str | None:
    valid = {letter.upper() for letter in valid_letters}
    candidate = (text or "").strip()
    try:
        payload = json.loads(candidate)
    except (TypeError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict) or set(payload) != {"answer"}:
        return None
    answer = str(payload["answer"]).strip().upper()
    return answer if answer in valid else None


def parse_frame_tool_calls(text: str, *, limit: int | None = None) -> list[FrameRequest]:
    calls: list[FrameRequest] = []
    for block in _TOOL_BLOCK_RE.findall(text or ""):
        for payload in json_objects(block):
            if payload.get("tool") != "frame_select":
                continue
            arguments = payload.get("arguments")
            if not isinstance(arguments, Mapping):
                continue
            try:
                has_nframes = "nframes" in arguments
                has_fps = "fps" in arguments
                if has_nframes == has_fps:
                    continue
                calls.append(
                    FrameRequest(
                        start_time=float(arguments["start_time"]),
                        end_time=float(arguments["end_time"]),
                        nframes=int(arguments["nframes"]) if has_nframes else None,
                        fps=float(arguments["fps"]) if has_fps else None,
                        resize=float(arguments.get("resize", 1.0)),
                        evidence_request=str(arguments.get("evidence_request", "")),
                    )
                )
            except (KeyError, TypeError, ValueError):
                continue
            if limit is not None and len(calls) >= limit:
                return calls
    return calls


def json_objects(text: str) -> list[dict[str, Any]]:
    decoder = json.JSONDecoder()
    payloads: list[dict[str, Any]] = []
    cursor = 0
    text = text or ""
    while cursor < len(text):
        start = text.find("{", cursor)
        if start < 0:
            break
        try:
            payload, consumed = decoder.raw_decode(text[start:])
        except json.JSONDecodeError:
            cursor = start + 1
            continue
        cursor = start + consumed
        if isinstance(payload, dict):
            payloads.append(payload)
    return payloads


def observation_content(observation: FrameObservation, instruction: str) -> list[dict[str, Any]]:
    content: list[dict[str, Any]] = [
        {
            "type": "text",
            "text": (
                f"<tool_response>Requested interval: "
                f"[{observation.resolved_start_time:.3f}, {observation.resolved_end_time:.3f}] seconds. "
                f"Actual frame timestamps: {[round(item, 3) for item in observation.timestamps]}.\n"
                f"{instruction}\n"
            ),
        }
    ]
    for path, timestamp in zip(observation.frame_paths, observation.timestamps):
        content.append({"type": "text", "text": f"Frame at {timestamp:.3f} seconds:"})
        content.append(
            {
                "type": "image_url",
                "image_url": {"url": Path(path).resolve().as_uri()},
            }
        )
    content.append({"type": "text", "text": "</tool_response>"})
    return content


def video_content(video: Path, text: str) -> list[dict[str, Any]]:
    return [
        {"type": "video_url", "video_url": {"url": video.resolve().as_uri()}},
        {"type": "text", "text": text},
    ]


def evidence_text(entries: list[EvidenceEntry]) -> str:
    records = [
        {
            "source": item.source,
            "interval": item.interval,
            "timestamps": item.timestamps,
            "content": item.content,
        }
        for item in entries
    ]
    return json.dumps(records, ensure_ascii=False, separators=(",", ":"), default=str)


def _read_frame_manifest(path: Path) -> tuple[list[Path], list[float], str] | None:
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        paths = [Path(item).resolve() for item in payload["frame_paths"]]
        timestamps = [float(item) for item in payload["timestamps"]]
        if not paths or len(paths) != len(timestamps) or not all(item.is_file() for item in paths):
            return None
        return paths, timestamps, str(payload["backend"])
    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
        return None


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _usage_number(usage: Mapping[str, Any], key: str) -> int:
    value = usage.get(key, 0)
    return int(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else 0


def _reasoning_tokens(usage: Mapping[str, Any]) -> int | None:
    direct = usage.get("reasoning_tokens")
    if isinstance(direct, (int, float)) and not isinstance(direct, bool):
        return int(direct)
    details = usage.get("completion_tokens_details")
    if isinstance(details, Mapping):
        value = details.get("reasoning_tokens")
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return int(value)
    return None


def _total_tokens(usage: Mapping[str, Any]) -> int:
    direct = _usage_number(usage, "total_tokens")
    return direct or _usage_number(usage, "prompt_tokens") + _usage_number(
        usage, "completion_tokens"
    )


def _api_visual_tokens(usage: Mapping[str, Any]) -> int | None:
    details = usage.get("prompt_tokens_details")
    if not isinstance(details, Mapping):
        return None
    multimodal = details.get("multimodal_tokens")
    if isinstance(multimodal, Mapping):
        value = multimodal.get("video", multimodal.get("image"))
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return int(value)
    for key in ("visual_tokens", "video_tokens", "image_tokens"):
        value = details.get(key)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return int(value)
    return None


def _messages_have_visual_media(messages: list[dict[str, Any]]) -> bool:
    serialized = json.dumps(messages, ensure_ascii=False, default=str)
    return '"video_url"' in serialized or '"image_url"' in serialized


def safe_session_id(sample: ModelSample, strategy: str) -> str:
    value = f"{sample.dataset}_{sample.sample_id}_{strategy}"
    return (_SAFE_ID_RE.sub("_", value).strip("._") or "sample")[:120]
