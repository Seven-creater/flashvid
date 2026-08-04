"""Frozen-candidate Hybrid using the official EVA evaluation loop.

The upstream ``eval-eva.py::single`` coroutine remains the agent state
machine.  This adapter only supplies the Qwen3.5 client, the benchmark sample,
an aggregate visual-token cap, audit fields, and the conservative v3c-style
candidate gate.
"""

from __future__ import annotations

import asyncio
from copy import deepcopy
import contextvars
import hashlib
import importlib.util
import json
import math
import re
import subprocess
import threading
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping

from .client import OpenAICompatibleClient
from .datasets import VideoIndex
from .eva_official import frame_tool_identity
from .media import probe_video
from .schemas import ModelSample, Sample


OFFICIAL_EVA_COMMIT = "758ad8d3dcb84a8086e5d70c9afb9a6f278e8f5a"

_ACTIVE_CLIENT: contextvars.ContextVar[OpenAICompatibleClient] = contextvars.ContextVar(
    "fast_hybrid_eva_client"
)
_ACTIVE_CALLS: contextvars.ContextVar[list[dict[str, Any]]] = contextvars.ContextVar(
    "fast_hybrid_eva_calls"
)
_ACTIVE_TOOL_STATE: contextvars.ContextVar[dict[str, Any]] = contextvars.ContextVar(
    "fast_hybrid_eva_tool_state"
)
_ACTIVE_STAGE: contextvars.ContextVar[str] = contextvars.ContextVar(
    "fast_hybrid_eva_stage", default="unknown"
)
_ACTIVE_GENERATION: contextvars.ContextVar[dict[str, Any]] = contextvars.ContextVar(
    "fast_hybrid_eva_generation", default={"temperature": 0.0, "seed": 0}
)
_ACTIVE_FORCED_TOOL_CALLS: contextvars.ContextVar[dict[str, Any] | None] = (
    contextvars.ContextVar("fast_hybrid_eva_forced_tool_calls", default=None)
)
_MODULE_LOCK = threading.Lock()
_OFFICIAL_MODULE: Any | None = None
_ORIGINAL_FRAME_SELECT: Any | None = None

_TRAJECTORY_CONTEXT_KEYS = {
    "experiment_config_sha256",
    "model_artifact_sha256",
    "manifest_sha256",
    "train600_manifest_sha256",
    "trajectory_schedule_id",
    "trajectory_variant_id",
    "trajectory_replica_id",
}


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _official_dir() -> Path:
    return _repo_root() / "third_party" / "EfficientVideoAgent"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _official_commit(path: Path) -> str:
    result = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
        timeout=5,
    )
    commit = result.stdout.strip().lower()
    if result.returncode != 0 or len(commit) != 40:
        raise RuntimeError(f"cannot identify official EVA checkout at {path}")
    return commit


def _usage_int(usage: dict[str, Any], key: str) -> int:
    value = usage.get(key, 0)
    return int(value) if isinstance(value, (int, float)) else 0


def _token_count(value: Any) -> int | None:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) < 0
    ):
        return None
    return int(value)


def _api_visual_tokens(usage: Mapping[str, Any]) -> int | None:
    """Read actual multimodal prompt tokens from an OpenAI-compatible usage."""

    for container_name in ("prompt_tokens_details", "input_tokens_details"):
        details = usage.get(container_name)
        if not isinstance(details, Mapping):
            continue
        multimodal = details.get("multimodal_tokens")
        if isinstance(multimodal, Mapping) and multimodal:
            values = [_token_count(value) for value in multimodal.values()]
            if all(value is not None for value in values):
                return sum(int(value) for value in values)
            return None
        direct = _token_count(multimodal)
        if direct is not None:
            return direct
        values = [
            _token_count(details.get(key))
            for key in ("visual_tokens", "video_tokens", "image_tokens")
            if details.get(key) is not None
        ]
        if values and all(value is not None for value in values):
            return sum(int(value) for value in values)
    return _token_count(usage.get("visual_tokens"))


def _messages_have_visual_media(messages: Any) -> bool:
    if not isinstance(messages, list):
        return False
    for message in messages:
        if not isinstance(message, Mapping):
            continue
        content = message.get("content")
        items = content if isinstance(content, list) else [content]
        for item in items:
            if not isinstance(item, Mapping):
                continue
            item_type = str(item.get("type") or "").lower()
            if item_type in {"image", "image_url", "video", "video_url"}:
                return True
            if "image_url" in item or "video_url" in item:
                return True
    return False


def _actual_visual_usage(calls: list[dict[str, Any]]) -> tuple[int | None, bool]:
    total = 0
    for call in calls:
        if call.get("source") != "model":
            continue
        has_media = _messages_have_visual_media(call.get("messages"))
        if call.get("error"):
            if has_media:
                return None, False
            continue
        if not has_media:
            continue
        usage = call.get("usage")
        visual = _api_visual_tokens(usage if isinstance(usage, Mapping) else {})
        if visual is None:
            return None, False
        total += visual
    return total, True


def _actual_total_usage_complete(calls: list[dict[str, Any]]) -> bool:
    model_calls = [
        call for call in calls if call.get("source") == "model"
    ]
    return bool(model_calls) and all(
        not call.get("error")
        and isinstance(call.get("usage"), Mapping)
        and all(
            _token_count(call["usage"].get(key)) is not None
            for key in ("prompt_tokens", "completion_tokens", "total_tokens")
        )
        for call in model_calls
    )


def _canonical_sha256(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _validated_trajectory_context(
    value: dict[str, Any] | None,
) -> dict[str, Any]:
    context = dict(value or {})
    unknown = sorted(set(context) - _TRAJECTORY_CONTEXT_KEYS)
    if unknown:
        raise ValueError(
            "unsupported trajectory_context keys: " + ", ".join(unknown)
        )
    for key in (
        "experiment_config_sha256",
        "model_artifact_sha256",
        "manifest_sha256",
        "train600_manifest_sha256",
    ):
        digest = context.get(key)
        if digest is None:
            continue
        normalized = str(digest).strip().lower()
        if len(normalized) != 64 or any(
            character not in "0123456789abcdef" for character in normalized
        ):
            raise ValueError(f"trajectory_context {key} must be a SHA-256 digest")
        context[key] = normalized
    for key in ("trajectory_schedule_id", "trajectory_variant_id"):
        identifier = context.get(key)
        if identifier is not None:
            normalized = str(identifier).strip()
            if not normalized:
                raise ValueError(f"trajectory_context {key} cannot be empty")
            context[key] = normalized
    replica = context.get("trajectory_replica_id")
    if replica is not None:
        if isinstance(replica, bool) or not isinstance(replica, int) or replica < 0:
            raise ValueError(
                "trajectory_context trajectory_replica_id must be a non-negative integer"
            )
    return context


def _tool_call_blocks(content: str) -> list[str]:
    return [
        match.group(0)
        for match in re.finditer(
            r"<tool_call>.*?</tool_call>", content or "", flags=re.DOTALL
        )
    ]


def _request_record(
    kwargs: dict[str, Any],
    *,
    attempt: int,
    content: str = "",
    reasoning_content: str = "",
    finish_reason: str | None = None,
    usage: dict[str, Any] | None = None,
    latency_s: float = 0.0,
    error: str | None = None,
    source: str = "model",
) -> dict[str, Any]:
    messages = deepcopy(list(kwargs.get("messages") or []))
    record: dict[str, Any] = {
        "stage": _ACTIVE_STAGE.get(),
        "branch": _ACTIVE_STAGE.get(),
        "request_kind": "eva_agent_turn",
        "source": source,
        "model": str(kwargs.get("model") or ""),
        "messages": messages,
        "content": content,
        "assistant_content": content,
        "assistant_tool_calls": _tool_call_blocks(content),
        "reasoning_content": reasoning_content,
        "finish_reason": finish_reason,
        "usage": dict(usage or {}),
        "latency_s": float(latency_s),
        "max_tokens": int(kwargs.get("max_tokens", 2048)),
        "temperature": float(kwargs.get("temperature", 0.0)),
        "seed": int(kwargs.get("seed", 0)),
        "enable_thinking": False,
        "attempt_index": int(attempt),
        "prompt_hash": _canonical_sha256(messages),
    }
    if error is not None:
        record["error"] = error
    return record


class _OfficialAsyncCompletions:
    async def create(self, **kwargs: Any) -> Any:
        client = _ACTIVE_CLIENT.get()
        call_log = _ACTIVE_CALLS.get()
        forced = _ACTIVE_FORCED_TOOL_CALLS.get()
        if forced is not None and int(forced["index"]) < len(forced["calls"]):
            call_index = int(forced["index"])
            forced["index"] = call_index + 1
            forced_call = forced["calls"][call_index]
            content = (
                "<tool_call>"
                + json.dumps(forced_call, separators=(",", ":"))
                + "</tool_call>"
            )
            call_log.append(
                _request_record(
                    kwargs,
                    attempt=0,
                    content=content,
                    finish_reason="forced_official_frame_select",
                    source=str(forced.get("source") or "deterministic_tool_schedule"),
                )
            )
            return SimpleNamespace(
                usage=SimpleNamespace(total_tokens=0),
                choices=[SimpleNamespace(message=SimpleNamespace(content=content))],
            )
        last_error: Exception | None = None
        generation = _ACTIVE_GENERATION.get()
        request_kwargs = dict(kwargs)
        request_kwargs["temperature"] = float(generation["temperature"])
        request_kwargs["seed"] = int(generation["seed"])
        for attempt in range(2):
            try:
                result = client.chat(
                    str(kwargs["model"]),
                    list(kwargs["messages"]),
                    max_tokens=int(kwargs.get("max_tokens", 2048)),
                    temperature=float(generation["temperature"]),
                    seed=int(generation["seed"]),
                    chat_template_kwargs={"enable_thinking": False},
                )
                call_log.append(
                    _request_record(
                        request_kwargs,
                        attempt=attempt + 1,
                        content=result.content,
                        reasoning_content=result.reasoning_content,
                        finish_reason=result.finish_reason,
                        usage=result.usage,
                        latency_s=result.latency_s,
                    )
                )
                return SimpleNamespace(
                    usage=SimpleNamespace(
                        total_tokens=_usage_int(result.usage, "total_tokens")
                    ),
                    choices=[
                        SimpleNamespace(
                            message=SimpleNamespace(content=result.content)
                        )
                    ],
                )
            except Exception as exc:  # one immediate retry, then fail the sample
                last_error = exc
                call_log.append(
                    _request_record(
                        request_kwargs,
                        attempt=attempt + 1,
                        error=f"{type(exc).__name__}: {exc}",
                    )
                )
        assert last_error is not None
        raise last_error


class _OfficialAsyncClient:
    def __init__(self) -> None:
        self.chat = SimpleNamespace(completions=_OfficialAsyncCompletions())


async def _budgeted_frame_select(
    video_path: str,
    arguments: dict[str, Any],
    fallback: bool = True,
    tool_version: str = "v3",
    tool_path: str | None = None,
) -> tuple[list[str] | None, list[int] | None]:
    """Apply only the plan's aggregate cap, then call EVA's official tool wrapper."""

    del tool_version
    state = _ACTIVE_TOOL_STATE.get()
    expected_calls = state.get("strict_expected_calls")
    if isinstance(expected_calls, list):
        call_index = len(state["trace"])
        if call_index >= len(expected_calls):
            state["strict_replay_violation"] = "unplanned_extra_frame_select"
            return None, None
        expected = expected_calls[call_index]
        for key in ("start_time", "end_time"):
            if not math.isclose(
                float(arguments.get(key, -1.0)),
                float(expected.get(key, -2.0)),
                rel_tol=0.0,
                abs_tol=1e-6,
            ):
                state["strict_replay_violation"] = (
                    f"planned_{key}_mismatch_at_call_{call_index}"
                )
                return None, None
    remaining = int(state["remaining"])
    if remaining <= 0:
        return None, None
    fitted = dict(arguments)
    nframes = max(1, int(fitted["nframes"]))
    resize = float(fitted.get("resize", 1.0) or 1.0)
    height = int(state["metadata"]["height"])
    width = int(state["metadata"]["width"])

    def estimate(frames: int, scale: float) -> int:
        h_units = max(1, round((height * scale) / 28))
        w_units = max(1, round((width * scale) / 28))
        return frames * h_units * w_units

    estimated = estimate(nframes, resize)
    if estimated > remaining:
        ratio = (remaining / estimated) ** (1.0 / 3.0)
        nframes = max(1, int(nframes * ratio))
        resize = max(0.05, resize * ratio)
        estimated = estimate(nframes, resize)
    while estimated > remaining and nframes > 1:
        nframes -= 1
        estimated = estimate(nframes, resize)
    if estimated > remaining:
        return None, None
    fitted["nframes"] = nframes
    fitted["resize"] = resize

    assert _ORIGINAL_FRAME_SELECT is not None
    paths, timestamps = await asyncio.wait_for(
        _ORIGINAL_FRAME_SELECT(
            video_path,
            fitted,
            fallback=fallback,
            tool_path=str(_official_dir() / "select_frame_fallback.py"),
        ),
        timeout=90.0,
    )
    if paths:
        resolved_paths = [str(Path(path).expanduser().resolve()) for path in paths]
        state["remaining"] = remaining - estimated
        state["trace"].append(
            {
                "start_time": float(fitted["start_time"]),
                "end_time": float(fitted["end_time"]),
                "nframes": len(paths),
                "resize": resize,
                "timestamps": list(timestamps or []),
                "frame_paths": resolved_paths,
                "estimated_visual_tokens": estimated,
                "backend": "official_eva_select_frame_fallback",
            }
        )
    return paths, timestamps


def _load_official_module() -> Any:
    global _OFFICIAL_MODULE, _ORIGINAL_FRAME_SELECT
    with _MODULE_LOCK:
        if _OFFICIAL_MODULE is not None:
            return _OFFICIAL_MODULE
        directory = _official_dir()
        commit = _official_commit(directory)
        if commit != OFFICIAL_EVA_COMMIT:
            raise RuntimeError(
                f"official EVA commit mismatch: expected {OFFICIAL_EVA_COMMIT}, got {commit}"
            )
        path = directory / "eval-eva.py"
        spec = importlib.util.spec_from_file_location(
            "flashvid_official_eva_eval", path
        )
        if spec is None or spec.loader is None:
            raise ImportError(f"cannot import official EVA evaluator: {path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _ORIGINAL_FRAME_SELECT = module.call_frame_select
        module.call_frame_select = _budgeted_frame_select
        module.aclient = _OfficialAsyncClient()
        module.FRAME_TOOL_PATH = str(directory / "select_frame_fallback.py")
        _OFFICIAL_MODULE = module
        return module


class _JsonTokenizer:
    def apply_chat_template(
        self, messages: list[dict[str, Any]], tokenize: bool = False
    ) -> str:
        if tokenize:
            raise ValueError("audit tokenizer only supports tokenize=False")
        return json.dumps(messages, ensure_ascii=False, default=str)


def _question(sample: ModelSample) -> str:
    choices = "\n".join(f"{letter}: {text}" for letter, text in sample.choices.items())
    return f"Question: {sample.question}\n{choices}"


def _verifier_system(candidate: str | None, version: str) -> str:
    candidate_text = candidate or "unavailable"
    extra = ""
    if version == "fast_hybrid_v2":
        extra = (
            " For explicit-time questions, inspect that time densely. For action, order, "
            "or counting questions, include the moments immediately before and after the event."
        )
    return (
        "Use the Frame Select Tool to analyze the video. Follow the official EVA "
        "protocol and call the tool at least once before answering. A tool request must be "
        "exactly <tool_call>{\"tool\":\"frame_select\",\"arguments\":{"
        "\"start_time\":NUMBER,\"end_time\":NUMBER,\"nframes\":INTEGER,"
        "\"resize\":NUMBER}}</tool_call>. Use valid seconds from the supplied video "
        "length, then wait for the role=tool <tool_response> observations. "
        f"Direct candidate: {candidate_text}. This is only a hypothesis, not ground truth. "
        "Try to disprove it with directly visible evidence. Keep it unless frames both "
        "contradict it and directly support another option. Do not infer unseen actions."
        f"{extra} When finished, put the final choice on the last line exactly as Answer: X."
    )


def _confirmation_system(first: str, other: str, version: str) -> str:
    extra = ""
    if version == "fast_hybrid_v2":
        extra = " Use dense before/after coverage when the distinction is temporal or action-based."
    return (
        "Independently verify a disagreement using the official EVA Frame Select Tool. "
        "Call frame_select at least once using the official <tool_call> JSON format before answering. "
        f"The two live hypotheses are {first} and {other}; neither is privileged. "
        "Inspect decisive visual evidence. Select a different or denser interval when useful. "
        "Only choose a hypothesis when the frames directly support it and contradict the other."
        f"{extra} End with exactly Answer: X."
    )


def _fixed_replay_system(candidate: str | None) -> str:
    candidate_text = candidate or "unavailable"
    return (
        "Replay the supplied frozen evidence schedule using the official EVA Frame Select "
        "Tool. The tool calls will be inserted deterministically; inspect every returned "
        "timestamped frame and do not request any additional interval. "
        f"Direct candidate: {candidate_text}. This is only a hypothesis, not ground truth. "
        "Keep it unless the visible evidence directly supports another option. After the "
        "scheduled observations, put the final choice on the last line exactly as Answer: X."
    )


def _validated_replay_calls(
    planned_calls: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if not planned_calls:
        raise ValueError("fixed evidence replay requires at least one planned call")
    calls: list[dict[str, Any]] = []
    for index, raw in enumerate(planned_calls):
        arguments = raw.get("arguments") if raw.get("tool") == "frame_select" else raw
        if not isinstance(arguments, dict):
            raise ValueError(f"planned call {index} must be an object")
        unknown = set(arguments) - {
            "start_time",
            "end_time",
            "nframes",
            "resize",
            "evidence_request",
            "source_actual_timestamps",
        }
        if unknown:
            raise ValueError(
                f"planned call {index} has unsupported fields: {sorted(unknown)}"
            )
        start = float(arguments["start_time"])
        end = float(arguments["end_time"])
        nframes = int(arguments["nframes"])
        resize = float(arguments.get("resize", 1.0))
        if not math.isfinite(start) or not math.isfinite(end) or end <= start:
            raise ValueError(f"planned call {index} has an invalid interval")
        if nframes <= 0 or not math.isfinite(resize) or resize <= 0:
            raise ValueError(f"planned call {index} has invalid frame settings")
        calls.append(
            {
                "tool": "frame_select",
                "arguments": {
                    "start_time": start,
                    "end_time": end,
                    "nframes": nframes,
                    "resize": resize,
                },
            }
        )
    return calls


_QUESTION_TIMESTAMP = re.compile(
    r"(?<!\d)(?:(\d{1,2}):)?([0-5]?\d):([0-5]\d)(?!\d)"
)


def _timestamp_seconds(match: re.Match[str]) -> float:
    return float(
        int(match.group(1) or 0) * 3600
        + int(match.group(2)) * 60
        + int(match.group(3))
    )


def _v2_confirmation_call(
    sample: ModelSample, first: dict[str, Any]
) -> dict[str, Any] | None:
    """Create the pre-registered dense re-observation for a proposed change."""

    calls = list(first.get("tool_calls") or [])
    if not calls:
        return None
    observed = calls[-1]
    start = float(observed["start_time"])
    end = float(observed["end_time"])
    timestamps = list(_QUESTION_TIMESTAMP.finditer(sample.question))
    if timestamps:
        first_second = _timestamp_seconds(timestamps[0])
        if len(timestamps) > 1:
            second = _timestamp_seconds(timestamps[1])
            start, end = min(first_second, second), max(first_second, second)
        else:
            start, end = first_second - 1.0, first_second + 1.0
        start = max(0.0, start - 1.0)
        end += 1.0
        nframes = min(96, max(8, int(math.ceil((end - start) * 4.0))))
    else:
        # Cover the visible before/after context around the verifier's own
        # decisive interval; the official visual-budget fallback may reduce it.
        start = max(0.0, start - 5.0)
        end += 5.0
        nframes = min(128, max(16, int(math.ceil(end - start))))
    return {
        "tool": "frame_select",
        "arguments": {
            "start_time": start,
            "end_time": end,
            "nframes": nframes,
            "resize": max(0.75, float(observed.get("resize", 1.0) or 1.0)),
        },
    }


class FastHybridEvaEvaluator:
    """Accuracy-first frozen-candidate Hybrid backed by official EVA ``single``."""

    def __init__(
        self,
        client: OpenAICompatibleClient,
        model: str,
        video_root: Path,
        frame_root: Path,
        *,
        version: str = "fast_hybrid_v1",
        max_turns: int = 6,
        max_call_visual_tokens: int = 12000,
        max_total_visual_tokens: int = 24000,
        candidate_results_sha256: str | None = None,
        teacher_model_sha256: str | None = None,
        experiment_config_sha256: str | None = None,
        scoring_deferred: bool = False,
        teacher_temperature: float = 0.0,
        generation_seed: int = 0,
        trajectory_context: dict[str, Any] | None = None,
    ) -> None:
        if version not in {"fast_hybrid_v1", "fast_hybrid_v2"}:
            raise ValueError(f"unsupported fast Hybrid version: {version}")
        self.client = client
        self.model = model
        self.index = VideoIndex(video_root)
        self.frame_root = frame_root.resolve()
        self.frame_root.mkdir(parents=True, exist_ok=True)
        self.version = version
        self.max_turns = max_turns
        self.max_call_visual_tokens = max_call_visual_tokens
        self.max_total_visual_tokens = max_total_visual_tokens
        self.candidate_results_sha256 = candidate_results_sha256
        self.teacher_model_sha256 = teacher_model_sha256
        self.trajectory_context = _validated_trajectory_context(trajectory_context)
        context_config = self.trajectory_context.get("experiment_config_sha256")
        if (
            experiment_config_sha256 is not None
            and context_config is not None
            and str(experiment_config_sha256).lower() != context_config
        ):
            raise ValueError(
                "experiment_config_sha256 conflicts with trajectory_context"
            )
        self.experiment_config_sha256 = (
            str(experiment_config_sha256).lower()
            if experiment_config_sha256 is not None
            else context_config
        )
        if (
            teacher_model_sha256 is not None
            and self.trajectory_context.get("model_artifact_sha256") is not None
            and str(teacher_model_sha256).lower()
            != self.trajectory_context["model_artifact_sha256"]
        ):
            raise ValueError("teacher model hash conflicts with trajectory_context")
        self.scoring_deferred = bool(scoring_deferred)
        self.teacher_temperature = float(teacher_temperature)
        self.generation_seed = int(generation_seed)
        if self.teacher_temperature < 0:
            raise ValueError("teacher_temperature cannot be negative")
        self.official = _load_official_module()
        self.official.FRAME_SAVE_ROOT = str(self.frame_root)

    def run_fingerprint(self) -> str:
        payload = {
            "runner": "fast_hybrid_eva_v1",
            "version": self.version,
            "model": self.model,
            "max_turns": self.max_turns,
            "max_call_visual_tokens": self.max_call_visual_tokens,
            "max_total_visual_tokens": self.max_total_visual_tokens,
            "candidate_results_sha256": self.candidate_results_sha256,
            "teacher_model_sha256": self.teacher_model_sha256,
            "experiment_config_sha256": self.experiment_config_sha256,
            "scoring_deferred": self.scoring_deferred,
            "teacher_temperature": self.teacher_temperature,
            "generation_seed": self.generation_seed,
            "trajectory_context": self.trajectory_context,
            "official_commit": OFFICIAL_EVA_COMMIT,
            "official_eval_sha256": _sha256(_official_dir() / "eval-eva.py"),
            "official_frame_tool": frame_tool_identity(),
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()

    def static_audit_fields(self) -> dict[str, Any]:
        context = dict(getattr(self, "trajectory_context", {}) or {})
        model_artifact = context.get("model_artifact_sha256") or getattr(
            self, "teacher_model_sha256", None
        )
        return {
            "candidate_results_sha256": getattr(
                self, "candidate_results_sha256", None
            ),
            "teacher_model_sha256": getattr(self, "teacher_model_sha256", None),
            "model_artifact_sha256": model_artifact,
            "experiment_config_sha256": (
                context.get("experiment_config_sha256")
                or getattr(self, "experiment_config_sha256", None)
            ),
            "manifest_sha256": context.get("manifest_sha256"),
            "train600_manifest_sha256": context.get("train600_manifest_sha256"),
            "trajectory_schedule_id": context.get("trajectory_schedule_id"),
            "trajectory_variant_id": context.get("trajectory_variant_id"),
            "trajectory_replica_id": context.get("trajectory_replica_id"),
            "teacher_temperature": float(
                getattr(self, "teacher_temperature", 0.0)
            ),
            "generation_seed": int(getattr(self, "generation_seed", 0)),
            "scoring_deferred": bool(getattr(self, "scoring_deferred", False)),
        }

    def _official_run(
        self,
        sample: ModelSample,
        system_prompt: str,
        visual_budget: int,
        stage: str,
        forced_tool_calls: list[dict[str, Any]] | None = None,
        *,
        strict_replay: bool = False,
        max_turns: int | None = None,
    ) -> dict[str, Any]:
        video = self.index.resolve(sample.video)
        metadata = probe_video(video)
        calls: list[dict[str, Any]] = []
        tool_state = {
            "remaining": max(0, visual_budget),
            "metadata": metadata,
            "trace": [],
            "strict_expected_calls": (
                [dict(call["arguments"]) for call in forced_tool_calls]
                if strict_replay and forced_tool_calls is not None
                else None
            ),
            "strict_replay_violation": None,
        }
        client_token = _ACTIVE_CLIENT.set(self.client)
        calls_token = _ACTIVE_CALLS.set(calls)
        tool_token = _ACTIVE_TOOL_STATE.set(tool_state)
        stage_token = _ACTIVE_STAGE.set(stage)
        generation_token = _ACTIVE_GENERATION.set(
            {
                "temperature": float(getattr(self, "teacher_temperature", 0.0)),
                "seed": int(getattr(self, "generation_seed", 0)),
            }
        )
        forced_token = _ACTIVE_FORCED_TOOL_CALLS.set(
            {
                "calls": list(forced_tool_calls),
                "index": 0,
                "source": (
                    "deterministic_fixed_evidence_replay"
                    if strict_replay
                    else "deterministic_confirmation_schedule"
                ),
            }
            if forced_tool_calls is not None
            else None
        )
        try:
            item = {
                "prompt": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": _question(sample)},
                ],
                "videos": [video.name],
            }
            record = asyncio.run(
                self.official.single(
                    0,
                    item,
                    self.model,
                    {"video_root": str(video.parent)},
                    _JsonTokenizer(),
                    max_turns if max_turns is not None else self.max_turns,
                    "seconds",
                    self.max_call_visual_tokens,
                    720,
                    True,
                )
            )
        finally:
            _ACTIVE_FORCED_TOOL_CALLS.reset(forced_token)
            _ACTIVE_GENERATION.reset(generation_token)
            _ACTIVE_STAGE.reset(stage_token)
            _ACTIVE_TOOL_STATE.reset(tool_token)
            _ACTIVE_CALLS.reset(calls_token)
            _ACTIVE_CLIENT.reset(client_token)
        answer = str(record.get("answer") or "").strip().upper()
        prediction = answer if answer in sample.option_letters else None
        usage = {
            "prompt_tokens": sum(
                _usage_int(call.get("usage", {}), "prompt_tokens") for call in calls
            ),
            "completion_tokens": sum(
                _usage_int(call.get("usage", {}), "completion_tokens") for call in calls
            ),
            "total_tokens": sum(
                _usage_int(call.get("usage", {}), "total_tokens") for call in calls
            ),
        }
        visual_tokens, visual_usage_complete = _actual_visual_usage(calls)
        visual_budget_estimate = visual_budget - int(tool_state["remaining"])
        trace = []
        for call in tool_state["trace"]:
            trace.append({"stage": stage, **call})
        serialized_messages = record.get("messages")
        try:
            messages = json.loads(serialized_messages or "[]")
        except (TypeError, json.JSONDecodeError):
            messages = []
        if not isinstance(messages, list):
            messages = []
        return {
            "prediction": prediction,
            "raw_response": calls[-1].get("content", "") if calls else "",
            "finish_reason": calls[-1].get("finish_reason") if calls else None,
            "rounds": int(record.get("num_rounds", len(calls))),
            "usage": usage,
            "latency_s": sum(float(call.get("latency_s", 0.0)) for call in calls),
            "visual_tokens": visual_tokens,
            "visual_usage_complete": visual_usage_complete,
            "visual_budget_estimate": visual_budget_estimate,
            "agent_total_tokens_complete": bool(
                not strict_replay and _actual_total_usage_complete(calls)
            ),
            "tool_calls": trace,
            "request_trace": calls,
            "messages": messages,
            "prompt_sha256": _canonical_sha256(
                [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": _question(sample)},
                ]
            ),
            "stage": stage,
            "call_records": calls,
            "stop_reason": record.get("stop_reason"),
            "error": record.get("error"),
            "strict_replay_violation": tool_state.get("strict_replay_violation"),
        }

    def fast_hybrid_eva(
        self, sample: Sample, candidate_answer: str | None
    ) -> dict[str, Any]:
        model_sample = ModelSample.from_sample(sample, candidate_answer)
        candidate = model_sample.candidate_answer
        first = self._official_run(
            model_sample,
            _verifier_system(candidate, self.version),
            self.max_total_visual_tokens,
            "verification",
        )
        verifier = first["prediction"]
        final = verifier
        confirmation: dict[str, Any] | None = None
        change_allowed = False
        remaining = max(
            0,
            self.max_total_visual_tokens - int(first["visual_budget_estimate"]),
        )
        if candidate and verifier and verifier != candidate and remaining > 0:
            forced_confirmation = (
                _v2_confirmation_call(model_sample, first)
                if self.version == "fast_hybrid_v2"
                else None
            )
            confirmation = self._official_run(
                model_sample,
                _confirmation_system(verifier, candidate, self.version),
                remaining,
                "change_confirmation",
                [forced_confirmation] if forced_confirmation is not None else None,
            )
            change_allowed = bool(
                first["tool_calls"]
                and confirmation["tool_calls"]
                and confirmation["prediction"] == verifier
            )
            if not change_allowed:
                final = candidate
        elif candidate and verifier != candidate:
            final = candidate
        if final not in model_sample.option_letters:
            final = candidate

        all_runs = [first] + ([confirmation] if confirmation is not None else [])
        run_stop_reasons = {
            str(run.get("stage") or "unknown"): run.get("stop_reason")
            for run in all_runs
        }
        run_errors = {
            str(run.get("stage") or "unknown"): str(run["error"])
            for run in all_runs
            if run.get("error")
        }
        required_run_failures: list[str] = []
        for run in all_runs:
            run_stage = str(run.get("stage") or "unknown")
            if run.get("error"):
                required_run_failures.append(f"{run_stage}: {run['error']}")
            if run.get("prediction") not in model_sample.option_letters:
                required_run_failures.append(
                    f"{run_stage}: no valid final answer "
                    f"(stop_reason={run.get('stop_reason')})"
                )
            if not run.get("tool_calls"):
                required_run_failures.append(
                    f"{run_stage}: no successful frame_select "
                    f"(stop_reason={run.get('stop_reason')})"
                )
        confirmation_required = bool(
            candidate and verifier and verifier != candidate
        )
        if confirmation_required and confirmation is None:
            required_run_failures.append(
                "change_confirmation: required confirmation was not run"
            )
        usage = {
            key: sum(int(run["usage"][key]) for run in all_runs)
            for key in ("prompt_tokens", "completion_tokens", "total_tokens")
        }
        tool_calls = [call for run in all_runs for call in run["tool_calls"]]
        request_trace = [
            request for run in all_runs for request in run.get("request_trace", [])
        ]
        conversation_traces = [
            {"stage": run.get("stage"), "messages": run.get("messages", [])}
            for run in all_runs
        ]
        messages = [
            {"stage": run.get("stage"), **message}
            for run in all_runs
            for message in run.get("messages", [])
            if isinstance(message, dict)
        ]
        prompt_hashes = {
            str(run.get("stage")): str(run.get("prompt_sha256"))
            for run in all_runs
            if run.get("stage") and run.get("prompt_sha256")
        }
        fallback = bool(
            candidate and final == candidate and verifier != candidate
        )
        aggregated_error = (
            "; ".join(dict.fromkeys(required_run_failures))
            if required_run_failures
            else None
        )
        confirmation_failed = bool(
            confirmation is not None
            and any(
                failure.startswith("change_confirmation:")
                for failure in required_run_failures
            )
        )
        visual_usage_complete = all(
            run.get("visual_usage_complete") is True for run in all_runs
        )
        visual_tokens = (
            sum(int(run["visual_tokens"]) for run in all_runs)
            if visual_usage_complete
            else None
        )
        return {
            "backend": "fast_hybrid_eva",
            "agent_version": self.version,
            "prediction": final,
            "final_prediction": final,
            "candidate_answer": candidate,
            "candidate_raw_response": "",
            "candidate_usage": {},
            "candidate_latency_s": 0.0,
            "candidate_rerun": 0,
            "candidate_changed": bool(candidate and final and final != candidate),
            "fallback_to_candidate": bool(fallback and candidate),
            "final_decision_source": (
                "confirmed_visual_change"
                if change_allowed
                else "candidate_confirmed"
                if final == candidate and verifier == candidate
                else "candidate_gate"
                if final == candidate and verifier != candidate
                else "verifier"
            ),
            "verifier_prediction": verifier,
            "change_confirmation_requested": confirmation is not None,
            "change_confirmation_observed": bool(
                confirmation and confirmation["tool_calls"]
            ),
            "change_gate_triggered": bool(candidate and verifier != candidate),
            "change_rejection_reason": (
                None
                if change_allowed or verifier == candidate
                else "verification_failed"
                if verifier not in model_sample.option_letters
                else "confirmation_not_run"
                if confirmation is None
                else "confirmation_failed"
                if confirmation_failed
                else "independent_confirmation_disagreed"
            ),
            "raw_response": first["raw_response"],
            "confirmation_raw_response": (
                confirmation["raw_response"] if confirmation else ""
            ),
            "finish_reason": first["finish_reason"],
            "run_stop_reasons": run_stop_reasons,
            "run_errors": run_errors,
            "required_run_failures": required_run_failures,
            "rounds": sum(int(run["rounds"]) for run in all_runs),
            "turn_count": sum(int(run["rounds"]) for run in all_runs),
            "visual_tokens": visual_tokens,
            "visual_usage_complete": visual_usage_complete,
            "visual_budget_estimate": sum(
                int(run["visual_budget_estimate"]) for run in all_runs
            ),
            "agent_total_tokens_complete": all(
                run.get("agent_total_tokens_complete") is True for run in all_runs
            ),
            "usage": usage,
            **usage,
            "latency_s": sum(float(run["latency_s"]) for run in all_runs),
            "tool_calls": tool_calls,
            "request_trace": request_trace,
            "conversation_traces": conversation_traces,
            "messages": messages,
            "observed_intervals": [
                [call["start_time"], call["end_time"]] for call in tool_calls
            ],
            "official_eva_commit": OFFICIAL_EVA_COMMIT,
            "official_eva_eval_sha256": _sha256(_official_dir() / "eval-eva.py"),
            "official_eva_frame_tool": frame_tool_identity(),
            **self.static_audit_fields(),
            "prompt_hashes": prompt_hashes,
            "prompt_sha256": _canonical_sha256(prompt_hashes),
            "teacher_identity_sha256": _canonical_sha256(
                {
                    "model": getattr(self, "model", ""),
                    "model_artifact_sha256": self.static_audit_fields()[
                        "model_artifact_sha256"
                    ],
                    "official_eva_commit": OFFICIAL_EVA_COMMIT,
                }
            ),
            "config_sha256": (
                getattr(self, "experiment_config_sha256", None)
                or self.run_fingerprint()
            ),
            "trajectory_schema_version": "fast_hybrid_eva_unscored_v1",
            "annotation_leak_check": "passed",
            "annotation_leak_reason": "model_sample_excludes_private_scoring_fields",
            "error": aggregated_error,
            "error_type": "required_run_failure" if aggregated_error else None,
        }

    def replay_fixed_evidence(
        self,
        sample: Sample | ModelSample,
        candidate_answer: str | None,
        planned_calls: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Replay frozen intervals through the official EVA loop, then answer once."""

        model_sample = (
            ModelSample(
                dataset=sample.dataset,
                sample_id=sample.sample_id,
                video=sample.video,
                question=sample.question,
                choices=dict(sample.choices),
                candidate_answer=candidate_answer,
            )
            if isinstance(sample, ModelSample)
            else ModelSample.from_sample(sample, candidate_answer)
        )
        calls = _validated_replay_calls(planned_calls)
        run = self._official_run(
            model_sample,
            _fixed_replay_system(model_sample.candidate_answer),
            self.max_total_visual_tokens,
            "compression_replay",
            calls,
            strict_replay=True,
            max_turns=max(self.max_turns, len(calls) + 1),
        )
        complete_schedule = (
            len(run["tool_calls"]) == len(calls)
            and run.get("strict_replay_violation") is None
        )
        prediction = run["prediction"] if complete_schedule else None
        return {
            "backend": "fast_hybrid_eva_replay",
            "agent_version": self.version,
            "prediction": prediction,
            "final_prediction": prediction,
            "candidate_answer": model_sample.candidate_answer,
            "candidate_raw_response": "",
            "candidate_usage": {},
            "candidate_latency_s": 0.0,
            "candidate_rerun": 0,
            "candidate_changed": bool(
                model_sample.candidate_answer
                and prediction
                and prediction != model_sample.candidate_answer
            ),
            "fallback_to_candidate": False,
            "final_decision_source": "fixed_evidence_replay",
            "raw_response": run["raw_response"],
            "finish_reason": run["finish_reason"],
            "rounds": run["rounds"],
            "turn_count": run["rounds"],
            "visual_tokens": run["visual_tokens"],
            "visual_usage_complete": run["visual_usage_complete"],
            "visual_budget_estimate": run["visual_budget_estimate"],
            "agent_total_tokens_complete": False,
            "usage": run["usage"],
            **run["usage"],
            "latency_s": run["latency_s"],
            "tool_calls": run["tool_calls"],
            "tool_steps": run["tool_calls"],
            "request_trace": run["request_trace"],
            "conversation_traces": [
                {"stage": run["stage"], "messages": run["messages"]}
            ],
            "messages": run["messages"],
            "planned_calls": calls,
            "planned_calls_completed": complete_schedule,
            "strict_replay_violation": run.get("strict_replay_violation"),
            "official_eva_commit": OFFICIAL_EVA_COMMIT,
            "official_eva_eval_sha256": _sha256(_official_dir() / "eval-eva.py"),
            "official_eva_frame_tool": frame_tool_identity(),
            **self.static_audit_fields(),
            "prompt_hashes": {run["stage"]: run["prompt_sha256"]},
            "prompt_sha256": run["prompt_sha256"],
            "trajectory_schema_version": "fast_hybrid_eva_replay_unscored_v1",
            "annotation_leak_check": "passed",
            "annotation_leak_reason": "model_sample_excludes_private_scoring_fields",
            "error": (
                run.get("error")
                or (
                    "fixed evidence schedule was not completed exactly"
                    if not complete_schedule
                    else None
                )
            ),
        }
