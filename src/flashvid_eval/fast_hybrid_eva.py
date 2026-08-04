"""Frozen-candidate Hybrid using the official EVA evaluation loop.

The upstream ``eval-eva.py::single`` coroutine remains the agent state
machine.  This adapter only supplies the Qwen3.5 client, the benchmark sample,
an aggregate visual-token cap, audit fields, and the conservative v3c-style
candidate gate.
"""

from __future__ import annotations

import asyncio
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
from typing import Any

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
_ACTIVE_FORCED_TOOL_CALL: contextvars.ContextVar[dict[str, Any] | None] = (
    contextvars.ContextVar("fast_hybrid_eva_forced_tool_call", default=None)
)
_MODULE_LOCK = threading.Lock()
_OFFICIAL_MODULE: Any | None = None
_ORIGINAL_FRAME_SELECT: Any | None = None


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


class _OfficialAsyncCompletions:
    async def create(self, **kwargs: Any) -> Any:
        client = _ACTIVE_CLIENT.get()
        call_log = _ACTIVE_CALLS.get()
        forced = _ACTIVE_FORCED_TOOL_CALL.get()
        if forced is not None and not forced["used"]:
            forced["used"] = True
            content = (
                "<tool_call>"
                + json.dumps(forced["call"], separators=(",", ":"))
                + "</tool_call>"
            )
            call_log.append(
                {
                    "attempt": 0,
                    "content": content,
                    "reasoning_content": "",
                    "finish_reason": "forced_official_confirmation_tool_call",
                    "usage": {},
                    "latency_s": 0.0,
                }
            )
            return SimpleNamespace(
                usage=SimpleNamespace(total_tokens=0),
                choices=[SimpleNamespace(message=SimpleNamespace(content=content))],
            )
        last_error: Exception | None = None
        for attempt in range(2):
            try:
                result = client.chat(
                    str(kwargs["model"]),
                    list(kwargs["messages"]),
                    max_tokens=int(kwargs.get("max_tokens", 2048)),
                    temperature=float(kwargs.get("temperature", 0.0)),
                    chat_template_kwargs={"enable_thinking": False},
                )
                call_log.append(
                    {
                        "attempt": attempt + 1,
                        "content": result.content,
                        "reasoning_content": result.reasoning_content,
                        "finish_reason": result.finish_reason,
                        "usage": dict(result.usage),
                        "latency_s": result.latency_s,
                    }
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
                    {
                        "attempt": attempt + 1,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
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
        state["remaining"] = remaining - estimated
        state["trace"].append(
            {
                "start_time": float(fitted["start_time"]),
                "end_time": float(fitted["end_time"]),
                "nframes": len(paths),
                "resize": resize,
                "timestamps": list(timestamps or []),
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
            "official_commit": OFFICIAL_EVA_COMMIT,
            "official_eval_sha256": _sha256(_official_dir() / "eval-eva.py"),
            "official_frame_tool": frame_tool_identity(),
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()

    def _official_run(
        self,
        sample: ModelSample,
        system_prompt: str,
        visual_budget: int,
        stage: str,
        forced_tool_call: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        video = self.index.resolve(sample.video)
        metadata = probe_video(video)
        calls: list[dict[str, Any]] = []
        tool_state = {
            "remaining": max(0, visual_budget),
            "metadata": metadata,
            "trace": [],
        }
        client_token = _ACTIVE_CLIENT.set(self.client)
        calls_token = _ACTIVE_CALLS.set(calls)
        tool_token = _ACTIVE_TOOL_STATE.set(tool_state)
        forced_token = _ACTIVE_FORCED_TOOL_CALL.set(
            {"call": forced_tool_call, "used": False}
            if forced_tool_call is not None
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
                    self.max_turns,
                    "seconds",
                    self.max_call_visual_tokens,
                    720,
                    True,
                )
            )
        finally:
            _ACTIVE_FORCED_TOOL_CALL.reset(forced_token)
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
        trace = []
        for call in tool_state["trace"]:
            trace.append({"stage": stage, **call})
        return {
            "prediction": prediction,
            "raw_response": calls[-1].get("content", "") if calls else "",
            "finish_reason": calls[-1].get("finish_reason") if calls else None,
            "rounds": int(record.get("num_rounds", len(calls))),
            "usage": usage,
            "latency_s": sum(float(call.get("latency_s", 0.0)) for call in calls),
            "visual_tokens": visual_budget - int(tool_state["remaining"]),
            "tool_calls": trace,
            "call_records": calls,
            "stop_reason": record.get("stop_reason"),
            "error": record.get("error"),
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
        remaining = max(0, self.max_total_visual_tokens - int(first["visual_tokens"]))
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
                forced_confirmation,
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
        usage = {
            key: sum(int(run["usage"][key]) for run in all_runs)
            for key in ("prompt_tokens", "completion_tokens", "total_tokens")
        }
        tool_calls = [call for run in all_runs for call in run["tool_calls"]]
        fallback = verifier not in model_sample.option_letters
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
                None if change_allowed or verifier == candidate else "independent_confirmation_failed"
            ),
            "raw_response": first["raw_response"],
            "confirmation_raw_response": (
                confirmation["raw_response"] if confirmation else ""
            ),
            "finish_reason": first["finish_reason"],
            "rounds": sum(int(run["rounds"]) for run in all_runs),
            "turn_count": sum(int(run["rounds"]) for run in all_runs),
            "visual_tokens": sum(int(run["visual_tokens"]) for run in all_runs),
            "usage": usage,
            **usage,
            "latency_s": sum(float(run["latency_s"]) for run in all_runs),
            "tool_calls": tool_calls,
            "observed_intervals": [
                [call["start_time"], call["end_time"]] for call in tool_calls
            ],
            "official_eva_commit": OFFICIAL_EVA_COMMIT,
            "official_eva_eval_sha256": _sha256(_official_dir() / "eval-eva.py"),
            "official_eva_frame_tool": frame_tool_identity(),
            "candidate_results_sha256": self.candidate_results_sha256,
            "annotation_leak_check": "passed",
            "annotation_leak_reason": "model_sample_excludes_private_scoring_fields",
            "error": first.get("error"),
        }
