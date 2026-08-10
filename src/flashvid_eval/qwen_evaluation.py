from __future__ import annotations

import hashlib
import json
import math
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol, Sequence

from .baseline_diagnostics import (
    DEFAULT_DURATION_BUCKET_EDGES_S,
    DIRECT_SAMPLING_SPECS,
    DirectSamplingSpec,
    OptionPermutation,
    format_choices_only,
    permute_choices,
)
from .client import ChatResult, OpenAICompatibleClient
from .datasets import VideoIndex
from .media import probe_video
from .privacy import (
    AnnotationLeakError,
    assert_annotation_free_request,
    assert_deferred_result_public,
)
from .qwen_protocol import (
    QwenInferenceProtocol,
    mcq_answer_response_format,
    next_length_retry_max_tokens,
    parse_strict_json_mcq_answer,
)
from .qwen_progress import emit_progress
from .qwen_token_accounting import enrich_usage_with_qwen_prompt_tokens
from .schemas import ModelSample, Sample, ScoringRecord


def _canonical_hash(payload: Mapping[str, Any]) -> str:
    serialized = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _question_prompt(question: str, choices: Mapping[str, str], *, video: bool) -> str:
    options = "\n".join(f"{letter}. {text}" for letter, text in choices.items())
    valid_letters = ", ".join(choices)
    evidence = (
        "Use the supplied video as the visual evidence."
        if video
        else "No video, image, subtitle, timestamp, or other visual evidence is available."
    )
    return (
        f"{evidence}\nQuestion: {question}\n{options}\n"
        "X must be one of ["
        f"{valid_letters}"
        ']. Return exactly one JSON object with no prose or extra keys: {"answer":"X"}'
    )


BASELINE_PROMPT_ID = "qwen_mcq_strict_json_v2"
BASELINE_PROMPT_TEMPLATE_SHA256 = hashlib.sha256(
    (
        "evidence_instruction\nQuestion: {question}\n{letter}. {choice}\n"
        "X must be one of [{valid_letters}]. "
        'Return exactly one JSON object with no prose or extra keys: {"answer":"X"}'
    ).encode("utf-8")
).hexdigest()


def _video_content(video: Path, prompt: str) -> list[dict[str, Any]]:
    return [
        {"type": "video_url", "video_url": {"url": video.resolve().as_uri()}},
        {"type": "text", "text": prompt},
    ]


def _token_int(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return int(value)


def _reasoning_tokens(usage: Mapping[str, Any]) -> int | None:
    direct = usage.get("reasoning_tokens")
    if (parsed := _token_int(direct)) is not None:
        return parsed
    details = usage.get("completion_tokens_details")
    if isinstance(details, Mapping):
        return _token_int(details.get("reasoning_tokens"))
    return None


def _visual_token_breakdown(usage: Mapping[str, Any]) -> dict[str, int] | None:
    """Extract vLLM/OpenAI visual usage without assuming one wire shape."""

    for container_name in ("prompt_tokens_details", "input_tokens_details"):
        details = usage.get(container_name)
        if not isinstance(details, Mapping):
            continue
        for name in ("visual_tokens", "video_tokens"):
            if (value := _token_int(details.get(name))) is not None:
                return {name: value}
        multimodal = details.get("multimodal_tokens")
        if (value := _token_int(multimodal)) is not None:
            return {"multimodal": value}
        if isinstance(multimodal, Mapping):
            parsed = {
                str(kind): value
                for kind, raw in multimodal.items()
                if (value := _token_int(raw)) is not None
            }
            if parsed:
                return parsed
    if (value := _token_int(usage.get("visual_tokens"))) is not None:
        return {"visual_tokens": value}
    return None


def _visual_tokens(usage: Mapping[str, Any]) -> int | None:
    breakdown = _visual_token_breakdown(usage)
    return sum(breakdown.values()) if breakdown is not None else None


def _sample_generation_seed(base_seed: int, sample: ModelSample) -> int:
    digest = hashlib.sha256(
        f"{base_seed}\0{sample.dataset}\0{sample.sample_id}".encode("utf-8")
    ).digest()
    return int.from_bytes(digest[:8], "big") & ((1 << 63) - 1)


def _sample_fingerprint(sample: Sample) -> str:
    return _canonical_hash(
        {
            "dataset": sample.dataset,
            "sample_id": sample.sample_id,
            "video": sample.video,
            "question": sample.question,
            "choices": sample.choices,
        }
    )


def _json_duration_edges(edges: Sequence[float]) -> list[float | str]:
    return ["inf" if math.isinf(value) else float(value) for value in edges]


class ControlUnavailableError(RuntimeError):
    """A diagnostic control cannot be formed without changing its protocol."""


class DataUnavailableError(FileNotFoundError):
    """The manifest's required media is not accessible under the frozen root."""


class ModelSampleRunner(Protocol):
    def run(self, sample: ModelSample) -> Mapping[str, Any]: ...

    def run_fingerprint(self) -> str: ...


@dataclass(frozen=True)
class QwenBaselineConfig:
    mode: str
    protocol: QwenInferenceProtocol
    direct_sampling: DirectSamplingSpec | None = None
    option_permutation_seed: int | None = None
    mismatched_videos: Mapping[str, str] | None = None
    allow_length_retry: bool = True
    generation_seed: int = 42
    run_context: Mapping[str, Any] | None = None
    max_model_len: int | None = None
    retry_context_reserve_tokens: int = 16
    annotation_leak_sentinels: tuple[str, ...] = ()
    mismatch_duration_bucket_edges_s: tuple[float, ...] = (
        DEFAULT_DURATION_BUCKET_EDGES_S
    )

    def __post_init__(self) -> None:
        allowed = {
            "question_choices",
            "choices_only",
            "permuted_choices",
            "direct",
            "mismatched_video",
        }
        if self.mode not in allowed:
            raise ValueError(f"unsupported Qwen baseline mode: {self.mode}")
        needs_video = self.mode in {"direct", "mismatched_video"}
        if needs_video != (self.direct_sampling is not None):
            raise ValueError("video baseline modes require exactly one Direct sampling spec")
        if (self.mode == "permuted_choices") != (
            self.option_permutation_seed is not None
        ):
            raise ValueError("permuted_choices requires exactly one permutation seed")
        if self.mode == "mismatched_video" and self.mismatched_videos is None:
            raise ValueError("mismatched_video requires a frozen sample-to-video mapping")
        if self.max_model_len is not None and self.max_model_len <= 0:
            raise ValueError("max_model_len must be positive")
        if self.retry_context_reserve_tokens < 0:
            raise ValueError("retry_context_reserve_tokens must be non-negative")
        edges = self.mismatch_duration_bucket_edges_s
        if (
            len(edges) < 2
            or edges[0] != 0.0
            or not math.isinf(edges[-1])
            or any(left >= right for left, right in zip(edges, edges[1:]))
        ):
            raise ValueError(
                "mismatch duration bucket edges must start at 0, end at infinity, "
                "and be strictly increasing"
            )


class QwenBaselineRunner:
    def __init__(
        self,
        client: OpenAICompatibleClient,
        model: str,
        video_root: Path,
        config: QwenBaselineConfig,
    ) -> None:
        self.client = client
        self.model = model
        self.config = config
        self.index = (
            VideoIndex(video_root)
            if config.mode in {"direct", "mismatched_video"}
            else None
        )

    def run_fingerprint(self) -> str:
        sampling = (
            asdict(self.config.direct_sampling)
            if self.config.direct_sampling is not None
            else None
        )
        return _canonical_hash(
            {
                "runner": "qwen_baseline_v5",
                "model": self.model,
                "mode": self.config.mode,
                "protocol": asdict(self.config.protocol),
                "direct_sampling": sampling,
                "option_permutation_seed": self.config.option_permutation_seed,
                "mismatched_videos": dict(self.config.mismatched_videos or {}),
                "allow_length_retry": self.config.allow_length_retry,
                "generation_seed": self.config.generation_seed,
                "run_context": dict(self.config.run_context or {}),
                "max_model_len": self.config.max_model_len,
                "retry_context_reserve_tokens": (
                    self.config.retry_context_reserve_tokens
                ),
                "annotation_leak_sentinel_hashes": [
                    hashlib.sha256(value.encode("utf-8")).hexdigest()
                    for value in self.config.annotation_leak_sentinels
                ],
                "mismatch_duration_bucket_edges_s": _json_duration_edges(
                    self.config.mismatch_duration_bucket_edges_s
                ),
                "prompt_id": BASELINE_PROMPT_ID,
                "prompt_template_sha256": BASELINE_PROMPT_TEMPLATE_SHA256,
            }
        )

    def _chat(
        self,
        messages: list[dict[str, Any]],
        *,
        generation_seed: int,
        valid_letters: Sequence[str],
        mm_processor_kwargs: dict[str, Any] | None = None,
        media_io_kwargs: dict[str, Any] | None = None,
    ) -> tuple[ChatResult, list[dict[str, Any]], float]:
        requested = self.config.protocol.max_tokens
        attempts: list[dict[str, Any]] = []
        latency = 0.0
        while True:
            kwargs = self.config.protocol.request_kwargs(max_tokens=requested)
            # This is always a final MCQ turn. The Qwen3 reasoning parser keeps
            # hidden reasoning separate while vLLM constrains formal content.
            kwargs["response_format"] = mcq_answer_response_format(valid_letters)
            kwargs["extra_body"] = {"return_token_ids": True}
            assert_annotation_free_request(
                {
                    "messages": messages,
                    "mm_processor_kwargs": mm_processor_kwargs,
                    "media_io_kwargs": media_io_kwargs,
                    "request_kwargs": kwargs,
                },
                secret_sentinels=self.config.annotation_leak_sentinels,
            )
            emit_progress(
                "api_request_started",
                runner="qwen_baseline",
                mode=self.config.mode,
            )
            result = self.client.chat(
                self.model,
                messages,
                seed=generation_seed,
                mm_processor_kwargs=mm_processor_kwargs,
                media_io_kwargs=media_io_kwargs,
                **kwargs,
            )
            emit_progress(
                "api_response",
                runner="qwen_baseline",
                mode=self.config.mode,
            )
            if isinstance(result.raw.get("prompt_token_ids"), list):
                result = replace(
                    result,
                    usage=enrich_usage_with_qwen_prompt_tokens(
                        result.usage,
                        result.raw,
                    ),
                )
            latency += result.latency_s
            attempts.append(
                {
                    "max_tokens": requested,
                    "finish_reason": result.finish_reason,
                    "usage": result.usage,
                    "reasoning_tokens": _reasoning_tokens(result.usage),
                    "visual_tokens": _visual_tokens(result.usage),
                    "latency_s": result.latency_s,
                }
            )
            retry = next_length_retry_max_tokens(
                self.config.protocol,
                result,
                requested,
                prompt_tokens=_token_int(result.usage.get("prompt_tokens")),
                max_model_len=self.config.max_model_len,
                reserve_tokens=self.config.retry_context_reserve_tokens,
            )
            unrestricted_retry = next_length_retry_max_tokens(
                self.config.protocol,
                result,
                requested,
            )
            attempts[-1]["length_retry_candidate_max_tokens"] = unrestricted_retry
            attempts[-1]["length_retry_max_tokens"] = retry
            attempts[-1]["length_retry_blocked_reason"] = (
                "insufficient_or_unknown_context_headroom"
                if unrestricted_retry is not None and retry is None
                else None
            )
            if not self.config.allow_length_retry or retry is None:
                return result, attempts, latency
            requested = retry

    def run(self, sample: ModelSample) -> Mapping[str, Any]:
        generation_seed = _sample_generation_seed(
            self.config.generation_seed,
            sample,
        )
        choices = dict(sample.choices)
        permutation: OptionPermutation | None = None
        if self.config.mode == "permuted_choices":
            permutation = permute_choices(choices, int(self.config.option_permutation_seed))
            choices = permutation.choices

        video: Path | None = None
        metadata: dict[str, float | int] | None = None
        sampled_frames: int | None = None
        mm_processor_kwargs: dict[str, Any] | None = None
        media_io_kwargs: dict[str, Any] | None = None
        if self.config.mode in {"direct", "mismatched_video"}:
            video_name = sample.video
            if self.config.mode == "mismatched_video":
                mapped = (self.config.mismatched_videos or {}).get(sample.sample_id)
                if mapped is None:
                    raise ControlUnavailableError(
                        "no same-dataset duration-bucket alternative for "
                        f"sample {sample.sample_id}"
                    )
                video_name = str(mapped)
            if self.index is None:
                raise AssertionError("validated video mode lost its video index")
            try:
                video = self.index.resolve(video_name)
            except FileNotFoundError as exc:
                raise DataUnavailableError(str(exc)) from exc
            metadata = probe_video(video)
            spec = self.config.direct_sampling
            if spec is None:
                raise AssertionError("validated video mode lost sampling spec")
            duration = float(metadata["duration"])
            mm_processor_kwargs = spec.mm_processor_kwargs()
            media_io_kwargs = spec.media_io_kwargs(duration)
            sampled_frames = spec.requested_num_frames(duration)

        if self.config.mode == "choices_only":
            prompt = format_choices_only(choices)
        else:
            prompt = _question_prompt(sample.question, choices, video=video is not None)
        content: Any = prompt if video is None else _video_content(video, prompt)
        messages = [{"role": "user", "content": content}]
        result, attempts, latency = self._chat(
            messages,
            generation_seed=generation_seed,
            valid_letters=tuple(choices),
            mm_processor_kwargs=mm_processor_kwargs,
            media_io_kwargs=media_io_kwargs,
        )
        displayed_prediction = parse_strict_json_mcq_answer(
            result.content,
            choices,
        )
        prediction = (
            permutation.remap_prediction(displayed_prediction)
            if permutation is not None
            else displayed_prediction
        )
        usage = dict(result.usage)
        prompt_token_accounting = usage.get("qwen_prompt_token_accounting")
        sampled_frames_actual = (
            int(sampled_frames)
            if video is not None
            and sampled_frames is not None
            and isinstance(prompt_token_accounting, Mapping)
            and prompt_token_accounting.get("media_kind") == "video"
            and isinstance(
                prompt_token_accounting.get("service_multimodal_tokens"),
                Mapping,
            )
            else None
        )
        final_prompt_tokens = _token_int(usage.get("prompt_tokens")) or 0
        final_completion_tokens = _token_int(usage.get("completion_tokens")) or 0
        final_total_tokens = (
            _token_int(usage.get("total_tokens"))
            or final_prompt_tokens + final_completion_tokens
        )
        executed_prompt_tokens = sum(
            _token_int(attempt["usage"].get("prompt_tokens")) or 0
            for attempt in attempts
        )
        executed_completion_tokens = sum(
            _token_int(attempt["usage"].get("completion_tokens")) or 0
            for attempt in attempts
        )
        executed_total_tokens = sum(
            _token_int(attempt["usage"].get("total_tokens"))
            or (_token_int(attempt["usage"].get("prompt_tokens")) or 0)
            + (_token_int(attempt["usage"].get("completion_tokens")) or 0)
            for attempt in attempts
        )
        reasoning_values = [
            _reasoning_tokens(attempt["usage"])
            for attempt in attempts
        ]
        executed_reasoning_tokens = (
            sum(value for value in reasoning_values if value is not None)
            if all(value is not None for value in reasoning_values)
            else None
        )
        visual_breakdowns = [
            _visual_token_breakdown(attempt["usage"])
            for attempt in attempts
        ]
        visual_usage_complete = all(value is not None for value in visual_breakdowns)
        executed_visual_breakdown: dict[str, int] = {}
        for breakdown in visual_breakdowns:
            for kind, value in (breakdown or {}).items():
                executed_visual_breakdown[kind] = (
                    executed_visual_breakdown.get(kind, 0) + value
                )
        executed_visual_tokens = (
            sum(executed_visual_breakdown.values())
            if visual_usage_complete
            else None
        )
        if video is None:
            executed_visual_tokens = 0
            executed_visual_breakdown = {}
            visual_usage_complete = True
        parse_error = (
            None if prediction is not None else "strict_json_answer_missing"
        )
        return {
            "prediction": prediction,
            "displayed_prediction": displayed_prediction,
            "content": result.content,
            "raw_response": result.content,
            "reasoning_content": result.reasoning_content,
            "finish_reason": result.finish_reason,
            "usage": usage,
            "executed_usage": {
                "prompt_tokens": executed_prompt_tokens,
                "completion_tokens": executed_completion_tokens,
                "reasoning_tokens": executed_reasoning_tokens,
                "visual_tokens": executed_visual_tokens,
                "visual_token_breakdown": executed_visual_breakdown,
                "visual_usage_complete": visual_usage_complete,
                "total_tokens": executed_total_tokens,
            },
            "prompt_tokens": executed_prompt_tokens,
            "completion_tokens": executed_completion_tokens,
            "reasoning_tokens": executed_reasoning_tokens,
            "visual_tokens": executed_visual_tokens,
            "visual_token_breakdown": executed_visual_breakdown,
            "visual_usage_complete": visual_usage_complete,
            "total_tokens": executed_total_tokens,
            "final_prompt_tokens": final_prompt_tokens,
            "final_completion_tokens": final_completion_tokens,
            "final_reasoning_tokens": _reasoning_tokens(usage),
            "final_visual_tokens": _visual_tokens(usage) if video is not None else 0,
            "final_total_tokens": final_total_tokens,
            "latency_s": latency,
            "request_attempts": attempts,
            "request_messages": messages,
            "length_retry_used": len(attempts) > 1,
            "length_retry_blocked_by_headroom": (
                bool(attempts[-1].get("length_retry_blocked_reason"))
                and self.config.allow_length_retry
            ),
            "generation_seed": generation_seed,
            "model": self.model,
            "run_context": dict(self.config.run_context or {}),
            "media_items": 1 if video is not None else 0,
            "input_video": str(video) if video is not None else None,
            "video_metadata": metadata,
            "sampling_id": (
                self.config.direct_sampling.sampling_id
                if self.config.direct_sampling is not None
                else None
            ),
            "baseline_mode": self.config.mode,
            "sampling_kwargs": {
                "mm_processor_kwargs": mm_processor_kwargs,
                "media_io_kwargs": media_io_kwargs,
            }
            if video is not None
            else None,
            "mm_processor_kwargs": mm_processor_kwargs,
            "media_io_kwargs": media_io_kwargs,
            "sampled_frames_estimated": sampled_frames,
            "sampled_frames_actual": sampled_frames_actual,
            "sampled_frames_source": (
                "vllm_explicit_num_frames_and_successful_video_accounting_v1"
                if sampled_frames_actual is not None
                else "estimated_from_request"
                if sampled_frames is not None
                else None
            ),
            "option_permutation_seed": self.config.option_permutation_seed,
            "displayed_to_original": (
                permutation.displayed_to_original if permutation is not None else None
            ),
            "protocol_id": self.config.protocol.protocol_id,
            "enable_thinking": self.config.protocol.enable_thinking,
            "protocol_request": {
                **self.config.protocol.request_kwargs(),
                "response_format": mcq_answer_response_format(tuple(choices)),
                "extra_body": {"return_token_ids": True},
            },
            "prompt_id": BASELINE_PROMPT_ID,
            "prompt_template_sha256": BASELINE_PROMPT_TEMPLATE_SHA256,
            "input_prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
            "run_fingerprint": self.run_fingerprint(),
            "parse_error": parse_error,
            "model_parse_failure": parse_error is not None,
            "failure_class": "model_parse_failure" if parse_error else None,
            "annotation_leak_check": "passed",
            "annotation_leak_reason": "runtime_request_guard",
            "annotation_leak_guard": "structured_keys_and_sentinels_v1",
        }


def evaluate_qwen_runner(
    samples: Sequence[Sample],
    runner: ModelSampleRunner,
    method_id: str,
    output_dir: Path,
    *,
    concurrency: int = 1,
    resume: bool = False,
    retry_errors: bool = False,
    result_adapter: Callable[[Any], Mapping[str, Any]] | None = None,
    defer_scoring: bool = False,
) -> dict[str, Any]:
    if not samples:
        raise ValueError("at least one sample is required")
    sample_by_id: dict[str, Sample] = {}
    for sample in samples:
        if sample.sample_id in sample_by_id:
            raise ValueError(f"duplicate input sample_id: {sample.sample_id}")
        sample_by_id[sample.sample_id] = sample
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{samples[0].dataset}_{method_id}.jsonl"
    runner_fingerprint = runner.run_fingerprint()
    fingerprint = runner_fingerprint
    if defer_scoring:
        # A trajectory file is a model-facing training artifact.  Give the
        # no-label output policy its own resume identity so that a scored run
        # can never be resumed into the same JSONL by accident.
        fingerprint = hashlib.sha256(
            json.dumps(
                {
                    "runner_fingerprint": runner_fingerprint,
                    "scoring_policy": "deferred_v1",
                },
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
    cached: dict[str, dict[str, Any]] = {}
    if resume and output_path.is_file():
        existing_lines = output_path.read_text(encoding="utf-8").splitlines()
        for line_number, line in enumerate(existing_lines, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                if line_number == len(existing_lines):
                    # An interrupted append can leave only the final JSON
                    # object incomplete. Its sample remains pending.
                    continue
                raise RuntimeError(
                    f"malformed non-final resume row {line_number} in {output_path}"
                )
            if row.get("run_fingerprint") != fingerprint:
                raise RuntimeError(f"resume fingerprint mismatch in {output_path}")
            if defer_scoring:
                if row.get("scoring_deferred") is not True:
                    raise RuntimeError(
                        f"deferred resume row is not marked unscored in {output_path}"
                    )
                assert_deferred_result_public(row)
            sample_id = str(row["sample_id"])
            sample = sample_by_id.get(sample_id)
            if sample is None:
                raise RuntimeError(
                    f"resume file contains sample outside current manifest: {sample_id}"
                )
            expected_sample_fingerprint = _sample_fingerprint(sample)
            stored_sample_fingerprint = row.get("sample_fingerprint")
            if (
                stored_sample_fingerprint is not None
                and stored_sample_fingerprint != expected_sample_fingerprint
            ):
                raise RuntimeError(
                    f"resume sample fingerprint mismatch for {sample_id}"
                )
            # A process can be interrupted after appending a replacement but
            # before the final canonical rewrite. Last-write-wins recovers that
            # state; the file is canonicalized below before any new work starts.
            cached[sample_id] = row

    def retryable(row: Mapping[str, Any]) -> bool:
        if row.get("parse_error"):
            return True
        request_trace = row.get("request_trace")
        if isinstance(request_trace, list) and any(
            isinstance(item, Mapping) and item.get("finish_reason") == "length"
            for item in request_trace
        ):
            return True
        if row.get("branch_failures"):
            return True
        if row.get("strategy"):
            visual_tokens = row.get("visual_tokens")
            if (
                row.get("visual_token_accounting_complete") is not True
                or not isinstance(visual_tokens, (int, float))
                or isinstance(visual_tokens, bool)
                or not math.isfinite(float(visual_tokens))
                or float(visual_tokens) < 0
            ):
                return True
        return bool(
            row.get("error")
            and not row.get("data_unavailable")
            and not row.get("control_unavailable")
        )

    retry_ids = {
        sample_id
        for sample_id, row in cached.items()
        if retry_errors and retryable(row)
    }
    if resume and output_path.is_file():
        # Remove retried rows before issuing requests. If the process dies at
        # any later point, the next --resume sees a missing row, never a stale
        # row plus a duplicate replacement.
        cached = {
            sample_id: row
            for sample_id, row in cached.items()
            if sample_id not in retry_ids
        }
        retained = [
            cached[sample.sample_id]
            for sample in samples
            if sample.sample_id in cached
        ]
        temporary = output_path.with_suffix(output_path.suffix + ".partial")
        temporary.write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in retained),
            encoding="utf-8",
        )
        temporary.replace(output_path)

    pending = [sample for sample in samples if sample.sample_id not in cached]

    def run_one(sample: Sample) -> dict[str, Any]:
        started = time.perf_counter()
        emit_progress(
            "evaluation_sample_started",
            dataset=sample.dataset,
            sample_id=sample.sample_id,
            method_id=method_id,
        )
        try:
            raw = runner.run(ModelSample.from_sample(sample, None))
            result = dict(result_adapter(raw) if result_adapter else raw)
            if defer_scoring:
                assert_deferred_result_public(result)
        except Exception as exc:
            annotation_leak = isinstance(exc, AnnotationLeakError)
            data_unavailable = isinstance(exc, DataUnavailableError)
            control_unavailable = isinstance(exc, ControlUnavailableError)
            failure_class = (
                "annotation_leak"
                if annotation_leak
                else "data_unavailable"
                if data_unavailable
                else "control_unavailable"
                if control_unavailable
                else "infrastructure_error"
            )
            result = {
                "prediction": None,
                "error": f"{type(exc).__name__}: {exc}",
                "error_type": type(exc).__name__,
                "data_unavailable": data_unavailable,
                "control_unavailable": control_unavailable,
                "model_parse_failure": False,
                "failure_class": failure_class,
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "reasoning_tokens": None,
                "visual_tokens": None,
                "total_tokens": 0,
                "latency_s": time.perf_counter() - started,
                "annotation_leak_check": "failed" if annotation_leak else "not_run",
                "run_fingerprint": fingerprint,
            }
        public_fields = {
            "dataset": sample.dataset,
            "sample_id": sample.sample_id,
            "video": sample.video,
            "method_id": method_id,
            "run_fingerprint": fingerprint,
            "runner_fingerprint": runner_fingerprint,
            "elapsed_s": time.perf_counter() - started,
            "candidate_rerun": 0,
            "sample_fingerprint": _sample_fingerprint(sample),
            "scoring_deferred": defer_scoring,
        }
        if not defer_scoring:
            scoring = ScoringRecord.from_sample(sample)
            public_fields.update(
                {
                    "question": sample.question,
                    "choices": dict(sample.choices),
                    "choice_text_lengths": {
                        letter: len(text) for letter, text in sample.choices.items()
                    },
                    "answer": scoring.answer,
                    "correct": result.get("prediction") == scoring.answer,
                }
            )
        result.update(public_fields)
        if defer_scoring:
            assert_deferred_result_public(result)
        emit_progress(
            "evaluation_sample_complete",
            dataset=sample.dataset,
            sample_id=sample.sample_id,
            method_id=method_id,
        )
        return result

    mode = "a" if resume else "w"
    with output_path.open(mode, encoding="utf-8") as output:
        with ThreadPoolExecutor(max_workers=max(1, concurrency)) as pool:
            futures = [pool.submit(run_one, sample) for sample in pending]
            for future in as_completed(futures):
                row = future.result()
                cached[row["sample_id"]] = row
                output.write(json.dumps(row, ensure_ascii=False) + "\n")
                output.flush()
                os.fsync(output.fileno())
                emit_progress(
                    "result_committed",
                    dataset=row["dataset"],
                    sample_id=row["sample_id"],
                    method_id=method_id,
                )
    ordered = [cached[sample.sample_id] for sample in samples if sample.sample_id in cached]
    temporary = output_path.with_suffix(output_path.suffix + ".partial")
    temporary.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in ordered),
        encoding="utf-8",
    )
    temporary.replace(output_path)
    correct = (
        None
        if defer_scoring
        else sum(bool(row.get("correct")) for row in ordered)
    )
    data_unavailable = sum(bool(row.get("data_unavailable")) for row in ordered)
    control_unavailable = sum(bool(row.get("control_unavailable")) for row in ordered)
    model_parse_failures = sum(
        bool(row.get("model_parse_failure") or row.get("parse_error"))
        for row in ordered
    )
    annotation_leak_failures = sum(
        row.get("failure_class") == "annotation_leak" for row in ordered
    )
    agent_policy_failures = sum(
        row.get("failure_class") == "agent_policy_failure" for row in ordered
    )
    infrastructure_errors = sum(
        bool(
            row.get("error")
            and not row.get("data_unavailable")
            and not row.get("control_unavailable")
            and row.get("failure_class")
            not in {
                "annotation_leak",
                "model_parse_failure",
                "agent_policy_failure",
            }
        )
        for row in ordered
    )
    errors = sum(
        bool(row.get("error") or row.get("parse_error")) for row in ordered
    )
    accessible_rows = [
        row
        for row in ordered
        if not row.get("data_unavailable") and not row.get("control_unavailable")
    ]
    common_valid_rows = [
        row
        for row in accessible_rows
        if not row.get("error")
        and not row.get("model_parse_failure")
        and not row.get("parse_error")
    ]
    accessible_correct = (
        None
        if defer_scoring
        else sum(bool(row.get("correct")) for row in accessible_rows)
    )
    common_valid_correct = (
        None
        if defer_scoring
        else sum(bool(row.get("correct")) for row in common_valid_rows)
    )
    nominal_total = len(samples)
    failure_class_counts = {
        "data_unavailable": data_unavailable,
        "control_unavailable": control_unavailable,
        "model_parse_failure": model_parse_failures,
        "agent_policy_failure": agent_policy_failures,
        "annotation_leak": annotation_leak_failures,
        "infrastructure_error": infrastructure_errors,
    }
    summary = {
        "dataset": samples[0].dataset,
        "method_id": method_id,
        "total": nominal_total,
        "completed": len(ordered),
        "missing": nominal_total - len(ordered),
        "correct": correct,
        "accuracy": (
            None if defer_scoring or not nominal_total else correct / nominal_total
        ),
        "accuracy_nominal": (
            None if defer_scoring or not nominal_total else correct / nominal_total
        ),
        "scoring_deferred": defer_scoring,
        "nominal": {
            "denominator": nominal_total,
            "correct": correct,
            "accuracy": (
                None if defer_scoring or not nominal_total else correct / nominal_total
            ),
        },
        "accessible": {
            "denominator": len(accessible_rows),
            "correct": accessible_correct,
            "accuracy": (
                accessible_correct / len(accessible_rows)
                if accessible_rows and not defer_scoring
                else None
            ),
        },
        "common_valid": {
            "denominator": len(common_valid_rows),
            "correct": common_valid_correct,
            "accuracy": (
                common_valid_correct / len(common_valid_rows)
                if common_valid_rows and not defer_scoring
                else None
            ),
        },
        "errors": errors,
        "failure_class_counts": failure_class_counts,
        "output": str(output_path),
        "run_fingerprint": fingerprint,
    }
    (output_dir / f"{samples[0].dataset}_{method_id}_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary


def direct_sampling_spec(sampling_id: str) -> DirectSamplingSpec:
    try:
        return DIRECT_SAMPLING_SPECS[sampling_id]
    except KeyError as exc:
        raise ValueError(f"unknown Direct sampling id: {sampling_id}") from exc
