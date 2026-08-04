"""Candidate-blind multimodal confirmation for deferred Fast Hybrid traces.

This module is intentionally separate from both the Fast Hybrid runner and the
offline label join.  It consumes an immutable run spec plus an unscored trace,
reuses only the frames already selected by that trace, and asks independent
Qwen Judges to answer from those frames.  It never plans, extracts frames, or
loads benchmark labels.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Protocol, Sequence

from .client import ChatResult
from .privacy import (
    AnnotationLeakError,
    assert_annotation_free_request,
    assert_deferred_result_public,
)
from .qwen_protocol import mcq_answer_response_format, parse_strict_json_mcq_answer


JUDGE_SEEDS = (17, 42, 73)
_OPTION_RE = re.compile(r"(?m)^([A-H]):\s+\S")
_EVA_PUBLIC_METADATA_RE = re.compile(
    r"\AVideo Length:\s*\d+(?:\.\d+)?\s*seconds\.\s*"
    r"Original video resolution:\s*\d+(?:\.\d+)?p\.\s*\Z"
)
_PROVENANCE_FIELDS = (
    "schema_version",
    "phase",
    "dataset",
    "sample_id",
    "schedule_id",
    "variant_id",
    "family_id",
    "replica_id",
    "trajectory_id",
    "planner_seed",
    "max_total_visual_tokens",
    "max_call_visual_tokens",
    "max_turns",
    "required_judge_seeds",
    "manifest_sha256",
    "train600_manifest_sha256",
    "dataset_manifest_sha256",
    "config_sha256",
    "controller_fingerprint",
    "run_spec_fingerprint",
)
_GENERATED_JUDGE_FIELDS = {
    "judge_confirmations",
    "judge_config_sha256",
    "judge_status",
    "judge_prompt_tokens",
    "judge_completion_tokens",
    "judge_total_tokens",
    "judge_visual_tokens",
    "judge_visual_tokens_complete",
    "judge_latency_s",
}
_JUDGE_SYSTEM_PROMPT = (
    "You are an independent multiple-choice Judge. Use only the supplied "
    "timestamped video frames. Do not assume unseen events and do not use any "
    "earlier model answer. Return exactly one JSON object and nothing else: "
    '{"answer":"X"}.'
)


class JudgeChatClient(Protocol):
    def chat(
        self,
        model: str,
        messages: list[dict[str, Any]],
        max_tokens: int = 32,
        *,
        temperature: float = 0.0,
        seed: int | None = None,
        response_format: dict[str, Any] | None = None,
        chat_template_kwargs: dict[str, Any] | None = None,
    ) -> ChatResult: ...


@dataclass(frozen=True)
class TrajectoryJudgeConfig:
    model: str = "Qwen3.5-9B"
    judge_seeds: tuple[int, ...] = JUDGE_SEEDS
    max_tokens: int = 512
    temperature: float = 0.2
    enable_thinking: bool = False

    def __post_init__(self) -> None:
        if not self.model.strip():
            raise ValueError("judge model cannot be empty")
        if not self.judge_seeds or len(set(self.judge_seeds)) != len(
            self.judge_seeds
        ):
            raise ValueError("judge seeds must be unique and non-empty")
        if self.max_tokens <= 0:
            raise ValueError("judge max_tokens must be positive")
        if self.temperature < 0:
            raise ValueError("judge temperature cannot be negative")

    def fingerprint(self) -> str:
        return canonical_sha256(
            {
                "runner": "fast_hybrid_trajectory_judge_v2",
                "config": asdict(self),
                "system_prompt": _JUDGE_SYSTEM_PROMPT,
                "answer_protocol": "strict_json_single_answer_v1",
                "evidence_protocol": "reuse_selected_frames_without_replanning_v2",
                "question_recovery_protocol": "strict_eva_public_metadata_v1",
            }
        )


@dataclass(frozen=True)
class TrajectoryJudgeJob:
    spec: dict[str, Any]
    trajectory: dict[str, Any]

    @property
    def trajectory_id(self) -> str:
        return str(self.spec["trajectory_id"])


def canonical_sha256(payload: Any) -> str:
    serialized = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _as_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if not isinstance(content, Sequence) or isinstance(content, (bytes, bytearray)):
        return ""
    return "".join(
        str(part.get("text") or "")
        for part in content
        if isinstance(part, Mapping) and part.get("type") == "text"
    )


def _question_and_choices_prompt(trajectory: Mapping[str, Any]) -> str:
    """Recover only the public initial user prompt from the audit trace."""

    containers: list[Any] = []
    for request in trajectory.get("request_trace") or []:
        if isinstance(request, Mapping) and request.get("stage") == "verification":
            containers.append(request.get("messages"))
    for conversation in trajectory.get("conversation_traces") or []:
        if (
            isinstance(conversation, Mapping)
            and conversation.get("stage") == "verification"
        ):
            containers.append(conversation.get("messages"))
    containers.append(trajectory.get("messages"))

    candidates: list[str] = []
    for messages in containers:
        if not isinstance(messages, Sequence) or isinstance(
            messages, (str, bytes, bytearray)
        ):
            continue
        for message in messages:
            if not isinstance(message, Mapping) or message.get("role") != "user":
                continue
            text = _as_text(message.get("content")).strip()
            question_index = text.find("Question:")
            if question_index < 0:
                continue
            prefix = text[:question_index]
            # Official EVA prepends only public video duration/resolution metadata.
            # Accept that exact shape (or no prefix), but reject arbitrary text so a
            # candidate hint can never be stripped and silently sent to the Judge.
            if prefix and _EVA_PUBLIC_METADATA_RE.fullmatch(prefix) is None:
                continue
            public_prompt = text[question_index:].strip()
            if len(_OPTION_RE.findall(public_prompt)) >= 2:
                candidates.append(public_prompt)
    if not candidates:
        raise ValueError("deferred trajectory has no public question/choices prompt")
    # Later official-EVA turns repeat the same initial prompt.  Reject genuinely
    # different prompts instead of silently judging the wrong sample.
    unique = list(dict.fromkeys(candidates))
    if len(unique) != 1:
        raise ValueError("deferred trajectory contains conflicting question prompts")
    return unique[0]


def _option_letters(question_prompt: str) -> tuple[str, ...]:
    letters = tuple(dict.fromkeys(_OPTION_RE.findall(question_prompt)))
    if len(letters) < 2:
        raise ValueError("question prompt contains fewer than two options")
    return letters


def _frame_evidence(trajectory: Mapping[str, Any]) -> tuple[list[dict[str, Any]], int]:
    raw_calls = trajectory.get("tool_calls", trajectory.get("tool_steps"))
    if not isinstance(raw_calls, list) or not raw_calls:
        raise ValueError("deferred trajectory has no selected frames")
    evidence: list[dict[str, Any]] = []
    estimated_visual_tokens = 0
    for call_index, raw in enumerate(raw_calls):
        if not isinstance(raw, Mapping):
            raise ValueError(f"tool call {call_index} is not an object")
        paths = raw.get("frame_paths")
        timestamps = raw.get("actual_timestamps", raw.get("timestamps"))
        if not isinstance(paths, list) or not paths:
            raise ValueError(f"tool call {call_index} has no frame_paths")
        if not isinstance(timestamps, list) or len(timestamps) != len(paths):
            raise ValueError(f"tool call {call_index} frame/timestamp mismatch")
        frames: list[dict[str, Any]] = []
        for frame_index, (raw_path, raw_timestamp) in enumerate(
            zip(paths, timestamps, strict=True)
        ):
            path = Path(str(raw_path)).expanduser()
            if not path.is_absolute():
                raise ValueError(
                    f"tool call {call_index} frame {frame_index} is not absolute"
                )
            resolved = path.resolve()
            if not resolved.is_file():
                raise FileNotFoundError(resolved)
            try:
                timestamp = float(raw_timestamp)
            except (TypeError, ValueError) as error:
                raise ValueError(
                    f"tool call {call_index} has an invalid timestamp"
                ) from error
            frames.append({"frame_path": str(resolved), "timestamp": timestamp})
        estimated = raw.get("estimated_visual_tokens", raw.get("visual_tokens", 0))
        if isinstance(estimated, (int, float)) and not isinstance(estimated, bool):
            estimated_visual_tokens += max(0, int(estimated))
        evidence.append(
            {
                "tool_call_index": call_index,
                "stage": str(raw.get("stage") or "verification"),
                "start_time": float(raw.get("start_time", frames[0]["timestamp"])),
                "end_time": float(raw.get("end_time", frames[-1]["timestamp"])),
                "frames": frames,
            }
        )
    return evidence, estimated_visual_tokens


def _judge_messages(
    question_prompt: str, evidence: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    content: list[dict[str, Any]] = [
        {
            "type": "text",
            "text": (
                f"{question_prompt}\n\n"
                "Review all timestamped frames below and answer independently."
            ),
        }
    ]
    for block in evidence:
        content.append(
            {
                "type": "text",
                "text": (
                    f"Evidence block {int(block['tool_call_index']) + 1}, "
                    f"interval [{float(block['start_time']):.3f}, "
                    f"{float(block['end_time']):.3f}] seconds:"
                ),
            }
        )
        for frame in block["frames"]:
            content.append(
                {
                    "type": "text",
                    "text": f"Frame at {float(frame['timestamp']):.3f} seconds:",
                }
            )
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": Path(frame["frame_path"]).as_uri()},
                }
            )
    return [
        {"role": "system", "content": _JUDGE_SYSTEM_PROMPT},
        {"role": "user", "content": content},
    ]


def _usage_int(usage: Mapping[str, Any], key: str) -> int:
    value = usage.get(key)
    return int(value) if isinstance(value, int) and not isinstance(value, bool) else 0


def _api_visual_tokens(usage: Mapping[str, Any]) -> int | None:
    for container_name in ("prompt_tokens_details", "input_tokens_details"):
        details = usage.get(container_name)
        if not isinstance(details, Mapping):
            continue
        multimodal = details.get("multimodal_tokens")
        if isinstance(multimodal, Mapping):
            values = [
                int(value)
                for value in multimodal.values()
                if isinstance(value, int) and not isinstance(value, bool)
            ]
            if values:
                return sum(values)
        if isinstance(multimodal, int) and not isinstance(multimodal, bool):
            return multimodal
        values = [
            int(details[key])
            for key in ("visual_tokens", "video_tokens", "image_tokens")
            if isinstance(details.get(key), int)
            and not isinstance(details.get(key), bool)
        ]
        if values:
            return sum(values)
    value = usage.get("visual_tokens")
    return int(value) if isinstance(value, int) and not isinstance(value, bool) else None


def _identity_matches(spec: Mapping[str, Any], trajectory: Mapping[str, Any]) -> None:
    for field in ("dataset", "sample_id"):
        if str(trajectory.get(field) or "") != str(spec.get(field) or ""):
            raise RuntimeError(f"trajectory {field} does not match its run spec")
    aliases = (
        ("schedule_id", "trajectory_schedule_id"),
        ("variant_id", "trajectory_variant_id"),
        ("replica_id", "trajectory_replica_id"),
        ("planner_seed", "generation_seed"),
        ("manifest_sha256", "manifest_sha256"),
        ("train600_manifest_sha256", "train600_manifest_sha256"),
        ("config_sha256", "experiment_config_sha256"),
    )
    for spec_field, trajectory_field in aliases:
        actual = trajectory.get(trajectory_field)
        if actual is not None and str(actual) != str(spec.get(spec_field)):
            raise RuntimeError(
                f"trajectory {trajectory_field} does not match run-spec {spec_field}"
            )
    explicit_id = trajectory.get("trajectory_id")
    if explicit_id is not None and str(explicit_id) != str(spec.get("trajectory_id")):
        raise RuntimeError("trajectory_id does not match its run spec")


def bind_trajectory_jobs(
    specs: Iterable[Mapping[str, Any]],
    trajectories: Iterable[Mapping[str, Any]],
    *,
    schedule_id: str,
) -> tuple[TrajectoryJudgeJob, ...]:
    """Join one schedule without labels and reject ambiguous/mixed inputs."""

    selected_specs = [
        dict(spec) for spec in specs if str(spec.get("schedule_id")) == schedule_id
    ]
    if not selected_specs:
        raise ValueError(f"no run specs for schedule_id={schedule_id}")
    spec_ids = [str(spec.get("trajectory_id") or "") for spec in selected_specs]
    if any(not value for value in spec_ids) or len(set(spec_ids)) != len(spec_ids):
        raise ValueError("selected run specs have missing or duplicate trajectory_id")
    for spec in selected_specs:
        assert_deferred_result_public(spec)

    trajectory_rows = [dict(row) for row in trajectories]
    for row in trajectory_rows:
        assert_deferred_result_public(row)
    used: set[int] = set()
    jobs: list[TrajectoryJudgeJob] = []
    for spec in selected_specs:
        matches: list[tuple[int, dict[str, Any]]] = []
        for index, row in enumerate(trajectory_rows):
            if index in used:
                continue
            if str(row.get("dataset") or "") != str(spec["dataset"]):
                continue
            if str(row.get("sample_id") or "") != str(spec["sample_id"]):
                continue
            explicit_id = row.get("trajectory_id")
            if explicit_id is not None and str(explicit_id) != str(spec["trajectory_id"]):
                continue
            result_schedule = row.get("trajectory_schedule_id", row.get("schedule_id"))
            if result_schedule is not None and str(result_schedule) != schedule_id:
                continue
            result_seed = row.get("generation_seed", row.get("planner_seed"))
            if result_seed is not None and int(result_seed) != int(spec["planner_seed"]):
                continue
            matches.append((index, row))
        if len(matches) != 1:
            raise RuntimeError(
                f"expected one deferred result for {spec['trajectory_id']}, found {len(matches)}"
            )
        index, trajectory = matches[0]
        used.add(index)
        if trajectory.get("scoring_deferred") is not True:
            raise RuntimeError(
                f"trajectory is not marked scoring_deferred: {spec['trajectory_id']}"
            )
        if int(trajectory.get("candidate_rerun") or 0) != 0:
            raise RuntimeError("deferred trajectory reran its frozen candidate")
        if trajectory.get("annotation_leak_check") != "passed":
            raise RuntimeError("deferred trajectory failed annotation leak audit")
        _identity_matches(spec, trajectory)
        jobs.append(TrajectoryJudgeJob(spec=spec, trajectory=trajectory))
    return tuple(sorted(jobs, key=lambda item: item.trajectory_id))


class FastHybridTrajectoryJudge:
    """Run three independent candidate-blind Judges on frozen selected frames."""

    def __init__(
        self,
        client: JudgeChatClient,
        config: TrajectoryJudgeConfig | None = None,
    ) -> None:
        self.client = client
        self.config = config or TrajectoryJudgeConfig()

    def _base_row(self, job: TrajectoryJudgeJob) -> dict[str, Any]:
        row = {
            key: value
            for key, value in job.trajectory.items()
            if key not in _GENERATED_JUDGE_FIELDS
        }
        for field in _PROVENANCE_FIELDS:
            if field in job.spec:
                row[field] = job.spec[field]
        row.update(
            {
                "trajectory_id": job.trajectory_id,
                "scoring_deferred": True,
                "judge_config_sha256": self.config.fingerprint(),
                "judge_confirmations": [],
            }
        )
        return row

    def _resume_row(
        self, job: TrajectoryJudgeJob, existing: Mapping[str, Any] | None
    ) -> dict[str, Any]:
        if existing is None:
            return self._base_row(job)
        assert_deferred_result_public(existing)
        row = dict(existing)
        if str(row.get("trajectory_id") or "") != job.trajectory_id:
            raise RuntimeError("resume trajectory_id mismatch")
        if row.get("judge_config_sha256") != self.config.fingerprint():
            raise RuntimeError("resume judge configuration fingerprint mismatch")
        for field in _PROVENANCE_FIELDS:
            if field in job.spec and row.get(field) != job.spec[field]:
                raise RuntimeError(f"resume provenance mismatch: {field}")
        confirmations = row.get("judge_confirmations")
        if not isinstance(confirmations, list):
            raise RuntimeError("resume judge_confirmations is not a list")
        seeds: list[int] = []
        for confirmation in confirmations:
            if not isinstance(confirmation, Mapping):
                raise RuntimeError("resume confirmation is not an object")
            try:
                seeds.append(int(confirmation["judge_seed"]))
            except (KeyError, TypeError, ValueError) as error:
                raise RuntimeError("resume confirmation has no valid judge_seed") from error
        if len(set(seeds)) != len(seeds) or any(
            seed not in self.config.judge_seeds for seed in seeds
        ):
            raise RuntimeError("resume has duplicate or unexpected judge seeds")
        return row

    @staticmethod
    def _failure_confirmation(seed: int, error: Exception) -> dict[str, Any]:
        annotation_failure = isinstance(error, AnnotationLeakError)
        frame_failure = isinstance(error, (FileNotFoundError, ValueError))
        failure_class = (
            "annotation_leak"
            if annotation_failure
            else "frame_input_error"
            if frame_failure
            else "infrastructure_error"
        )
        return {
            "judge_seed": seed,
            "prediction": None,
            "final_prediction": None,
            "raw_response": "",
            "reasoning_content": "",
            "finish_reason": None,
            "usage": {},
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "visual_tokens": 0,
            "visual_token_source": "not_run",
            "latency_s": 0.0,
            "request_messages": [],
            "request_prompt_sha256": None,
            "evidence_sha256": None,
            "frame_count": 0,
            "parse_error": None,
            "api_error": None if frame_failure or annotation_failure else str(error),
            "frame_error": str(error) if frame_failure else None,
            "error": f"{type(error).__name__}: {error}",
            "error_type": type(error).__name__,
            "failure_class": failure_class,
            "fallback_used": False,
            "fallback_to_candidate": False,
            "candidate_blind": True,
            "annotation_leak_check": "failed" if annotation_failure else "not_run",
        }

    def _judge_one(
        self,
        *,
        seed: int,
        messages: list[dict[str, Any]],
        option_letters: tuple[str, ...],
        evidence_sha256: str,
        frame_count: int,
        estimated_visual_tokens: int,
    ) -> dict[str, Any]:
        response_format = mcq_answer_response_format(option_letters)
        request = {
            "messages": messages,
            "request_kwargs": {"response_format": response_format},
        }
        try:
            assert_annotation_free_request(request)
            result = self.client.chat(
                self.config.model,
                messages,
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
                seed=seed,
                response_format=response_format,
                chat_template_kwargs={
                    "enable_thinking": self.config.enable_thinking
                },
            )
        except Exception as error:
            failed = self._failure_confirmation(seed, error)
            failed["request_messages"] = messages
            failed["request_prompt_sha256"] = canonical_sha256(messages)
            failed["evidence_sha256"] = evidence_sha256
            failed["frame_count"] = frame_count
            return failed

        usage = dict(result.usage)
        prediction = parse_strict_json_mcq_answer(result.content, option_letters)
        actual_visual_tokens = _api_visual_tokens(usage)
        visual_tokens = (
            actual_visual_tokens
            if actual_visual_tokens is not None
            else estimated_visual_tokens
        )
        parse_error = None if prediction is not None else "strict_json_answer_missing"
        return {
            "judge_seed": seed,
            "prediction": prediction,
            "final_prediction": prediction,
            "raw_response": result.content,
            "reasoning_content": result.reasoning_content,
            "finish_reason": result.finish_reason,
            "usage": usage,
            "prompt_tokens": _usage_int(usage, "prompt_tokens"),
            "completion_tokens": _usage_int(usage, "completion_tokens"),
            "total_tokens": _usage_int(usage, "total_tokens"),
            "visual_tokens": visual_tokens,
            "visual_token_source": (
                "api_usage" if actual_visual_tokens is not None else "source_estimate"
            ),
            "latency_s": float(result.latency_s),
            "request_messages": messages,
            "request_prompt_sha256": canonical_sha256(messages),
            "evidence_sha256": evidence_sha256,
            "frame_count": frame_count,
            "parse_error": parse_error,
            "api_error": None,
            "frame_error": None,
            "error": None,
            "error_type": None,
            "failure_class": None if prediction is not None else "model_parse_failure",
            "fallback_used": False,
            "fallback_to_candidate": False,
            "candidate_blind": True,
            "annotation_leak_check": "passed",
        }

    @staticmethod
    def _finalize(row: dict[str, Any]) -> None:
        confirmations = list(row["judge_confirmations"])
        row["judge_confirmations"] = sorted(
            confirmations, key=lambda item: int(item["judge_seed"])
        )
        row["judge_prompt_tokens"] = sum(
            int(item.get("prompt_tokens") or 0) for item in confirmations
        )
        row["judge_completion_tokens"] = sum(
            int(item.get("completion_tokens") or 0) for item in confirmations
        )
        row["judge_total_tokens"] = sum(
            int(item.get("total_tokens") or 0) for item in confirmations
        )
        visual_complete = all(
            item.get("visual_token_source") == "api_usage" for item in confirmations
        )
        row["judge_visual_tokens"] = sum(
            int(item.get("visual_tokens") or 0) for item in confirmations
        )
        row["judge_visual_tokens_complete"] = visual_complete
        row["judge_latency_s"] = sum(
            float(item.get("latency_s") or 0.0) for item in confirmations
        )
        required = set(int(seed) for seed in row.get("required_judge_seeds") or [])
        observed = {int(item["judge_seed"]) for item in confirmations}
        row["judge_status"] = (
            "complete"
            if observed == required
            and not any(item.get("error") for item in confirmations)
            and not any(item.get("parse_error") for item in confirmations)
            else "complete_with_failures"
            if observed == required
            else "incomplete"
        )

    def judge(
        self,
        job: TrajectoryJudgeJob,
        *,
        existing: Mapping[str, Any] | None = None,
        on_update: Callable[[dict[str, Any]], None] | None = None,
    ) -> dict[str, Any]:
        row = self._resume_row(job, existing)
        required = tuple(int(seed) for seed in job.spec.get("required_judge_seeds") or ())
        if required != self.config.judge_seeds:
            raise RuntimeError("run-spec judge seeds do not match Judge configuration")
        completed = {
            int(item["judge_seed"]) for item in row.get("judge_confirmations") or []
        }
        try:
            question_prompt = _question_and_choices_prompt(job.trajectory)
            option_letters = _option_letters(question_prompt)
            evidence, estimated_visual_tokens = _frame_evidence(job.trajectory)
            messages = _judge_messages(question_prompt, evidence)
            evidence_sha256 = canonical_sha256(evidence)
            frame_count = sum(len(block["frames"]) for block in evidence)
            preparation_error: Exception | None = None
        except Exception as error:
            question_prompt = ""
            option_letters = ()
            evidence = []
            estimated_visual_tokens = 0
            messages = []
            evidence_sha256 = ""
            frame_count = 0
            preparation_error = error

        for seed in self.config.judge_seeds:
            if seed in completed:
                continue
            confirmation = (
                self._failure_confirmation(seed, preparation_error)
                if preparation_error is not None
                else self._judge_one(
                    seed=seed,
                    messages=messages,
                    option_letters=option_letters,
                    evidence_sha256=evidence_sha256,
                    frame_count=frame_count,
                    estimated_visual_tokens=estimated_visual_tokens,
                )
            )
            row["judge_confirmations"].append(confirmation)
            self._finalize(row)
            if on_update is not None:
                on_update(dict(row))
        self._finalize(row)
        assert_deferred_result_public(row)
        return row
