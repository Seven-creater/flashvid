from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Protocol

from .client import ChatResult
from .privacy import AnnotationLeakError, assert_annotation_free_request
from .qwen_agents import (
    AgentStrategy,
    AgentTrace,
    FrameRequest,
    InferenceProtocol,
    parse_answer_json,
)
from .qwen_agents.core import evidence_text, sample_question, safe_session_id
from .qwen_agents.strategies import EvidenceAgentBase
from .qwen_sft import materialize_trajectory_identity
from .schemas import ModelSample


class TrajectoryChatClient(Protocol):
    def chat(
        self,
        model: str,
        messages: list[dict[str, Any]],
        max_tokens: int = 32,
        *,
        temperature: float = 0.0,
        seed: int | None = None,
        chat_template_kwargs: dict[str, Any] | None = None,
        sampling_params: dict[str, Any] | None = None,
    ) -> ChatResult: ...


@dataclass(frozen=True)
class TrajectoryGenerationConfig:
    schedule_id: str
    dataset_manifest_sha256: str
    train600_manifest_sha256: str
    experiment_config_sha256: str
    agent_config_sha256: str
    model_artifact_sha256: str
    judge_seeds: tuple[int, ...] = (17, 42, 73)
    replica_id: int = 0

    def __post_init__(self) -> None:
        if not self.schedule_id.strip():
            raise ValueError("schedule_id cannot be empty")
        if len(set(self.judge_seeds)) != len(self.judge_seeds) or not self.judge_seeds:
            raise ValueError("judge_seeds must be unique and non-empty")
        for name in (
            "dataset_manifest_sha256",
            "train600_manifest_sha256",
            "experiment_config_sha256",
            "agent_config_sha256",
            "model_artifact_sha256",
        ):
            value = str(getattr(self, name)).lower()
            if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
                raise ValueError(f"{name} must be a lowercase SHA-256")


class QwenTrajectoryRunner:
    """Wrap a frozen Agent with annotation-blind, fixed-evidence Judge replicas."""

    def __init__(
        self,
        *,
        strategy: AgentStrategy,
        client: TrajectoryChatClient,
        model: str,
        protocol: InferenceProtocol,
        config: TrajectoryGenerationConfig,
    ) -> None:
        self.strategy = strategy
        self.client = client
        self.model = model
        self.protocol = protocol
        self.config = config

    def run_fingerprint(self) -> str:
        payload = {
            "runner": "qwen_trajectory_v1",
            "strategy_fingerprint": self.strategy.run_fingerprint(),
            "model": self.model,
            "protocol": asdict(self.protocol),
            "config": asdict(self.config),
            "implementation_files": {
                Path(__file__).name: _sha256_file(Path(__file__)),
                "qwen_sft.py": _sha256_file(Path(__file__).with_name("qwen_sft.py")),
            },
        }
        serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()

    def run(self, sample: ModelSample) -> Mapping[str, Any]:
        if sample.candidate_answer is not None:
            raise ValueError("Qwen-only trajectory generation cannot receive a candidate answer")
        trace = self.strategy.run(sample)
        if not isinstance(trace, AgentTrace):
            raise TypeError("trajectory strategy must return AgentTrace")
        row = trace.to_result_dict()
        row["base_agent_run_fingerprint"] = trace.run_fingerprint
        row["trajectory_runner_fingerprint"] = self.run_fingerprint()
        row["agent_config_sha256"] = self.config.agent_config_sha256
        row["model_artifact_sha256"] = self.config.model_artifact_sha256
        row = materialize_trajectory_identity(
            row,
            schedule_id=self.config.schedule_id,
            replica_id=self.config.replica_id,
            judge_seed="nested",
            manifest_sha256=self.config.train600_manifest_sha256,
            dataset_manifest_sha256=self.config.dataset_manifest_sha256,
            config_sha256=self.config.experiment_config_sha256,
        )
        if trace.annotation_leak_check != "passed":
            row["judge_confirmations"] = []
            row["confirmation_status"] = "skipped_annotation_leak"
            row["confirmation_failure_counts"] = {
                "annotation_leak": 0,
                "infrastructure_error": 0,
                "model_parse_failure": 0,
            }
            return row

        confirmations: list[dict[str, Any]] = []
        for judge_seed in self.config.judge_seeds:
            confirmation = self._confirm(sample, trace, judge_seed)
            confirmations.append(confirmation)
            if confirmation.get("failure_class") == "annotation_leak":
                break
        row["judge_confirmations"] = confirmations
        self._aggregate_confirmation_failures(row, confirmations)
        return row

    @staticmethod
    def _aggregate_confirmation_failures(
        row: dict[str, Any], confirmations: list[dict[str, Any]]
    ) -> None:
        counts = {
            "annotation_leak": 0,
            "infrastructure_error": 0,
            "model_parse_failure": 0,
        }
        for confirmation in confirmations:
            failure_class = confirmation.get("failure_class")
            if failure_class in counts:
                counts[str(failure_class)] += 1
        row["confirmation_failure_counts"] = counts
        if not any(counts.values()):
            row["confirmation_status"] = "passed_engineering"
            return

        row["confirmation_status"] = "failed"
        if counts["annotation_leak"]:
            row["annotation_leak_check"] = "failed"
            row["failure_class"] = "annotation_leak"
            row["error_type"] = "AnnotationLeakError"
            row["error"] = "trajectory confirmation request failed annotation leak guard"
        elif counts["infrastructure_error"]:
            row["failure_class"] = "infrastructure_error"
            row["error_type"] = "TrajectoryConfirmationError"
            row["error"] = "one or more trajectory confirmation API calls failed"
        else:
            row["failure_class"] = "model_parse_failure"
            row["model_parse_failure"] = True
            row["parse_error"] = "trajectory_confirmation_strict_json_answer_missing"

    def _confirm(
        self,
        sample: ModelSample,
        trace: AgentTrace,
        judge_seed: int,
    ) -> dict[str, Any]:
        messages = [
            {
                "role": "system",
                "content": (
                    "You are an independent multiple-choice Judge validating a fixed Qwen-only "
                    "trajectory. Use only the timestamped evidence; do not invent unseen events. "
                    'Return exactly one JSON object and nothing else: {"answer":"X"}.'
                ),
            },
            {
                "role": "user",
                "content": (
                    f"{sample_question(sample)}\nTimestamped evidence:\n"
                    f"{evidence_text(trace.evidence_memory)}"
                ),
            },
        ]
        requested = self.protocol.judge_max_tokens
        attempts: list[dict[str, Any]] = []
        try:
            assert_annotation_free_request(messages)
            while True:
                result = self.client.chat(
                    self.model,
                    messages,
                    max_tokens=requested,
                    temperature=self.protocol.temperature,
                    seed=judge_seed,
                    chat_template_kwargs={
                        "enable_thinking": self.protocol.enable_thinking
                    },
                    sampling_params=self.protocol.sampling_params(),
                )
                attempts.append(
                    {
                        "max_tokens": requested,
                        "finish_reason": result.finish_reason,
                        "usage": dict(result.usage),
                        "latency_s": result.latency_s,
                    }
                )
                retry = self.protocol.length_retry_max_tokens
                if result.finish_reason != "length" or retry is None or requested >= retry:
                    break
                prompt_tokens = _usage_int(result.usage, "prompt_tokens")
                headroom = (
                    self.protocol.server_max_model_len
                    - prompt_tokens
                    - self.protocol.context_safety_tokens
                )
                next_max = min(retry, headroom)
                if next_max <= requested:
                    break
                requested = next_max
            prediction = parse_answer_json(result.content, sample.option_letters)
            parse_error = None if prediction is not None else "strict_json_answer_missing"
            return {
                "judge_seed": judge_seed,
                "prediction": prediction,
                "final_prediction": prediction,
                "raw_response": result.content,
                "reasoning_content": result.reasoning_content,
                "finish_reason": result.finish_reason,
                "usage": dict(result.usage),
                "request_attempts": attempts,
                "parse_error": parse_error,
                "error": None,
                "error_type": None,
                "failure_class": (
                    None if prediction is not None else "model_parse_failure"
                ),
                "model_parse_failure": prediction is None,
                "fallback_used": False,
                "annotation_leak_check": "passed",
            }
        except AnnotationLeakError as error:
            return {
                "judge_seed": judge_seed,
                "prediction": None,
                "final_prediction": None,
                "raw_response": "",
                "reasoning_content": "",
                "finish_reason": None,
                "usage": {},
                "request_attempts": attempts,
                "parse_error": None,
                "error": f"{type(error).__name__}: {error}",
                "error_type": type(error).__name__,
                "failure_class": "annotation_leak",
                "model_parse_failure": False,
                "fallback_used": False,
                "annotation_leak_check": "failed",
            }
        except Exception as error:
            return {
                "judge_seed": judge_seed,
                "prediction": None,
                "final_prediction": None,
                "raw_response": "",
                "reasoning_content": "",
                "finish_reason": None,
                "usage": {},
                "request_attempts": attempts,
                "parse_error": None,
                "error": f"{type(error).__name__}: {error}",
                "error_type": type(error).__name__,
                "failure_class": "infrastructure_error",
                "model_parse_failure": False,
                "fallback_used": False,
                "annotation_leak_check": "not_run",
            }


class FixedEvidenceReplayStrategy(EvidenceAgentBase):
    """Replay a counterfactual with frozen intervals and altered frame counts.

    The strategy intentionally has no planner: only the pure frame tool sees
    the frozen calls, while Qwen independently observes the resulting frames
    and judges the answer.  This keeps a counterfactual change attributable to
    evidence cost rather than to a newly sampled localization plan.
    """

    strategy_id = "counterfactual_fixed_evidence"

    def __init__(
        self,
        *,
        planned_calls: tuple[FrameRequest, ...],
        source_fingerprint: str,
        **kwargs: Any,
    ) -> None:
        if not planned_calls:
            raise ValueError("counterfactual replay requires planned calls")
        if len(source_fingerprint) != 64 or any(
            character not in "0123456789abcdef"
            for character in source_fingerprint.lower()
        ):
            raise ValueError("source_fingerprint must be a SHA-256")
        self.planned_calls = planned_calls
        self.source_fingerprint = source_fingerprint.lower()
        super().__init__(**kwargs)

    def run_fingerprint(self) -> str:
        payload = {
            "runner": "qwen_counterfactual_fixed_evidence_v1",
            "base_agent_fingerprint": super().run_fingerprint(),
            "source_fingerprint": self.source_fingerprint,
            "planned_calls": [item.to_tool_arguments() for item in self.planned_calls],
            "implementation_sha256": _sha256_file(Path(__file__)),
        }
        serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()

    def _run(self, sample: ModelSample, trace: AgentTrace) -> None:
        if sample.candidate_answer is not None:
            raise ValueError("counterfactual replay cannot receive a candidate answer")
        video = self._resolve_video(sample)
        session = self.frame_tool.open_session(
            video,
            safe_session_id(sample, self.strategy_id),
        )
        for index, request in enumerate(self.planned_calls):
            self._observe(
                sample,
                trace,
                session,
                request,
                (
                    "Report only directly visible facts relevant to the question and choices. "
                    "Preserve temporal order and explicitly mark unresolved evidence."
                ),
                branch="counterfactual_evidence",
                source=f"counterfactual_step_{index}",
                seed_offset=index,
            )
        answer = self._judge(
            sample,
            trace,
            branch="counterfactual_judge",
            seed_offset=len(self.planned_calls),
        )
        if answer is None:
            trace.error = "counterfactual judge returned invalid answer JSON"
            trace.error_type = "AnswerParseError"
            return
        trace.prediction = trace.final_prediction = answer


def _usage_int(usage: Mapping[str, Any], key: str) -> int:
    value = usage.get(key)
    return int(value) if isinstance(value, int) and not isinstance(value, bool) else 0


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
