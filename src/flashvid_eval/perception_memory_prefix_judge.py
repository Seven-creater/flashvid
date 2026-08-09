"""Candidate-blind, text-only prefix verification for Perception-Memory EVA.

The online half of the process-SFT gate deliberately has no access to benchmark
labels or the frozen Direct candidate.  Every prefix is judged from exactly two
public objects: ``public_sample`` and ``memory_after``.  No frame is re-encoded,
and tools are explicitly disabled.  Ground-truth answers are joined later by
``perception_memory_selection``.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Protocol, Sequence

from . import eva_official as _eva_official_module
from . import perception_memory_eva as _perception_memory_module
from . import privacy as _privacy_module
from .client import ChatResult
from .perception_memory_eva import (
    EvidenceEvent,
    EvidenceMemory,
    OptionLedger,
    PERCEPTION_NORMALIZATION_VERSION,
    build_judge_messages,
    messages_have_media,
)
from .privacy import AnnotationLeakError, assert_annotation_free_request
from .qwen_agents import core as _qwen_core_module
from .schemas import ModelSample


PREFIX_JUDGE_SEEDS = (17, 42, 73)
_PUBLIC_SAMPLE_KEYS = frozenset(
    {"dataset", "sample_id", "video", "question", "choices"}
)
_MEMORY_KEYS = frozenset(
    {"event_ledger", "option_ledger", "unresolved", "observed_intervals"}
)
_EVENT_KEYS = frozenset(
    {"evidence_id", "id", "interval", "timestamp", "fact", "source"}
)
_JUDGE_RESPONSE_KEYS = frozenset({"answer", "evidence_ids"})
_PREFIX_JUDGE_SYSTEM_PROMPT = (
    "You are an independent, candidate-blind evidence-only Judge. You have no "
    "tools and cannot inspect any image or video. Use only the timestamped evidence "
    "ledger. Make the best evidence-only answer and cite only existing evidence IDs. "
    "Return exactly one JSON object with answer and evidence_ids. Three independent "
    "correct responses are the offline completeness criterion; do not claim access "
    "to ground truth."
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


@lru_cache(maxsize=1)
def implementation_dependency_hashes() -> dict[str, str]:
    """Hash every local implementation file that affects prefix Judge semantics."""

    paths = {
        "prefix_judge": Path(__file__).resolve(),
        "perception_memory_eva": Path(_perception_memory_module.__file__).resolve(),
        "privacy": Path(_privacy_module.__file__).resolve(),
        "qwen_agent_core": Path(_qwen_core_module.__file__).resolve(),
        "eva_official": Path(_eva_official_module.__file__).resolve(),
    }
    return {
        name: hashlib.sha256(path.read_bytes()).hexdigest()
        for name, path in sorted(paths.items())
    }


def _text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be non-empty text")
    return " ".join(value.strip().split())


def _number(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{field} must be finite")
    return result


def _string_array(value: Any, field: str) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise ValueError(f"{field} must be an array")
    return tuple(_text(item, field) for item in value)


def _public_sample(value: Any) -> ModelSample:
    if not isinstance(value, Mapping) or set(value) != _PUBLIC_SAMPLE_KEYS:
        raise ValueError("public_sample must contain exactly the public sample fields")
    choices = value["choices"]
    if not isinstance(choices, Mapping) or len(choices) < 2:
        raise ValueError("public_sample.choices must contain at least two options")
    normalized_choices: dict[str, str] = {}
    for raw_letter, raw_choice in choices.items():
        letter = str(raw_letter).strip().upper()
        if len(letter) != 1 or not "A" <= letter <= "H" or letter in normalized_choices:
            raise ValueError("public_sample choices must use unique A-H letters")
        normalized_choices[letter] = _text(raw_choice, f"choices.{letter}")
    public = {
        "dataset": _text(value["dataset"], "dataset"),
        "sample_id": _text(value["sample_id"], "sample_id"),
        "video": _text(value["video"], "video"),
        "question": _text(value["question"], "question"),
        "choices": normalized_choices,
    }
    assert_annotation_free_request(public)
    return ModelSample(candidate_answer=None, **public)


def _interval(value: Any, field: str) -> tuple[float, float]:
    if not isinstance(value, list) or len(value) != 2:
        raise ValueError(f"{field} must be an interval pair")
    start = _number(value[0], f"{field}[0]")
    end = _number(value[1], f"{field}[1]")
    if end <= start:
        raise ValueError(f"{field} must have end > start")
    return start, end


def _public_memory(value: Any, option_letters: Sequence[str]) -> EvidenceMemory:
    if not isinstance(value, Mapping) or set(value) != _MEMORY_KEYS:
        raise ValueError("memory_after must contain exactly the frozen ledger fields")
    assert_annotation_free_request(value)

    raw_events = value["event_ledger"]
    if not isinstance(raw_events, list) or not raw_events:
        raise ValueError("memory_after.event_ledger must be non-empty")
    events: list[EvidenceEvent] = []
    evidence_ids: set[str] = set()
    for index, raw in enumerate(raw_events):
        if not isinstance(raw, Mapping):
            raise ValueError(f"event_ledger[{index}] must be an object")
        keys = set(raw)
        if not keys <= _EVENT_KEYS or not {
            "interval",
            "timestamp",
            "fact",
            "source",
        } <= keys:
            raise ValueError(f"event_ledger[{index}] has an invalid schema")
        raw_id = raw.get("evidence_id", raw.get("id"))
        evidence_id = _text(raw_id, f"event_ledger[{index}].evidence_id")
        if evidence_id in evidence_ids:
            raise ValueError("memory_after contains duplicate evidence IDs")
        evidence_ids.add(evidence_id)
        interval = _interval(raw["interval"], f"event_ledger[{index}].interval")
        timestamp = raw["timestamp"]
        parsed_timestamp = (
            None
            if timestamp is None
            else _number(timestamp, f"event_ledger[{index}].timestamp")
        )
        events.append(
            EvidenceEvent(
                evidence_id=evidence_id,
                interval=interval,
                timestamp=parsed_timestamp,
                fact=_text(raw["fact"], f"event_ledger[{index}].fact"),
                source=_text(raw["source"], f"event_ledger[{index}].source"),
            )
        )

    raw_options = value["option_ledger"]
    if not isinstance(raw_options, Mapping) or set(raw_options) != set(option_letters):
        raise ValueError("memory_after.option_ledger must match the sample options")
    options: dict[str, OptionLedger] = {}
    for letter in option_letters:
        raw = raw_options[letter]
        if not isinstance(raw, Mapping) or set(raw) != {"supports", "contradicts"}:
            raise ValueError(f"option_ledger.{letter} has an invalid schema")
        supports = _string_array(raw["supports"], f"option_ledger.{letter}.supports")
        contradicts = _string_array(
            raw["contradicts"], f"option_ledger.{letter}.contradicts"
        )
        if any(item not in evidence_ids for item in (*supports, *contradicts)):
            raise ValueError(f"option_ledger.{letter} cites an unknown evidence ID")
        options[letter] = OptionLedger(
            tuple(dict.fromkeys(supports)), tuple(dict.fromkeys(contradicts))
        )

    observed = value["observed_intervals"]
    if not isinstance(observed, list):
        raise ValueError("memory_after.observed_intervals must be an array")
    observed_intervals = [
        _interval(item, f"observed_intervals[{index}]")
        for index, item in enumerate(observed)
    ]
    unresolved = list(_string_array(value["unresolved"], "unresolved"))
    return EvidenceMemory(
        option_letters=tuple(option_letters),
        event_ledger=events,
        option_ledger=options,
        unresolved=unresolved,
        observed_intervals=observed_intervals,
    )


@dataclass(frozen=True)
class PrefixJudgeDecision:
    answer: str
    evidence_ids: tuple[str, ...]


def parse_prefix_judge_response(
    text: str,
    valid_letters: Iterable[str],
    valid_evidence_ids: Iterable[str],
) -> PrefixJudgeDecision | None:
    """Parse one strict evidence-only answer with valid ledger citations."""

    letters = {str(item).strip().upper() for item in valid_letters}
    evidence = set(valid_evidence_ids)
    try:
        payload = json.loads((text or "").strip())
        if not isinstance(payload, Mapping) or set(payload) != _JUDGE_RESPONSE_KEYS:
            return None
        answer = str(payload["answer"]).strip().upper()
        if answer not in letters:
            return None
        evidence_ids = _string_array(payload["evidence_ids"], "evidence_ids")
        evidence_ids = tuple(dict.fromkeys(evidence_ids))
        if not evidence_ids or any(item not in evidence for item in evidence_ids):
            return None
        return PrefixJudgeDecision(
            answer=answer,
            evidence_ids=evidence_ids,
        )
    except (TypeError, ValueError, json.JSONDecodeError):
        return None


def prefix_response_format(option_letters: Sequence[str]) -> dict[str, Any]:
    letters = list(option_letters)
    if len(letters) < 2 or len(set(letters)) != len(letters):
        raise ValueError("response schema requires unique options")
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "perception_memory_prefix_judge",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    # This exact shape is recognized as a public MCQ schema by
                    # the central annotation-leak guard.
                    "answer": {"type": "string", "enum": letters},
                    "evidence_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                        "minItems": 1,
                        "uniqueItems": True,
                    },
                },
                "required": sorted(_JUDGE_RESPONSE_KEYS),
                "additionalProperties": False,
            },
        },
    }


def build_prefix_judge_messages(
    sample: ModelSample, memory: EvidenceMemory
) -> list[dict[str, Any]]:
    """Build one candidate-blind request from public sample + text memory only."""

    base = build_judge_messages(sample, memory)
    messages = [
        {
            "role": "system",
            "content": _PREFIX_JUDGE_SYSTEM_PROMPT,
        },
        {"role": "user", "content": base[1]["content"]},
    ]
    if messages_have_media(messages):
        raise AssertionError("prefix Judge must be text-only")
    return messages


@dataclass(frozen=True)
class PrefixJudgeConfig:
    model: str = "Qwen3.5-9B"
    judge_seeds: tuple[int, ...] = PREFIX_JUDGE_SEEDS
    max_tokens: int = 512
    temperature: float = 0.2
    enable_thinking: bool = False

    def __post_init__(self) -> None:
        if not self.model.strip():
            raise ValueError("model cannot be empty")
        if len(self.judge_seeds) != 3 or len(set(self.judge_seeds)) != 3:
            raise ValueError("prefix completeness requires exactly three unique seeds")
        if self.max_tokens <= 0:
            raise ValueError("max_tokens must be positive")
        if self.temperature < 0:
            raise ValueError("temperature cannot be negative")
        if self.enable_thinking:
            raise ValueError("prefix Judge must keep hidden thinking disabled")

    def fingerprint(self) -> str:
        return canonical_sha256(
            {
                "runner": "perception_memory_prefix_judge_v1",
                "config": asdict(self),
                "tool_choice": "none",
                "system_prompt": _PREFIX_JUDGE_SYSTEM_PROMPT,
                "response_schema": prefix_response_format(("A", "B")),
                "parser": "strict_complete_answer_citations_v1",
                "implementation_dependencies": implementation_dependency_hashes(),
            }
        )


@dataclass(frozen=True)
class PrefixJudgeJob:
    trajectory_id: str
    prefix_id: str
    prefix_index: int
    dataset: str
    sample_id: str
    sample: ModelSample
    memory: EvidenceMemory
    source_sha256: str


class PrefixJudgeClient(Protocol):
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
        extra_body: dict[str, Any] | None = None,
    ) -> ChatResult: ...


def bind_prefix_jobs(
    trajectories: Iterable[Mapping[str, Any]],
) -> tuple[PrefixJudgeJob, ...]:
    """Expand public deferred trajectories into immutable per-prefix jobs."""

    jobs: list[PrefixJudgeJob] = []
    seen_trajectories: set[str] = set()
    seen_prefixes: set[str] = set()
    for row_index, raw_row in enumerate(trajectories):
        if not isinstance(raw_row, Mapping):
            raise ValueError(f"trajectory row {row_index} must be an object")
        if raw_row.get("scoring_deferred") is not True:
            raise ValueError("prefix judging requires scoring_deferred trajectories")
        if raw_row.get("annotation_leak_check") != "passed":
            raise ValueError("trajectory failed annotation leak audit")
        if int(raw_row.get("candidate_rerun") or 0) != 0:
            raise ValueError("trajectory reran its frozen candidate")
        if (
            raw_row.get("perception_normalization_version")
            != PERCEPTION_NORMALIZATION_VERSION
        ):
            raise ValueError(
                "trajectory has incompatible perception normalization"
            )
        trajectory_id = _text(raw_row.get("trajectory_id"), "trajectory_id")
        if trajectory_id in seen_trajectories:
            raise ValueError("duplicate trajectory_id")
        seen_trajectories.add(trajectory_id)
        sample = _public_sample(raw_row.get("public_sample"))
        if str(raw_row.get("dataset")) != sample.dataset or str(
            raw_row.get("sample_id")
        ) != sample.sample_id:
            raise ValueError("trajectory identity differs from public_sample")
        states = raw_row.get("perception_states")
        if not isinstance(states, list) or not states:
            raise ValueError("trajectory has no perception prefixes")
        for expected_index, raw_state in enumerate(states):
            if not isinstance(raw_state, Mapping):
                raise ValueError("perception state must be an object")
            prefix_index = int(raw_state.get("step_index", expected_index))
            if prefix_index != expected_index:
                raise ValueError("prefix indices must be contiguous and ordered")
            memory = _public_memory(raw_state.get("memory_after"), sample.option_letters)
            public_sample = {
                "dataset": sample.dataset,
                "sample_id": sample.sample_id,
                "video": sample.video,
                "question": sample.question,
                "choices": dict(sample.choices),
            }
            memory_dict = memory.to_dict()
            source = {
                "trajectory_id": trajectory_id,
                "prefix_index": prefix_index,
                "public_sample": public_sample,
                "memory_after": memory_dict,
                "config_sha256": raw_row.get("config_sha256"),
                "run_fingerprint": raw_row.get("run_fingerprint"),
                "perception_normalization_version": (
                    PERCEPTION_NORMALIZATION_VERSION
                ),
            }
            prefix_id = f"{trajectory_id}#prefix-{prefix_index:03d}"
            if prefix_id in seen_prefixes:
                raise ValueError("duplicate prefix_id")
            seen_prefixes.add(prefix_id)
            jobs.append(
                PrefixJudgeJob(
                    trajectory_id=trajectory_id,
                    prefix_id=prefix_id,
                    prefix_index=prefix_index,
                    dataset=sample.dataset,
                    sample_id=sample.sample_id,
                    sample=sample,
                    memory=memory,
                    source_sha256=canonical_sha256(source),
                )
            )
    return tuple(sorted(jobs, key=lambda item: item.prefix_id))


def _usage_int(usage: Mapping[str, Any], key: str) -> int:
    value = usage.get(key)
    return int(value) if isinstance(value, int) and not isinstance(value, bool) else 0


class PerceptionMemoryPrefixJudge:
    """Run three independent text-only Qwen Judges for one evidence prefix."""

    def __init__(
        self,
        client: PrefixJudgeClient,
        config: PrefixJudgeConfig | None = None,
    ) -> None:
        self.client = client
        self.config = config or PrefixJudgeConfig()

    def _base_row(self, job: PrefixJudgeJob) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "dataset": job.dataset,
            "sample_id": job.sample_id,
            "trajectory_id": job.trajectory_id,
            "prefix_id": job.prefix_id,
            "prefix_index": job.prefix_index,
            "source_sha256": job.source_sha256,
            "prefix_judge_config_sha256": self.config.fingerprint(),
            "scoring_deferred": True,
            "candidate_blind": True,
            "tools_disabled": True,
            "media_count": 0,
            "annotation_leak_check": "passed",
            "judge_confirmations": [],
            "judge_status": "incomplete",
        }

    def _resume_row(
        self,
        job: PrefixJudgeJob,
        existing: Mapping[str, Any] | None,
        *,
        retry_errors: bool = False,
    ) -> dict[str, Any]:
        if existing is None:
            return self._base_row(job)
        row = dict(existing)
        expected = self._base_row(job)
        for field in (
            "dataset",
            "sample_id",
            "trajectory_id",
            "prefix_id",
            "prefix_index",
            "source_sha256",
            "prefix_judge_config_sha256",
        ):
            if row.get(field) != expected[field]:
                raise RuntimeError(f"resume prefix provenance mismatch: {field}")
        confirmations = row.get("judge_confirmations")
        if not isinstance(confirmations, list):
            raise RuntimeError("resume judge_confirmations must be an array")
        if any(not isinstance(item, Mapping) for item in confirmations):
            raise RuntimeError("resume judge_confirmations must contain objects")
        seeds = [int(item["judge_seed"]) for item in confirmations]
        if len(seeds) != len(set(seeds)) or any(
            seed not in self.config.judge_seeds for seed in seeds
        ):
            raise RuntimeError("resume contains duplicate or unexpected Judge seeds")
        if retry_errors:
            row["judge_confirmations"] = [
                dict(item)
                for item in confirmations
                if item.get("error") is None and item.get("parsed_valid") is True
            ]
        self._finalize(row)
        return row

    def prepare_resume(
        self,
        job: PrefixJudgeJob,
        existing: Mapping[str, Any],
        *,
        retry_errors: bool = False,
    ) -> dict[str, Any]:
        """Validate/sanitize one persisted row without issuing model requests."""

        return self._resume_row(job, existing, retry_errors=retry_errors)

    @staticmethod
    def _failure(
        seed: int,
        error: Exception,
        messages: list[dict[str, Any]],
        request_kwargs: dict[str, Any],
    ) -> dict[str, Any]:
        return {
            "judge_seed": seed,
            "prediction": None,
            "evidence_ids": [],
            "parsed_valid": False,
            "raw_response": "",
            "reasoning_content": "",
            "finish_reason": None,
            "usage": {},
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "latency_s": 0.0,
            "request_messages": messages,
            "request_kwargs": request_kwargs,
            "request_prompt_sha256": canonical_sha256(
                {"messages": messages, "request_kwargs": request_kwargs}
            ),
            "error": f"{type(error).__name__}: {error}",
            "error_type": type(error).__name__,
            "failure_class": (
                "annotation_leak"
                if isinstance(error, AnnotationLeakError)
                else "infrastructure_error"
            ),
            "candidate_blind": True,
            "tools_disabled": True,
            "media_count": 0,
            "annotation_leak_check": (
                "failed" if isinstance(error, AnnotationLeakError) else "passed"
            ),
        }

    def _one(
        self,
        job: PrefixJudgeJob,
        seed: int,
        messages: list[dict[str, Any]],
    ) -> dict[str, Any]:
        response_format = prefix_response_format(job.sample.option_letters)
        request_kwargs = {
            "response_format": response_format,
            "chat_template_kwargs": {"enable_thinking": False},
            "tool_choice": "none",
        }
        try:
            if messages_have_media(messages):
                raise ValueError("prefix Judge request contains media")
            assert_annotation_free_request(
                {"messages": messages, "request_kwargs": request_kwargs}
            )
            result = self.client.chat(
                self.config.model,
                messages,
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
                seed=seed,
                response_format=response_format,
                chat_template_kwargs={"enable_thinking": False},
                extra_body={"tool_choice": "none"},
            )
        except Exception as error:
            return self._failure(seed, error, messages, request_kwargs)
        decision = parse_prefix_judge_response(
            result.content, job.sample.option_letters, job.memory.evidence_ids
        )
        usage = dict(result.usage)
        return {
            "judge_seed": seed,
            "prediction": decision.answer if decision else None,
            "evidence_ids": list(decision.evidence_ids) if decision else [],
            "parsed_valid": decision is not None,
            "raw_response": result.content,
            "reasoning_content": result.reasoning_content,
            "finish_reason": result.finish_reason,
            "usage": usage,
            "prompt_tokens": _usage_int(usage, "prompt_tokens"),
            "completion_tokens": _usage_int(usage, "completion_tokens"),
            "total_tokens": _usage_int(usage, "total_tokens"),
            "latency_s": float(result.latency_s),
            "request_messages": messages,
            "request_kwargs": request_kwargs,
            "request_prompt_sha256": canonical_sha256(
                {"messages": messages, "request_kwargs": request_kwargs}
            ),
            "error": None,
            "error_type": None,
            "failure_class": None if decision else "model_parse_failure",
            "candidate_blind": True,
            "tools_disabled": True,
            "media_count": 0,
            "annotation_leak_check": "passed",
        }

    def _finalize(self, row: dict[str, Any]) -> None:
        confirmations = sorted(
            row["judge_confirmations"], key=lambda item: int(item["judge_seed"])
        )
        row["judge_confirmations"] = confirmations
        row["judge_prompt_tokens"] = sum(
            int(item.get("prompt_tokens") or 0) for item in confirmations
        )
        row["judge_completion_tokens"] = sum(
            int(item.get("completion_tokens") or 0) for item in confirmations
        )
        row["judge_total_tokens"] = sum(
            int(item.get("total_tokens") or 0) for item in confirmations
        )
        row["judge_latency_s"] = sum(
            float(item.get("latency_s") or 0.0) for item in confirmations
        )
        observed = {int(item["judge_seed"]) for item in confirmations}
        required = set(self.config.judge_seeds)
        row["judge_status"] = (
            "complete"
            if observed == required
            and all(item.get("parsed_valid") is True for item in confirmations)
            and all(item.get("error") is None for item in confirmations)
            else "complete_with_failures"
            if observed == required
            else "incomplete"
        )

    def judge(
        self,
        job: PrefixJudgeJob,
        *,
        existing: Mapping[str, Any] | None = None,
        on_update: Callable[[dict[str, Any]], None] | None = None,
        retry_errors: bool = False,
    ) -> dict[str, Any]:
        row = self._resume_row(job, existing, retry_errors=retry_errors)
        messages = build_prefix_judge_messages(job.sample, job.memory)
        completed = {
            int(item["judge_seed"]) for item in row.get("judge_confirmations") or []
        }
        for seed in self.config.judge_seeds:
            if seed in completed:
                continue
            row["judge_confirmations"].append(self._one(job, seed, messages))
            self._finalize(row)
            if on_update is not None:
                on_update(dict(row))
        self._finalize(row)
        return row


__all__ = [
    "PREFIX_JUDGE_SEEDS",
    "PerceptionMemoryPrefixJudge",
    "PrefixJudgeConfig",
    "PrefixJudgeDecision",
    "PrefixJudgeJob",
    "bind_prefix_jobs",
    "build_prefix_judge_messages",
    "canonical_sha256",
    "parse_prefix_judge_response",
    "prefix_response_format",
]
