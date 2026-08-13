"""Visual-only counterfactual completeness checks for evidence prefixes.

The verifier receives the public question and the exact frames accumulated by a
prefix.  It never receives the Direct candidate, benchmark answer, text ledger,
or prior model reasoning.  Ground truth is joined only by the offline selector.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from copy import deepcopy
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Protocol, Sequence

from .client import ChatResult
from .privacy import AnnotationLeakError, assert_annotation_free_request
from .schemas import ModelSample


VISUAL_CSV_SCHEMA_VERSION = "visual_csv_v4"
VISUAL_CSV_SEEDS = (17, 42, 73)
_PUBLIC_SAMPLE_KEYS = frozenset(
    {"dataset", "sample_id", "video", "question", "choices"}
)
_DECISION_KEYS = frozenset(
    {"answer", "frame_indices", "evidence_complete", "missing_evidence"}
)
_PRIVATE_TRAJECTORY_KEYS = frozenset(
    {
        "answer",
        "correct_answer",
        "right_answer",
        "ground_truth",
        "gt",
        "candidate",
        "candidate_answer",
        "direct_candidate",
        "time_range",
        "clue_intervals",
        "question_type",
        "metadata",
        "annotation",
        "annotations",
    }
)
_CONFIRMATION_KEYS = frozenset(
    {
        "judge_seed",
        "prediction",
        "frame_indices",
        "evidence_complete",
        "missing_evidence",
        "parsed_valid",
        "raw_response",
        "finish_reason",
        "usage",
        "prompt_tokens",
        "completion_tokens",
        "total_tokens",
        "latency_s",
        "request_messages",
        "request_kwargs",
        "request_prompt_sha256",
        "error",
        "error_type",
        "failure_class",
        "candidate_blind",
        "tools_disabled",
        "media_count",
        "annotation_leak_check",
        "fallback_used",
    }
)
_RESULT_KEYS = frozenset(
    {
        "schema_version",
        "visual_csv_schema_version",
        "dataset",
        "sample_id",
        "trajectory_id",
        "prefix_id",
        "prefix_index",
        "source_sha256",
        "visual_csv_config",
        "visual_csv_config_sha256",
        "verifier_artifact_sha256",
        "visual_csv_judge_seeds",
        "candidate_blind",
        "tools_disabled",
        "media_count",
        "frame_paths_sha256",
        "frame_content_sha256s",
        "timestamps_sha256",
        "annotation_leak_check",
        "visual_csv_confirmations",
        "visual_csv_status",
        "visual_csv_prompt_tokens",
        "visual_csv_completion_tokens",
        "visual_csv_total_tokens",
        "visual_csv_latency_s",
    }
)


def _text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be non-empty text")
    return " ".join(value.strip().split())


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def frame_content_sha256s(frame_paths: Sequence[str]) -> tuple[str, ...]:
    """Hash the bytes behind an ordered local-frame inventory."""

    values: list[str] = []
    for raw_path in frame_paths:
        path = Path(str(raw_path)).resolve()
        if not path.is_absolute() or not path.is_file():
            raise ValueError(f"visual CSV frame is missing: {path}")
        values.append(_file_sha256(path))
    if not values:
        raise ValueError("visual CSV requires at least one frame")
    return tuple(values)


def _strip_private_trajectory_fields(row: dict[str, Any]) -> None:
    for key in tuple(row):
        if str(key).strip().casefold() in _PRIVATE_TRAJECTORY_KEYS:
            row.pop(key, None)


def _public_sample(value: Any) -> ModelSample:
    if not isinstance(value, Mapping) or set(value) != _PUBLIC_SAMPLE_KEYS:
        raise ValueError("public_sample must contain exactly the public fields")
    choices = value["choices"]
    if not isinstance(choices, Mapping) or len(choices) < 2:
        raise ValueError("public_sample.choices must contain at least two options")
    normalized: dict[str, str] = {}
    for raw_letter, raw_choice in choices.items():
        letter = str(raw_letter).strip().upper()
        if len(letter) != 1 or not "A" <= letter <= "H" or letter in normalized:
            raise ValueError("public choices must use unique A-H letters")
        normalized[letter] = _text(raw_choice, f"choices.{letter}")
    sample = ModelSample(
        dataset=_text(value["dataset"], "dataset"),
        sample_id=_text(value["sample_id"], "sample_id"),
        video=_text(value["video"], "video"),
        question=_text(value["question"], "question"),
        choices=normalized,
        candidate_answer=None,
    )
    assert_annotation_free_request(
        {
            "dataset": sample.dataset,
            "sample_id": sample.sample_id,
            "video": sample.video,
            "question": sample.question,
            "choices": dict(sample.choices),
        }
    )
    return sample


@dataclass(frozen=True)
class VisualCsvDecision:
    answer: str
    frame_indices: tuple[int, ...]
    evidence_complete: bool
    missing_evidence: tuple[str, ...]


@dataclass(frozen=True)
class VisualCsvConfig:
    verifier_artifact_sha256: str
    model: str = "Qwen3.5-9B"
    seeds: tuple[int, ...] = VISUAL_CSV_SEEDS
    max_tokens: int = 256
    temperature: float = 0.2
    enable_thinking: bool = False

    def __post_init__(self) -> None:
        if not self.model.strip():
            raise ValueError("model cannot be empty")
        artifact = str(self.verifier_artifact_sha256 or "").strip().lower()
        if len(artifact) != 64 or any(
            character not in "0123456789abcdef" for character in artifact
        ):
            raise ValueError("verifier_artifact_sha256 must be a SHA-256")
        object.__setattr__(self, "verifier_artifact_sha256", artifact)
        if len(self.seeds) != 3 or len(set(self.seeds)) != 3:
            raise ValueError("visual CSV requires exactly three unique seeds")
        if self.max_tokens <= 0 or self.temperature < 0:
            raise ValueError("invalid visual CSV generation parameters")
        if self.enable_thinking:
            raise ValueError("visual CSV must keep hidden thinking disabled")

    def fingerprint(self) -> str:
        return canonical_sha256(
            {
                "schema_version": VISUAL_CSV_SCHEMA_VERSION,
                "config": self.to_dict(),
                "prompt": _VISUAL_CSV_SYSTEM,
                "parser": "strict_visual_csv_four_field_v1",
            }
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            **asdict(self),
            "seeds": list(self.seeds),
        }

    @classmethod
    def from_dict(cls, value: Any) -> VisualCsvConfig:
        if not isinstance(value, Mapping) or set(value) != {
            "verifier_artifact_sha256",
            "model",
            "seeds",
            "max_tokens",
            "temperature",
            "enable_thinking",
        }:
            raise ValueError("visual CSV result has an invalid verifier config")
        seeds = value["seeds"]
        if not isinstance(seeds, list):
            raise ValueError("visual CSV verifier config seeds must be an array")
        return cls(
            verifier_artifact_sha256=value["verifier_artifact_sha256"],
            model=value["model"],
            seeds=tuple(seeds),
            max_tokens=value["max_tokens"],
            temperature=value["temperature"],
            enable_thinking=value["enable_thinking"],
        )


@dataclass(frozen=True)
class VisualCsvJob:
    trajectory_id: str
    prefix_id: str
    prefix_index: int
    dataset: str
    sample_id: str
    sample: ModelSample
    frame_paths: tuple[str, ...]
    timestamps: tuple[float, ...]
    frame_content_sha256s: tuple[str, ...]
    source_sha256: str

    def __post_init__(self) -> None:
        if (
            not self.frame_paths
            or len(self.frame_paths) != len(self.timestamps)
            or len(self.frame_paths) != len(self.frame_content_sha256s)
        ):
            raise ValueError("visual CSV job frame provenance must be aligned")
        if any(
            len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
            for value in self.frame_content_sha256s
        ):
            raise ValueError("visual CSV frame content SHA-256 is invalid")


_VISUAL_CSV_SYSTEM = (
    "You are a candidate-blind visual completeness verifier. Use only the public "
    "multiple-choice question and the attached frames. You have no tools and must "
    "ignore any prior textual reasoning. Return the best answer supported by these "
    "frames and cite the zero-based frame indices that directly support it. Decide "
    "whether the visible evidence is complete enough to answer. Return one raw JSON "
    "object with exactly answer, frame_indices, evidence_complete, and "
    "missing_evidence. frame_indices must be non-empty. If evidence_complete is "
    "true, missing_evidence must be empty; if false, list at least one concrete "
    "piece of missing visual evidence."
)


def visual_csv_response_format(option_letters: Sequence[str]) -> dict[str, Any]:
    letters = tuple(str(item).strip().upper() for item in option_letters)
    if len(letters) < 2 or len(set(letters)) != len(letters):
        raise ValueError("response schema requires unique answer options")
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "visual_counterfactual_answer",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "answer": {"type": "string", "enum": list(letters)},
                    "frame_indices": {
                        "type": "array",
                        "items": {"type": "integer", "minimum": 0},
                        "minItems": 1,
                        "uniqueItems": True,
                    },
                    "evidence_complete": {"type": "boolean"},
                    "missing_evidence": {
                        "type": "array",
                        "items": {"type": "string", "minLength": 1},
                        "uniqueItems": True,
                    },
                },
                "required": sorted(_DECISION_KEYS),
                "additionalProperties": False,
            },
        },
    }


def parse_visual_csv_response(
    text: str,
    valid_letters: Iterable[str],
    frame_count: int,
) -> VisualCsvDecision | None:
    letters = {str(item).strip().upper() for item in valid_letters}
    try:
        payload = json.loads((text or "").strip())
        if not isinstance(payload, Mapping) or set(payload) != _DECISION_KEYS:
            return None
        answer = str(payload["answer"]).strip().upper()
        indices = payload["frame_indices"]
        complete = payload["evidence_complete"]
        missing = payload["missing_evidence"]
        if answer not in letters or not isinstance(indices, list) or not indices:
            return None
        if any(isinstance(item, bool) or not isinstance(item, int) for item in indices):
            return None
        unique = tuple(dict.fromkeys(indices))
        if len(unique) != len(indices) or any(
            item < 0 or item >= frame_count for item in unique
        ):
            return None
        if not isinstance(complete, bool) or not isinstance(missing, list):
            return None
        if any(not isinstance(item, str) or not item.strip() for item in missing):
            return None
        normalized_missing = tuple(" ".join(item.strip().split()) for item in missing)
        if len(set(normalized_missing)) != len(normalized_missing):
            return None
        if complete != (not normalized_missing):
            return None
        return VisualCsvDecision(
            answer=answer,
            frame_indices=unique,
            evidence_complete=complete,
            missing_evidence=normalized_missing,
        )
    except (TypeError, ValueError, json.JSONDecodeError):
        return None


def build_visual_csv_messages(job: VisualCsvJob) -> list[dict[str, Any]]:
    choices = "\n".join(
        f"{letter}. {text}" for letter, text in job.sample.choices.items()
    )
    content: list[dict[str, Any]] = [
        {
            "type": "text",
            "text": (
                f"Question: {job.sample.question}\nOptions:\n{choices}\n"
                "Answer using only the following observed frames."
            ),
        }
    ]
    for index, (path, timestamp) in enumerate(
        zip(job.frame_paths, job.timestamps, strict=True)
    ):
        content.append(
            {
                "type": "text",
                "text": f"Frame index {index}, timestamp {timestamp:.3f} seconds",
            }
        )
        content.append(
            {
                "type": "image_url",
                "image_url": {"url": Path(path).resolve().as_uri()},
            }
        )
    messages = [
        {"role": "system", "content": _VISUAL_CSV_SYSTEM},
        {"role": "user", "content": content},
    ]
    assert_annotation_free_request(messages)
    return messages


def bind_visual_csv_jobs(
    trajectories: Iterable[Mapping[str, Any]],
) -> tuple[VisualCsvJob, ...]:
    jobs: list[VisualCsvJob] = []
    seen_trajectories: set[str] = set()
    seen_prefixes: set[str] = set()
    frame_hash_cache: dict[str, str] = {}
    for row_index, row in enumerate(trajectories):
        if not isinstance(row, Mapping):
            raise ValueError(f"trajectory row {row_index} must be an object")
        if row.get("annotation_leak_check") != "passed":
            raise ValueError("trajectory failed annotation leak audit")
        if int(row.get("candidate_rerun") or 0) != 0:
            raise ValueError("trajectory reran its frozen candidate")
        trajectory_id = _text(row.get("trajectory_id"), "trajectory_id")
        if trajectory_id in seen_trajectories:
            raise ValueError("duplicate trajectory_id")
        seen_trajectories.add(trajectory_id)
        sample = _public_sample(row.get("public_sample"))
        if str(row.get("dataset")) != sample.dataset or str(
            row.get("sample_id")
        ) != sample.sample_id:
            raise ValueError("trajectory identity differs from public_sample")
        states = row.get("perception_states")
        if not isinstance(states, list) or not states:
            raise ValueError("trajectory has no perception prefixes")
        accumulated: list[tuple[str, float]] = []
        observed_pairs: set[tuple[str, float]] = set()
        for expected_index, state in enumerate(states):
            if not isinstance(state, Mapping):
                raise ValueError("perception state must be an object")
            prefix_index = int(state.get("step_index", expected_index))
            if prefix_index != expected_index:
                raise ValueError("prefix indices must be contiguous and ordered")
            paths = state.get("frame_paths")
            timestamps = state.get("timestamps")
            if not isinstance(paths, list) or not isinstance(timestamps, list):
                raise ValueError("perception prefix requires frame paths and timestamps")
            if not paths or len(paths) != len(timestamps):
                raise ValueError("frame paths and timestamps must be non-empty and aligned")
            for path_value, timestamp_value in zip(paths, timestamps, strict=True):
                path = Path(_text(path_value, "frame_path"))
                if not path.is_absolute() or not path.is_file():
                    raise ValueError(f"cached frame is missing: {path}")
                if isinstance(timestamp_value, bool) or not isinstance(
                    timestamp_value, (int, float)
                ):
                    raise ValueError("frame timestamp must be numeric")
                timestamp = float(timestamp_value)
                if not math.isfinite(timestamp) or timestamp < 0:
                    raise ValueError("frame timestamp must be finite and non-negative")
                pair = (str(path.resolve()), timestamp)
                if pair not in observed_pairs:
                    observed_pairs.add(pair)
                    accumulated.append(pair)
            prefix_id = f"{trajectory_id}#prefix-{prefix_index:03d}"
            if prefix_id in seen_prefixes:
                raise ValueError("duplicate visual CSV prefix_id")
            seen_prefixes.add(prefix_id)
            frame_paths = tuple(item[0] for item in accumulated)
            frame_timestamps = tuple(item[1] for item in accumulated)
            for path in frame_paths:
                if path not in frame_hash_cache:
                    frame_hash_cache[path] = _file_sha256(Path(path))
            content_sha256s = tuple(frame_hash_cache[path] for path in frame_paths)
            public_sample = {
                "dataset": sample.dataset,
                "sample_id": sample.sample_id,
                "video": sample.video,
                "question": sample.question,
                "choices": dict(sample.choices),
            }
            source_sha256 = canonical_sha256(
                {
                    "schema_version": VISUAL_CSV_SCHEMA_VERSION,
                    "trajectory_id": trajectory_id,
                    "prefix_index": prefix_index,
                    "public_sample": public_sample,
                    "frame_paths": frame_paths,
                    "timestamps": frame_timestamps,
                    "frame_content_sha256s": content_sha256s,
                    "source_run_fingerprint": row.get("run_fingerprint"),
                }
            )
            jobs.append(
                VisualCsvJob(
                    trajectory_id=trajectory_id,
                    prefix_id=prefix_id,
                    prefix_index=prefix_index,
                    dataset=sample.dataset,
                    sample_id=sample.sample_id,
                    sample=sample,
                    frame_paths=frame_paths,
                    timestamps=frame_timestamps,
                    frame_content_sha256s=content_sha256s,
                    source_sha256=source_sha256,
                )
            )
    return tuple(sorted(jobs, key=lambda item: item.prefix_id))


class VisualCsvClient(Protocol):
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


def _usage_int(usage: Mapping[str, Any], key: str) -> int:
    value = usage.get(key)
    return int(value) if isinstance(value, int) and not isinstance(value, bool) else 0


def _request_material(job: VisualCsvJob) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    messages = build_visual_csv_messages(job)
    request_kwargs = {
        "response_format": visual_csv_response_format(job.sample.option_letters),
        "chat_template_kwargs": {"enable_thinking": False},
        "tool_choice": "none",
    }
    return messages, request_kwargs


def _retryable_infrastructure_failure(item: Mapping[str, Any]) -> bool:
    return (
        bool(item.get("error"))
        and item.get("failure_class") == "infrastructure_error"
    )


def _validate_confirmation(
    item: Mapping[str, Any],
    job: VisualCsvJob,
    expected_seeds: set[int],
) -> int:
    if set(item) != _CONFIRMATION_KEYS:
        raise ValueError("visual CSV confirmation has unexpected fields")
    raw_seed = item.get("judge_seed")
    if isinstance(raw_seed, bool) or not isinstance(raw_seed, int):
        raise ValueError("visual CSV confirmation seed must be an integer")
    seed = raw_seed
    if seed not in expected_seeds:
        raise ValueError("visual CSV confirmation seed is not configured")
    expected_messages, expected_request_kwargs = _request_material(job)
    expected_prompt_sha256 = canonical_sha256(
        {
            "messages": expected_messages,
            "request_kwargs": expected_request_kwargs,
        }
    )
    if (
        item.get("request_messages") != expected_messages
        or item.get("request_kwargs") != expected_request_kwargs
        or item.get("request_prompt_sha256") != expected_prompt_sha256
        or item.get("candidate_blind") is not True
        or item.get("tools_disabled") is not True
        or item.get("media_count") != len(job.frame_paths)
        or item.get("fallback_used") is not False
    ):
        raise ValueError("visual CSV confirmation provenance mismatch")
    assert_annotation_free_request(
        {
            "messages": item["request_messages"],
            "request_kwargs": item["request_kwargs"],
        }
    )
    error = item.get("error")
    parsed_valid = item.get("parsed_valid")
    if error is not None:
        if (
            not isinstance(error, str)
            or not error.strip()
            or parsed_valid is not False
            or item.get("prediction") is not None
            or item.get("frame_indices") != []
            or item.get("evidence_complete") is not None
            or item.get("missing_evidence") != []
            or item.get("raw_response") != ""
            or item.get("failure_class")
            not in {"infrastructure_error", "annotation_leak"}
        ):
            raise ValueError("visual CSV error confirmation is inconsistent")
        return seed
    is_truncated = item.get("failure_class") == "model_truncation"
    decision = (
        None
        if is_truncated
        else parse_visual_csv_response(
            str(item.get("raw_response") or ""),
            job.sample.option_letters,
            len(job.frame_paths),
        )
    )
    if parsed_valid is True:
        if decision is None or (
            item.get("prediction") != decision.answer
            or item.get("frame_indices") != list(decision.frame_indices)
            or item.get("evidence_complete") != decision.evidence_complete
            or item.get("missing_evidence") != list(decision.missing_evidence)
            or item.get("failure_class") is not None
            or item.get("annotation_leak_check") != "passed"
        ):
            raise ValueError("visual CSV parsed confirmation differs from raw response")
    elif (
        parsed_valid is not False
        or decision is not None
        or item.get("prediction") is not None
        or item.get("frame_indices") != []
        or item.get("evidence_complete") is not None
        or item.get("missing_evidence") != []
        or item.get("failure_class") not in {"model_parse_failure", "model_truncation"}
        or (is_truncated and item.get("finish_reason") != "length")
    ):
        raise ValueError("visual CSV invalid model response audit is inconsistent")
    return seed


def _validate_result_for_job(result: Mapping[str, Any], job: VisualCsvJob) -> None:
    if frame_content_sha256s(job.frame_paths) != job.frame_content_sha256s:
        raise ValueError("visual CSV source frame content changed")
    if set(result) != _RESULT_KEYS:
        raise ValueError("visual CSV result has unexpected fields")
    verifier_config = VisualCsvConfig.from_dict(result.get("visual_csv_config"))
    if result.get("visual_csv_config") != verifier_config.to_dict():
        raise ValueError("visual CSV verifier config is not canonical")
    artifact = str(result.get("verifier_artifact_sha256") or "")
    if len(artifact) != 64 or any(
        character not in "0123456789abcdef" for character in artifact
    ):
        raise ValueError("visual CSV verifier artifact SHA-256 is invalid")
    if (
        result.get("schema_version") != 1
        or result.get("visual_csv_schema_version") != VISUAL_CSV_SCHEMA_VERSION
        or result.get("dataset") != job.dataset
        or result.get("sample_id") != job.sample_id
        or result.get("trajectory_id") != job.trajectory_id
        or result.get("prefix_id") != job.prefix_id
        or result.get("prefix_index") != job.prefix_index
        or result.get("source_sha256") != job.source_sha256
        or result.get("visual_csv_config_sha256") != verifier_config.fingerprint()
        or artifact != verifier_config.verifier_artifact_sha256
        or result.get("candidate_blind") is not True
        or result.get("tools_disabled") is not True
        or result.get("media_count") != len(job.frame_paths)
        or result.get("frame_paths_sha256") != canonical_sha256(job.frame_paths)
        or tuple(result.get("frame_content_sha256s") or ())
        != job.frame_content_sha256s
        or result.get("timestamps_sha256") != canonical_sha256(job.timestamps)
        or result.get("annotation_leak_check") != "passed"
    ):
        raise ValueError("visual CSV result identity or frame provenance mismatch")
    raw_seeds = result.get("visual_csv_judge_seeds")
    if (
        not isinstance(raw_seeds, list)
        or len(raw_seeds) != 3
        or any(isinstance(seed, bool) or not isinstance(seed, int) for seed in raw_seeds)
        or len(set(raw_seeds)) != 3
    ):
        raise ValueError("visual CSV result requires exactly three configured seeds")
    confirmations = result.get("visual_csv_confirmations")
    if not isinstance(confirmations, list) or len(confirmations) != 3:
        raise ValueError("visual CSV result requires exactly three confirmations")
    observed = {
        _validate_confirmation(item, job, set(raw_seeds))
        for item in confirmations
        if isinstance(item, Mapping)
    }
    if len(observed) != len(confirmations) or observed != set(raw_seeds):
        raise ValueError("visual CSV confirmations do not match configured seeds")
    if result.get("visual_csv_status") not in {
        "complete",
        "complete_with_failures",
    }:
        raise ValueError("visual CSV result does not contain all three seeds")


class VisualCsvVerifier:
    """Run the three-seed visual-only answer check for one immutable prefix."""

    def __init__(
        self,
        clients: VisualCsvClient | Sequence[VisualCsvClient],
        config: VisualCsvConfig,
    ) -> None:
        values = tuple(clients) if isinstance(clients, Sequence) else (clients,)
        if not values or any(not callable(getattr(item, "chat", None)) for item in values):
            raise ValueError("visual CSV requires one or more clients")
        self.clients = values
        self.config = config

    def _client(self, prefix_id: str) -> VisualCsvClient:
        digest = hashlib.sha256(prefix_id.encode("utf-8")).digest()
        return self.clients[int.from_bytes(digest[:8], "big") % len(self.clients)]

    def _one(
        self, job: VisualCsvJob, seed: int, messages: list[dict[str, Any]]
    ) -> dict[str, Any]:
        expected_messages, request_kwargs = _request_material(job)
        if messages != expected_messages:
            raise ValueError("visual CSV request messages differ from immutable job")
        response_format = request_kwargs["response_format"]
        try:
            assert_annotation_free_request(
                {"messages": messages, "request_kwargs": request_kwargs}
            )
            result = self._client(job.prefix_id).chat(
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
            return {
                "judge_seed": seed,
                "prediction": None,
                "frame_indices": [],
                "evidence_complete": None,
                "missing_evidence": [],
                "parsed_valid": False,
                "raw_response": "",
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
                "media_count": len(job.frame_paths),
                "annotation_leak_check": (
                    "failed" if isinstance(error, AnnotationLeakError) else "passed"
                ),
                "fallback_used": False,
            }
        decision = (
            None
            if result.finish_reason == "length"
            else parse_visual_csv_response(
                result.content, job.sample.option_letters, len(job.frame_paths)
            )
        )
        return {
            "judge_seed": seed,
            "prediction": decision.answer if decision else None,
            "frame_indices": list(decision.frame_indices) if decision else [],
            "evidence_complete": (
                decision.evidence_complete if decision is not None else None
            ),
            "missing_evidence": (
                list(decision.missing_evidence) if decision is not None else []
            ),
            "parsed_valid": decision is not None,
            "raw_response": result.content,
            "finish_reason": result.finish_reason,
            "usage": dict(result.usage),
            "prompt_tokens": _usage_int(result.usage, "prompt_tokens"),
            "completion_tokens": _usage_int(result.usage, "completion_tokens"),
            "total_tokens": _usage_int(result.usage, "total_tokens"),
            "latency_s": float(result.latency_s),
            "request_messages": messages,
            "request_kwargs": request_kwargs,
            "request_prompt_sha256": canonical_sha256(
                {"messages": messages, "request_kwargs": request_kwargs}
            ),
            "error": None,
            "error_type": None,
            "failure_class": (
                None
                if decision
                else "model_truncation"
                if result.finish_reason == "length"
                else "model_parse_failure"
            ),
            "candidate_blind": True,
            "tools_disabled": True,
            "media_count": len(job.frame_paths),
            "annotation_leak_check": "passed",
            "fallback_used": False,
        }

    def _base_row(self, job: VisualCsvJob) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "visual_csv_schema_version": VISUAL_CSV_SCHEMA_VERSION,
            "dataset": job.dataset,
            "sample_id": job.sample_id,
            "trajectory_id": job.trajectory_id,
            "prefix_id": job.prefix_id,
            "prefix_index": job.prefix_index,
            "source_sha256": job.source_sha256,
            "visual_csv_config": self.config.to_dict(),
            "visual_csv_config_sha256": self.config.fingerprint(),
            "verifier_artifact_sha256": self.config.verifier_artifact_sha256,
            "visual_csv_judge_seeds": list(self.config.seeds),
            "candidate_blind": True,
            "tools_disabled": True,
            "media_count": len(job.frame_paths),
            "frame_paths_sha256": canonical_sha256(job.frame_paths),
            "frame_content_sha256s": list(job.frame_content_sha256s),
            "timestamps_sha256": canonical_sha256(job.timestamps),
            "annotation_leak_check": "passed",
            "visual_csv_confirmations": [],
            "visual_csv_status": "incomplete",
        }

    def _finalize(self, row: dict[str, Any]) -> None:
        confirmations = sorted(
            row["visual_csv_confirmations"],
            key=lambda item: int(item["judge_seed"]),
        )
        row["visual_csv_confirmations"] = confirmations
        observed = {int(item["judge_seed"]) for item in confirmations}
        required = set(self.config.seeds)
        row["visual_csv_status"] = (
            "complete"
            if observed == required
            and all(item.get("parsed_valid") is True for item in confirmations)
            and all(item.get("error") is None for item in confirmations)
            else "complete_with_failures"
            if observed == required
            else "incomplete"
        )
        row["visual_csv_prompt_tokens"] = sum(
            int(item.get("prompt_tokens") or 0) for item in confirmations
        )
        row["visual_csv_completion_tokens"] = sum(
            int(item.get("completion_tokens") or 0) for item in confirmations
        )
        row["visual_csv_total_tokens"] = sum(
            int(item.get("total_tokens") or 0) for item in confirmations
        )
        row["visual_csv_latency_s"] = sum(
            float(item.get("latency_s") or 0.0) for item in confirmations
        )

    def _resume_row(
        self,
        job: VisualCsvJob,
        existing: Mapping[str, Any] | None,
        *,
        retry_errors: bool = False,
    ) -> dict[str, Any]:
        if frame_content_sha256s(job.frame_paths) != job.frame_content_sha256s:
            raise RuntimeError("visual CSV frame content changed before request/resume")
        expected = self._base_row(job)
        if existing is None:
            return expected
        row = dict(existing)
        if set(row) != _RESULT_KEYS:
            raise RuntimeError("visual CSV resume row has unexpected fields")
        for field in (
            "schema_version",
            "visual_csv_schema_version",
            "dataset",
            "sample_id",
            "trajectory_id",
            "prefix_id",
            "prefix_index",
            "source_sha256",
            "visual_csv_config",
            "visual_csv_config_sha256",
            "verifier_artifact_sha256",
            "visual_csv_judge_seeds",
            "candidate_blind",
            "tools_disabled",
            "media_count",
            "frame_paths_sha256",
            "frame_content_sha256s",
            "timestamps_sha256",
            "annotation_leak_check",
        ):
            if row.get(field) != expected[field]:
                raise RuntimeError(f"visual CSV resume mismatch: {field}")
        confirmations = row.get("visual_csv_confirmations")
        if not isinstance(confirmations, list) or any(
            not isinstance(item, Mapping) for item in confirmations
        ):
            raise RuntimeError("visual CSV confirmations must be an object array")
        try:
            seeds = [int(item["judge_seed"]) for item in confirmations]
        except (KeyError, TypeError, ValueError) as error:
            raise RuntimeError("visual CSV confirmation has invalid seed") from error
        if len(seeds) != len(set(seeds)) or any(
            seed not in self.config.seeds for seed in seeds
        ):
            raise RuntimeError("visual CSV resume has duplicate or unexpected seeds")
        for item in confirmations:
            try:
                _validate_confirmation(item, job, set(self.config.seeds))
            except (AnnotationLeakError, ValueError) as error:
                raise RuntimeError(
                    "visual CSV resume confirmation provenance mismatch"
                ) from error
        row["visual_csv_confirmations"] = [
            dict(item)
            for item in confirmations
            if not retry_errors
            or not _retryable_infrastructure_failure(item)
        ]
        self._finalize(row)
        return row

    def prepare_resume(
        self,
        job: VisualCsvJob,
        existing: Mapping[str, Any],
        *,
        retry_errors: bool = False,
    ) -> dict[str, Any]:
        """Validate one persisted row and drop only failed seeds when requested."""

        return self._resume_row(job, existing, retry_errors=retry_errors)

    def verify(
        self,
        job: VisualCsvJob,
        *,
        existing: Mapping[str, Any] | None = None,
        on_update: Callable[[dict[str, Any]], None] | None = None,
        retry_errors: bool = False,
    ) -> dict[str, Any]:
        row = self._resume_row(job, existing, retry_errors=retry_errors)
        observed = {
            int(item["judge_seed"]) for item in row["visual_csv_confirmations"]
        }
        messages = build_visual_csv_messages(job)
        for seed in self.config.seeds:
            if seed in observed:
                continue
            row["visual_csv_confirmations"].append(self._one(job, seed, messages))
            self._finalize(row)
            if on_update is not None:
                on_update(dict(row))
        self._finalize(row)
        return row


def attach_visual_csv_results(
    trajectories: Iterable[Mapping[str, Any]],
    results: Iterable[Mapping[str, Any]],
    *,
    verifier_artifact_sha256: str,
) -> tuple[dict[str, Any], ...]:
    """Attach immutable visual-only confirmations to their exact source prefix."""

    expected_verifier = str(verifier_artifact_sha256 or "").strip().lower()
    if len(expected_verifier) != 64 or any(
        character not in "0123456789abcdef" for character in expected_verifier
    ):
        raise ValueError("expected visual CSV verifier artifact must be a SHA-256")

    sources = list(trajectories)
    jobs = bind_visual_csv_jobs(sources)
    job_by_prefix = {job.prefix_id: job for job in jobs}
    result_by_prefix: dict[str, Mapping[str, Any]] = {}
    config_hashes: set[str] = set()
    verifier_artifact_hashes: set[str] = set()
    configured_seed_sets: set[tuple[int, ...]] = set()
    for raw in results:
        prefix_id = _text(raw.get("prefix_id"), "visual CSV prefix_id")
        if prefix_id in result_by_prefix:
            raise ValueError("duplicate visual CSV result prefix_id")
        job = job_by_prefix.get(prefix_id)
        if job is None:
            raise ValueError(f"visual CSV result has unknown prefix: {prefix_id}")
        _validate_result_for_job(raw, job)
        if raw.get("verifier_artifact_sha256") != expected_verifier:
            raise ValueError("visual CSV verifier artifact differs from expected weights")
        result_by_prefix[prefix_id] = raw
        config_hashes.add(str(raw["visual_csv_config_sha256"]))
        verifier_artifact_hashes.add(str(raw["verifier_artifact_sha256"]))
        configured_seed_sets.add(tuple(raw["visual_csv_judge_seeds"]))
    if (
        len(config_hashes) != 1
        or len(verifier_artifact_hashes) != 1
        or len(configured_seed_sets) != 1
    ):
        raise ValueError(
            "visual CSV results mix verifier configs, artifacts, or seed schemas"
        )
    output: list[dict[str, Any]] = []
    expected: set[str] = set()
    for source in sources:
        row = deepcopy(dict(source))
        _strip_private_trajectory_fields(row)
        trajectory_id = _text(row.get("trajectory_id"), "trajectory_id")
        states = row.get("perception_states")
        if not isinstance(states, list) or not states:
            raise ValueError("trajectory has no perception prefixes")
        for index, state in enumerate(states):
            if not isinstance(state, dict):
                raise ValueError("perception prefix must be an object")
            prefix_index = int(state.get("step_index", index))
            if prefix_index != index:
                raise ValueError("prefix indices must be contiguous and ordered")
            prefix_id = f"{trajectory_id}#prefix-{prefix_index:03d}"
            expected.add(prefix_id)
            result = result_by_prefix.get(prefix_id)
            if result is None:
                raise ValueError(f"missing visual CSV result: {prefix_id}")
            if (
                result.get("trajectory_id") != trajectory_id
                or int(result.get("prefix_index", -1)) != prefix_index
                or result.get("dataset") != row.get("dataset")
                or result.get("sample_id") != row.get("sample_id")
            ):
                raise ValueError("visual CSV result identity mismatch")
            if result.get("visual_csv_status") not in {
                "complete",
                "complete_with_failures",
            }:
                raise ValueError("visual CSV result does not contain all three seeds")
            confirmations = result.get("visual_csv_confirmations")
            if not isinstance(confirmations, list) or len(confirmations) != 3:
                raise ValueError("visual CSV result requires exactly three confirmations")
            state["visual_csv_confirmations"] = deepcopy(confirmations)
            state["visual_csv_source_sha256"] = result.get("source_sha256")
            state["visual_csv_config_sha256"] = result.get(
                "visual_csv_config_sha256"
            )
            state["visual_csv_config"] = deepcopy(result.get("visual_csv_config"))
            state["visual_csv_verifier_artifact_sha256"] = result.get(
                "verifier_artifact_sha256"
            )
            state["visual_csv_judge_seeds"] = deepcopy(
                result.get("visual_csv_judge_seeds")
            )
            state["visual_csv_frame_paths_sha256"] = result.get(
                "frame_paths_sha256"
            )
            state["visual_csv_frame_content_sha256s"] = deepcopy(
                result.get("frame_content_sha256s")
            )
            state["visual_csv_timestamps_sha256"] = result.get(
                "timestamps_sha256"
            )
            # Ground truth is joined by label_visual_csv_prefixes only.  Attaching
            # model outputs alone must never turn a prefix into a STOP target.
            state["evidence_complete"] = False
        output.append(row)
    extras = sorted(set(result_by_prefix) - expected)
    if extras:
        raise ValueError(f"visual CSV results contain extra prefixes: {extras[:3]}")
    return tuple(output)


def label_visual_csv_prefixes(
    trajectories: Iterable[Mapping[str, Any]],
    answers: Mapping[tuple[str, str], str],
    *,
    candidate_answers: Mapping[tuple[str, str], str] | None = None,
) -> tuple[dict[str, Any], ...]:
    """Join labels offline and materialize STOP/CONTINUE without serializing GT."""

    sources = list(trajectories)
    job_by_prefix = {job.prefix_id: job for job in bind_visual_csv_jobs(sources)}
    output: list[dict[str, Any]] = []
    for source in sources:
        row = deepcopy(dict(source))
        _strip_private_trajectory_fields(row)
        identity = (str(row.get("dataset") or ""), str(row.get("sample_id") or ""))
        answer = str(answers.get(identity) or "").strip().upper()
        if len(answer) != 1 or not "A" <= answer <= "H":
            raise ValueError(f"missing offline answer for {identity[0]}/{identity[1]}")
        if candidate_answers is not None:
            candidate = str(candidate_answers.get(identity) or "").strip().upper()
            if len(candidate) != 1 or not "A" <= candidate <= "H":
                raise ValueError(
                    f"missing frozen candidate for {identity[0]}/{identity[1]}"
                )
            row["candidate_training_stratum"] = (
                "candidate_correct" if candidate == answer else "candidate_wrong"
            )
        states = row.get("perception_states")
        if not isinstance(states, list) or not states:
            raise ValueError("trajectory has no visual CSV prefixes")
        trajectory_id = _text(row.get("trajectory_id"), "trajectory_id")
        for prefix_index, state in enumerate(states):
            if not isinstance(state, dict):
                raise ValueError("perception prefix must be an object")
            prefix_id = f"{trajectory_id}#prefix-{prefix_index:03d}"
            job = job_by_prefix[prefix_id]
            confirmations = state.get("visual_csv_confirmations")
            if not isinstance(confirmations, list) or len(confirmations) != 3:
                raise ValueError("prefix lacks three visual CSV confirmations")
            configured_seeds = state.get("visual_csv_judge_seeds")
            if (
                not isinstance(configured_seeds, list)
                or len(configured_seeds) != 3
                or any(
                    isinstance(seed, bool) or not isinstance(seed, int)
                    for seed in configured_seeds
                )
                or len(set(configured_seeds)) != 3
            ):
                raise ValueError("prefix lacks exactly three configured visual CSV seeds")
            if (
                state.get("visual_csv_frame_paths_sha256")
                != canonical_sha256(job.frame_paths)
                or tuple(state.get("visual_csv_frame_content_sha256s") or ())
                != job.frame_content_sha256s
                or frame_content_sha256s(job.frame_paths)
                != job.frame_content_sha256s
                or state.get("visual_csv_timestamps_sha256")
                != canonical_sha256(job.timestamps)
            ):
                raise ValueError("prefix visual CSV frame provenance mismatch")
            seeds: set[int] = set()
            predictions: list[str] = []
            valid = True
            for confirmation in confirmations:
                if not isinstance(confirmation, Mapping):
                    raise ValueError("visual CSV confirmation must be an object")
                seed = _validate_confirmation(
                    confirmation, job, set(configured_seeds)
                )
                if seed in seeds:
                    raise ValueError("duplicate visual CSV confirmation seed")
                seeds.add(seed)
                prediction = str(confirmation.get("prediction") or "").upper()
                predictions.append(prediction)
                valid = valid and (
                    confirmation.get("parsed_valid") is True
                    and confirmation.get("error") is None
                    and confirmation.get("candidate_blind") is True
                    and confirmation.get("annotation_leak_check") == "passed"
                    and isinstance(confirmation.get("frame_indices"), list)
                    and bool(confirmation.get("frame_indices"))
                )
            if seeds != set(configured_seeds):
                raise ValueError("visual CSV confirmations differ from configured seeds")
            # The model's own evidence_complete boolean is diagnostic only.  The
            # STOP label is the counterfactual 3/3 answer-correctness outcome.
            state["evidence_complete"] = bool(
                valid and all(prediction == answer for prediction in predictions)
            )
            state["visual_csv_label_valid"] = valid
            state["evidence_prediction"] = (
                predictions[0]
                if valid and len(set(predictions)) == 1
                else None
            )
            state["completion_gate_kind"] = "visual_csv_3of3_offline_label"
        row["offline_label_join"] = "ground_truth_used_for_boolean_only_not_serialized"
        output.append(row)
    return tuple(output)


def _non_negative_number(value: Any) -> float:
    if (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
        and float(value) >= 0
    ):
        return float(value)
    return float("inf")


def _optional_non_negative_number(value: Any) -> float:
    number = _non_negative_number(value)
    return 0.0 if math.isinf(number) else number


def _request_usage_total(request: Mapping[str, Any]) -> float:
    usage = request.get("usage")
    return (
        _optional_non_negative_number(usage.get("total_tokens"))
        if isinstance(usage, Mapping)
        else 0.0
    )


def _trace_through_visual_prefix(
    row: Mapping[str, Any], prefix_index: int
) -> list[dict[str, Any]]:
    trace = row.get("request_trace")
    if not isinstance(trace, list) or not trace:
        raise ValueError("stable visual CSV trajectory requires request_trace")
    observation_stages = {"perception", "observation", "observer"}
    matching: list[int] = []
    for position, raw in enumerate(trace):
        if not isinstance(raw, Mapping):
            raise ValueError("request_trace entry must be an object")
        stage = str(raw.get("stage") or raw.get("request_kind") or "").casefold()
        if stage in observation_stages and raw.get("prefix_index") == prefix_index:
            matching.append(position)
    if not matching:
        raise ValueError("complete prefix has no accepted Observer request")
    retained = [deepcopy(dict(item)) for item in trace[: matching[-1] + 1]]

    def accepted_planners(index: int) -> int:
        return sum(
            str(item.get("stage") or item.get("request_kind") or "").casefold()
            in {"controller", "planner"}
            and item.get("prefix_index") == index
            and item.get("action_accepted") is True
            for item in retained
        )

    if accepted_planners(-1) != 1:
        raise ValueError("stable trajectory requires one accepted initial Planner action")
    for index in range(prefix_index):
        if accepted_planners(index) != 1:
            raise ValueError(
                f"incomplete prefix {index} requires one accepted next Planner action"
            )
    return retained


def _selected_cost(row: Mapping[str, Any]) -> tuple[float, float, int, float, str]:
    return (
        _non_negative_number(row.get("retained_total_tokens")),
        _non_negative_number(row.get("retained_visual_tokens")),
        int(row.get("retained_tool_steps") or 0),
        _non_negative_number(row.get("retained_latency_s")),
        str(row.get("trajectory_id") or ""),
    )


def _truncate_visual_csv_trajectory(
    row: Mapping[str, Any], prefix_index: int, prediction: str
) -> dict[str, Any]:
    selected = deepcopy(dict(row))
    states = selected.get("perception_states")
    steps = selected.get("tool_steps")
    if not isinstance(states, list) or prefix_index >= len(states):
        raise ValueError("complete visual CSV prefix is outside perception_states")
    if not isinstance(steps, list) or len(steps) <= prefix_index:
        raise ValueError("complete visual CSV prefix is outside tool_steps")
    retained_states = states[: prefix_index + 1]
    retained_steps = steps[: prefix_index + 1]
    retained_trace = _trace_through_visual_prefix(selected, prefix_index)
    final_memory = retained_states[-1].get("memory_after")
    if not isinstance(final_memory, Mapping):
        raise ValueError("complete visual CSV prefix has no memory_after")
    source_request_tokens = sum(_request_usage_total(item) for item in retained_trace)
    visual_csv_tokens = sum(
        _request_usage_total(item)
        for item in retained_states[-1]["visual_csv_confirmations"]
        if isinstance(item, Mapping)
    )
    retained_visual_tokens = sum(
        _optional_non_negative_number(item.get("visual_tokens"))
        for item in retained_steps
        if isinstance(item, Mapping)
    )
    retained_latency = sum(
        _optional_non_negative_number(item.get("latency_s"))
        for item in (*retained_trace, *retained_steps)
        if isinstance(item, Mapping)
    )
    selected.update(
        {
            "perception_states": retained_states,
            "tool_steps": retained_steps,
            "request_trace": retained_trace,
            "prediction": prediction,
            "final_prediction": prediction,
            "evidence_complete": True,
            "stop_reason": "earliest_visual_csv_3of3_prefix",
            "earliest_complete_prefix_index": prefix_index,
            "event_ledger": deepcopy(final_memory.get("event_ledger") or []),
            "option_ledger": deepcopy(final_memory.get("option_ledger") or {}),
            "unresolved": deepcopy(final_memory.get("unresolved") or []),
            "observed_intervals": deepcopy(
                final_memory.get("observed_intervals") or []
            ),
            "rounds": prefix_index + 1,
            "turn_count": len(retained_trace),
            "retained_source_total_tokens": source_request_tokens,
            "retained_visual_csv_total_tokens": visual_csv_tokens,
            "retained_total_tokens": source_request_tokens + visual_csv_tokens,
            "retained_visual_tokens": retained_visual_tokens,
            "retained_tool_steps": prefix_index + 1,
            "retained_latency_s": retained_latency,
            "total_tokens": source_request_tokens + visual_csv_tokens,
            "visual_tokens": retained_visual_tokens,
            "latency_s": retained_latency,
            "_selection_stable": True,
        }
    )
    usage = selected.get("usage")
    if isinstance(usage, Mapping):
        selected["usage"] = {
            **deepcopy(dict(usage)),
            "total_tokens": source_request_tokens + visual_csv_tokens,
        }
    _strip_private_trajectory_fields(selected)
    return selected


def select_visual_csv_trajectories(
    trajectories: Iterable[Mapping[str, Any]],
    results: Iterable[Mapping[str, Any]],
    answers: Mapping[tuple[str, str], str],
    *,
    verifier_artifact_sha256: str,
) -> tuple[tuple[dict[str, Any], ...], tuple[dict[str, Any], ...], dict[str, Any]]:
    """Offline-label visual prefixes and select one deterministic stable trace/sample."""

    sources = [dict(row) for row in trajectories]
    candidate_answers: dict[tuple[str, str], str] = {}
    source_samples: set[tuple[str, str]] = set()
    for source in sources:
        identity = (str(source.get("dataset") or ""), str(source.get("sample_id") or ""))
        candidate = str(
            source.get("candidate_answer", source.get("direct_candidate", "")) or ""
        ).strip().upper()
        if len(candidate) != 1 or not "A" <= candidate <= "H":
            raise ValueError(f"trajectory lacks frozen candidate for {identity}")
        previous = candidate_answers.setdefault(identity, candidate)
        if previous != candidate:
            raise ValueError(f"frozen candidate drift across trajectories for {identity}")
        source_samples.add(identity)

    attached = attach_visual_csv_results(
        sources,
        results,
        verifier_artifact_sha256=verifier_artifact_sha256,
    )
    labeled_rows = list(
        label_visual_csv_prefixes(
            attached, answers, candidate_answers=candidate_answers
        )
    )
    stable_by_sample: dict[tuple[str, str], list[dict[str, Any]]] = {}
    prefix_counts: Counter[str] = Counter()
    for row in labeled_rows:
        states = row["perception_states"]
        earliest: tuple[int, str] | None = None
        for index, state in enumerate(states):
            complete = state.get("evidence_complete") is True
            prefix_counts["complete" if complete else "incomplete"] += 1
            prediction = str(state.get("evidence_prediction") or "").upper()
            if complete and earliest is None:
                if len(prediction) != 1 or not "A" <= prediction <= "H":
                    raise ValueError("complete visual CSV prefix has no unanimous answer")
                earliest = (index, prediction)
        row["_selection_stable"] = False
        identity = (str(row["dataset"]), str(row["sample_id"]))
        stable = (
            earliest is not None
            and row.get("annotation_leak_check") == "passed"
            and int(row.get("candidate_rerun") or 0) == 0
            and row.get("fallback_used") is not True
            and row.get("fallback_to_candidate") is not True
            and not any(row.get(field) for field in ("error", "error_type", "api_error"))
            and all(
                state.get("visual_csv_label_valid") is True
                for state in states[: earliest[0] + 1]
            )
        )
        if stable:
            assert earliest is not None
            stable_row = _truncate_visual_csv_trajectory(
                row, earliest[0], earliest[1]
            )
            stable_by_sample.setdefault(identity, []).append(stable_row)

    selected = [
        deepcopy(min(stable_by_sample[identity], key=_selected_cost))
        for identity in sorted(stable_by_sample)
    ]
    selected_by_dataset = Counter(str(row["dataset"]) for row in selected)
    selected_by_stratum = Counter(
        str(row["candidate_training_stratum"]) for row in selected
    )
    all_answer_samples = set(answers)
    no_stable = sorted(all_answer_samples - set(stable_by_sample))
    no_stable_ids = {
        dataset: [
            sample_id
            for sample_dataset, sample_id in no_stable
            if sample_dataset == dataset
        ]
        for dataset in sorted({item[0] for item in all_answer_samples})
    }
    summary = {
        "trajectories": len(sources),
        "samples": len(all_answer_samples),
        "samples_with_trajectories": len(source_samples),
        "prefixes": sum(prefix_counts.values()),
        "prefix_status": dict(sorted(prefix_counts.items())),
        "stable_trajectories": sum(len(rows) for rows in stable_by_sample.values()),
        "selected": len(selected),
        "selected_by_dataset": dict(sorted(selected_by_dataset.items())),
        "selected_by_candidate_stratum": dict(sorted(selected_by_stratum.items())),
        "candidate_fixes": selected_by_stratum.get("candidate_wrong", 0),
        "no_stable": len(no_stable),
        "no_stable_by_dataset": {
            dataset: len(ids) for dataset, ids in no_stable_ids.items()
        },
        "no_stable_sample_ids": no_stable_ids,
        "judge_seeds": list(VISUAL_CSV_SEEDS),
        "answers_serialized_into_selected": 0,
        "candidates_serialized_into_selected": 0,
        "training_quantity_policy": "advisory_only",
    }
    return tuple(labeled_rows), tuple(selected), summary


__all__ = [
    "VISUAL_CSV_SCHEMA_VERSION",
    "VISUAL_CSV_SEEDS",
    "VisualCsvConfig",
    "VisualCsvDecision",
    "VisualCsvJob",
    "VisualCsvVerifier",
    "attach_visual_csv_results",
    "bind_visual_csv_jobs",
    "build_visual_csv_messages",
    "label_visual_csv_prefixes",
    "parse_visual_csv_response",
    "select_visual_csv_trajectories",
    "visual_csv_response_format",
]
