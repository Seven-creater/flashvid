"""Frozen-input paired role ablations for Perception-Memory EVA.

The end-to-end Planner cells measure a system intervention.  Observer, Verifier,
and Answerer cells instead use this module so both arms receive byte-identical
public inputs.  Ground-truth labels are deliberately absent; scoring is a later
offline join.
"""

from __future__ import annotations

import copy
import hashlib
import math
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any, Protocol
from urllib.parse import unquote, urlparse
from urllib.request import url2pathname

from .client import ChatResult
from .perception_memory_eva import (
    EvidenceMemory,
    bind_perception_state,
    build_perception_messages,
    build_perception_retry_messages,
    build_cited_judge_messages,
    build_runtime_visual_csv_messages,
    parse_evidence_decision,
    perception_response_format,
)
from .perception_memory_visual_csv import (
    parse_visual_csv_response,
    visual_csv_response_format,
)
from .privacy import assert_annotation_free_request
from .qwen_agents.core import FrameObservation, FrameRequest
from .qwen_sft import canonical_sha256
from .schemas import ModelSample


FROZEN_ROLE_INPUT_VERSION = "role_ablation_frozen_inputs_v1"
PAIRED_ROLE_RESULT_VERSION = "role_ablation_paired_result_v2"
PAIRED_ROLES = ("observer", "verifier", "answerer")
ROLE_ABLATION_DEV30_COUNTS = {
    "lvbench": 10,
    "lsdbench": 10,
    "cgbench": 10,
}
_PUBLIC_SAMPLE_KEYS = {"dataset", "sample_id", "video", "question", "choices"}
_OBSERVER_STAGES = {"perception", "observer", "confirmation_perception"}
_ANSWERER_STAGES = {
    "evidence_judge",
    "answerer",
    "confirmation_judge",
}
_OBSERVER_RETRY_REASONS = {
    "finish_reason_length",
    "invalid_json_or_schema",
    "invalid_observation_binding",
    "legacy_timestamp_schema",
}


class PairedRoleClient(Protocol):
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


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _lower_sha(value: Any, field: str) -> str:
    text = str(value or "")
    if len(text) != 64 or any(character not in "0123456789abcdef" for character in text):
        raise ValueError(f"{field} must be a lowercase SHA-256")
    return text


def _public_sample(value: Any) -> ModelSample:
    if not isinstance(value, Mapping) or set(value) != _PUBLIC_SAMPLE_KEYS:
        raise ValueError("public_sample must contain exactly the public fields")
    choices = value.get("choices")
    if not isinstance(choices, Mapping) or len(choices) < 2:
        raise ValueError("public_sample.choices must contain at least two options")
    normalized: dict[str, str] = {}
    for raw_letter, raw_text in choices.items():
        letter = str(raw_letter).strip().upper()
        text = str(raw_text).strip()
        if (
            len(letter) != 1
            or not "A" <= letter <= "H"
            or letter in normalized
            or not text
        ):
            raise ValueError("public_sample choices must use unique A-H labels")
        normalized[letter] = text
    sample = ModelSample(
        dataset=str(value["dataset"]).strip(),
        sample_id=str(value["sample_id"]).strip(),
        video=str(value["video"]).strip(),
        question=str(value["question"]).strip(),
        choices=normalized,
        candidate_answer=None,
    )
    payload = {
        "dataset": sample.dataset,
        "sample_id": sample.sample_id,
        "video": sample.video,
        "question": sample.question,
        "choices": dict(sample.choices),
    }
    assert_annotation_free_request(payload)
    return sample


def _frame_records(
    paths: Sequence[Any], timestamps: Sequence[Any]
) -> list[dict[str, Any]]:
    if not paths or len(paths) != len(timestamps):
        raise ValueError("frame paths and timestamps must be non-empty and aligned")
    frames: list[dict[str, Any]] = []
    for raw_path, raw_timestamp in zip(paths, timestamps, strict=True):
        path = Path(str(raw_path)).resolve()
        if not path.is_absolute() or not path.is_file():
            raise ValueError(f"frozen role frame is missing: {path}")
        if isinstance(raw_timestamp, bool) or not isinstance(raw_timestamp, (int, float)):
            raise ValueError("frozen role timestamp must be numeric")
        timestamp = float(raw_timestamp)
        if not math.isfinite(timestamp) or timestamp < 0:
            raise ValueError("frozen role timestamp must be finite and non-negative")
        frames.append(
            {
                "path": str(path),
                "timestamp": timestamp,
                "file_sha256": _file_sha256(path),
            }
        )
    return frames


def _message_media_paths(messages: Sequence[Mapping[str, Any]]) -> tuple[str, ...]:
    paths: list[str] = []
    for message in messages:
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for item in content:
            if not isinstance(item, Mapping) or item.get("type") != "image_url":
                continue
            image = item.get("image_url")
            url = image.get("url") if isinstance(image, Mapping) else image
            if not isinstance(url, str) or not url:
                raise ValueError("frozen role media item has no URL")
            if url.startswith("file:"):
                parsed = urlparse(url)
                path = Path(url2pathname(unquote(parsed.path)))
                if parsed.netloc:
                    path = Path(f"//{parsed.netloc}{url2pathname(unquote(parsed.path))}")
            else:
                path = Path(url)
            paths.append(str(path.resolve()))
    return tuple(paths)


def _terminal_trace(
    trace: Sequence[Any], stages: set[str], step_index: int | None = None
) -> Mapping[str, Any] | None:
    attempts: list[tuple[int, int, Mapping[str, Any]]] = []
    for position, raw in enumerate(trace):
        if not isinstance(raw, Mapping):
            raise ValueError("request_trace entries must be objects")
        if str(raw.get("stage") or "").casefold() not in stages:
            continue
        if step_index is not None and raw.get("step_index") != step_index:
            continue
        attempt = raw.get("attempt_index", 0)
        if isinstance(attempt, bool) or not isinstance(attempt, int) or attempt < 0:
            raise ValueError("request trace attempt_index must be non-negative")
        attempts.append((attempt, position, raw))
    if not attempts:
        return None
    terminal = max(attempts, key=lambda item: (item[0], item[1]))[2]
    if terminal.get("finish_reason") == "length" or any(
        terminal.get(field) for field in ("error", "error_type", "attempt_error")
    ):
        raise ValueError("frozen role source has no successful terminal request")
    return terminal


def _observer_terminal_messages(
    trace: Sequence[Any],
    *,
    step_index: int,
    canonical_messages: Sequence[Mapping[str, Any]],
    valid_letters: Sequence[str],
    frame_count: int,
    interval: tuple[float, float],
) -> tuple[Mapping[str, Any], list[dict[str, Any]], dict[str, Any] | None]:
    """Validate and reproduce the only supported same-frame Observer retry."""

    matching: list[Mapping[str, Any]] = []
    for raw in trace:
        if not isinstance(raw, Mapping):
            raise ValueError("request_trace entries must be objects")
        if (
            str(raw.get("stage") or "").casefold() in _OBSERVER_STAGES
            and raw.get("step_index") == step_index
        ):
            matching.append(raw)
    terminal = _terminal_trace(trace, _OBSERVER_STAGES, step_index)
    if terminal is None:
        raise ValueError(f"source has no observer request for step {step_index}")
    terminal_attempt = terminal.get("attempt_index", 0)
    if terminal_attempt == 0:
        expected = copy.deepcopy(list(canonical_messages))
        if terminal.get("messages") != expected:
            raise ValueError(
                "frozen Observer request does not match the canonical current-frame prompt"
            )
        return terminal, expected, None
    if terminal_attempt != 1:
        raise ValueError("frozen Observer supports only one bounded retry")

    attempt_zero = [item for item in matching if item.get("attempt_index", 0) == 0]
    attempt_one = [item for item in matching if item.get("attempt_index", 0) == 1]
    if len(attempt_zero) != 1 or len(attempt_one) != 1:
        raise ValueError("frozen Observer retry requires exactly one prior attempt 0")
    prior = attempt_zero[0]
    if prior.get("stage") != terminal.get("stage"):
        raise ValueError("frozen Observer retry changed request stage")
    if prior.get("messages") != list(canonical_messages):
        raise ValueError(
            "frozen Observer retry attempt 0 is not the canonical current-frame prompt"
        )

    reason = prior.get("attempt_error")
    if (
        reason not in _OBSERVER_RETRY_REASONS
        or prior.get("retry_reason") != reason
        or prior.get("retry_triggered") is not True
        or prior.get("retry_of_attempt") is not None
        or terminal.get("retry_reason") != reason
        or terminal.get("retry_of_attempt") != 0
        or terminal.get("retry_triggered") is not False
    ):
        raise ValueError("frozen Observer retry audit is inconsistent")
    if (reason == "finish_reason_length") != (prior.get("finish_reason") == "length"):
        raise ValueError("frozen Observer retry failure reason is inconsistent")

    retry_group_id = canonical_sha256(canonical_messages)
    if (
        prior.get("retry_group_id") != retry_group_id
        or terminal.get("retry_group_id") != retry_group_id
        or prior.get("prompt_hash") != retry_group_id
        or prior.get("failure_class") != "model_parse_failure"
    ):
        raise ValueError("frozen Observer retry group audit is inconsistent")
    expected = build_perception_retry_messages(
        canonical_messages,
        reason,
        valid_letters=valid_letters,
        frame_count=frame_count,
        interval=interval,
        role_separated=True,
    )
    terminal_prompt_sha256 = canonical_sha256(expected)
    if (
        terminal.get("messages") != expected
        or terminal.get("prompt_hash") != terminal_prompt_sha256
    ):
        raise ValueError(
            "frozen Observer retry does not match the canonical correction prompt"
        )
    retry_contract = {
        "attempt_index": 1,
        "retry_of_attempt": 0,
        "retry_reason": reason,
        "retry_group_id": retry_group_id,
        "base_messages_sha256": retry_group_id,
        "terminal_messages_sha256": terminal_prompt_sha256,
    }
    return terminal, expected, retry_contract


def _latest_terminal_trace(
    trace: Sequence[Any], stages: set[str]
) -> Mapping[str, Any] | None:
    """Return the chronologically final terminal call across distinct stages."""

    matching: list[Mapping[str, Any]] = []
    for raw in trace:
        if not isinstance(raw, Mapping):
            raise ValueError("request_trace entries must be objects")
        if str(raw.get("stage") or "").casefold() in stages:
            matching.append(raw)
    if not matching:
        return None
    terminal = matching[-1]
    if terminal.get("finish_reason") == "length" or any(
        terminal.get(field) for field in ("error", "error_type", "attempt_error")
    ):
        raise ValueError("frozen role source has no successful terminal request")
    return terminal


def _request_contract(
    request: Mapping[str, Any],
    *,
    fallback_max_tokens: int,
    fallback_response_format: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    messages = request.get("messages")
    if not isinstance(messages, list) or not messages:
        raise ValueError("frozen role request requires messages")
    assert_annotation_free_request({"messages": messages})
    max_tokens = request.get("max_tokens", fallback_max_tokens)
    seed = request.get("seed")
    if (
        isinstance(max_tokens, bool)
        or not isinstance(max_tokens, int)
        or max_tokens <= 0
        or isinstance(seed, bool)
        or not isinstance(seed, int)
    ):
        raise ValueError("frozen role request has invalid generation parameters")
    response_format = request.get("response_format", fallback_response_format)
    contract = {
        "messages": copy.deepcopy(messages),
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "seed": seed,
        "response_format": copy.deepcopy(response_format),
        "chat_template_kwargs": {"enable_thinking": False},
        "extra_body": {
            "return_token_ids": True,
            **({"tool_choice": "none"} if request.get("tools_disabled") else {}),
        },
    }
    contract["request_sha256"] = canonical_sha256(contract)
    return contract


def _assert_request_public(request: Mapping[str, Any]) -> None:
    assert_annotation_free_request({"messages": request.get("messages")})
    if request.get("response_format") is not None:
        assert_annotation_free_request(
            {
                "request_kwargs": {
                    "response_format": request.get("response_format")
                }
            }
        )


def _deduplicated_inventory(states: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    inventory: list[dict[str, Any]] = []
    seen: set[tuple[str, float]] = set()
    for state in states:
        for frame in _frame_records(state["frame_paths"], state["timestamps"]):
            key = (frame["path"], frame["timestamp"])
            if key not in seen:
                seen.add(key)
                inventory.append(frame)
    return inventory


def _inventory_pairs(frames: Sequence[Mapping[str, Any]]) -> list[tuple[str, float]]:
    return [(str(frame["path"]), float(frame["timestamp"])) for frame in frames]


def _canonical_observer_messages(
    sample: ModelSample,
    state: Mapping[str, Any],
    frames: Sequence[Mapping[str, Any]],
    interval: Sequence[float],
) -> list[dict[str, Any]]:
    raw_request = state.get("request")
    if not isinstance(raw_request, Mapping):
        raise ValueError("frozen Observer state requires its frame request")
    evidence_request = raw_request.get("evidence_request")
    if not isinstance(evidence_request, str) or not evidence_request.strip():
        raise ValueError("frozen Observer state requires evidence_request")
    has_nframes = "nframes" in raw_request
    has_fps = "fps" in raw_request
    expected_keys = {
        "start_time",
        "end_time",
        "resize",
        "evidence_request",
        "nframes" if has_nframes else "fps",
    }
    if has_nframes == has_fps or set(raw_request) != expected_keys:
        raise ValueError("frozen Observer state has a non-canonical frame request")
    try:
        request = FrameRequest(
            start_time=float(raw_request["start_time"]),
            end_time=float(raw_request["end_time"]),
            resize=float(raw_request["resize"]),
            nframes=(int(raw_request["nframes"]) if has_nframes else None),
            fps=(float(raw_request["fps"]) if has_fps else None),
            evidence_request=evidence_request,
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("frozen Observer state has an invalid frame request") from exc
    observation = FrameObservation(
        request=request,
        resolved_start_time=float(interval[0]),
        resolved_end_time=float(interval[1]),
        resolved_nframes=len(frames),
        frame_paths=tuple(str(frame["path"]) for frame in frames),
        timestamps=tuple(float(frame["timestamp"]) for frame in frames),
        backend="frozen_role_ablation",
        cache_hit=True,
        estimated_visual_tokens=0,
        latency_s=0.0,
    )
    return build_perception_messages(
        sample,
        observation,
        evidence_request,
        use_frame_indices=True,
        role_separated=True,
    )


def freeze_role_ablation_input(source: Mapping[str, Any]) -> dict[str, Any]:
    """Freeze one public, label-free input bundle from a Base runtime result."""

    if source.get("annotation_leak_check") != "passed":
        raise ValueError("source result failed annotation leak audit")
    if int(source.get("candidate_rerun") or 0) != 0:
        raise ValueError("source result reran the frozen candidate")
    sample = _public_sample(source.get("public_sample"))
    if (str(source.get("dataset")), str(source.get("sample_id"))) != (
        sample.dataset,
        sample.sample_id,
    ):
        raise ValueError("source identity differs from public_sample")
    states = source.get("perception_states")
    trace = source.get("request_trace")
    if not isinstance(states, list) or not states or not isinstance(trace, list):
        raise ValueError("source requires perception states and request trace")

    observer_calls: list[dict[str, Any]] = []
    for step_index, raw_state in enumerate(states):
        if not isinstance(raw_state, Mapping) or raw_state.get("step_index", step_index) != step_index:
            raise ValueError("source perception states must be contiguous")
        frames = _frame_records(raw_state.get("frame_paths"), raw_state.get("timestamps"))
        response = raw_state.get("perception_response")
        interval = response.get("interval") if isinstance(response, Mapping) else None
        if (
            not isinstance(interval, (list, tuple))
            or len(interval) != 2
            or any(
                isinstance(item, bool) or not isinstance(item, (int, float))
                for item in interval
            )
        ):
            raise ValueError("frozen Observer state has no resolved interval")
        expected_messages = _canonical_observer_messages(
            sample, raw_state, frames, interval
        )
        terminal, expected_terminal_messages, retry_contract = (
            _observer_terminal_messages(
                trace,
                step_index=step_index,
                canonical_messages=expected_messages,
                valid_letters=sample.option_letters,
                frame_count=len(frames),
                interval=(float(interval[0]), float(interval[1])),
            )
        )
        if terminal.get("messages") != expected_terminal_messages:
            raise AssertionError("validated Observer terminal request changed")
        contract = _request_contract(
            terminal,
            fallback_max_tokens=int(source.get("perception_max_tokens") or 1024),
            fallback_response_format=perception_response_format(
                sample.option_letters, len(frames), role_separated=True
            ),
        )
        expected_paths = tuple(frame["path"] for frame in frames)
        if _message_media_paths(contract["messages"]) != expected_paths:
            raise ValueError("frozen Observer request media differs from state frames")
        observer_calls.append(
            {
                "step_index": step_index,
                "frames": frames,
                "actual_timestamps": [frame["timestamp"] for frame in frames],
                "resolved_interval": [float(interval[0]), float(interval[1])],
                "source_retry_contract": retry_contract,
                "request": contract,
            }
        )

    inventory = _deduplicated_inventory(states)
    verifier_calls: list[dict[str, Any]] = []
    cumulative_count = 0
    cumulative: list[dict[str, Any]] = []
    seen_frames: set[tuple[str, float]] = set()
    for step_index, state in enumerate(states):
        for frame in _frame_records(state["frame_paths"], state["timestamps"]):
            key = (frame["path"], frame["timestamp"])
            if key not in seen_frames:
                seen_frames.add(key)
                cumulative.append(frame)
        messages = build_runtime_visual_csv_messages(
            sample, _inventory_pairs(cumulative), prefix_index=step_index
        )
        request = {
            "messages": messages,
            "max_tokens": 256,
            "temperature": 0.0,
            "seed": 42,
            "response_format": visual_csv_response_format(sample.option_letters),
            "chat_template_kwargs": {"enable_thinking": False},
            "extra_body": {"return_token_ids": True, "tool_choice": "none"},
        }
        request["request_sha256"] = canonical_sha256(request)
        cumulative_count = len(cumulative)
        verifier_calls.append(
            {
                "prefix_index": step_index,
                "frames": copy.deepcopy(cumulative),
                "request": request,
            }
        )
    if cumulative_count != len(inventory):
        raise AssertionError("frozen verifier inventory is inconsistent")

    answerer: dict[str, Any] | None = None
    terminal_answerer = _latest_terminal_trace(trace, _ANSWERER_STAGES)
    if terminal_answerer is not None:
        final_memory = terminal_answerer.get("evidence_memory_at_stage")
        if final_memory is None:
            final_memory = states[-1].get("memory_after")
        if not isinstance(final_memory, Mapping):
            raise ValueError("frozen Answerer requires final memory_after")
        expected_memory_sha256 = terminal_answerer.get("evidence_memory_sha256")
        if (
            expected_memory_sha256 is not None
            and expected_memory_sha256 != canonical_sha256(final_memory)
        ):
            raise ValueError("frozen Answerer trace ledger SHA-256 changed")
        evidence_ids = [
            str(item.get("evidence_id", item.get("id", "")))
            for item in final_memory.get("event_ledger", [])
            if isinstance(item, Mapping)
        ]
        if not evidence_ids or any(not item for item in evidence_ids):
            raise ValueError("frozen Answerer memory has invalid evidence IDs")
        answerer_request = _request_contract(
            terminal_answerer,
            fallback_max_tokens=int(source.get("judge_max_tokens") or 512),
        )
        answerer_stage = str(terminal_answerer.get("stage") or "").casefold()
        decisive_field = (
            "confirmation_decisive_frame_indices"
            if answerer_stage == "confirmation_judge"
            else "decisive_frame_indices"
        )
        answerer = {
            "memory": copy.deepcopy(dict(final_memory)),
            "memory_sha256": canonical_sha256(final_memory),
            "valid_evidence_ids": evidence_ids,
            "frames": copy.deepcopy(inventory),
            "cited_frame_indices": list(
                terminal_answerer.get(
                    "cited_frame_indices", source.get(decisive_field) or []
                )
            ),
            "source_stage": answerer_stage,
            "request": answerer_request,
        }
        indices = answerer["cited_frame_indices"]
        if (
            not indices
            or len(indices) > 16
            or any(
                isinstance(index, bool)
                or not isinstance(index, int)
                or index < 0
                or index >= len(inventory)
                for index in indices
            )
        ):
            raise ValueError("frozen Answerer has invalid cited frame indices")
        expected_paths = tuple(inventory[index]["path"] for index in dict.fromkeys(indices))
        if _message_media_paths(answerer["request"]["messages"]) != expected_paths:
            raise ValueError("frozen Answerer request media differs from cited frames")
        expected_messages = build_cited_judge_messages(
            sample,
            EvidenceMemory.from_dict(final_memory, sample.option_letters),
            _inventory_pairs(inventory),
            indices,
        )
        if answerer["request"]["messages"] != expected_messages:
            raise ValueError(
                "frozen Answerer request differs from its stage ledger and cited frames"
            )

    if answerer is not None:
        fixed_answerer_call: dict[str, Any] | None = answerer
    else:
        fixed_answerer_call = None

    public_sample = {
        "dataset": sample.dataset,
        "sample_id": sample.sample_id,
        "video": sample.video,
        "question": sample.question,
        "choices": dict(sample.choices),
    }
    payload = {
        "schema_version": FROZEN_ROLE_INPUT_VERSION,
        "dataset": sample.dataset,
        "sample_id": sample.sample_id,
        "public_sample": public_sample,
        "source_run_fingerprint": _lower_sha(
            source.get("run_fingerprint"), "source_run_fingerprint"
        ),
        "observer_calls": observer_calls,
        "verifier_calls": verifier_calls,
        "answerer_call": fixed_answerer_call,
        "offline_scoring_required": True,
        "labels_serialized": False,
    }
    payload["frozen_input_sha256"] = canonical_sha256(payload)
    return payload


def freeze_role_ablation_inputs(
    sources: Iterable[Mapping[str, Any]],
) -> tuple[dict[str, Any], ...]:
    rows = [freeze_role_ablation_input(row) for row in sources]
    identities = [(row["dataset"], row["sample_id"]) for row in rows]
    if len(identities) != len(set(identities)):
        raise ValueError("frozen role inputs contain duplicate samples")
    return tuple(sorted(rows, key=lambda row: (row["dataset"], row["sample_id"])))


def validate_role_ablation_dev30_scope(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, int]:
    """Require the pre-registered three-dataset, ten-sample Dev30 scope."""
    identities: list[tuple[str, str]] = []
    counts = {dataset: 0 for dataset in ROLE_ABLATION_DEV30_COUNTS}
    for row in rows:
        dataset = str(row.get("dataset") or "").strip()
        sample_id = str(row.get("sample_id") or "").strip()
        if dataset not in counts or not sample_id:
            raise ValueError("role ablation input is outside the frozen Dev30 scope")
        identities.append((dataset, sample_id))
        counts[dataset] += 1
    if len(identities) != len(set(identities)):
        raise ValueError("role ablation Dev30 contains duplicate samples")
    if counts != ROLE_ABLATION_DEV30_COUNTS:
        raise ValueError(
            "role ablation requires exactly ten samples from each Dev dataset"
        )
    return counts


def validate_frozen_role_input(row: Mapping[str, Any]) -> None:
    if row.get("schema_version") != FROZEN_ROLE_INPUT_VERSION:
        raise ValueError("frozen role input schema version changed")
    if row.get("labels_serialized") is not False or row.get("offline_scoring_required") is not True:
        raise ValueError("frozen role input must remain label-free")
    expected_sha = str(row.get("frozen_input_sha256") or "")
    payload = {key: copy.deepcopy(value) for key, value in row.items() if key != "frozen_input_sha256"}
    if expected_sha != canonical_sha256(payload):
        raise ValueError("frozen role input SHA-256 changed")
    _public_sample(row.get("public_sample"))
    for role_field in ("observer_calls", "verifier_calls"):
        calls = row.get(role_field)
        if not isinstance(calls, list) or not calls:
            raise ValueError(f"frozen role input requires {role_field}")
        for call in calls:
            for frame in call.get("frames", []):
                path = Path(str(frame.get("path"))).resolve()
                if _file_sha256(path) != frame.get("file_sha256"):
                    raise ValueError("frozen role frame content SHA-256 changed")
            request = call.get("request")
            if not isinstance(request, Mapping):
                raise ValueError("frozen role call requires request")
            material = {key: copy.deepcopy(value) for key, value in request.items() if key != "request_sha256"}
            if request.get("request_sha256") != canonical_sha256(material):
                raise ValueError("frozen role request SHA-256 changed")
            _assert_request_public(material)
    answerer = row.get("answerer_call")
    if answerer is not None:
        if not isinstance(answerer, Mapping):
            raise ValueError("answerer_call must be null or an object")
        if answerer.get("memory_sha256") != canonical_sha256(answerer.get("memory")):
            raise ValueError("frozen Answerer ledger SHA-256 changed")
        for frame in answerer.get("frames", []):
            if _file_sha256(Path(str(frame.get("path"))).resolve()) != frame.get("file_sha256"):
                raise ValueError("frozen Answerer frame content SHA-256 changed")
        request = answerer.get("request")
        material = {key: copy.deepcopy(value) for key, value in request.items() if key != "request_sha256"}
        if request.get("request_sha256") != canonical_sha256(material):
            raise ValueError("frozen Answerer request SHA-256 changed")
        _assert_request_public(material)


def validate_paired_role_result(
    result: Mapping[str, Any], frozen: Mapping[str, Any]
) -> None:
    """Fail closed on missing/extra calls or either-arm engineering failures."""

    if result.get("schema_version") != PAIRED_ROLE_RESULT_VERSION:
        raise ValueError("paired role result schema version changed")
    if result.get("frozen_input_sha256") != frozen.get("frozen_input_sha256"):
        raise ValueError("paired role result uses a different frozen input")
    if not str(result.get("materialized_run_id") or "").strip():
        raise ValueError("paired role result lacks its materialized run ID")
    _lower_sha(result.get("materialized_run_sha256"), "materialized_run_sha256")
    role = str(result.get("role") or "")
    expected_calls = (
        len(frozen["observer_calls"])
        if role == "observer"
        else len(frozen["verifier_calls"])
        if role == "verifier"
        else 1
        if role == "answerer" and isinstance(frozen.get("answerer_call"), Mapping)
        else 0
    )
    calls = result.get("paired_calls")
    if (
        role not in PAIRED_ROLES
        or not isinstance(calls, list)
        or len(calls) != expected_calls
        or result.get("call_count") != expected_calls
    ):
        raise ValueError("paired role result call coverage differs from frozen input")
    not_applicable_reason = result.get("not_applicable_reason")
    if role == "answerer" and expected_calls == 0:
        if not_applicable_reason != "base_runtime_did_not_reach_answerer":
            raise ValueError("missing frozen Answerer call must be explicitly not applicable")
    elif not_applicable_reason is not None:
        raise ValueError("paired role result has a spurious not-applicable reason")
    for index, call in enumerate(calls):
        if call.get("call_index") != index:
            raise ValueError("paired role call indices must be contiguous")
        control = call.get("control")
        treatment = call.get("treatment")
        if not isinstance(control, Mapping) or not isinstance(treatment, Mapping):
            raise ValueError("paired role result is missing one arm")
        if (
            control.get("request_sha256") != call.get("request_sha256")
            or treatment.get("request_sha256") != call.get("request_sha256")
        ):
            raise ValueError("paired role arms did not use the same request")
        for arm_name, arm in (("control", control), ("treatment", treatment)):
            if arm.get("finish_reason") == "length" or arm.get("parsed_valid") is not True:
                raise ValueError(
                    f"paired role {arm_name} arm has no valid terminal response"
                )
    downstream = result.get("conditioned_downstream")
    if downstream is not None and (
        not isinstance(downstream, Mapping)
        or downstream.get("parsed_valid") is not True
    ):
        raise ValueError("paired Observer conditioned downstream evaluation failed")
    payload = {
        key: copy.deepcopy(value) for key, value in result.items() if key != "result_sha256"
    }
    if result.get("result_sha256") != canonical_sha256(payload):
        raise ValueError("paired role result SHA-256 changed")


def _call(client: PairedRoleClient, model: str, request: Mapping[str, Any]) -> ChatResult:
    return client.chat(
        model,
        copy.deepcopy(request["messages"]),
        max_tokens=int(request["max_tokens"]),
        temperature=float(request["temperature"]),
        seed=int(request["seed"]),
        response_format=copy.deepcopy(request.get("response_format")),
        chat_template_kwargs=copy.deepcopy(request.get("chat_template_kwargs")),
        extra_body=copy.deepcopy(request.get("extra_body")),
    )


def _arm_result(
    *,
    role: str,
    call: Mapping[str, Any],
    sample: ModelSample,
    client: PairedRoleClient,
    model: str,
    artifact_sha256: str,
) -> dict[str, Any]:
    request = call["request"]
    result = _call(client, model, request)
    parsed_valid = False
    parsed: dict[str, Any] | None = None
    if role == "observer":
        state, reference_mode = bind_perception_state(
            result.content,
            sample.option_letters,
            call["actual_timestamps"],
            allow_timestamp_schema=False,
            allow_role_separated_schema=True,
        )
        parsed_valid = state is not None and reference_mode == "frame_index"
        if state is not None and tuple(state.interval) != tuple(call["resolved_interval"]):
            parsed_valid = False
        parsed = state.to_dict() if state is not None else None
    elif role == "verifier":
        decision = parse_visual_csv_response(
            result.content, sample.option_letters, len(call["frames"])
        )
        parsed_valid = decision is not None
        if decision is not None:
            parsed = {
                "answer": decision.answer,
                "frame_indices": list(decision.frame_indices),
                "evidence_complete": decision.evidence_complete,
                "missing_evidence": list(decision.missing_evidence),
            }
    else:
        decision = parse_evidence_decision(
            result.content,
            sample.option_letters,
            call["valid_evidence_ids"],
        )
        parsed_valid = decision is not None
        if decision is not None:
            parsed = {"answer": decision.answer, "evidence_ids": list(decision.evidence_ids)}
    return {
        "model": model,
        "artifact_sha256": _lower_sha(artifact_sha256, "artifact_sha256"),
        "request_sha256": request["request_sha256"],
        "content": result.content,
        "finish_reason": result.finish_reason,
        "usage": copy.deepcopy(result.usage),
        "latency_s": result.latency_s,
        "parsed_valid": parsed_valid,
        "parsed": parsed,
    }


def _observer_memory(
    paired_calls: Sequence[Mapping[str, Any]],
    arm: str,
    calls: Sequence[Mapping[str, Any]],
    sample: ModelSample,
) -> EvidenceMemory | None:
    memory = EvidenceMemory(sample.option_letters)
    for paired, call in zip(paired_calls, calls, strict=True):
        result = paired[arm]
        state, reference_mode = bind_perception_state(
            str(result.get("content") or ""),
            sample.option_letters,
            call["actual_timestamps"],
            allow_timestamp_schema=False,
            allow_role_separated_schema=True,
        )
        if state is None or reference_mode != "frame_index":
            return None
        if tuple(state.interval) != tuple(call["resolved_interval"]):
            return None
        memory.merge(state)
    return memory


def _condition_observer_with_base_downstream(
    *,
    frozen: Mapping[str, Any],
    sample: ModelSample,
    paired_calls: Sequence[Mapping[str, Any]],
    verifier_client: PairedRoleClient,
    verifier_model: str,
    verifier_artifact_sha256: str,
    answerer_client: PairedRoleClient,
    answerer_model: str,
    answerer_artifact_sha256: str,
) -> dict[str, Any]:
    observer_calls = list(frozen["observer_calls"])
    control_memory = _observer_memory(paired_calls, "control", observer_calls, sample)
    treatment_memory = _observer_memory(
        paired_calls, "treatment", observer_calls, sample
    )
    if control_memory is None or treatment_memory is None:
        return {
            "parsed_valid": False,
            "error": "observer output could not be bound to frozen frames",
        }
    verifier_call = frozen["verifier_calls"][-1]
    verifier = _arm_result(
        role="verifier",
        call=verifier_call,
        sample=sample,
        client=verifier_client,
        model=verifier_model,
        artifact_sha256=verifier_artifact_sha256,
    )
    parsed = verifier.get("parsed")
    if not verifier.get("parsed_valid") or not isinstance(parsed, Mapping):
        return {
            "parsed_valid": False,
            "error": "frozen Base Verifier response was invalid",
            "verifier": verifier,
        }
    cited = parsed.get("frame_indices")
    inventory = verifier_call["frames"]
    if not isinstance(cited, list) or not cited:
        return {
            "parsed_valid": False,
            "error": "frozen Base Verifier cited no frames",
            "verifier": verifier,
        }

    answerer_results: dict[str, Any] = {}
    for arm, memory in (
        ("control", control_memory),
        ("treatment", treatment_memory),
    ):
        messages = build_cited_judge_messages(
            sample,
            memory,
            _inventory_pairs(inventory),
            cited,
        )
        request = {
            "messages": messages,
            "max_tokens": 512,
            "temperature": 0.0,
            "seed": 42,
            "response_format": {"type": "json_object"},
            "chat_template_kwargs": {"enable_thinking": False},
            "extra_body": {"return_token_ids": True, "tool_choice": "none"},
        }
        request["request_sha256"] = canonical_sha256(request)
        result = _call(answerer_client, answerer_model, request)
        decision = parse_evidence_decision(
            result.content, sample.option_letters, memory.evidence_ids
        )
        answerer_results[arm] = {
            "model": answerer_model,
            "artifact_sha256": _lower_sha(
                answerer_artifact_sha256, "answerer_artifact_sha256"
            ),
            "request_sha256": request["request_sha256"],
            "memory_sha256": canonical_sha256(memory.to_dict()),
            "content": result.content,
            "finish_reason": result.finish_reason,
            "usage": copy.deepcopy(result.usage),
            "latency_s": result.latency_s,
            "parsed_valid": decision is not None,
            "parsed": (
                {
                    "answer": decision.answer,
                    "evidence_ids": list(decision.evidence_ids),
                }
                if decision is not None
                else None
            ),
        }
    return {
        "parsed_valid": all(
            item["parsed_valid"] for item in answerer_results.values()
        ),
        "verifier": verifier,
        "answerer": answerer_results,
        "condition": (
            "same_frozen_frame_schedule_and_base_verifier_answerer; "
            "only Observer-derived ledger differs"
        ),
    }


def run_paired_role_ablation(
    frozen: Mapping[str, Any],
    *,
    role: str,
    control_client: PairedRoleClient,
    control_model: str,
    control_artifact_sha256: str,
    treatment_client: PairedRoleClient,
    treatment_model: str,
    treatment_artifact_sha256: str,
    materialized_run_id: str,
    materialized_run_sha256: str,
    downstream_verifier_client: PairedRoleClient | None = None,
    downstream_verifier_model: str | None = None,
    downstream_verifier_artifact_sha256: str | None = None,
    downstream_answerer_client: PairedRoleClient | None = None,
    downstream_answerer_model: str | None = None,
    downstream_answerer_artifact_sha256: str | None = None,
) -> dict[str, Any]:
    """Run both arms over identical frozen calls; never joins benchmark labels."""

    if role not in PAIRED_ROLES:
        raise ValueError(f"paired role must be one of {PAIRED_ROLES}")
    run_id = str(materialized_run_id or "").strip()
    if not run_id:
        raise ValueError("materialized_run_id must be non-empty")
    run_sha256 = _lower_sha(
        materialized_run_sha256, "materialized_run_sha256"
    )
    validate_frozen_role_input(frozen)
    sample = _public_sample(frozen["public_sample"])
    calls: list[Mapping[str, Any]]
    if role == "observer":
        calls = list(frozen["observer_calls"])
    elif role == "verifier":
        calls = list(frozen["verifier_calls"])
    else:
        answerer = frozen.get("answerer_call")
        calls = [answerer] if isinstance(answerer, Mapping) else []
    paired_calls: list[dict[str, Any]] = []
    for index, call in enumerate(calls):
        control = _arm_result(
            role=role,
            call=call,
            sample=sample,
            client=control_client,
            model=control_model,
            artifact_sha256=control_artifact_sha256,
        )
        treatment = _arm_result(
            role=role,
            call=call,
            sample=sample,
            client=treatment_client,
            model=treatment_model,
            artifact_sha256=treatment_artifact_sha256,
        )
        if control["request_sha256"] != treatment["request_sha256"]:
            raise AssertionError("paired role arms received different requests")
        paired_calls.append(
            {
                "call_index": index,
                "request_sha256": control["request_sha256"],
                "control": control,
                "treatment": treatment,
            }
        )
    conditioned_downstream: dict[str, Any] | None = None
    if role == "observer":
        downstream = (
            downstream_verifier_client,
            downstream_verifier_model,
            downstream_verifier_artifact_sha256,
            downstream_answerer_client,
            downstream_answerer_model,
            downstream_answerer_artifact_sha256,
        )
        if any(item is not None for item in downstream) and any(
            item is None for item in downstream
        ):
            raise ValueError("Observer conditioned downstream requires all Base bindings")
        if all(item is not None for item in downstream):
            conditioned_downstream = _condition_observer_with_base_downstream(
                frozen=frozen,
                sample=sample,
                paired_calls=paired_calls,
                verifier_client=downstream_verifier_client,
                verifier_model=str(downstream_verifier_model),
                verifier_artifact_sha256=str(
                    downstream_verifier_artifact_sha256
                ),
                answerer_client=downstream_answerer_client,
                answerer_model=str(downstream_answerer_model),
                answerer_artifact_sha256=str(
                    downstream_answerer_artifact_sha256
                ),
            )
    result = {
        "schema_version": PAIRED_ROLE_RESULT_VERSION,
        "dataset": frozen["dataset"],
        "sample_id": frozen["sample_id"],
        "role": role,
        "materialized_run_id": run_id,
        "materialized_run_sha256": run_sha256,
        "frozen_input_sha256": frozen["frozen_input_sha256"],
        "paired_calls": paired_calls,
        "call_count": len(paired_calls),
        "not_applicable_reason": (
            "base_runtime_did_not_reach_answerer"
            if role == "answerer" and not paired_calls
            else None
        ),
        "conditioned_downstream": conditioned_downstream,
        "offline_scoring_required": True,
        "labels_serialized": False,
    }
    result["result_sha256"] = canonical_sha256(result)
    validate_paired_role_result(result, frozen)
    return result


__all__ = [
    "FROZEN_ROLE_INPUT_VERSION",
    "PAIRED_ROLE_RESULT_VERSION",
    "PAIRED_ROLES",
    "freeze_role_ablation_input",
    "freeze_role_ablation_inputs",
    "run_paired_role_ablation",
    "validate_frozen_role_input",
    "validate_paired_role_result",
]
