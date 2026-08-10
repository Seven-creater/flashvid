"""Build process-SFT episodes from Perception-Memory EVA trajectories.

The exporter deliberately remains a thin adapter around :mod:`qwen_sft`:
every model request becomes one loss-isolated ms-swift record, while user,
tool, media, and historical assistant turns stay masked by the shared builder.
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from copy import deepcopy
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse
from urllib.request import url2pathname

from .perception_memory_eva import (
    PERCEPTION_NORMALIZATION_VERSION,
    bind_perception_state,
    normalize_perception_state,
)
from .qwen_sft import (
    build_sft_record,
    canonical_sha256,
    validate_exported_sft_record,
)


_PRIVATE_KEYS = frozenset(
    {
        "answer",
        "correct_answer",
        "right_answer",
        "ground_truth",
        "gt",
        "time_range",
        "clue_intervals",
        "question_type",
    }
)
_PRIVATE_SENTINELS = ("ANNOTATION_SENTINEL", "GROUND_TRUTH_SENTINEL")
_CANDIDATE_MARKERS = ("direct candidate", "candidate_answer", "candidate answer")
_ERROR_FIELDS = ("error", "error_type", "api_error", "frame_error", "parse_error")
_TOOL_CALL_RE = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)
_ANSWER_RE = re.compile(r"(?:Answer:\s*)?([A-H])", re.IGNORECASE)

_OBSERVATION_STAGES = frozenset(
    {"perception", "observation", "observer", "confirmation_perception"}
)
_CONTROLLER_STAGES = frozenset({"controller", "planner", "confirmation_controller"})
_PRIMARY_CONTROLLER_STAGES = frozenset({"controller", "planner"})
_FINAL_STAGES = frozenset({"evidence_judge", "final_judge", "judge", "final"})


def _normalized_key(value: Any) -> str:
    return str(value).strip().casefold().replace(" ", "_")


def _private_path(value: Any, path: str = "$") -> str | None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            child = f"{path}.{key}"
            if _normalized_key(key) in _PRIVATE_KEYS:
                return child
            found = _private_path(item, child)
            if found:
                return found
    elif isinstance(value, list):
        for index, item in enumerate(value):
            found = _private_path(item, f"{path}[{index}]")
            if found:
                return found
    elif isinstance(value, str) and any(
        marker in value for marker in _PRIVATE_SENTINELS
    ):
        return path
    return None


def _assert_public_messages(messages: Any, *, candidate_blind: bool) -> None:
    private = _private_path(messages, "$.messages")
    if private:
        raise ValueError(f"private annotation in process-SFT messages at {private}")
    if candidate_blind:
        serialized = json.dumps(messages, ensure_ascii=False).casefold()
        marker = next((item for item in _CANDIDATE_MARKERS if item in serialized), None)
        if marker:
            raise ValueError(
                f"candidate leaked into candidate-blind messages: {marker}"
            )


def _request_messages(request: Mapping[str, Any]) -> list[dict[str, Any]]:
    raw_messages = request.get("messages", request.get("request_messages"))
    if not isinstance(raw_messages, list) or not raw_messages:
        raise ValueError("process-SFT request has no message snapshot")
    _assert_public_messages(raw_messages, candidate_blind=True)
    messages: list[dict[str, Any]] = []
    for index, raw in enumerate(raw_messages):
        if not isinstance(raw, Mapping):
            raise ValueError(f"request messages[{index}] must be an object")
        role = str(raw.get("role") or "")
        if role not in {"system", "user", "assistant", "tool"}:
            raise ValueError(f"unsupported process-SFT message role: {role!r}")
        message = {"role": role, "content": deepcopy(raw.get("content", ""))}
        for key in ("name", "tool_call_id", "tool_calls", "function_call"):
            if key in raw:
                message[key] = deepcopy(raw[key])
        if role == "assistant":
            message["target_type"] = "context"
        messages.append(message)
    if messages[0]["role"] != "system" or messages[-1]["role"] not in {"user", "tool"}:
        raise ValueError("process-SFT request snapshot has invalid role ordering")
    _assert_public_messages(messages, candidate_blind=True)
    return messages


def _media_paths(messages: Sequence[Mapping[str, Any]]) -> tuple[str, ...]:
    paths: list[str] = []
    for message in messages:
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for item in content:
            if not isinstance(item, Mapping):
                continue
            item_type = str(item.get("type") or "")
            if item_type == "video_url":
                raise ValueError(
                    "Perception-Memory SFT accepts frames, not video inputs"
                )
            if item_type != "image_url":
                continue
            image_url = item.get("image_url")
            url = image_url.get("url") if isinstance(image_url, Mapping) else image_url
            if not isinstance(url, str):
                raise ValueError("image_url must contain a local file URI")
            parsed = urlparse(url)
            if parsed.scheme.casefold() != "file":
                raise ValueError("process-SFT images must use local file:// URIs")
            path = Path(url2pathname(unquote(parsed.path))).resolve()
            paths.append(str(path))
    return tuple(paths)


def _json_object(content: Any, field: str) -> dict[str, Any]:
    if not isinstance(content, str) or not content.strip():
        raise ValueError(f"{field} must be non-empty JSON text")
    text = content.strip()
    if text.startswith("```") and text.endswith("```"):
        lines = text.splitlines()
        if len(lines) < 3 or lines[-1].strip() != "```":
            raise ValueError(f"{field} has malformed JSON fencing")
        text = "\n".join(lines[1:-1]).strip()
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as error:
        raise ValueError(f"{field} is not valid JSON") from error
    if not isinstance(payload, Mapping):
        raise ValueError(f"{field} must be a JSON object")
    return dict(payload)


def _canonical_json(value: Mapping[str, Any]) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _validate_perception_response(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("perception_response must be an object")
    response = deepcopy(dict(value))
    private = _private_path(response, "$.perception_response")
    if private:
        raise ValueError(f"private annotation in perception response at {private}")
    required = {
        "interval",
        "timestamped_facts",
        "option_evidence",
        "temporal_changes",
        "unresolved",
        "evidence_sufficient",
        "next_evidence_needed",
    }
    if set(response) != required:
        raise ValueError(
            "perception_response must contain exactly the frozen observation schema"
        )
    interval = response["interval"]
    if (
        not isinstance(interval, list)
        or len(interval) != 2
        or any(
            isinstance(item, bool) or not isinstance(item, (int, float))
            for item in interval
        )
        or float(interval[1]) <= float(interval[0])
    ):
        raise ValueError(
            "perception_response.interval must be an increasing numeric pair"
        )
    facts = response["timestamped_facts"]
    if not isinstance(facts, list):
        raise ValueError("timestamped_facts must be a list")
    for index, fact in enumerate(facts):
        if not isinstance(fact, Mapping) or set(fact) != {"time", "fact"}:
            raise ValueError(f"timestamped_facts[{index}] has invalid schema")
        timestamp = fact["time"]
        if isinstance(timestamp, bool) or not isinstance(timestamp, (int, float)):
            raise ValueError(f"timestamped_facts[{index}].time must be numeric")
        if not isinstance(fact["fact"], str) or not fact["fact"].strip():
            raise ValueError(f"timestamped_facts[{index}].fact must be non-empty text")
    option_evidence = response["option_evidence"]
    if not isinstance(option_evidence, Mapping):
        raise ValueError("option_evidence must be an object")
    for option, evidence in option_evidence.items():
        if len(str(option)) != 1 or not "A" <= str(option).upper() <= "H":
            raise ValueError("option_evidence keys must be A-H letters")
        if not isinstance(evidence, Mapping) or set(evidence) != {
            "supports",
            "contradicts",
        }:
            raise ValueError(f"option_evidence[{option}] has invalid schema")
        for relation in ("supports", "contradicts"):
            values = evidence[relation]
            if not isinstance(values, list) or any(
                not isinstance(item, str) for item in values
            ):
                raise ValueError(
                    f"option_evidence[{option}].{relation} must be a string list"
                )
    for field in ("temporal_changes", "unresolved"):
        values = response[field]
        if not isinstance(values, list) or any(
            not isinstance(item, str) for item in values
        ):
            raise ValueError(f"perception_response.{field} must be a string list")
    if not isinstance(response["evidence_sufficient"], bool):
        raise ValueError("evidence_sufficient must be boolean")
    if not isinstance(response["next_evidence_needed"], str):
        raise ValueError("next_evidence_needed must be text")
    return response


def _normalize_raw_perception_response(
    value: Any,
    *,
    valid_letters: Sequence[str],
    resolved_start_time: Any,
    resolved_end_time: Any,
    actual_timestamps: Sequence[Any],
) -> dict[str, Any]:
    """Apply the runtime's exact validation/compaction to one raw response."""

    if not isinstance(value, str):
        raise ValueError("perception response must be raw JSON text")
    state, _reference_mode = bind_perception_state(
        value, valid_letters, actual_timestamps
    )
    if state is None:
        raise ValueError("raw perception response failed the frozen parser")
    private = _private_path(state.to_dict(), "$.raw_perception_response")
    if private:
        raise ValueError(f"private annotation in perception response at {private}")
    normalized = normalize_perception_state(
        state,
        resolved_start_time=resolved_start_time,
        resolved_end_time=resolved_end_time,
        actual_timestamps=actual_timestamps,
    )
    return normalized.to_dict()


def _public_option_letters(trajectory: Mapping[str, Any]) -> tuple[str, ...]:
    public = trajectory.get("public_sample")
    choices = public.get("choices") if isinstance(public, Mapping) else None
    if not isinstance(choices, Mapping) or not choices:
        raise ValueError("selected trajectory requires public_sample.choices")
    letters = tuple(str(item).strip().upper() for item in choices)
    if len(letters) != len(set(letters)) or any(
        len(item) != 1 or not "A" <= item <= "H" for item in letters
    ):
        raise ValueError("public_sample.choices has invalid option labels")
    return letters


def _numeric_tuple(value: Any, field: str) -> tuple[float, ...]:
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{field} must be a numeric array")
    result: list[float] = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            raise ValueError(f"{field} must be a numeric array")
        number = float(item)
        if not math.isfinite(number):
            raise ValueError(f"{field} must contain finite numbers")
        result.append(number)
    return tuple(result)


def _tool_state_contract(
    tool_steps: Sequence[Any], state: Mapping[str, Any], step_index: int
) -> tuple[float, float, tuple[float, ...]]:
    if step_index >= len(tool_steps) or not isinstance(tool_steps[step_index], Mapping):
        raise ValueError(f"prefix {step_index} has no corresponding tool_step")
    tool = tool_steps[step_index]
    state_paths = tuple(str(Path(str(item)).resolve()) for item in state["frame_paths"])
    tool_paths_raw = tool.get("frame_paths")
    if not isinstance(tool_paths_raw, (list, tuple)):
        raise ValueError(f"tool_steps[{step_index}].frame_paths must be an array")
    tool_paths = tuple(str(Path(str(item)).resolve()) for item in tool_paths_raw)
    if state_paths != tool_paths:
        raise ValueError(f"prefix {step_index} state/tool frame paths differ")
    state_timestamps = _numeric_tuple(state["timestamps"], "state timestamps")
    tool_timestamps = _numeric_tuple(
        tool.get("actual_timestamps", tool.get("timestamps")),
        f"tool_steps[{step_index}] timestamps",
    )
    if state_timestamps != tool_timestamps or len(tool_timestamps) != len(tool_paths):
        raise ValueError(f"prefix {step_index} state/tool timestamps differ")
    start = tool.get("resolved_start_time", tool.get("start_time"))
    end = tool.get("resolved_end_time", tool.get("end_time"))
    if (
        isinstance(start, bool)
        or not isinstance(start, (int, float))
        or isinstance(end, bool)
        or not isinstance(end, (int, float))
    ):
        raise ValueError(f"tool_steps[{step_index}] has invalid resolved interval")
    return float(start), float(end), tool_timestamps


def _state_memory(state: Mapping[str, Any]) -> dict[str, Any]:
    raw = state.get("memory_after")
    if raw is None:
        raw = {
            "event_ledger": state.get("event_ledger"),
            "option_ledger": state.get("option_ledger"),
            "unresolved": state.get("unresolved"),
        }
    if not isinstance(raw, Mapping):
        raise ValueError("perception state requires memory_after")
    memory = deepcopy(dict(raw))
    if not isinstance(memory.get("event_ledger"), list):
        raise ValueError("memory_after.event_ledger must be a list")
    if not isinstance(memory.get("option_ledger"), Mapping):
        raise ValueError("memory_after.option_ledger must be an object")
    if not isinstance(memory.get("unresolved"), list):
        raise ValueError("memory_after.unresolved must be a list")
    private = _private_path(memory, "$.memory_after")
    if private:
        raise ValueError(f"private annotation in evidence memory at {private}")
    return memory


def _terminal_requests(
    trace: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], ...]:
    groups: dict[tuple[str, int, int, str, str], list[tuple[int, dict[str, Any]]]] = {}
    for index, raw in enumerate(trace):
        request = dict(raw)
        step_index = request.get("step_index", request.get("prefix_index"))
        if (
            isinstance(step_index, bool)
            or not isinstance(step_index, int)
            or step_index < 0
        ):
            raise ValueError(f"request_trace[{index}] has invalid step_index")
        prefix_index = request.get("prefix_index")
        if (
            isinstance(prefix_index, bool)
            or not isinstance(prefix_index, int)
            or prefix_index < -1
        ):
            raise ValueError(f"request_trace[{index}] has invalid prefix_index")
        messages = request.get("messages", request.get("request_messages"))
        prompt_hash = str(request.get("prompt_hash") or canonical_sha256(messages))
        retry_group = str(request.get("retry_group_id") or prompt_hash)
        key = (
            str(request.get("stage") or request.get("request_kind") or "").casefold(),
            step_index,
            prefix_index,
            retry_group,
            str(request.get("seed") if request.get("seed") is not None else ""),
        )
        groups.setdefault(key, []).append((index, request))
    terminal: list[tuple[int, dict[str, Any]]] = []
    for attempts in groups.values():
        indices = [item[1].get("attempt_index", 0) for item in attempts]
        if any(isinstance(item, bool) or not isinstance(item, int) for item in indices):
            raise ValueError("request attempt_index must be an integer")
        if len(indices) != len(set(indices)):
            raise ValueError("request retries require unique attempt_index values")
        position, request = max(
            attempts, key=lambda item: int(item[1].get("attempt_index", 0))
        )
        if request.get("finish_reason") == "length":
            raise ValueError("terminal process-SFT request is truncated")
        if any(request.get(field) for field in _ERROR_FIELDS):
            raise ValueError("terminal process-SFT request contains an error")
        terminal.append((position, request))
    return tuple(request for _position, request in sorted(terminal))


def _official_tool_target(content: Any) -> str | None:
    if not isinstance(content, str):
        return None
    matches = list(_TOOL_CALL_RE.finditer(content))
    if not matches:
        return None
    if content.count("<tool_call>") != len(matches) or content.count(
        "</tool_call>"
    ) != len(matches):
        raise ValueError("controller response has malformed tool markup")
    blocks: list[str] = []
    for match in matches:
        try:
            payload = json.loads(match.group(1))
        except json.JSONDecodeError as error:
            raise ValueError(
                "controller frame_select call is not valid JSON"
            ) from error
        if not isinstance(payload, Mapping) or payload.get("tool") != "frame_select":
            raise ValueError("controller may only call official frame_select")
        arguments = payload.get("arguments")
        if not isinstance(arguments, Mapping):
            raise ValueError("frame_select arguments must be an object")
        if ("nframes" in arguments) == ("fps" in arguments):
            raise ValueError("frame_select requires exactly one of nframes or fps")
        blocks.append(
            "<tool_call>"
            + json.dumps(
                payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
            )
            + "</tool_call>"
        )
    return "\n".join(blocks)


def _plan_target(content: Any) -> str | None:
    try:
        payload = _json_object(content, "controller response")
    except ValueError:
        return None
    action = str(payload.get("action") or payload.get("decision") or "").casefold()
    return _canonical_json(payload) if action in {"continue", "plan"} else None


def _stop_target(content: Any) -> str | None:
    try:
        payload = _json_object(content, "stop response")
    except ValueError:
        return None
    action = str(payload.get("action") or payload.get("decision") or "").casefold()
    return _canonical_json(payload) if action == "stop" else None


def _answer_target(
    content: Any, *, valid_evidence_ids: frozenset[str] | None = None
) -> str | None:
    if not isinstance(content, str):
        return None
    stripped = content.strip()
    try:
        payload = _json_object(stripped, "final response")
    except ValueError:
        return None
    if set(payload) != {"answer", "evidence_ids"}:
        return None
    answer = str(payload["answer"] or "").strip().upper()
    if len(answer) != 1 or not "A" <= answer <= "H":
        return None
    evidence_ids = payload["evidence_ids"]
    if (
        not isinstance(evidence_ids, list)
        or not evidence_ids
        or len(evidence_ids) != len(set(evidence_ids))
        or any(not isinstance(item, str) or not item for item in evidence_ids)
    ):
        return None
    if valid_evidence_ids is not None and any(
        item not in valid_evidence_ids for item in evidence_ids
    ):
        return None
    # Keep the exact runtime protocol. Training a legacy ``Answer: X`` target
    # would make the strict evidence Judge parser fail after SFT.
    return _canonical_json({"answer": answer, "evidence_ids": evidence_ids})


def _target_answer(content: str) -> str | None:
    match = _ANSWER_RE.fullmatch(content.strip())
    if match:
        return match.group(1).upper()
    try:
        payload = json.loads(content)
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, Mapping):
        return None
    answer = str(payload.get("answer") or "").strip().upper()
    return answer if len(answer) == 1 and "A" <= answer <= "H" else None


def _stage(request: Mapping[str, Any]) -> str:
    return str(request.get("stage") or request.get("request_kind") or "").casefold()


def _prefix_index(request: Mapping[str, Any]) -> int:
    return int(request["prefix_index"])


def _requests_for(
    requests: Sequence[Mapping[str, Any]], step_index: int, stages: frozenset[str]
) -> list[Mapping[str, Any]]:
    return [
        request
        for request in requests
        if _prefix_index(request) == step_index and _stage(request) in stages
    ]


def _one_request(
    requests: Sequence[Mapping[str, Any]],
    step_index: int,
    stages: frozenset[str],
    label: str,
) -> Mapping[str, Any]:
    matches = _requests_for(requests, step_index, stages)
    if len(matches) != 1:
        raise ValueError(f"prefix {step_index} requires exactly one {label} request")
    return matches[0]


def _accepted_controller_requests(
    requests: Sequence[Mapping[str, Any]], prefix_index: int
) -> list[Mapping[str, Any]]:
    matches = _requests_for(requests, prefix_index, _CONTROLLER_STAGES)
    accepted: list[Mapping[str, Any]] = []
    for request in matches:
        status = request.get("action_accepted")
        if not isinstance(status, bool):
            raise ValueError("controller request requires boolean action_accepted")
        if status:
            accepted.append(request)
    return accepted


def _confirmation_gate(state: Mapping[str, Any], prediction: str) -> None:
    confirmations = state.get(
        "judge_confirmations", state.get("completion_confirmations")
    )
    if not isinstance(confirmations, list) or len(confirmations) != 3:
        raise ValueError("complete prefix requires exactly 3 Judge confirmations")
    memory = _state_memory(state)
    valid_evidence_ids = {
        str(item.get("evidence_id", item.get("id", "")))
        for item in memory["event_ledger"]
        if isinstance(item, Mapping)
    }
    valid_evidence_ids.discard("")
    seeds: set[str] = set()
    for index, raw in enumerate(confirmations):
        if not isinstance(raw, Mapping):
            raise ValueError(f"judge_confirmations[{index}] must be an object")
        seed = str(raw.get("judge_seed", raw.get("seed", "")))
        if not seed or seed in seeds:
            raise ValueError("complete prefix requires 3 unique Judge seeds")
        seeds.add(seed)
        answer = (
            str(
                raw.get(
                    "final_prediction", raw.get("prediction", raw.get("answer", ""))
                )
            )
            .strip()
            .upper()
        )
        if answer != prediction or raw.get("evidence_complete") is not True:
            raise ValueError("complete prefix is not 3/3 evidence-correct")
        evidence_ids = raw.get("evidence_ids")
        if (
            not isinstance(evidence_ids, list)
            or not evidence_ids
            or any(item not in valid_evidence_ids for item in evidence_ids)
        ):
            raise ValueError("Judge confirmation cites invalid evidence IDs")
        if raw.get("annotation_leak_check") != "passed":
            raise ValueError("Judge confirmation failed annotation leak audit")
        if raw.get("fallback_used") is True or raw.get("fallback_to_candidate") is True:
            raise ValueError("Judge confirmation may not use candidate fallback")
        if any(raw.get(field) for field in _ERROR_FIELDS):
            raise ValueError("Judge confirmation contains an error")
        messages = raw.get("request_messages", raw.get("messages"))
        if messages is not None:
            _assert_public_messages(messages, candidate_blind=True)


def _final_request_from_confirmation(
    state: Mapping[str, Any], step_index: int, prediction: str
) -> dict[str, Any]:
    """Materialize one 3/3 prefix Judge call as the runtime-compatible target."""

    _confirmation_gate(state, prediction)
    confirmations = sorted(
        state["judge_confirmations"],
        key=lambda item: int(item.get("judge_seed", item.get("seed", 0))),
    )
    confirmation = confirmations[0]
    messages = confirmation.get("request_messages", confirmation.get("messages"))
    if not isinstance(messages, list) or not messages:
        raise ValueError("Judge confirmation requires public request_messages")
    memory = _state_memory(state)
    valid_evidence_ids = frozenset(
        str(item.get("evidence_id", item.get("id", "")))
        for item in memory["event_ledger"]
        if isinstance(item, Mapping)
        and str(item.get("evidence_id", item.get("id", "")))
    )
    raw_response = str(confirmation.get("raw_response") or "")
    target = _answer_target(raw_response, valid_evidence_ids=valid_evidence_ids)
    if target is None or _target_answer(target) != prediction:
        raise ValueError("Judge confirmation has no runtime-compatible final target")
    return {
        "stage": "evidence_judge",
        "step_index": step_index,
        "prefix_index": step_index,
        "prompt_hash": str(
            confirmation.get("prompt_hash") or canonical_sha256(messages)
        ),
        "seed": int(confirmation.get("judge_seed", confirmation.get("seed", 0))),
        "attempt_index": 0,
        "messages": deepcopy(messages),
        "content": target,
        "finish_reason": confirmation.get("finish_reason") or "stop",
        "usage": deepcopy(confirmation.get("usage") or {}),
    }


def _episode_record(
    trajectory: Mapping[str, Any],
    request: Mapping[str, Any],
    *,
    step_index: int,
    target_type: str,
    target_content: str,
    role: str,
    memory: Mapping[str, Any],
    prefix_complete: bool,
) -> dict[str, Any]:
    messages = _request_messages(request)
    if role != "perception" and _media_paths(messages):
        raise ValueError(f"{role} request must be text-only")
    messages.append(
        {"role": "assistant", "content": target_content, "target_type": target_type}
    )
    materialized = dict(trajectory)
    materialized["training_messages"] = messages
    record = build_sft_record(
        materialized,
        require_complete_trajectory=False,
        episode_metadata={
            "episode_id": f"{trajectory['trajectory_id']}#prefix-{step_index:03d}-{role}",
            "prefix_index": step_index,
            "prefix_complete": prefix_complete,
            "process_role": role,
            "stage": _stage(request),
            "prompt_hash": str(
                request.get("prompt_hash") or canonical_sha256(request.get("messages"))
            ),
            "finish_reason": request.get("finish_reason"),
            "episode_target_type": target_type,
            "evidence_memory_sha256": canonical_sha256(memory),
            "usage": deepcopy(request.get("usage") or {}),
        },
    )
    validate_exported_sft_record(record)
    return record


def validate_selected_trajectory(trajectory: Mapping[str, Any]) -> None:
    """Enforce the per-trajectory gate before process-SFT export."""

    if (
        trajectory.get("perception_normalization_version")
        != PERCEPTION_NORMALIZATION_VERSION
    ):
        raise ValueError(
            "selected trajectory has incompatible perception normalization"
        )
    _public_option_letters(trajectory)
    if trajectory.get("_selection_stable") is not True:
        raise ValueError("process SFT requires a stable selected trajectory")
    if trajectory.get("annotation_leak_check") != "passed":
        raise ValueError("selected trajectory failed annotation leak audit")
    if int(trajectory.get("candidate_rerun") or 0) != 0:
        raise ValueError("selected trajectory reran its frozen candidate")
    if (
        trajectory.get("fallback_used") is True
        or trajectory.get("fallback_to_candidate") is True
    ):
        raise ValueError("selected trajectory may not depend on candidate fallback")
    if any(trajectory.get(field) for field in _ERROR_FIELDS):
        raise ValueError("selected trajectory contains an engineering error")
    prediction = (
        str(trajectory.get("final_prediction", trajectory.get("prediction", "")))
        .strip()
        .upper()
    )
    if len(prediction) != 1 or not "A" <= prediction <= "H":
        raise ValueError("selected trajectory has no A-H final_prediction")
    tool_steps = trajectory.get("tool_steps")
    states = trajectory.get("perception_states")
    trace = trajectory.get("request_trace")
    if not isinstance(tool_steps, list) or not tool_steps:
        raise ValueError("selected trajectory requires at least one frame_select")
    if (
        not isinstance(states, list)
        or not states
        or not isinstance(trace, list)
        or not trace
    ):
        raise ValueError(
            "selected trajectory requires perception_states and request_trace"
        )
    for index, raw in enumerate(states):
        if not isinstance(raw, Mapping):
            raise ValueError(f"perception_states[{index}] must be an object")
        step_index = raw.get("step_index", index)
        if step_index != index:
            raise ValueError("perception state indices must be contiguous and ordered")
        _validate_perception_response(raw.get("perception_response"))
        _state_memory(raw)
        frame_paths = raw.get("frame_paths")
        timestamps = raw.get("timestamps")
        if not isinstance(frame_paths, list) or not frame_paths:
            raise ValueError("perception state requires current-step frame_paths")
        if not isinstance(timestamps, list) or len(timestamps) != len(frame_paths):
            raise ValueError("perception state frame_paths/timestamps count differs")
        for frame_path in frame_paths:
            path = Path(str(frame_path)).resolve()
            if not path.is_absolute() or not path.is_file():
                raise ValueError(f"perception state frame does not exist: {path}")
        complete = raw.get("evidence_complete")
        if not isinstance(complete, bool):
            raise ValueError("perception state evidence_complete must be boolean")
    if states[-1].get("evidence_complete") is not True:
        raise ValueError("the final perception prefix must be complete")
    for state in states:
        if state.get("evidence_complete") is True:
            _confirmation_gate(state, prediction)
    _terminal_requests(
        [dict(item) if isinstance(item, Mapping) else item for item in trace]
    )


def build_perception_memory_sft_records(
    trajectory: Mapping[str, Any],
) -> tuple[dict[str, Any], ...]:
    """Export memory/action/final episodes with prefix-completeness safeguards."""

    validate_selected_trajectory(trajectory)
    states = trajectory["perception_states"]
    valid_letters = _public_option_letters(trajectory)
    tool_steps = trajectory["tool_steps"]
    first_complete_index = next(
        index for index, state in enumerate(states) if state["evidence_complete"]
    )
    training_states = states[: first_complete_index + 1]
    requests = _terminal_requests(trajectory["request_trace"])
    prediction = str(
        trajectory.get("final_prediction", trajectory.get("prediction"))
    ).upper()
    records: list[dict[str, Any]] = []
    initial_requests = _accepted_controller_requests(requests, -1)
    initial_actions: list[tuple[Mapping[str, Any], str, str]] = []
    for request in initial_requests:
        tool = _official_tool_target(request.get("content"))
        plan = None if tool else _plan_target(request.get("content"))
        if tool:
            initial_actions.append((request, "tool", tool))
        elif plan:
            initial_actions.append((request, "plan", plan))
    if len(initial_actions) != 1:
        raise ValueError("unobserved prefix requires one initial tool/plan target")
    initial_request, initial_type, initial_content = initial_actions[0]
    records.append(
        _episode_record(
            trajectory,
            initial_request,
            step_index=-1,
            target_type=initial_type,
            target_content=initial_content,
            role="controller",
            memory={},
            prefix_complete=False,
        )
    )
    for step_index, state in enumerate(training_states):
        memory = _state_memory(state)
        observation = _one_request(
            requests, step_index, _OBSERVATION_STAGES, "perception"
        )
        observation_messages = _request_messages(observation)
        observed_paths = _media_paths(observation_messages)
        if not observed_paths:
            raise ValueError(f"prefix {step_index} perception request has no frames")
        expected_paths = tuple(
            str(Path(str(path)).resolve()) for path in state["frame_paths"]
        )
        if observed_paths != expected_paths:
            raise ValueError(
                f"prefix {step_index} perception request does not contain only current frames"
            )
        resolved_start, resolved_end, actual_timestamps = _tool_state_contract(
            tool_steps, state, step_index
        )
        response = _validate_perception_response(state["perception_response"])
        observed_payload = _normalize_raw_perception_response(
            observation.get("content"),
            valid_letters=valid_letters,
            resolved_start_time=resolved_start,
            resolved_end_time=resolved_end,
            actual_timestamps=actual_timestamps,
        )
        if canonical_sha256(observed_payload) != canonical_sha256(response):
            raise ValueError("perception state differs from the actual model response")
        model_target = state.get("perception_model_target")
        if model_target is None:
            target_content = _canonical_json(response)
        else:
            target_payload = _normalize_raw_perception_response(
                model_target,
                valid_letters=valid_letters,
                resolved_start_time=resolved_start,
                resolved_end_time=resolved_end,
                actual_timestamps=actual_timestamps,
            )
            if canonical_sha256(target_payload) != canonical_sha256(response):
                raise ValueError(
                    "perception model target differs from normalized state"
                )
            target_content = _canonical_json(
                _json_object(model_target, "perception_model_target")
            )
        complete = bool(state["evidence_complete"])
        records.append(
            _episode_record(
                trajectory,
                observation,
                step_index=step_index,
                target_type="memory",
                target_content=target_content,
                role="perception",
                memory=memory,
                prefix_complete=complete,
            )
        )

        controller_requests = _accepted_controller_requests(requests, step_index)
        if not complete:
            actions: list[tuple[Mapping[str, Any], str, str]] = []
            for request in controller_requests:
                tool = _official_tool_target(request.get("content"))
                plan = None if tool else _plan_target(request.get("content"))
                if tool:
                    actions.append((request, "tool", tool))
                elif plan:
                    actions.append((request, "plan", plan))
            if len(actions) != 1:
                raise ValueError(
                    f"incomplete prefix {step_index} requires one next tool/plan target"
                )
            request, target_type, target_content = actions[0]
            records.append(
                _episode_record(
                    trajectory,
                    request,
                    step_index=step_index,
                    target_type=target_type,
                    target_content=target_content,
                    role="controller",
                    memory=memory,
                    prefix_complete=False,
                )
            )
            continue

        primary_controller_requests = _requests_for(
            requests, step_index, _PRIMARY_CONTROLLER_STAGES
        )
        stop_matches: list[tuple[Mapping[str, Any], str]] = []
        for request in primary_controller_requests:
            target = _stop_target(request.get("content"))
            if target:
                stop_matches.append((request, target))
        if len(stop_matches) > 1:
            raise ValueError("complete prefix has multiple stop targets")
        if stop_matches:
            stop_request, stop_content = stop_matches[0]
        else:
            # A prefix can become complete only after offline 3/3 verification.
            # Counterfactually teach the Controller to stop at that exact text
            # memory instead of imitating the Teacher's now-redundant next call.
            stop_request = (
                primary_controller_requests[0] if primary_controller_requests else None
            )
            stop_content = '{"action":"stop"}'
        if stop_request is not None:
            records.append(
                _episode_record(
                    trajectory,
                    stop_request,
                    step_index=step_index,
                    target_type="stop",
                    target_content=stop_content,
                    role="controller",
                    memory=memory,
                    prefix_complete=True,
                )
            )
        final_request = _final_request_from_confirmation(state, step_index, prediction)
        final_target = str(final_request["content"])
        records.append(
            _episode_record(
                trajectory,
                final_request,
                step_index=step_index,
                target_type="final",
                target_content=final_target,
                role="final_judge",
                memory=memory,
                prefix_complete=True,
            )
        )
        break

    target_counts = Counter(
        target
        for record in records
        for target in record["metadata"]["assistant_target_types"]
    )
    if target_counts["final"] != 1:
        raise ValueError("selected trajectory must export exactly one final target")
    if target_counts["memory"] != len(training_states):
        raise ValueError(
            "every retained perception prefix must export one memory target"
        )
    return tuple(records)


def summarize_perception_memory_sft(
    trajectories: Iterable[Mapping[str, Any]],
    records: Iterable[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    rows = [dict(row) for row in trajectories]
    exported = [dict(record) for record in records]
    by_dataset: Counter[str] = Counter()
    fixes_by_dataset: Counter[str] = Counter()
    prefixes = Counter()
    seen: set[tuple[str, str]] = set()
    for row in rows:
        validate_selected_trajectory(row)
        identity = (str(row.get("dataset") or ""), str(row.get("sample_id") or ""))
        if identity in seen:
            raise ValueError(f"duplicate selected sample: {identity[0]}/{identity[1]}")
        seen.add(identity)
        by_dataset[identity[0]] += 1
        candidate = str(row.get("candidate_answer") or "").strip().upper()
        prediction = str(row.get("final_prediction") or "").strip().upper()
        if candidate and candidate != prediction:
            fixes_by_dataset[identity[0]] += 1
        prefixes["unobserved"] += 1
        for state in row["perception_states"]:
            prefixes["complete" if state["evidence_complete"] else "incomplete"] += 1
    targets: Counter[str] = Counter()
    record_trajectory_ids: set[str] = set()
    for record in exported:
        validate_exported_sft_record(record)
        targets.update(record["metadata"]["assistant_target_types"])
        record_trajectory_ids.add(str(record["metadata"].get("trajectory_id") or ""))
    selected_trajectory_ids = {str(row.get("trajectory_id") or "") for row in rows}
    if exported and record_trajectory_ids != selected_trajectory_ids:
        raise ValueError("selected trajectories and process-SFT record coverage differ")
    if exported:
        expected_memory_targets = sum(
            next(
                index + 1
                for index, state in enumerate(row["perception_states"])
                if state["evidence_complete"]
            )
            for row in rows
        )
        if targets["memory"] != expected_memory_targets or targets["final"] != len(
            rows
        ):
            raise ValueError(
                "process-SFT target coverage does not match selected prefixes"
            )
    return {
        "selected_trajectories": len(rows),
        "selected_by_dataset": dict(sorted(by_dataset.items())),
        "candidate_fixes": sum(fixes_by_dataset.values()),
        "candidate_fixes_by_dataset": dict(sorted(fixes_by_dataset.items())),
        "prefixes": dict(sorted(prefixes.items())),
        "sft_records": len(exported),
        "record_coverage_passed": bool(exported),
        "assistant_targets": dict(sorted(targets.items())),
    }


def enforce_perception_memory_selection_gate(
    trajectories: Iterable[Mapping[str, Any]],
    records: Iterable[Mapping[str, Any]] = (),
    *,
    minimum_total: int = 360,
    minimum_per_dataset: int = 100,
    minimum_candidate_fixes: int = 90,
    minimum_candidate_fixes_per_dataset: int = 20,
) -> dict[str, Any]:
    """Validate corpus-level gates and return the frozen summary."""

    limits = (
        minimum_total,
        minimum_per_dataset,
        minimum_candidate_fixes,
        minimum_candidate_fixes_per_dataset,
    )
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value < 0
        for value in limits
    ):
        raise ValueError("selection gate thresholds must be non-negative integers")
    summary = summarize_perception_memory_sft(trajectories, records)
    failures: list[str] = []
    if summary["selected_trajectories"] < minimum_total:
        failures.append(f"selected<{minimum_total}")
    for dataset in ("lvbench", "lsdbench", "cgbench"):
        if summary["selected_by_dataset"].get(dataset, 0) < minimum_per_dataset:
            failures.append(f"{dataset}<{minimum_per_dataset}")
        if (
            summary["candidate_fixes_by_dataset"].get(dataset, 0)
            < minimum_candidate_fixes_per_dataset
        ):
            failures.append(
                f"{dataset}_candidate_fixes<{minimum_candidate_fixes_per_dataset}"
            )
    if summary["candidate_fixes"] < minimum_candidate_fixes:
        failures.append(f"candidate_fixes<{minimum_candidate_fixes}")
    if failures:
        raise ValueError("process-SFT selection gate failed: " + ", ".join(failures))
    return summary


__all__ = [
    "build_perception_memory_sft_records",
    "enforce_perception_memory_selection_gate",
    "summarize_perception_memory_sft",
    "validate_selected_trajectory",
]
