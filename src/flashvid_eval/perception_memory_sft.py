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
    EvidenceEvent,
    EvidenceMemory,
    OptionLedger,
    PERCEPTION_NORMALIZATION_VERSION,
    bind_perception_state,
    build_controller_messages,
    build_role_separated_controller_messages,
    normalize_perception_state,
    parse_controller_action,
    perception_state_payload,
)
from .question_time import parse_question_time_range
from .qwen_sft import (
    build_sft_record,
    canonical_sha256,
    validate_exported_sft_record,
)
from .schemas import ModelSample


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
_CONTROLLER_VIDEO_RE = re.compile(
    r"Video duration:\s*([0-9]+(?:\.[0-9]+)?) seconds;\s*"
    r"resolution:\s*(\d+)x(\d+)"
)
_COMPLETION_GATE_KINDS = frozenset(
    {"auto", "visual_csv", "legacy_prefix_judge"}
)

_OBSERVATION_STAGES = frozenset({"perception", "observation", "observer"})
_PRIMARY_CONTROLLER_STAGES = frozenset({"controller", "planner"})
_FINAL_STAGES = frozenset({"evidence_judge", "final_judge", "judge", "final"})

QUALITY_CONTRACT_VERSION = "role_separated_process_sft_quality_v1"
VISUAL_PATH_CLASSIFIER_VERSION = "visual_path_classifier_v1"
VISUAL_PATH_FAMILIES = (
    "single_frame_select",
    "timestamp_grounded_select",
    "hierarchical_refinement",
    "multi_interval_exploration",
)
CANDIDATE_TRAINING_STRATA = ("candidate_correct", "candidate_wrong")
QUALITY_RATIO_MIN = 0.9
QUALITY_RATIO_MAX = 1.1
_INTERVAL_TOLERANCE = 1e-6


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


def _validate_perception_response(
    value: Any, *, allow_role_separated_schema: bool = False
) -> dict[str, Any]:
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
    role_separated_required = required - {
        "evidence_sufficient",
        "next_evidence_needed",
    }
    expected = role_separated_required if allow_role_separated_schema else required
    if set(response) != expected:
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
    if not allow_role_separated_schema:
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
    allow_role_separated_schema: bool = False,
) -> dict[str, Any]:
    """Apply the runtime's exact validation/compaction to one raw response."""

    if not isinstance(value, str):
        raise ValueError("perception response must be raw JSON text")
    state, _reference_mode = bind_perception_state(
        value,
        valid_letters,
        actual_timestamps,
        allow_role_separated_schema=allow_role_separated_schema,
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
    return perception_state_payload(
        normalized, role_separated=allow_role_separated_schema
    )


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


def _normalized_frame_request(value: Any, field: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field} must be an object")
    allowed = {
        "start_time",
        "end_time",
        "nframes",
        "fps",
        "resize",
        "evidence_request",
    }
    if set(value) - allowed:
        raise ValueError(f"{field} contains unsupported fields")
    if ("nframes" in value) == ("fps" in value):
        raise ValueError(f"{field} requires exactly one of nframes or fps")
    try:
        start = float(value["start_time"])
        end = float(value["end_time"])
        resize = float(value.get("resize", 1.0))
        if "nframes" in value:
            raw_nframes = value["nframes"]
            if isinstance(raw_nframes, bool) or not isinstance(raw_nframes, int):
                raise ValueError
            sampling: dict[str, Any] = {"nframes": raw_nframes}
        else:
            raw_fps = value["fps"]
            if isinstance(raw_fps, bool):
                raise ValueError
            sampling = {"fps": float(raw_fps)}
    except (KeyError, TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{field} has invalid frame_select arguments") from error
    if (
        not all(math.isfinite(item) for item in (start, end, resize))
        or end <= start
        or not 0.05 <= resize <= 2.0
    ):
        raise ValueError(f"{field} has invalid frame_select arguments")
    if "nframes" in sampling and sampling["nframes"] <= 0:
        raise ValueError(f"{field} has invalid nframes")
    if "fps" in sampling and (
        not math.isfinite(sampling["fps"]) or sampling["fps"] <= 0
    ):
        raise ValueError(f"{field} has invalid fps")
    evidence_request = value.get("evidence_request", "")
    if not isinstance(evidence_request, str):
        raise ValueError(f"{field}.evidence_request must be text")
    return {
        "start_time": start,
        "end_time": end,
        "resize": resize,
        **sampling,
        **(
            {"evidence_request": " ".join(evidence_request.split())}
            if evidence_request.strip()
            else {}
        ),
    }


def _accepted_frame_requests(
    trace: Sequence[Mapping[str, Any]], retained_count: int,
) -> tuple[dict[str, Any], ...]:
    requests: list[dict[str, Any]] = []
    for index, raw in enumerate(trace):
        if not isinstance(raw, Mapping):
            raise ValueError(f"request_trace[{index}] must be an object")
        if (
            _stage(raw)
            not in (_PRIMARY_CONTROLLER_STAGES | {"confirmation_controller"})
            or raw.get("action_accepted") is not True
        ):
            continue
        content = raw.get("content")
        action = parse_controller_action(content if isinstance(content, str) else "")
        if action is None:
            continue
        if action.action == "observe":
            assert action.request is not None
            if raw.get("prefix_index") != len(requests) - 1:
                raise ValueError("accepted Planner frame_select prefix order differs")
            requests.append(action.request.to_tool_arguments())
            if len(requests) == retained_count:
                break
    return tuple(requests)


def _intervals_for_classifier(
    trajectory: Mapping[str, Any],
) -> tuple[tuple[float, float], ...]:
    steps = trajectory.get("tool_steps")
    if not isinstance(steps, list) or not steps:
        raise ValueError("visual path classifier requires real frame_select steps")
    states = trajectory.get("perception_states")
    if not isinstance(states, list) or not states:
        raise ValueError("visual path classifier requires perception states")
    try:
        retained_count = next(
            index + 1
            for index, state in enumerate(states)
            if isinstance(state, Mapping) and state.get("evidence_complete") is True
        )
    except StopIteration as error:
        raise ValueError("visual path classifier requires a complete prefix") from error
    if len(steps) < retained_count:
        raise ValueError("visual path classifier has fewer tools than retained prefixes")
    intervals: list[tuple[float, float]] = []
    for index, raw in enumerate(steps[:retained_count]):
        if not isinstance(raw, Mapping):
            raise ValueError(f"tool_steps[{index}] must be an object")
        start = raw.get("resolved_start_time", raw.get("start_time"))
        end = raw.get("resolved_end_time", raw.get("end_time"))
        if (
            isinstance(start, bool)
            or not isinstance(start, (int, float))
            or isinstance(end, bool)
            or not isinstance(end, (int, float))
            or not math.isfinite(float(start))
            or not math.isfinite(float(end))
            or float(end) <= float(start)
        ):
            raise ValueError(f"tool_steps[{index}] has invalid classifier interval")
        intervals.append((float(start), float(end)))
    return tuple(intervals)


def classify_visual_path(trajectory: Mapping[str, Any]) -> str:
    """Classify one retained trajectory with the frozen v1 precedence rules."""

    intervals = _intervals_for_classifier(trajectory)
    public = trajectory.get("public_sample")
    question = public.get("question") if isinstance(public, Mapping) else None
    if not isinstance(question, str):
        raise ValueError("visual path classifier requires the public question")
    explicit = parse_question_time_range(question, padding_s=1.0)
    if len(intervals) == 1 and explicit is not None and any(
        min(end, explicit[1]) - max(start, explicit[0]) > _INTERVAL_TOLERANCE
        for start, end in intervals
    ):
        return "timestamp_grounded_select"
    if len(intervals) == 1:
        return "single_frame_select"
    if all(
        outer[0] - _INTERVAL_TOLERANCE <= inner[0]
        and inner[1] <= outer[1] + _INTERVAL_TOLERANCE
        and (inner[1] - inner[0]) < (outer[1] - outer[0]) - _INTERVAL_TOLERANCE
        for outer, inner in zip(intervals[:-1], intervals[1:], strict=True)
    ):
        return "hierarchical_refinement"
    if all(
        min(first[1], second[1]) - max(first[0], second[0])
        <= _INTERVAL_TOLERANCE
        for index, first in enumerate(intervals)
        for second in intervals[index + 1 :]
    ):
        return "multi_interval_exploration"
    raise ValueError(
        "trajectory does not match a frozen visual path family: "
        + str(trajectory.get("trajectory_id") or "<missing>")
    )


def _candidate_training_stratum(trajectory: Mapping[str, Any]) -> str:
    declared = str(trajectory.get("candidate_training_stratum") or "").strip()
    if declared and declared not in CANDIDATE_TRAINING_STRATA:
        raise ValueError("candidate_training_stratum is not a frozen public stratum")
    candidate = str(trajectory.get("candidate_answer") or "").strip().upper()
    prediction = str(
        trajectory.get("final_prediction", trajectory.get("prediction", ""))
    ).strip().upper()
    derived = ""
    if candidate:
        if len(candidate) != 1 or not "A" <= candidate <= "H":
            raise ValueError("selected trajectory has an invalid frozen candidate")
        derived = (
            "candidate_correct" if candidate == prediction else "candidate_wrong"
        )
    if declared and derived and declared != derived:
        raise ValueError("candidate_training_stratum contradicts the frozen candidate")
    if not declared and not derived:
        raise ValueError(
            "selected trajectory requires a sanitized candidate_training_stratum"
        )
    return declared or derived


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
    *,
    completed_prefix_index: int | None = None,
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
        ignorable_completed_planner = (
            completed_prefix_index is not None
            and _stage(request) in _PRIMARY_CONTROLLER_STAGES
            and request.get("prefix_index") == completed_prefix_index
        )
        if request.get("finish_reason") == "length" and ignorable_completed_planner:
            continue
        if request.get("finish_reason") == "length":
            raise ValueError("terminal process-SFT request is truncated")
        if any(request.get(field) for field in _ERROR_FIELDS) and ignorable_completed_planner:
            continue
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
    requests: Sequence[Mapping[str, Any]],
    prefix_index: int,
    stages: frozenset[str] = _PRIMARY_CONTROLLER_STAGES,
) -> list[Mapping[str, Any]]:
    matches = _requests_for(requests, prefix_index, stages)
    accepted: list[Mapping[str, Any]] = []
    for request in matches:
        status = request.get("action_accepted")
        if not isinstance(status, bool):
            raise ValueError("controller request requires boolean action_accepted")
        if status:
            accepted.append(request)
    return accepted


def _confirmation_source(
    state: Mapping[str, Any], requested_kind: str
) -> tuple[str, list[Any]]:
    kind = str(requested_kind).strip().casefold()
    if kind not in _COMPLETION_GATE_KINDS:
        raise ValueError(f"unsupported completion gate kind: {requested_kind!r}")
    visual = state.get("visual_csv_confirmations")
    legacy = state.get(
        "judge_confirmations", state.get("completion_confirmations")
    )
    if kind in {"auto", "visual_csv"} and isinstance(visual, list):
        return "visual_csv", visual
    if kind == "visual_csv":
        raise ValueError("visual_csv completion gate requires visual_csv_confirmations")
    if isinstance(legacy, list):
        return "legacy_prefix_judge", legacy
    raise ValueError("prefix has no 3/3 completion confirmations")


def _confirmation_gate(
    state: Mapping[str, Any],
    prediction: str,
    *,
    expected_complete: bool,
    requested_kind: str,
) -> str:
    gate_kind, confirmations = _confirmation_source(state, requested_kind)
    if len(confirmations) != 3:
        raise ValueError("prefix requires exactly 3 completion confirmations")
    memory = _state_memory(state)
    valid_evidence_ids = {
        str(item.get("evidence_id", item.get("id", "")))
        for item in memory["event_ledger"]
        if isinstance(item, Mapping)
    }
    valid_evidence_ids.discard("")
    seeds: set[str] = set()
    predictions: set[str] = set()
    for index, raw in enumerate(confirmations):
        if not isinstance(raw, Mapping):
            raise ValueError(f"completion confirmation[{index}] must be an object")
        seed = str(raw.get("judge_seed", raw.get("seed", "")))
        if not seed or seed in seeds:
            raise ValueError("prefix requires 3 unique confirmation seeds")
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
        if (
            len(answer) != 1
            or not "A" <= answer <= "H"
            or raw.get("parsed_valid", True) is not True
        ):
            raise ValueError("completion confirmation has no valid A-H prediction")
        predictions.add(answer)
        if gate_kind == "visual_csv":
            frame_indices = raw.get("frame_indices")
            if (
                raw.get("candidate_blind") is not True
                or not isinstance(frame_indices, list)
                or not frame_indices
                or len(frame_indices) != len(set(frame_indices))
                or any(
                    isinstance(item, bool) or not isinstance(item, int) or item < 0
                    for item in frame_indices
                )
            ):
                raise ValueError(
                    "visual_csv confirmation requires candidate-blind frame_indices"
                )
        else:
            evidence_ids = raw.get("evidence_ids")
            if (
                not isinstance(evidence_ids, list)
                or not evidence_ids
                or any(item not in valid_evidence_ids for item in evidence_ids)
            ):
                raise ValueError("prefix Judge confirmation cites invalid evidence IDs")
        if raw.get("annotation_leak_check") != "passed":
            raise ValueError("completion confirmation failed annotation leak audit")
        if raw.get("fallback_used") is True or raw.get("fallback_to_candidate") is True:
            raise ValueError("completion confirmation may not use candidate fallback")
        if any(raw.get(field) for field in _ERROR_FIELDS):
            raise ValueError("completion confirmation contains an error")
        messages = raw.get("request_messages", raw.get("messages"))
        if messages is not None:
            _assert_public_messages(messages, candidate_blind=True)
    complete = predictions == {prediction}
    if complete != expected_complete:
        label = "complete" if expected_complete else "incomplete"
        raise ValueError(f"3/3 completion confirmations do not support {label} label")
    return gate_kind


def _role_episode_record(
    trajectory: Mapping[str, Any],
    messages: Sequence[Mapping[str, Any]],
    *,
    role: str,
    turns: Sequence[Mapping[str, Any]],
    completion_gate_kind: str,
    terminal_prefix_index: int,
    episode_schema: str,
    episode_id: str,
) -> dict[str, Any]:
    materialized = dict(trajectory)
    materialized["training_messages"] = deepcopy(list(messages))
    path_family = classify_visual_path(trajectory)
    record = build_sft_record(
        materialized,
        require_complete_trajectory=True,
        require_final_target=False,
        episode_metadata={
            "episode_id": episode_id,
            "episode_schema": episode_schema,
            "terminal_prefix_index": terminal_prefix_index,
            "prefix_complete": True,
            "process_role": role,
            "completion_gate_kind": completion_gate_kind,
            "observer_sufficiency_fields_masked": role == "observer",
            "quality_contract_version": QUALITY_CONTRACT_VERSION,
            "visual_path_classifier_version": VISUAL_PATH_CLASSIFIER_VERSION,
            "visual_path_family": path_family,
            "candidate_training_stratum": _candidate_training_stratum(trajectory),
            "episode_turns": deepcopy(list(turns)),
        },
    )
    validate_exported_sft_record(record)
    return record


def _append_snapshot_turn(
    episode: list[dict[str, Any]],
    request: Mapping[str, Any],
    *,
    target_type: str,
    target_content: str,
) -> None:
    snapshot = _request_messages(request)
    system = [message for message in snapshot if message["role"] == "system"]
    non_system = [message for message in snapshot if message["role"] != "system"]
    if len(system) != 1 or len(non_system) != 1 or non_system[0]["role"] != "user":
        raise ValueError("role episode requires one system and one user per snapshot")
    if not episode:
        episode.append(system[0])
    elif episode[0] != system[0]:
        raise ValueError("role episode system prompt changed between turns")
    episode.append(non_system[0])
    episode.append(
        {
            "role": "assistant",
            "content": target_content,
            "target_type": target_type,
        }
    )


def _public_message(value: Mapping[str, Any]) -> dict[str, Any]:
    message = {
        key: deepcopy(value[key])
        for key in ("role", "content", "name", "tool_call_id", "tool_calls", "function_call")
        if key in value
    }
    if message.get("role") == "assistant":
        action = _planner_action({"content": message.get("content")})
        if action is None:
            raise ValueError("Planner accepted history contains an invalid action")
        message["content"] = action[1]
    return message


def _append_role_separated_planner_turn(
    episode: list[dict[str, Any]],
    request: Mapping[str, Any],
    *,
    target_type: str,
    target_content: str,
) -> None:
    """Append one target only when the trace contains the exact accepted history."""

    snapshot = _request_messages(request)
    if snapshot[0]["role"] != "system" or snapshot[-1]["role"] != "user":
        raise ValueError("role-separated Planner snapshot has invalid boundaries")
    public_snapshot = [_public_message(message) for message in snapshot]
    if not episode:
        if len(public_snapshot) != 2:
            raise ValueError("initial Planner request must have no accepted history")
        episode.extend(public_snapshot)
    else:
        expected_prefix = [_public_message(message) for message in episode]
        expected_prefix[0] = public_snapshot[0]
        expected = [*expected_prefix, public_snapshot[-1]]
        if public_snapshot != expected:
            raise ValueError(
                "Planner request accepted-history differs from the training episode"
            )
        episode[0] = public_snapshot[0]
        episode.append(public_snapshot[-1])
    episode.append(
        {
            "role": "assistant",
            "content": target_content,
            "target_type": target_type,
        }
    )


def _accepted_planner_history(
    episode: Sequence[Mapping[str, Any]],
) -> list[tuple[dict[str, Any], str]]:
    history: list[tuple[dict[str, Any], str]] = []
    for index, message in enumerate(episode):
        if message.get("role") != "assistant" or message.get("target_type") != "tool":
            continue
        if index == 0 or episode[index - 1].get("role") != "user":
            raise ValueError("Planner tool target lacks its exact user snapshot")
        history.append(
            (
                _public_message(episode[index - 1]),
                str(message.get("content") or ""),
            )
        )
    return history


def _planner_action(request: Mapping[str, Any]) -> tuple[str, str] | None:
    tool = _official_tool_target(request.get("content"))
    if tool:
        return "tool", tool
    plan = _plan_target(request.get("content"))
    return ("plan", plan) if plan else None


def _rehydrate_memory(
    value: Mapping[str, Any], valid_letters: Sequence[str]
) -> EvidenceMemory:
    events: list[EvidenceEvent] = []
    for index, raw in enumerate(value.get("event_ledger") or []):
        if not isinstance(raw, Mapping):
            raise ValueError(f"event_ledger[{index}] must be an object")
        interval = _numeric_tuple(raw.get("interval"), f"event_ledger[{index}].interval")
        if len(interval) != 2 or interval[1] <= interval[0]:
            raise ValueError(f"event_ledger[{index}] has invalid interval")
        timestamp = raw.get("timestamp")
        if timestamp is not None:
            timestamp = _numeric_tuple([timestamp], "event timestamp")[0]
        evidence_id = str(raw.get("evidence_id", raw.get("id", ""))).strip()
        fact = str(raw.get("fact") or "").strip()
        source = str(raw.get("source") or "").strip()
        if not evidence_id or not fact or not source:
            raise ValueError(f"event_ledger[{index}] lacks id/fact/source")
        events.append(
            EvidenceEvent(evidence_id, (interval[0], interval[1]), timestamp, fact, source)
        )
    raw_options = value.get("option_ledger")
    if not isinstance(raw_options, Mapping) or set(raw_options) != set(valid_letters):
        raise ValueError("option_ledger does not match public option labels")
    options: dict[str, OptionLedger] = {}
    for letter in valid_letters:
        raw = raw_options[letter]
        if not isinstance(raw, Mapping):
            raise ValueError(f"option_ledger[{letter}] must be an object")
        supports = raw.get("supports")
        contradicts = raw.get("contradicts")
        if (
            not isinstance(supports, list)
            or not isinstance(contradicts, list)
            or any(not isinstance(item, str) for item in supports + contradicts)
        ):
            raise ValueError(f"option_ledger[{letter}] has invalid evidence IDs")
        options[letter] = OptionLedger(tuple(supports), tuple(contradicts))
    unresolved = value.get("unresolved")
    if not isinstance(unresolved, list) or any(
        not isinstance(item, str) for item in unresolved
    ):
        raise ValueError("evidence memory unresolved must be a string list")
    raw_intervals = value.get("observed_intervals")
    if raw_intervals is None:
        raw_intervals = [list(event.interval) for event in events]
    intervals: list[tuple[float, float]] = []
    for index, raw in enumerate(raw_intervals):
        interval = _numeric_tuple(raw, f"observed_intervals[{index}]")
        if len(interval) != 2 or interval[1] <= interval[0]:
            raise ValueError(f"observed_intervals[{index}] has invalid interval")
        pair = (interval[0], interval[1])
        if pair not in intervals:
            intervals.append(pair)
    return EvidenceMemory(
        tuple(valid_letters),
        event_ledger=events,
        option_ledger=options,
        unresolved=list(unresolved),
        observed_intervals=intervals,
    )


def _controller_video_metadata(
    requests: Sequence[Mapping[str, Any]],
    trajectory: Mapping[str, Any] | None = None,
) -> dict[str, float | int]:
    for request in requests:
        if _stage(request) not in _PRIMARY_CONTROLLER_STAGES:
            continue
        for message in _request_messages(request):
            content = message.get("content")
            if not isinstance(content, str):
                continue
            match = _CONTROLLER_VIDEO_RE.search(content)
            if match:
                return {
                    "duration": float(match.group(1)),
                    "width": int(match.group(2)),
                    "height": int(match.group(3)),
                }
    if trajectory is not None:
        tool_steps = trajectory.get("tool_steps")
        if isinstance(tool_steps, list) and tool_steps:
            ends = [
                step.get("resolved_end_time", step.get("end_time"))
                for step in tool_steps
                if isinstance(step, Mapping)
            ]
            if ends and all(
                isinstance(item, (int, float)) and not isinstance(item, bool)
                for item in ends
            ):
                return {"duration": max(float(item) for item in ends), "width": 0, "height": 0}
    raise ValueError("controller trace has no frozen video metadata snapshot")


def _synthetic_stop_request(
    trajectory: Mapping[str, Any],
    requests: Sequence[Mapping[str, Any]],
    state: Mapping[str, Any],
    prefix_index: int,
    *,
    accepted_actions: Sequence[tuple[Mapping[str, Any], str]],
    role_separated: bool,
) -> dict[str, Any]:
    public = trajectory.get("public_sample")
    if not isinstance(public, Mapping):
        raise ValueError("selected trajectory requires public_sample")
    sample = ModelSample(
        dataset=str(public.get("dataset") or trajectory.get("dataset") or ""),
        sample_id=str(public.get("sample_id") or trajectory.get("sample_id") or ""),
        video=str(public.get("video") or ""),
        question=str(public.get("question") or ""),
        choices=deepcopy(dict(public.get("choices") or {})),
        candidate_answer=None,
    )
    memory = _rehydrate_memory(_state_memory(state), sample.option_letters)
    metadata = _controller_video_metadata(requests, trajectory)
    if role_separated:
        messages = build_role_separated_controller_messages(
            sample, memory, metadata, accepted_actions
        )
    else:
        messages = build_controller_messages(sample, memory, metadata)
        for source_request in requests:
            if _stage(source_request) not in _PRIMARY_CONTROLLER_STAGES:
                continue
            source_messages = _request_messages(source_request)
            source_system = [
                message for message in source_messages if message["role"] == "system"
            ]
            if len(source_system) == 1:
                messages[0] = source_system[0]
                break
    return {
        "stage": "planner",
        "step_index": prefix_index + 1,
        "prefix_index": prefix_index,
        "prompt_hash": canonical_sha256(messages),
        "messages": messages,
        "content": '{"action":"stop"}',
        "finish_reason": "offline_visual_csv_stop_target",
        "usage": {},
        "action_accepted": True,
    }


def _turn_metadata(
    request: Mapping[str, Any],
    *,
    prefix_index: int,
    target_type: str,
    target_origin: str,
) -> dict[str, Any]:
    return {
        "prefix_index": prefix_index,
        "stage": _stage(request),
        "prompt_hash": str(
            request.get("prompt_hash") or canonical_sha256(request.get("messages"))
        ),
        "target_type": target_type,
        "target_origin": target_origin,
    }


def validate_selected_trajectory(
    trajectory: Mapping[str, Any], *, completion_gate_kind: str = "auto"
) -> None:
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
    completed_prefix_index = next(
        (
            index
            for index, state in enumerate(states)
            if isinstance(state, Mapping) and state.get("evidence_complete") is True
        ),
        len(states) - 1,
    )
    retained_count = completed_prefix_index + 1
    if len(tool_steps) < retained_count:
        raise ValueError("selected trajectory has fewer tool steps than states")
    retained_tool_steps = tool_steps[:retained_count]
    accepted_requests = _accepted_frame_requests(trace, retained_count)
    if len(accepted_requests) != retained_count:
        raise ValueError("accepted Planner frame_select count differs from tool steps")
    for index, raw in enumerate(states):
        if not isinstance(raw, Mapping):
            raise ValueError(f"perception_states[{index}] must be an object")
        step_index = raw.get("step_index", index)
        if step_index != index:
            raise ValueError("perception state indices must be contiguous and ordered")
        _validate_perception_response(
            raw.get("perception_response"),
            allow_role_separated_schema=completion_gate_kind == "visual_csv",
        )
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
        if index < retained_count:
            tool_request = retained_tool_steps[index].get("request")
            if tool_request is None:
                tool_request = {
                    key: retained_tool_steps[index][key]
                    for key in (
                        "start_time",
                        "end_time",
                        "nframes",
                        "fps",
                        "resize",
                        "evidence_request",
                    )
                    if key in retained_tool_steps[index]
                }
            normalized_controller = _normalized_frame_request(
                accepted_requests[index], f"accepted frame_select[{index}]"
            )
            normalized_state = _normalized_frame_request(
                raw.get("request"), f"perception_states[{index}].request"
            )
            normalized_tool = _normalized_frame_request(
                tool_request, f"tool_steps[{index}].request"
            )
            if (
                normalized_controller != normalized_state
                or normalized_state != normalized_tool
            ):
                raise ValueError(
                    f"frame_select[{index}] controller/state/tool request binding differs"
                )
        complete = raw.get("evidence_complete")
        if not isinstance(complete, bool):
            raise ValueError("perception state evidence_complete must be boolean")
    if states[-1].get("evidence_complete") is not True:
        raise ValueError("the final perception prefix must be complete")
    for state in states:
        complete = state.get("evidence_complete") is True
        if completion_gate_kind == "visual_csv":
            if (
                trajectory.get("offline_label_join")
                != "ground_truth_used_for_boolean_only_not_serialized"
                or state.get("completion_gate_kind")
                != "visual_csv_3of3_offline_label"
            ):
                raise ValueError(
                    "visual_csv STOP/CONTINUE labels require the frozen offline label join"
                )
            for field in ("visual_csv_source_sha256", "visual_csv_config_sha256"):
                value = str(state.get(field) or "")
                if len(value) != 64 or any(
                    character not in "0123456789abcdef" for character in value
                ):
                    raise ValueError(f"visual_csv prefix requires lowercase {field}")
        if completion_gate_kind == "visual_csv" or complete:
            _confirmation_gate(
                state,
                prediction,
                expected_complete=complete,
                requested_kind=completion_gate_kind,
            )
    completed_prefix_index = next(
        index for index, state in enumerate(states) if state["evidence_complete"]
    )
    _terminal_requests(
        [dict(item) if isinstance(item, Mapping) else item for item in trace],
        completed_prefix_index=completed_prefix_index,
    )
    declared_classifier = trajectory.get("visual_path_classifier_version")
    if declared_classifier not in (None, VISUAL_PATH_CLASSIFIER_VERSION):
        raise ValueError("selected trajectory visual path classifier version drifted")
    declared_contract = trajectory.get("quality_contract_version")
    if declared_contract not in (None, QUALITY_CONTRACT_VERSION):
        raise ValueError("selected trajectory quality contract version drifted")
    classified = classify_visual_path(trajectory)
    declared_family = trajectory.get("visual_path_family")
    if declared_family not in (None, classified):
        raise ValueError("selected trajectory visual path family is inconsistent")
    _candidate_training_stratum(trajectory)


def build_perception_memory_sft_records(
    trajectory: Mapping[str, Any],
    *,
    include_observer: bool = False,
    completion_gate_kind: str = "auto",
) -> tuple[dict[str, Any], ...]:
    """Export one complete Planner episode and an optional Observer episode."""

    validate_selected_trajectory(
        trajectory, completion_gate_kind=completion_gate_kind
    )
    states = trajectory["perception_states"]
    valid_letters = _public_option_letters(trajectory)
    tool_steps = trajectory["tool_steps"]
    first_complete_index = next(
        index for index, state in enumerate(states) if state["evidence_complete"]
    )
    training_states = states[: first_complete_index + 1]
    requests = _terminal_requests(
        trajectory["request_trace"],
        completed_prefix_index=first_complete_index,
    )
    prediction = str(
        trajectory.get("final_prediction", trajectory.get("prediction"))
    ).upper()
    actual_gate_kinds = {
        _confirmation_gate(
            state,
            prediction,
            expected_complete=bool(state["evidence_complete"]),
            requested_kind=completion_gate_kind,
        )
        for state in training_states
        if completion_gate_kind == "visual_csv" or state["evidence_complete"]
    }
    if len(actual_gate_kinds) != 1:
        raise ValueError("one role episode cannot mix completion gate kinds")
    actual_gate_kind = next(iter(actual_gate_kinds))
    role_separated_planner = actual_gate_kind == "visual_csv"

    planner_messages: list[dict[str, Any]] = []
    planner_turns: list[dict[str, Any]] = []
    initial_requests = _accepted_controller_requests(requests, -1)
    initial_actions: list[tuple[Mapping[str, Any], str, str]] = []
    for request in initial_requests:
        action = _planner_action(request)
        if action:
            initial_actions.append((request, action[0], action[1]))
    if len(initial_actions) != 1:
        raise ValueError("unobserved prefix requires one initial tool/plan target")
    initial_request, initial_type, initial_content = initial_actions[0]
    planner_append = (
        _append_role_separated_planner_turn
        if role_separated_planner
        else _append_snapshot_turn
    )
    planner_append(
        planner_messages,
        initial_request,
        target_type=initial_type,
        target_content=initial_content,
    )
    planner_turns.append(
        _turn_metadata(
            initial_request,
            prefix_index=-1,
            target_type=initial_type,
            target_origin="accepted_planner_trace",
        )
    )

    observer_records: list[dict[str, Any]] = []
    for step_index, state in enumerate(training_states):
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
        response = _validate_perception_response(
            state["perception_response"],
            allow_role_separated_schema=role_separated_planner,
        )
        observed_payload = _normalize_raw_perception_response(
            observation.get("content"),
            valid_letters=valid_letters,
            resolved_start_time=resolved_start,
            resolved_end_time=resolved_end,
            actual_timestamps=actual_timestamps,
            allow_role_separated_schema=role_separated_planner,
        )
        if canonical_sha256(observed_payload) != canonical_sha256(response):
            raise ValueError("perception state differs from the actual model response")
        model_target = state.get("perception_model_target")
        if model_target is None:
            target_payload = response
        else:
            normalized_target = _normalize_raw_perception_response(
                model_target,
                valid_letters=valid_letters,
                resolved_start_time=resolved_start,
                resolved_end_time=resolved_end,
                actual_timestamps=actual_timestamps,
                allow_role_separated_schema=role_separated_planner,
            )
            if canonical_sha256(normalized_target) != canonical_sha256(response):
                raise ValueError(
                    "perception model target differs from normalized state"
                )
            target_payload = _json_object(model_target, "perception_model_target")
        complete = bool(state["evidence_complete"])
        # Sufficiency belongs to the Planner's offline visual-CSV supervision,
        # not to the Observer adapter.  Do not reproduce the old self-report
        # conflict by teaching either field here.
        target_payload = deepcopy(dict(target_payload))
        target_payload.pop("evidence_sufficient", None)
        target_payload.pop("next_evidence_needed", None)
        target_content = _canonical_json(target_payload)
        if include_observer:
            observer_messages: list[dict[str, Any]] = []
            _append_snapshot_turn(
                observer_messages,
                observation,
                target_type="memory",
                target_content=target_content,
            )
            observer_turn = (
                _turn_metadata(
                    observation,
                    prefix_index=step_index,
                    target_type="memory",
                    target_origin="accepted_observer_trace_sufficiency_masked",
                )
            )
            observer_records.append(
                _role_episode_record(
                    trajectory,
                    observer_messages,
                    role="observer",
                    turns=[observer_turn],
                    completion_gate_kind=actual_gate_kind,
                    terminal_prefix_index=step_index,
                    episode_schema="observer_current_frame_episode_v1",
                    episode_id=(
                        f"{trajectory['trajectory_id']}#role-observer"
                        f"#prefix-{step_index:03d}"
                    ),
                )
            )

        if not complete:
            controller_requests = _accepted_controller_requests(requests, step_index)
            actions: list[tuple[Mapping[str, Any], str, str]] = []
            for request in controller_requests:
                action = _planner_action(request)
                if action:
                    actions.append((request, action[0], action[1]))
            if len(actions) != 1:
                raise ValueError(
                    f"incomplete prefix {step_index} requires one next tool/plan target"
                )
            request, target_type, target_content = actions[0]
            planner_append(
                planner_messages,
                request,
                target_type=target_type,
                target_content=target_content,
            )
            planner_turns.append(
                _turn_metadata(
                    request,
                    prefix_index=step_index,
                    target_type=target_type,
                    target_origin=f"{actual_gate_kind}_3of3_incomplete",
                )
            )
            continue

        accepted_history = _accepted_planner_history(planner_messages)
        stop_request = _synthetic_stop_request(
            trajectory,
            requests,
            state,
            step_index,
            accepted_actions=accepted_history,
            role_separated=role_separated_planner,
        )
        planner_append(
            planner_messages,
            stop_request,
            target_type="stop",
            target_content='{"action":"stop"}',
        )
        planner_turns.append(
            _turn_metadata(
                stop_request,
                prefix_index=step_index,
                target_type="stop",
                target_origin=(
                    "offline_visual_csv_stop"
                    if actual_gate_kind == "visual_csv"
                    else f"{actual_gate_kind}_3of3_complete"
                ),
            )
        )
        break

    records = [
        _role_episode_record(
            trajectory,
            planner_messages,
            role="planner",
            turns=planner_turns,
            completion_gate_kind=actual_gate_kind,
            terminal_prefix_index=first_complete_index,
            episode_schema="planner_complete_episode_v1",
            episode_id=f"{trajectory['trajectory_id']}#role-planner",
        )
    ]
    if include_observer:
        records.extend(observer_records)
    target_counts = Counter(
        target for record in records for target in record["metadata"]["assistant_target_types"]
    )
    if target_counts["stop"] != 1 or target_counts["final"]:
        raise ValueError("Planner episode must export exactly one stop and no Judge target")
    if include_observer and target_counts["memory"] != len(training_states):
        raise ValueError(
            "every retained perception prefix must export one memory target"
        )
    return tuple(records)


def summarize_perception_memory_sft(
    trajectories: Iterable[Mapping[str, Any]],
    records: Iterable[Mapping[str, Any]] = (),
    *,
    completion_gate_kind: str = "auto",
) -> dict[str, Any]:
    rows = [dict(row) for row in trajectories]
    exported = [dict(record) for record in records]
    by_dataset: Counter[str] = Counter()
    fixes_by_dataset: Counter[str] = Counter()
    candidate_strata: Counter[str] = Counter()
    path_families: Counter[str] = Counter()
    prefixes = Counter()
    seen: set[tuple[str, str]] = set()
    for row in rows:
        validate_selected_trajectory(row, completion_gate_kind=completion_gate_kind)
        identity = (str(row.get("dataset") or ""), str(row.get("sample_id") or ""))
        if identity in seen:
            raise ValueError(f"duplicate selected sample: {identity[0]}/{identity[1]}")
        seen.add(identity)
        by_dataset[identity[0]] += 1
        candidate = str(row.get("candidate_answer") or "").strip().upper()
        prediction = str(row.get("final_prediction") or "").strip().upper()
        if candidate and candidate != prediction:
            fixes_by_dataset[identity[0]] += 1
        candidate_strata[_candidate_training_stratum(row)] += 1
        path_families[classify_visual_path(row)] += 1
        prefixes["unobserved"] += 1
        first_complete_index = next(
            index
            for index, state in enumerate(row["perception_states"])
            if state["evidence_complete"]
        )
        for state in row["perception_states"][: first_complete_index + 1]:
            prefixes["complete" if state["evidence_complete"] else "incomplete"] += 1
    targets: Counter[str] = Counter()
    roles: Counter[str] = Counter()
    record_trajectory_ids: set[str] = set()
    role_episode_ids: set[str] = set()
    observer_prefixes: set[tuple[str, int]] = set()
    for record in exported:
        validate_exported_sft_record(record)
        metadata = record["metadata"]
        targets.update(metadata["assistant_target_types"])
        trajectory_id = str(metadata.get("trajectory_id") or "")
        role = str(metadata.get("process_role") or "")
        if role not in {"planner", "observer"}:
            raise ValueError("process-SFT record has invalid role")
        episode_id = str(metadata.get("episode_id") or "")
        if not episode_id or episode_id in role_episode_ids:
            raise ValueError("duplicate or missing role episode_id")
        role_episode_ids.add(episode_id)
        if role == "observer":
            prefix_index = metadata.get("terminal_prefix_index")
            if (
                isinstance(prefix_index, bool)
                or not isinstance(prefix_index, int)
                or prefix_index < 0
            ):
                raise ValueError("Observer current-frame episode has invalid prefix")
            observer_pair = (trajectory_id, prefix_index)
            if observer_pair in observer_prefixes:
                raise ValueError("duplicate Observer current-frame episode")
            observer_prefixes.add(observer_pair)
        roles[role] += 1
        record_trajectory_ids.add(trajectory_id)
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
        if roles["planner"] != len(rows) or targets["stop"] != len(rows):
            raise ValueError("every selected trajectory requires one Planner episode")
        if roles["observer"] not in {0, expected_memory_targets}:
            raise ValueError("Observer current-frame episodes must cover every prefix or none")
        if targets["memory"] != (
            expected_memory_targets if roles["observer"] else 0
        ) or targets["final"]:
            raise ValueError(
                "process-SFT target coverage does not match selected prefixes"
            )
    observed_continue = prefixes["incomplete"]
    stop_count = targets["stop"] if exported else prefixes["complete"]
    return {
        "quality_contract_version": QUALITY_CONTRACT_VERSION,
        "visual_path_classifier_version": VISUAL_PATH_CLASSIFIER_VERSION,
        "selected_trajectories": len(rows),
        "selected_by_dataset": dict(sorted(by_dataset.items())),
        "candidate_fixes": sum(fixes_by_dataset.values()),
        "candidate_fixes_by_dataset": dict(sorted(fixes_by_dataset.items())),
        "candidate_training_strata": {
            name: candidate_strata[name] for name in CANDIDATE_TRAINING_STRATA
        },
        "visual_path_distribution": {
            name: path_families[name] for name in VISUAL_PATH_FAMILIES
        },
        "prefixes": dict(sorted(prefixes.items())),
        "planner_decisions": {
            "observed_incomplete_continue": observed_continue,
            "stop": stop_count,
        },
        "sft_records": len(exported),
        "record_coverage_passed": bool(exported),
        "role_episodes": dict(sorted(roles.items())),
        "assistant_targets": dict(sorted(targets.items())),
    }


def _require_balanced_pair(
    counts: Mapping[str, int], first: str, second: str, label: str
) -> None:
    left = int(counts.get(first, 0))
    right = int(counts.get(second, 0))
    if left <= 0 or right <= 0:
        raise ValueError(f"{label} requires both non-empty strata")
    ratio = left / right
    if not QUALITY_RATIO_MIN <= ratio <= QUALITY_RATIO_MAX:
        raise ValueError(
            f"{label} ratio must be in [{QUALITY_RATIO_MIN}, {QUALITY_RATIO_MAX}], "
            f"found {left}:{right}"
        )


def _enforce_visual_csv_quality(summary: Mapping[str, Any]) -> None:
    if summary.get("quality_contract_version") != QUALITY_CONTRACT_VERSION:
        raise ValueError("process-SFT quality contract version drifted")
    if (
        summary.get("visual_path_classifier_version")
        != VISUAL_PATH_CLASSIFIER_VERSION
    ):
        raise ValueError("visual path classifier version drifted")
    _require_balanced_pair(
        summary["candidate_training_strata"],
        "candidate_correct",
        "candidate_wrong",
        "candidate training strata",
    )
    _require_balanced_pair(
        summary["planner_decisions"],
        "observed_incomplete_continue",
        "stop",
        "observed CONTINUE/STOP",
    )
    distribution = summary["visual_path_distribution"]
    counts = [int(distribution.get(name, 0)) for name in VISUAL_PATH_FAMILIES]
    if any(count <= 0 for count in counts):
        raise ValueError("visual path balance requires all four non-empty families")
    ratio = max(counts) / min(counts)
    if ratio > QUALITY_RATIO_MAX:
        raise ValueError(
            f"visual path max/min ratio must be <= {QUALITY_RATIO_MAX}, found {ratio:.6f}"
        )


def enforce_perception_memory_selection_gate(
    trajectories: Iterable[Mapping[str, Any]],
    records: Iterable[Mapping[str, Any]] = (),
    *,
    completion_gate_kind: str = "auto",
) -> dict[str, Any]:
    """Validate quality/provenance gates; sample counts remain advisory metadata."""

    summary = summarize_perception_memory_sft(
        trajectories, records, completion_gate_kind=completion_gate_kind
    )
    if completion_gate_kind == "visual_csv":
        _enforce_visual_csv_quality(summary)
    return summary


__all__ = [
    "CANDIDATE_TRAINING_STRATA",
    "QUALITY_CONTRACT_VERSION",
    "QUALITY_RATIO_MAX",
    "QUALITY_RATIO_MIN",
    "VISUAL_PATH_CLASSIFIER_VERSION",
    "VISUAL_PATH_FAMILIES",
    "build_perception_memory_sft_records",
    "classify_visual_path",
    "enforce_perception_memory_selection_gate",
    "summarize_perception_memory_sft",
    "validate_selected_trajectory",
]
