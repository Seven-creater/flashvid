"""Export selected Fast Hybrid EVA traces into ms-swift SFT episodes."""

from __future__ import annotations

import json
import re
from copy import deepcopy
from pathlib import Path
from typing import Any, Iterable, Mapping

from .answers import extract_strict_answer_letter
from .privacy import assert_deferred_result_public
from .qwen_sft import build_sft_record, validate_exported_sft_record


_TOOL_BLOCK = re.compile(r"<tool_call>.*?</tool_call>", re.DOTALL)


def _terminal_requests(trace: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Keep the final retry for each immutable model prompt."""

    groups: dict[tuple[str, str, int], list[tuple[int, dict[str, Any]]]] = {}
    for index, raw in enumerate(trace):
        request = dict(raw)
        key = (
            str(request.get("stage") or request.get("branch") or ""),
            str(request.get("prompt_hash") or ""),
            int(request.get("seed") or 0),
        )
        groups.setdefault(key, []).append((index, request))
    selected: list[tuple[int, dict[str, Any]]] = []
    for requests in groups.values():
        position, terminal = max(
            requests, key=lambda item: int(item[1].get("attempt_index") or 0)
        )
        if terminal.get("error"):
            raise ValueError("terminal Fast Hybrid request is an API error")
        if terminal.get("finish_reason") == "length":
            raise ValueError("terminal Fast Hybrid request is truncated")
        selected.append((position, terminal))
    return [request for _position, request in sorted(selected)]


def _action(content: str) -> tuple[str, str]:
    blocks = _TOOL_BLOCK.findall(content or "")
    if blocks:
        for block in blocks:
            payload = json.loads(block[len("<tool_call>") : -len("</tool_call>")])
            if not isinstance(payload, Mapping) or payload.get("tool") != "frame_select":
                raise ValueError("Fast Hybrid trace contains a non-frame_select tool")
        return "tool", "\n".join(blocks)
    answer = extract_strict_answer_letter(content, tuple("ABCDEFGH"))
    if answer is not None:
        return "answer", f"Answer: {answer}"
    raise ValueError("Fast Hybrid response has neither an official tool call nor strict answer")


def _episode_messages(
    request: Mapping[str, Any], *, target_type: str, target_content: str
) -> list[dict[str, Any]]:
    source = request.get("messages")
    if not isinstance(source, list) or not source:
        raise ValueError("Fast Hybrid request has no message snapshot")
    messages: list[dict[str, Any]] = []
    for index, raw in enumerate(source):
        if not isinstance(raw, Mapping):
            raise ValueError(f"request messages[{index}] must be an object")
        role = str(raw.get("role") or "")
        if role not in {"system", "user", "assistant", "tool"}:
            raise ValueError(f"unsupported Fast Hybrid message role: {role!r}")
        message = {"role": role, "content": deepcopy(raw.get("content", ""))}
        if role == "assistant":
            message["target_type"] = "context"
        messages.append(message)
    if messages[0]["role"] != "system" or messages[-1]["role"] not in {"user", "tool"}:
        raise ValueError("Fast Hybrid request snapshot has invalid role ordering")
    messages.append(
        {
            "role": "assistant",
            "content": target_content,
            "target_type": target_type,
        }
    )
    return messages


def build_fast_hybrid_sft_records(
    trajectory: Mapping[str, Any],
) -> tuple[dict[str, Any], ...]:
    """Create one loss-isolated episode for each Fast Hybrid assistant action."""

    assert_deferred_result_public(trajectory)
    if trajectory.get("_selection_stable") is not True:
        raise ValueError("Fast Hybrid SFT requires a 3/3-stable selected trajectory")
    raw_trace = trajectory.get("request_trace")
    if not isinstance(raw_trace, list) or not raw_trace:
        raise ValueError("selected Fast Hybrid trajectory has no request_trace")
    terminal = _terminal_requests(raw_trace)
    classified = [(_action(str(item.get("content") or "")), item) for item in terminal]
    answer_indices = [
        index for index, ((kind, _), _request) in enumerate(classified) if kind == "answer"
    ]
    if not answer_indices:
        raise ValueError("Fast Hybrid trajectory has no final answer action")
    final_index = max(answer_indices)
    prediction = str(
        trajectory.get("final_prediction", trajectory.get("prediction")) or ""
    ).strip().upper()
    final_answer = extract_strict_answer_letter(
        classified[final_index][0][1], tuple("ABCDEFGH")
    )
    if final_answer != prediction:
        raise ValueError("final SFT answer differs from Fast Hybrid final_prediction")

    records: list[dict[str, Any]] = []
    for index, ((kind, content), request) in enumerate(classified):
        # A verifier answer that triggers the external candidate gate is only a
        # proposal.  Training it (especially when the confirmation later
        # rejects it) would supervise an answer the complete Agent did not
        # accept.  Keep every official tool action, but only the single final
        # accepted answer.
        if kind == "answer" and index != final_index:
            continue
        target_type = "tool" if kind == "tool" else "final"
        materialized = dict(trajectory)
        materialized["training_messages"] = _episode_messages(
            request,
            target_type=target_type,
            target_content=content,
        )
        record = build_sft_record(
            materialized,
            require_complete_trajectory=False,
            episode_metadata={
                "episode_id": f"{trajectory['trajectory_id']}#turn-{index:03d}",
                "turn_index": index,
                "stage": request.get("stage"),
                "prompt_hash": request.get("prompt_hash"),
                "finish_reason": request.get("finish_reason"),
                "episode_target_type": target_type,
                "usage": deepcopy(request.get("usage") or {}),
            },
        )
        validate_exported_sft_record(record)
        records.append(record)
    if sum(
        record["metadata"]["assistant_target_types"].count("final")
        for record in records
    ) != 1:
        raise ValueError("Fast Hybrid trajectory must export exactly one final target")
    if not any(
        "tool" in record["metadata"]["assistant_target_types"] for record in records
    ):
        raise ValueError("Fast Hybrid trajectory must export at least one tool target")
    return tuple(records)


def validate_fast_hybrid_frame_files(trajectory: Mapping[str, Any]) -> None:
    calls = trajectory.get("tool_steps", trajectory.get("tool_calls"))
    if not isinstance(calls, list) or not calls:
        raise ValueError("Fast Hybrid trajectory has no tool calls")
    for call_index, call in enumerate(calls):
        if not isinstance(call, Mapping):
            raise ValueError(f"tool call {call_index} is not an object")
        paths = call.get("frame_paths")
        timestamps = call.get("actual_timestamps", call.get("timestamps"))
        if not isinstance(paths, list) or not paths:
            raise ValueError(f"tool call {call_index} has no frame paths")
        if not isinstance(timestamps, list) or len(timestamps) != len(paths):
            raise ValueError(f"tool call {call_index} frame/timestamp count differs")
        for path in paths:
            resolved = Path(str(path)).expanduser()
            if not resolved.is_absolute() or not resolved.is_file():
                raise ValueError(f"missing Fast Hybrid frame file: {resolved}")
