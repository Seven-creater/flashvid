from __future__ import annotations

import json
import math
import random
import re
import time
from copy import copy
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import replace
from pathlib import Path
from typing import Any, Iterable, Mapping

from .answers import extract_answer_letter, extract_strict_answer_letter
from .client import OpenAICompatibleClient
from .datasets import VideoIndex
from .eva_official import select_frames as official_select_frames
from .media import estimate_visual_tokens, probe_video
from .privacy import assert_deferred_result_public
from .schemas import ModelSample, Sample, ScoringRecord


def format_question(sample: Sample) -> str:
    choices = "\n".join(f"{letter}: {text}" for letter, text in sample.choices.items())
    return (
        "Answer the video multiple-choice question. Return only one option letter "
        "from the choices, using the format `Answer: X`.\n\n"
        f"Question: {sample.question}\n{choices}"
    )


def format_text_only_question(sample: Sample) -> str:
    choices = "\n".join(f"{letter}: {text}" for letter, text in sample.choices.items())
    return (
        "Answer the multiple-choice question using only the question and answer "
        "choices below. No video, images, subtitles, timestamps, or other visual "
        "evidence are provided. Choose the best option and return only one option "
        "letter using the format `Answer: X`.\n\n"
        f"Question: {sample.question}\n{choices}"
    )


def _content_with_video(video: Path, text: str) -> list[dict[str, Any]]:
    return [
        {"type": "video_url", "video_url": {"url": video.as_uri()}},
        {"type": "text", "text": text},
    ]


_TIMESTAMP = re.compile(r"(?<!\d)(?:(\d{1,2}):)?([0-5]?\d):([0-5]\d)(?!\d)")


def _timestamp_seconds(match: re.Match[str]) -> float:
    hours = int(match.group(1) or 0)
    minutes = int(match.group(2))
    seconds = int(match.group(3))
    return float(hours * 3600 + minutes * 60 + seconds)


def parse_question_time_range(question: str) -> tuple[float, float] | None:
    """Parse timestamps present in the question text, never dataset metadata."""

    matches = list(_TIMESTAMP.finditer(question))
    if not matches:
        return None
    first = _timestamp_seconds(matches[0])
    if len(matches) == 1:
        return max(0.0, first - 1.0), first + 1.0
    between = question[matches[0].end():matches[1].start()]
    if not re.search(r"(?:-|\u2013|\u2014|~|to|through|until|from)", between, re.IGNORECASE):
        return None
    second = _timestamp_seconds(matches[1])
    return (min(first, second), max(first, second))


def _parse_tool_calls(text: str) -> list[dict[str, Any]]:
    """Parse only official EVA ``<tool_call>`` wrapped frame requests."""

    decoder = json.JSONDecoder()
    calls: list[dict[str, Any]] = []
    for block in re.findall(r"<tool_call>(.*?)</tool_call>", text, flags=re.DOTALL):
        cursor = 0
        while cursor < len(block):
            start = block.find("{", cursor)
            if start < 0:
                break
            try:
                payload, consumed = decoder.raw_decode(block[start:])
            except json.JSONDecodeError:
                cursor = start + 1
                continue
            cursor = start + consumed
            if not isinstance(payload, dict) or payload.get("tool") != "frame_select":
                continue
            arguments = payload.get("arguments", payload.get("parameters", {}))
            if not isinstance(arguments, dict):
                continue
            try:
                call = {
                    "start_time": float(arguments["start_time"]),
                    "end_time": float(arguments["end_time"]),
                    "nframes": int(arguments["nframes"]),
                    "resize": float(arguments.get("resize", 1.0)),
                }
            except (KeyError, TypeError, ValueError):
                continue
            if (
                call["end_time"] <= call["start_time"]
                or not 1 <= call["nframes"] <= 64
                or not 0.05 <= call["resize"] <= 2.0
            ):
                continue
            calls.append(call)
    return calls


def _ranges_overlap(left: tuple[float, float], right: tuple[float, float]) -> bool:
    return max(left[0], right[0]) < min(left[1], right[1])


def _ranges_redundant(left: tuple[float, float], right: tuple[float, float]) -> bool:
    intersection = max(0.0, min(left[1], right[1]) - max(left[0], right[0]))
    union = max(left[1], right[1]) - min(left[0], right[0])
    return union > 0.0 and intersection / union >= 0.5


def _fit_tool_budget(
    metadata: dict[str, float | int],
    call: dict[str, Any],
    budget: int,
) -> tuple[dict[str, Any], int] | None:
    """Reduce resolution first, then frame count, to fit a visual-token budget."""

    if budget <= 0:
        return None
    fitted = dict(call)
    estimated = estimate_visual_tokens(metadata, fitted["nframes"], fitted["resize"])
    if estimated > budget:
        resize_factor = math.sqrt(budget / estimated)
        fitted["resize"] = max(0.05, fitted["resize"] * resize_factor)
        estimated = estimate_visual_tokens(metadata, fitted["nframes"], fitted["resize"])
    if estimated > budget:
        fitted["nframes"] = max(1, int(fitted["nframes"] * budget / estimated))
        estimated = estimate_visual_tokens(metadata, fitted["nframes"], fitted["resize"])
    if estimated > budget:
        return None
    return fitted, estimated


def _tool_content(
    frames: list[Path],
    timestamps: list[float],
    question: str,
    observed_intervals: list[tuple[float, float]],
    remaining_visual_tokens: int,
) -> list[dict[str, Any]]:
    """Build the official EVA-style tool response message."""
    content: list[dict[str, Any]] = [
        {"type": "text", "text": "<tool_response>"}
    ]
    for frame, timestamp in zip(frames, timestamps):
        content.append({"type": "text", "text": f"\nFrame at {int(timestamp)} seconds:"})
        content.append({"type": "image_url", "image_url": {"url": f"file://{frame}"}})
    content.append(
        {
            "type": "text",
            "text": (
                "\nObserved intervals (seconds): "
                f"{[[round(a, 2), round(b, 2)] for a, b in observed_intervals]}. "
                f"Remaining visual-token budget: {remaining_visual_tokens}. "
                "If evidence is sufficient, answer exactly as Answer: X. Otherwise, briefly "
                "identify the unresolved options and call a non-overlapping interval.\n\n"
                f"Question: {question}</tool_response>"
            ),
        }
    )
    return content


_AGENT_CONFIGS: dict[str, dict[str, Any]] = {
    "v2a": {"overview_frames": 12, "overview_resize": 0.25, "max_calls_per_turn": 1},
    "v2b": {"overview_frames": 16, "overview_resize": 0.20, "max_calls_per_turn": 1},
    "v2c": {"overview_frames": 16, "overview_resize": 0.20, "max_calls_per_turn": 2},
    "v2d": {"overview_frames": 12, "overview_resize": 0.35, "max_calls_per_turn": 1},
    # Accuracy-first verifier used after a separate Direct candidate call.
    "hybrid_v1": {"overview_frames": 16, "overview_resize": 0.75, "max_calls_per_turn": 2},
    "hybrid_v2": {"overview_frames": 16, "overview_resize": 0.75, "max_calls_per_turn": 2},
    # Hybrid v3 explores independent verifier strategies while keeping Direct
    # candidate fallback semantics unchanged.
    "hybrid_v3a": {
        "overview_frames": 12,
        "overview_resize": 0.35,
        "max_calls_per_turn": 1,
        "tool_round_limit": 4,
    },
    "hybrid_v3b": {
        "overview_frames": 16,
        "overview_resize": 0.35,
        "max_calls_per_turn": 1,
        "tool_round_limit": 5,
    },
    "hybrid_v3c": {
        "overview_frames": 16,
        "overview_resize": 0.35,
        "max_calls_per_turn": 1,
        "tool_round_limit": 6,
    },
    "hybrid_v3d": {
        "overview_frames": 20,
        "overview_resize": 0.30,
        "max_calls_per_turn": 2,
        "tool_round_limit": 6,
    },
    "hybrid_v3e": {
        "overview_frames": 16,
        "overview_resize": 0.75,
        "max_calls_per_turn": 2,
        "tool_round_limit": 6,
    },
    "hybrid_v3f": {
        "overview_frames": 16,
        "overview_resize": 0.35,
        "max_calls_per_turn": 1,
        "tool_round_limit": 6,
    },
    # Meta-controller only; hybrid() delegates each question to v2 or v3c.
    "hybrid_v3g": {
        "overview_frames": 16,
        "overview_resize": 0.35,
        "max_calls_per_turn": 1,
        "tool_round_limit": 6,
    },
}

_HYBRID_V3_ROUTE_CALLS: dict[str, dict[str, dict[str, float | int]]] = {
    "hybrid_v3a": {
        "global_overview": {"nframes": 12, "resize": 0.35},
        "temporal_event": {"nframes": 12, "resize": 0.35},
        "action_event": {"nframes": 12, "resize": 0.40},
        "ocr_detail": {"nframes": 8, "resize": 0.70},
    },
    "hybrid_v3b": {
        "global_overview": {"nframes": 16, "resize": 0.32},
        "temporal_event": {"nframes": 12, "resize": 0.42},
        "action_event": {"nframes": 12, "resize": 0.45},
        "ocr_detail": {"nframes": 8, "resize": 0.85},
    },
    "hybrid_v3c": {
        "global_overview": {"nframes": 16, "resize": 0.32},
        "temporal_event": {"nframes": 12, "resize": 0.42},
        "action_event": {"nframes": 12, "resize": 0.45},
        "ocr_detail": {"nframes": 8, "resize": 0.85},
    },
    "hybrid_v3d": {
        "global_overview": {"nframes": 20, "resize": 0.30},
        "temporal_event": {"nframes": 16, "resize": 0.35},
        "action_event": {"nframes": 16, "resize": 0.40},
        "ocr_detail": {"nframes": 10, "resize": 0.80},
    },
    "hybrid_v3e": {
        "global_overview": {"nframes": 16, "resize": 0.75},
        "temporal_event": {"nframes": 12, "resize": 0.42},
        "action_event": {"nframes": 12, "resize": 0.45},
        "ocr_detail": {"nframes": 16, "resize": 0.75},
    },
    "hybrid_v3f": {
        "global_overview": {"nframes": 16, "resize": 0.32},
        "temporal_event": {"nframes": 12, "resize": 0.42},
        "action_event": {"nframes": 12, "resize": 0.45},
        "ocr_detail": {"nframes": 8, "resize": 0.85},
    },
}

_UNCERTAIN_LANGUAGE_RE = re.compile(
    r"(?is)\b("
    r"unclear|not clear|not visible|not directly visible|ambiguous|missing|"
    r"insufficient|cannot|can't|could be|might|maybe|likely|appears|seems|"
    r"suggests|implied|no clear|doesn't show|do not show|not enough evidence|"
    r"uncertain|uncertainty|guess|guessed|hard to tell|can't tell|cannot tell|"
    r"no obvious|not obvious|maybe|probably|possibly"
    r")\b"
)
_EVENT_CONFIRMATION_KEYWORDS = re.compile(
    r"(?is)\b("
    r"after|before|while|during|then|first|last|next|sequence|steps|"
    r"what did|what does|what was|how did|what happens|what is the change"
    r")\b"
)
_OCR_DETAIL_KEYWORDS = re.compile(
    r"(?is)\b("
    r"text|written|writing|word|words|read|reads|says|sign|subtitle|caption|"
    r"label|logo|screen|display|banner|poster|letter|number|symbol"
    r")\b"
)
_ACTION_EVENT_KEYWORDS = re.compile(
    r"(?is)\b("
    r"what did .+ do|what does .+ do|what action|what activity|how did|how does|"
    r"gesture|movement|move|turn|pick up|put down|open|close|enter|leave|"
    r"change in|changes in|reaction|interact"
    r")\b"
)
_TEMPORAL_ROUTE_KEYWORDS = re.compile(
    r"(?is)\b("
    r"before|after|while|during|then|first|last|next|earlier|later|sequence|"
    r"order|timeline|at the end|at the beginning|finally"
    r")\b"
)


def _initial_tool_call(
    question: str,
    duration: float,
    version: str,
    metadata: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], str]:
    explicit = parse_question_time_range(question)
    if explicit is not None:
        start_time = max(0.0, min(explicit[0] - 1.0, duration - 0.1))
        end_time = min(duration, explicit[1] + 1.0)
        return (
            {
                "start_time": start_time,
                "end_time": max(start_time + 0.1, end_time),
                "nframes": 8,
                "resize": 1.0,
            },
            "explicit_question_time",
        )
    route = _question_route(question, metadata or {}) if version in _HYBRID_V3_ROUTE_CALLS else "global_overview"
    config = _AGENT_CONFIGS[version]
    if version in _HYBRID_V3_ROUTE_CALLS:
        route_call = _HYBRID_V3_ROUTE_CALLS[version].get(route)
        if route_call is not None:
            return (
                {
                    "start_time": 0.0,
                    "end_time": max(0.1, duration),
                    "nframes": int(route_call["nframes"]),
                    "resize": float(route_call["resize"]),
                },
                route,
            )
    return (
        {
            "start_time": 0.0,
            "end_time": max(0.1, duration),
            "nframes": int(config["overview_frames"]),
            "resize": float(config["overview_resize"]),
        },
        route,
    )


def _agent_system_prompt(
    metadata: dict[str, float | int],
    initial_call: dict[str, Any],
    route: str,
    version: str,
    max_call_visual_tokens: int,
    max_total_visual_tokens: int,
    candidate_answer: str | None = None,
) -> str:
    config = _AGENT_CONFIGS[version]
    first_action = json.dumps(
        {"tool": "frame_select", "arguments": initial_call},
        ensure_ascii=False,
        separators=(",", ":"),
    )
    version_instruction = ""
    if version == "v2b":
        version_instruction = (
            "Before selecting a zoom interval, eliminate options using the overview and choose "
            "the interval that best separates the remaining options. "
        )
    elif version == "v2c":
        version_instruction = (
            "Before any second action, compare evidence supporting and contradicting every option. "
            "You may request up to two disjoint intervals only when they test different unresolved options. "
        )
    elif version == "v2d":
        version_instruction = (
            "For a global overview, you must request exactly one non-overlapping local zoom before any answer, "
            "even if the overview suggests a choice. Use the zoom to verify the most discriminative detail. "
        )
    elif version == "hybrid_v1":
        version_instruction = (
            "This is a hypothesis-verification task. Independently identify the evidence required by the question, "
            "then actively search for evidence that could disprove the Direct candidate. Record support and opposition "
            "for each option, and request new non-overlapping evidence whenever uncertainty remains. Change away from "
            "the Direct candidate only when selected frames clearly contradict it and support another option. "
        )
    elif version == "hybrid_v2":
        version_instruction = (
            "This is a conservative hypothesis-verification task. Treat the Direct candidate as the default answer. "
            "Do not fill missing parts of an action chain from common sense or prior knowledge. Change away from the "
            "Direct candidate only when selected frames show a visible contradiction and directly support another "
            "option. If evidence is unclear, implied, incomplete, or not visible, keep the Direct candidate. For action, "
            "event, and time-order questions on a global overview route, request one additional non-overlapping "
            "confirmation zoom before any changed answer. Final answers must be one line only: Answer: X. "
        )
    elif version.startswith("hybrid_v3"):
        version_instruction = _hybrid_v3_instruction(version, route)
    candidate_instruction = ""
    if candidate_answer:
        if version.startswith("hybrid_v3"):
            candidate_instruction = (
                f"Direct candidate hypothesis: {candidate_answer}. It is not ground truth. Treat it as the default "
                "only until selected frames directly contradict it; do not replace it on implication or memory. "
            )
        else:
            candidate_instruction = (
                f"Direct candidate hypothesis: {candidate_answer}. It is not ground truth. Try to disprove it with "
                "visual evidence before accepting it; do not copy it without evidence. "
            )
    return (
        "You are a query-driven long-video QA agent using the official EVA frame-selection protocol. "
        "You never receive the full video and must answer only from selected frames. "
        f"Video duration: {float(metadata['duration']):.3f} seconds. Resolution: "
        f"{int(metadata['width'])}x{int(metadata['height'])}. "
        "All tool times are seconds; MM:SS means minutes and seconds, never a decimal. "
        "Internally follow: identify evidence type, summarize observed evidence, eliminate choices, "
        "then select the most discriminative unseen interval. Never use dataset annotations. "
        f"The question-derived route is {route}. The controller will execute this first action "
        f"before your first reply: <tool_call>{first_action}</tool_call>. "
        "After observing it, either answer exactly as Answer: X or emit official calls in the same wrapper. "
        "For a global_overview route, request one targeted zoom before answering unless the overview "
        "already contains direct, unambiguous evidence. "
        "Do not repeat an observed interval. A zoom request should normally contain 6 to 8 frames. "
        f"At most {int(config['max_calls_per_turn'])} call(s) are allowed per turn; each call has "
        f"a {max_call_visual_tokens} visual-token limit and the whole sample has a "
        f"{max_total_visual_tokens} visual-token limit. "
        f"{candidate_instruction}{version_instruction}Do not reveal chain-of-thought or use knowledge not supported by frames."
    )


def _final_answer_prompt(version: str, candidate_answer: str | None = None) -> str:
    if version.startswith("hybrid_v3") and candidate_answer:
        return (
            f"Final turn. The Direct candidate is {candidate_answer} and stays the default unless the selected "
            "frames clearly contradict it. Use only observed frames, do not explain, and output exactly one line: "
            "Answer: X."
        )
    if version.startswith("hybrid_v3"):
        return (
            "Final turn. Use only observed frames, do not explain, and output exactly one line: Answer: X."
        )
    if version == "hybrid_v2" and candidate_answer:
        return (
            f"Final turn. The Direct candidate is {candidate_answer} and remains the default. "
            "Use only selected frames. If they do not clearly contradict the Direct candidate and directly support "
            "another option, output the Direct candidate. Do not explain. Output exactly one line: Answer: X."
        )
    if version == "hybrid_v1" and candidate_answer:
        return (
            f"Final turn. The Direct candidate is {candidate_answer}. Internally compare visual evidence for and "
            "against every option. Change away from the Direct candidate only if observed frames clearly contradict "
            "it and directly support another choice; if evidence is ambiguous or missing, output the Direct candidate. "
            "Output only the best supported choice exactly as Answer: X. Do not request another tool."
        )
    if version in {"v2c", "v2d", "hybrid_v1", "hybrid_v2"}:
        return (
            "Final turn. Internally compare visual evidence for and against every option, explicitly test whether "
            "the Direct candidate is contradicted, then output "
            "only the best supported choice exactly as Answer: X. Do not request another tool."
        )
    return (
        "Final turn. Eliminate unsupported choices using only observed frames and output exactly "
        "Answer: X. Do not request another tool."
    )


def _hybrid_change_confirmation_prompt(candidate_answer: str, proposed_answer: str) -> str:
    return (
        f"You proposed Answer: {proposed_answer}, which differs from the Direct candidate {candidate_answer}. "
        "Re-check all observed frames. Keep the changed answer only if a specific observed visual detail contradicts "
        "the Direct candidate and supports the changed answer. If evidence is ambiguous, incomplete, or not visible, "
        f"return to the Direct candidate. Output exactly Answer: {proposed_answer} or Answer: {candidate_answer}."
    )


def _hybrid_v2_confirmation_tool_prompt(candidate_answer: str, proposed_answer: str) -> str:
    return (
        f"You proposed Answer: {proposed_answer}, which differs from the Direct candidate {candidate_answer}. "
        "Before changing the answer, request one additional non-overlapping frame_select interval that directly "
        "checks the visual contradiction. Emit only the official <tool_call> JSON now."
    )


def _minimum_tool_rounds(version: str, route: str, candidate_answer: str | None) -> int:
    if route != "global_overview":
        if version == "hybrid_v3a" and route == "ocr_detail":
            return 1
        if version.startswith("hybrid_v3") and route != "explicit_question_time":
            return 2
        return 1
    if version == "v2d":
        return 2
    if version.startswith("hybrid_v3"):
        return 2
    if version in {"hybrid_v1", "hybrid_v2"} and candidate_answer is not None:
        return 2
    return 1


def _usage_sum(records: list[dict[str, Any]]) -> dict[str, int]:
    return {
        key: sum(int(record.get("usage", {}).get(key, 0) or 0) for record in records)
        for key in ("prompt_tokens", "completion_tokens", "total_tokens")
    }


def _usage_fields(usage: dict[str, Any]) -> dict[str, int]:
    return {
        "prompt_tokens": int(usage.get("prompt_tokens", 0) or 0),
        "completion_tokens": int(usage.get("completion_tokens", 0) or 0),
        "total_tokens": int(usage.get("total_tokens", 0) or 0),
    }


def _question_type_labels(metadata: dict[str, Any]) -> set[str]:
    raw = metadata.get("question_type")
    if raw is None:
        return set()
    if isinstance(raw, list):
        return {str(item).strip().lower() for item in raw if str(item).strip()}
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            parsed = None
        if isinstance(parsed, list):
            return {str(item).strip().lower() for item in parsed if str(item).strip()}
        try:
            import ast

            parsed = ast.literal_eval(raw)
        except Exception:
            parsed = None
        if isinstance(parsed, list):
            return {str(item).strip().lower() for item in parsed if str(item).strip()}
        return {raw.strip().lower()} if raw.strip() else set()
    return {str(raw).strip().lower()} if str(raw).strip() else set()


def _question_route(question: str, metadata: dict[str, Any]) -> str:
    # Routing is derived from the question text only. Dataset question_type is
    # retained for offline audit, but must not influence what evidence the model sees.
    del metadata
    if parse_question_time_range(question) is not None:
        return "explicit_question_time"
    if _OCR_DETAIL_KEYWORDS.search(question):
        return "ocr_detail"
    if _ACTION_EVENT_KEYWORDS.search(question):
        return "action_event"
    if _TEMPORAL_ROUTE_KEYWORDS.search(question):
        return "temporal_event"
    return "global_overview"


def _route_prompt_hint(route: str) -> str:
    if route == "explicit_question_time":
        return (
            "The question already exposes a concrete time window, so anchor the first zoom around that window "
            "and prefer local evidence over whole-video speculation. "
        )
    if route == "ocr_detail":
        return (
            "This is a detail/OCR question. Favor readability: use fewer frames, keep resolution high, and "
            "revisit the scene where text or symbols are most likely to be visible. "
        )
    if route == "action_event":
        return (
            "This is an action/event question. Track the visible state change, compare the key before/after "
            "moments, and require extra evidence if the answer hinges on a transition. "
        )
    if route == "temporal_event":
        return (
            "This is a temporal question. Compare the order of events directly and avoid inventing unseen steps "
            "between the observed frames. "
        )
    return (
        "This is a broad overview question. Cover the timeline first, then zoom into the most discriminative "
        "unseen segment before changing any answer. "
    )


def _hybrid_v3_instruction(version: str, route: str) -> str:
    base = (
        "Treat the Direct candidate as provisional and only change it when the selected frames show a visible "
        "contradiction and directly support another option. Never use common sense to fill missing evidence. "
    )
    if version == "hybrid_v3c":
        base += "If you propose a changed answer, verify it again with an additional non-overlapping evidence pass before finalizing. "
    elif version == "hybrid_v3d":
        base += (
            "You may request up to two non-overlapping frame_select intervals in one turn when they test different "
            "unresolved options, and you should use that coverage to separate competing answers. If you propose a "
            "changed answer, verify it again with an additional evidence pass before finalizing. "
        )
    elif version == "hybrid_v3e":
        base += (
            "For broad overview and OCR/detail questions, use dense high-resolution coverage and one conservative "
            "change-confirmation pass. For action and temporal questions, verify every proposed changed answer with "
            "a second non-overlapping evidence pass before finalizing. "
        )
    elif version == "hybrid_v3f":
        base += (
            "Treat an uncertain first change proposal as a reason to gather one more non-overlapping evidence pass, "
            "not as permission to change. Finalize the change only if the new evidence produces the same strict "
            "answer again without uncertainty. "
        )
    return base + _route_prompt_hint(route)


def _hybrid_v3_confirmation_tool_prompt(
    candidate_answer: str,
    proposed_answer: str,
    allow_two_intervals: bool = False,
) -> str:
    extra = (
        "You may request up to two non-overlapping frame_select intervals if they test different unresolved "
        "evidence."
        if allow_two_intervals
        else "Request one additional non-overlapping frame_select interval that directly checks the contradiction."
    )
    return (
        f"You proposed Answer: {proposed_answer}, which differs from the Direct candidate {candidate_answer}. "
        f"Re-check the contradiction using the official EVA protocol. {extra} Emit only the official <tool_call> JSON now."
    )


def _requires_change_confirmation(sample: Sample, route: str) -> bool:
    if route != "global_overview":
        return False
    if sample.dataset == "lsdbench":
        return True
    question_types = _question_type_labels(sample.metadata)
    if {"event understanding", "temporal grounding"} & question_types:
        return True
    return bool(_EVENT_CONFIRMATION_KEYWORDS.search(sample.question))


def _strict_final_answer(text: str, valid_letters: Iterable[str]) -> str | None:
    return extract_strict_answer_letter(text, valid_letters)


def _change_gate_reason(text: str) -> str | None:
    if not text or not text.strip():
        return "empty_response"
    if _UNCERTAIN_LANGUAGE_RE.search(text):
        return "uncertain_language"
    return None


def _uses_strict_final_answer(version: str) -> bool:
    return version == "hybrid_v2" or version.startswith("hybrid_v3")


def _strategy_version_name(version: str) -> str:
    if version.startswith("hybrid_v3"):
        return f"official_eva_{version}"
    if version == "hybrid_v2":
        return "official_eva_hybrid_v2"
    if version == "hybrid_v1":
        return "official_eva_hybrid_v1"
    return "official_eva_budgeted_v2"


class Evaluator:
    def __init__(
        self,
        client: OpenAICompatibleClient,
        model: str,
        video_root: Path,
        frame_root: Path,
        max_turns: int = 3,
        max_call_visual_tokens: int = 4000,
        max_total_visual_tokens: int = 8000,
        agent_version: str = "v2a",
    ):
        if agent_version not in _AGENT_CONFIGS:
            raise ValueError(f"unsupported agent version: {agent_version}")
        if max_call_visual_tokens <= 0 or max_total_visual_tokens <= 0:
            raise ValueError("visual-token budgets must be positive")
        self.client = client
        self.model = model
        self.index = VideoIndex(video_root)
        self.frame_root = frame_root.resolve()
        self.max_turns = max(2, min(6, max_turns))
        self.max_call_visual_tokens = max_call_visual_tokens
        self.max_total_visual_tokens = max_total_visual_tokens
        self.agent_version = agent_version

    def _resolve(self, sample: Sample) -> Path:
        return self.index.resolve(sample.video)

    def direct(self, sample: Sample) -> dict[str, Any]:
        video = self._resolve(sample)
        result = self.client.chat(
            self.model,
            [{"role": "user", "content": _content_with_video(video, format_question(sample))}],
        )
        prediction = extract_answer_letter(result.content, sample.option_letters)
        record = {
            "prediction": prediction,
            "raw_response": result.content,
            "correct": prediction == sample.answer,
            "rounds": 1,
            "turn_count": 1,
            "visual_tokens": None,
            "usage": result.usage,
            "latency_s": result.latency_s,
            "fallback_used": False,
            "tool_calls": [],
            "gate_reason": None,
            "question_type": sorted(_question_type_labels(sample.metadata)) or None,
            "question_route": _question_route(sample.question, sample.metadata),
        }
        record.update(_usage_fields(result.usage))
        return record

    def text_only(self, sample: Sample) -> dict[str, Any]:
        result = self.client.chat(
            self.model,
            [{"role": "user", "content": format_text_only_question(sample)}],
            max_tokens=128,
        )
        prediction = extract_answer_letter(result.content, sample.option_letters)
        record = {
            "prediction": prediction,
            "raw_response": result.content,
            "correct": prediction == sample.answer,
            "rounds": 1,
            "turn_count": 1,
            "visual_tokens": 0,
            "usage": result.usage,
            "latency_s": result.latency_s,
            "fallback_used": False,
            "tool_calls": [],
            "gate_reason": None,
            "prompt_id": "text_only_mcq_v1",
            "media_items": 0,
            "annotation_leak_check": "passed",
            "annotation_leak_reason": "question_and_choices_only",
        }
        record.update(_usage_fields(result.usage))
        return record

    def _agent_verify(
        self,
        sample: Sample,
        candidate_answer: str | None = None,
    ) -> dict[str, Any]:
        video = self._resolve(sample)
        metadata = probe_video(video)
        duration = max(0.1, float(metadata["duration"]))
        agent_question = "\n".join(
            [f"Question: {sample.question}"]
            + [f"{letter}: {text}" for letter, text in sample.choices.items()]
        )
        initial_call, route = _initial_tool_call(
            sample.question,
            duration,
            self.agent_version,
            sample.metadata,
        )
        messages: list[dict[str, Any]] = [
            {
                "role": "system",
                "content": _agent_system_prompt(
                    metadata,
                    initial_call,
                    route,
                    self.agent_version,
                    self.max_call_visual_tokens,
                    self.max_total_visual_tokens,
                    candidate_answer,
                ),
            },
            {"role": "user", "content": [{"type": "text", "text": agent_question}]},
        ]
        call_records: list[dict[str, Any]] = []
        tool_trace: list[dict[str, Any]] = []
        observed_intervals: list[tuple[float, float]] = []
        visual_tokens = 0
        fallback_used = False
        strict_verifier_answer: str | None = None
        candidate_change_reviewed = False
        candidate_change_rejected = False
        candidate_change_finalized = False
        change_gate_triggered = False
        change_gate_deferred = False
        change_gate_reason: str | None = None
        change_confirmation_requested = False
        change_confirmation_observed = False
        pending_change_answer: str | None = None
        question_type = sorted(_question_type_labels(sample.metadata)) or None
        sample_frame_dir = self.frame_root / sample.dataset / re.sub(r"[^A-Za-z0-9_.-]", "_", sample.sample_id)

        def execute_calls(requested: list[dict[str, Any]], turn: int) -> bool:
            nonlocal visual_tokens
            accepted: list[tuple[dict[str, Any], int]] = []
            max_calls = int(_AGENT_CONFIGS[self.agent_version]["max_calls_per_turn"])
            for requested_call in requested[:max_calls]:
                start_time = max(0.0, min(float(requested_call["start_time"]), duration - 0.1))
                end_time = max(start_time + 0.1, min(float(requested_call["end_time"]), duration))
                interval = (start_time, end_time)
                if any(_ranges_redundant(interval, seen) for seen in observed_intervals):
                    continue
                if any(_ranges_overlap(interval, (call["start_time"], call["end_time"])) for call, _ in accepted):
                    continue
                remaining = self.max_total_visual_tokens - visual_tokens - sum(cost for _, cost in accepted)
                fitted = _fit_tool_budget(
                    metadata,
                    {**requested_call, "start_time": start_time, "end_time": end_time},
                    min(self.max_call_visual_tokens, remaining),
                )
                if fitted is not None:
                    accepted.append(fitted)
            if not accepted:
                return False

            combined_frames: list[Path] = []
            combined_timestamps: list[float] = []
            for call_index, (tool, estimated) in enumerate(accepted, start=1):
                frames, timestamps, frame_backend = official_select_frames(
                    video,
                    tool["start_time"],
                    tool["end_time"],
                    tool["nframes"],
                    tool["resize"],
                    sample_frame_dir / f"turn_{turn:02d}_call_{call_index:02d}",
                )
                visual_tokens += estimated
                observed_intervals.append((tool["start_time"], tool["end_time"]))
                combined_frames.extend(frames)
                combined_timestamps.extend(timestamps)
                tool_trace.append(
                    {
                        "turn": turn,
                        "call_index": call_index,
                        "start_time": tool["start_time"],
                        "end_time": tool["end_time"],
                        "nframes": len(frames),
                        "resize": tool["resize"],
                        "backend": frame_backend,
                        "timestamps": timestamps,
                        "estimated_visual_tokens": estimated,
                    }
                )
            messages.append(
                {
                    "role": "tool",
                    "content": _tool_content(
                        combined_frames,
                        combined_timestamps,
                        agent_question,
                        observed_intervals,
                        self.max_total_visual_tokens - visual_tokens,
                    ),
                }
            )
            return True

        # The first, low-cost observation is controller-scheduled. This avoids
        # spending a planning round on a repeated full-timeline request while
        # preserving the official assistant/tool message protocol.
        initial_payload = json.dumps(
            {"tool": "frame_select", "arguments": initial_call},
            ensure_ascii=False,
            separators=(",", ":"),
        )
        messages.append({"role": "assistant", "content": f"<tool_call>{initial_payload}</tool_call>"})
        if not execute_calls([initial_call], 1):
            raise RuntimeError("initial official frame selection did not fit the visual-token budget")
        tool_rounds = 1
        minimum_tool_rounds = _minimum_tool_rounds(self.agent_version, route, candidate_answer)
        tool_round_limit = int(
            _AGENT_CONFIGS[self.agent_version].get(
                "tool_round_limit",
                5 if self.agent_version in {"hybrid_v1", "hybrid_v2"} else 2,
            )
        )

        for turn in range(self.max_turns - 1):
            final_turn = (
                turn == self.max_turns - 2
                or tool_rounds >= tool_round_limit
                or visual_tokens >= self.max_total_visual_tokens
            )
            if final_turn:
                messages.append(
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": _final_answer_prompt(self.agent_version, candidate_answer),
                            }
                        ],
                    }
                )
            response = self.client.chat(self.model, messages, max_tokens=256 if final_turn else 512)
            call_records.append({"usage": response.usage, "latency_s": response.latency_s})
            messages.append({"role": "assistant", "content": response.content})
            prediction = (
                _strict_final_answer(response.content, sample.option_letters)
                if _uses_strict_final_answer(self.agent_version)
                else extract_answer_letter(response.content, sample.option_letters)
            )
            if prediction:
                strict_verifier_answer = prediction
            if prediction and (tool_rounds >= minimum_tool_rounds or final_turn):
                raw_response = response.content
                if final_turn and tool_rounds < minimum_tool_rounds and candidate_answer and prediction != candidate_answer:
                    candidate_change_reviewed = True
                    candidate_change_rejected = True
                    change_gate_triggered = True
                    change_gate_reason = "minimum_evidence_not_met"
                    prediction = None
                if pending_change_answer is not None:
                    if not change_confirmation_observed:
                        if final_turn:
                            candidate_change_rejected = True
                            change_gate_triggered = True
                            change_gate_reason = "confirmation_missing"
                            pending_change_answer = None
                            prediction = None
                        else:
                            messages.append(
                                {
                                    "role": "user",
                                    "content": [
                                        {
                                            "type": "text",
                                            "text": _hybrid_v3_confirmation_tool_prompt(
                                                candidate_answer or "",
                                                pending_change_answer,
                                                allow_two_intervals=self.agent_version == "hybrid_v3d",
                                            ),
                                        }
                                    ],
                                }
                            )
                            continue
                    elif prediction == pending_change_answer:
                        confirmation_gate_reason = (
                            _change_gate_reason(raw_response)
                            if self.agent_version == "hybrid_v3f"
                            else None
                        )
                        if confirmation_gate_reason:
                            candidate_change_rejected = True
                            change_gate_triggered = True
                            change_gate_reason = confirmation_gate_reason
                            prediction = None
                        else:
                            candidate_change_finalized = True
                        pending_change_answer = None
                    else:
                        candidate_change_rejected = True
                        change_gate_triggered = True
                        change_gate_reason = "confirmation_disagreed"
                        pending_change_answer = None
                        prediction = None
                if (
                    not candidate_change_finalized
                    and self.agent_version == "hybrid_v1"
                    and candidate_answer
                    and prediction
                    and prediction != candidate_answer
                ):
                    candidate_change_reviewed = True
                    messages.append(
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": _hybrid_change_confirmation_prompt(candidate_answer, prediction),
                                }
                            ],
                        }
                    )
                    confirmation = self.client.chat(self.model, messages, max_tokens=256)
                    call_records.append({"usage": confirmation.usage, "latency_s": confirmation.latency_s})
                    messages.append({"role": "assistant", "content": confirmation.content})
                    confirmed = extract_answer_letter(confirmation.content, sample.option_letters)
                    raw_response = confirmation.content
                    if confirmed is None or confirmed == candidate_answer:
                        prediction = None
                        candidate_change_rejected = True
                    else:
                        prediction = confirmed
                elif (
                    not candidate_change_finalized
                    and self.agent_version == "hybrid_v2"
                    and candidate_answer
                    and prediction
                    and prediction != candidate_answer
                ):
                    candidate_change_reviewed = True
                    change_gate_reason = _change_gate_reason(raw_response)
                    if change_gate_reason:
                        change_gate_triggered = True
                        candidate_change_rejected = True
                        prediction = None
                    elif _requires_change_confirmation(sample, route) and not change_confirmation_observed:
                        change_confirmation_requested = True
                        messages.append(
                            {
                                "role": "user",
                                "content": [
                                    {
                                        "type": "text",
                                        "text": _hybrid_v2_confirmation_tool_prompt(candidate_answer, prediction),
                                    }
                                ],
                            }
                        )
                        continue
                elif (
                    not candidate_change_finalized
                    and self.agent_version in {"hybrid_v3a", "hybrid_v3b"}
                    and candidate_answer
                    and prediction
                    and prediction != candidate_answer
                ):
                    candidate_change_reviewed = True
                    change_gate_reason = _change_gate_reason(raw_response)
                    if change_gate_reason:
                        change_gate_triggered = True
                        candidate_change_rejected = True
                        prediction = None
                    else:
                        needs_confirmation = (
                            route in {"global_overview", "temporal_event", "action_event"}
                            if self.agent_version == "hybrid_v3a"
                            else route != "explicit_question_time"
                        )
                        if needs_confirmation and not change_confirmation_observed:
                            if final_turn:
                                candidate_change_rejected = True
                                change_gate_triggered = True
                                change_gate_reason = "confirmation_missing"
                                prediction = None
                            else:
                                change_confirmation_requested = True
                                messages.append(
                                    {
                                        "role": "user",
                                        "content": [
                                            {
                                                "type": "text",
                                                "text": _hybrid_v3_confirmation_tool_prompt(
                                                    candidate_answer,
                                                    prediction,
                                                ),
                                            }
                                        ],
                                    }
                                )
                                continue
                elif (
                    not candidate_change_finalized
                    and self.agent_version in {"hybrid_v3c", "hybrid_v3d"}
                    and candidate_answer
                    and prediction
                    and prediction != candidate_answer
                ):
                    candidate_change_reviewed = True
                    change_gate_reason = _change_gate_reason(raw_response)
                    if change_gate_reason:
                        change_gate_triggered = True
                        candidate_change_rejected = True
                        prediction = None
                    elif final_turn:
                        candidate_change_rejected = True
                        change_gate_triggered = True
                        change_gate_reason = "confirmation_missing"
                        prediction = None
                    else:
                        pending_change_answer = prediction
                        change_confirmation_requested = True
                        messages.append(
                            {
                                "role": "user",
                                "content": [
                                    {
                                        "type": "text",
                                        "text": _hybrid_v3_confirmation_tool_prompt(
                                            candidate_answer,
                                            prediction,
                                            allow_two_intervals=self.agent_version == "hybrid_v3d",
                                        ),
                                    }
                                ],
                            }
                        )
                        continue
                elif (
                    not candidate_change_finalized
                    and self.agent_version == "hybrid_v3e"
                    and candidate_answer
                    and prediction
                    and prediction != candidate_answer
                ):
                    candidate_change_reviewed = True
                    change_gate_reason = _change_gate_reason(raw_response)
                    if change_gate_reason:
                        change_gate_triggered = True
                        candidate_change_rejected = True
                        prediction = None
                    elif final_turn:
                        candidate_change_rejected = True
                        change_gate_triggered = True
                        change_gate_reason = "confirmation_missing"
                        prediction = None
                    elif route in {"temporal_event", "action_event", "explicit_question_time"}:
                        pending_change_answer = prediction
                        change_confirmation_requested = True
                        messages.append(
                            {
                                "role": "user",
                                "content": [
                                    {
                                        "type": "text",
                                        "text": _hybrid_v3_confirmation_tool_prompt(candidate_answer, prediction),
                                    }
                                ],
                            }
                        )
                        continue
                    elif not change_confirmation_observed:
                        change_confirmation_requested = True
                        messages.append(
                            {
                                "role": "user",
                                "content": [
                                    {
                                        "type": "text",
                                        "text": _hybrid_v3_confirmation_tool_prompt(candidate_answer, prediction),
                                    }
                                ],
                            }
                        )
                        continue
                elif (
                    not candidate_change_finalized
                    and self.agent_version == "hybrid_v3f"
                    and candidate_answer
                    and prediction
                    and prediction != candidate_answer
                ):
                    candidate_change_reviewed = True
                    initial_gate_reason = _change_gate_reason(raw_response)
                    if initial_gate_reason:
                        change_gate_triggered = True
                        change_gate_deferred = True
                        change_gate_reason = initial_gate_reason
                    if final_turn:
                        candidate_change_rejected = True
                        change_gate_reason = "confirmation_missing"
                        prediction = None
                    else:
                        pending_change_answer = prediction
                        change_confirmation_requested = True
                        messages.append(
                            {
                                "role": "user",
                                "content": [
                                    {
                                        "type": "text",
                                        "text": _hybrid_v3_confirmation_tool_prompt(candidate_answer, prediction),
                                    }
                                ],
                            }
                        )
                        continue
                usage = _usage_sum(call_records)
                record = {
                    "prediction": prediction,
                    "raw_response": raw_response,
                    "correct": prediction == sample.answer,
                    "rounds": len(call_records),
                    "turn_count": len(call_records),
                    "visual_tokens": visual_tokens,
                    "usage": usage,
                    "latency_s": sum(record.get("latency_s", 0.0) for record in call_records),
                    "fallback_used": fallback_used,
                    "tool_calls": tool_trace,
                    "observed_intervals": observed_intervals,
                    "route": route,
                    "question_route": route,
                    "question_type": question_type,
                    "prompt_version": self.agent_version,
                    "strategy_version": _strategy_version_name(self.agent_version),
                    "strict_verifier_answer": strict_verifier_answer,
                    "change_gate_triggered": change_gate_triggered,
                    "change_gate_deferred": change_gate_deferred,
                    "change_rejection_reason": change_gate_reason,
                    "gate_reason": change_gate_reason,
                    "change_confirmation_requested": change_confirmation_requested,
                    "change_confirmation_observed": change_confirmation_observed,
                    "candidate_change_reviewed": candidate_change_reviewed,
                    "candidate_change_rejected": candidate_change_rejected,
                }
                if candidate_change_reviewed:
                    record["candidate_change_reviewed"] = True
                if candidate_change_rejected:
                    record["candidate_change_rejected"] = True
                record.update(_usage_fields(usage))
                return record
            if final_turn:
                continue

            requested = _parse_tool_calls(response.content)
            if change_confirmation_requested and not change_confirmation_observed and not requested:
                confirmation_interval_text = (
                    "Request up to two additional non-overlapping frame_select intervals"
                    if self.agent_version == "hybrid_v3d"
                    else "Request one additional non-overlapping frame_select interval"
                )
                messages.append(
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": f"{confirmation_interval_text} to confirm the candidate change. Emit only the official tool call.",
                            }
                        ],
                    }
                )
                continue
            if prediction and tool_rounds < minimum_tool_rounds:
                messages.append(
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": "Do not answer yet. The low-resolution overview is not sufficient; emit one non-overlapping local frame_select zoom now.",
                            }
                        ],
                    }
                )
                continue
            if not requested or tool_rounds >= tool_round_limit:
                if tool_rounds < minimum_tool_rounds:
                    messages.append(
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": "You must select one non-overlapping local zoom before answering. Emit only the official frame_select tool call now.",
                                }
                            ],
                        }
                    )
                    continue
                messages.append(
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": "The requested interval was invalid, repeated, or over budget. Answer now exactly as Answer: X.",
                            }
                        ],
                    }
                )
                continue
            if execute_calls(requested, tool_rounds + 1):
                tool_rounds += 1
                if change_confirmation_requested and not change_confirmation_observed:
                    change_confirmation_observed = True
            else:
                messages.append(
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": "The requested interval was invalid, repeated, or over budget. Answer now exactly as Answer: X.",
                            }
                        ],
                    }
                )
        usage = _usage_sum(call_records)
        record = {
            "prediction": None,
            "raw_response": messages[-1].get("content", "") if messages else "",
            "correct": False,
            "rounds": len(call_records),
            "turn_count": len(call_records),
            "visual_tokens": visual_tokens,
            "usage": usage,
            "latency_s": sum(record.get("latency_s", 0.0) for record in call_records),
            "fallback_used": fallback_used,
            "tool_calls": tool_trace,
            "observed_intervals": observed_intervals,
            "route": route,
            "question_route": route,
            "question_type": question_type,
            "prompt_version": self.agent_version,
            "strategy_version": _strategy_version_name(self.agent_version),
            "strict_verifier_answer": strict_verifier_answer,
            "change_gate_triggered": change_gate_triggered,
            "change_gate_deferred": change_gate_deferred,
            "change_rejection_reason": change_gate_reason,
            "gate_reason": change_gate_reason,
            "change_confirmation_requested": change_confirmation_requested,
            "change_confirmation_observed": change_confirmation_observed,
            "candidate_change_reviewed": candidate_change_reviewed,
            "candidate_change_rejected": candidate_change_rejected,
            "error": "no_answer",
        }
        record.update(_usage_fields(usage))
        return record

    def agent(self, sample: Sample) -> dict[str, Any]:
        """Run the ordinary EVA-style agent without a Direct hypothesis."""
        return self._agent_verify(sample, None)

    def hybrid(self, sample: Sample) -> dict[str, Any]:
        """Run Direct candidate generation followed by official EVA evidence verification."""
        candidate_result: dict[str, Any] | None = None
        candidate_error: str | None = None
        try:
            # This call is internal to Hybrid and never writes or mutates the frozen
            # Direct baseline.  Its raw explanation is intentionally not passed on.
            candidate_result = self.direct(sample)
        except Exception as exc:  # verifier should still get a chance to answer
            candidate_error = f"{type(exc).__name__}: {exc}"

        candidate_answer = None
        if candidate_result is not None:
            parsed = candidate_result.get("prediction")
            if parsed in sample.option_letters:
                candidate_answer = parsed
        try:
            if getattr(self, "agent_version", None) == "hybrid_v3g":
                question_route = _question_route(sample.question, {})
                delegated_version = (
                    "hybrid_v2"
                    if question_route in {"global_overview", "ocr_detail"}
                    else "hybrid_v3c"
                )
                verifier = copy(self)
                verifier.agent_version = delegated_version
                # v2 is retained as a frozen controller, but v3g must not let
                # dataset annotations influence its confirmation policy.
                verifier_sample = replace(sample, metadata={})
                verifier_result = verifier._agent_verify(verifier_sample, candidate_answer)
                verifier_result.update(
                    {
                        "prompt_version": "hybrid_v3g",
                        "strategy_version": "official_eva_hybrid_v3g",
                        "delegated_version": delegated_version,
                        "question_route": question_route,
                        "route": question_route,
                        "question_type": sorted(_question_type_labels(sample.metadata)) or None,
                    }
                )
            else:
                verifier_result = self._agent_verify(sample, candidate_answer)
        except Exception as exc:
            verifier_result = {
                "prediction": None,
                "raw_response": "",
                "correct": False,
                "rounds": 0,
                "turn_count": 0,
                "visual_tokens": 0,
                "usage": {},
                "latency_s": 0.0,
                "fallback_used": False,
                "tool_calls": [],
                "observed_intervals": [],
                "gate_reason": None,
                "question_type": sorted(_question_type_labels(sample.metadata)) or None,
                "question_route": _question_route(sample.question, sample.metadata),
                "error": f"{type(exc).__name__}: {exc}",
            }
        return _merge_hybrid_result(
            sample,
            candidate_result,
            verifier_result,
            candidate_error=candidate_error,
        )

    def hybrid_frozen(
        self,
        sample: Sample,
        candidate_answer: str | None,
    ) -> dict[str, Any]:
        """Run the v3 verifier with a read-only candidate and no Direct rerun."""

        candidate = (
            candidate_answer
            if candidate_answer in sample.option_letters
            else None
        )
        safe_sample = replace(sample, metadata={})
        verifier_result = self._agent_verify(safe_sample, candidate)
        candidate_result = (
            {
                "prediction": candidate,
                "raw_response": "",
                "usage": {},
                "latency_s": 0.0,
            }
            if candidate is not None
            else None
        )
        result = _merge_hybrid_result(
            sample,
            candidate_result,
            verifier_result,
        )
        result.update(
            {
                "backend": "hybrid_frozen",
                "candidate_rerun": 0,
                "candidate_raw_response": "",
                "candidate_usage": {},
                "candidate_latency_s": 0.0,
            }
        )
        return result


def _merge_hybrid_result(
    sample: Sample,
    candidate_result: dict[str, Any] | None,
    verifier_result: dict[str, Any],
    candidate_error: str | None = None,
) -> dict[str, Any]:
    """Merge candidate and verifier records without exposing candidate reasoning."""
    candidate_answer = None
    if candidate_result is not None:
        parsed = candidate_result.get("prediction")
        if parsed in sample.option_letters:
            candidate_answer = parsed
    verifier_prediction = verifier_result.get("prediction")
    final_prediction = verifier_prediction if verifier_prediction in sample.option_letters else None
    fallback_to_candidate = final_prediction is None and candidate_answer is not None
    if fallback_to_candidate:
        final_prediction = candidate_answer
    strict_verifier_answer = verifier_result.get("strict_verifier_answer")
    change_gate_triggered = bool(verifier_result.get("change_gate_triggered"))
    change_rejection_reason = verifier_result.get("change_rejection_reason")
    if fallback_to_candidate:
        final_decision_source = "candidate_gate" if change_gate_triggered else "candidate_fallback"
    elif candidate_answer and final_prediction == candidate_answer:
        final_decision_source = "candidate_confirmed"
    elif candidate_answer and final_prediction is not None and final_prediction != candidate_answer:
        final_decision_source = "verifier_change"
    elif final_prediction is not None:
        final_decision_source = "verifier"
    else:
        final_decision_source = "verifier_error"

    candidate_usage = _usage_fields(candidate_result.get("usage", {})) if candidate_result else _usage_fields({})
    verifier_usage = _usage_fields(verifier_result.get("usage", {}))
    usage = {
        key: candidate_usage[key] + verifier_usage[key]
        for key in ("prompt_tokens", "completion_tokens", "total_tokens")
    }
    candidate_latency = float(candidate_result.get("latency_s", 0.0) or 0.0) if candidate_result else 0.0
    verifier_latency = float(verifier_result.get("latency_s", 0.0) or 0.0)
    result = dict(verifier_result)
    result.update(
        {
            "backend": "hybrid",
            "prediction": final_prediction,
            "final_prediction": final_prediction,
            "correct": final_prediction == sample.answer,
            "candidate_answer": candidate_answer,
            "candidate_raw_response": candidate_result.get("raw_response", "") if candidate_result else "",
            "candidate_usage": candidate_result.get("usage", {}) if candidate_result else {},
            "candidate_latency_s": candidate_latency,
            "candidate_correct": candidate_answer == sample.answer if candidate_answer else False,
            "candidate_changed": bool(candidate_answer and final_prediction and candidate_answer != final_prediction),
            "fallback_to_candidate": fallback_to_candidate,
            "final_decision_source": final_decision_source,
            "strict_verifier_answer": strict_verifier_answer,
            "change_gate_triggered": change_gate_triggered,
            "change_gate_deferred": bool(verifier_result.get("change_gate_deferred")),
            "change_rejection_reason": change_rejection_reason,
            "gate_reason": verifier_result.get("gate_reason", change_rejection_reason),
            "change_confirmation_requested": bool(verifier_result.get("change_confirmation_requested")),
            "change_confirmation_observed": bool(verifier_result.get("change_confirmation_observed")),
            "candidate_change_reviewed": bool(verifier_result.get("candidate_change_reviewed")),
            "candidate_change_rejected": bool(verifier_result.get("candidate_change_rejected")),
            "turn_count": int(verifier_result.get("turn_count", verifier_result.get("rounds", 0)) or 0),
            "question_type": verifier_result.get("question_type", sorted(_question_type_labels(sample.metadata)) or None),
            "question_route": verifier_result.get("question_route", verifier_result.get("route")),
            "usage": usage,
            "prompt_tokens": usage["prompt_tokens"],
            "completion_tokens": usage["completion_tokens"],
            "total_tokens": usage["total_tokens"],
            "latency_s": candidate_latency + verifier_latency,
        }
    )
    if candidate_error:
        result["candidate_error"] = candidate_error
    if verifier_result.get("error"):
        result["verifier_error"] = verifier_result["error"]
    # A verifier error is actionable only when it did not yield a usable fallback.
    if final_prediction is not None:
        result.pop("error", None)
    elif verifier_result.get("error"):
        result["error"] = verifier_result["error"]
    return result


def select_manifest(samples: list[Sample], count: int | None, seed: int) -> list[Sample]:
    if count is None or count >= len(samples):
        return samples
    if count <= 0:
        raise ValueError("sample count must be positive")
    rng = random.Random(seed)
    selected = rng.sample(samples, count)
    return sorted(selected, key=lambda item: item.sample_id)


def available_samples(samples: list[Sample], video_root: Path) -> list[Sample]:
    """Return samples whose videos have already landed under a partial dataset root."""
    index = VideoIndex(video_root)
    available: list[Sample] = []
    for sample in samples:
        try:
            index.resolve(sample.video)
        except FileNotFoundError:
            continue
        available.append(sample)
    return available


def _record_needs_retry(record: dict[str, Any]) -> bool:
    """Return whether a resumed row represents an engineering failure.

    Some model-facing failures deliberately fall back to the frozen candidate,
    so they do not necessarily populate the top-level ``error`` field.  They
    still must be retried when ``--retry-errors`` is requested; otherwise a
    parse failure can be mistaken for a completed evaluation row forever.
    """

    return bool(
        record.get("error")
        or record.get("verifier_error")
        or record.get("failure_stage")
        or record.get("parse_error")
        or record.get("trajectory_valid") is False
    )


def frozen_candidate_costs(record: Mapping[str, Any]) -> dict[str, Any]:
    """Return cumulative Direct cost, failing closed on incomplete accounting."""

    executed = record.get("executed_usage")
    if isinstance(executed, Mapping):
        usage_source: Mapping[str, Any] = executed
    elif all(record.get(key) is not None for key in ("prompt_tokens", "completion_tokens", "total_tokens")):
        usage_source = record
    else:
        raw_usage = record.get("usage")
        usage_source = raw_usage if isinstance(raw_usage, Mapping) else {}

    usage: dict[str, int] = {}
    usage_complete = True
    for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
        value = usage_source.get(key)
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or float(value) < 0
        ):
            usage_complete = False
            usage[key] = 0
        else:
            usage[key] = int(value)

    visual_value = record.get("visual_tokens")
    visual_complete = record.get("visual_usage_complete") is True
    if visual_value is None and isinstance(executed, Mapping):
        visual_value = executed.get("visual_tokens")
    if not visual_complete and isinstance(executed, Mapping):
        visual_complete = executed.get("visual_usage_complete") is True
    if (
        isinstance(visual_value, bool)
        or not isinstance(visual_value, (int, float))
        or not math.isfinite(float(visual_value))
        or float(visual_value) < 0
    ):
        visual_tokens = None
        visual_complete = False
    else:
        visual_tokens = int(visual_value)

    usage_complete = bool(usage_complete and not record.get("error"))
    visual_complete = bool(visual_complete and not record.get("error"))
    return {
        "usage": usage,
        "visual_tokens": visual_tokens,
        "usage_complete": usage_complete,
        "visual_complete": visual_complete,
        "complete": bool(usage_complete and visual_complete),
    }


def evaluate(
    samples: list[Sample],
    evaluator: Any,
    backend: str,
    output_dir: Path,
    concurrency: int = 1,
    resume: bool = False,
    retry_errors: bool = False,
    candidate_answers: dict[str, str] | None = None,
    candidate_sources: dict[str, str] | None = None,
    candidate_records: Mapping[str, Mapping[str, Any]] | None = None,
    defer_scoring: bool = False,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{samples[0].dataset}_{backend}.jsonl"
    fingerprint_method = getattr(evaluator, "run_fingerprint", None)
    expected_run_fingerprint = (
        fingerprint_method() if callable(fingerprint_method) else None
    )
    cached: dict[str, dict[str, Any]] = {}
    if resume and output_path.exists():
        for line in output_path.read_text(encoding="utf-8").splitlines():
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if (
                expected_run_fingerprint is not None
                and record.get("run_fingerprint") != expected_run_fingerprint
            ):
                if backend == "fast_hybrid_eva" and record.get("run_fingerprint") is None:
                    # Early Fast Hybrid rows predate per-row fingerprints. The
                    # CLI's frozen-input file already rejects config changes;
                    # stamp these exact legacy rows during their first resume.
                    record["run_fingerprint"] = expected_run_fingerprint
                else:
                    raise RuntimeError(
                        f"resume fingerprint mismatch in {output_path}: "
                        f"expected {expected_run_fingerprint}, "
                        f"found {record.get('run_fingerprint')}"
                    )
            cached[str(record.get("sample_id"))] = record
    pending = [
        sample for sample in samples
        if sample.sample_id not in cached
        or (
            retry_errors
            and _record_needs_retry(cached[sample.sample_id])
        )
    ]

    def run_one(sample: Sample) -> dict[str, Any]:
        started = time.perf_counter()
        candidate = (candidate_answers or {}).get(sample.sample_id)
        try:
            if backend == "direct":
                result = evaluator.direct(sample)
            elif backend == "text_only":
                result = evaluator.text_only(sample)
            elif backend == "agent":
                result = evaluator.agent(sample)
            elif backend == "hybrid":
                result = evaluator.hybrid(sample)
            elif backend == "hybrid_frozen":
                result = evaluator.hybrid_frozen(sample, candidate)
            elif backend == "fast_hybrid_eva":
                result = evaluator.fast_hybrid_eva(sample, candidate)
            elif backend == "perception_memory_eva":
                result = evaluator.run(ModelSample.from_sample(sample, candidate))
            elif backend == "flashvid_hybrid":
                model_sample = ModelSample.from_sample(sample, candidate)
                result = evaluator.flashvid_hybrid(model_sample)
            else:
                raise ValueError(f"unsupported backend: {backend}")
        except Exception as exc:
            fallback = candidate if candidate in sample.option_letters else None
            data_unavailable = isinstance(exc, FileNotFoundError)
            result = {
                "prediction": fallback,
                "final_prediction": fallback,
                "error": f"{type(exc).__name__}: {exc}",
                "error_type": type(exc).__name__,
                "data_unavailable": data_unavailable,
                "rounds": 0,
                "turn_count": 0,
                "visual_tokens": None,
                "usage": {},
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
                "latency_s": time.perf_counter() - started,
                "fallback_used": False,
                "tool_calls": [],
                "candidate_answer": fallback,
                "fallback_to_candidate": fallback is not None,
                "decision_source": "candidate_fallback" if fallback else "no_valid_answer",
                "annotation_leak_check": "not_run",
            }
            if backend in {
                "fast_hybrid_eva",
                "perception_memory_eva",
                "flashvid_hybrid",
            }:
                # An exception can escape before the backend constructs its
                # normal result object.  Preserve immutable run provenance on
                # the failed row so bulk audits can distinguish an ordinary
                # infrastructure failure from a mixed or changed experiment.
                static_audit_fields = getattr(evaluator, "static_audit_fields", None)
                if callable(static_audit_fields):
                    result.update(static_audit_fields())
            if backend in {
                "fast_hybrid_eva",
                "perception_memory_eva",
                "flashvid_hybrid",
            } and data_unavailable:
                # Video resolution fails before any model request.  This is
                # the one exceptional path whose leak check can be certified
                # without inspecting a request trace.
                result["annotation_leak_check"] = "passed"
                result["annotation_leak_reason"] = (
                    "no_model_request_source_video_unavailable"
                )
            if expected_run_fingerprint is not None:
                result["run_fingerprint"] = expected_run_fingerprint
        if defer_scoring:
            # Training trajectories must be persisted before private benchmark
            # labels are joined.  Fail closed if a backend accidentally copied
            # scoring metadata into its model-facing result.
            assert_deferred_result_public(result)
            result.update(
                {
                    "dataset": sample.dataset,
                    "sample_id": sample.sample_id,
                    "video": sample.video,
                    "scoring_deferred": True,
                }
            )
        else:
            scoring = ScoringRecord.from_sample(sample)
            result.update(
                {
                    "dataset": scoring.dataset,
                    "sample_id": scoring.sample_id,
                    "video": sample.video,
                    "answer": scoring.answer,
                    "correct": result.get("prediction") == scoring.answer,
                }
            )
        result.setdefault("turn_count", result.get("rounds", 0))
        if backend in {
            "hybrid_frozen",
            "fast_hybrid_eva",
            "perception_memory_eva",
            "flashvid_hybrid",
        }:
            result.setdefault(
                "candidate_source",
                (candidate_sources or {}).get(
                    sample.sample_id,
                    "parsed" if candidate else "none",
                ),
            )
            result.setdefault("candidate_rerun", 0)
            candidate_record = (candidate_records or {}).get(sample.sample_id)
            if candidate_record is not None:
                candidate_cost = frozen_candidate_costs(candidate_record)
                candidate_usage = candidate_cost["usage"]
                agent_usage_source = result.get("usage")
                agent_usage_mapping = (
                    agent_usage_source
                    if isinstance(agent_usage_source, Mapping)
                    else result
                )
                agent_total_tokens_complete = (
                    result.get("agent_total_tokens_complete") is True
                )
                agent_usage: dict[str, int] = {}
                for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
                    value = agent_usage_mapping.get(key)
                    if (
                        isinstance(value, bool)
                        or not isinstance(value, (int, float))
                        or not math.isfinite(float(value))
                        or float(value) < 0
                    ):
                        agent_total_tokens_complete = False
                        agent_usage[key] = 0
                    else:
                        agent_usage[key] = int(value)
                if not any(agent_usage.values()) and isinstance(
                    result.get("request_trace"), list
                ):
                    for request in result["request_trace"]:
                        if not isinstance(request, Mapping):
                            continue
                        request_usage = request.get("usage")
                        if not isinstance(request_usage, Mapping):
                            continue
                        for key in agent_usage:
                            agent_usage[key] += int(request_usage.get(key, 0) or 0)
                candidate_visual_tokens = candidate_cost["visual_tokens"]
                agent_visual_value = result.get("visual_tokens")
                agent_visual_tokens_complete = (
                    result.get("visual_usage_complete") is True
                )
                if (
                    isinstance(agent_visual_value, bool)
                    or not isinstance(agent_visual_value, (int, float))
                    or not math.isfinite(float(agent_visual_value))
                    or float(agent_visual_value) < 0
                ):
                    agent_visual_tokens = None
                    agent_visual_tokens_complete = False
                else:
                    agent_visual_tokens = int(agent_visual_value)
                end_to_end_visual_tokens_complete = bool(
                    candidate_cost["visual_complete"]
                    and agent_visual_tokens_complete
                )
                end_to_end_visual_tokens = (
                    int(candidate_visual_tokens) + int(agent_visual_tokens)
                    if end_to_end_visual_tokens_complete
                    else None
                )
                end_to_end_total_tokens_complete = bool(
                    candidate_cost["usage_complete"]
                    and agent_total_tokens_complete
                )
                result.update(
                    {
                        "candidate_usage": candidate_usage,
                        "candidate_latency_s": float(
                            candidate_record.get(
                                "latency_s", candidate_record.get("elapsed_s", 0.0)
                            )
                            or 0.0
                        ),
                        "candidate_visual_tokens": candidate_visual_tokens,
                        "candidate_usage_complete": candidate_cost["usage_complete"],
                        "candidate_visual_tokens_complete": candidate_cost[
                            "visual_complete"
                        ],
                        "candidate_cost_complete": candidate_cost["complete"],
                        "agent_usage": agent_usage,
                        "agent_visual_tokens": agent_visual_tokens,
                        "agent_total_tokens_complete": agent_total_tokens_complete,
                        "agent_visual_tokens_complete": agent_visual_tokens_complete,
                        "end_to_end_prompt_tokens": (
                            candidate_usage["prompt_tokens"]
                            + agent_usage["prompt_tokens"]
                        ),
                        "end_to_end_completion_tokens": (
                            candidate_usage["completion_tokens"]
                            + agent_usage["completion_tokens"]
                        ),
                        "end_to_end_total_tokens": (
                            candidate_usage["total_tokens"]
                            + agent_usage["total_tokens"]
                        ),
                        "end_to_end_total_tokens_complete": (
                            end_to_end_total_tokens_complete
                        ),
                        "end_to_end_visual_tokens": end_to_end_visual_tokens,
                        "end_to_end_visual_tokens_complete": (
                            end_to_end_visual_tokens_complete
                        ),
                        "end_to_end_latency_s": (
                            float(
                                candidate_record.get(
                                    "latency_s",
                                    candidate_record.get("elapsed_s", 0.0),
                                )
                                or 0.0
                            )
                            + float(result.get("latency_s", 0.0) or 0.0)
                        ),
                    }
                )
        result.setdefault("gate_reason", result.get("change_rejection_reason"))
        if not defer_scoring:
            result.setdefault("question_type", sorted(_question_type_labels(sample.metadata)) or None)
            result.setdefault(
                "question_route",
                result.get("planner_route") or result.get("route") or _question_route(sample.question, sample.metadata),
            )
        result.setdefault("elapsed_s", time.perf_counter() - started)
        if expected_run_fingerprint is not None:
            result.setdefault("run_fingerprint", expected_run_fingerprint)
        if defer_scoring:
            assert_deferred_result_public(result)
        return result

    mode = "a" if resume else "w"
    with output_path.open(mode, encoding="utf-8") as output:
        with ThreadPoolExecutor(max_workers=max(1, concurrency)) as pool:
            futures = [pool.submit(run_one, sample) for sample in pending]
            for future in as_completed(futures):
                record = future.result()
                cached[record["sample_id"]] = record
                output.write(json.dumps(record, ensure_ascii=False) + "\n")
                output.flush()
    ordered = [cached[sample.sample_id] for sample in samples if sample.sample_id in cached]
    compact = output_path.with_suffix(output_path.suffix + ".partial")
    compact.write_text(
        "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in ordered),
        encoding="utf-8",
    )
    compact.replace(output_path)
    correct = None if defer_scoring else sum(bool(item.get("correct")) for item in ordered)
    errors = sum(_record_needs_retry(item) for item in ordered)
    summary = {
        "dataset": samples[0].dataset,
        "backend": backend,
        "total": len(samples),
        "completed": len(ordered),
        "correct": correct,
        "accuracy": (
            None
            if defer_scoring or not ordered
            else int(correct) / len(ordered)
        ),
        "errors": errors,
        "scoring_deferred": defer_scoring,
        "output": str(output_path),
    }
    (output_dir / f"{samples[0].dataset}_{backend}_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return summary
