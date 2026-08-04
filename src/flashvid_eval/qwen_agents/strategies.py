from __future__ import annotations

import json
import math
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping

from flashvid_eval.schemas import ModelSample
from flashvid_eval.baseline_diagnostics import DIRECT_SAMPLING_SPECS
from flashvid_eval.qwen_protocol import mcq_answer_response_format

from .core import (
    AgentConfig,
    AgentTrace,
    BaseQwenAgent,
    DuplicateFrameRequestError,
    EvidenceEntry,
    FrameObservation,
    FrameRequest,
    FrameSession,
    FrameTool,
    InferenceProtocol,
    ToolStep,
    evidence_text,
    json_objects,
    observation_content,
    parse_answer_json,
    parse_frame_tool_calls,
    safe_session_id,
    sample_question,
    video_content,
)


_ANSWER_INSTRUCTION = (
    'When ready, return exactly one JSON object and nothing else: {"answer":"X"}. '
    "X must be one of the supplied option letters."
)

_A3_ROOT_NODES = 8
_A3_BRANCH_NODES = 2


def _tool_instruction(max_calls: int = 1) -> str:
    suffix = "call" if max_calls == 1 else f"up to {max_calls} calls"
    return (
        "If more visual evidence is needed, return only official EVA tool markup with "
        f"{suffix}: <tool_call>{{\"tool\":\"frame_select\",\"arguments\":"
        "{\"start_time\":0.0,\"end_time\":30.0,\"nframes\":16,"
        "\"resize\":0.75,\"evidence_request\":\"what to verify\"}}}</tool_call>. "
        "Use exactly one of nframes or fps."
    )


def _metadata_text(metadata: Mapping[str, float | int]) -> str:
    return (
        f"Video duration: {float(metadata['duration']):.3f} seconds; "
        f"resolution: {int(metadata['width'])}x{int(metadata['height'])}."
    )


def _canonical_tool_call(request: FrameRequest) -> str:
    payload = {"tool": "frame_select", "arguments": request.to_tool_arguments()}
    return f"<tool_call>{json.dumps(payload, ensure_ascii=False, separators=(',', ':'))}</tool_call>"


def _fallback_intervals(duration: float, count: int, window_s: float) -> list[tuple[float, float]]:
    if duration <= window_s:
        return [(0.0, duration)]
    intervals: list[tuple[float, float]] = []
    for index in range(count):
        center = duration * (index + 1) / (count + 1)
        start = max(0.0, min(duration - window_s, center - window_s / 2))
        intervals.append((start, min(duration, start + window_s)))
    return intervals


def _parse_intervals(text: str, duration: float, limit: int) -> list[tuple[float, float]]:
    for payload in reversed(json_objects(text)):
        raw = payload.get("intervals")
        if not isinstance(raw, list):
            continue
        parsed: list[tuple[float, float]] = []
        for item in raw:
            if isinstance(item, Mapping):
                item = [item.get("start_time"), item.get("end_time")]
            if not isinstance(item, (list, tuple)) or len(item) != 2:
                continue
            try:
                start, end = float(item[0]), float(item[1])
            except (TypeError, ValueError):
                continue
            start = max(0.0, min(start, max(0.0, duration - 0.001)))
            end = max(start + 0.001, min(end, duration))
            interval = (start, end)
            if not any(abs(start - old[0]) < 0.001 and abs(end - old[1]) < 0.001 for old in parsed):
                parsed.append(interval)
            if len(parsed) >= limit:
                break
        if parsed:
            return parsed
    return []


def _strict_string_list(value: Any, *, allow_empty: bool) -> list[str] | None:
    if not isinstance(value, list) or (not allow_empty and not value):
        return None
    normalized: list[str] = []
    for item in value:
        if not isinstance(item, str) or not item.strip():
            return None
        normalized.append(item.strip())
    return normalized


def _parse_a2_overview(
    text: str,
    duration: float,
    limit: int,
) -> tuple[dict[str, Any], list[tuple[float, float]]] | None:
    try:
        payload = json.loads((text or "").strip())
    except (TypeError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict) or set(payload) != {
        "atomic_claims",
        "unresolved",
        "intervals",
    }:
        return None
    claims = _strict_string_list(payload["atomic_claims"], allow_empty=False)
    unresolved = _strict_string_list(payload["unresolved"], allow_empty=True)
    raw_intervals = payload["intervals"]
    if claims is None or unresolved is None:
        return None
    if not isinstance(raw_intervals, list) or not 1 <= len(raw_intervals) <= limit:
        return None
    intervals: list[tuple[float, float]] = []
    for item in raw_intervals:
        if not isinstance(item, list) or len(item) != 2:
            return None
        if any(isinstance(value, bool) for value in item):
            return None
        try:
            start, end = float(item[0]), float(item[1])
        except (TypeError, ValueError):
            return None
        if not all(math.isfinite(value) for value in (start, end)):
            return None
        if start < 0 or end > duration or start >= end:
            return None
        intervals.append((start, end))
    intervals.sort()
    if any(right[0] < left[1] for left, right in zip(intervals, intervals[1:])):
        return None
    normalized = {
        "atomic_claims": claims,
        "unresolved": unresolved,
        "intervals": [[start, end] for start, end in intervals],
    }
    return normalized, intervals


def _parse_a2_local(
    text: str,
    option_letters: tuple[str, ...],
) -> dict[str, Any] | None:
    try:
        payload = json.loads((text or "").strip())
    except (TypeError, json.JSONDecodeError):
        return None
    keys = {"observed_facts", "supports", "contradicts", "unresolved"}
    if not isinstance(payload, dict) or set(payload) != keys:
        return None
    facts = _strict_string_list(payload["observed_facts"], allow_empty=False)
    unresolved = _strict_string_list(payload["unresolved"], allow_empty=True)
    supports = _strict_string_list(payload["supports"], allow_empty=True)
    contradicts = _strict_string_list(payload["contradicts"], allow_empty=True)
    if None in (facts, unresolved, supports, contradicts):
        return None
    valid = {letter.upper() for letter in option_letters}
    normalized_supports = [letter.upper() for letter in supports or []]
    normalized_contradicts = [letter.upper() for letter in contradicts or []]
    if not normalized_supports and not normalized_contradicts:
        return None
    if not set(normalized_supports + normalized_contradicts) <= valid:
        return None
    if set(normalized_supports) & set(normalized_contradicts):
        return None
    return {
        "observed_facts": facts,
        "supports": normalized_supports,
        "contradicts": normalized_contradicts,
        "unresolved": unresolved,
    }


def _parse_selected_node(text: str, node_count: int) -> int | None:
    for payload in reversed(json_objects(text)):
        value = payload.get("selected_node")
        if isinstance(value, int) and not isinstance(value, bool) and 0 <= value < node_count:
            return value
    return None


def _split_interval(start: float, end: float, count: int) -> list[tuple[float, float]]:
    width = (end - start) / count
    return [
        (start + width * index, end if index == count - 1 else start + width * (index + 1))
        for index in range(count)
    ]


class EvidenceAgentBase(BaseQwenAgent):
    def _observe(
        self,
        sample: ModelSample,
        trace: AgentTrace,
        session: FrameSession,
        request: FrameRequest,
        instruction: str,
        *,
        branch: str,
        source: str,
        seed_offset: int = 0,
        record: bool = True,
    ) -> tuple[FrameObservation, str]:
        observation = session.select(request)
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a visual evidence observer. Report only facts visible in the supplied "
                    "frames. Never fill gaps between frames or use an answer candidate."
                ),
            },
            {
                "role": "user",
                "content": observation_content(
                    observation,
                    f"{sample_question(sample)}\n{instruction}",
                ),
            },
        ]
        result = self._chat(
            trace,
            messages,
            branch=branch,
            request_kind="observer",
            seed_offset=seed_offset,
        )
        if record:
            self._record_observation(trace, branch, observation, result.content, source)
        else:
            trace.tool_steps.append(ToolStep.from_observation(branch, observation))
        return observation, result.content

    def _judge(
        self,
        sample: ModelSample,
        trace: AgentTrace,
        *,
        branch: str,
        seed_offset: int = 0,
    ) -> str | None:
        messages = [
            {
                "role": "system",
                "content": (
                    "You are an independent multiple-choice judge. Use only the timestamped evidence "
                    "provided. Do not invent unseen events. "
                    + _ANSWER_INSTRUCTION
                ),
            },
            {
                "role": "user",
                "content": (
                    f"{sample_question(sample)}\nEvidence memory:\n"
                    f"{evidence_text(trace.evidence_memory)}"
                ),
            },
        ]
        result = self._chat(
            trace,
            messages,
            branch=branch,
            request_kind="judge",
            seed_offset=seed_offset,
            response_format=mcq_answer_response_format(sample.option_letters),
        )
        trace.raw_response = result.content
        return parse_answer_json(result.content, sample.option_letters)


class EvaCleanStrategy(EvidenceAgentBase):
    strategy_id = "a0_eva_clean"

    def _run(self, sample: ModelSample, trace: AgentTrace) -> None:
        video = self._resolve_video(sample)
        session = self.frame_tool.open_session(video, safe_session_id(sample, self.strategy_id))
        metadata = session.metadata
        messages: list[dict[str, Any]] = [
            {
                "role": "system",
                "content": (
                    "You are a Qwen-only long-video agent. Inspect the video through frame_select. "
                    "After each tool result, write a concise <evidence> block containing only visible "
                    "facts, then either request a new non-duplicate interval or answer. "
                    f"{_metadata_text(metadata)} {_tool_instruction()} {_ANSWER_INSTRUCTION}"
                ),
            },
            {"role": "user", "content": sample_question(sample)},
        ]
        last_observations: list[FrameObservation] = []
        force_final_answer = False
        for turn in range(self.config.max_turns):
            if (
                turn == self.config.max_turns - 1
                and trace.tool_steps
                and not force_final_answer
            ):
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            "This is the final turn. Use the evidence already observed and return "
                            "the required one-key answer JSON now."
                        ),
                    }
                )
                force_final_answer = True
            result = self._chat(
                trace,
                messages,
                branch="evidence",
                request_kind=(
                    "judge"
                    if force_final_answer
                    else "planner"
                    if turn == 0
                    else "observer"
                ),
                seed_offset=turn,
                response_format=(
                    mcq_answer_response_format(sample.option_letters)
                    if force_final_answer
                    else None
                ),
            )
            messages.append({"role": "assistant", "content": result.content})
            if last_observations:
                for observation in last_observations:
                    trace.evidence_memory.append(
                        EvidenceEntry(
                            source="eva_response",
                            interval=(observation.resolved_start_time, observation.resolved_end_time),
                            timestamps=observation.timestamps,
                            content=result.content,
                        )
                    )
                last_observations = []
            answer = parse_answer_json(result.content, sample.option_letters)
            calls = parse_frame_tool_calls(result.content, limit=self.config.max_intervals)
            if answer is not None and not calls:
                trace.prediction = trace.final_prediction = answer
                trace.raw_response = result.content
                return
            if not calls and not trace.tool_steps:
                calls = [
                    FrameRequest(
                        start_time=0.0,
                        end_time=float(metadata["duration"]),
                        nframes=self.config.overview_frames,
                        resize=self.config.resize,
                        evidence_request="Obtain a full-timeline overview relevant to the choices.",
                    )
                ]
                messages.append({"role": "assistant", "content": _canonical_tool_call(calls[0])})
                trace.fallback_used = True
            elif not calls:
                messages.append(
                    {
                        "role": "user",
                        "content": "Return the final answer now as the required one-key JSON object.",
                    }
                )
                force_final_answer = True
                continue

            tool_content: list[dict[str, Any]] = [
                {"type": "text", "text": "<tool_response>"}
            ]
            duplicate_seen = False
            for request in calls:
                try:
                    observation = session.select(request)
                except DuplicateFrameRequestError:
                    trace.fallback_used = True
                    duplicate_seen = True
                    continue
                trace.tool_steps.append(ToolStep.from_observation("evidence", observation))
                last_observations.append(observation)
                tool_content.extend(
                    observation_content(
                        observation,
                        request.evidence_request or "Inspect the visible evidence.",
                    )
                )
            if last_observations:
                if duplicate_seen:
                    tool_content.append(
                        {
                            "type": "text",
                            "text": "One duplicate request was skipped because it was already observed.",
                        }
                    )
                tool_content.append({"type": "text", "text": "</tool_response>"})
                messages.append({"role": "tool", "content": tool_content})
            elif duplicate_seen:
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            "That exact frame request was already observed. Request a different "
                            "interval or return the final answer JSON."
                        ),
                    }
                )
        trace.error = "agent did not return a valid answer within max_turns"
        trace.error_type = "NoAnswerError"


class StoryboardZoomStrategy(EvidenceAgentBase):
    strategy_id = "a1_storyboard_zoom"

    overview_instruction = (
        "Build a storyboard of the visible events. Return JSON with `intervals` containing up to "
        "the requested number of [start_seconds,end_seconds] ranges most useful for distinguishing "
        "the options, plus a short `evidence` field."
    )
    local_instruction = (
        "Describe visible actions, entities, state changes, and contradictions between options as "
        "compact JSON. Do not infer events between sampled frames."
    )

    def _run(self, sample: ModelSample, trace: AgentTrace) -> None:
        video = self._resolve_video(sample)
        session = self.frame_tool.open_session(video, safe_session_id(sample, self.strategy_id))
        duration = float(session.metadata["duration"])
        _, overview_text = self._observe(
            sample,
            trace,
            session,
            FrameRequest(
                0.0,
                duration,
                nframes=self.config.overview_frames,
                resize=self.config.resize,
                evidence_request="Create a full-timeline storyboard.",
            ),
            f"{self.overview_instruction} Select at most {self.config.max_intervals} intervals.",
            branch="evidence",
            source="storyboard_overview",
        )
        local_limit = min(self.config.max_intervals, max(1, self.config.max_turns - 2))
        intervals = _parse_intervals(overview_text, duration, local_limit)
        if not intervals:
            intervals = _fallback_intervals(
                duration,
                local_limit,
                min(duration, self.config.local_window_s),
            )
            trace.fallback_used = True
        for index, (start, end) in enumerate(intervals):
            self._observe(
                sample,
                trace,
                session,
                FrameRequest(
                    start,
                    end,
                    fps=self.config.local_fps,
                    resize=self.config.resize,
                    evidence_request="Inspect this candidate interval in temporal order.",
                ),
                self.local_instruction,
                branch="evidence",
                source="local_zoom",
                seed_offset=index + 1,
            )
        answer = self._judge(sample, trace, branch="judge")
        if answer is None:
            trace.error = "judge returned invalid answer JSON"
            trace.error_type = "AnswerParseError"
            return
        trace.prediction = trace.final_prediction = answer


class MultiClueMemoryStrategy(StoryboardZoomStrategy):
    strategy_id = "a2_multi_clue_memory"
    overview_instruction = (
        "Decompose the answer options into discriminative visual claims. Return JSON with "
        "`atomic_claims`, `unresolved`, and up to the requested number of non-identical `intervals` "
        "as [start_seconds,end_seconds]. Seek multiple disjoint clues when counting, order, or "
        "cross-event aggregation is required."
    )
    local_instruction = (
        "Return compact JSON with `observed_facts`, `supports`, `contradicts`, and `unresolved`. "
        "Refer to option letters and include temporal order only when directly visible."
    )

    @staticmethod
    def _canonical_evidence(payload: Mapping[str, Any]) -> str:
        return json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )

    @staticmethod
    def _append_validated_evidence(
        trace: AgentTrace,
        observation: FrameObservation,
        payload: Mapping[str, Any],
        source: str,
    ) -> None:
        trace.evidence_memory.append(
            EvidenceEntry(
                source=source,
                interval=(
                    observation.resolved_start_time,
                    observation.resolved_end_time,
                ),
                timestamps=observation.timestamps,
                content=MultiClueMemoryStrategy._canonical_evidence(payload),
            )
        )

    def _run(self, sample: ModelSample, trace: AgentTrace) -> None:
        video = self._resolve_video(sample)
        session = self.frame_tool.open_session(
            video, safe_session_id(sample, self.strategy_id)
        )
        duration = float(session.metadata["duration"])
        local_limit = min(
            self.config.max_intervals, max(1, self.config.max_turns - 2)
        )
        overview, overview_text = self._observe(
            sample,
            trace,
            session,
            FrameRequest(
                0.0,
                duration,
                nframes=self.config.overview_frames,
                resize=self.config.resize,
                evidence_request="Find multiple disjoint clues across the full timeline.",
            ),
            f"{self.overview_instruction} Select at most {local_limit} intervals.",
            branch="evidence",
            source="storyboard_overview",
            record=False,
        )
        parsed_overview = _parse_a2_overview(
            overview_text, duration, local_limit
        )
        if parsed_overview is None:
            intervals = _fallback_intervals(
                duration,
                local_limit,
                min(duration, self.config.local_window_s),
            )
            trace.fallback_used = True
        else:
            overview_payload, intervals = parsed_overview
            self._append_validated_evidence(
                trace, overview, overview_payload, "storyboard_overview"
            )

        valid_local_evidence = 0
        for index, (start, end) in enumerate(intervals):
            observation, local_text = self._observe(
                sample,
                trace,
                session,
                FrameRequest(
                    start,
                    end,
                    fps=self.config.local_fps,
                    resize=self.config.resize,
                    evidence_request="Verify option claims using directly visible facts.",
                ),
                self.local_instruction,
                branch="evidence",
                source="local_zoom",
                seed_offset=index + 1,
                record=False,
            )
            local_payload = _parse_a2_local(
                local_text, sample.option_letters
            )
            if local_payload is None:
                continue
            self._append_validated_evidence(
                trace, observation, local_payload, "local_zoom"
            )
            valid_local_evidence += 1

        if valid_local_evidence == 0:
            trace.error = "A2 produced no schema-valid local visual evidence"
            trace.error_type = "EvidenceValidationError"
            trace.failure_class = "model_parse_failure"
            return
        answer = self._judge(sample, trace, branch="judge")
        if answer is None:
            trace.error = "judge returned invalid answer JSON"
            trace.error_type = "AnswerParseError"
            return
        trace.prediction = trace.final_prediction = answer



class HierarchicalSearchStrategy(EvidenceAgentBase):
    strategy_id = "a3_hierarchical_search"

    def _run(self, sample: ModelSample, trace: AgentTrace) -> None:
        video = self._resolve_video(sample)
        session = self.frame_tool.open_session(video, safe_session_id(sample, self.strategy_id))
        current = (0.0, float(session.metadata["duration"]))
        search_depth = min(self.config.hierarchy_depth, max(0, self.config.max_turns - 2))
        for depth in range(search_depth):
            if current[1] - current[0] <= self.config.local_window_s:
                break
            node_count = _A3_ROOT_NODES if depth == 0 else _A3_BRANCH_NODES
            search_stage = "eight-way coarse overview" if depth == 0 else "binary refinement"
            nodes = _split_interval(current[0], current[1], node_count)
            node_text = "\n".join(
                f"Node {index}: [{start:.3f}, {end:.3f}] seconds"
                for index, (start, end) in enumerate(nodes)
            )
            _, response = self._observe(
                sample,
                trace,
                session,
                FrameRequest(
                    current[0],
                    current[1],
                    nframes=node_count,
                    resize=self.config.resize,
                    evidence_request=f"Compare nodes during {search_stage} at depth {depth}.",
                ),
                (
                    f"The frames represent these ordered nodes:\n{node_text}\n"
                    f"This is a {search_stage}. "
                    "Return JSON with `selected_node` (zero-based integer) and concise "
                    "`node_summaries`. Select the single node most likely to contain evidence "
                    "that distinguishes the options."
                ),
                branch="evidence",
                source=f"hierarchy_depth_{depth}",
                seed_offset=depth,
            )
            selected = _parse_selected_node(response, len(nodes))
            if selected is None:
                selected = len(nodes) // 2
                trace.fallback_used = True
            current = nodes[selected]

        dense_end = min(current[1], current[0] + self.config.local_window_s)
        self._observe(
            sample,
            trace,
            session,
            FrameRequest(
                current[0],
                dense_end,
                fps=self.config.local_fps,
                resize=self.config.resize,
                evidence_request="Inspect the selected leaf interval densely and in temporal order.",
            ),
            (
                "Return compact JSON with visible events, option support, option contradictions, "
                "and unresolved details. Do not infer unseen transitions."
            ),
            branch="evidence",
            source="hierarchy_leaf",
            seed_offset=search_depth,
        )
        answer = self._judge(sample, trace, branch="judge")
        if answer is None:
            trace.error = "judge returned invalid answer JSON"
            trace.error_type = "AnswerParseError"
            return
        trace.prediction = trace.final_prediction = answer


class IndependentArbitrationStrategy(EvidenceAgentBase):
    strategy_id = "a4_independent_arbitration"

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        evidence_config = replace(self.config, strategy=self.config.evidence_strategy)
        strategy_cls = _STRATEGIES.get(evidence_config.strategy)
        if strategy_cls is None or strategy_cls is IndependentArbitrationStrategy:
            raise ValueError(f"invalid A4 evidence strategy: {evidence_config.strategy}")
        self.evidence_agent = strategy_cls(
            client=self.client,
            model=self.model,
            video_root=self.index.root,
            frame_tool=self.frame_tool,
            config=evidence_config,
            protocol=self.protocol,
        )

    def _run(self, sample: ModelSample, trace: AgentTrace) -> None:
        video = self._resolve_video(sample)
        direct_messages = [
            {
                "role": "system",
                "content": (
                    "Answer the video multiple-choice question independently from the full video. "
                    + _ANSWER_INSTRUCTION
                ),
            },
            {"role": "user", "content": video_content(video, sample_question(sample))},
        ]
        direct_sampling = DIRECT_SAMPLING_SPECS[self.config.direct_sampling]
        direct_duration = float(self.frame_tool.probe(video)["duration"])
        direct_result = self._chat(
            trace,
            direct_messages,
            branch="direct",
            request_kind="direct",
            response_format=mcq_answer_response_format(sample.option_letters),
            mm_processor_kwargs=direct_sampling.mm_processor_kwargs(),
            media_io_kwargs=direct_sampling.media_io_kwargs(direct_duration),
        )
        direct_answer = parse_answer_json(direct_result.content, sample.option_letters)

        evidence_trace = self.evidence_agent.run(sample)
        for item in evidence_trace.request_trace:
            trace.request_trace.append(replace(item, branch=f"evidence/{item.branch}"))
        for item in evidence_trace.tool_steps:
            trace.tool_steps.append(replace(item, branch=f"evidence/{item.branch}"))
        trace.evidence_memory.extend(evidence_trace.evidence_memory)
        evidence_answer = evidence_trace.final_prediction

        if direct_answer is not None and direct_answer == evidence_answer:
            trace.prediction = trace.final_prediction = direct_answer
            trace.raw_response = direct_result.content
            return
        if direct_answer is None and evidence_answer is not None:
            trace.prediction = trace.final_prediction = evidence_answer
            trace.raw_response = evidence_trace.raw_response
            trace.fallback_used = True
            return
        if evidence_answer is None and direct_answer is not None:
            trace.prediction = trace.final_prediction = direct_answer
            trace.raw_response = direct_result.content
            trace.fallback_used = True
            return
        if direct_answer is None and evidence_answer is None:
            trace.error = "both independent branches returned invalid answers"
            trace.error_type = "AnswerParseError"
            return

        arbiter_messages = [
            {
                "role": "system",
                "content": (
                    "You are an independent arbiter. The two answers are hypotheses, not labels. "
                    "Use timestamped evidence to choose one. If one visual detail must be confirmed, "
                    f"request one interval with official EVA markup. {_tool_instruction()} "
                    f"Otherwise {_ANSWER_INSTRUCTION}"
                ),
            },
            {
                "role": "user",
                "content": (
                    f"{sample_question(sample)}\nFull-video branch answer: {direct_answer}\n"
                    f"Evidence-agent answer: {evidence_answer}\nTimestamped evidence:\n"
                    f"{evidence_text(trace.evidence_memory)}"
                ),
            },
        ]
        arbiter = self._chat(
            trace,
            arbiter_messages,
            branch="arbiter",
            request_kind="judge",
        )
        answer = parse_answer_json(arbiter.content, sample.option_letters)
        confirmation = parse_frame_tool_calls(arbiter.content, limit=1)
        if confirmation:
            session = self.frame_tool.open_session(video, safe_session_id(sample, "a4_confirmation"))
            observation = session.select(confirmation[0])
            trace.tool_steps.append(ToolStep.from_observation("arbiter", observation))
            trace.evidence_memory.append(
                EvidenceEntry(
                    source="arbitration_confirmation",
                    interval=(observation.resolved_start_time, observation.resolved_end_time),
                    timestamps=observation.timestamps,
                    content="confirmation frames supplied to arbiter",
                )
            )
            arbiter_messages.extend(
                [
                    {"role": "assistant", "content": arbiter.content},
                    {
                        "role": "tool",
                        "content": observation_content(
                            observation,
                            "Resolve the disagreement using only directly visible evidence. "
                            + _ANSWER_INSTRUCTION,
                        ),
                    },
                ]
            )
            final = self._chat(
                trace,
                arbiter_messages,
                branch="arbiter",
                request_kind="judge",
                seed_offset=1,
                response_format=mcq_answer_response_format(sample.option_letters),
            )
            trace.raw_response = final.content
            answer = parse_answer_json(final.content, sample.option_letters)
        else:
            trace.raw_response = arbiter.content
        if answer is None:
            trace.prediction = trace.final_prediction = direct_answer
            trace.fallback_used = True
            return
        trace.prediction = trace.final_prediction = answer


_STRATEGIES: dict[str, type[BaseQwenAgent]] = {
    "a0": EvaCleanStrategy,
    "a0_eva_clean": EvaCleanStrategy,
    "a1": StoryboardZoomStrategy,
    "a1_storyboard_zoom": StoryboardZoomStrategy,
    "a2": MultiClueMemoryStrategy,
    "a2_multi_clue_memory": MultiClueMemoryStrategy,
    "a3": HierarchicalSearchStrategy,
    "a3_hierarchical_search": HierarchicalSearchStrategy,
    "a4": IndependentArbitrationStrategy,
    "a4_independent_arbitration": IndependentArbitrationStrategy,
}


def build_strategy(
    config: AgentConfig | Mapping[str, Any],
    *,
    client: Any,
    model: str,
    video_root: Path,
    frame_root: Path,
    protocol: InferenceProtocol | None = None,
    frame_tool: FrameTool | None = None,
) -> BaseQwenAgent:
    if not isinstance(config, AgentConfig):
        config = AgentConfig.from_mapping(config)
    strategy_cls = _STRATEGIES.get(config.strategy)
    if strategy_cls is None:
        raise ValueError(f"unknown Qwen agent strategy: {config.strategy}")
    tool = frame_tool or FrameTool(
        frame_root,
        max_frames_per_call=config.max_frames_per_call,
    )
    return strategy_cls(
        client=client,
        model=model,
        video_root=video_root,
        frame_tool=tool,
        config=config,
        protocol=protocol,
    )
