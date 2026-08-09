"""Offline label join and stable trajectory selection for process SFT.

This module is the *only* Perception-Memory component allowed to load frozen
Train600 answers.  It runs after every prefix Judge call has finished, labels a
prefix complete iff all three candidate-blind evidence-only Judges cite valid
evidence and answer correctly, preserves every incomplete prefix, and emits
selected trajectories without serializing benchmark answers into model data.
"""

from __future__ import annotations

import copy
import math
from collections import Counter
from typing import Any, Iterable, Mapping, Sequence

from .perception_memory_prefix_judge import (
    PREFIX_JUDGE_SEEDS,
    PrefixJudgeJob,
    bind_prefix_jobs,
)
from .perception_memory_eva import messages_have_media
from .privacy import assert_annotation_free_request
from .privacy import assert_deferred_result_public


def load_frozen_answers(
    rows: Iterable[Mapping[str, Any]],
) -> dict[tuple[str, str], str]:
    """Load private answers into an in-memory join table with no fuzzy matching."""

    answers: dict[tuple[str, str], str] = {}
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise ValueError(f"answer row {index} must be an object")
        dataset = str(row.get("dataset") or "").strip()
        sample_id = str(row.get("sample_id") or "").strip()
        answer = str(row.get("answer") or "").strip().upper()
        if not dataset or not sample_id:
            raise ValueError("frozen answer row has no dataset/sample_id")
        if len(answer) != 1 or not "A" <= answer <= "H":
            raise ValueError("frozen answer row has no A-H answer")
        identity = (dataset, sample_id)
        if identity in answers:
            raise ValueError("frozen answers contain a duplicate sample identity")
        answers[identity] = answer
    if not answers:
        raise ValueError("frozen answer table is empty")
    return answers


def _judge_map(
    jobs: Sequence[PrefixJudgeJob], rows: Iterable[Mapping[str, Any]]
) -> dict[str, dict[str, Any]]:
    expected = {job.prefix_id: job for job in jobs}
    judged: dict[str, dict[str, Any]] = {}
    fingerprints: set[str] = set()
    for index, raw in enumerate(rows):
        if not isinstance(raw, Mapping):
            raise ValueError(f"prefix Judge row {index} must be an object")
        row = dict(raw)
        prefix_id = str(row.get("prefix_id") or "")
        if prefix_id not in expected:
            raise ValueError(f"unexpected prefix Judge identity: {prefix_id!r}")
        if prefix_id in judged:
            raise ValueError("duplicate prefix Judge row")
        job = expected[prefix_id]
        for field, wanted in (
            ("dataset", job.dataset),
            ("sample_id", job.sample_id),
            ("trajectory_id", job.trajectory_id),
            ("prefix_index", job.prefix_index),
            ("source_sha256", job.source_sha256),
        ):
            if row.get(field) != wanted:
                raise ValueError(f"prefix Judge provenance mismatch: {field}")
        if row.get("candidate_blind") is not True:
            raise ValueError("prefix Judge was not candidate-blind")
        if row.get("tools_disabled") is not True or int(row.get("media_count") or 0) != 0:
            raise ValueError("prefix Judge used tools or media")
        if row.get("annotation_leak_check") != "passed":
            raise ValueError("prefix Judge failed annotation leak audit")
        if row.get("scoring_deferred") is not True:
            raise ValueError("prefix Judge row is not deferred-scoring output")
        fingerprint = str(row.get("prefix_judge_config_sha256") or "")
        if not fingerprint:
            raise ValueError("prefix Judge row has no configuration fingerprint")
        fingerprints.add(fingerprint)
        confirmations = row.get("judge_confirmations")
        if not isinstance(confirmations, list) or len(confirmations) != 3:
            raise ValueError("all three prefix Judge calls must finish before scoring")
        seeds: set[int] = set()
        for confirmation in confirmations:
            if not isinstance(confirmation, Mapping):
                raise ValueError("prefix Judge confirmation must be an object")
            seed = int(confirmation.get("judge_seed"))
            if seed in seeds:
                raise ValueError("duplicate prefix Judge seed")
            seeds.add(seed)
            if confirmation.get("candidate_blind") is not True:
                raise ValueError("confirmation was not candidate-blind")
            if confirmation.get("tools_disabled") is not True:
                raise ValueError("confirmation did not disable tools")
            if int(confirmation.get("media_count") or 0) != 0:
                raise ValueError("confirmation unexpectedly contains media")
            messages = confirmation.get("request_messages")
            if not isinstance(messages, list) or not messages:
                raise ValueError("confirmation has no complete public request")
            if messages_have_media(messages):
                raise ValueError("confirmation request contains media")
            serialized = str(messages).casefold()
            if "direct candidate" in serialized or "candidate_answer" in serialized:
                raise ValueError("candidate leaked into prefix Judge request")
            request_kwargs = confirmation.get("request_kwargs")
            if not isinstance(request_kwargs, Mapping):
                raise ValueError("confirmation has no request kwargs")
            if request_kwargs.get("tool_choice") != "none":
                raise ValueError("confirmation request did not disable tools")
            assert_annotation_free_request(
                {"messages": messages, "request_kwargs": request_kwargs}
            )
            evidence_ids = confirmation.get("evidence_ids")
            if confirmation.get("parsed_valid") is True and (
                not isinstance(evidence_ids, list)
                or not evidence_ids
                or any(item not in job.memory.evidence_ids for item in evidence_ids)
            ):
                raise ValueError("parsed confirmation cites invalid evidence IDs")
        if seeds != set(PREFIX_JUDGE_SEEDS):
            raise ValueError("prefix Judge seeds must be exactly 17/42/73")
        judged[prefix_id] = row
    if set(judged) != set(expected):
        missing = sorted(set(expected) - set(judged))
        raise ValueError(f"prefix Judge matrix is incomplete: {missing[:3]}")
    if len(fingerprints) != 1:
        raise ValueError("prefix Judge rows mix multiple configurations")
    return judged


def _unanimous_evidence_prediction(row: Mapping[str, Any]) -> str | None:
    confirmations = row["judge_confirmations"]
    predictions: set[str] = set()
    for confirmation in confirmations:
        prediction = str(confirmation.get("prediction") or "").strip().upper()
        if (
            confirmation.get("parsed_valid") is not True
            or len(prediction) != 1
            or not "A" <= prediction <= "H"
            or confirmation.get("error")
            or confirmation.get("annotation_leak_check") != "passed"
            or not isinstance(confirmation.get("evidence_ids"), list)
            or not confirmation.get("evidence_ids")
        ):
            return None
        predictions.add(prediction)
    return next(iter(predictions)) if len(predictions) == 1 else None


def _training_confirmation(
    confirmation: Mapping[str, Any], *, prefix_complete: bool
) -> dict[str, Any]:
    """Copy the public audit fields expected by the process-SFT gate."""

    return {
        "judge_seed": int(confirmation["judge_seed"]),
        "prediction": confirmation.get("prediction"),
        "final_prediction": confirmation.get("prediction"),
        "evidence_ids": list(confirmation.get("evidence_ids") or []),
        "evidence_complete": prefix_complete,
        "annotation_leak_check": confirmation.get("annotation_leak_check"),
        "fallback_used": False,
        "fallback_to_candidate": False,
        "parsed_valid": confirmation.get("parsed_valid") is True,
        "request_messages": copy.deepcopy(confirmation.get("request_messages") or []),
        "raw_response": str(confirmation.get("raw_response") or ""),
        "reasoning_content": str(confirmation.get("reasoning_content") or ""),
        "finish_reason": confirmation.get("finish_reason"),
        "usage": copy.deepcopy(confirmation.get("usage") or {}),
        "error": confirmation.get("error"),
        "error_type": confirmation.get("error_type"),
    }


def _number(value: Any, field: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) < 0
    ):
        raise ValueError(f"{field} must be a finite non-negative number")
    return float(value)


def _request_total_tokens(request: Mapping[str, Any]) -> float:
    usage = request.get("usage")
    if not isinstance(usage, Mapping):
        raise ValueError("retained request has no usage object")
    return _number(usage.get("total_tokens", 0), "request usage.total_tokens")


def _trace_through_prefix(
    row: Mapping[str, Any], prefix_index: int
) -> list[dict[str, Any]]:
    trace = row.get("request_trace")
    if not isinstance(trace, list) or not trace:
        raise ValueError("stable replay trajectory requires request_trace")
    retained: list[dict[str, Any]] = []
    reached = False
    for raw in trace:
        if not isinstance(raw, Mapping):
            raise ValueError("request_trace entry must be an object")
        request = copy.deepcopy(dict(raw))
        retained.append(request)
        stage = str(request.get("stage") or "").casefold()
        observed_prefix = request.get("prefix_index")
        if (
            stage
            in {
                "perception",
                "observation",
                "confirmation_perception",
            }
            and observed_prefix == prefix_index
        ):
            reached = True
            break
    if not reached:
        raise ValueError("request_trace has no perception request for complete prefix")
    return retained


def _judge_request(
    confirmation: Mapping[str, Any], prefix_index: int, model: str
) -> dict[str, Any]:
    messages = confirmation.get("request_messages")
    if not isinstance(messages, list) or not messages:
        raise ValueError("complete prefix Judge has no public request messages")
    return {
        "stage": "evidence_judge",
        "model": model,
        "messages": copy.deepcopy(messages),
        "content": str(confirmation.get("raw_response") or ""),
        "reasoning_content": str(confirmation.get("reasoning_content") or ""),
        "finish_reason": confirmation.get("finish_reason"),
        "usage": copy.deepcopy(confirmation.get("usage") or {}),
        "latency_s": _number(
            confirmation.get("latency_s", 0), "Judge confirmation latency_s"
        ),
        "seed": int(confirmation["judge_seed"]),
        "step_index": prefix_index,
        "prefix_index": prefix_index,
        "prompt_hash": str(confirmation.get("request_prompt_sha256") or ""),
        "selection_confirmation": True,
    }


def _truncate_stable_trajectory(
    row: Mapping[str, Any],
    *,
    prefix_index: int,
    evidence_prediction: str,
) -> dict[str, Any]:
    """Materialize the executable prefix instead of retaining redundant tail cost."""

    truncated = copy.deepcopy(dict(row))
    states = truncated.get("perception_states")
    steps = truncated.get("tool_steps")
    if not isinstance(states, list) or prefix_index >= len(states):
        raise ValueError("complete prefix is outside perception_states")
    if not isinstance(steps, list) or len(steps) <= prefix_index:
        raise ValueError("complete prefix is outside tool_steps")
    full_states = len(states)
    full_steps = len(steps)
    source_total = truncated.get("total_tokens")
    source_visual = truncated.get("visual_tokens")
    source_latency = truncated.get("latency_s")
    truncated["perception_states"] = states[: prefix_index + 1]
    truncated["tool_steps"] = steps[: prefix_index + 1]

    retained_trace = _trace_through_prefix(truncated, prefix_index)
    final_state = truncated["perception_states"][-1]
    confirmations = final_state.get("judge_confirmations")
    if not isinstance(confirmations, list) or len(confirmations) != 3:
        raise ValueError("complete prefix requires three Judge confirmations")
    canonical_confirmation = min(
        confirmations, key=lambda item: int(item["judge_seed"])
    )
    retained_trace.append(
        _judge_request(
            canonical_confirmation,
            prefix_index,
            str(truncated.get("model") or "Qwen3.5-9B"),
        )
    )
    retained_source_total = sum(
        _request_total_tokens(request) for request in retained_trace[:-1]
    )
    retained_judge_total = _request_total_tokens(retained_trace[-1])
    retained_visual = sum(
        _number(step.get("visual_tokens"), "tool_steps.visual_tokens")
        for step in truncated["tool_steps"]
        if isinstance(step, Mapping)
    )
    if len(truncated["tool_steps"]) != sum(
        isinstance(step, Mapping) for step in truncated["tool_steps"]
    ):
        raise ValueError("tool_steps entries must be objects")
    retained_latency = sum(
        _number(request.get("latency_s", 0), "request latency_s")
        for request in retained_trace
    ) + sum(
        _number(step.get("latency_s", 0), "tool step latency_s")
        for step in truncated["tool_steps"]
    )

    memory = final_state.get("memory_after")
    if not isinstance(memory, Mapping):
        raise ValueError("complete prefix has no evidence memory")
    truncated.update(
        {
            "prediction": evidence_prediction,
            "final_prediction": evidence_prediction,
            "evidence_answer": evidence_prediction,
            "evidence_complete": True,
            "stop_reason": "earliest_stable_evidence_prefix",
            "earliest_complete_prefix_index": prefix_index,
            "event_ledger": copy.deepcopy(memory.get("event_ledger") or []),
            "option_ledger": copy.deepcopy(memory.get("option_ledger") or {}),
            "unresolved": copy.deepcopy(memory.get("unresolved") or []),
            "observed_intervals": copy.deepcopy(
                memory.get("observed_intervals") or []
            ),
            "request_trace": retained_trace,
            "rounds": prefix_index + 1,
            "turn_count": len(retained_trace),
            "full_trajectory_states": full_states,
            "full_trajectory_tool_steps": full_steps,
            "full_trajectory_total_tokens": source_total,
            "full_trajectory_visual_tokens": source_visual,
            "full_trajectory_latency_s": source_latency,
            "retained_source_total_tokens": retained_source_total,
            "retained_judge_total_tokens": retained_judge_total,
            "retained_total_tokens": retained_source_total + retained_judge_total,
            "retained_visual_tokens": retained_visual,
            "retained_tool_steps": prefix_index + 1,
            "retained_latency_s": retained_latency,
            "total_tokens": retained_source_total + retained_judge_total,
            "visual_tokens": retained_visual,
            "latency_s": retained_latency,
        }
    )
    usage = truncated.get("usage")
    if isinstance(usage, Mapping):
        truncated["usage"] = {
            **copy.deepcopy(dict(usage)),
            "total_tokens": retained_source_total + retained_judge_total,
        }
    return truncated


def _cost_key(row: Mapping[str, Any]) -> tuple[float, float, int, float, str]:
    def number(field: str) -> float:
        value = row.get(field)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return float(value)
        return float("inf")

    steps = row.get("retained_tool_steps")
    return (
        number("retained_total_tokens"),
        number("retained_visual_tokens"),
        int(steps) if isinstance(steps, int) and not isinstance(steps, bool) else 10**9,
        number("retained_latency_s"),
        str(row.get("trajectory_id") or ""),
    )


def label_and_select_trajectories(
    trajectories: Iterable[Mapping[str, Any]],
    prefix_judgments: Iterable[Mapping[str, Any]],
    frozen_answers: Mapping[tuple[str, str], str],
) -> tuple[tuple[dict[str, Any], ...], tuple[dict[str, Any], ...], dict[str, Any]]:
    """Join labels offline, preserve prefixes, and choose one stable trace/sample."""

    source_rows = [dict(row) for row in trajectories]
    jobs = bind_prefix_jobs(source_rows)
    judgments = _judge_map(jobs, prefix_judgments)
    jobs_by_trajectory: dict[str, list[PrefixJudgeJob]] = {}
    for job in jobs:
        jobs_by_trajectory.setdefault(job.trajectory_id, []).append(job)

    labeled: list[dict[str, Any]] = []
    stable_by_sample: dict[tuple[str, str], list[dict[str, Any]]] = {}
    source_samples: set[tuple[str, str]] = set()
    prefix_counts: Counter[str] = Counter()
    for source in source_rows:
        row = copy.deepcopy(source)
        identity = (str(row.get("dataset") or ""), str(row.get("sample_id") or ""))
        if identity not in frozen_answers:
            raise ValueError(f"frozen answer missing for {identity}")
        source_samples.add(identity)
        answer = frozen_answers[identity]
        states = row["perception_states"]
        trajectory_jobs = sorted(
            jobs_by_trajectory[str(row["trajectory_id"])],
            key=lambda item: item.prefix_index,
        )
        if len(states) != len(trajectory_jobs):
            raise ValueError("trajectory prefix count changed during offline join")
        earliest_complete: tuple[int, str] | None = None
        for prefix_index, (state, job) in enumerate(
            zip(states, trajectory_jobs, strict=True)
        ):
            judgment = judgments[job.prefix_id]
            evidence_prediction = _unanimous_evidence_prediction(judgment)
            complete = evidence_prediction == answer
            state["evidence_complete"] = complete
            state["evidence_prediction"] = evidence_prediction
            state["judge_confirmations"] = [
                _training_confirmation(item, prefix_complete=complete)
                for item in sorted(
                    judgment["judge_confirmations"],
                    key=lambda item: int(item["judge_seed"]),
                )
            ]
            state["prefix_judge_source_sha256"] = judgment["source_sha256"]
            state["prefix_judge_config_sha256"] = judgment[
                "prefix_judge_config_sha256"
            ]
            prefix_counts["complete" if complete else "incomplete"] += 1
            if complete and earliest_complete is None:
                assert evidence_prediction is not None
                earliest_complete = (prefix_index, evidence_prediction)

        stable = (
            earliest_complete is not None
            and row.get("annotation_leak_check") == "passed"
            and int(row.get("candidate_rerun") or 0) == 0
            and row.get("fallback_used") is not True
            and row.get("fallback_to_candidate") is not True
            and not row.get("error")
            and isinstance(row.get("tool_steps"), list)
            and bool(row.get("tool_steps"))
        )
        # Labeled/research output cannot be passed to the trainer by accident.
        row["_selection_stable"] = False
        assert_deferred_result_public(row)
        labeled.append(row)
        if stable:
            assert earliest_complete is not None
            prefix_index, evidence_prediction = earliest_complete
            stable_row = _truncate_stable_trajectory(
                row,
                prefix_index=prefix_index,
                evidence_prediction=evidence_prediction,
            )
            stable_by_sample.setdefault(identity, []).append(stable_row)

    selected: list[dict[str, Any]] = []
    for identity in sorted(stable_by_sample):
        winner = copy.deepcopy(min(stable_by_sample[identity], key=_cost_key))
        winner["_selection_stable"] = True
        assert_deferred_result_public(winner)
        selected.append(winner)

    selected_by_dataset = Counter(str(row["dataset"]) for row in selected)
    fixes_by_dataset = Counter(
        str(row["dataset"])
        for row in selected
        if row.get("candidate_answer") is not None
        and row.get("candidate_answer") != row.get("final_prediction")
    )
    answer_samples = set(frozen_answers)
    no_stable_samples = sorted(answer_samples - set(stable_by_sample))
    source_datasets = {dataset for dataset, _sample_id in answer_samples}
    no_stable_sample_ids = {
        dataset: [
            sample_id
            for sample_dataset, sample_id in no_stable_samples
            if sample_dataset == dataset
        ]
        for dataset in sorted(source_datasets)
    }
    no_stable_by_dataset = {
        dataset: len(sample_ids)
        for dataset, sample_ids in no_stable_sample_ids.items()
    }
    summary = {
        "trajectories": len(source_rows),
        "samples": len(answer_samples),
        "samples_with_trajectories": len(source_samples),
        "prefixes": len(jobs),
        "prefix_status": dict(sorted(prefix_counts.items())),
        "stable_trajectories": sum(len(items) for items in stable_by_sample.values()),
        "selected": len(selected),
        "selected_by_dataset": dict(sorted(selected_by_dataset.items())),
        "no_stable": len(no_stable_samples),
        "no_stable_by_dataset": no_stable_by_dataset,
        "no_stable_sample_ids": no_stable_sample_ids,
        "candidate_fixes": sum(fixes_by_dataset.values()),
        "candidate_fixes_by_dataset": dict(sorted(fixes_by_dataset.items())),
        "judge_seeds": list(PREFIX_JUDGE_SEEDS),
        "answers_serialized_into_selected": 0,
    }
    return tuple(labeled), tuple(selected), summary


__all__ = [
    "label_and_select_trajectories",
    "load_frozen_answers",
]
