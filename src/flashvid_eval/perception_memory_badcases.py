from __future__ import annotations

import json
import re
from collections import Counter
from collections.abc import Iterable, Mapping
from pathlib import Path
from statistics import fmean
from typing import Any


_CLOCK = re.compile(r"(?<!\d)(?:(\d{1,2}):)?([0-5]?\d):([0-5]\d)(?!\d)")


def read_result_files(paths: Iterable[Path]) -> dict[tuple[str, str], dict[str, Any]]:
    records: dict[tuple[str, str], dict[str, Any]] = {}
    for path in paths:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError(f"{path}:{line_number}: row must be an object")
                dataset = str(value.get("dataset") or "").strip().lower()
                sample_id = str(value.get("sample_id") or "").strip()
                if not dataset or not sample_id:
                    raise ValueError(
                        f"{path}:{line_number}: missing dataset/sample_id"
                    )
                key = (dataset, sample_id)
                if key in records:
                    raise ValueError(f"duplicate result identity: {dataset}/{sample_id}")
                records[key] = value
    return records


def _seconds(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value or "").strip()
    try:
        return float(text)
    except ValueError:
        match = _CLOCK.fullmatch(text)
        if not match:
            return None
        return float(
            int(match.group(1) or 0) * 3600
            + int(match.group(2)) * 60
            + int(match.group(3))
        )


def _annotation_intervals(value: Any) -> list[tuple[float, float]]:
    if isinstance(value, Mapping):
        start = _seconds(
            value.get("start", value.get("start_time", value.get("begin")))
        )
        end = _seconds(value.get("end", value.get("end_time", value.get("stop"))))
        if start is not None and end is not None and end >= start:
            return [(start, end)]
        intervals: list[tuple[float, float]] = []
        for item in value.values():
            intervals.extend(_annotation_intervals(item))
        return intervals
    if isinstance(value, (list, tuple)):
        if len(value) == 2:
            start, end = (_seconds(value[0]), _seconds(value[1]))
            if start is not None and end is not None and end >= start:
                return [(start, end)]
        intervals: list[tuple[float, float]] = []
        for item in value:
            intervals.extend(_annotation_intervals(item))
        return intervals
    if isinstance(value, str):
        matches = list(_CLOCK.finditer(value))
        if len(matches) >= 2:
            start, end = (_seconds(matches[0].group(0)), _seconds(matches[1].group(0)))
            if start is not None and end is not None and end >= start:
                return [(start, end)]
    return []


def read_manifest_annotations(
    paths: Iterable[Path],
) -> dict[tuple[str, str], list[tuple[float, float]]]:
    """Read private temporal labels for post-hoc diagnostics only."""

    result: dict[tuple[str, str], list[tuple[float, float]]] = {}
    for path in paths:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                row = json.loads(line)
                dataset = str(row.get("dataset") or "").strip().lower()
                sample_id = str(row.get("sample_id") or "").strip()
                if not dataset or not sample_id:
                    raise ValueError(
                        f"{path}:{line_number}: manifest row lacks dataset/sample_id"
                    )
                key = (dataset, sample_id)
                if key in result:
                    raise ValueError(f"duplicate manifest identity: {dataset}/{sample_id}")
                metadata = row.get("metadata")
                metadata = metadata if isinstance(metadata, Mapping) else {}
                intervals: list[tuple[float, float]] = []
                for field in ("time_range", "clue_intervals", "time_reference"):
                    intervals.extend(_annotation_intervals(metadata.get(field)))
                result[key] = sorted(set(intervals))
    return result


def attach_target_diagnostics(
    rows: list[dict[str, Any]],
    annotations: Mapping[tuple[str, str], list[tuple[float, float]]],
) -> None:
    """Join private labels only after inference and compute localization metrics."""

    for row in rows:
        identity = (str(row["dataset"]), str(row["sample_id"]))
        intervals = list(annotations.get(identity) or [])
        row["diagnostic_target_intervals"] = [list(item) for item in intervals]
        for prefix in ("untrained", "sft"):
            timestamps = [
                float(timestamp)
                for call in row.get(f"{prefix}_tool_trace") or []
                for timestamp in call.get("timestamps") or []
                if isinstance(timestamp, (int, float)) and not isinstance(timestamp, bool)
            ]
            count = sum(
                any(start <= timestamp <= end for start, end in intervals)
                for timestamp in timestamps
            )
            hit = bool(count) if intervals else None
            row[f"{prefix}_target_frame_count"] = count if intervals else None
            row[f"{prefix}_funnel"]["target_hit"] = hit


def _prediction(row: Mapping[str, Any]) -> str | None:
    value = str(row.get("final_prediction") or row.get("prediction") or "").upper()
    return value if len(value) == 1 and "A" <= value <= "H" else None


def _answer(row: Mapping[str, Any]) -> str | None:
    value = str(row.get("answer") or "").upper()
    return value if len(value) == 1 and "A" <= value <= "H" else None


def _tool_calls(row: Mapping[str, Any]) -> list[dict[str, Any]]:
    value = row.get("tool_calls", row.get("tool_steps"))
    return [dict(item) for item in value or [] if isinstance(item, Mapping)]


def _tool_call_audit(row: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Keep the evidence-location fields needed for paired offline diagnosis."""

    audited: list[dict[str, Any]] = []
    for index, call in enumerate(_tool_calls(row), 1):
        timestamps = call.get("timestamps")
        frame_paths = call.get("frame_paths")
        audited.append(
            {
                "call_index": index,
                "stage": call.get("stage"),
                "start_time": call.get("start_time"),
                "end_time": call.get("end_time"),
                "nframes_requested": call.get("nframes"),
                "nframes_returned": (
                    len(frame_paths)
                    if isinstance(frame_paths, list)
                    else len(timestamps)
                    if isinstance(timestamps, list)
                    else None
                ),
                "resize": call.get("resize"),
                "timestamps": timestamps if isinstance(timestamps, list) else [],
                "estimated_visual_tokens": call.get("estimated_visual_tokens"),
                "actual_visual_tokens": call.get("visual_tokens"),
                "backend": call.get("backend"),
            }
        )
    return audited


def _request_audit(row: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Retain model outputs and usage without copying potentially huge prompts."""

    requests = row.get("request_trace")
    if not isinstance(requests, list):
        return []
    audited: list[dict[str, Any]] = []
    for index, request in enumerate(requests, 1):
        if not isinstance(request, Mapping):
            continue
        usage = request.get("usage")
        audited.append(
            {
                "request_index": index,
                "stage": request.get("stage"),
                "finish_reason": request.get("finish_reason"),
                "assistant_output": request.get("assistant_content")
                or request.get("content")
                or "",
                "reasoning_content": request.get("reasoning_content") or "",
                "usage": dict(usage) if isinstance(usage, Mapping) else {},
                "latency_s": request.get("latency_s"),
                "error": request.get("error"),
            }
        )
    return audited


def _number(row: Mapping[str, Any], *keys: str) -> float | None:
    for key in keys:
        value = row.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            continue
        return float(value)
    return None


def _funnel(row: Mapping[str, Any]) -> dict[str, bool | None]:
    target_hit = row.get("target_hit")
    if not isinstance(target_hit, bool):
        target_hit = None
    valid = row.get("evidence_state_valid")
    if not isinstance(valid, bool):
        states = row.get("evidence_states")
        valid = (
            bool(states)
            and all(
                isinstance(item, Mapping) and item.get("valid") is not False
                for item in states
            )
            if isinstance(states, list)
            else None
        )
    complete = row.get("evidence_complete")
    if not isinstance(complete, bool):
        complete = None
    answer = _answer(row)
    prediction = _prediction(row)
    return {
        "target_hit": target_hit,
        "evidence_state_valid": valid,
        "evidence_complete": complete,
        "judge_correct": prediction == answer if answer and prediction else None,
    }


def pair_badcases(
    untrained: Mapping[tuple[str, str], Mapping[str, Any]],
    sft: Mapping[tuple[str, str], Mapping[str, Any]],
) -> list[dict[str, Any]]:
    if set(untrained) != set(sft):
        missing_sft = sorted(set(untrained) - set(sft))
        missing_untrained = sorted(set(sft) - set(untrained))
        raise ValueError(
            "paired result IDs differ: "
            f"missing_sft={missing_sft[:5]}, missing_untrained={missing_untrained[:5]}"
        )
    paired: list[dict[str, Any]] = []
    for dataset, sample_id in sorted(untrained):
        base = untrained[(dataset, sample_id)]
        tuned = sft[(dataset, sample_id)]
        answer = _answer(base)
        if answer != _answer(tuned):
            raise ValueError(f"answer mismatch for {dataset}/{sample_id}")
        base_prediction = _prediction(base)
        tuned_prediction = _prediction(tuned)
        base_correct = bool(answer and base_prediction == answer)
        tuned_correct = bool(answer and tuned_prediction == answer)
        if base_correct and not tuned_correct:
            flip = "untrained_correct_sft_wrong"
        elif not base_correct and tuned_correct:
            flip = "untrained_wrong_sft_correct"
        elif base_correct:
            flip = "both_correct"
        else:
            flip = "both_wrong"

        base_calls = _tool_calls(base)
        tuned_calls = _tool_calls(tuned)
        candidate = str(
            tuned.get("candidate_answer") or base.get("candidate_answer") or ""
        ).upper() or None
        failure_modes: list[str] = []
        if tuned.get("error"):
            failure_modes.append("engineering_failure")
        if flip == "untrained_correct_sft_wrong" and len(tuned_calls) < len(base_calls):
            failure_modes.append("possible_early_stop")
        if candidate and candidate != answer and tuned_prediction == candidate:
            failure_modes.append("wrong_candidate_preserved")
        if candidate and candidate == answer and tuned_prediction != candidate:
            failure_modes.append("correct_candidate_regressed")
        if tuned.get("stopped_with_incomplete_evidence") is True:
            failure_modes.append("incomplete_evidence_stop")
        if tuned.get("evidence_state_valid") is False:
            failure_modes.append("invalid_evidence_state")
        if not failure_modes and not tuned_correct:
            failure_modes.append("unclassified_reasoning_or_localization")

        paired.append(
            {
                "dataset": dataset,
                "sample_id": sample_id,
                "answer": answer,
                "candidate_answer": candidate,
                "untrained_prediction": base_prediction,
                "sft_prediction": tuned_prediction,
                "flip_type": flip,
                "failure_modes": failure_modes,
                "untrained_tool_calls": len(base_calls),
                "sft_tool_calls": len(tuned_calls),
                "untrained_tool_trace": _tool_call_audit(base),
                "sft_tool_trace": _tool_call_audit(tuned),
                "untrained_request_trace": _request_audit(base),
                "sft_request_trace": _request_audit(tuned),
                "untrained_intervals": base.get("observed_intervals") or [],
                "sft_intervals": tuned.get("observed_intervals") or [],
                "untrained_visual_tokens": _number(base, "visual_tokens"),
                "sft_visual_tokens": _number(tuned, "visual_tokens"),
                "untrained_total_tokens": _number(base, "total_tokens"),
                "sft_total_tokens": _number(tuned, "total_tokens"),
                "untrained_stop_reason": base.get("stop_reason")
                or base.get("run_stop_reasons"),
                "sft_stop_reason": tuned.get("stop_reason")
                or tuned.get("run_stop_reasons"),
                "untrained_candidate_changed": bool(base.get("candidate_changed")),
                "sft_candidate_changed": bool(tuned.get("candidate_changed")),
                "untrained_fallback_to_candidate": bool(
                    base.get("fallback_to_candidate")
                ),
                "sft_fallback_to_candidate": bool(
                    tuned.get("fallback_to_candidate")
                ),
                "untrained_raw_response": base.get("raw_response") or "",
                "sft_raw_response": tuned.get("raw_response") or "",
                "untrained_confirmation_raw_response": base.get(
                    "confirmation_raw_response"
                )
                or "",
                "sft_confirmation_raw_response": tuned.get(
                    "confirmation_raw_response"
                )
                or "",
                "untrained_funnel": _funnel(base),
                "sft_funnel": _funnel(tuned),
            }
        )
    return paired


def _failure_taxonomy(row: Mapping[str, Any]) -> tuple[str, ...]:
    """Map a wrong SFT result into the preregistered audit taxonomy."""

    if row.get("flip_type") in {"both_correct", "untrained_wrong_sft_correct"}:
        return ()
    modes = set(str(item) for item in row.get("failure_modes") or [])
    funnel = row.get("sft_funnel") if isinstance(row.get("sft_funnel"), Mapping) else {}
    categories: list[str] = []
    if "engineering_failure" in modes:
        categories.append("engineering")
    if funnel.get("target_hit") is False:
        categories.append("localization")
    if funnel.get("evidence_state_valid") is False or "invalid_evidence_state" in modes:
        categories.append("visual_fact_extraction")
    if (
        funnel.get("evidence_state_valid") is True
        and funnel.get("evidence_complete") is False
        and int(row.get("sft_tool_calls") or 0) > 1
    ):
        categories.append("cross_interval_memory")
    if (
        funnel.get("evidence_complete") is False
        or "possible_early_stop" in modes
        or "incomplete_evidence_stop" in modes
    ):
        categories.append("incomplete_evidence_early_stop")
    if funnel.get("evidence_complete") is True and funnel.get("judge_correct") is False:
        categories.append("judging")
    if {"wrong_candidate_preserved", "correct_candidate_regressed"}.intersection(modes):
        categories.append("candidate_gate")
    # Old Fast-Hybrid traces do not always expose every funnel field.  A wrong
    # model answer with no observable engineering/localization signal is a
    # judging failure, not silently "unclassified".
    if not categories:
        categories.append("judging")
    return tuple(dict.fromkeys(categories))


def summarize_pairs(rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    flip_names = (
        "untrained_correct_sft_wrong",
        "untrained_wrong_sft_correct",
        "both_correct",
        "both_wrong",
    )
    failure_names = (
        "localization",
        "visual_fact_extraction",
        "cross_interval_memory",
        "incomplete_evidence_early_stop",
        "judging",
        "candidate_gate",
        "engineering",
    )
    by_dataset: dict[str, dict[str, Any]] = {}
    for dataset in sorted({str(row["dataset"]) for row in rows}):
        selected = [row for row in rows if row["dataset"] == dataset]
        flips = Counter(str(row["flip_type"]) for row in selected)
        failures = Counter(
            mode for row in selected for mode in _failure_taxonomy(row)
        )
        token_fields: dict[str, float | None] = {}
        for key in (
            "untrained_visual_tokens",
            "sft_visual_tokens",
            "untrained_total_tokens",
            "sft_total_tokens",
        ):
            values = [float(row[key]) for row in selected if row.get(key) is not None]
            token_fields[f"mean_{key}"] = fmean(values) if values else None
        by_dataset[dataset] = {
            "samples": len(selected),
            "untrained_correct": flips["both_correct"]
            + flips["untrained_correct_sft_wrong"],
            "sft_correct": flips["both_correct"]
            + flips["untrained_wrong_sft_correct"],
            "flips": {name: int(flips[name]) for name in flip_names},
            "failure_modes": {name: int(failures[name]) for name in failure_names},
            **token_fields,
            "funnel": {
                stage: {
                    "known": sum(
                        row.get("sft_funnel", {}).get(stage) is not None
                        for row in selected
                    ),
                    "passed": sum(
                        row.get("sft_funnel", {}).get(stage) is True
                        for row in selected
                    ),
                }
                for stage in (
                    "target_hit",
                    "evidence_state_valid",
                    "evidence_complete",
                    "judge_correct",
                )
            },
        }
    flip_totals = Counter(str(row["flip_type"]) for row in rows)
    failure_totals = Counter(mode for row in rows for mode in _failure_taxonomy(row))
    wrong_rows = [
        row
        for row in rows
        if row.get("flip_type") in {"untrained_correct_sft_wrong", "both_wrong"}
    ]
    taxonomy_classified = sum(bool(_failure_taxonomy(row)) for row in wrong_rows)
    taxonomy_coverage_passed = taxonomy_classified == len(wrong_rows)
    scope_passed = taxonomy_coverage_passed and len(rows) == 300 and {
        dataset: value["samples"] for dataset, value in by_dataset.items()
    } == {"cgbench": 100, "lsdbench": 100, "lvbench": 100}
    return {
        "schema_version": 1,
        "status": "passed" if scope_passed else "failed",
        "scope_passed": scope_passed,
        "samples": len(rows),
        "paired_samples": len(rows),
        "datasets": by_dataset,
        "flip_totals": {name: int(flip_totals[name]) for name in flip_names},
        "failure_mode_totals": {
            name: int(failure_totals[name]) for name in failure_names
        },
        "taxonomy_required": len(wrong_rows),
        "taxonomy_classified": taxonomy_classified,
        "taxonomy_coverage_passed": taxonomy_coverage_passed,
        "limitations": [
            "target_hit is null unless a result already contains post-hoc annotation diagnostics",
            "possible_early_stop is a paired heuristic and must be verified from request traces",
        ],
    }


def write_badcase_report(
    output_dir: Path,
    rows: list[dict[str, Any]],
    summary: Mapping[str, Any],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "paired_badcases.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    lines = ["# Current Fast Hybrid SFT paired badcases", ""]
    lines.append("| Dataset | Untrained | SFT | Regressions | Gains |")
    lines.append("|---|---:|---:|---:|---:|")
    for dataset, value in summary["datasets"].items():
        flips = value["flips"]
        lines.append(
            f"| {dataset} | {value['untrained_correct']}/{value['samples']} | "
            f"{value['sft_correct']}/{value['samples']} | "
            f"{flips.get('untrained_correct_sft_wrong', 0)} | "
            f"{flips.get('untrained_wrong_sft_correct', 0)} |"
        )
    lines.extend(["", "## Failure modes", ""])
    for name, count in summary["failure_mode_totals"].items():
        lines.append(f"- {name}: {count}")
    lines.extend(["", "## Limitations", ""])
    for item in summary["limitations"]:
        lines.append(f"- {item}")
    (output_dir / "badcases.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
