#!/usr/bin/env python3
"""Hard preflight for the FlashVID budget SFT dataset.

The exported SFT JSONL intentionally contains no ground-truth answer.  To prove
that its rows are positive trajectories, this checker cross-links every SFT row
with the selected trajectory JSONL and the frozen training manifests.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import re
import sys
import tempfile
from typing import Any, Iterable, Mapping, Sequence


ALLOWED_BUDGETS = (0.10, 0.25, 0.50, 1.00)
_TOOL_CALL_RE = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)
_FINAL_ANSWER_RE = re.compile(r"^\s*Answer:\s*([A-H])\s*$")
_PRIVATE_TEXT_RE = re.compile(
    r"(?i)(?:ground[_ -]?truth|correct[_ -]?answer|right[_ -]?answer|"
    r"time[_ -]?range|clue[_ -]?intervals?|question[_ -]?type)\s*[:=]"
)
_PRIVATE_KEYS = {
    "answer",
    "correct_answer",
    "right_answer",
    "ground_truth",
    "time_range",
    "clue_intervals",
    "question_type",
    "reward_model",
}
_ERROR_FIELDS = (
    "error",
    "api_error",
    "frame_error",
    "transcode_error",
    "perception_error",
    "parse_error",
    "failure_stage",
)


class ErrorCollector:
    def __init__(self, limit: int = 100):
        self.limit = limit
        self.total = 0
        self.items: list[str] = []

    def add(self, message: str) -> None:
        self.total += 1
        if len(self.items) < self.limit:
            self.items.append(message)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"{path}:{line_number}: invalid JSON: {exc}"
                ) from exc
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: row must be an object")
            records.append(value)
    return records


def _identity(record: Mapping[str, Any], *, metadata: bool) -> tuple[str, str, str]:
    source = record.get("metadata") if metadata else record
    if not isinstance(source, Mapping):
        raise ValueError("metadata must be an object")
    values = tuple(
        str(source.get(key) or "").strip()
        for key in ("dataset", "sample_id", "trajectory_id")
    )
    if not all(values):
        raise ValueError("dataset, sample_id, and trajectory_id are required")
    return values


def _normalized_key(value: Any) -> str:
    return str(value).strip().casefold().replace("-", "_").replace(" ", "_")


def _find_private_value(value: Any, location: str = "$") -> str | None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            if _normalized_key(key) in _PRIVATE_KEYS:
                return f"{location}.{key}"
            found = _find_private_value(item, f"{location}.{key}")
            if found:
                return found
    elif isinstance(value, list):
        for index, item in enumerate(value):
            found = _find_private_value(item, f"{location}[{index}]")
            if found:
                return found
    elif isinstance(value, str):
        if _PRIVATE_TEXT_RE.search(value):
            return location
        stripped = value.strip()
        if stripped.startswith(("{", "[")):
            try:
                parsed = json.loads(stripped)
            except json.JSONDecodeError:
                parsed = None
            if parsed is not None:
                return _find_private_value(parsed, location)
    return None


def _parse_assistant_targets(
    messages: Any,
) -> tuple[list[float], str | None]:
    if not isinstance(messages, list) or not messages:
        return [], "messages must be a non-empty list"
    tool_ratios: list[float] = []
    tool_target_count = 0
    final_target_count = 0
    tool_response_count = 0
    for index, message in enumerate(messages):
        if not isinstance(message, Mapping):
            return [], f"messages[{index}] must be an object"
        role = str(message.get("role") or "")
        content = message.get("content")
        if role not in {"system", "user", "assistant", "tool"}:
            return [], f"messages[{index}] has unsupported role {role!r}"
        if not isinstance(content, str):
            return [], f"messages[{index}].content must be a string"
        if role == "tool":
            tool_response_count += 1
            continue
        if role != "assistant":
            continue
        final_match = _FINAL_ANSWER_RE.fullmatch(content)
        if final_match:
            final_target_count += 1
            continue
        blocks = list(_TOOL_CALL_RE.finditer(content))
        if not blocks:
            return [], (
                f"messages[{index}] assistant target is neither an official "
                "tool call nor a strict final answer"
            )
        residual = _TOOL_CALL_RE.sub("", content)
        if residual.strip():
            return [], f"messages[{index}] mixes tool calls with free-form text"
        for block in blocks:
            try:
                payload = json.loads(block.group(1))
            except json.JSONDecodeError as exc:
                return [], f"messages[{index}] has invalid tool JSON: {exc}"
            if not isinstance(payload, dict) or payload.get("tool") != "frame_select":
                return [], f"messages[{index}] uses a non-frame_select tool"
            arguments = payload.get("arguments")
            if not isinstance(arguments, dict):
                return [], f"messages[{index}] tool arguments must be an object"
            try:
                ratio = float(arguments["retention_ratio"])
            except (KeyError, TypeError, ValueError):
                return [], f"messages[{index}] has no valid retention_ratio"
            if ratio not in ALLOWED_BUDGETS:
                return [], f"messages[{index}] has unsupported budget {ratio}"
            tool_ratios.append(ratio)
            tool_target_count += 1
    if tool_target_count == 0:
        return [], "no assistant frame_select training target"
    if tool_response_count == 0:
        return [], "no tool observation"
    if final_target_count != 1:
        return [], f"expected exactly one strict final target, found {final_target_count}"
    last = messages[-1]
    if (
        not isinstance(last, Mapping)
        or last.get("role") != "assistant"
        or not isinstance(last.get("content"), str)
        or _FINAL_ANSWER_RE.fullmatch(last["content"]) is None
    ):
        return [], "strict final answer must be the last message"
    return tool_ratios, None


def _answer_from_manifest(record: Mapping[str, Any]) -> str | None:
    value = (
        record.get("answer")
        or record.get("correct_answer")
        or record.get("right_answer")
        or (
            record.get("reward_model", {}).get("ground_truth")
            if isinstance(record.get("reward_model"), Mapping)
            else None
        )
    )
    answer = str(value or "").strip().upper()
    return answer if re.fullmatch(r"[A-H]", answer) else None


def _selected_budget_sequence(record: Mapping[str, Any]) -> list[float]:
    sequence = record.get("budget_sequence")
    if isinstance(sequence, list) and sequence:
        try:
            result = [float(value) for value in sequence]
        except (TypeError, ValueError):
            return []
        return result if all(value in ALLOWED_BUDGETS for value in result) else []
    steps = record.get("tool_steps")
    if not isinstance(steps, list):
        return []
    try:
        result = [
            float(step["retention_ratio"])
            for step in steps
            if isinstance(step, Mapping)
        ]
    except (KeyError, TypeError, ValueError):
        return []
    return result if result and all(value in ALLOWED_BUDGETS for value in result) else []


def validate_sft_data(
    sft_records: Sequence[Mapping[str, Any]],
    selected_records: Sequence[Mapping[str, Any]],
    answer_records: Sequence[Mapping[str, Any]],
    *,
    minimum_positive_trajectories: int = 300,
    minimum_budget_levels: int = 3,
    maximum_budget_share: float = 0.70,
) -> dict[str, Any]:
    errors = ErrorCollector()

    answers: dict[tuple[str, str], str] = {}
    for index, record in enumerate(answer_records):
        dataset = str(record.get("dataset") or "").strip()
        sample_id = str(record.get("sample_id") or "").strip()
        answer = _answer_from_manifest(record)
        key = (dataset, sample_id)
        if not dataset or not sample_id or answer is None:
            errors.add(f"answer[{index}] has invalid dataset/sample_id/answer")
        elif key in answers:
            errors.add(f"duplicate answer key: {dataset}/{sample_id}")
        else:
            answers[key] = answer

    selected: dict[tuple[str, str, str], Mapping[str, Any]] = {}
    positive_selected_trace_count = 0
    positive_primary_sample_ids: set[tuple[str, str]] = set()
    primary_selection_counts: Counter[tuple[str, str]] = Counter()
    for index, record in enumerate(selected_records):
        try:
            identity = _identity(record, metadata=False)
        except ValueError as exc:
            errors.add(f"selected[{index}]: {exc}")
            continue
        if identity in selected:
            errors.add(f"duplicate selected trajectory: {'/'.join(identity)}")
            continue
        selected[identity] = record
        answer = answers.get(identity[:2])
        prediction = str(
            record.get("final_prediction") or record.get("prediction") or ""
        ).strip().upper()
        role = str(record.get("_selection_role") or "")
        if role == "primary":
            primary_selection_counts[identity[:2]] += 1
        if answer is None:
            errors.add(f"selected trajectory has no answer: {'/'.join(identity)}")
        elif prediction != answer:
            errors.add(
                f"selected trajectory is not positive: {'/'.join(identity)} "
                f"prediction={prediction!r} answer={answer!r}"
            )
        elif record.get("trajectory_valid") is not True:
            errors.add(f"selected trajectory is not valid: {'/'.join(identity)}")
        elif record.get("annotation_leak_check") != "passed":
            errors.add(f"selected trajectory failed leak check: {'/'.join(identity)}")
        elif any(record.get(field) for field in _ERROR_FIELDS):
            errors.add(f"selected trajectory has an error: {'/'.join(identity)}")
        elif role not in {"primary", "secondary_changed_candidate"}:
            errors.add(f"selected trajectory has invalid selection role: {'/'.join(identity)}")
        else:
            positive_selected_trace_count += 1
            if role == "primary":
                positive_primary_sample_ids.add(identity[:2])

    for (dataset, sample_id), count in sorted(primary_selection_counts.items()):
        if count > 1:
            errors.add(
                "multiple primary selected trajectories for sample: "
                f"{dataset}/{sample_id} count={count}"
            )

    positive_primary_sample_count = len(positive_primary_sample_ids)

    sft: dict[tuple[str, str, str], Mapping[str, Any]] = {}
    budget_counts: Counter[float] = Counter()
    trajectories_by_budget: Counter[float] = Counter()
    for index, record in enumerate(sft_records):
        try:
            identity = _identity(record, metadata=True)
        except ValueError as exc:
            errors.add(f"sft[{index}]: {exc}")
            continue
        if identity in sft:
            errors.add(f"duplicate SFT trajectory: {'/'.join(identity)}")
            continue
        sft[identity] = record
        private_location = _find_private_value(record)
        if private_location:
            errors.add(
                f"SFT trajectory contains private field/marker at {private_location}: "
                f"{'/'.join(identity)}"
            )
        ratios, target_error = _parse_assistant_targets(record.get("messages"))
        if target_error:
            errors.add(f"SFT trajectory {'/'.join(identity)}: {target_error}")
            continue
        budget_counts.update(ratios)
        trajectories_by_budget.update(set(ratios))

        source = selected.get(identity)
        if source is None:
            errors.add(f"SFT trajectory has no selected source: {'/'.join(identity)}")
            continue
        metadata = record.get("metadata")
        assert isinstance(metadata, Mapping)
        if metadata.get("selection_role") != source.get("_selection_role"):
            errors.add(f"selection role mismatch: {'/'.join(identity)}")
        source_cost = source.get("retained_visual_tokens")
        metadata_cost = metadata.get("retained_visual_tokens")
        try:
            costs_match = math.isclose(
                float(source_cost), float(metadata_cost), rel_tol=0.0, abs_tol=1e-6
            )
        except (TypeError, ValueError):
            costs_match = False
        if not costs_match:
            errors.add(f"retained token cost mismatch: {'/'.join(identity)}")
        source_ratios = _selected_budget_sequence(source)
        if ratios != source_ratios:
            errors.add(
                f"budget sequence mismatch: {'/'.join(identity)} "
                f"SFT={ratios} selected={source_ratios}"
            )

    for missing in sorted(set(selected) - set(sft)):
        errors.add(f"selected trajectory missing from SFT: {'/'.join(missing)}")

    total_budget_calls = sum(budget_counts.values())
    shares = {
        f"{ratio:.2f}": (
            budget_counts[ratio] / total_budget_calls if total_budget_calls else 0.0
        )
        for ratio in ALLOWED_BUDGETS
    }
    used_levels = sum(count > 0 for count in budget_counts.values())
    largest_share = max(shares.values(), default=0.0)

    constraints = {
        "positive_trajectories": {
            "actual": positive_primary_sample_count,
            "minimum": minimum_positive_trajectories,
            "unit": "unique_dataset_sample_with_verified_primary_selected_trajectory",
            "passed": positive_primary_sample_count >= minimum_positive_trajectories,
        },
        "budget_levels": {
            "actual": used_levels,
            "minimum": minimum_budget_levels,
            "passed": used_levels >= minimum_budget_levels,
        },
        "maximum_budget_share": {
            "actual": largest_share,
            "maximum": maximum_budget_share,
            "unit": "share_of_assistant_frame_select_targets",
            "passed": largest_share <= maximum_budget_share,
        },
        "private_fields": {
            "actual": sum("private field/marker" in item for item in errors.items),
            "maximum": 0,
            "passed": not any("private field/marker" in item for item in errors.items),
        },
        "source_alignment": {
            "sft_records": len(sft),
            "selected_records": len(selected),
            "passed": set(sft) == set(selected),
        },
    }
    for name, check in constraints.items():
        if not check["passed"]:
            errors.add(f"constraint failed: {name}")

    return {
        "status": "passed" if errors.total == 0 else "failed",
        "constraints": constraints,
        "counts": {
            "sft_records": len(sft_records),
            "selected_records": len(selected_records),
            "selected_trace_count": len(selected_records),
            "answer_records": len(answer_records),
            # Compatibility name: this is deliberately the number used by the
            # >=300 gate, i.e. unique samples with a verified primary trace.
            "verified_positive_trajectories": positive_primary_sample_count,
            "verified_positive_primary_samples": positive_primary_sample_count,
            "verified_positive_selected_traces": positive_selected_trace_count,
            "assistant_frame_select_targets": total_budget_calls,
        },
        "budget_distribution": {
            "tool_call_counts": {
                f"{ratio:.2f}": budget_counts[ratio] for ratio in ALLOWED_BUDGETS
            },
            "tool_call_shares": shares,
            "trajectory_presence_counts": {
                f"{ratio:.2f}": trajectories_by_budget[ratio]
                for ratio in ALLOWED_BUDGETS
            },
        },
        "error_count": errors.total,
        "errors": errors.items,
        "errors_truncated": errors.total > len(errors.items),
    }


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    temporary.replace(path)


def _flatten(paths: Iterable[Path]) -> list[dict[str, Any]]:
    return [record for path in paths for record in _read_jsonl(path)]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Verify positive-trajectory provenance, budget diversity, privacy, "
            "and assistant targets before FlashVID budget SFT."
        )
    )
    parser.add_argument("--sft-data", type=Path, required=True)
    parser.add_argument(
        "--selected-trajectories", type=Path, nargs="+", required=True
    )
    parser.add_argument("--answer-manifests", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--minimum-positive-trajectories", type=int, default=300)
    parser.add_argument("--minimum-budget-levels", type=int, default=3)
    parser.add_argument("--maximum-budget-share", type=float, default=0.70)
    args = parser.parse_args(argv)

    try:
        if args.minimum_positive_trajectories < 1:
            raise ValueError("minimum-positive-trajectories must be positive")
        if not 1 <= args.minimum_budget_levels <= len(ALLOWED_BUDGETS):
            raise ValueError("minimum-budget-levels is outside the supported range")
        if not 0 < args.maximum_budget_share <= 1:
            raise ValueError("maximum-budget-share must be in (0, 1]")
        report = validate_sft_data(
            _read_jsonl(args.sft_data),
            _flatten(args.selected_trajectories),
            _flatten(args.answer_manifests),
            minimum_positive_trajectories=args.minimum_positive_trajectories,
            minimum_budget_levels=args.minimum_budget_levels,
            maximum_budget_share=args.maximum_budget_share,
        )
        report["inputs"] = {
            "sft_data": str(args.sft_data.resolve()),
            "sft_sha256": _sha256(args.sft_data),
            "selected_trajectories": [
                {"path": str(path.resolve()), "sha256": _sha256(path)}
                for path in args.selected_trajectories
            ],
            "answer_manifests": [
                {"path": str(path.resolve()), "sha256": _sha256(path)}
                for path in args.answer_manifests
            ],
        }
    except Exception as exc:
        report = {
            "status": "failed",
            "error_count": 1,
            "errors": [f"{type(exc).__name__}: {exc}"],
        }

    _atomic_write_json(args.output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report.get("status") == "passed" else 1


if __name__ == "__main__":
    sys.exit(main())
