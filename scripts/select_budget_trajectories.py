from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from flashvid_eval.offline_budget import (
    retained_visual_tokens,
    select_training_trajectories,
    write_sft_jsonl,
)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _read_answers(path: Path) -> dict[str, str]:
    answers: dict[str, str] = {}
    for record in _read_jsonl(path):
        sample_id = str(record.get("sample_id") or "")
        if not sample_id:
            raise ValueError(f"answer record without sample_id in {path}")
        if sample_id in answers:
            raise ValueError(f"duplicate answer sample_id in {path}: {sample_id}")
        value = (
            record.get("answer")
            or record.get("correct_answer")
            or record.get("right_answer")
            or (record.get("reward_model") or {}).get("ground_truth")
        )
        answer = str(value or "").strip().upper()
        if not answer:
            raise ValueError(f"answer record without answer in {path}: {sample_id}")
        answers[sample_id] = answer
    return answers


def _write_jsonl(path: Path, records: tuple[dict[str, Any], ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Reject invalid trajectories, select cheapest correct traces, and export SFT JSONL."
    )
    parser.add_argument("--trajectories", type=Path, nargs="+", required=True)
    parser.add_argument("--answers", type=Path, required=True)
    parser.add_argument("--selected-output", type=Path, required=True)
    parser.add_argument("--sft-output", type=Path, required=True)
    parser.add_argument("--summary-output", type=Path)
    parser.add_argument("--second-trace-limit", type=float, default=1.2)
    parser.add_argument("--no-secondary", action="store_true")
    args = parser.parse_args()

    trajectories = [
        record
        for path in args.trajectories
        for record in _read_jsonl(path)
    ]
    answers = _read_answers(args.answers)
    selection = select_training_trajectories(
        trajectories,
        answers,
        second_trace_limit=None if args.no_secondary else args.second_trace_limit,
    )
    _write_jsonl(args.selected_output, selection.selected)
    sft_count = write_sft_jsonl(selection.selected, args.sft_output)

    primary_count = sum(
        record.get("_selection_role") == "primary"
        for record in selection.selected
    )
    secondary_count = len(selection.selected) - primary_count
    summary = {
        "inputs": {
            "trajectory_files": [str(path.resolve()) for path in args.trajectories],
            "trajectory_count": len(trajectories),
            "answer_file": str(args.answers.resolve()),
            "answer_count": len(answers),
        },
        "selection": {
            "primary_count": primary_count,
            "secondary_changed_candidate_count": secondary_count,
            "sft_record_count": sft_count,
            "no_positive_count": len(selection.no_positive_sample_ids),
            "no_positive_sample_ids": list(selection.no_positive_sample_ids),
            "unknown_sample_ids": list(selection.unknown_sample_ids),
            "total_retained_visual_tokens": sum(
                retained_visual_tokens(record) for record in selection.selected
            ),
            "second_trace_limit": None if args.no_secondary else args.second_trace_limit,
        },
        "outputs": {
            "selected": str(args.selected_output.resolve()),
            "sft": str(args.sft_output.resolve()),
        },
    }
    summary_path = args.summary_output or args.selected_output.with_suffix(".summary.json")
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = summary_path.with_suffix(summary_path.suffix + ".partial")
    temporary.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(summary_path)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
