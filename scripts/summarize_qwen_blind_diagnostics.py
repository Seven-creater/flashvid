#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import statistics
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping

from flashvid_eval.qwen_reporting import load_jsonl, paired_compare


TOKEN_RE = re.compile(r"[A-Za-z0-9]+|[\u4e00-\u9fff]{1,8}")


def _mean(values: Iterable[float]) -> float | None:
    clean = list(values)
    return statistics.fmean(clean) if clean else None


def _tokens(text: str) -> list[str]:
    return [token.casefold() for token in TOKEN_RE.findall(text)]


def summarize_run(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("diagnostic run is empty")
    answer_letters = Counter(str(row.get("answer") or "NONE") for row in rows)
    prediction_letters = Counter(str(row.get("prediction") or "NONE") for row in rows)
    displayed_letters = Counter(
        str(row.get("displayed_prediction") or "NONE") for row in rows
    )
    correct_by_answer: dict[str, dict[str, Any]] = {}
    for letter in sorted(answer_letters):
        subset = [row for row in rows if str(row.get("answer") or "NONE") == letter]
        correct = sum(bool(row.get("correct")) for row in subset)
        correct_by_answer[letter] = {
            "count": len(subset),
            "correct": correct,
            "accuracy": correct / len(subset) if subset else None,
        }

    correct_option_lengths: list[float] = []
    predicted_option_lengths: list[float] = []
    all_option_lengths: list[float] = []
    correct_option_tokens: Counter[str] = Counter()
    distractor_tokens: Counter[str] = Counter()
    for row in rows:
        choices = row.get("choices")
        if not isinstance(choices, Mapping):
            continue
        answer = str(row.get("answer") or "")
        prediction = str(row.get("prediction") or "")
        for letter, text in choices.items():
            length = float(len(str(text)))
            all_option_lengths.append(length)
            if str(letter) == answer:
                correct_option_lengths.append(length)
                correct_option_tokens.update(_tokens(str(text)))
            else:
                distractor_tokens.update(_tokens(str(text)))
            if str(letter) == prediction:
                predicted_option_lengths.append(length)

    errors = sum(
        bool(row.get("error") or row.get("parse_error")) for row in rows
    )
    correct = sum(bool(row.get("correct")) for row in rows)
    return {
        "dataset": rows[0].get("dataset"),
        "model": rows[0].get("model"),
        "baseline_mode": rows[0].get("baseline_mode"),
        "protocol_id": rows[0].get("protocol_id"),
        "total": len(rows),
        "correct": correct,
        "accuracy": correct / len(rows),
        "errors": errors,
        "media_items_total": sum(int(row.get("media_items") or 0) for row in rows),
        "answer_letter_counts": dict(sorted(answer_letters.items())),
        "prediction_letter_counts": dict(sorted(prediction_letters.items())),
        "displayed_prediction_letter_counts": dict(sorted(displayed_letters.items())),
        "accuracy_by_answer_letter": correct_by_answer,
        "option_length": {
            "mean_all": _mean(all_option_lengths),
            "mean_correct": _mean(correct_option_lengths),
            "mean_predicted": _mean(predicted_option_lengths),
        },
        "top_correct_option_tokens": correct_option_tokens.most_common(30),
        "top_distractor_tokens": distractor_tokens.most_common(30),
    }


def build_report(named_paths: list[tuple[str, Path]]) -> dict[str, Any]:
    runs = {name: load_jsonl(path) for name, path in named_paths}
    summaries = {name: summarize_run(rows) for name, rows in runs.items()}
    comparisons: dict[str, Any] = {}
    names = sorted(runs)
    for left_index, left in enumerate(names):
        for right in names[left_index + 1 :]:
            left_summary = summaries[left]
            right_summary = summaries[right]
            if (
                left_summary["dataset"] == right_summary["dataset"]
                and left_summary["model"] == right_summary["model"]
                and left_summary["protocol_id"] == right_summary["protocol_id"]
            ):
                comparisons[f"{left}__vs__{right}"] = paired_compare(
                    runs[left], runs[right]
                )
    return {
        "schema_version": 1,
        "diagnostic_only": True,
        "must_not_tune_agent_prompts": True,
        "runs": summaries,
        "paired_comparisons": comparisons,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Summarize Qwen no-video, permutation, and wrong-video controls."
    )
    parser.add_argument(
        "--run",
        action="append",
        required=True,
        metavar="NAME=JSONL",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    named_paths: list[tuple[str, Path]] = []
    seen: set[str] = set()
    for value in args.run:
        if "=" not in value:
            raise ValueError("--run must be NAME=JSONL")
        name, raw_path = value.split("=", 1)
        if not name or name in seen:
            raise ValueError("diagnostic run names must be unique and non-empty")
        path = Path(raw_path)
        if not path.is_file():
            raise FileNotFoundError(path)
        seen.add(name)
        named_paths.append((name, path))
    report = build_report(named_paths)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".partial")
    temporary.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(args.output)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
