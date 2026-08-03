from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Iterable

from .schemas import Sample

_LETTERS = "ABCDEFGH"
_OPTION_RE = re.compile(
    r"(?m)^\s*(?:\(([A-H])\)|([A-H])[:.])\s*(.*?)(?=^\s*(?:\([A-H]\)|[A-H][:.])\s*|\Z)",
    re.DOTALL,
)


def _read_records(path: Path) -> list[dict[str, Any]]:
    if path.suffix.lower() == ".jsonl":
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        return data
    for key in ("data", "items", "annotations", "questions"):
        if isinstance(data.get(key), list):
            return data[key]
    raise ValueError(f"Expected a list of records in {path}")


def _letters_for_count(count: int) -> list[str]:
    if not 1 <= count <= len(_LETTERS):
        raise ValueError(f"Unsupported option count: {count}")
    return list(_LETTERS[:count])


def _choices_from_value(value: Any) -> dict[str, str]:
    if isinstance(value, dict):
        result = {str(key).strip().upper(): str(item).strip() for key, item in value.items()}
        if set(result) - set(_LETTERS):
            raise ValueError(f"Invalid option keys: {sorted(result)}")
        return {letter: result[letter] for letter in _LETTERS if letter in result}
    if isinstance(value, list):
        return dict(zip(_letters_for_count(len(value)), (str(item).strip() for item in value)))
    raise ValueError(f"Unsupported choices value: {type(value).__name__}")


def _sample_id(raw: dict[str, Any], fallback: int) -> str:
    for key in ("sample_id", "id", "qid", "uid", "index"):
        value = raw.get(key)
        if value is not None:
            return str(value)
    return str(fallback)


def _load_cgbench(records: Iterable[dict[str, Any]]) -> list[Sample]:
    samples = []
    for index, raw in enumerate(records):
        choices = _choices_from_value(raw.get("choices"))
        answer = str(raw.get("right_answer", "")).strip().upper()
        if answer not in choices:
            raise ValueError(f"CG-Bench item {_sample_id(raw, index)} has invalid answer {answer!r}")
        video_uid = str(raw.get("video_uid", "")).strip()
        if not video_uid:
            raise ValueError(f"CG-Bench item {_sample_id(raw, index)} has no video_uid")
        samples.append(
            Sample(
                dataset="cgbench",
                sample_id=_sample_id(raw, index),
                video=f"{video_uid}.mp4",
                question=str(raw.get("question", "")).strip(),
                choices=choices,
                answer=answer,
                metadata={
                    "clue_intervals": raw.get("clue_intervals"),
                    "domain": raw.get("domain"),
                    "sub_category": raw.get("sub_category"),
                    "duration": raw.get("duration"),
                },
            )
        )
    return samples


def _load_lsdbench(records: Iterable[dict[str, Any]]) -> list[Sample]:
    samples = []
    for index, raw in enumerate(records):
        choices = _choices_from_value(raw.get("options"))
        answer = str(raw.get("correct_answer", "")).strip().upper()
        if answer not in choices:
            raise ValueError(f"LSDBench item {_sample_id(raw, index)} has invalid answer {answer!r}")
        video_id = str(raw.get("video_id", "")).strip()
        if not video_id:
            raise ValueError(f"LSDBench item {_sample_id(raw, index)} has no video_id")
        samples.append(
            Sample(
                dataset="lsdbench",
                sample_id=_sample_id(raw, index),
                video=f"{video_id}.mp4",
                question=str(raw.get("question", "")).strip(),
                choices=choices,
                answer=answer,
                metadata={"time_range": raw.get("time_range"), "segment": raw.get("segment")},
            )
        )
    return samples


def _parse_lv_prompt(prompt: Any) -> tuple[str, dict[str, str]]:
    if isinstance(prompt, list):
        text = "\n".join(str(turn.get("content", "")) for turn in prompt if turn.get("role") == "user")
    else:
        text = str(prompt)
    # Some LVBench exports store line breaks as literal ``\n`` sequences.
    text = text.replace("\\r\\n", "\n").replace("\\n", "\n")
    matches = list(_OPTION_RE.finditer(text))
    if not matches:
        raise ValueError("LVBench prompt has no A/B/C/D-style options")
    question = text[: matches[0].start()].strip()
    choices = {
        (match.group(1) or match.group(2)).upper(): " ".join(match.group(3).split())
        for match in matches
    }
    return question, choices


def _load_lvbench(records: Iterable[dict[str, Any]]) -> list[Sample]:
    samples = []
    for index, raw in enumerate(records):
        question, choices = _parse_lv_prompt(raw.get("prompt", raw.get("question", "")))
        answer = str(raw.get("reward_model", {}).get("ground_truth", raw.get("answer", ""))).strip().upper()
        if answer not in choices:
            raise ValueError(f"LVBench item {_sample_id(raw, index)} has invalid answer {answer!r}")
        videos = raw.get("videos") or [raw.get("video")]
        video = str(videos[0] or "").strip()
        if not video:
            raise ValueError(f"LVBench item {_sample_id(raw, index)} has no video")
        samples.append(
            Sample(
                dataset="lvbench",
                sample_id=_sample_id(raw, index),
                video=video,
                question=question,
                choices=choices,
                answer=answer,
                metadata={
                    "time_reference": raw.get("time_reference"),
                    "question_type": raw.get("question_type"),
                    "uid": raw.get("uid"),
                    "video_type": raw.get("video_type"),
                },
            )
        )
    return samples


class VideoIndex:
    """Resolve dataset-relative names without requiring a flat directory."""

    def __init__(self, root: Path):
        self.root = root.resolve()
        self._by_name: dict[str, Path] = {}
        self._by_stem: dict[str, Path] = {}
        for path in root.rglob("*"):
            if path.is_file() and path.suffix.lower() in {".mp4", ".mkv", ".webm", ".mov"}:
                self._by_name.setdefault(path.name.lower(), path)
                self._by_stem.setdefault(path.stem.lower(), path)

    def resolve(self, relative: str) -> Path:
        candidate = (self.root / relative).resolve()
        if candidate.is_file():
            return candidate
        by_name = self._by_name.get(Path(relative).name.lower())
        if by_name:
            return by_name
        by_stem = self._by_stem.get(Path(relative).stem.lower())
        if by_stem:
            return by_stem
        raise FileNotFoundError(f"Video not found under {self.root}: {relative}")


def load_samples(dataset: str, annotations: str | Path) -> list[Sample]:
    """Load one supported dataset into the canonical representation."""

    dataset = dataset.lower()
    records = _read_records(Path(annotations))
    if dataset == "cgbench":
        samples = _load_cgbench(records)
    elif dataset == "lvbench":
        samples = _load_lvbench(records)
    elif dataset == "lsdbench":
        samples = _load_lsdbench(records)
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")
    if not samples:
        raise ValueError(f"No samples loaded from {annotations}")
    return samples
