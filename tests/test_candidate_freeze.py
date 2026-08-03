from __future__ import annotations

import json
from pathlib import Path

import pytest

from flashvid_eval.client import ChatResult
from flashvid_eval.schemas import Sample
from scripts.evaluate_mcq import _normalize_frozen_candidates


class _NormalizerClient:
    def __init__(self, answer: str = "B"):
        self.answer = answer
        self.calls = 0

    def chat(self, model, messages, **kwargs):
        self.calls += 1
        return ChatResult(
            content=f"Answer: {self.answer}",
            usage={},
            raw={},
            latency_s=0.0,
        )


def _sample(sample_id: str) -> Sample:
    return Sample(
        dataset="demo",
        sample_id=sample_id,
        video=f"{sample_id}.mp4",
        question="What happens?",
        choices={"A": "opens", "B": "closes"},
        answer="A",
    )


def _write_candidates(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_candidate_normalization_cache_expands_and_is_read_only_across_models(
    tmp_path: Path,
) -> None:
    candidates = tmp_path / "direct.jsonl"
    cache = tmp_path / "normalized.jsonl"
    first, second = _sample("one"), _sample("two")
    _write_candidates(
        candidates,
        [
            {
                "dataset": "demo",
                "sample_id": "one",
                "video": "one.mp4",
                "prediction": "A",
                "raw_response": "Answer: A",
            },
            {
                "dataset": "demo",
                "sample_id": "two",
                "video": "two.mp4",
                "prediction": None,
                "raw_response": "The model did not format its output.",
            },
        ],
    )
    client = _NormalizerClient("B")

    smoke = _normalize_frozen_candidates(
        candidates,
        [first],
        client,
        "Qwen3.5-9B",
        cache,
    )
    assert smoke[0] == {"one": "A"}
    assert smoke[3] == 0

    full = _normalize_frozen_candidates(
        candidates,
        [first, second],
        client,
        "Qwen3.5-9B",
        cache,
    )
    assert full[0] == {"one": "A", "two": "B"}
    assert full[3] == 1
    assert len(cache.read_text(encoding="utf-8").splitlines()) == 2

    read_only_client = _NormalizerClient("A")
    reused = _normalize_frozen_candidates(
        candidates,
        [first, second],
        read_only_client,
        "Qwen3.5-4B-Agent-SFT",
        cache,
        read_only=True,
    )
    assert reused[0] == full[0]
    assert reused[4] == full[4]
    assert read_only_client.calls == 0


def test_candidate_normalization_read_only_rejects_missing_rows(
    tmp_path: Path,
) -> None:
    candidates = tmp_path / "direct.jsonl"
    cache = tmp_path / "normalized.jsonl"
    first, second = _sample("one"), _sample("two")
    _write_candidates(
        candidates,
        [
            {"sample_id": "one", "video": "one.mp4", "prediction": "A"},
            {"sample_id": "two", "video": "two.mp4", "prediction": "B"},
        ],
    )
    client = _NormalizerClient()
    _normalize_frozen_candidates(
        candidates,
        [first],
        client,
        "Qwen3.5-9B",
        cache,
    )
    with pytest.raises(RuntimeError, match="read-only candidate normalization cache"):
        _normalize_frozen_candidates(
            candidates,
            [first, second],
            client,
            "Qwen3.5-4B",
            cache,
            read_only=True,
        )


def test_candidate_file_must_cover_manifest_and_match_video(tmp_path: Path) -> None:
    candidates = tmp_path / "direct.jsonl"
    cache = tmp_path / "normalized.jsonl"
    sample = _sample("one")
    client = _NormalizerClient()
    _write_candidates(candidates, [])
    with pytest.raises(ValueError, match="missing 1 manifest samples"):
        _normalize_frozen_candidates(
            candidates,
            [sample],
            client,
            "Qwen3.5-9B",
            cache,
        )

    _write_candidates(
        candidates,
        [{"sample_id": "one", "video": "different.mp4", "prediction": "A"}],
    )
    with pytest.raises(ValueError, match="candidate video mismatch"):
        _normalize_frozen_candidates(
            candidates,
            [sample],
            client,
            "Qwen3.5-9B",
            cache,
        )
