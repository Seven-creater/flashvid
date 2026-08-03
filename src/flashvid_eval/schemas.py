from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class Sample:
    """One benchmark item in the evaluator's canonical representation."""

    dataset: str
    sample_id: str
    video: str
    question: str
    choices: dict[str, str]
    answer: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def option_letters(self) -> tuple[str, ...]:
        return tuple(self.choices)


@dataclass(frozen=True)
class ModelSample:
    """Annotation-free item that may be passed to model-facing code."""

    dataset: str
    sample_id: str
    video: str
    question: str
    choices: dict[str, str]
    candidate_answer: str | None

    @classmethod
    def from_sample(cls, sample: Sample, candidate_answer: str | None) -> "ModelSample":
        candidate = candidate_answer if candidate_answer in sample.option_letters else None
        return cls(
            dataset=sample.dataset,
            sample_id=sample.sample_id,
            video=sample.video,
            question=sample.question,
            choices=dict(sample.choices),
            candidate_answer=candidate,
        )

    @property
    def option_letters(self) -> tuple[str, ...]:
        return tuple(self.choices)


@dataclass(frozen=True)
class ScoringRecord:
    """Private labels joined only after model inference has completed."""

    dataset: str
    sample_id: str
    answer: str
    metadata: dict[str, Any]

    @classmethod
    def from_sample(cls, sample: Sample) -> "ScoringRecord":
        return cls(sample.dataset, sample.sample_id, sample.answer, dict(sample.metadata))
