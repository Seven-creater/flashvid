from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Iterable


FORBIDDEN_ANNOTATION_KEYS = frozenset(
    {
        "answer",
        "correct_answer",
        "right_answer",
        "time_range",
        "clue_intervals",
        "question_type",
        "ground_truth",
    }
)

# Public marker names used by tests and preflight. A caller can additionally
# pass a per-run secret value without persisting that value in result records.
DEFAULT_ANNOTATION_SENTINELS = (
    "__ANNOTATION_LEAK_SENTINEL__",
    "SECRET_ANNOTATION_SENTINEL",
)

_PUBLIC_ANSWER_SCHEMA_PARENT = (
    "$request.request_kwargs.response_format.json_schema.schema.properties"
)


class AnnotationLeakError(ValueError):
    """A model-bound payload contains benchmark-private annotation data."""


def _is_safe_public_answer_schema(value: Any) -> bool:
    if not isinstance(value, Mapping) or set(value) != {"type", "enum"}:
        return False
    if value["type"] != "string":
        return False
    choices = value["enum"]
    if not isinstance(choices, list) or len(choices) < 2:
        return False
    if any(
        not isinstance(choice, str)
        or len(choice) != 1
        or not choice.isascii()
        or not choice.isalpha()
        or choice != choice.upper()
        for choice in choices
    ):
        return False
    return len(set(choices)) == len(choices)


def assert_annotation_free_request(
    payload: Any,
    *,
    secret_sentinels: Iterable[str] = (),
) -> None:
    """Reject private annotation keys and sentinel values recursively.

    Annotation names are checked as structured keys, not substrings in natural
    language prompts. Sentinel values are checked in every string so tests can
    prove that a private value cannot cross the request boundary.
    """

    sentinels = tuple(
        value
        for value in (*DEFAULT_ANNOTATION_SENTINELS, *secret_sentinels)
        if value
    )

    def visit(value: Any, path: str) -> None:
        if isinstance(value, Mapping):
            for raw_key, child in value.items():
                key = str(raw_key).strip().lower()
                is_public_answer_schema = (
                    key == "answer" and path == _PUBLIC_ANSWER_SCHEMA_PARENT
                    and _is_safe_public_answer_schema(child)
                )
                if key in FORBIDDEN_ANNOTATION_KEYS and not is_public_answer_schema:
                    raise AnnotationLeakError(
                        f"forbidden annotation key at {path}.{key}"
                    )
                visit(child, f"{path}.{key}")
            return
        if isinstance(value, str):
            if any(sentinel in value for sentinel in sentinels):
                raise AnnotationLeakError(f"annotation sentinel found at {path}")
            return
        if isinstance(value, Sequence) and not isinstance(
            value, (bytes, bytearray)
        ):
            for index, child in enumerate(value):
                visit(child, f"{path}[{index}]")

    visit(payload, "$request")


def assert_deferred_result_public(payload: Any) -> None:
    """Fail closed when an unscored training artifact contains private labels."""

    assert_annotation_free_request(payload)

    def visit(value: Any, path: str) -> None:
        if isinstance(value, Mapping):
            for raw_key, child in value.items():
                key = str(raw_key).strip().lower()
                if key in {"correct", "is_correct", "reward", "label"}:
                    raise AnnotationLeakError(
                        f"forbidden deferred-scoring key at {path}.{key}"
                    )
                visit(child, f"{path}.{key}")
            return
        if isinstance(value, Sequence) and not isinstance(
            value, (str, bytes, bytearray)
        ):
            for index, child in enumerate(value):
                visit(child, f"{path}[{index}]")

    visit(payload, "$deferred_result")
