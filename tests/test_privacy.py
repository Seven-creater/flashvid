from __future__ import annotations

import pytest

from flashvid_eval.privacy import (
    AnnotationLeakError,
    assert_annotation_free_request,
    assert_deferred_result_public,
)


@pytest.mark.parametrize(
    "key",
    (
        "answer",
        "correct_answer",
        "right_answer",
        "time_range",
        "clue_intervals",
        "question_type",
        "ground_truth",
    ),
)
def test_annotation_guard_rejects_private_keys_recursively(key: str) -> None:
    with pytest.raises(AnnotationLeakError, match="forbidden annotation key"):
        assert_annotation_free_request(
            [{"role": "user", "content": {"nested": {key: "PRIVATE"}}}]
        )


def test_annotation_guard_rejects_secret_sentinel_and_accepts_prompt_answer_text() -> None:
    assert_annotation_free_request(
        [{"role": "user", "content": 'Return JSON {"answer":"X"}'}]
    )
    with pytest.raises(AnnotationLeakError, match="sentinel"):
        assert_annotation_free_request(
            [{"role": "user", "content": "hidden=SENTINEL-42"}],
            secret_sentinels=("SENTINEL-42",),
        )


def test_annotation_guard_allows_only_the_public_answer_schema_property() -> None:
    assert_annotation_free_request(
        {
            "request_kwargs": {
                "response_format": {
                    "json_schema": {
                        "schema": {
                            "properties": {
                                "answer": {"type": "string", "enum": ["A", "B"]}
                            }
                        }
                    }
                }
            }
        }
    )
    with pytest.raises(AnnotationLeakError):
        assert_annotation_free_request(
            {"request_kwargs": {"other": {"properties": {"answer": "B"}}}}
        )


@pytest.mark.parametrize(
    "answer_schema",
    (
        {"const": "B"},
        {"type": "string", "enum": ["A", "B"], "description": "B is correct"},
        {"type": "string", "enum": ["A", "B"], "metadata": {"value": "B"}},
        {"type": "string", "enum": ["B"]},
        {"type": "string", "enum": ["A", "A"]},
        {"type": "string", "enum": ["A", "correct=B"]},
        {"type": "number", "enum": ["A", "B"]},
    ),
)
def test_annotation_guard_rejects_unsafe_public_answer_schema(
    answer_schema: dict[str, object],
) -> None:
    with pytest.raises(AnnotationLeakError, match="forbidden annotation key"):
        assert_annotation_free_request(
            {
                "request_kwargs": {
                    "response_format": {
                        "json_schema": {
                            "schema": {
                                "properties": {"answer": answer_schema}
                            }
                        }
                    }
                }
            }
        )


def test_deferred_result_guard_rejects_labels_but_allows_model_predictions() -> None:
    assert_deferred_result_public(
        {
            "prediction": "B",
            "request_trace": [{"content": '{"answer":"B"}'}],
        }
    )
    for payload in (
        {"answer": "B"},
        {"correct": True},
        {"nested": {"ground_truth": "B"}},
    ):
        with pytest.raises(AnnotationLeakError):
            assert_deferred_result_public(payload)
