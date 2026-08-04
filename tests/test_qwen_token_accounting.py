from __future__ import annotations

import pytest

from flashvid_eval.qwen_token_accounting import (
    PromptTokenAccountingError,
    enrich_usage_with_qwen_prompt_tokens,
    qwen_prompt_token_accounting,
)


def test_qwen_prompt_token_accounting_counts_exact_visual_tokens() -> None:
    ids = [1, 248053, 248057, 248057, 248054, 2]
    audit = qwen_prompt_token_accounting(
        {"prompt_token_ids": ids},
        {"prompt_tokens": len(ids)},
    )
    assert audit["video_tokens"] == 2
    assert audit["image_tokens"] == 0
    assert audit["visual_tokens"] == 2
    assert audit["vision_segment_count"] == 1
    assert audit["processed_video_frame_slots"] == 2


def test_qwen_prompt_token_accounting_enriches_usage_without_mutation() -> None:
    usage = {"prompt_tokens": 7, "completion_tokens": 1}
    ids = [1, 248053, 248056, 248056, 248056, 248054, 2]
    enriched = enrich_usage_with_qwen_prompt_tokens(
        usage,
        {"prompt_token_ids": ids},
    )
    assert usage == {"prompt_tokens": 7, "completion_tokens": 1}
    assert enriched["prompt_tokens_details"]["multimodal_tokens"] == {
        "image": 3,
        "video": 0,
    }
    assert enriched["qwen_prompt_token_accounting"]["media_kind"] == "image"


def test_qwen_prompt_token_accounting_fails_closed() -> None:
    with pytest.raises(PromptTokenAccountingError, match="missing"):
        qwen_prompt_token_accounting({}, {"prompt_tokens": 1})
    with pytest.raises(PromptTokenAccountingError, match="does not match"):
        qwen_prompt_token_accounting(
            {"prompt_token_ids": [1, 2]},
            {"prompt_tokens": 1},
        )
    with pytest.raises(PromptTokenAccountingError, match="unbalanced"):
        qwen_prompt_token_accounting(
            {"prompt_token_ids": [248053, 248057]},
            {"prompt_tokens": 2},
        )
