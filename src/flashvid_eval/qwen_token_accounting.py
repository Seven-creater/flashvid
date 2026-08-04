from __future__ import annotations

from collections import Counter
from copy import deepcopy
from typing import Any, Mapping


# Qwen3.5-4B and Qwen3.5-9B share this tokenizer vocabulary.  Every live
# service preflight validates these IDs through returned prompt token IDs.
QWEN_VISION_START_TOKEN_ID = 248053
QWEN_VISION_END_TOKEN_ID = 248054
QWEN_IMAGE_PAD_TOKEN_ID = 248056
QWEN_VIDEO_PAD_TOKEN_ID = 248057
QWEN_SPECIAL_TOKEN_IDS = {
    "<|vision_start|>": QWEN_VISION_START_TOKEN_ID,
    "<|vision_end|>": QWEN_VISION_END_TOKEN_ID,
    "<|image_pad|>": QWEN_IMAGE_PAD_TOKEN_ID,
    "<|video_pad|>": QWEN_VIDEO_PAD_TOKEN_ID,
}


class PromptTokenAccountingError(ValueError):
    """Returned token IDs cannot support exact, auditable accounting."""


def qwen_prompt_token_accounting(
    raw_response: Mapping[str, Any],
    usage: Mapping[str, Any],
) -> dict[str, Any]:
    """Count exact Qwen visual prompt tokens returned by vLLM.

    vLLM 0.25 exposes ``prompt_token_ids`` when a request sets
    ``return_token_ids=true``.  Qwen expands every image/video patch into
    dedicated pad-token IDs, so counting those IDs is exact and avoids an
    extra decode/tokenize request.
    """

    raw_ids = raw_response.get("prompt_token_ids")
    if not isinstance(raw_ids, list) or any(
        isinstance(value, bool) or not isinstance(value, int) for value in raw_ids
    ):
        raise PromptTokenAccountingError("response is missing integer prompt_token_ids")
    prompt_tokens = usage.get("prompt_tokens")
    if (
        isinstance(prompt_tokens, int)
        and not isinstance(prompt_tokens, bool)
        and prompt_tokens != len(raw_ids)
    ):
        raise PromptTokenAccountingError(
            "prompt_token_ids length does not match usage.prompt_tokens: "
            f"{len(raw_ids)} != {prompt_tokens}"
        )

    counts = Counter(raw_ids)
    vision_start = counts[QWEN_VISION_START_TOKEN_ID]
    vision_end = counts[QWEN_VISION_END_TOKEN_ID]
    if vision_start != vision_end:
        raise PromptTokenAccountingError(
            f"unbalanced Qwen vision delimiters: {vision_start} != {vision_end}"
        )
    image_tokens = counts[QWEN_IMAGE_PAD_TOKEN_ID]
    video_tokens = counts[QWEN_VIDEO_PAD_TOKEN_ID]
    if image_tokens and video_tokens:
        media_kind = "mixed"
    elif video_tokens:
        media_kind = "video"
    elif image_tokens:
        media_kind = "image"
    else:
        media_kind = "none"
    return {
        "source": "vllm_returned_prompt_token_ids_v1",
        "prompt_token_ids_count": len(raw_ids),
        "vision_segment_count": vision_start,
        "image_tokens": image_tokens,
        "video_tokens": video_tokens,
        "visual_pad_tokens": image_tokens + video_tokens,
        "media_kind": media_kind,
    }


def enrich_usage_with_qwen_prompt_tokens(
    usage: Mapping[str, Any],
    raw_response: Mapping[str, Any],
) -> dict[str, Any]:
    """Return a copied usage object with exact multimodal-token details."""

    result = deepcopy(dict(usage))
    audit = qwen_prompt_token_accounting(raw_response, result)
    details = result.get("prompt_tokens_details")
    if details is None:
        details = {}
    elif not isinstance(details, Mapping):
        raise PromptTokenAccountingError("usage.prompt_tokens_details is not an object")
    else:
        details = deepcopy(dict(details))
    returned = details.get("multimodal_tokens")
    pad_tokens = {
        "image": int(audit["image_tokens"]),
        "video": int(audit["video_tokens"]),
    }
    if audit["media_kind"] != "none" and not isinstance(returned, Mapping):
        raise PromptTokenAccountingError(
            "response is missing service multimodal token details"
        )
    service_tokens: dict[str, int] = {}
    if isinstance(returned, Mapping):
        for kind, raw_value in returned.items():
            if (
                isinstance(raw_value, bool)
                or not isinstance(raw_value, int)
                or raw_value < 0
            ):
                raise PromptTokenAccountingError(
                    f"service {kind} multimodal token detail is invalid"
                )
            service_tokens[str(kind)] = int(raw_value)
        for kind, value in pad_tokens.items():
            if value and service_tokens.get(kind, 0) <= 0:
                raise PromptTokenAccountingError(
                    f"service is missing positive {kind} multimodal token detail"
                )
    # The service's multimodal total includes more than the repeated Qwen pad
    # IDs (for example video boundary/metadata tokens), so the two values are
    # complementary audit signals rather than quantities that must be equal.
    details["qwen_multimodal_pad_tokens"] = pad_tokens
    result["prompt_tokens_details"] = details
    audit["service_multimodal_tokens"] = service_tokens
    audit["service_visual_tokens"] = sum(service_tokens.values())
    audit["visual_token_source"] = "vllm_prompt_tokens_details_v1"
    result["qwen_prompt_token_accounting"] = audit
    return result
