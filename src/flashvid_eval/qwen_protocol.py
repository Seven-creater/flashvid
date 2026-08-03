from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Iterable

from .client import ChatResult


@dataclass(frozen=True)
class QwenInferenceProtocol:
    """Frozen request settings for an auditable Qwen MCQ run."""

    protocol_id: str
    enable_thinking: bool
    max_tokens: int
    length_retry_max_tokens: int | None = None
    temperature: float = 0.0
    top_p: float | None = None
    top_k: int | None = None
    min_p: float | None = None
    presence_penalty: float | None = None
    repetition_penalty: float | None = None

    def request_kwargs(
        self,
        *,
        max_tokens: int | None = None,
        json_mode: bool | None = None,
    ) -> dict[str, Any]:
        # vLLM applies structured-output grammar before its reasoning parser.
        # Constraining a thinking response to JSON therefore constrains the
        # hidden reasoning too and can prevent the parser from separating it.
        if json_mode is None:
            json_mode = not self.enable_thinking
        elif self.enable_thinking:
            json_mode = False
        kwargs: dict[str, Any] = {
            "max_tokens": self.max_tokens if max_tokens is None else max_tokens,
            "temperature": self.temperature,
            "chat_template_kwargs": {"enable_thinking": self.enable_thinking},
        }
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        sampling = {
            "top_p": self.top_p,
            "top_k": self.top_k,
            "min_p": self.min_p,
            "presence_penalty": self.presence_penalty,
            "repetition_penalty": self.repetition_penalty,
        }
        sampling = {key: value for key, value in sampling.items() if value is not None}
        if sampling:
            kwargs["sampling_params"] = sampling
        return kwargs


NO_THINK_PROTOCOL = QwenInferenceProtocol(
    protocol_id="no_think_v1",
    enable_thinking=False,
    max_tokens=512,
    temperature=0.7,
    top_p=0.8,
    top_k=20,
    min_p=0.0,
    presence_penalty=1.5,
    repetition_penalty=1.0,
)

THINK_PROTOCOL = QwenInferenceProtocol(
    protocol_id="think_v1",
    enable_thinking=True,
    max_tokens=8192,
    length_retry_max_tokens=32768,
    temperature=1.0,
    top_p=0.95,
    top_k=20,
    min_p=0.0,
    presence_penalty=1.5,
    repetition_penalty=1.0,
)

QWEN_PROTOCOLS = {
    "no_think": NO_THINK_PROTOCOL,
    "think": THINK_PROTOCOL,
}


def parse_strict_json_mcq_answer(
    content: str,
    valid_letters: Iterable[str],
) -> str | None:
    """Parse exactly ``{"answer": "X"}``, with no prose or extra fields."""

    valid = {str(letter).strip().upper() for letter in valid_letters}
    if not valid:
        return None
    try:
        payload = json.loads(content)
    except (TypeError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict) or set(payload) != {"answer"}:
        return None
    answer = payload["answer"]
    if not isinstance(answer, str):
        return None
    normalized = answer.strip().upper()
    if len(normalized) != 1 or normalized not in valid:
        return None
    return normalized


def next_length_retry_max_tokens(
    protocol: QwenInferenceProtocol,
    result: ChatResult,
    requested_max_tokens: int,
    *,
    prompt_tokens: int | None = None,
    max_model_len: int | None = None,
    reserve_tokens: int = 0,
) -> int | None:
    """Return the one allowed retry size after a truncated response."""

    retry_max = protocol.length_retry_max_tokens
    if (
        result.finish_reason != "length"
        or retry_max is None
        or requested_max_tokens >= retry_max
    ):
        return None
    if reserve_tokens < 0:
        raise ValueError("reserve_tokens must be non-negative")
    if max_model_len is not None:
        if max_model_len <= 0:
            raise ValueError("max_model_len must be positive")
        if prompt_tokens is None:
            return None
        retry_max = min(retry_max, max_model_len - prompt_tokens - reserve_tokens)
        if retry_max <= requested_max_tokens:
            return None
    return retry_max
