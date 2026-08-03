from __future__ import annotations

import io
import json

from flashvid_eval.client import ChatResult, OpenAICompatibleClient
from flashvid_eval.qwen_protocol import (
    NO_THINK_PROTOCOL,
    THINK_PROTOCOL,
    next_length_retry_max_tokens,
    parse_strict_json_mcq_answer,
)


def test_client_captures_reasoning_finish_reason_and_explicit_request_fields(
    monkeypatch,
) -> None:
    captured: dict = {}
    response = {
        "choices": [
            {
                "message": {
                    "content": '{"answer":"B"}',
                    "reasoning": "private reasoning",
                    "reasoning_content": "legacy reasoning",
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 2},
    }

    def fake_urlopen(request, timeout):
        captured.update(json.loads(request.data.decode("utf-8")))
        return io.BytesIO(json.dumps(response).encode("utf-8"))

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    result = OpenAICompatibleClient("http://localhost:8000/v1").chat(
        "Qwen",
        [{"role": "user", "content": "Q"}],
        max_tokens=512,
        chat_template_kwargs={"enable_thinking": False},
        sampling_params={"top_p": 0.8},
        mm_processor_kwargs={"num_frames": 32},
        media_io_kwargs={"video_load_backend": "opencv"},
    )

    assert result.content == '{"answer":"B"}'
    assert result.reasoning_content == "private reasoning"
    assert result.finish_reason == "stop"
    assert captured["chat_template_kwargs"] == {"enable_thinking": False}
    assert captured["top_p"] == 0.8
    assert captured["mm_processor_kwargs"] == {"num_frames": 32}
    assert captured["media_io_kwargs"] == {"video_load_backend": "opencv"}


def test_chat_result_new_fields_are_backward_compatible() -> None:
    result = ChatResult("A", {}, {}, 0.0)
    assert result.reasoning_content == ""
    assert result.finish_reason is None


def test_protocols_expose_thinking_and_length_retry_policy() -> None:
    no_think = NO_THINK_PROTOCOL.request_kwargs()
    assert no_think["max_tokens"] == 512
    assert no_think["temperature"] == 0.7
    assert no_think["response_format"] == {"type": "json_object"}
    assert no_think["chat_template_kwargs"] == {"enable_thinking": False}
    assert no_think["sampling_params"] == {
        "top_p": 0.8,
        "top_k": 20,
        "min_p": 0.0,
        "presence_penalty": 1.5,
        "repetition_penalty": 1.0,
    }
    assert THINK_PROTOCOL.request_kwargs()["max_tokens"] == 8192
    assert THINK_PROTOCOL.request_kwargs()["sampling_params"]["top_p"] == 0.95
    assert THINK_PROTOCOL.request_kwargs()["chat_template_kwargs"] == {
        "enable_thinking": True
    }
    assert "response_format" not in THINK_PROTOCOL.request_kwargs()
    assert "response_format" not in THINK_PROTOCOL.request_kwargs(json_mode=True)
    truncated = ChatResult("", {}, {}, 0.0, finish_reason="length")
    assert next_length_retry_max_tokens(THINK_PROTOCOL, truncated, 8192) == 32768
    assert next_length_retry_max_tokens(THINK_PROTOCOL, truncated, 32768) is None
    assert next_length_retry_max_tokens(NO_THINK_PROTOCOL, truncated, 512) is None
    assert next_length_retry_max_tokens(
        THINK_PROTOCOL,
        truncated,
        8192,
        prompt_tokens=120_000,
        max_model_len=131_072,
        reserve_tokens=16,
    ) == 11_056
    assert next_length_retry_max_tokens(
        THINK_PROTOCOL,
        truncated,
        8192,
        prompt_tokens=123_000,
        max_model_len=131_072,
        reserve_tokens=16,
    ) is None


def test_client_accepts_legacy_reasoning_content(monkeypatch) -> None:
    response = {
        "choices": [
            {
                "message": {
                    "content": '{"answer":"A"}',
                    "reasoning_content": "legacy private reasoning",
                },
                "finish_reason": "stop",
            }
        ]
    }

    def fake_urlopen(request, timeout):
        return io.BytesIO(json.dumps(response).encode("utf-8"))

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    result = OpenAICompatibleClient("http://localhost:8000/v1").chat(
        "Qwen", [{"role": "user", "content": "Q"}]
    )
    assert result.reasoning_content == "legacy private reasoning"


def test_strict_json_answer_parser_rejects_prose_fences_and_extra_fields() -> None:
    valid = list("ABCD")
    assert parse_strict_json_mcq_answer('{"answer":"B"}', valid) == "B"
    assert parse_strict_json_mcq_answer('{"answer":"b"}', valid) == "B"
    assert parse_strict_json_mcq_answer('Answer: B', valid) is None
    assert parse_strict_json_mcq_answer('```json\n{"answer":"B"}\n```', valid) is None
    assert parse_strict_json_mcq_answer('{"answer":"B","reason":"x"}', valid) is None
    assert parse_strict_json_mcq_answer('{"answer":"H"}', valid) is None
