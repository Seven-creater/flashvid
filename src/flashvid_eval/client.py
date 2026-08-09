from __future__ import annotations

import json
import time
import urllib.error
import urllib.parse
import urllib.request
from copy import deepcopy
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ChatResult:
    content: str
    usage: dict[str, Any]
    raw: dict[str, Any]
    latency_s: float
    reasoning_content: str = ""
    finish_reason: str | None = None


def _message_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, list):
        return "".join(
            str(item.get("text", ""))
            for item in value
            if isinstance(item, dict)
        )
    return str(value)


class OpenAICompatibleClient:
    def __init__(
        self,
        base_url: str,
        api_key: str = "no",
        timeout: float = 900.0,
        *,
        local_file_urls_as_paths: bool = False,
    ):
        self.endpoint = base_url.rstrip("/") + "/chat/completions"
        self.api_key = api_key
        self.timeout = timeout
        self.local_file_urls_as_paths = local_file_urls_as_paths

    @staticmethod
    def _local_media_paths(value: Any) -> Any:
        if isinstance(value, list):
            return [OpenAICompatibleClient._local_media_paths(item) for item in value]
        if not isinstance(value, dict):
            return value
        result = {
            key: OpenAICompatibleClient._local_media_paths(item)
            for key, item in value.items()
        }
        url = result.get("url")
        if isinstance(url, str) and url.startswith("file://"):
            parsed = urllib.parse.urlparse(url)
            path = urllib.request.url2pathname(parsed.path)
            if parsed.netloc:
                path = f"//{parsed.netloc}{path}"
            result["url"] = path
        return result

    def chat(
        self,
        model: str,
        messages: list[dict[str, Any]],
        max_tokens: int = 32,
        *,
        temperature: float = 0.0,
        seed: int | None = None,
        response_format: dict[str, Any] | None = None,
        logprobs: bool = False,
        top_logprobs: int | None = None,
        chat_template_kwargs: dict[str, Any] | None = None,
        sampling_params: dict[str, Any] | None = None,
        mm_processor_kwargs: dict[str, Any] | None = None,
        media_io_kwargs: dict[str, Any] | None = None,
        extra_body: dict[str, Any] | None = None,
    ) -> ChatResult:
        request_messages = messages
        if self.local_file_urls_as_paths:
            request_messages = self._local_media_paths(deepcopy(messages))
        payload = {
            "model": model,
            "messages": request_messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if seed is not None:
            payload["seed"] = seed
        if response_format is not None:
            payload["response_format"] = response_format
        if logprobs:
            payload["logprobs"] = True
            if top_logprobs is not None:
                payload["top_logprobs"] = top_logprobs
        if chat_template_kwargs is not None:
            payload["chat_template_kwargs"] = dict(chat_template_kwargs)
        if sampling_params:
            conflicts = sorted(set(payload).intersection(sampling_params))
            if conflicts:
                raise ValueError(
                    "sampling_params duplicates explicit request fields: "
                    + ", ".join(conflicts)
                )
            payload.update(sampling_params)
        if mm_processor_kwargs is not None:
            payload["mm_processor_kwargs"] = dict(mm_processor_kwargs)
        if media_io_kwargs is not None:
            payload["media_io_kwargs"] = dict(media_io_kwargs)
        if extra_body:
            payload.update(extra_body)
        request = urllib.request.Request(
            self.endpoint,
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
        )
        started = time.perf_counter()
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                raw = json.load(response)
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"HTTP {exc.code}: {body[:1000]}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"request failed: {exc.reason}") from exc
        latency = time.perf_counter() - started
        choices = raw.get("choices") or []
        if not choices:
            raise RuntimeError(f"API response has no choices: {raw}")
        choice = choices[0]
        message = choice.get("message") or {}
        reasoning = message.get("reasoning")
        if reasoning is None:
            # Older OpenAI-compatible servers used the non-standard
            # ``reasoning_content`` field.  vLLM exposes ``reasoning``.
            reasoning = message.get("reasoning_content")
        return ChatResult(
            content=_message_text(message.get("content")),
            usage=raw.get("usage") or {},
            raw=raw,
            latency_s=latency,
            reasoning_content=_message_text(reasoning),
            finish_reason=(
                str(choice["finish_reason"])
                if choice.get("finish_reason") is not None
                else None
            ),
        )
