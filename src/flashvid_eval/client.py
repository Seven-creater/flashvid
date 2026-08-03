from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ChatResult:
    content: str
    usage: dict[str, Any]
    raw: dict[str, Any]
    latency_s: float


class OpenAICompatibleClient:
    def __init__(self, base_url: str, api_key: str = "no", timeout: float = 900.0):
        self.endpoint = base_url.rstrip("/") + "/chat/completions"
        self.api_key = api_key
        self.timeout = timeout

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
        extra_body: dict[str, Any] | None = None,
    ) -> ChatResult:
        payload = {
            "model": model,
            "messages": messages,
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
        message = choices[0].get("message") or {}
        content = message.get("content", "")
        if isinstance(content, list):
            content = "".join(str(item.get("text", "")) for item in content if isinstance(item, dict))
        return ChatResult(str(content), raw.get("usage") or {}, raw, latency)
