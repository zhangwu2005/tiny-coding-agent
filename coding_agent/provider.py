"""Minimal OpenAI-compatible HTTP client implemented with the standard library."""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Any


class ProviderError(RuntimeError):
    """A model request failed or returned an unusable response."""


class OpenAICompatibleClient:
    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        base_url: str = "https://api.openai.com/v1",
        timeout: float = 45.0,
    ) -> None:
        if not api_key:
            raise ValueError("API key is empty")
        if timeout <= 0:
            raise ValueError("timeout must be positive")
        self.api_key = api_key
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    @classmethod
    def from_environment(cls) -> "OpenAICompatibleClient":
        deepseek_key = os.environ.get("DEEPSEEK_API_KEY", "").strip()
        openai_key = os.environ.get("OPENAI_API_KEY", "").strip()
        # A provider-specific key is an explicit request to use that provider.
        use_deepseek_defaults = bool(deepseek_key)
        return cls(
            api_key=openai_key or deepseek_key,
            model=(
                os.environ.get("DEEPSEEK_MODEL", "")
                if use_deepseek_defaults
                else os.environ.get("OPENAI_MODEL", "")
            )
            or ("deepseek-v4-flash" if use_deepseek_defaults else "gpt-4o-mini"),
            base_url=(
                os.environ.get("DEEPSEEK_BASE_URL", "")
                if use_deepseek_defaults
                else os.environ.get("LLM_BASE_URL", "")
            )
            or ("https://api.deepseek.com" if use_deepseek_defaults else "https://api.openai.com/v1"),
            timeout=cls._timeout_from_environment(),
        )

    @staticmethod
    def _timeout_from_environment() -> float:
        raw_timeout = os.environ.get("LLM_TIMEOUT", "45").strip()
        try:
            timeout = float(raw_timeout)
        except ValueError as exc:
            raise ValueError("LLM_TIMEOUT must be a number") from exc
        if timeout <= 0:
            raise ValueError("LLM_TIMEOUT must be positive")
        return timeout

    def complete(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> dict[str, Any]:
        payload = {"model": self.model, "messages": messages, "tools": tools, "tool_choice": "auto"}
        request = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                result = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")[:2000]
            raise ProviderError(f"model HTTP {exc.code}: {body}") from exc
        except (urllib.error.URLError, TimeoutError, OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise ProviderError(f"model request failed: {exc}") from exc

        try:
            message = result["choices"][0]["message"]
        except (KeyError, IndexError, TypeError) as exc:
            raise ProviderError(f"unexpected model response: {result!r}") from exc
        if not isinstance(message, dict):
            raise ProviderError("model message is not an object")
        return message
