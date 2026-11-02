"""Anthropic backend via the messages API."""
from __future__ import annotations

import anthropic

from .base import TEMPERATURE, TIMEOUT_S, LLMError


def _wrap(e: Exception) -> LLMError:
    if isinstance(e, anthropic.APITimeoutError):
        return LLMError("Request to Anthropic timed out")
    if isinstance(e, anthropic.APIConnectionError):
        return LLMError("Cannot connect to Anthropic")
    if isinstance(e, anthropic.AuthenticationError):
        return LLMError("Authentication failed: check the Anthropic API key")
    if isinstance(e, anthropic.PermissionDeniedError):
        return LLMError("Permission denied by the Anthropic API")
    if isinstance(e, anthropic.NotFoundError):
        return LLMError("Model not found: check the model name")
    if isinstance(e, anthropic.RateLimitError):
        return LLMError("Rate limited or quota exceeded")
    if isinstance(e, anthropic.APIStatusError):
        return LLMError(f"Anthropic API error {e.status_code}: {str(e.message)[:200]}")
    return LLMError(f"LLM request failed: {e}")


class AnthropicBackend:
    def __init__(self, model: str, api_key: str) -> None:
        self.model = model
        self._client = anthropic.Anthropic(api_key=api_key, timeout=TIMEOUT_S, max_retries=1)

    def complete(self, system_prompt: str, text: str) -> str:
        if not self.model:
            raise LLMError("No model selected")
        try:
            r = self._client.messages.create(
                model=self.model,
                max_tokens=4096,
                temperature=TEMPERATURE,
                system=system_prompt,
                messages=[{"role": "user", "content": text}],
            )
        except Exception as e:
            raise _wrap(e) from e
        return "".join(b.text for b in r.content if b.type == "text")

    def list_models(self) -> list[str]:
        try:
            return sorted(m.id for m in self._client.models.list(limit=1000))
        except Exception as e:
            raise _wrap(e) from e
