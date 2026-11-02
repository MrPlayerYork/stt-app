"""OpenAI and OpenAI-compatible (local server) backend via chat completions."""
from __future__ import annotations

import openai

from .base import TEMPERATURE, TIMEOUT_S, LLMError


def _wrap(e: Exception, base_url: str | None) -> LLMError:
    where = base_url or "OpenAI"
    if isinstance(e, openai.APITimeoutError):
        return LLMError(f"Request to {where} timed out")
    if isinstance(e, openai.APIConnectionError):
        return LLMError(f"Cannot connect to {where} (is the server running?)")
    if isinstance(e, openai.AuthenticationError):
        return LLMError("Authentication failed: check the API key")
    if isinstance(e, openai.PermissionDeniedError):
        return LLMError("Permission denied by the API")
    if isinstance(e, openai.NotFoundError):
        return LLMError("Model or endpoint not found: check the model name")
    if isinstance(e, openai.RateLimitError):
        return LLMError("Rate limited or quota exceeded")
    if isinstance(e, openai.APIStatusError):
        return LLMError(f"API error {e.status_code}: {str(e.message)[:200]}")
    return LLMError(f"LLM request failed: {e}")


class OpenAICompatBackend:
    def __init__(self, model: str, api_key: str | None, base_url: str | None = None) -> None:
        self.model = model
        self.base_url = base_url or None
        self._client = openai.OpenAI(
            api_key=api_key or "not-needed", base_url=self.base_url, timeout=TIMEOUT_S, max_retries=1
        )

    def complete(self, system_prompt: str, text: str) -> str:
        if not self.model:
            raise LLMError("No model selected")
        kwargs = dict(
            model=self.model,
            messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": text}],
        )
        try:
            try:
                r = self._client.chat.completions.create(temperature=TEMPERATURE, **kwargs)
            except openai.BadRequestError as e:
                # some models (e.g. reasoning ones) only accept the default temperature
                if "temperature" not in str(e).lower():
                    raise
                r = self._client.chat.completions.create(**kwargs)
        except Exception as e:
            raise _wrap(e, self.base_url) from e
        return (r.choices[0].message.content or "") if r.choices else ""

    def list_models(self) -> list[str]:
        try:
            return sorted(m.id for m in self._client.models.list())
        except Exception as e:
            raise _wrap(e, self.base_url) from e
