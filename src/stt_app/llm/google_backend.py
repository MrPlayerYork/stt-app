"""Google Gemini backend via google-genai."""
from __future__ import annotations

from google import genai
from google.genai import errors, types

from .base import TEMPERATURE, TIMEOUT_S, LLMError


def _wrap(e: Exception) -> LLMError:
    if isinstance(e, errors.APIError):
        code = e.code
        if code in (401, 403) or "API key" in str(e.message):
            return LLMError("Authentication failed: check the Google API key")
        if code == 404:
            return LLMError("Model not found: check the model name")
        if code == 429:
            return LLMError("Rate limited or quota exceeded")
        if code == 408 or code == 504:
            return LLMError("Request to Google timed out")
        return LLMError(f"Google API error {code}: {str(e.message)[:200]}")
    name = type(e).__name__
    if "Timeout" in name:
        return LLMError("Request to Google timed out")
    if "Connect" in name or "Network" in name:
        return LLMError("Cannot connect to Google")
    return LLMError(f"LLM request failed: {e}")


class GoogleBackend:
    def __init__(self, model: str, api_key: str) -> None:
        self.model = model
        self._client = genai.Client(
            api_key=api_key, http_options=types.HttpOptions(timeout=int(TIMEOUT_S * 1000))
        )

    def complete(self, system_prompt: str, text: str) -> str:
        if not self.model:
            raise LLMError("No model selected")
        try:
            r = self._client.models.generate_content(
                model=self.model,
                contents=text,
                config=types.GenerateContentConfig(
                    system_instruction=system_prompt, temperature=TEMPERATURE
                ),
            )
        except Exception as e:
            raise _wrap(e) from e
        return r.text or ""

    def list_models(self) -> list[str]:
        try:
            names = [
                (m.name or "").removeprefix("models/")
                for m in self._client.models.list()
                if not m.supported_actions or "generateContent" in m.supported_actions
            ]
        except Exception as e:
            raise _wrap(e) from e
        return sorted(n for n in names if n)
