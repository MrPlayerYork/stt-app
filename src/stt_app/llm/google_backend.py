"""Google Gemini backend via google-genai."""
from __future__ import annotations

from google import genai
from google.genai import errors, types

from .base import TEMPERATURE, TIMEOUT_S, LLMError, LLMResult


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
    def __init__(self, model: str, api_key: str, reasoning: str = "default") -> None:
        self.model = model
        self.reasoning = reasoning or "default"
        self._client = genai.Client(
            api_key=api_key, http_options=types.HttpOptions(timeout=int(TIMEOUT_S * 1000))
        )

    def _thinking(self) -> types.ThinkingConfig | None:
        # Mapping: Gemini 3 takes thinking_level (low/medium/high; "off" -> MINIMAL). Gemini 2.5 takes
        # thinking_budget tokens (low 1024 / medium 8192 / high 24576; "off" -> 0, or 128 on 2.5 Pro, which
        # cannot disable thinking). include_thoughts=True so thought summaries come back.
        lvl = self.reasoning
        if lvl == "default":
            return None
        if "gemini-3" in self.model:
            level = types.ThinkingLevel.MINIMAL if lvl == "off" else types.ThinkingLevel(lvl.upper())
            return types.ThinkingConfig(thinking_level=level, include_thoughts=lvl != "off")
        budget = {"off": 128 if "pro" in self.model else 0, "low": 1024, "medium": 8192, "high": 24576}[lvl]
        return types.ThinkingConfig(thinking_budget=budget, include_thoughts=lvl != "off")

    def _generate(self, system_prompt: str, text: str, thinking):
        cfg = types.GenerateContentConfig(
            system_instruction=system_prompt, temperature=TEMPERATURE, thinking_config=thinking
        )
        return self._client.models.generate_content(model=self.model, contents=text, config=cfg)

    def complete(self, system_prompt: str, text: str) -> LLMResult:
        if not self.model:
            raise LLMError("No model selected")
        thinking = self._thinking()
        note = ""
        try:
            try:
                r = self._generate(system_prompt, text, thinking)
            except errors.APIError as e:
                if thinking is None or e.code != 400 or "think" not in str(e.message).lower():
                    raise
                note = "model rejected the reasoning setting"
                r = self._generate(system_prompt, text, None)
        except Exception as e:
            raise _wrap(e) from e
        return self._parse(r, note)

    @staticmethod
    def _parse(r, note: str = "") -> LLMResult:
        answer, thoughts = [], []
        cands = getattr(r, "candidates", None) or []
        parts = (cands[0].content.parts or []) if cands and cands[0].content else []
        for p in parts:
            if p.text:
                (thoughts if p.thought else answer).append(p.text)
        usage = getattr(r, "usage_metadata", None)
        tokens = getattr(usage, "thoughts_token_count", None)
        if tokens is not None:
            reasoned = tokens > 0
        else:
            reasoned = True if thoughts else (False if usage is not None else None)
        return LLMResult("".join(answer), "\n\n".join(t.strip() for t in thoughts), tokens, reasoned, note)

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

    def unload(self) -> None:
        """Cloud models have nothing to unload."""
