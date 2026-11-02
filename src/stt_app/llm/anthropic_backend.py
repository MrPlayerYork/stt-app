"""Anthropic backend via the messages API."""
from __future__ import annotations

import anthropic

from .base import TEMPERATURE, TIMEOUT_S, LLMError, LLMResult


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


# Reasoning mapping. Claude 5.x models think adaptively by default and reject manual budgets (400); effort is
# set via output_config.effort. low/medium/high -> thinking {"type": "adaptive", "display": "summarized"} +
# output_config.effort=<level>. "off" -> thinking {"type": "disabled"} (Sonnet 5.5 only takes "between_tools";
# Opus 5.5 can't disable). Pre-5 models (Haiku 4.5, Sonnet/Opus 4.5) reject adaptive and need
# {"type": "enabled", "budget_tokens": N} (low 1024 / medium 4096 / high 8192, max_tokens raised above the
# budget). Rejected variants are skipped on a 400 that mentions thinking/effort. Newer models also 400 on any
# non-default temperature, and thinking forbids it on older ones, so temperature is dropped when thinking is
# configured or when the server complains about it.
BUDGETS = {"low": 1024, "medium": 4096, "high": 8192}
MAX_TOKENS = 4096


class AnthropicBackend:
    def __init__(self, model: str, api_key: str, reasoning: str = "default") -> None:
        self.model = model
        self.reasoning = reasoning or "default"
        self._client = anthropic.Anthropic(api_key=api_key, timeout=TIMEOUT_S, max_retries=1)

    def _variants(self) -> list[dict]:
        lvl = self.reasoning
        if lvl == "default":
            return [{}]
        if lvl == "off":
            return [{"thinking": {"type": "disabled"}}, {"thinking": {"type": "between_tools"}}, {}]
        return [
            {"thinking": {"type": "adaptive", "display": "summarized"}, "output_config": {"effort": lvl}},
            {"thinking": {"type": "enabled", "budget_tokens": BUDGETS[lvl], "display": "summarized"},
             "max_tokens": MAX_TOKENS + BUDGETS[lvl]},
            {},
        ]

    def complete(self, system_prompt: str, text: str) -> LLMResult:
        if not self.model:
            raise LLMError("No model selected")
        base = dict(
            model=self.model, max_tokens=MAX_TOKENS, system=system_prompt,
            messages=[{"role": "user", "content": text}],
        )
        variants = self._variants()
        note = ""
        try:
            for i, extra in enumerate(variants):
                kw = {**base, **extra}
                if "thinking" not in extra:
                    kw["temperature"] = TEMPERATURE
                try:
                    try:
                        r = self._client.messages.create(**kw)
                    except anthropic.BadRequestError as e:
                        if "temperature" not in kw or "temperature" not in str(e).lower():
                            raise
                        kw.pop("temperature")
                        r = self._client.messages.create(**kw)
                    break
                except anthropic.BadRequestError as e:
                    low = str(e).lower()
                    if i == len(variants) - 1 or not ("thinking" in low or "effort" in low or "budget" in low):
                        raise
                    if variants[i + 1] == {}:
                        note = "model rejected the reasoning setting"
        except Exception as e:
            raise _wrap(e) from e
        return self._parse(r, note)

    @staticmethod
    def _parse(r, note: str = "") -> LLMResult:
        answer, thoughts = [], []
        for b in r.content:
            if b.type == "text":
                answer.append(b.text)
            elif b.type == "thinking" and getattr(b, "thinking", ""):
                thoughts.append(b.thinking)
        has_block = any(b.type in ("thinking", "redacted_thinking") for b in r.content)
        details = getattr(getattr(r, "usage", None), "output_tokens_details", None)
        tokens = getattr(details, "thinking_tokens", None)
        reasoned = (tokens > 0) if tokens is not None else (True if has_block else None)
        return LLMResult("".join(answer), "\n\n".join(thoughts).strip(), tokens, reasoned, note)

    def list_models(self) -> list[str]:
        try:
            return sorted(m.id for m in self._client.models.list(limit=1000))
        except Exception as e:
            raise _wrap(e) from e

    def unload(self) -> None:
        """Cloud models have nothing to unload."""
