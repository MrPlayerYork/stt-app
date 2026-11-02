"""LLM post-processing of transcripts."""
from __future__ import annotations

from typing import TYPE_CHECKING

from .base import REASONING_LEVELS, LLMBackend, LLMError, LLMResult
from .prompts import CLEANUP_PROMPT, PRESETS, all_presets, system_prompt_for

if TYPE_CHECKING:
    from stt_app.config import LLMConfig

PROVIDERS = {
    "openai": "OpenAI",
    "openai_compat": "Local / OpenAI-compatible (Ollama, LM Studio, vLLM, llama-server)",
    "anthropic": "Anthropic",
    "google": "Google Gemini",
}
LOCAL_PRESETS = {
    "Ollama": "http://localhost:11434/v1",
    "LM Studio": "http://localhost:1234/v1",
    "vLLM": "http://localhost:8000/v1",
    "llama-server": "http://localhost:8080/v1",
}
DEFAULT_MODELS = {
    "openai": "gpt-5.4-mini",
    "openai_compat": "",
    "anthropic": "claude-haiku-5-5",
    "google": "gemini-2.5-flash",
}


def make_backend(cfg: LLMConfig, api_key: str | None) -> LLMBackend:
    p = cfg.provider
    reasoning = getattr(cfg, "reasoning", "default")
    if p not in PROVIDERS:
        raise LLMError(f"Unknown LLM provider: {p!r}")
    if p != "openai_compat" and not api_key:
        raise LLMError(f"No API key set for {PROVIDERS[p]} (env var {cfg.api_key_env or '?'})")
    if p == "openai_compat":
        from .openai_compat import OpenAICompatBackend

        if not cfg.base_url:
            raise LLMError("No base URL set for the local server")
        return OpenAICompatBackend(cfg.model, api_key, cfg.base_url, cfg.idle_ttl_seconds, reasoning)
    if p == "openai":
        from .openai_compat import OpenAICompatBackend

        return OpenAICompatBackend(cfg.model, api_key, reasoning=reasoning)
    if p == "anthropic":
        from .anthropic_backend import AnthropicBackend

        return AnthropicBackend(cfg.model, api_key, reasoning)  # type: ignore[arg-type]
    from .google_backend import GoogleBackend

    return GoogleBackend(cfg.model, api_key, reasoning)  # type: ignore[arg-type]


def process(cfg: LLMConfig, api_key: str | None, text: str) -> LLMResult:
    r = make_backend(cfg, api_key).complete(system_prompt_for(cfg), text)
    r.text = r.text.strip()
    return r


__all__ = [
    "LLMResult", "REASONING_LEVELS", "PROVIDERS", "LOCAL_PRESETS", "DEFAULT_MODELS", "LLMBackend", "LLMError", "CLEANUP_PROMPT",
    "PRESETS", "all_presets", "system_prompt_for", "make_backend", "process",
]
