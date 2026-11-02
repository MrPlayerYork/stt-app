"""Shared LLM backend types."""
from __future__ import annotations

from typing import Protocol

TIMEOUT_S = 60.0
TEMPERATURE = 0.2


class LLMError(Exception):
    """Any LLM backend failure, with a short human-readable message."""


class LLMBackend(Protocol):
    def complete(self, system_prompt: str, text: str) -> str: ...

    def list_models(self) -> list[str]: ...

    def unload(self) -> None: ...
