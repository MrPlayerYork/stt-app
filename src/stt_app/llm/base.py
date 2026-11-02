"""Shared LLM backend types."""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Protocol

TIMEOUT_S = 60.0
TEMPERATURE = 0.2
REASONING_LEVELS = ("default", "off", "low", "medium", "high")


class LLMError(Exception):
    """Any LLM backend failure, with a short human-readable message."""


@dataclass
class LLMResult:
    text: str  # final answer only; never contains reasoning/<think> text
    reasoning: str = ""  # reasoning/thinking text if the provider exposed it
    reasoning_tokens: int | None = None
    reasoned: bool | None = None  # True/False when known, None when the provider can't tell
    note: str = ""  # short human note, e.g. "model has no reasoning control"


class LLMBackend(Protocol):
    def complete(self, system_prompt: str, text: str) -> LLMResult: ...

    def list_models(self) -> list[str]: ...

    def unload(self) -> None: ...


_THINK_CLOSED = re.compile(r"<think(?:ing)?>(.*?)</think(?:ing)?>", re.S | re.I)
_THINK_OPEN = re.compile(r"^\s*<think(?:ing)?>(.*)$", re.S | re.I)
_THINK_STRAY_CLOSE = re.compile(r"^(.*?)</think(?:ing)?>", re.S | re.I)


def split_think(content: str) -> tuple[str, str]:
    """Split inline <think>…</think> blocks out of content -> (answer, reasoning).

    Handles closed blocks, an unterminated leading <think> (everything after it is reasoning), and a
    lone </think> with no opening tag (servers that strip the opening tag).
    """
    reasoning: list[str] = []

    def grab(m: re.Match) -> str:
        reasoning.append(m.group(1).strip())
        return ""

    out = _THINK_CLOSED.sub(grab, content)
    m = _THINK_OPEN.match(out)
    if m:
        reasoning.append(m.group(1).strip())
        out = ""
    else:
        m = _THINK_STRAY_CLOSE.match(out)
        if m and "<think" not in m.group(1).lower():
            reasoning.append(m.group(1).strip())
            out = out[m.end():]
    return out.strip(), "\n\n".join(r for r in reasoning if r)
