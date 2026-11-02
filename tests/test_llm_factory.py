"""Tests for stt_app.llm.make_backend and process (no network calls)."""
from __future__ import annotations

import pytest

import stt_app.llm as llm
from stt_app.config import LLMConfig
from stt_app.llm import LLMError, make_backend, process
from stt_app.llm.anthropic_backend import AnthropicBackend
from stt_app.llm.google_backend import GoogleBackend
from stt_app.llm.openai_compat import OpenAICompatBackend
from stt_app.llm.prompts import CLEANUP_PROMPT


def test_unknown_provider_raises():
    with pytest.raises(LLMError, match="Unknown LLM provider"):
        make_backend(LLMConfig(provider="nope"), "key")


@pytest.mark.parametrize("provider", ["openai", "anthropic", "google"])
def test_cloud_provider_without_key_raises(provider):
    with pytest.raises(LLMError, match="No API key"):
        make_backend(LLMConfig(provider=provider, api_key_env="SOME_ENV"), None)


@pytest.mark.parametrize("provider", ["openai", "anthropic", "google"])
def test_cloud_provider_with_empty_key_raises(provider):
    with pytest.raises(LLMError):
        make_backend(LLMConfig(provider=provider), "")


def test_openai_compat_requires_base_url():
    with pytest.raises(LLMError, match="base URL"):
        make_backend(LLMConfig(provider="openai_compat", base_url=""), None)


def test_openai_compat_works_without_key():
    backend = make_backend(
        LLMConfig(provider="openai_compat", model="m", base_url="http://localhost:1234/v1"), None
    )
    assert isinstance(backend, OpenAICompatBackend)
    assert backend.model == "m"
    assert backend.base_url == "http://localhost:1234/v1"


def test_openai_backend_with_dummy_key():
    backend = make_backend(LLMConfig(provider="openai", model="gpt-x"), "sk-dummy")
    assert isinstance(backend, OpenAICompatBackend)
    assert backend.model == "gpt-x"


def test_anthropic_backend_with_dummy_key():
    backend = make_backend(LLMConfig(provider="anthropic", model="claude-x"), "sk-ant-dummy")
    assert isinstance(backend, AnthropicBackend)
    assert backend.model == "claude-x"


def test_google_backend_with_dummy_key():
    backend = make_backend(LLMConfig(provider="google", model="gemini-x"), "dummy")
    assert isinstance(backend, GoogleBackend)
    assert backend.model == "gemini-x"


class _FakeBackend:
    def __init__(self, output: str):
        self.output = output
        self.calls: list[tuple[str, str]] = []

    def complete(self, system_prompt: str, text: str) -> str:
        self.calls.append((system_prompt, text))
        return self.output


def test_process_returns_stripped_output(monkeypatch):
    fake = _FakeBackend("  \n Hello there. \t\n")
    seen = {}

    def fake_make(cfg, api_key):
        seen["args"] = (cfg, api_key)
        return fake

    monkeypatch.setattr(llm, "make_backend", fake_make)
    cfg = LLMConfig(provider="openai", model="m")
    out = process(cfg, "k", "um hello there")
    assert out == "Hello there."
    assert seen["args"] == (cfg, "k")
    assert fake.calls == [(CLEANUP_PROMPT, "um hello there")]
