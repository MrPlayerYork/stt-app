"""Tests for stt_app.llm.prompts: system prompt selection and preset listing."""
from __future__ import annotations

import pytest

from stt_app.config import LLMConfig
from stt_app.llm.prompts import (
    CLEANUP_PROMPT,
    PRESETS,
    WORD_CHOICE_ADDENDUM,
    all_presets,
    system_prompt_for,
)


def test_cleanup_mode_uses_cleanup_prompt():
    cfg = LLMConfig(mode="cleanup", preset="markdown")
    assert system_prompt_for(cfg) == CLEANUP_PROMPT


@pytest.mark.parametrize("preset", list(PRESETS))
def test_each_builtin_preset_in_prompt_mode(preset):
    cfg = LLMConfig(mode="prompt", preset=preset)
    assert system_prompt_for(cfg) == PRESETS[preset][1]
    assert system_prompt_for(cfg) != CLEANUP_PROMPT


def test_custom_preset_overrides_prompt():
    cfg = LLMConfig(mode="prompt", preset="mine", custom_prompts={"mine": "Be a pirate."})
    assert system_prompt_for(cfg) == "Be a pirate."


def test_custom_preset_can_shadow_builtin_name():
    cfg = LLMConfig(mode="prompt", preset="concise", custom_prompts={"concise": "Custom concise."})
    assert system_prompt_for(cfg) == "Custom concise."


def test_unknown_preset_falls_back_to_deramble():
    cfg = LLMConfig(mode="prompt", preset="does-not-exist")
    assert system_prompt_for(cfg) == PRESETS["deramble"][1]


def test_all_presets_lists_builtins_and_customs():
    cfg = LLMConfig(custom_prompts={"mine": "x", "Concise": "y"})
    presets = all_presets(cfg)
    assert presets["deramble"] == "De-ramble"
    assert presets["concise"] == "Concise"
    assert presets["markdown"] == "Markdown"
    assert presets["mine"] == "mine"
    assert presets["Concise"] == "Concise"
    assert set(PRESETS) <= set(presets)


def test_all_presets_without_customs_is_builtins_only():
    assert set(all_presets(LLMConfig())) == set(PRESETS)


def test_word_choice_addendum_added_to_builtin_preset():
    cfg = LLMConfig(mode="prompt", preset="concise", word_choice=True)
    assert system_prompt_for(cfg) == PRESETS["concise"][1] + WORD_CHOICE_ADDENDUM


def test_word_choice_addendum_added_to_custom_prompt():
    cfg = LLMConfig(mode="prompt", preset="mine", custom_prompts={"mine": "Be a pirate."}, word_choice=True)
    assert system_prompt_for(cfg) == "Be a pirate." + WORD_CHOICE_ADDENDUM


def test_word_choice_addendum_absent_when_disabled():
    cfg = LLMConfig(mode="prompt", preset="deramble", word_choice=False)
    assert WORD_CHOICE_ADDENDUM not in system_prompt_for(cfg)
    assert system_prompt_for(cfg) == PRESETS["deramble"][1]


def test_word_choice_addendum_absent_in_cleanup_mode():
    cfg = LLMConfig(mode="cleanup", word_choice=True)
    assert system_prompt_for(cfg) == CLEANUP_PROMPT
    assert WORD_CHOICE_ADDENDUM not in system_prompt_for(cfg)
