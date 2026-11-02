"""Tests for stt_app.config: TOML persistence, defaults and API key lookup."""
from __future__ import annotations

import keyring
import pytest

from stt_app import config as cfgmod
from stt_app.config import (
    AppConfig,
    AudioConfig,
    LLMConfig,
    WhisperConfig,
    get_api_key,
    load_config,
    save_config,
)


@pytest.fixture
def cfg_file(tmp_path, monkeypatch):
    path = tmp_path / "config.toml"
    monkeypatch.setenv("STT_APP_CONFIG", str(path))
    return path


def test_config_path_honours_env_override(cfg_file):
    assert cfgmod.config_path() == cfg_file


def test_load_returns_none_when_missing(cfg_file):
    assert not cfg_file.exists()
    assert load_config() is None


def test_round_trip_defaults(cfg_file):
    save_config(AppConfig())
    assert load_config() == AppConfig()


def test_round_trip_null_fields_and_custom_prompts(cfg_file):
    original = AppConfig(
        audio=AudioConfig(device_name=None, samplerate=48000),
        whisper=WhisperConfig(language=None, model="small"),
        llm=LLMConfig(
            enabled=True,
            provider="anthropic",
            model="claude-haiku-5-5",
            mode="prompt",
            preset="mine",
            custom_prompts={"mine": "Translate to French.", "other": "Be terse."},
        ),
        history_size=5,
    )
    save_config(original)
    loaded = load_config()
    assert loaded == original
    assert loaded.audio.device_name is None
    assert loaded.whisper.language is None
    assert loaded.llm.custom_prompts == {"mine": "Translate to French.", "other": "Be terse."}


def test_round_trip_preserves_set_device_name(cfg_file):
    save_config(AppConfig(audio=AudioConfig(device_name="USB Mic")))
    assert load_config().audio.device_name == "USB Mic"


def test_unknown_keys_are_ignored(cfg_file):
    cfg_file.write_text(
        'history_size = 7\nmystery = 1\n'
        '[audio]\nsamplerate = 16000\nbogus = "x"\n'
        '[llm]\nenabled = true\nfuture_flag = false\n',
        encoding="utf-8",
    )
    loaded = load_config()
    assert loaded.history_size == 7
    assert loaded.audio == AudioConfig()
    assert loaded.llm.enabled is True
    assert not hasattr(loaded.llm, "future_flag")


def test_missing_keys_fall_back_to_defaults(cfg_file):
    cfg_file.write_text('[whisper]\nmodel = "tiny"\n', encoding="utf-8")
    loaded = load_config()
    assert loaded.whisper.model == "tiny"
    assert loaded.whisper.language == WhisperConfig().language
    assert loaded.audio == AudioConfig()
    assert loaded.llm == LLMConfig()
    assert loaded.history_size == 20


def test_empty_file_gives_all_defaults(cfg_file):
    cfg_file.write_text("", encoding="utf-8")
    assert load_config() == AppConfig()


def test_get_api_key_prefers_keyring(monkeypatch):
    monkeypatch.setattr(keyring, "get_password", lambda svc, user: "from-keyring")
    monkeypatch.setenv("TEST_KEY_ENV", "from-env")
    assert get_api_key("anthropic", "TEST_KEY_ENV") == "from-keyring"


@pytest.mark.parametrize("keyring_result", [None, ""])
def test_get_api_key_falls_back_to_env_when_keyring_empty(monkeypatch, keyring_result):
    monkeypatch.setattr(keyring, "get_password", lambda svc, user: keyring_result)
    monkeypatch.setenv("TEST_KEY_ENV", "from-env")
    assert get_api_key("openai", "TEST_KEY_ENV") == "from-env"


def test_get_api_key_falls_back_to_env_when_keyring_raises(monkeypatch):
    def boom(svc, user):
        raise keyring.errors.KeyringError("no backend")

    monkeypatch.setattr(keyring, "get_password", boom)
    monkeypatch.setenv("TEST_KEY_ENV", "from-env")
    assert get_api_key("openai", "TEST_KEY_ENV") == "from-env"


def test_get_api_key_none_when_nothing_available(monkeypatch):
    monkeypatch.setattr(keyring, "get_password", lambda svc, user: None)
    monkeypatch.delenv("TEST_KEY_ENV_UNSET", raising=False)
    assert get_api_key("openai", "TEST_KEY_ENV_UNSET") is None
    assert get_api_key("openai") is None


def test_get_api_key_empty_env_value_is_none(monkeypatch):
    monkeypatch.setattr(keyring, "get_password", lambda svc, user: None)
    monkeypatch.setenv("TEST_KEY_ENV", "")
    assert get_api_key("openai", "TEST_KEY_ENV") is None


def test_api_key_never_written_to_toml(cfg_file, monkeypatch):
    secret = "sk-SUPER-SECRET-123"
    monkeypatch.setattr(keyring, "get_password", lambda svc, user: secret)
    monkeypatch.setenv("STT_TEST_KEY", secret)
    assert get_api_key("openai", "STT_TEST_KEY") == secret
    save_config(AppConfig(llm=LLMConfig(enabled=True, provider="openai", api_key_env="STT_TEST_KEY")))
    text = cfg_file.read_text(encoding="utf-8")
    assert secret not in text
    assert "api_key =" not in text
    assert "STT_TEST_KEY" in text  # only the env var name is persisted


def test_new_whisper_fields_default_for_old_config(cfg_file):
    cfg_file.write_text('[whisper]\nmodel = "small"\n')
    cfg = load_config()
    assert cfg.whisper.model == "small"
    assert cfg.whisper.models_dir == "" and cfg.whisper.download_connections == 8


def test_resolved_models_dir(tmp_path):
    cfg = AppConfig()
    assert cfgmod.resolved_models_dir(cfg).name == "models"
    cfg.whisper.models_dir = str(tmp_path / "m")
    assert cfgmod.resolved_models_dir(cfg) == tmp_path / "m"


def test_llm_ttl_defaults_and_old_config_compat(cfg_file):
    assert LLMConfig().idle_ttl_seconds == 120
    assert LLMConfig().unload_on_exit is False
    cfg_file.write_text('[llm]\nenabled = true\nmodel = "m"\n', encoding="utf-8")
    loaded = load_config()
    assert (loaded.llm.idle_ttl_seconds, loaded.llm.unload_on_exit) == (120, False)


def test_llm_ttl_round_trip(cfg_file):
    save_config(AppConfig(llm=LLMConfig(idle_ttl_seconds=0, unload_on_exit=True)))
    loaded = load_config()
    assert loaded.llm.idle_ttl_seconds == 0 and loaded.llm.unload_on_exit is True


def test_reasoning_and_word_choice_round_trip(cfg_file):
    save_config(AppConfig(llm=LLMConfig(reasoning="high", word_choice=True)))
    loaded = load_config()
    assert loaded.llm.reasoning == "high" and loaded.llm.word_choice is True


def test_reasoning_and_word_choice_defaults_for_old_config(cfg_file):
    cfg_file.write_text('[llm]\nenabled = true\nmode = "prompt"\n', encoding="utf-8")
    loaded = load_config()
    assert (loaded.llm.reasoning, loaded.llm.word_choice) == ("default", False)


@pytest.mark.parametrize("value", ["extreme", "HIGH", "", "none"])
def test_invalid_reasoning_value_falls_back_to_default(cfg_file, value):
    cfg_file.write_text(f'[llm]\nreasoning = "{value}"\n', encoding="utf-8")
    assert load_config().llm.reasoning == "default"
