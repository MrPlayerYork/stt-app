"""Persistent settings for the STT app, stored as TOML with API keys kept in the keyring."""

import os
import tempfile
import tomllib
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

import keyring
import platformdirs
import tomli_w

SERVICE = "stt-app"


@dataclass
class AudioConfig:
    device_name: str | None = None  # chosen microphone (sounddevice device name)
    samplerate: int = 16000


@dataclass
class WhisperConfig:
    model: str = "large-v3-turbo"
    device: str = "auto"  # "auto" | "cuda" | "cpu"
    compute_type: str = "auto"
    language: str | None = "en"  # None = auto-detect


@dataclass
class LLMConfig:
    enabled: bool = False
    provider: str = "openai_compat"  # "openai" | "openai_compat" | "anthropic" | "google"
    model: str = ""
    base_url: str = ""  # used by openai_compat, e.g. http://localhost:1234/v1
    api_key_env: str = ""  # optional env var name holding the API key
    mode: str = "cleanup"  # "cleanup" | "prompt"
    preset: str = "deramble"  # Prompt Mode preset key
    custom_prompts: dict[str, str] = field(default_factory=dict)  # name -> system prompt


@dataclass
class AppConfig:
    audio: AudioConfig = field(default_factory=AudioConfig)
    whisper: WhisperConfig = field(default_factory=WhisperConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    history_size: int = 20


def config_path() -> Path:
    override = os.environ.get("STT_APP_CONFIG")
    if override:
        return Path(override)
    return platformdirs.user_config_path(SERVICE) / "config.toml"


def _section(cls, data: dict, nullable: tuple[str, ...] = ()):
    names = {f.name for f in fields(cls)}
    kwargs = {k: v for k, v in data.items() if k in names}
    for name in nullable:
        if kwargs.get(name) == "":
            kwargs[name] = None
    return cls(**kwargs)


def load_config() -> AppConfig | None:
    path = config_path()
    if not path.exists():
        return None
    with path.open("rb") as f:
        raw = tomllib.load(f)
    return AppConfig(
        audio=_section(AudioConfig, raw.get("audio", {}), nullable=("device_name",)),
        whisper=_section(WhisperConfig, raw.get("whisper", {}), nullable=("language",)),
        llm=_section(LLMConfig, raw.get("llm", {})),
        history_size=raw.get("history_size", 20),
    )


def save_config(cfg: AppConfig) -> None:
    data = asdict(cfg)
    for section in data.values():
        if isinstance(section, dict):
            for key, value in section.items():
                if value is None:
                    section[key] = ""  # TOML has no null; "" maps back to None on load
    path = config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".config-", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(tomli_w.dumps(data))
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def get_api_key(provider: str, env_name: str = "") -> str | None:
    try:
        key = keyring.get_password(SERVICE, provider)
    except Exception:
        key = None
    if key:
        return key
    if env_name:
        return os.environ.get(env_name) or None
    return None


def set_api_key(provider: str, key: str) -> None:
    keyring.set_password(SERVICE, provider, key)
