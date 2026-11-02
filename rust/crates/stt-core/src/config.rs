//! Persistent settings: `~/.config/stt-app/config.toml`, field-for-field identical to the Python app's
//! `src/stt_app/config.py`, so an existing file loads unchanged and the Python app can still read files
//! the Rust app writes.
//!
//! Compatibility rules (mirroring the Python loader):
//! - Every section and key is optional; missing ones take the defaults below (`#[serde(default)]`).
//! - Unknown keys and sections are ignored (and, like Python, dropped on the next save).
//! - TOML has no null: `audio.device_name` and `whisper.language` use `""` for "none" both ways.
//! - Enum-valued strings (`whisper.device`, `whisper.quantization`, `llm.provider`, `llm.mode`, `llm.reasoning`) accept unknown
//!   values and fall back to the default instead of failing the whole file.
//! - New Rust-only keys must be additive: add the field with `#[serde(default)]` and a default that keeps
//!   the Python behaviour.
//!
//! Load/save (atomic write via a temp file + rename in the same directory) are owned by P2-1; until then
//! [`load_config`] and [`save_config`] are `todo!()`. Parsing and serializing ([`AppConfig::from_toml_str`],
//! [`AppConfig::to_toml_string`]) and the path helpers are implemented.

use std::path::{Path, PathBuf};

use indexmap::IndexMap;
use serde::{Deserialize, Deserializer, Serialize, Serializer};

use crate::APP_NAME;
use crate::error::ConfigError;

/// Env var that overrides the config file path (same as Python; used by tests).
pub const CONFIG_ENV: &str = "STT_APP_CONFIG";

/// Defines a fieldless enum stored in TOML as a fixed lowercase string. Unknown strings deserialize to
/// the default variant (Python falls back the same way for `reasoning`, and treats any mode other than
/// "prompt" as cleanup).
macro_rules! string_enum {
    (
        $(#[$meta:meta])*
        pub enum $name:ident {
            $( $(#[$vmeta:meta])* $var:ident = $s:literal $(, alias $alias:literal)* ; )+
        }
        default = $def:ident;
    ) => {
        $(#[$meta])*
        #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
        pub enum $name {
            $( $(#[$vmeta])* $var, )+
        }

        impl $name {
            /// Every variant, in menu order.
            pub const ALL: &'static [Self] = &[$(Self::$var),+];

            /// The exact string stored in the config file.
            pub fn as_str(self) -> &'static str {
                match self { $(Self::$var => $s,)+ }
            }

            /// Parse a config string (trimmed, case-insensitive; aliases accepted). `None` if unknown.
            pub fn parse(s: &str) -> Option<Self> {
                let s = s.trim().to_ascii_lowercase();
                match s.as_str() {
                    $( $s $(| $alias)* => Some(Self::$var), )+
                    _ => None,
                }
            }
        }

        impl Default for $name {
            fn default() -> Self { Self::$def }
        }

        impl std::fmt::Display for $name {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                f.write_str(self.as_str())
            }
        }

        impl Serialize for $name {
            fn serialize<S: Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
                s.serialize_str(self.as_str())
            }
        }

        impl<'de> Deserialize<'de> for $name {
            fn deserialize<D: Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
                let s = String::deserialize(d)?;
                Ok(Self::parse(&s).unwrap_or_default())
            }
        }
    };
}

string_enum! {
    /// `whisper.device`: where Whisper runs.
    ///
    /// The Python values are `"auto" | "cuda" | "cpu"`. The Rust app runs on Vulkan, but writes the GPU
    /// choice as `"cuda"` so the Python app keeps working on the same file; `"gpu"` and `"vulkan"` are
    /// accepted on read.
    pub enum WhisperDevice {
        /// Try the GPU, fall back to CPU (default).
        Auto = "auto";
        /// GPU only (no fallback). Stored as "cuda" for Python compatibility.
        Gpu = "cuda", alias "gpu", alias "vulkan";
        /// CPU only.
        Cpu = "cpu";
    }
    default = Auto;
}

string_enum! {
    /// `whisper.quantization` (Rust-only, additive): which ggml weight variant of the model to load.
    /// The Python app ignores this key. Default `q5_0`: in the P1-b gate it gave the same text as f16 with
    /// about half the VRAM. Not every model has every variant on Hugging Face; stt-whisper picks the
    /// nearest available one (and says which in [`crate::LoadInfo`]).
    pub enum WhisperQuant {
        /// Full-precision f16 weights (largest, most VRAM).
        F16 = "f16";
        /// 8-bit quantized.
        Q8_0 = "q8_0";
        /// 5-bit quantized. Default.
        Q5_0 = "q5_0";
    }
    default = Q5_0;
}

string_enum! {
    /// `llm.provider`. An unknown provider string loads as the default (`openai_compat`); the Python app
    /// kept the string and failed when building the backend.
    pub enum Provider {
        /// OpenAI (cloud, needs an API key).
        OpenAi = "openai";
        /// Any OpenAI-compatible server: LM Studio, Ollama, vLLM, llama-server (API key optional). Default.
        OpenAiCompat = "openai_compat";
        /// Anthropic (cloud, needs an API key).
        Anthropic = "anthropic";
        /// Google Gemini (cloud, needs an API key).
        Google = "google";
    }
    default = OpenAiCompat;
}

string_enum! {
    /// `llm.mode`: what the LLM does with the transcript. Any value other than "prompt" means cleanup
    /// (as in Python's `system_prompt_for`).
    pub enum LlmMode {
        /// Word-for-word cleanup (`CLEANUP_PROMPT`). Default.
        Cleanup = "cleanup";
        /// Prompt Mode: rewrite with a preset or custom prompt (`llm.preset`).
        Prompt = "prompt";
    }
    default = Cleanup;
}

string_enum! {
    /// `llm.reasoning`: requested reasoning/thinking effort. Unknown values load as `Default`
    /// (Python: `if llm.reasoning not in REASONING_LEVELS: llm.reasoning = "default"`).
    pub enum ReasoningLevel {
        /// Send nothing; the provider/model default applies. Default.
        Default = "default";
        /// Ask for no reasoning (OpenAI-compatible "none"/"minimal", thinking disabled, ...).
        Off = "off";
        /// Low effort.
        Low = "low";
        /// Medium effort.
        Medium = "medium";
        /// High effort.
        High = "high";
    }
    default = Default;
}

impl Provider {
    /// Human label shown in menus (Python `PROVIDERS`).
    pub fn label(self) -> &'static str {
        match self {
            Self::OpenAi => "OpenAI",
            Self::OpenAiCompat => {
                "Local / OpenAI-compatible (Ollama, LM Studio, vLLM, llama-server)"
            }
            Self::Anthropic => "Anthropic",
            Self::Google => "Google Gemini",
        }
    }

    /// Model suggested when the user switches to this provider (Python `DEFAULT_MODELS`; `""` = pick from
    /// the server's list).
    pub fn default_model(self) -> &'static str {
        match self {
            Self::OpenAi => "gpt-5.4-mini",
            Self::OpenAiCompat => "",
            Self::Anthropic => "claude-haiku-5-5",
            Self::Google => "gemini-2.5-flash",
        }
    }

    /// Cloud providers need an API key; `openai_compat` works without one.
    pub fn needs_api_key(self) -> bool {
        !matches!(self, Self::OpenAiCompat)
    }

    /// Keyring username for this provider's API key (the config string, as in Python).
    pub fn keyring_user(self) -> &'static str {
        self.as_str()
    }
}

/// Base-URL presets for `openai_compat` (Python `LOCAL_PRESETS`), in menu order.
pub const LOCAL_PRESETS: &[(&str, &str)] = &[
    ("Ollama", "http://localhost:11434/v1"),
    ("LM Studio", "http://localhost:1234/v1"),
    ("vLLM", "http://localhost:8000/v1"),
    ("llama-server", "http://localhost:8080/v1"),
];

/// Serde helper: `Option<String>` stored as a TOML string, with `""` meaning `None` (both directions).
/// Exactly `""` maps to `None`, like Python's `_section(..., nullable=...)`.
mod empty_as_none {
    use serde::{Deserialize, Deserializer, Serializer};

    pub fn serialize<S: Serializer>(v: &Option<String>, s: S) -> Result<S::Ok, S::Error> {
        s.serialize_str(v.as_deref().unwrap_or(""))
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(d: D) -> Result<Option<String>, D::Error> {
        let s = String::deserialize(d)?;
        Ok(if s.is_empty() { None } else { Some(s) })
    }
}

/// The whole config file. Top level holds `history_size` plus the `[audio]`, `[whisper]` and `[llm]`
/// sections.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct AppConfig {
    /// TOML key `history_size`: how many dictations the history keeps (default 20; the UI uses at least 1).
    pub history_size: usize,
    /// `[audio]` section.
    pub audio: AudioConfig,
    /// `[whisper]` section.
    pub whisper: WhisperConfig,
    /// `[llm]` section.
    pub llm: LlmConfig,
}

impl Default for AppConfig {
    fn default() -> Self {
        Self {
            history_size: 20,
            audio: AudioConfig::default(),
            whisper: WhisperConfig::default(),
            llm: LlmConfig::default(),
        }
    }
}

/// `[audio]` section.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct AudioConfig {
    /// Chosen microphone by device name; `None` (stored as `""`) = not chosen yet, which triggers
    /// calibration on start. Matched with [`crate::audio::find_device`] (exact, then case-insensitive
    /// substring). Note: the Python app stored PortAudio/JACK names; cpal names may differ, in which case
    /// the lookup fails and the user re-picks.
    #[serde(with = "empty_as_none")]
    pub device_name: Option<String>,
    /// Kept for compatibility (default 16000). Recording always resamples to [`crate::SAMPLE_RATE`].
    pub samplerate: u32,
}

impl Default for AudioConfig {
    fn default() -> Self {
        Self {
            device_name: None,
            samplerate: 16_000,
        }
    }
}

/// `[whisper]` section.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct WhisperConfig {
    /// Model name, one of [`crate::transcribe::WHISPER_MODELS`] (default "large-v3-turbo"). stt-whisper
    /// maps it to a ggml file.
    pub model: String,
    /// Device choice (default auto).
    pub device: WhisperDevice,
    /// Python/CTranslate2 compute type (default "auto"). Unused by the Rust app; kept so it round-trips.
    pub compute_type: String,
    /// Language code; `None` (stored as `""`) = auto-detect. Default `Some("en")`.
    #[serde(with = "empty_as_none")]
    pub language: Option<String>,
    /// Models directory; `""` = `<user data dir>/stt-app/models`. See [`resolved_models_dir`].
    pub models_dir: String,
    /// Parallel connections per downloaded file (default 8; the UI allows 1..=32).
    pub download_connections: u32,
    /// ggml weight variant (Rust-only key, default q5_0). See [`WhisperQuant`].
    pub quantization: WhisperQuant,
}

impl Default for WhisperConfig {
    fn default() -> Self {
        Self {
            model: "large-v3-turbo".into(),
            device: WhisperDevice::Auto,
            compute_type: "auto".into(),
            language: Some("en".into()),
            models_dir: String::new(),
            download_connections: 8,
            quantization: WhisperQuant::Q5_0,
        }
    }
}

/// `[llm]` section.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct LlmConfig {
    /// Post-process transcripts with an LLM (default false = Raw mode).
    pub enabled: bool,
    /// Provider (default `openai_compat`).
    pub provider: Provider,
    /// Model id (default `""`).
    pub model: String,
    /// Base URL for `openai_compat`, e.g. `http://localhost:1234/v1` (default `""`).
    pub base_url: String,
    /// Optional env var holding the API key, used when the keyring has none (default `""`).
    pub api_key_env: String,
    /// Cleanup or Prompt Mode (default cleanup).
    pub mode: LlmMode,
    /// Prompt Mode preset key: a built-in preset ("deramble", "concise", "markdown") or a key of
    /// `custom_prompts` (default "deramble").
    pub preset: String,
    /// TOML table `[llm.custom_prompts]`: name → system prompt. Insertion order is preserved (menus list
    /// prompts in this order, like Python's dict).
    pub custom_prompts: IndexMap<String, String>,
    /// Local servers: unload the model after this many idle seconds; 0 = server default (default 120).
    /// LM Studio gets `ttl`, Ollama gets `keep_alive`.
    pub idle_ttl_seconds: u64,
    /// Local servers: ask the server to unload the model when the app quits, if the LLM was used this
    /// session (default false).
    pub unload_on_exit: bool,
    /// Reasoning effort (default `default`).
    pub reasoning: ReasoningLevel,
    /// Prompt Mode only: also replace words that clearly don't fit the context (default false).
    pub word_choice: bool,
}

impl Default for LlmConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            provider: Provider::OpenAiCompat,
            model: String::new(),
            base_url: String::new(),
            api_key_env: String::new(),
            mode: LlmMode::Cleanup,
            preset: "deramble".into(),
            custom_prompts: IndexMap::new(),
            idle_ttl_seconds: 120,
            unload_on_exit: false,
            reasoning: ReasoningLevel::Default,
            word_choice: false,
        }
    }
}

impl AppConfig {
    /// Parse a config file's contents (Python-written files included). Missing keys take defaults,
    /// unknown keys are ignored.
    pub fn from_toml_str(s: &str) -> Result<Self, ConfigError> {
        Ok(toml::from_str(s)?)
    }

    /// Serialize to TOML in the same shape the Python app writes (every key present, `None` as `""`).
    pub fn to_toml_string(&self) -> Result<String, ConfigError> {
        Ok(toml::to_string(self)?)
    }
}

/// Path of the config file: `$STT_APP_CONFIG` if set and non-empty, else
/// `<XDG config dir>/stt-app/config.toml` (normally `~/.config/stt-app/config.toml`).
pub fn config_path() -> Result<PathBuf, ConfigError> {
    if let Some(p) = std::env::var_os(CONFIG_ENV).filter(|p| !p.is_empty()) {
        return Ok(PathBuf::from(p));
    }
    Ok(dirs::config_dir()
        .ok_or(ConfigError::NoConfigDir)?
        .join(APP_NAME)
        .join("config.toml"))
}

/// Default models directory: `<XDG data dir>/stt-app/models` (normally `~/.local/share/stt-app/models`).
pub fn default_models_dir() -> Result<PathBuf, ConfigError> {
    Ok(dirs::data_dir()
        .ok_or(ConfigError::NoConfigDir)?
        .join(APP_NAME)
        .join("models"))
}

/// The models directory to use: `whisper.models_dir` (trimmed, `~` expanded) or [`default_models_dir`].
pub fn resolved_models_dir(cfg: &AppConfig) -> Result<PathBuf, ConfigError> {
    let custom = cfg.whisper.models_dir.trim();
    if custom.is_empty() {
        return default_models_dir();
    }
    Ok(expand_tilde(custom))
}

/// Expand a leading `~` or `~/` to the home directory (like Python's `Path.expanduser` for the current
/// user). Other paths are returned unchanged.
fn expand_tilde(p: &str) -> PathBuf {
    if let Some(home) = dirs::home_dir() {
        if p == "~" {
            return home;
        }
        if let Some(rest) = p.strip_prefix("~/") {
            return home.join(rest);
        }
    }
    PathBuf::from(p)
}

/// Load the config from [`config_path`]. `Ok(None)` when the file does not exist (first run).
///
/// **Stub, owned by P2-1.**
pub fn load_config() -> Result<Option<AppConfig>, ConfigError> {
    load_config_from(&config_path()?)
}

/// Load the config from `path`. `Ok(None)` when the file does not exist.
///
/// **Stub, owned by P2-1** (read the file, [`AppConfig::from_toml_str`], map I/O errors to
/// [`ConfigError::Io`]).
pub fn load_config_from(path: &Path) -> Result<Option<AppConfig>, ConfigError> {
    let _ = path;
    todo!("P2-1: load_config_from")
}

/// Save the config to [`config_path`], creating parent directories.
///
/// **Stub, owned by P2-1.**
pub fn save_config(cfg: &AppConfig) -> Result<(), ConfigError> {
    save_config_to(cfg, &config_path()?)
}

/// Save the config to `path` atomically: write a temp file in the same directory
/// (`.config-*.tmp`), then rename over `path`; remove the temp file on failure.
///
/// **Stub, owned by P2-1.**
pub fn save_config_to(cfg: &AppConfig, path: &Path) -> Result<(), ConfigError> {
    let _ = (cfg, path);
    todo!("P2-1: save_config_to (atomic write)")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tilde_expansion() {
        let home = dirs::home_dir().unwrap();
        assert_eq!(expand_tilde("~/m"), home.join("m"));
        assert_eq!(expand_tilde("/abs/m"), PathBuf::from("/abs/m"));
        assert_eq!(expand_tilde("rel/~"), PathBuf::from("rel/~"));
    }

    #[test]
    fn custom_models_dir_is_trimmed() {
        let mut cfg = AppConfig::default();
        cfg.whisper.models_dir = "  /data/models ".into();
        assert_eq!(
            resolved_models_dir(&cfg).unwrap(),
            PathBuf::from("/data/models")
        );
    }

    #[test]
    fn enum_parse_is_lenient() {
        assert_eq!(WhisperDevice::parse("GPU"), Some(WhisperDevice::Gpu));
        assert_eq!(WhisperDevice::parse("cuda"), Some(WhisperDevice::Gpu));
        assert_eq!(WhisperDevice::Gpu.as_str(), "cuda");
        assert_eq!(ReasoningLevel::parse("extreme"), None);
        assert_eq!(Provider::ALL.len(), 4);
    }
}
