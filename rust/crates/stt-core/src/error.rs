//! Error types for every library crate (`thiserror`). The binary wraps them in `anyhow`.
//!
//! `Display` strings are shown to the user as-is (like the Python app's exception messages), so keep them
//! short, human-readable and free of debug formatting.

use std::path::PathBuf;

use thiserror::Error;

/// Reading, parsing or writing the config file.
#[derive(Debug, Error)]
pub enum ConfigError {
    /// The file could not be read or written.
    #[error("{}: {source}", path.display())]
    Io {
        /// File that failed.
        path: PathBuf,
        /// Underlying I/O error.
        #[source]
        source: std::io::Error,
    },
    /// The file is not valid TOML or has a value of the wrong type.
    #[error("invalid config: {0}")]
    Parse(#[from] toml::de::Error),
    /// The config could not be serialized (should not happen with the schema in [`crate::config`]).
    #[error("could not serialize config: {0}")]
    Serialize(#[from] toml::ser::Error),
    /// No platform config/data directory could be determined (no `$HOME`).
    #[error("could not determine the user config directory")]
    NoConfigDir,
}

/// Keyring access failed. A missing entry is *not* an error: lookups return `Ok(None)`.
#[derive(Debug, Error)]
pub enum SecretError {
    /// No keyring backend is available (no Secret Service running, locked collection, ...).
    #[error("keyring unavailable: {0}")]
    Unavailable(String),
    /// The backend answered but the operation failed.
    #[error("keyring error: {0}")]
    Backend(String),
}

/// Audio device or capture failure.
#[derive(Debug, Error)]
pub enum AudioError {
    /// No input devices exist.
    #[error("no input devices found")]
    NoDevices,
    /// The requested device is not (or no longer) present.
    #[error("input device not found: {0}")]
    DeviceNotFound(String),
    /// The device exists but a stream could not be opened on it.
    #[error("could not open {device}: {message}")]
    Open {
        /// Device name.
        device: String,
        /// Backend message.
        message: String,
    },
    /// The stream failed while running.
    #[error("audio stream error: {0}")]
    Stream(String),
    /// Any other backend (cpal / host) failure.
    #[error("audio backend error: {0}")]
    Backend(String),
}

/// Model download failure (Hugging Face). Downloads resume on the next attempt.
#[derive(Debug, Error)]
pub enum DownloadError {
    /// An HTTP status that retrying will not fix (401/403 gated repo or bad token, 404).
    #[error("HTTP {status} for {url}: {message}")]
    Http {
        /// HTTP status code.
        status: u16,
        /// Requested URL (never includes the token).
        url: String,
        /// Human hint, e.g. "the repo may be gated, add a Hugging Face token".
        message: String,
    },
    /// Network failure after all retries.
    #[error("{0}")]
    Network(String),
    /// Local file I/O failed.
    #[error("{}: {source}", path.display())]
    Io {
        /// File that failed.
        path: PathBuf,
        /// Underlying I/O error.
        #[source]
        source: std::io::Error,
    },
    /// The model name is not known.
    #[error("unknown model '{0}'")]
    UnknownModel(String),
    /// Download stopped early (cancelled or connection loss); it will resume next time.
    #[error("download of {0} was interrupted (it will resume next time)")]
    Interrupted(String),
    /// Files are missing or malformed after the download finished.
    #[error("{0}")]
    Incomplete(String),
}

/// Whisper model fetch/load/transcription failure.
#[derive(Debug, Error)]
pub enum TranscribeError {
    /// The model could not be downloaded.
    #[error("could not download model '{model}': {source}")]
    Download {
        /// Model name from the config.
        model: String,
        /// Underlying download error.
        #[source]
        source: DownloadError,
    },
    /// The model could not be loaded on any device. `message` lists every attempt (GPU, then CPU).
    #[error("could not load model '{model}': {message}")]
    Load {
        /// Model name from the config.
        model: String,
        /// One line per attempt, joined with " | " (like the Python app).
        message: String,
    },
    /// Inference failed.
    #[error("transcription failed: {0}")]
    Transcribe(String),
}

/// Any LLM backend failure. `Display` is a short message shown in the UI; the app then copies the raw
/// transcript instead (parity: "LLM failure → copy raw text").
#[derive(Debug, Error)]
pub enum LlmError {
    /// Provider key is not one of the four known providers.
    #[error("unknown LLM provider: {0:?}")]
    UnknownProvider(String),
    /// A cloud provider has no API key (keyring or env var).
    #[error("no API key set for {provider} (env var {env})")]
    NoApiKey {
        /// Provider label, e.g. "Anthropic".
        provider: String,
        /// Configured env var name, or "?" when none.
        env: String,
    },
    /// `openai_compat` has no base URL.
    #[error("no base URL set for the local server")]
    NoBaseUrl,
    /// The server could not be reached (connection refused, DNS, TLS).
    #[error("could not connect to {url}: {message}")]
    Connection {
        /// Base URL or endpoint.
        url: String,
        /// Transport message.
        message: String,
    },
    /// The request timed out.
    #[error("request timed out")]
    Timeout,
    /// The server answered with an error status.
    #[error("HTTP {status}: {message}")]
    Http {
        /// HTTP status code.
        status: u16,
        /// Provider error message (from the JSON body when possible).
        message: String,
    },
    /// The response could not be parsed or had no usable content.
    #[error("invalid response: {0}")]
    InvalidResponse(String),
}

/// Copying to the clipboard failed.
#[derive(Debug, Error)]
pub enum ClipboardError {
    /// None of `wl-copy`, `xclip`, `xsel` is installed / usable.
    #[error("no clipboard tool found: install wl-clipboard (Wayland), xclip or xsel")]
    NoTool,
    /// The tool exited with an error.
    #[error("{tool} failed: {message}")]
    Failed {
        /// Command name, e.g. "wl-copy".
        tool: String,
        /// Its stderr, or the spawn error.
        message: String,
    },
    /// The tool did not exit within the timeout (5 s in the Python app).
    #[error("{tool} timed out")]
    Timeout {
        /// Command name.
        tool: String,
    },
}
