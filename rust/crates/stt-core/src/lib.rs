//! Shared contracts for the STT app (Rust rewrite of `src/stt_app/`).
//!
//! This crate owns:
//! - [`config`]: the TOML settings schema, identical to the Python app's `config.py`, so the existing
//!   `~/.config/stt-app/config.toml` loads unchanged.
//! - [`secrets`]: API keys / tokens in the system keyring (service `stt-app`, username = provider), with an
//!   env-var fallback.
//! - The traits every other crate codes against: [`Transcriber`], [`AudioSource`] / [`Recorder`] /
//!   [`MultiRecorder`], [`LlmBackend`] / [`LlmFactory`], [`Clipboard`], [`SecretStore`], plus the shared
//!   value types ([`LlmResult`], [`InputDevice`], [`LoadEvent`], [`DownloadProgress`], ...).
//! - All library error types ([`error`]).
//! - With the `fakes` feature: in-memory fakes of every trait for other crates' tests (`fakes` module).
//!
//! Dependency rule: no crate except the `stt` binary depends on `stt-whisper`; everything else uses the
//! [`Transcriber`] trait and, in tests, `fakes::FakeTranscriber`.
#![warn(missing_docs)]

pub mod audio;
pub mod clipboard;
pub mod config;
pub mod error;
pub mod llm;
pub mod progress;
pub mod recover;
pub mod secrets;
pub mod transcribe;

#[cfg(any(test, feature = "fakes"))]
pub mod fakes;

pub use audio::{AudioSource, DeviceId, InputDevice, MultiRecorder, Recorder, find_device};
pub use clipboard::{Clipboard, ClipboardMethod};
pub use config::{
    AppConfig, AudioConfig, LlmConfig, LlmMode, Provider, ReasoningLevel, WhisperConfig,
    WhisperDecoding, WhisperDevice, WhisperQuant,
};
pub use error::{
    AudioError, ClipboardError, ConfigError, DownloadError, LlmError, SecretError, TranscribeError,
};
pub use llm::{LlmBackend, LlmFactory, LlmResult, ServerKind};
pub use progress::{DownloadProgress, DownloadProgressFn, LoadEvent, LoadProgressFn};
pub use secrets::{KeyringStore, SecretStore};
pub use transcribe::{ComputeDevice, LoadInfo, Transcriber, TranscriberFactory};

/// Sample rate of all audio passed between crates: mono `f32` samples at 16 kHz, range `-1.0..=1.0`.
pub const SAMPLE_RATE: u32 = 16_000;

/// Keyring service name and config directory name, shared with the Python app.
pub const APP_NAME: &str = "stt-app";
