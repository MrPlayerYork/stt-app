//! The speech-to-text contract. The only real implementation lives in `stt-whisper` (whisper.cpp with
//! Vulkan, GPU→CPU fallback); everything else holds an `Arc<dyn Transcriber>` and tests use
//! `fakes::FakeTranscriber`.

use std::path::PathBuf;
use std::sync::Arc;

use crate::config::AppConfig;
use crate::error::TranscribeError;
use crate::progress::{DownloadProgressFn, LoadProgressFn};
use crate::secrets::SecretStore;

/// Whisper model names offered in settings, in menu order (Python `transcribe.MODELS`). Stored in
/// `whisper.model`.
pub const WHISPER_MODELS: &[&str] = &["large-v3", "large-v3-turbo", "distil-large-v3", "medium", "small"];

/// Where a model ended up running.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ComputeDevice {
    /// A GPU through the given backend.
    Gpu {
        /// Backend name, e.g. "Vulkan".
        backend: String,
        /// Device name if the backend reports one, e.g. "NVIDIA GeForce RTX 4070 SUPER".
        name: Option<String>,
    },
    /// The CPU.
    Cpu,
}

impl std::fmt::Display for ComputeDevice {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Gpu { backend, .. } => write!(f, "gpu ({backend})"),
            Self::Cpu => f.write_str("cpu"),
        }
    }
}

/// What [`Transcriber::load`] produced.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LoadInfo {
    /// Model name from the config, e.g. "large-v3-turbo".
    pub model: String,
    /// Model file actually loaded (ggml file path).
    pub path: PathBuf,
    /// Device the model runs on.
    pub device: ComputeDevice,
    /// Extra detail for the status line, e.g. quantization "q5_0"; `""` when none.
    pub detail: String,
    /// `Some(reason)` when the GPU was tried and failed, so this is the CPU fallback.
    pub fallback_reason: Option<String>,
}

impl LoadInfo {
    /// One-line status text, e.g. `large-v3-turbo on gpu (Vulkan) q5_0` or
    /// `large-v3-turbo on cpu (GPU failed: ...)`. Python: `Transcriber.description`.
    pub fn description(&self) -> String {
        let mut s = format!("{} on {}", self.model, self.device);
        if !self.detail.is_empty() {
            s.push(' ');
            s.push_str(&self.detail);
        }
        if let Some(r) = &self.fallback_reason {
            s.push_str(&format!(" (GPU failed: {r})"));
        }
        s
    }
}

/// A loaded-on-demand speech-to-text model.
///
/// Shared between the UI's loader thread, the record flow and calibration, so all methods take `&self`
/// and implementations use interior mutability. `load` must be safe to call concurrently: one caller
/// loads, the others block until it finishes and get the same result, and every waiting caller's
/// `on_progress` also receives the in-flight download/load events (Python registers listeners before
/// taking the load lock).
pub trait Transcriber: Send + Sync {
    /// Model name this transcriber was created for (`whisper.model`).
    fn model_name(&self) -> &str;

    /// Make sure the model file is local, downloading it if needed (resumable). Returns its path.
    /// Used by the settings screen's "download now" as well as by `load`.
    fn fetch(&self, on_progress: Option<DownloadProgressFn>) -> Result<PathBuf, TranscribeError>;

    /// Fetch (if needed) and load the model, trying the GPU first and falling back to the CPU per
    /// `whisper.device`, and warm it up with a short silent inference so GPU problems surface here.
    /// Returns immediately with the existing info if already loaded.
    fn load(&self, on_progress: Option<LoadProgressFn>) -> Result<LoadInfo, TranscribeError>;

    /// Info about the loaded model, `None` if not loaded.
    fn info(&self) -> Option<LoadInfo>;

    /// Transcribe mono 16 kHz `f32` audio (see [`crate::SAMPLE_RATE`]) to text, segments joined with
    /// spaces and trimmed. Empty audio returns `Ok("")` without loading. Loads the model first if needed.
    /// Uses voice-activity filtering / no-speech suppression where available so silence gives `""`.
    fn transcribe(&self, audio: &[f32]) -> Result<String, TranscribeError>;

    /// Free the model (VRAM/RAM). A later `transcribe`/`load` loads it again.
    fn unload(&self);
}

/// Builds a [`Transcriber`] from the current settings. The binary provides the real one (stt-whisper);
/// the UI calls it at start and whenever whisper settings change.
pub trait TranscriberFactory: Send + Sync {
    /// Create a transcriber for `cfg.whisper` (model, device, language, models dir, download
    /// connections). The Hugging Face token comes from `secrets` (see
    /// [`crate::secrets::HF_TOKEN_USER`]). Must not load or download anything yet.
    fn create(
        &self,
        cfg: &AppConfig,
        secrets: &dyn SecretStore,
    ) -> Result<Arc<dyn Transcriber>, TranscribeError>;
}
