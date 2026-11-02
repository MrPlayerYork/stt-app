//! Progress callbacks for model downloads and model loading.
//!
//! Callbacks are `Arc<dyn Fn + Send + Sync>` because they are called from worker threads (one per
//! download connection, the model loader thread) and may be stored while a load is in progress. They must
//! be cheap and must not block: the UI typically stores the latest event behind a mutex and redraws on
//! its own tick.

use std::sync::Arc;

use crate::transcribe::{ComputeDevice, LoadInfo};

/// Bytes downloaded so far for one file. Reported at most every ~200 ms per file, plus once at the end.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DownloadProgress {
    /// File name being downloaded, e.g. `ggml-large-v3-turbo.bin`.
    pub file: String,
    /// Bytes present locally (including bytes from a resumed `.part` file).
    pub done: u64,
    /// Total file size in bytes (known before the first report).
    pub total: u64,
}

impl DownloadProgress {
    /// Completion in percent, `0..=100` (0 when `total` is 0).
    pub fn percent(&self) -> u8 {
        if self.total == 0 {
            return 0;
        }
        (self.done.min(self.total) * 100 / self.total) as u8
    }
}

/// Callback receiving download progress (Python `ProgressFn(done, total, name)`).
pub type DownloadProgressFn = Arc<dyn Fn(&DownloadProgress) + Send + Sync>;

/// One step of [`crate::Transcriber::load`], in order: zero or more `Download`, then `Loading` per device
/// attempt (with `Fallback` between a failed GPU attempt and the CPU attempt), then `Ready`.
#[derive(Debug, Clone, PartialEq)]
pub enum LoadEvent {
    /// Model file download progress (only when the file is not local yet).
    Download(DownloadProgress),
    /// Loading the model onto `device` (includes the warm-up inference).
    Loading {
        /// Device being tried.
        device: ComputeDevice,
    },
    /// The GPU attempt failed; trying the CPU next.
    Fallback {
        /// Why the GPU attempt failed (short message).
        reason: String,
    },
    /// Loaded and warmed up.
    Ready(LoadInfo),
}

/// Callback receiving model-load events.
pub type LoadProgressFn = Arc<dyn Fn(&LoadEvent) + Send + Sync>;
