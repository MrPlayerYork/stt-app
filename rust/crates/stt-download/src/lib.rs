//! Fast, resumable Hugging Face model downloads (blocking `ureq`, threads).
//!
//! Multi-connection ranged downloads with resume (`<file>.part` + `<file>.progress.json` sidecar, the
//! same layout as `src/stt_app/download.py`, so partial downloads resume across the Python and Rust
//! apps), HF file listing, bearer token (dropped on cross-host redirects), retries with backoff, fatal
//! 401/403/404 handling, reuse of existing files, and progress via [`stt_core::DownloadProgressFn`].
//! Port of `reference/hf_downloader.py` + `src/stt_app/download.py`.

mod engine;
mod http;
#[cfg(test)]
mod tests;

use std::path::Path;
use std::time::Duration;

use stt_core::{DownloadError, DownloadProgressFn};

const HF_API: &str = "https://huggingface.co/api/models";
const HF_RESOLVE: &str = "https://huggingface.co";

/// Tunables; tests shorten them.
#[derive(Debug, Clone)]
pub(crate) struct Options {
    pub max_retries: u32,
    /// Sleep before retry `n` is `retry_backoff * n`.
    pub retry_backoff: Duration,
    pub save_interval: Duration,
    pub report_interval: Duration,
}

impl Default for Options {
    fn default() -> Self {
        Self {
            max_retries: 5,
            retry_backoff: Duration::from_secs(3),
            save_interval: Duration::from_millis(500),
            report_interval: Duration::from_millis(200),
        }
    }
}

/// Percent-encode like Python's `urllib.parse.quote` (`/` kept).
fn quote(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for b in s.bytes() {
        if b.is_ascii_alphanumeric() || matches!(b, b'_' | b'.' | b'-' | b'~' | b'/') {
            out.push(b as char);
        } else {
            out.push_str(&format!("%{b:02X}"));
        }
    }
    out
}

/// `https://huggingface.co/<repo>/resolve/main/<file>` (file path segments URL-encoded).
pub fn hf_resolve_url(repo: &str, file: &str) -> String {
    format!("{HF_RESOLVE}/{repo}/resolve/main/{}", quote(file))
}

/// File paths in a Hugging Face model repo (Python `_repo_files`, without the faster-whisper filter).
pub fn list_repo_files(repo: &str, token: Option<&str>) -> Result<Vec<String>, DownloadError> {
    engine::list_repo_files_at(HF_API, repo, token)
}

/// Download `url` to `dest` with `connections` parallel ranged requests, resuming from
/// `dest.part` + `dest.progress.json`. Returns at once if `dest` exists (Python `download_file`).
/// `dest`'s parent directory is created if missing. Blocks until done.
pub fn download_file(
    url: &str,
    dest: &Path,
    token: Option<&str>,
    connections: u32,
    on_progress: Option<DownloadProgressFn>,
) -> Result<(), DownloadError> {
    engine::download(
        &Options::default(),
        url,
        dest,
        token,
        connections,
        on_progress,
    )
}
