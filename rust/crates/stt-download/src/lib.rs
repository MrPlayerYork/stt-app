//! Fast, resumable Hugging Face model downloads (blocking `ureq`, threads).
//!
//! Owns (P2-4): multi-connection ranged downloads with resume (`<file>.part` + `<file>.progress.json`
//! sidecar, same layout as `src/stt_app/download.py`), HF file listing, bearer token (dropped on
//! cross-host redirects), retries with backoff, fatal 401/403/404 handling, reuse of existing files,
//! and progress via [`stt_core::DownloadProgressFn`]. Errors are [`stt_core::DownloadError`].
//! Port of `reference/hf_downloader.py` + `src/stt_app/download.py`.
//!
//! The public signatures below are fixed now, because stt-whisper (P2-8) builds against them in parallel.

use std::path::Path;

use stt_core::{DownloadError, DownloadProgressFn};

/// `https://huggingface.co/<repo>/resolve/main/<file>` (file path segments URL-encoded).
pub fn hf_resolve_url(repo: &str, file: &str) -> String {
    let _ = (repo, file);
    todo!("P2-4")
}

/// File paths in a Hugging Face model repo (Python `_repo_files`).
pub fn list_repo_files(repo: &str, token: Option<&str>) -> Result<Vec<String>, DownloadError> {
    let _ = (repo, token);
    todo!("P2-4")
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
    let _ = (url, dest, token, connections, on_progress);
    todo!("P2-4")
}
