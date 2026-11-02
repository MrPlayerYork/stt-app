//! Fast, resumable Hugging Face model downloads (blocking `ureq`, threads).
//!
//! Owns (P2-4): multi-connection ranged downloads with resume (`<file>.part` + `<file>.progress.json`
//! sidecar, same layout as `src/stt_app/download.py`), HF file listing, bearer token (dropped on
//! cross-host redirects), retries with backoff, fatal 401/403/404 handling, reuse of existing files,
//! and progress via [`stt_core::DownloadProgressFn`]. Errors are [`stt_core::DownloadError`].
//! Port of `reference/hf_downloader.py` + `src/stt_app/download.py`.
