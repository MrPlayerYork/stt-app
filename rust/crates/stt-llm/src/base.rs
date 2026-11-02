//! Shared constants and `<think>` handling (Python `llm/base.py`). Owner: P2-6. The constants are fixed
//! here so P2-7 can use them.

use std::time::Duration;

/// Request timeout for every provider (Python `TIMEOUT_S`).
pub const TIMEOUT: Duration = Duration::from_secs(60);

/// Sampling temperature for every provider (Python `TEMPERATURE`).
pub const TEMPERATURE: f32 = 0.2;

/// Split inline `<think>…</think>` / `<thinking>…</thinking>` blocks out of `content` →
/// `(answer, reasoning)`, handling an unclosed opening tag and a stray closing tag like Python
/// `split_think`.
pub fn split_think(content: &str) -> (String, String) {
    let _ = content;
    todo!("P2-6")
}
