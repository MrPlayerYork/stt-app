//! OpenAI and OpenAI-compatible backend, server detection, idle TTL and unload (Python
//! `llm/openai_compat.py`). Owner: P2-6.

use stt_core::{LlmBackend, LlmError, LlmResult, ReasoningLevel, ServerKind};

/// Detect what kind of server is behind `base_url` (Python `detect_server`).
pub fn detect_server(base_url: &str, api_key: Option<&str>) -> ServerKind {
    let _ = (base_url, api_key);
    todo!("P2-6")
}

/// OpenAI (`base_url = None`) or any OpenAI-compatible server.
pub struct OpenAiCompatBackend {}

impl OpenAiCompatBackend {
    /// No network on construction. `idle_ttl_seconds = 0` means no TTL.
    pub fn new(
        model: &str,
        api_key: Option<&str>,
        base_url: Option<&str>,
        idle_ttl_seconds: u64,
        reasoning: ReasoningLevel,
    ) -> Self {
        let _ = (model, api_key, base_url, idle_ttl_seconds, reasoning);
        todo!("P2-6")
    }
}

impl LlmBackend for OpenAiCompatBackend {
    fn complete(&self, system_prompt: &str, text: &str) -> Result<LlmResult, LlmError> {
        let _ = (system_prompt, text);
        todo!("P2-6")
    }
    fn list_models(&self) -> Result<Vec<String>, LlmError> {
        todo!("P2-6")
    }
}
