//! Google backend over raw HTTPS (Python `llm/google_backend.py`). Owner: P2-7.

use stt_core::{LlmBackend, LlmError, LlmResult, ReasoningLevel};

/// Google backend. The constructor signature is fixed (P2-6's factory calls it); fields are P2-7's choice.
pub struct GoogleBackend {}

impl GoogleBackend {
    /// No network on construction.
    pub fn new(model: &str, api_key: &str, reasoning: ReasoningLevel) -> Self {
        let _ = (model, api_key, reasoning);
        todo!("P2-7")
    }
}

impl LlmBackend for GoogleBackend {
    fn complete(&self, system_prompt: &str, text: &str) -> Result<LlmResult, LlmError> {
        let _ = (system_prompt, text);
        todo!("P2-7")
    }
    fn list_models(&self) -> Result<Vec<String>, LlmError> {
        todo!("P2-7")
    }
}
