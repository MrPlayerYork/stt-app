//! LLM post-processing contracts. Real backends (OpenAI, OpenAI-compatible, Anthropic, Google) and the
//! prompts live in `stt-llm`; tests use `fakes::FakeLlm`.

use crate::config::LlmConfig;
use crate::error::LlmError;

/// Input used by "Test connection" in the LLM settings (Python `ui.SAMPLE_TEXT`).
pub const TEST_SAMPLE_TEXT: &str =
    "um so i think we should uh meet on tuesday no wait wednesday at three pm";

/// Result of one completion.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct LlmResult {
    /// Final answer only, trimmed. Never contains reasoning or `<think>` text: this is what reaches the
    /// clipboard.
    pub text: String,
    /// Reasoning/thinking text if the provider exposed it (separate field or stripped `<think>` blocks);
    /// `""` otherwise. Shown via the "\[v\] view" reasoning line.
    pub reasoning: String,
    /// Reasoning tokens reported by the provider, if any.
    pub reasoning_tokens: Option<u64>,
    /// `Some(true/false)` when it is known whether the model reasoned, `None` when the provider can't tell.
    pub reasoned: Option<bool>,
    /// Short human note, e.g. "model has no reasoning control"; `""` when none.
    pub note: String,
}

/// What kind of OpenAI-compatible server is behind a base URL (Python `detect_server`). Decides how the
/// idle TTL is sent (LM Studio `ttl`, Ollama `keep_alive`) and how unload works.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum ServerKind {
    /// LM Studio.
    LmStudio,
    /// Ollama.
    Ollama,
    /// Anything else (vLLM, llama-server, cloud providers). Default.
    #[default]
    Other,
}

/// One configured LLM backend. Built per use from the current [`LlmConfig`] by an [`LlmFactory`]
/// (cheap: no network on construction). Calls are blocking (ureq); the UI runs them on a worker thread,
/// hence `Send + Sync`.
///
/// Settings that apply per request (idle TTL, reasoning level, base URL) are taken from the config at
/// construction.
pub trait LlmBackend: Send + Sync {
    /// Run `system_prompt` over `text` (the raw transcript, sent as the user message). Applies the
    /// configured reasoning level (with the provider-specific fallbacks, e.g. retry without
    /// `reasoning_effort` on HTTP 400) and, for local servers, the idle TTL.
    fn complete(&self, system_prompt: &str, text: &str) -> Result<LlmResult, LlmError>;

    /// Model ids available to this key / server, sorted for display.
    fn list_models(&self) -> Result<Vec<String>, LlmError>;

    /// Ask a local server to unload the model now (used on quit when `llm.unload_on_exit` is set and the
    /// LLM was used). Default: nothing to do (cloud providers).
    fn unload(&self) -> Result<(), LlmError> {
        Ok(())
    }

    /// Server kind for `openai_compat` (detected once, cached); [`ServerKind::Other`] for cloud providers.
    fn server_kind(&self) -> ServerKind {
        ServerKind::Other
    }

    /// "Test connection": a full completion of [`TEST_SAMPLE_TEXT`] with `system_prompt` (the prompt the
    /// current mode would use), exactly like Python's `test_llm`.
    fn test_connection(&self, system_prompt: &str) -> Result<LlmResult, LlmError> {
        self.complete(system_prompt, TEST_SAMPLE_TEXT)
    }
}

/// Builds backends from settings (Python `make_backend`). The binary provides the `stt-llm` one; UI
/// tests use `fakes::FakeLlmFactory`.
pub trait LlmFactory: Send + Sync {
    /// Build the backend for `cfg.provider`. Errors: [`LlmError::NoApiKey`] for a cloud provider without
    /// a key, [`LlmError::NoBaseUrl`] for `openai_compat` without `base_url`.
    fn create(
        &self,
        cfg: &LlmConfig,
        api_key: Option<&str>,
    ) -> Result<Box<dyn LlmBackend>, LlmError>;
}
