//! Provider dispatch (Python `make_backend`, `process`, `DEFAULT_MODELS`). Owner: P2-6.

use stt_core::{LlmBackend, LlmConfig, LlmError, LlmFactory, LlmResult};

/// The real [`LlmFactory`]: `openai`/`openai_compat` → [`crate::openai_compat::OpenAiCompatBackend`],
/// `anthropic` → [`crate::anthropic::AnthropicBackend::new`], `google` →
/// [`crate::google::GoogleBackend::new`].
pub struct HttpLlmFactory;

impl LlmFactory for HttpLlmFactory {
    fn create(
        &self,
        cfg: &LlmConfig,
        api_key: Option<&str>,
    ) -> Result<Box<dyn LlmBackend>, LlmError> {
        let _ = (cfg, api_key);
        todo!("P2-6")
    }
}

/// Build the backend and run the current mode's prompt over `text`, answer trimmed (Python `process`).
pub fn process(
    factory: &dyn LlmFactory,
    cfg: &LlmConfig,
    api_key: Option<&str>,
    text: &str,
) -> Result<LlmResult, LlmError> {
    let _ = (factory, cfg, api_key, text);
    todo!("P2-6")
}
