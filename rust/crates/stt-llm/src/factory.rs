//! Provider dispatch (Python `make_backend`, `process`, `DEFAULT_MODELS`). Owner: P2-6.

use stt_core::{LlmBackend, LlmConfig, LlmError, LlmFactory, LlmResult, Provider};

use crate::anthropic::AnthropicBackend;
use crate::google::GoogleBackend;
use crate::openai_compat::OpenAiCompatBackend;
use crate::prompts::system_prompt_for;

/// Default model per provider (Python `DEFAULT_MODELS`).
pub const DEFAULT_MODELS: &[(Provider, &str)] = &[
    (Provider::OpenAi, "gpt-5.4-mini"),
    (Provider::OpenAiCompat, ""),
    (Provider::Anthropic, "claude-haiku-5-5"),
    (Provider::Google, "gemini-2.5-flash"),
];

/// The default model id for `provider` (`""` for `openai_compat`).
pub fn default_model(provider: Provider) -> &'static str {
    DEFAULT_MODELS
        .iter()
        .find(|(p, _)| *p == provider)
        .map_or("", |(_, m)| m)
}

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
        let provider = cfg.provider;
        let key = api_key.filter(|k| !k.is_empty());
        if provider != Provider::OpenAiCompat && key.is_none() {
            return Err(LlmError::NoApiKey {
                provider: provider.label().to_string(),
                env: if cfg.api_key_env.is_empty() {
                    "?".to_string()
                } else {
                    cfg.api_key_env.clone()
                },
            });
        }
        let reasoning = cfg.reasoning;
        Ok(match provider {
            Provider::OpenAiCompat => {
                if cfg.base_url.is_empty() {
                    return Err(LlmError::NoBaseUrl);
                }
                Box::new(OpenAiCompatBackend::new(
                    &cfg.model,
                    api_key,
                    Some(&cfg.base_url),
                    cfg.idle_ttl_seconds,
                    reasoning,
                ))
            }
            Provider::OpenAi => Box::new(OpenAiCompatBackend::new(
                &cfg.model, api_key, None, 0, reasoning,
            )),
            Provider::Anthropic => Box::new(AnthropicBackend::new(
                &cfg.model,
                key.unwrap_or_default(),
                reasoning,
            )),
            Provider::Google => Box::new(GoogleBackend::new(
                &cfg.model,
                key.unwrap_or_default(),
                reasoning,
            )),
        })
    }
}

/// Build the backend and run the current mode's prompt over `text`, answer trimmed (Python `process`).
pub fn process(
    factory: &dyn LlmFactory,
    cfg: &LlmConfig,
    api_key: Option<&str>,
    text: &str,
) -> Result<LlmResult, LlmError> {
    let mut r = factory
        .create(cfg, api_key)?
        .complete(&system_prompt_for(cfg), text)?;
    r.text = r.text.trim().to_string();
    Ok(r)
}
