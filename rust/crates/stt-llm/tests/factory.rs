//! Port of `tests/test_llm_factory.py` (no network except a local fake server).

mod openai_support;

use openai_support::{Fake, Reply, chat};
use serde_json::json;
use stt_core::fakes::{FakeLlm, FakeLlmFactory};
use stt_core::{LlmConfig, LlmError, LlmFactory, Provider};
use stt_llm::factory::{DEFAULT_MODELS, default_model};
use stt_llm::{HttpLlmFactory, process};

fn cfg(provider: Provider) -> LlmConfig {
    LlmConfig {
        provider,
        ..LlmConfig::default()
    }
}

fn err(c: &LlmConfig, key: Option<&str>) -> LlmError {
    HttpLlmFactory
        .create(c, key)
        .err()
        .expect("expected an error")
}

#[test]
fn cloud_provider_without_key_errors() {
    for p in [Provider::OpenAi, Provider::Anthropic, Provider::Google] {
        let c = LlmConfig {
            api_key_env: "SOME_ENV".into(),
            ..cfg(p)
        };
        for key in [None, Some("")] {
            match err(&c, key) {
                LlmError::NoApiKey { provider, env } => {
                    assert_eq!((provider.as_str(), env.as_str()), (p.label(), "SOME_ENV"));
                }
                e => panic!("{e}"),
            }
        }
    }
    assert!(
        err(&cfg(Provider::Google), None)
            .to_string()
            .contains("env var ?")
    );
}

#[test]
fn openai_compat_requires_base_url() {
    assert!(matches!(
        err(&cfg(Provider::OpenAiCompat), None),
        LlmError::NoBaseUrl
    ));
}

#[test]
fn backends_build_without_network() {
    for (p, model) in [
        (Provider::OpenAi, "gpt-x"),
        (Provider::Anthropic, "claude-x"),
        (Provider::Google, "gemini-x"),
    ] {
        let c = LlmConfig {
            model: model.into(),
            ..cfg(p)
        };
        assert!(HttpLlmFactory.create(&c, Some("dummy")).is_ok());
    }
}

#[test]
fn openai_compat_works_without_key_and_passes_settings() {
    let f = Fake::start(|_| Reply::json(200, chat(json!({"content": "  ok \n"}), None)));
    let c = LlmConfig {
        model: "mm".into(),
        base_url: f.url.clone(),
        ..cfg(Provider::OpenAiCompat)
    };
    let b = HttpLlmFactory.create(&c, None).unwrap();
    assert_eq!(b.complete("s", "t").unwrap().text, "ok");
    assert_eq!(f.chat_bodies()[0]["model"], "mm");
}

#[test]
fn default_models_match_python() {
    assert_eq!(DEFAULT_MODELS.len(), 4);
    assert_eq!(default_model(Provider::OpenAi), "gpt-5.4-mini");
    assert_eq!(default_model(Provider::OpenAiCompat), "");
    assert_eq!(default_model(Provider::Anthropic), "claude-haiku-5-5");
    assert_eq!(default_model(Provider::Google), "gemini-2.5-flash");
}

#[test]
fn process_returns_trimmed_output() {
    // Uses prompts::system_prompt_for (P2-7); panics until that lands.
    let factory = FakeLlmFactory::new(FakeLlm::new("  \n Hello there. \t\n"));
    let c = cfg(Provider::OpenAi);
    let out = process(&factory, &c, Some("k"), "um hello there").unwrap();
    assert_eq!(out.text, "Hello there.");
    assert_eq!(factory.requests(), vec![c]);
}
