//! Port of the OpenAI-compatible parts of `tests/test_llm_reasoning.py`, plus wire behaviour.

mod openai_support;

use openai_support::{Fake, Reply, chat};
use serde_json::{Value, json};
use stt_core::{LlmBackend, LlmError, ReasoningLevel};
use stt_llm::openai_compat::OpenAiCompatBackend;

fn backend(url: &str, level: ReasoningLevel) -> OpenAiCompatBackend {
    OpenAiCompatBackend::new("m", None, Some(url), 0, level)
}

fn ok_server(message: Value, usage: Option<Value>) -> Fake {
    Fake::start(move |_| Reply::json(200, chat(message.clone(), usage.clone())))
}

#[test]
fn reasoning_content_field() {
    let f = ok_server(
        json!({"content": "Hi.", "reasoning_content": "thought hard"}),
        None,
    );
    let r = backend(&f.url, ReasoningLevel::Default)
        .complete("s", "t")
        .unwrap();
    assert_eq!(
        (r.text.as_str(), r.reasoning.as_str(), r.reasoned),
        ("Hi.", "thought hard", Some(true))
    );
    assert!(f.chat_bodies()[0].get("reasoning_effort").is_none());
}

#[test]
fn reasoning_field_ollama_style() {
    let f = ok_server(
        json!({"content": "Hi.", "reasoning": "ollama thoughts"}),
        None,
    );
    let r = backend(&f.url, ReasoningLevel::Default)
        .complete("s", "t")
        .unwrap();
    assert_eq!(
        (r.text.as_str(), r.reasoning.as_str()),
        ("Hi.", "ollama thoughts")
    );
}

#[test]
fn inline_think() {
    let f = ok_server(json!({"content": "<think>hmm</think>\n\nHi."}), None);
    let r = backend(&f.url, ReasoningLevel::Default)
        .complete("s", "t")
        .unwrap();
    assert_eq!(
        (r.text.as_str(), r.reasoning.as_str(), r.reasoned),
        ("Hi.", "hmm", Some(true))
    );
}

#[test]
fn field_and_inline_reasoning_are_joined() {
    let f = ok_server(
        json!({"content": "<think>b</think>Hi.", "reasoning": "a"}),
        None,
    );
    let r = backend(&f.url, ReasoningLevel::Default)
        .complete("s", "t")
        .unwrap();
    assert_eq!(r.reasoning, "a\n\nb");
}

#[test]
fn unterminated_inline_think() {
    let f = ok_server(json!({"content": "<think>cut off"}), None);
    let r = backend(&f.url, ReasoningLevel::Default)
        .complete("s", "t")
        .unwrap();
    assert_eq!((r.text.as_str(), r.reasoning.as_str()), ("", "cut off"));
}

#[test]
fn null_content_is_empty() {
    let f = ok_server(json!({"content": null}), None);
    let r = backend(&f.url, ReasoningLevel::Default)
        .complete("s", "t")
        .unwrap();
    assert_eq!(r.text, "");
}

#[test]
fn reasoning_tokens_usage() {
    let usage = |n| {
        json!({"prompt_tokens": 1, "completion_tokens": 9, "total_tokens": 10,
        "completion_tokens_details": {"reasoning_tokens": n}})
    };
    let f = ok_server(json!({"content": "Hi."}), Some(usage(7)));
    let r = backend(&f.url, ReasoningLevel::Default)
        .complete("s", "t")
        .unwrap();
    assert_eq!(
        (r.reasoning_tokens, r.reasoned, r.reasoning.as_str()),
        (Some(7), Some(true), "")
    );
    let f = ok_server(json!({"content": "Hi."}), Some(usage(0)));
    let r = backend(&f.url, ReasoningLevel::Default)
        .complete("s", "t")
        .unwrap();
    assert_eq!((r.reasoning_tokens, r.reasoned), (Some(0), Some(false)));
}

#[test]
fn unknown_when_no_signal() {
    let f = ok_server(json!({"content": "Hi."}), None);
    let r = backend(&f.url, ReasoningLevel::Default)
        .complete("s", "t")
        .unwrap();
    assert_eq!((r.reasoned, r.reasoning_tokens), (None, None));
}

#[test]
fn reasoning_effort_only_when_not_default() {
    use ReasoningLevel::*;
    for (level, sent) in [
        (Default, None),
        (Off, Some("none")),
        (Low, Some("low")),
        (Medium, Some("medium")),
        (High, Some("high")),
    ] {
        let f = ok_server(json!({"content": "ok"}), None);
        backend(&f.url, level).complete("s", "t").unwrap();
        let b = &f.chat_bodies()[0];
        assert_eq!(
            b.get("reasoning_effort").and_then(Value::as_str),
            sent,
            "{level:?}"
        );
        assert_eq!(b.get("reasoning_effort").is_some(), level != Default);
    }
}

#[test]
fn exact_request_body_and_headers() {
    let f = ok_server(json!({"content": "ok"}), None);
    OpenAiCompatBackend::new(
        "m",
        Some("sk-1"),
        Some(&format!("{}/", f.url)),
        0,
        ReasoningLevel::High,
    )
    .complete("sys", "user text")
    .unwrap();
    let req = f.log().remove(0);
    assert_eq!(
        (req.method.as_str(), req.path.as_str()),
        ("POST", "/v1/chat/completions")
    );
    assert_eq!(req.auth.as_deref(), Some("Bearer sk-1"));
    assert_eq!(
        req.body.unwrap(),
        json!({"model": "m", "temperature": 0.2, "reasoning_effort": "high",
               "messages": [{"role": "system", "content": "sys"}, {"role": "user", "content": "user text"}]})
    );
}

#[test]
fn placeholder_key_without_key() {
    let f = ok_server(json!({"content": "ok"}), None);
    backend(&f.url, ReasoningLevel::Default)
        .complete("s", "t")
        .unwrap();
    assert_eq!(f.log()[0].auth.as_deref(), Some("Bearer not-needed"));
}

fn rejects_effort(r: &openai_support::Req) -> Reply {
    if r.body
        .as_ref()
        .is_some_and(|b| b.get("reasoning_effort").is_some())
    {
        Reply::json(
            400,
            json!({"error": {"message": "Unknown parameter: reasoning_effort", "type": "invalid_request_error"}}),
        )
    } else {
        Reply::json(200, chat(json!({"content": "ok"}), None))
    }
}

#[test]
fn bad_request_on_reasoning_effort_retries_without_param() {
    let f = Fake::start(rejects_effort);
    let r = backend(&f.url, ReasoningLevel::High)
        .complete("s", "t")
        .unwrap();
    assert_eq!(
        (r.text.as_str(), r.note.as_str()),
        ("ok", "server ignored reasoning setting")
    );
    let bodies = f.chat_bodies();
    assert!(bodies[0].get("reasoning_effort").is_some());
    assert!(
        bodies[1..]
            .iter()
            .all(|b| b.get("reasoning_effort").is_none())
    );
}

#[test]
fn off_retries_once_then_drops_the_field() {
    let f = Fake::start(rejects_effort);
    let r = backend(&f.url, ReasoningLevel::Off)
        .complete("s", "t")
        .unwrap();
    assert_eq!(r.note, "server ignored reasoning setting");
    let bodies = f.chat_bodies();
    assert_eq!(bodies.len(), 2);
    assert_eq!(bodies[0]["reasoning_effort"], "none");
}

#[test]
fn unrelated_400_still_errors() {
    let f = Fake::start(|_| Reply::json(400, json!({"error": {"message": "context too long"}})));
    let e = backend(&f.url, ReasoningLevel::High)
        .complete("s", "t")
        .unwrap_err();
    assert!(
        matches!(e, LlmError::Http { status: 400, ref message } if message == "context too long"),
        "{e}"
    );
    assert_eq!(f.chat_bodies().len(), 1);
}

#[test]
fn temperature_400_retries_without_temperature() {
    let f = Fake::start(|r| {
        if r.body
            .as_ref()
            .is_some_and(|b| b.get("temperature").is_some())
        {
            Reply::json(
                400,
                json!({"error": {"message": "Unsupported value: 'temperature' does not support 0.2"}}),
            )
        } else {
            Reply::json(200, chat(json!({"content": "ok"}), None))
        }
    });
    let r = backend(&f.url, ReasoningLevel::Default)
        .complete("s", "t")
        .unwrap();
    assert_eq!(r.text, "ok");
    assert!(f.chat_bodies()[1].get("temperature").is_none());
}

#[test]
fn status_errors_have_human_messages() {
    for (code, text) in [
        (401, "Authentication failed: check the API key"),
        (403, "Permission denied by the API"),
        (404, "Model or endpoint not found: check the model name"),
        (429, "Rate limited or quota exceeded"),
    ] {
        let f = Fake::start(move |_| Reply::json(code, json!({"error": {"message": "x"}})));
        let e = backend(&f.url, ReasoningLevel::Default)
            .complete("s", "t")
            .unwrap_err();
        assert!(
            matches!(&e, LlmError::Http { status, message } if *status == code && message == text),
            "{e}"
        );
    }
    let f = Fake::start(|_| Reply::text(418, &"x".repeat(500)));
    let e = backend(&f.url, ReasoningLevel::Default)
        .complete("s", "t")
        .unwrap_err();
    assert!(
        matches!(&e, LlmError::Http { status: 418, message } if message.len() == 200),
        "{e}"
    );
}

#[test]
fn one_retry_on_server_error() {
    let n = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let n2 = n.clone();
    let f = Fake::start(move |_| {
        if n2.fetch_add(1, std::sync::atomic::Ordering::SeqCst) == 0 {
            Reply::json(500, json!({"error": {"message": "boom"}}))
        } else {
            Reply::json(200, chat(json!({"content": "ok"}), None))
        }
    });
    assert_eq!(
        backend(&f.url, ReasoningLevel::Default)
            .complete("s", "t")
            .unwrap()
            .text,
        "ok"
    );
    assert_eq!(f.chat_bodies().len(), 2);
}

#[test]
fn unreachable_server_is_a_connection_error() {
    let e = backend("http://127.0.0.1:9/v1", ReasoningLevel::Default)
        .complete("s", "t")
        .unwrap_err();
    assert!(matches!(e, LlmError::Connection { .. }), "{e}");
}

#[test]
fn no_model_is_an_error() {
    let b = OpenAiCompatBackend::new(
        "",
        None,
        Some("http://127.0.0.1:9/v1"),
        0,
        ReasoningLevel::Default,
    );
    assert!(b.complete("s", "t").is_err());
}

#[test]
fn list_models_sorted() {
    let f = Fake::start(|r| {
        assert_eq!((r.method.as_str(), r.path.as_str()), ("GET", "/v1/models"));
        Reply::json(
            200,
            json!({"object": "list", "data": [{"id": "b"}, {"id": "a"}, {"id": "C"}]}),
        )
    });
    assert_eq!(
        backend(&f.url, ReasoningLevel::Default)
            .list_models()
            .unwrap(),
        ["C", "a", "b"]
    );
}

#[test]
fn list_models_auth_error() {
    let f = Fake::start(|_| Reply::json(401, json!({"error": {"message": "bad key"}})));
    let e = backend(&f.url, ReasoningLevel::Default)
        .list_models()
        .unwrap_err();
    assert!(matches!(e, LlmError::Http { status: 401, .. }), "{e}");
}
