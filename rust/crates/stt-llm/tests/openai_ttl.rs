//! Port of `tests/test_llm_ttl.py`: TTL / unload against fake LM Studio / Ollama / other servers.

mod openai_support;

use openai_support::{Fake, Reply, chat};
use serde_json::{Value, json};
use stt_core::{LlmBackend, LlmError, ReasoningLevel, ServerKind};
use stt_llm::openai_compat::{OpenAiCompatBackend, detect_server};

fn server(kind: &'static str) -> Fake {
    Fake::start(move |r| match (r.method.as_str(), r.path.as_str()) {
        ("GET", "/") if kind == "ollama" => Reply::text(200, "Ollama is running"),
        ("GET", "/api/version") if kind == "ollama" => Reply::json(200, json!({"version": "0.1"})),
        ("GET", "/api/v0/models") if kind == "lmstudio" => Reply::json(
            401,
            json!({"error": {"message": "An LM Studio API token is required"}}),
        ),
        ("GET", _) => Reply::json(404, json!({"error": "nope"})),
        ("POST", "/v1/chat/completions") => Reply::json(200, chat(json!({"content": "ok"}), None)),
        _ => Reply::json(200, json!({})),
    })
}

fn backend(url: &str, ttl: u64) -> OpenAiCompatBackend {
    OpenAiCompatBackend::new("m", Some("tok"), Some(url), ttl, ReasoningLevel::Default)
}

fn posts(f: &Fake) -> Vec<(String, Option<Value>)> {
    f.log()
        .into_iter()
        .filter(|r| r.method == "POST")
        .map(|r| (r.path, r.body))
        .collect()
}

#[test]
fn detection() {
    for (name, kind) in [
        ("ollama", ServerKind::Ollama),
        ("lmstudio", ServerKind::LmStudio),
        ("other", ServerKind::Other),
    ] {
        let f = server(name);
        assert_eq!(detect_server(&f.url, None), kind, "{name}");
    }
}

#[test]
fn detection_never_fails() {
    assert_eq!(
        detect_server("http://127.0.0.1:1/v1", None),
        ServerKind::Other
    );
    assert_eq!(detect_server("not a url", None), ServerKind::Other);
}

#[test]
fn lmstudio_sends_ttl() {
    let f = server("lmstudio");
    assert_eq!(backend(&f.url, 120).complete("s", "hi").unwrap().text, "ok");
    assert_eq!(f.chat_bodies()[0]["ttl"], 120);
}

#[test]
fn ollama_sets_keep_alive_natively() {
    let f = server("ollama");
    backend(&f.url, 90).complete("s", "hi").unwrap();
    assert!(f.chat_bodies()[0].get("ttl").is_none());
    assert!(posts(&f).contains(&(
        "/api/generate".into(),
        Some(json!({"model": "m", "keep_alive": 90}))
    )));
}

#[test]
fn other_sends_nothing_extra() {
    let f = server("other");
    backend(&f.url, 120).complete("s", "hi").unwrap();
    let mut keys: Vec<_> = f.chat_bodies()[0]
        .as_object()
        .unwrap()
        .keys()
        .cloned()
        .collect();
    keys.sort();
    assert_eq!(keys, ["messages", "model", "temperature"]);
    assert_eq!(posts(&f).len(), 1);
}

#[test]
fn ttl_zero_sends_nothing_extra() {
    let f = server("lmstudio");
    backend(&f.url, 0).complete("s", "hi").unwrap();
    assert!(f.chat_bodies()[0].get("ttl").is_none());
    assert!(
        f.log().iter().all(|r| r.method == "POST"),
        "no detection probes without a TTL"
    );
}

#[test]
fn unload_endpoints() {
    let f = server("lmstudio");
    backend(&f.url, 0).unload().unwrap();
    assert!(posts(&f).contains(&(
        "/api/v1/models/unload".into(),
        Some(json!({"instance_id": "m"}))
    )));
    let f = server("ollama");
    backend(&f.url, 0).unload().unwrap();
    assert!(posts(&f).contains(&(
        "/api/generate".into(),
        Some(json!({"model": "m", "keep_alive": 0}))
    )));
    let f = server("other");
    backend(&f.url, 0).unload().unwrap();
    assert!(posts(&f).is_empty());
}

#[test]
fn unload_http_error() {
    let f = Fake::start(|_| Reply::json(500, json!({})));
    let e = backend(&f.url, 0)
        .with_server_kind(ServerKind::LmStudio)
        .unload()
        .unwrap_err();
    assert!(matches!(e, LlmError::Http { status: 500, .. }), "{e}");
}

#[test]
fn unload_failure_is_an_error() {
    let b = backend("http://127.0.0.1:9/v1/", 0).with_server_kind(ServerKind::LmStudio);
    assert!(matches!(
        b.unload().unwrap_err(),
        LlmError::Connection { .. }
    ));
}

#[test]
fn cloud_unload_is_a_noop() {
    OpenAiCompatBackend::new("gpt", Some("k"), None, 0, ReasoningLevel::Default)
        .unload()
        .unwrap();
}

#[test]
fn detection_cached() {
    let f = server("ollama");
    let b = backend(&f.url, 60);
    b.complete("s", "a").unwrap();
    b.complete("s", "b").unwrap();
    let root_gets = f
        .log()
        .iter()
        .filter(|r| r.method == "GET" && r.path == "/")
        .count();
    assert_eq!(root_gets, 1);
    assert_eq!(b.server_kind(), ServerKind::Ollama);
}

#[test]
fn server_kind_is_other_without_base_url() {
    let b = OpenAiCompatBackend::new("gpt", Some("k"), None, 120, ReasoningLevel::Default);
    assert_eq!(b.server_kind(), ServerKind::Other);
}
