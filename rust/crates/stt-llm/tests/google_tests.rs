//! Google backend against a local fake HTTP server (never the real API). Port of the Google part of
//! `tests/test_llm_reasoning.py`, asserting on the exact JSON bodies and headers sent.

use std::io::{Read, Write};
use std::net::TcpListener;
use std::sync::{Arc, Mutex};
use std::thread;

use serde_json::{Value, json};
use stt_core::{LlmBackend, LlmError, ReasoningLevel};
use stt_llm::google::GoogleBackend;

#[derive(Debug, Clone)]
struct Req {
    request_line: String,
    headers: Vec<(String, String)>,
    body: Value,
}

impl Req {
    fn header(&self, name: &str) -> Option<&str> {
        self.headers
            .iter()
            .find(|(k, _)| k.eq_ignore_ascii_case(name))
            .map(|(_, v)| v.as_str())
    }
}

/// Serves the scripted `(status, body)` replies in order, one per connection, and records requests.
fn serve(replies: Vec<(u16, String)>) -> (String, Arc<Mutex<Vec<Req>>>) {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let url = format!("http://{}", listener.local_addr().unwrap());
    let seen = Arc::new(Mutex::new(Vec::new()));
    let log = seen.clone();
    thread::spawn(move || {
        for (status, body) in replies {
            let Ok((mut s, _)) = listener.accept() else {
                return;
            };
            let mut buf = Vec::new();
            let mut chunk = [0u8; 4096];
            let (head_end, len) = loop {
                let n = s.read(&mut chunk).unwrap_or(0);
                buf.extend_from_slice(&chunk[..n]);
                if let Some(p) = buf.windows(4).position(|w| w == b"\r\n\r\n") {
                    let head = String::from_utf8_lossy(&buf[..p]).to_lowercase();
                    let len = head
                        .lines()
                        .find_map(|l| l.strip_prefix("content-length:"))
                        .and_then(|v| v.trim().parse::<usize>().ok())
                        .unwrap_or(0);
                    break (p + 4, len);
                }
                if n == 0 {
                    return;
                }
            };
            while buf.len() < head_end + len {
                let n = s.read(&mut chunk).unwrap_or(0);
                if n == 0 {
                    break;
                }
                buf.extend_from_slice(&chunk[..n]);
            }
            let head = String::from_utf8_lossy(&buf[..head_end - 4]).to_string();
            let mut lines = head.lines();
            let request_line = lines.next().unwrap_or("").to_string();
            let headers = lines
                .filter_map(|l| l.split_once(':'))
                .map(|(k, v)| (k.trim().to_string(), v.trim().to_string()))
                .collect();
            let parsed = serde_json::from_slice(&buf[head_end..]).unwrap_or(Value::Null);
            log.lock().unwrap().push(Req {
                request_line,
                headers,
                body: parsed,
            });
            let resp = format!(
                "HTTP/1.1 {status} X\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
                body.len()
            );
            let _ = s.write_all(resp.as_bytes());
        }
    });
    (url, seen)
}

fn ok(text: &str) -> (u16, String) {
    (
        200,
        json!({"candidates": [{"content": {"parts": [{"text": text}]}}]}).to_string(),
    )
}

fn api_error(code: u16, msg: &str) -> (u16, String) {
    (
        code,
        json!({"error": {"code": code, "message": msg, "status": "X"}}).to_string(),
    )
}

fn backend(url: &str, model: &str, lvl: ReasoningLevel) -> GoogleBackend {
    GoogleBackend::new(model, "g-key", lvl).with_base_url(url)
}

fn thinking_sent(model: &str, lvl: ReasoningLevel) -> Option<Value> {
    let (url, seen) = serve(vec![ok("ok")]);
    backend(&url, model, lvl).complete("s", "t").unwrap();
    let body = seen.lock().unwrap()[0].body.clone();
    body["generationConfig"].get("thinkingConfig").cloned()
}

#[test]
fn default_level_body_and_headers() {
    let (url, seen) = serve(vec![ok(" Answer. ")]);
    let r = backend(&url, "gemini-2.5-flash", ReasoningLevel::Default)
        .complete("sys", "txt")
        .unwrap();
    assert_eq!((r.text.as_str(), r.reasoning.as_str()), ("Answer.", ""));
    assert_eq!((r.reasoned, r.reasoning_tokens), (None, None));
    let req = &seen.lock().unwrap()[0];
    assert!(
        req.request_line
            .starts_with("POST /v1beta/models/gemini-2.5-flash:generateContent ")
    );
    assert_eq!(req.header("x-goog-api-key"), Some("g-key"));
    assert_eq!(req.header("content-type"), Some("application/json"));
    assert_eq!(
        req.body,
        json!({
            "contents": [{"role": "user", "parts": [{"text": "txt"}]}],
            "systemInstruction": {"parts": [{"text": "sys"}]},
            "generationConfig": {"temperature": 0.2},
        })
    );
}

#[test]
fn thinking_config_per_model_and_level() {
    let cases = [
        (
            "gemini-2.5-flash",
            ReasoningLevel::Off,
            json!({"thinkingBudget": 0, "includeThoughts": false}),
        ),
        (
            "gemini-2.5-pro",
            ReasoningLevel::Off,
            json!({"thinkingBudget": 128, "includeThoughts": false}),
        ),
        (
            "gemini-2.5-flash",
            ReasoningLevel::Low,
            json!({"thinkingBudget": 1024, "includeThoughts": true}),
        ),
        (
            "gemini-2.5-flash",
            ReasoningLevel::Medium,
            json!({"thinkingBudget": 8192, "includeThoughts": true}),
        ),
        (
            "gemini-2.5-flash",
            ReasoningLevel::High,
            json!({"thinkingBudget": 24576, "includeThoughts": true}),
        ),
        (
            "gemini-3-pro-preview",
            ReasoningLevel::Low,
            json!({"thinkingLevel": "LOW", "includeThoughts": true}),
        ),
        (
            "gemini-3-pro-preview",
            ReasoningLevel::Medium,
            json!({"thinkingLevel": "MEDIUM", "includeThoughts": true}),
        ),
        (
            "gemini-3-pro-preview",
            ReasoningLevel::High,
            json!({"thinkingLevel": "HIGH", "includeThoughts": true}),
        ),
        (
            "gemini-3-flash",
            ReasoningLevel::Off,
            json!({"thinkingLevel": "MINIMAL", "includeThoughts": false}),
        ),
    ];
    for (model, lvl, want) in cases {
        assert_eq!(thinking_sent(model, lvl), Some(want), "{model} {lvl:?}");
    }
    assert_eq!(
        thinking_sent("gemini-2.5-pro", ReasoningLevel::Default),
        None
    );
}

#[test]
fn retry_without_thinking_on_400() {
    let (url, seen) = serve(vec![
        api_error(400, "thinking budget not supported"),
        ok("ok"),
    ]);
    let r = backend(&url, "gemini-2.5-pro", ReasoningLevel::Off)
        .complete("s", "t")
        .unwrap();
    assert_eq!(r.text, "ok");
    assert_eq!(r.note, "model rejected the reasoning setting");
    let reqs = seen.lock().unwrap();
    assert_eq!(reqs.len(), 2);
    assert!(reqs[0].body["generationConfig"]["thinkingConfig"].is_object());
    assert!(
        reqs[1].body["generationConfig"]
            .get("thinkingConfig")
            .is_none()
    );
}

#[test]
fn unrelated_400_is_not_retried() {
    let (url, seen) = serve(vec![api_error(400, "bad contents")]);
    let err = backend(&url, "gemini-2.5-pro", ReasoningLevel::Off)
        .complete("s", "t")
        .unwrap_err();
    assert!(
        matches!(&err, LlmError::Http { status: 400, message } if message == "Google API error 400: bad contents"),
        "{err:?}"
    );
    assert_eq!(seen.lock().unwrap().len(), 1);
}

#[test]
fn thought_parts_go_to_reasoning_only() {
    let body = json!({
        "candidates": [{"content": {"parts": [
            {"text": " considering ", "thought": true},
            {"text": "Answer."},
            {"text": "more", "thought": true},
        ]}}],
        "usageMetadata": {"thoughtsTokenCount": 33},
    });
    let (url, _) = serve(vec![(200, body.to_string())]);
    let r = backend(&url, "gemini-2.5-flash", ReasoningLevel::High)
        .complete("s", "t")
        .unwrap();
    assert_eq!(r.text, "Answer.");
    assert_eq!(r.reasoning, "considering\n\nmore");
    assert_eq!((r.reasoning_tokens, r.reasoned), (Some(33), Some(true)));
}

#[test]
fn no_thoughts_with_usage_means_not_reasoned() {
    let body = json!({
        "candidates": [{"content": {"parts": [{"text": "A"}]}}],
        "usageMetadata": {"promptTokenCount": 3},
    });
    let (url, _) = serve(vec![(200, body.to_string())]);
    let r = backend(&url, "gemini-2.5-flash", ReasoningLevel::Default)
        .complete("s", "t")
        .unwrap();
    assert_eq!((r.text.as_str(), r.reasoned), ("A", Some(false)));
    assert_eq!(r.reasoning_tokens, None);
}

#[test]
fn error_statuses_use_human_messages() {
    for (status, msg, want) in [
        (
            401,
            "nope",
            "Authentication failed: check the Google API key",
        ),
        (
            403,
            "nope",
            "Authentication failed: check the Google API key",
        ),
        (
            400,
            "API key not valid",
            "Authentication failed: check the Google API key",
        ),
        (404, "nope", "Model not found: check the model name"),
        (429, "nope", "Rate limited or quota exceeded"),
        (408, "nope", "Request to Google timed out"),
        (504, "nope", "Request to Google timed out"),
        (500, "boom", "Google API error 500: boom"),
    ] {
        let (url, _) = serve(vec![api_error(status, msg)]);
        let err = backend(&url, "m", ReasoningLevel::Default)
            .complete("s", "t")
            .unwrap_err();
        assert!(
            matches!(&err, LlmError::Http { status: s, message } if *s == status && message == want),
            "{status}: {err:?}"
        );
    }
}

#[test]
fn unreachable_server_is_a_connection_error() {
    let url = {
        let l = TcpListener::bind("127.0.0.1:0").unwrap();
        format!("http://{}", l.local_addr().unwrap())
    };
    let err = backend(&url, "m", ReasoningLevel::Default)
        .complete("s", "t")
        .unwrap_err();
    assert!(matches!(err, LlmError::Connection { .. }), "{err:?}");
}

#[test]
fn empty_model_is_rejected_without_a_request() {
    let (url, seen) = serve(vec![]);
    let err = backend(&url, "", ReasoningLevel::Default)
        .complete("s", "t")
        .unwrap_err();
    assert!(err.to_string().contains("No model selected"));
    assert!(seen.lock().unwrap().is_empty());
}

#[test]
fn list_models_filters_strips_prefix_and_sorts() {
    let body = json!({"models": [
        {"name": "models/gemini-b", "supportedGenerationMethods": ["generateContent"]},
        {"name": "models/embed-1", "supportedGenerationMethods": ["embedContent"]},
        {"name": "models/gemini-a", "supportedGenerationMethods": ["generateContent", "countTokens"]},
        {"name": "models/no-actions"},
    ]});
    let (url, seen) = serve(vec![(200, body.to_string())]);
    let ids = backend(&url, "m", ReasoningLevel::Default)
        .list_models()
        .unwrap();
    assert_eq!(ids, ["gemini-a", "gemini-b", "no-actions"]);
    let req = &seen.lock().unwrap()[0];
    assert!(
        req.request_line
            .starts_with("GET /v1beta/models?pageSize=1000 ")
    );
    assert_eq!(req.header("x-goog-api-key"), Some("g-key"));
}

#[test]
fn list_models_follows_pages() {
    let p1 = json!({"models": [{"name": "models/b"}], "nextPageToken": "tok"});
    let p2 = json!({"models": [{"name": "models/a"}]});
    let (url, seen) = serve(vec![(200, p1.to_string()), (200, p2.to_string())]);
    let ids = backend(&url, "m", ReasoningLevel::Default)
        .list_models()
        .unwrap();
    assert_eq!(ids, ["a", "b"]);
    assert!(
        seen.lock().unwrap()[1]
            .request_line
            .contains("pageToken=tok")
    );
}

#[test]
fn list_models_auth_error() {
    let (url, _) = serve(vec![api_error(403, "denied")]);
    let err = backend(&url, "m", ReasoningLevel::Default)
        .list_models()
        .unwrap_err();
    assert!(err.to_string().contains("check the Google API key"));
}
