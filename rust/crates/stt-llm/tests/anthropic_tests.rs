//! Anthropic backend against a local fake HTTP server (never the real API). Port of the Anthropic part
//! of `tests/test_llm_reasoning.py`, asserting on the exact JSON bodies and headers sent.

use std::io::{Read, Write};
use std::net::TcpListener;
use std::sync::{Arc, Mutex};
use std::thread;

use serde_json::{Value, json};
use stt_core::{LlmBackend, LlmError, ReasoningLevel};
use stt_llm::anthropic::AnthropicBackend;

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
        json!({"content": [{"type": "text", "text": text}]}).to_string(),
    )
}

fn bad_request(msg: &str) -> (u16, String) {
    (
        400,
        json!({"type": "error", "error": {"type": "invalid_request_error", "message": msg}})
            .to_string(),
    )
}

fn backend(url: &str, model: &str, lvl: ReasoningLevel) -> AnthropicBackend {
    AnthropicBackend::new(model, "sk-test", lvl).with_base_url(url)
}

#[test]
fn default_level_body_and_headers() {
    let (url, seen) = serve(vec![ok("  hello \n")]);
    let r = backend(&url, "m", ReasoningLevel::Default)
        .complete("sys", "txt")
        .unwrap();
    assert_eq!((r.text.as_str(), r.reasoning.as_str()), ("hello", ""));
    assert_eq!((r.reasoned, r.reasoning_tokens), (None, None));
    let req = &seen.lock().unwrap()[0];
    assert!(req.request_line.starts_with("POST /v1/messages "));
    assert_eq!(req.header("x-api-key"), Some("sk-test"));
    assert_eq!(req.header("anthropic-version"), Some("2023-06-01"));
    assert_eq!(req.header("content-type"), Some("application/json"));
    assert_eq!(
        req.body,
        json!({
            "model": "m", "max_tokens": 4096, "system": "sys",
            "messages": [{"role": "user", "content": "txt"}],
            "temperature": 0.2,
        })
    );
}

#[test]
fn off_level_sends_disabled_thinking_without_temperature() {
    let (url, seen) = serve(vec![ok("ok")]);
    backend(&url, "m", ReasoningLevel::Off)
        .complete("s", "t")
        .unwrap();
    let body = seen.lock().unwrap()[0].body.clone();
    assert_eq!(body["thinking"], json!({"type": "disabled"}));
    assert!(body.get("temperature").is_none());
}

#[test]
fn medium_level_sends_adaptive_thinking_and_effort() {
    let (url, seen) = serve(vec![ok("ok")]);
    backend(&url, "m", ReasoningLevel::Medium)
        .complete("s", "t")
        .unwrap();
    let body = seen.lock().unwrap()[0].body.clone();
    assert_eq!(
        body["thinking"],
        json!({"type": "adaptive", "display": "summarized"})
    );
    assert_eq!(body["output_config"], json!({"effort": "medium"}));
    assert_eq!(body["max_tokens"], 4096);
    assert!(body.get("temperature").is_none());
}

#[test]
fn falls_back_to_budget_then_none() {
    let (url, seen) = serve(vec![
        bad_request("adaptive thinking is not supported on this model"),
        ok("ok"),
    ]);
    let r = backend(&url, "old", ReasoningLevel::High)
        .complete("s", "t")
        .unwrap();
    assert_eq!((r.text.as_str(), r.note.as_str()), ("ok", ""));
    {
        let reqs = seen.lock().unwrap();
        assert_eq!(reqs.len(), 2);
        assert_eq!(
            reqs[1].body["thinking"],
            json!({"type": "enabled", "budget_tokens": 8192, "display": "summarized"})
        );
        assert_eq!(reqs[1].body["max_tokens"], 4096 + 8192);
        assert!(reqs[1].body.get("output_config").is_none());
    }

    let (url, seen) = serve(vec![
        bad_request("thinking is not allowed"),
        bad_request("thinking is not allowed"),
        ok("ok"),
    ]);
    let r = backend(&url, "old", ReasoningLevel::Low)
        .complete("s", "t")
        .unwrap();
    assert_eq!(r.note, "model rejected the reasoning setting");
    let reqs = seen.lock().unwrap();
    assert_eq!(reqs.len(), 3);
    assert_eq!(reqs[1].body["thinking"]["budget_tokens"], 1024);
    assert!(reqs[2].body.get("thinking").is_none());
    assert_eq!(reqs[2].body["temperature"], 0.2);
}

#[test]
fn off_level_tries_between_tools_then_none() {
    let (url, seen) = serve(vec![
        bad_request("thinking disabled not allowed"),
        bad_request("thinking type invalid"),
        ok("ok"),
    ]);
    let r = backend(&url, "m", ReasoningLevel::Off)
        .complete("s", "t")
        .unwrap();
    assert_eq!(r.note, "model rejected the reasoning setting");
    let reqs = seen.lock().unwrap();
    assert_eq!(reqs[1].body["thinking"], json!({"type": "between_tools"}));
    assert!(reqs[2].body.get("thinking").is_none());
}

#[test]
fn drops_temperature_on_400() {
    let (url, seen) = serve(vec![
        bad_request("temperature is not supported on this model"),
        ok("ok"),
    ]);
    let r = backend(&url, "m", ReasoningLevel::Default)
        .complete("s", "t")
        .unwrap();
    assert_eq!(r.text, "ok");
    let reqs = seen.lock().unwrap();
    assert!(reqs[0].body.get("temperature").is_some());
    assert!(reqs[1].body.get("temperature").is_none());
}

#[test]
fn unrelated_400_is_not_retried() {
    let (url, seen) = serve(vec![bad_request("messages: bad shape")]);
    let err = backend(&url, "m", ReasoningLevel::High)
        .complete("s", "t")
        .unwrap_err();
    assert!(
        matches!(&err, LlmError::Http { status: 400, message } if message == "Anthropic API error 400: messages: bad shape"),
        "{err:?}"
    );
    assert_eq!(seen.lock().unwrap().len(), 1);
}

#[test]
fn thinking_blocks_go_to_reasoning_only() {
    let body = json!({
        "content": [
            {"type": "thinking", "thinking": "step 1"},
            {"type": "text", "text": "Final."},
            {"type": "redacted_thinking"},
            {"type": "thinking", "thinking": "step 2"},
        ],
        "usage": {"output_tokens_details": {"thinking_tokens": 12}},
    });
    let (url, _) = serve(vec![(200, body.to_string())]);
    let r = backend(&url, "m", ReasoningLevel::High)
        .complete("s", "t")
        .unwrap();
    assert_eq!(r.text, "Final.");
    assert_eq!(r.reasoning, "step 1\n\nstep 2");
    assert_eq!((r.reasoning_tokens, r.reasoned), (Some(12), Some(true)));
}

#[test]
fn thinking_block_without_token_count_means_reasoned() {
    let body = json!({"content": [
        {"type": "redacted_thinking"}, {"type": "text", "text": "A"}]});
    let (url, _) = serve(vec![(200, body.to_string())]);
    let r = backend(&url, "m", ReasoningLevel::High)
        .complete("s", "t")
        .unwrap();
    assert_eq!((r.text.as_str(), r.reasoning.as_str()), ("A", ""));
    assert_eq!((r.reasoned, r.reasoning_tokens), (Some(true), None));
}

#[test]
fn zero_thinking_tokens_means_not_reasoned() {
    let body = json!({"content": [{"type": "text", "text": "A"}],
        "usage": {"output_tokens_details": {"thinking_tokens": 0}}});
    let (url, _) = serve(vec![(200, body.to_string())]);
    let r = backend(&url, "m", ReasoningLevel::Default)
        .complete("s", "t")
        .unwrap();
    assert_eq!((r.reasoned, r.reasoning_tokens), (Some(false), Some(0)));
}

#[test]
fn error_statuses_use_human_messages() {
    for (status, want) in [
        (401, "Authentication failed: check the Anthropic API key"),
        (403, "Permission denied by the Anthropic API"),
        (404, "Model not found: check the model name"),
        (400, "Anthropic API error 400: nope"),
        (418, "Anthropic API error 418: nope"),
    ] {
        let body = json!({"error": {"message": "nope"}}).to_string();
        let (url, _) = serve(vec![(status, body)]);
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
fn rate_limit_retries_once_then_reports() {
    let rl = (429, json!({"error": {"message": "slow"}}).to_string());
    let (url, seen) = serve(vec![rl.clone(), rl.clone()]);
    let err = backend(&url, "m", ReasoningLevel::Default)
        .complete("s", "t")
        .unwrap_err();
    assert!(
        matches!(&err, LlmError::Http { status: 429, message } if message == "Rate limited or quota exceeded")
    );
    assert_eq!(seen.lock().unwrap().len(), 2);

    let (url, _) = serve(vec![rl, ok("fine")]);
    let r = backend(&url, "m", ReasoningLevel::Default)
        .complete("s", "t")
        .unwrap();
    assert_eq!(r.text, "fine");
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
fn list_models_sorted_with_headers() {
    let body = json!({"data": [{"id": "claude-b"}, {"id": "claude-a"}]}).to_string();
    let (url, seen) = serve(vec![(200, body)]);
    let ids = backend(&url, "m", ReasoningLevel::Default)
        .list_models()
        .unwrap();
    assert_eq!(ids, ["claude-a", "claude-b"]);
    let req = &seen.lock().unwrap()[0];
    assert!(req.request_line.starts_with("GET /v1/models?limit=1000 "));
    assert_eq!(req.header("x-api-key"), Some("sk-test"));
    assert_eq!(req.header("anthropic-version"), Some("2023-06-01"));
}

#[test]
fn list_models_auth_error() {
    let (url, _) = serve(vec![(
        401,
        json!({"error": {"message": "bad"}}).to_string(),
    )]);
    let err = backend(&url, "m", ReasoningLevel::Default)
        .list_models()
        .unwrap_err();
    assert!(err.to_string().contains("check the Anthropic API key"));
}
