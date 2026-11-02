//! OpenAI and OpenAI-compatible backend, server detection, idle TTL and unload (Python
//! `llm/openai_compat.py`). Owner: P2-6.
//!
//! Python goes through the `openai` SDK; this reproduces its wire behaviour: `POST {base}/chat/completions`
//! and `GET {base}/models` with `Authorization: Bearer <key or "not-needed">`, a 60 s timeout and one retry
//! (connection errors, timeouts, 408/409/429/5xx).

use std::io::Read;
use std::sync::OnceLock;
use std::time::Duration;

use serde_json::{Value, json};
use stt_core::{LlmBackend, LlmError, LlmResult, ReasoningLevel, ServerKind};

use crate::base::{TEMPERATURE, TIMEOUT, split_think};

/// Where OpenAI itself lives (used when no base URL is configured).
const OPENAI_BASE: &str = "https://api.openai.com/v1";
const PROBE_TIMEOUT: Duration = Duration::from_millis(1500);
const UNLOAD_TIMEOUT: Duration = Duration::from_secs(3);
/// The SDK's `max_retries=1` back-off when the server sends no `Retry-After`.
const RETRY_DELAY: Duration = Duration::from_millis(500);
const BODY_CAP: u64 = 65536;

/// Server root (`scheme://host:port`) for a base URL such as `http://localhost:1234/v1`.
fn root(base_url: &str) -> String {
    let Some((scheme, rest)) = base_url.split_once("://") else {
        return String::new();
    };
    let end = rest.find(['/', '?', '#']).unwrap_or(rest.len());
    format!("{scheme}://{}", &rest[..end])
}

/// Port of a base URL, if it has one.
fn port_of(base_url: &str) -> Option<u16> {
    let root = root(base_url);
    let authority = root.split_once("://")?.1;
    let host = authority.rsplit('@').next()?;
    let (before, port) = host.rsplit_once(':')?;
    if !before.contains('[') || before.ends_with(']') {
        port.parse().ok()
    } else {
        None
    }
}

/// Transport-level failure of one HTTP call.
enum Transport {
    Timeout,
    Connect(String),
}

/// A finished HTTP exchange (any status).
struct Resp {
    status: u16,
    body: String,
    retry_after: Option<Duration>,
}

fn agent(timeout: Duration) -> ureq::Agent {
    ureq::Agent::config_builder()
        .http_status_as_error(false)
        .timeout_global(Some(timeout))
        .build()
        .into()
}

fn send(
    method: &str,
    url: &str,
    api_key: Option<&str>,
    body: Option<&Value>,
    timeout: Duration,
) -> Result<Resp, Transport> {
    let agent = agent(timeout);
    let auth = api_key.map(|k| format!("Bearer {k}"));
    let result = if method == "GET" {
        let mut req = agent.get(url).header("Accept", "application/json");
        if let Some(a) = &auth {
            req = req.header("Authorization", a);
        }
        req.call()
    } else {
        let mut req = agent
            .post(url)
            .header("Accept", "application/json")
            .header("Content-Type", "application/json");
        if let Some(a) = &auth {
            req = req.header("Authorization", a);
        }
        req.send(body.map(Value::to_string).unwrap_or_default())
    };
    match result {
        Ok(mut resp) => {
            let status = resp.status().as_u16();
            let retry_after = resp
                .headers()
                .get("retry-after")
                .and_then(|v| v.to_str().ok())
                .and_then(|v| v.trim().parse::<f64>().ok())
                .filter(|s| *s > 0.0 && *s <= 60.0)
                .map(Duration::from_secs_f64);
            let mut buf = Vec::new();
            // A truncated or unreadable body is treated like an empty one.
            let _ = resp
                .body_mut()
                .as_reader()
                .take(BODY_CAP)
                .read_to_end(&mut buf);
            Ok(Resp {
                status,
                body: String::from_utf8_lossy(&buf).into_owned(),
                retry_after,
            })
        }
        Err(ureq::Error::Timeout(_)) => Err(Transport::Timeout),
        Err(ureq::Error::Io(e))
            if matches!(
                e.kind(),
                std::io::ErrorKind::TimedOut | std::io::ErrorKind::WouldBlock
            ) =>
        {
            Err(Transport::Timeout)
        }
        Err(e) => Err(Transport::Connect(e.to_string())),
    }
}

/// Python `_http`: `(status, text)`, status 0 when the server was unreachable. Never fails.
fn probe(
    method: &str,
    url: &str,
    api_key: Option<&str>,
    body: Option<&Value>,
    timeout: Duration,
) -> (u16, String) {
    match send(method, url, api_key, body, timeout) {
        Ok(r) => (r.status, r.body),
        Err(_) => (0, String::new()),
    }
}

/// Detect what kind of server is behind `base_url` (Python `detect_server`). Cheap GET probes; never fails.
pub fn detect_server(base_url: &str, api_key: Option<&str>) -> ServerKind {
    let root = root(base_url);
    if root.is_empty() {
        return ServerKind::Other;
    }
    // Ollama: GET / -> "Ollama is running"; GET /api/version -> {"version": ...}
    let (_, text) = probe("GET", &format!("{root}/"), api_key, None, PROBE_TIMEOUT);
    if text.contains("Ollama is running") {
        return ServerKind::Ollama;
    }
    if text.contains("LM Studio") {
        return ServerKind::LmStudio;
    }
    let (status, text) = probe(
        "GET",
        &format!("{root}/api/version"),
        api_key,
        None,
        PROBE_TIMEOUT,
    );
    if status == 200 && text.contains("\"version\"") {
        return ServerKind::Ollama;
    }
    if text.contains("LM Studio") {
        return ServerKind::LmStudio;
    }
    // LM Studio native REST (answers 401 with an LM Studio message when a token is required)
    let (status, text) = probe(
        "GET",
        &format!("{root}/api/v0/models"),
        api_key,
        None,
        PROBE_TIMEOUT,
    );
    if text.contains("LM Studio") || (status == 200 && text.contains("\"data\"")) {
        return ServerKind::LmStudio;
    }
    if status != 0 {
        // something answered but is neither: vLLM, llama-server, ...
        return ServerKind::Other;
    }
    match port_of(base_url) {
        Some(1234) => ServerKind::LmStudio,
        Some(11434) => ServerKind::Ollama,
        _ => ServerKind::Other,
    }
}

/// Failure of one SDK-style call.
enum CallErr {
    /// HTTP status >= 400. `raw` is the response body (used for the "mentions X" checks).
    Status {
        code: u16,
        message: String,
        raw: String,
    },
    Other(LlmError),
}

impl CallErr {
    /// Python `_wrap`: human message for the UI.
    fn wrap(self, base_url: Option<&str>) -> LlmError {
        match self {
            CallErr::Other(e) => e,
            CallErr::Status { code, message, .. } => {
                let text = match code {
                    401 => "Authentication failed: check the API key".to_string(),
                    403 => "Permission denied by the API".to_string(),
                    404 => "Model or endpoint not found: check the model name".to_string(),
                    429 => "Rate limited or quota exceeded".to_string(),
                    _ => message.chars().take(200).collect(),
                };
                let _ = base_url;
                LlmError::Http {
                    status: code,
                    message: text,
                }
            }
        }
    }
}

/// The SDK's error message for a failed response: `error.message`, else the text.
fn error_message(raw: &str) -> String {
    if let Ok(v) = serde_json::from_str::<Value>(raw) {
        match v.get("error") {
            Some(Value::Object(o)) => {
                if let Some(m) = o.get("message").and_then(Value::as_str) {
                    return m.to_string();
                }
            }
            Some(Value::String(s)) => return s.clone(),
            _ => {}
        }
        if let Some(m) = v.get("message").and_then(Value::as_str) {
            return m.to_string();
        }
    }
    raw.to_string()
}

fn retryable(status: u16) -> bool {
    matches!(status, 408 | 409 | 429) || status >= 500
}

/// OpenAI (`base_url = None`) or any OpenAI-compatible server.
pub struct OpenAiCompatBackend {
    model: String,
    reasoning: ReasoningLevel,
    base_url: Option<String>,
    api_key: Option<String>,
    idle_ttl_seconds: u64,
    kind: OnceLock<ServerKind>,
}

impl OpenAiCompatBackend {
    /// No network on construction. `idle_ttl_seconds = 0` means no TTL.
    pub fn new(
        model: &str,
        api_key: Option<&str>,
        base_url: Option<&str>,
        idle_ttl_seconds: u64,
        reasoning: ReasoningLevel,
    ) -> Self {
        Self {
            model: model.to_string(),
            reasoning,
            base_url: base_url.filter(|u| !u.is_empty()).map(str::to_string),
            api_key: api_key.filter(|k| !k.is_empty()).map(str::to_string),
            idle_ttl_seconds,
            kind: OnceLock::new(),
        }
    }

    /// Test hook: skip detection and pretend the server kind is already known.
    #[doc(hidden)]
    pub fn with_server_kind(self, kind: ServerKind) -> Self {
        let _ = self.kind.set(kind);
        self
    }

    fn api_base(&self) -> String {
        self.base_url
            .as_deref()
            .unwrap_or(OPENAI_BASE)
            .trim_end_matches('/')
            .to_string()
    }

    /// The SDK always sends a key; a placeholder when none is configured.
    fn sdk_key(&self) -> &str {
        self.api_key.as_deref().unwrap_or("not-needed")
    }

    fn detected_kind(&self) -> ServerKind {
        *self.kind.get_or_init(|| match &self.base_url {
            Some(u) => detect_server(u, self.api_key.as_deref()),
            None => ServerKind::Other,
        })
    }

    /// One SDK-style call: auth header, timeout, one retry on retryable failures.
    fn call(&self, method: &str, path: &str, body: Option<&Value>) -> Result<Value, CallErr> {
        let url = format!("{}{path}", self.api_base());
        let mut attempt = 0;
        loop {
            let outcome = send(method, &url, Some(self.sdk_key()), body, TIMEOUT);
            let (err, delay) = match outcome {
                Ok(r) if r.status < 400 => {
                    return serde_json::from_str(&r.body).map_err(|e| {
                        CallErr::Other(LlmError::InvalidResponse(format!("not valid JSON: {e}")))
                    });
                }
                Ok(r) => {
                    let delay = r.retry_after.unwrap_or(RETRY_DELAY);
                    let retry = retryable(r.status);
                    let e = CallErr::Status {
                        code: r.status,
                        message: error_message(&r.body),
                        raw: r.body,
                    };
                    (e, retry.then_some(delay))
                }
                Err(Transport::Timeout) => (CallErr::Other(LlmError::Timeout), Some(RETRY_DELAY)),
                Err(Transport::Connect(m)) => (
                    CallErr::Other(LlmError::Connection {
                        url: self
                            .base_url
                            .clone()
                            .unwrap_or_else(|| "OpenAI".to_string()),
                        message: format!("is the server running? ({m})"),
                    }),
                    Some(RETRY_DELAY),
                ),
            };
            match delay {
                Some(d) if attempt < 1 => {
                    attempt += 1;
                    std::thread::sleep(d);
                }
                _ => return Err(err),
            }
        }
    }

    /// `reasoning_effort` candidates, tried in order. low/medium/high pass through; "off" -> "none"
    /// (OpenAI gpt-5.1+, Ollama; official OpenAI also tries "minimal" for older gpt-5 models). Servers
    /// that reject the field fall through to `None` (nothing sent) and the result notes it.
    fn efforts(&self) -> Vec<Option<&'static str>> {
        match self.reasoning {
            ReasoningLevel::Default => vec![None],
            ReasoningLevel::Off if self.base_url.is_some() => vec![Some("none")],
            ReasoningLevel::Off => vec![Some("none"), Some("minimal")],
            ReasoningLevel::Low => vec![Some("low")],
            ReasoningLevel::Medium => vec![Some("medium")],
            ReasoningLevel::High => vec![Some("high")],
        }
    }

    fn create(&self, body: &Value) -> Result<Value, CallErr> {
        let mut with_temp = body.clone();
        with_temp["temperature"] = json!(TEMPERATURE.to_string().parse::<f64>().unwrap_or(0.2));
        match self.call("POST", "/chat/completions", Some(&with_temp)) {
            Err(CallErr::Status {
                code: 400,
                raw,
                message,
            }) if format!("{message} {raw}")
                .to_lowercase()
                .contains("temperature") =>
            {
                // some models (e.g. reasoning ones) only accept the default temperature
                self.call("POST", "/chat/completions", Some(body))
            }
            other => other,
        }
    }

    fn set_ollama_keep_alive(&self) {
        // Ollama's /v1 endpoint does not document keep_alive; the native API does. An /api/generate call
        // without a prompt only updates the loaded model's expiry (best effort, errors ignored).
        if let Some(u) = &self.base_url {
            probe(
                "POST",
                &format!("{}/api/generate", root(u)),
                self.api_key.as_deref(),
                Some(&json!({"model": self.model, "keep_alive": self.idle_ttl_seconds})),
                UNLOAD_TIMEOUT,
            );
        }
    }

    fn parse(r: &Value, note: &str) -> LlmResult {
        let msg = r.pointer("/choices/0/message");
        let content = msg
            .and_then(|m| m.get("content"))
            .and_then(Value::as_str)
            .unwrap_or("");
        // LM Studio / vLLM / llama-server: reasoning_content; Ollama /v1 and others: reasoning
        let mut reasoning = String::new();
        for name in ["reasoning_content", "reasoning"] {
            if let Some(v) = msg.and_then(|m| m.get(name)).and_then(Value::as_str)
                && !v.trim().is_empty()
            {
                reasoning = v.trim().to_string();
                break;
            }
        }
        let (answer, inline) = split_think(content);
        if !inline.is_empty() {
            reasoning = if reasoning.is_empty() {
                inline
            } else {
                format!("{reasoning}\n\n{inline}").trim().to_string()
            };
        }
        let tokens = r
            .pointer("/usage/completion_tokens_details/reasoning_tokens")
            .and_then(Value::as_u64);
        let reasoned = if !reasoning.is_empty() || tokens.unwrap_or(0) > 0 {
            Some(true)
        } else if tokens == Some(0) {
            Some(false)
        } else {
            None
        };
        LlmResult {
            text: answer,
            reasoning,
            reasoning_tokens: tokens,
            reasoned,
            note: note.to_string(),
        }
    }
}

impl LlmBackend for OpenAiCompatBackend {
    fn complete(&self, system_prompt: &str, text: &str) -> Result<LlmResult, LlmError> {
        if self.model.is_empty() {
            return Err(LlmError::InvalidResponse("No model selected".into()));
        }
        let body = json!({
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": text},
            ],
        });
        let use_ttl = self.idle_ttl_seconds > 0
            && self.base_url.is_some()
            && self.detected_kind() == ServerKind::LmStudio;
        let mut note = "";
        let mut candidates = self.efforts();
        if candidates != [None] {
            candidates.push(None);
        }
        let mut result = None;
        for (i, effort) in candidates.iter().enumerate() {
            let mut b = body.clone();
            if let Some(e) = effort {
                b["reasoning_effort"] = json!(e);
            }
            if use_ttl {
                b["ttl"] = json!(self.idle_ttl_seconds); // seconds; LM Studio applies it to JIT loads
            }
            match self.create(&b) {
                Ok(r) => {
                    result = Some(r);
                    break;
                }
                Err(CallErr::Status {
                    code: 400,
                    message,
                    raw,
                }) => {
                    let s = format!("{message} {raw}").to_lowercase();
                    if effort.is_none() || (!s.contains("reasoning") && !s.contains("effort")) {
                        return Err(CallErr::Status {
                            code: 400,
                            message,
                            raw,
                        }
                        .wrap(self.base_url.as_deref()));
                    }
                    if candidates.get(i + 1) == Some(&None) {
                        note = "server ignored reasoning setting";
                    }
                }
                Err(e) => return Err(e.wrap(self.base_url.as_deref())),
            }
        }
        let r = result.ok_or_else(|| LlmError::InvalidResponse("no response".into()))?;
        if self.idle_ttl_seconds > 0
            && self.base_url.is_some()
            && self.detected_kind() == ServerKind::Ollama
        {
            self.set_ollama_keep_alive();
        }
        Ok(Self::parse(&r, note))
    }

    fn list_models(&self) -> Result<Vec<String>, LlmError> {
        let v = self
            .call("GET", "/models", None)
            .map_err(|e| e.wrap(self.base_url.as_deref()))?;
        let mut ids: Vec<String> = v
            .get("data")
            .and_then(Value::as_array)
            .map(|a| {
                a.iter()
                    .filter_map(|m| m.get("id").and_then(Value::as_str).map(str::to_string))
                    .collect()
            })
            .unwrap_or_default();
        ids.sort();
        Ok(ids)
    }

    /// Ask LM Studio / Ollama to unload the model now. Other servers: no-op.
    fn unload(&self) -> Result<(), LlmError> {
        let Some(base) = &self.base_url else {
            return Ok(());
        };
        if self.model.is_empty() {
            return Ok(());
        }
        let root = root(base);
        let (kind, path, body) = match self.detected_kind() {
            ServerKind::LmStudio => (
                "lmstudio",
                "/api/v1/models/unload",
                json!({"instance_id": self.model}),
            ),
            ServerKind::Ollama => (
                "ollama",
                "/api/generate",
                json!({"model": self.model, "keep_alive": 0}),
            ),
            ServerKind::Other => return Ok(()),
        };
        let (status, _) = probe(
            "POST",
            &format!("{root}{path}"),
            self.api_key.as_deref(),
            Some(&body),
            UNLOAD_TIMEOUT,
        );
        if status == 0 {
            return Err(LlmError::Connection {
                url: root,
                message: "could not reach the server to unload the model".into(),
            });
        }
        if status >= 400 {
            return Err(LlmError::Http {
                status,
                message: format!("Unload failed ({kind})"),
            });
        }
        Ok(())
    }

    fn server_kind(&self) -> ServerKind {
        if self.base_url.is_some() {
            self.detected_kind()
        } else {
            ServerKind::Other
        }
    }
}
