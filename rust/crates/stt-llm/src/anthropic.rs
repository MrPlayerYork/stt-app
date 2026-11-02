//! Anthropic backend over raw HTTPS (Python `llm/anthropic_backend.py`). Owner: P2-7.
//!
//! Wire behaviour of the vendor SDK is reproduced by hand: `x-api-key` + `anthropic-version` headers, one
//! retry on 408/409/429/5xx and connection errors (SDK `max_retries=1`), and the reasoning "variants" fall
//! back in the same order as the Python `_variants`.

use serde_json::{Map, Value, json};
use stt_core::{LlmBackend, LlmError, LlmResult, ReasoningLevel};

use crate::base::{TEMPERATURE, TIMEOUT};

const DEFAULT_BASE_URL: &str = "https://api.anthropic.com";
const API_VERSION: &str = "2023-06-01";
const MAX_TOKENS: u64 = 4096;
const REJECTED_NOTE: &str = "model rejected the reasoning setting";

/// Anthropic backend (Messages API).
pub struct AnthropicBackend {
    model: String,
    api_key: String,
    reasoning: ReasoningLevel,
    base_url: String,
    agent: ureq::Agent,
}

/// Outcome of one HTTP exchange: status and body text.
struct Reply {
    status: u16,
    body: String,
}

impl AnthropicBackend {
    /// No network on construction.
    pub fn new(model: &str, api_key: &str, reasoning: ReasoningLevel) -> Self {
        let agent: ureq::Agent = ureq::Agent::config_builder()
            .timeout_global(Some(TIMEOUT))
            .http_status_as_error(false)
            .build()
            .into();
        Self {
            model: model.to_string(),
            api_key: api_key.to_string(),
            reasoning,
            base_url: DEFAULT_BASE_URL.to_string(),
            agent,
        }
    }

    /// Point the backend at another server (tests). A trailing `/` is ignored.
    pub fn with_base_url(mut self, url: &str) -> Self {
        self.base_url = url.trim_end_matches('/').to_string();
        self
    }

    /// Extra request fields per attempt, in fallback order (Python `_variants`).
    fn variants(&self) -> Vec<Map<String, Value>> {
        let obj = |v: Value| match v {
            Value::Object(m) => m,
            _ => Map::new(),
        };
        let budget = match self.reasoning {
            ReasoningLevel::Low => 1024,
            ReasoningLevel::Medium => 4096,
            _ => 8192,
        };
        match self.reasoning {
            ReasoningLevel::Default => vec![Map::new()],
            ReasoningLevel::Off => vec![
                obj(json!({"thinking": {"type": "disabled"}})),
                obj(json!({"thinking": {"type": "between_tools"}})),
                Map::new(),
            ],
            lvl => vec![
                obj(json!({
                    "thinking": {"type": "adaptive", "display": "summarized"},
                    "output_config": {"effort": lvl.as_str()},
                })),
                obj(json!({
                    "thinking": {"type": "enabled", "budget_tokens": budget, "display": "summarized"},
                    "max_tokens": MAX_TOKENS + budget,
                })),
                Map::new(),
            ],
        }
    }

    /// One POST/GET with the SDK's single retry on transient failures.
    fn send(&self, path: &str, body: Option<&Value>) -> Result<Reply, LlmError> {
        let url = format!("{}{}", self.base_url, path);
        let mut last = None;
        for attempt in 0..2 {
            let result = match body {
                Some(b) => self
                    .agent
                    .post(&url)
                    .header("x-api-key", &self.api_key)
                    .header("anthropic-version", API_VERSION)
                    .header("content-type", "application/json")
                    .send(b.to_string()),
                None => self
                    .agent
                    .get(&url)
                    .header("x-api-key", &self.api_key)
                    .header("anthropic-version", API_VERSION)
                    .call(),
            };
            match result {
                Ok(resp) => {
                    let status = resp.status().as_u16();
                    let text = resp
                        .into_body()
                        .read_to_string()
                        .map_err(|e| transport_error(&e, &self.base_url))?;
                    if attempt == 0 && matches!(status, 408 | 409 | 429 | 500..=599) {
                        last = Some(Reply { status, body: text });
                        continue;
                    }
                    return Ok(Reply { status, body: text });
                }
                Err(e) => {
                    if attempt == 0 {
                        last = None;
                        let err = transport_error(&e, &self.base_url);
                        if matches!(err, LlmError::Timeout | LlmError::Connection { .. }) {
                            continue;
                        }
                        return Err(err);
                    }
                    return Err(transport_error(&e, &self.base_url));
                }
            }
        }
        last.ok_or_else(|| LlmError::InvalidResponse("no response".into()))
    }

    fn post_messages(&self, kw: &Map<String, Value>) -> Result<Reply, LlmError> {
        self.send("/v1/messages", Some(&Value::Object(kw.clone())))
    }
}

fn transport_error(e: &ureq::Error, url: &str) -> LlmError {
    match e {
        ureq::Error::Timeout(_) => LlmError::Timeout,
        other => LlmError::Connection {
            url: url.to_string(),
            message: other.to_string(),
        },
    }
}

/// The API's error message (`error.message`), or the raw body if it isn't the usual JSON.
fn error_message(body: &str) -> String {
    serde_json::from_str::<Value>(body)
        .ok()
        .and_then(|v| v["error"]["message"].as_str().map(str::to_string))
        .unwrap_or_else(|| body.trim().to_string())
}

/// Map an error status to the human message (Python `_wrap`).
fn wrap_status(status: u16, body: &str) -> LlmError {
    let message = match status {
        401 => "Authentication failed: check the Anthropic API key".to_string(),
        403 => "Permission denied by the Anthropic API".to_string(),
        404 => "Model not found: check the model name".to_string(),
        429 => "Rate limited or quota exceeded".to_string(),
        _ => {
            let m: String = error_message(body).chars().take(200).collect();
            return LlmError::Http {
                status,
                message: format!("Anthropic API error {status}: {m}"),
            };
        }
    };
    LlmError::Http { status, message }
}

fn parse_response(body: &str, note: &str) -> Result<LlmResult, LlmError> {
    let v: Value = serde_json::from_str(body)
        .map_err(|e| LlmError::InvalidResponse(format!("not JSON: {e}")))?;
    let blocks = v["content"]
        .as_array()
        .ok_or_else(|| LlmError::InvalidResponse("no content in response".into()))?;
    let (mut answer, mut thoughts) = (String::new(), Vec::new());
    let mut has_block = false;
    for b in blocks {
        match b["type"].as_str() {
            Some("text") => answer.push_str(b["text"].as_str().unwrap_or("")),
            Some("thinking") => {
                has_block = true;
                if let Some(t) = b["thinking"].as_str().filter(|t| !t.is_empty()) {
                    thoughts.push(t.to_string());
                }
            }
            Some("redacted_thinking") => has_block = true,
            _ => {}
        }
    }
    let tokens = v["usage"]["output_tokens_details"]["thinking_tokens"].as_u64();
    let reasoned = match tokens {
        Some(t) => Some(t > 0),
        None if has_block => Some(true),
        None => None,
    };
    Ok(LlmResult {
        text: answer.trim().to_string(),
        reasoning: thoughts.join("\n\n").trim().to_string(),
        reasoning_tokens: tokens,
        reasoned,
        note: note.to_string(),
    })
}

impl LlmBackend for AnthropicBackend {
    fn complete(&self, system_prompt: &str, text: &str) -> Result<LlmResult, LlmError> {
        if self.model.is_empty() {
            return Err(LlmError::InvalidResponse("No model selected".into()));
        }
        let variants = self.variants();
        let mut note = "";
        for (i, extra) in variants.iter().enumerate() {
            let mut kw = Map::new();
            kw.insert("model".into(), json!(self.model));
            kw.insert("max_tokens".into(), json!(MAX_TOKENS));
            kw.insert("system".into(), json!(system_prompt));
            kw.insert(
                "messages".into(),
                json!([{"role": "user", "content": text}]),
            );
            for (k, v) in extra {
                kw.insert(k.clone(), v.clone());
            }
            if !extra.contains_key("thinking") {
                kw.insert("temperature".into(), temperature());
            }
            let mut reply = self.post_messages(&kw)?;
            if reply.status == 400
                && kw.contains_key("temperature")
                && reply.body.to_lowercase().contains("temperature")
            {
                kw.remove("temperature");
                reply = self.post_messages(&kw)?;
            }
            if reply.status == 400 {
                let low = reply.body.to_lowercase();
                let thinking_related =
                    low.contains("thinking") || low.contains("effort") || low.contains("budget");
                if i + 1 < variants.len() && thinking_related {
                    if variants[i + 1].is_empty() {
                        note = REJECTED_NOTE;
                    }
                    continue;
                }
            }
            if reply.status >= 300 {
                return Err(wrap_status(reply.status, &reply.body));
            }
            return parse_response(&reply.body, note);
        }
        Err(LlmError::InvalidResponse("no response".into()))
    }

    fn list_models(&self) -> Result<Vec<String>, LlmError> {
        let reply = self.send("/v1/models?limit=1000", None)?;
        if reply.status >= 300 {
            return Err(wrap_status(reply.status, &reply.body));
        }
        let v: Value = serde_json::from_str(&reply.body)
            .map_err(|e| LlmError::InvalidResponse(format!("not JSON: {e}")))?;
        let mut ids: Vec<String> = v["data"]
            .as_array()
            .map(|a| {
                a.iter()
                    .filter_map(|m| m["id"].as_str().map(str::to_string))
                    .collect()
            })
            .unwrap_or_default();
        ids.sort();
        Ok(ids)
    }
}

/// `TEMPERATURE` as the JSON number Python sends (0.2, not the f32 widening 0.20000000298...).
fn temperature() -> Value {
    json!(TEMPERATURE.to_string().parse::<f64>().unwrap_or(0.2))
}
