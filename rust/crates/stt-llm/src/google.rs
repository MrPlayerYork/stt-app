//! Google Gemini backend over raw HTTPS (Python `llm/google_backend.py`). Owner: P2-7.
//!
//! Gemini `generateContent` with the key in the `x-goog-api-key` header; `thinkingConfig` per reasoning
//! level exactly as Python `_thinking`.

use serde_json::{Value, json};
use stt_core::{LlmBackend, LlmError, LlmResult, ReasoningLevel};

use crate::base::{TEMPERATURE, TIMEOUT};

const DEFAULT_BASE_URL: &str = "https://generativelanguage.googleapis.com";
const REJECTED_NOTE: &str = "model rejected the reasoning setting";

/// Google Gemini backend.
pub struct GoogleBackend {
    model: String,
    api_key: String,
    reasoning: ReasoningLevel,
    base_url: String,
    agent: ureq::Agent,
}

struct Reply {
    status: u16,
    body: String,
}

impl GoogleBackend {
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

    /// `thinkingConfig` for the reasoning level, `None` for `default` (Python `_thinking`).
    /// Gemini 3 takes `thinkingLevel` ("off" -> MINIMAL); 2.5 takes `thinkingBudget` tokens
    /// ("off" -> 0, or 128 on Pro, which cannot disable thinking).
    fn thinking(&self) -> Option<Value> {
        let lvl = self.reasoning;
        if lvl == ReasoningLevel::Default {
            return None;
        }
        let include = lvl != ReasoningLevel::Off;
        if self.model.contains("gemini-3") {
            let level = if lvl == ReasoningLevel::Off {
                "MINIMAL".to_string()
            } else {
                lvl.as_str().to_uppercase()
            };
            return Some(json!({"thinkingLevel": level, "includeThoughts": include}));
        }
        let budget = match lvl {
            ReasoningLevel::Off if self.model.contains("pro") => 128,
            ReasoningLevel::Off => 0,
            ReasoningLevel::Low => 1024,
            ReasoningLevel::Medium => 8192,
            _ => 24576,
        };
        Some(json!({"thinkingBudget": budget, "includeThoughts": include}))
    }

    fn model_path(&self) -> String {
        if self.model.starts_with("models/") {
            self.model.clone()
        } else {
            format!("models/{}", self.model)
        }
    }

    fn get(&self, path_and_query: &str) -> Result<Reply, LlmError> {
        let url = format!("{}{}", self.base_url, path_and_query);
        let resp = self
            .agent
            .get(&url)
            .header("x-goog-api-key", &self.api_key)
            .call()
            .map_err(|e| transport_error(&e, &self.base_url))?;
        read_reply(resp, &self.base_url)
    }

    fn generate(
        &self,
        system_prompt: &str,
        text: &str,
        thinking: Option<&Value>,
    ) -> Result<Reply, LlmError> {
        let mut gen_cfg = json!({"temperature": temperature()});
        if let Some(t) = thinking {
            gen_cfg["thinkingConfig"] = t.clone();
        }
        let body = json!({
            "contents": [{"role": "user", "parts": [{"text": text}]}],
            "systemInstruction": {"parts": [{"text": system_prompt}]},
            "generationConfig": gen_cfg,
        });
        let url = format!(
            "{}/v1beta/{}:generateContent",
            self.base_url,
            self.model_path()
        );
        let resp = self
            .agent
            .post(&url)
            .header("x-goog-api-key", &self.api_key)
            .header("content-type", "application/json")
            .send(body.to_string())
            .map_err(|e| transport_error(&e, &self.base_url))?;
        read_reply(resp, &self.base_url)
    }
}

fn read_reply(resp: ureq::http::Response<ureq::Body>, url: &str) -> Result<Reply, LlmError> {
    let status = resp.status().as_u16();
    let body = resp
        .into_body()
        .read_to_string()
        .map_err(|e| transport_error(&e, url))?;
    Ok(Reply { status, body })
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

/// The API's error message (`error.message`), or the raw body.
fn error_message(body: &str) -> String {
    serde_json::from_str::<Value>(body)
        .ok()
        .and_then(|v| v["error"]["message"].as_str().map(str::to_string))
        .unwrap_or_else(|| body.trim().to_string())
}

/// Map an error status to the human message (Python `_wrap`).
fn wrap_status(status: u16, body: &str) -> LlmError {
    let msg = error_message(body);
    let message = if matches!(status, 401 | 403) || msg.contains("API key") {
        "Authentication failed: check the Google API key".to_string()
    } else if status == 404 {
        "Model not found: check the model name".to_string()
    } else if status == 429 {
        "Rate limited or quota exceeded".to_string()
    } else if matches!(status, 408 | 504) {
        "Request to Google timed out".to_string()
    } else {
        let m: String = msg.chars().take(200).collect();
        format!("Google API error {status}: {m}")
    };
    LlmError::Http { status, message }
}

fn parse_response(body: &str, note: &str) -> Result<LlmResult, LlmError> {
    let v: Value = serde_json::from_str(body)
        .map_err(|e| LlmError::InvalidResponse(format!("not JSON: {e}")))?;
    let (mut answer, mut thoughts) = (String::new(), Vec::new());
    if let Some(parts) = v["candidates"][0]["content"]["parts"].as_array() {
        for p in parts {
            let Some(t) = p["text"].as_str().filter(|t| !t.is_empty()) else {
                continue;
            };
            if p["thought"].as_bool().unwrap_or(false) {
                thoughts.push(t.trim().to_string());
            } else {
                answer.push_str(t);
            }
        }
    }
    let usage = v.get("usageMetadata").filter(|u| !u.is_null());
    let tokens = usage.and_then(|u| u["thoughtsTokenCount"].as_u64());
    let reasoned = match tokens {
        Some(t) => Some(t > 0),
        None if !thoughts.is_empty() => Some(true),
        None if usage.is_some() => Some(false),
        None => None,
    };
    Ok(LlmResult {
        text: answer.trim().to_string(),
        reasoning: thoughts.join("\n\n"),
        reasoning_tokens: tokens,
        reasoned,
        note: note.to_string(),
    })
}

impl LlmBackend for GoogleBackend {
    fn complete(&self, system_prompt: &str, text: &str) -> Result<LlmResult, LlmError> {
        if self.model.is_empty() {
            return Err(LlmError::InvalidResponse("No model selected".into()));
        }
        let thinking = self.thinking();
        let mut reply = self.generate(system_prompt, text, thinking.as_ref())?;
        let mut note = "";
        if reply.status == 400
            && thinking.is_some()
            && error_message(&reply.body).to_lowercase().contains("think")
        {
            note = REJECTED_NOTE;
            reply = self.generate(system_prompt, text, None)?;
        }
        if reply.status >= 300 {
            return Err(wrap_status(reply.status, &reply.body));
        }
        parse_response(&reply.body, note)
    }

    fn list_models(&self) -> Result<Vec<String>, LlmError> {
        let mut names: Vec<String> = Vec::new();
        let mut token: Option<String> = None;
        loop {
            let path = match &token {
                Some(t) => format!("/v1beta/models?pageSize=1000&pageToken={t}"),
                None => "/v1beta/models?pageSize=1000".to_string(),
            };
            let reply = self.get(&path)?;
            if reply.status >= 300 {
                return Err(wrap_status(reply.status, &reply.body));
            }
            let v: Value = serde_json::from_str(&reply.body)
                .map_err(|e| LlmError::InvalidResponse(format!("not JSON: {e}")))?;
            for m in v["models"].as_array().into_iter().flatten() {
                let actions = m["supportedGenerationMethods"]
                    .as_array()
                    .or_else(|| m["supportedActions"].as_array());
                let ok = match actions {
                    Some(a) if !a.is_empty() => a.iter().any(|x| x == "generateContent"),
                    _ => true,
                };
                let name = m["name"].as_str().unwrap_or("");
                let name = name.strip_prefix("models/").unwrap_or(name);
                if ok && !name.is_empty() {
                    names.push(name.to_string());
                }
            }
            match v["nextPageToken"].as_str().filter(|t| !t.is_empty()) {
                Some(t) => token = Some(t.to_string()),
                None => break,
            }
        }
        names.sort();
        Ok(names)
    }
}

/// `TEMPERATURE` as the JSON number Python sends (0.2, not the f32 widening 0.20000000298...).
fn temperature() -> Value {
    json!(TEMPERATURE.to_string().parse::<f64>().unwrap_or(0.2))
}
