//! LLM post-processing over raw HTTPS (blocking `ureq` + `serde_json`).
//!
//! Owns:
//! - P2-6: OpenAI and OpenAI-compatible backends, local server detection (LM Studio / Ollama / other),
//!   idle TTL (LM Studio `ttl`, Ollama `keep_alive`), unload, reasoning levels + capture, and `<think>`
//!   stripping (`split_think`) so reasoning never reaches the clipboard.
//! - P2-7: Anthropic and Google backends; prompts, presets (deramble, concise, markdown), custom prompts
//!   and word choice (`system_prompt_for`, `all_presets`).
//!
//! Implements [`stt_core::LlmBackend`] and [`stt_core::LlmFactory`] (Python `make_backend`).
