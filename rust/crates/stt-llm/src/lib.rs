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
//!
//! Module ownership (Phase 2; signatures below are fixed so P2-6 and P2-7 can work in parallel):
//! - P2-6: [`base`], [`openai_compat`], [`factory`]
//! - P2-7: [`anthropic`], [`google`], [`prompts`]
//!
//! Each agent edits only its own files. Don't touch another agent's module; change a shared signature
//! only after a board `note`.

pub mod anthropic;
pub mod base;
pub mod factory;
pub mod google;
pub mod openai_compat;
pub mod prompts;

pub use factory::{HttpLlmFactory, process};
pub use prompts::{all_presets, system_prompt_for};
