//! Cleanup prompt, Prompt Mode presets, custom prompts and word choice (Python `llm/prompts.py`).
//! Owner: P2-7.

use stt_core::LlmConfig;

/// The system prompt the current mode uses (Python `system_prompt_for`).
pub fn system_prompt_for(cfg: &LlmConfig) -> String {
    let _ = cfg;
    todo!("P2-7")
}

/// Built-in presets then custom prompts, in menu order: `(key, prompt)` (Python `all_presets`).
pub fn all_presets(cfg: &LlmConfig) -> Vec<(String, String)> {
    let _ = cfg;
    todo!("P2-7")
}
