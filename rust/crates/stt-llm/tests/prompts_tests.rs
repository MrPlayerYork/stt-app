//! Port of `tests/test_prompts.py`, plus a byte-for-byte check against the Python prompt texts
//! (`fixtures/prompts_python.json`, dumped from `stt_app.llm.prompts`).

use stt_core::{LlmConfig, LlmMode};
use stt_llm::prompts::{CLEANUP_PROMPT, PRESETS, WORD_CHOICE_ADDENDUM};
use stt_llm::{all_presets, system_prompt_for};

fn cfg(mode: LlmMode, preset: &str) -> LlmConfig {
    LlmConfig {
        mode,
        preset: preset.into(),
        ..LlmConfig::default()
    }
}

fn preset_prompt(key: &str) -> &'static str {
    PRESETS.iter().find(|p| p.0 == key).unwrap().2
}

#[test]
fn texts_match_python_byte_for_byte() {
    let v: serde_json::Value =
        serde_json::from_str(include_str!("fixtures/prompts_python.json")).unwrap();
    assert_eq!(CLEANUP_PROMPT, v["cleanup"].as_str().unwrap());
    assert_eq!(WORD_CHOICE_ADDENDUM, v["word"].as_str().unwrap());
    for (key, _, prompt) in PRESETS {
        assert_eq!(prompt, v[key].as_str().unwrap(), "preset {key}");
    }
}

#[test]
fn cleanup_mode_uses_cleanup_prompt() {
    assert_eq!(
        system_prompt_for(&cfg(LlmMode::Cleanup, "markdown")),
        CLEANUP_PROMPT
    );
}

#[test]
fn each_builtin_preset_in_prompt_mode() {
    for (key, _, prompt) in PRESETS {
        let got = system_prompt_for(&cfg(LlmMode::Prompt, key));
        assert_eq!(got, prompt);
        assert_ne!(got, CLEANUP_PROMPT);
    }
}

#[test]
fn custom_preset_overrides_prompt() {
    let mut c = cfg(LlmMode::Prompt, "mine");
    c.custom_prompts
        .insert("mine".into(), "Be a pirate.".into());
    assert_eq!(system_prompt_for(&c), "Be a pirate.");
}

#[test]
fn custom_preset_can_shadow_builtin_name() {
    let mut c = cfg(LlmMode::Prompt, "concise");
    c.custom_prompts
        .insert("concise".into(), "Custom concise.".into());
    assert_eq!(system_prompt_for(&c), "Custom concise.");
}

#[test]
fn unknown_preset_falls_back_to_deramble() {
    assert_eq!(
        system_prompt_for(&cfg(LlmMode::Prompt, "does-not-exist")),
        preset_prompt("deramble")
    );
}

#[test]
fn all_presets_lists_builtins_and_customs_in_order() {
    let mut c = LlmConfig::default();
    c.custom_prompts.insert("mine".into(), "x".into());
    c.custom_prompts.insert("Concise".into(), "y".into());
    let got = all_presets(&c);
    let pairs: Vec<(&str, &str)> = got.iter().map(|(a, b)| (a.as_str(), b.as_str())).collect();
    assert_eq!(
        pairs,
        [
            ("deramble", "De-ramble"),
            ("concise", "Concise"),
            ("markdown", "Markdown"),
            ("mine", "mine"),
            ("Concise", "Concise"),
        ]
    );
}

#[test]
fn all_presets_without_customs_is_builtins_only() {
    let keys: Vec<String> = all_presets(&LlmConfig::default())
        .into_iter()
        .map(|p| p.0)
        .collect();
    assert_eq!(keys, ["deramble", "concise", "markdown"]);
}

#[test]
fn custom_named_like_builtin_key_adds_no_duplicate() {
    let mut c = LlmConfig::default();
    c.custom_prompts.insert("concise".into(), "z".into());
    assert_eq!(all_presets(&c).len(), 3);
}

#[test]
fn word_choice_added_to_builtin_and_custom() {
    let mut c = cfg(LlmMode::Prompt, "concise");
    c.word_choice = true;
    assert_eq!(
        system_prompt_for(&c),
        format!("{}{}", preset_prompt("concise"), WORD_CHOICE_ADDENDUM)
    );
    let mut c = cfg(LlmMode::Prompt, "mine");
    c.word_choice = true;
    c.custom_prompts
        .insert("mine".into(), "Be a pirate.".into());
    assert_eq!(
        system_prompt_for(&c),
        format!("Be a pirate.{WORD_CHOICE_ADDENDUM}")
    );
}

#[test]
fn word_choice_absent_when_disabled_or_cleanup() {
    let c = cfg(LlmMode::Prompt, "deramble");
    assert_eq!(system_prompt_for(&c), preset_prompt("deramble"));
    let mut c = cfg(LlmMode::Cleanup, "deramble");
    c.word_choice = true;
    assert_eq!(system_prompt_for(&c), CLEANUP_PROMPT);
}
