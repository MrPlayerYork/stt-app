//! Settings tests: drive the real `Ui` with the Settings tab on screen, using the fakes.

use std::time::Duration;

use crossterm::event::KeyCode;
use stt_core::fakes::FakeLlm;
use stt_core::secrets::HF_TOKEN_USER;
use stt_core::{AppConfig, Provider, ReasoningLevel, WhisperDecoding};

use super::SettingsTab;
use crate::app::{App, FlashKind};
use crate::runner::Ui;
use crate::testkit::{ch, ctrl, key, render_ui, settle, test_app, test_app_with};

fn ui() -> Ui {
    Ui::with_tabs([
        Box::new(SettingsTab::new()),
        Box::new(SettingsTab::new()),
        Box::new(SettingsTab::new()),
    ])
}

fn typ(ui: &mut Ui, app: &mut App, s: &str) {
    for c in s.chars() {
        ui.on_key(app, ch(c));
    }
}

fn press(ui: &mut Ui, app: &mut App, code: KeyCode) {
    ui.on_key(app, key(code));
}

fn screen(ui: &mut Ui, app: &App) -> String {
    render_ui(ui, app, 80, 24)
}

/// Tick (bounded) until only `overlays` overlays remain, or the screen contains `text`.
fn tick_until_overlays(ui: &mut Ui, app: &mut App, overlays: usize) {
    for _ in 0..400 {
        ui.tick(app);
        if ui.overlay_count() == overlays {
            return;
        }
        std::thread::sleep(Duration::from_millis(5));
    }
    panic!("overlays never reached {overlays}");
}

fn tick_until_text(ui: &mut Ui, app: &mut App, text: &str) -> String {
    for _ in 0..400 {
        ui.tick(app);
        let s = screen(ui, app);
        if s.contains(text) {
            return s;
        }
        std::thread::sleep(Duration::from_millis(5));
    }
    panic!("never saw {text:?}:\n{}", screen(ui, app));
}

fn llm_cfg() -> AppConfig {
    let mut cfg = AppConfig::default();
    cfg.llm.base_url = "http://localhost:1234/v1".into();
    cfg
}

#[test]
fn section_list_shows_summaries_and_help() {
    let mut cfg = AppConfig::default();
    cfg.audio.device_name = Some("Blue Snowball".into());
    let (app, _h) = test_app_with(cfg, FakeLlm::new("x"));
    let mut ui = ui();
    let s = screen(&mut ui, &app);
    assert!(s.contains("› 1 Microphone"), "{s}");
    assert!(s.contains("Blue Snowball"), "{s}");
    assert!(
        s.contains("large-v3-turbo · auto · q5_0 · beam 5 · en"),
        "{s}"
    );
    assert!(s.contains("off · Local server"), "{s}");
    assert!(
        s.contains("Which microphone"),
        "help for the selected row: {s}"
    );
}

#[test]
fn enter_opens_section_and_esc_closes() {
    let (mut app, _h) = test_app();
    let mut ui = ui();
    press(&mut ui, &mut app, KeyCode::Down);
    press(&mut ui, &mut app, KeyCode::Enter);
    assert_eq!(ui.overlay_count(), 1);
    let s = screen(&mut ui, &app);
    assert!(s.contains("› Whisper"), "{s}");
    assert!(s.contains("Decoding"), "{s}");
    press(&mut ui, &mut app, KeyCode::Esc);
    assert_eq!(ui.overlay_count(), 0);
}

#[test]
fn whisper_decoding_cycles_saves_and_reloads_once_on_close() {
    let (mut app, h) = test_app();
    let mut ui = ui();
    typ(&mut ui, &mut app, "2"); // open Whisper
    typ(&mut ui, &mut app, "4"); // decoding: beam -> greedy
    assert_eq!(app.cfg.whisper.decoding, WhisperDecoding::Greedy);
    let s = screen(&mut ui, &app);
    assert!(s.contains("◂ greedy (fast) ▸"), "{s}");
    assert!(s.contains("Beam 5 tries several"), "help text: {s}");
    press(&mut ui, &mut app, KeyCode::Right);
    assert_eq!(app.cfg.whisper.decoding, WhisperDecoding::Beam);
    press(&mut ui, &mut app, KeyCode::Left);
    assert_eq!(app.cfg.whisper.decoding, WhisperDecoding::Greedy);
    assert_eq!(h.config.saves().len(), 3);
    assert_eq!(h.transcribers.created().len(), 0, "no reload while open");
    press(&mut ui, &mut app, KeyCode::Esc);
    assert_eq!(h.transcribers.created().len(), 1, "one reload on close");

    // Reopen and close without changes: no reload.
    typ(&mut ui, &mut app, "2");
    press(&mut ui, &mut app, KeyCode::Esc);
    assert_eq!(h.transcribers.created().len(), 1);
}

#[test]
fn whisper_model_pick_and_device_cycle() {
    let (mut app, _h) = test_app();
    let mut ui = ui();
    typ(&mut ui, &mut app, "2");
    typ(&mut ui, &mut app, "1");
    assert_eq!(ui.overlay_count(), 2);
    typ(&mut ui, &mut app, "4"); // large-v3 is 1, turbo 2, distil 3, medium 4
    assert_eq!(app.cfg.whisper.model, "medium");
    assert_eq!(ui.overlay_count(), 1);
    typ(&mut ui, &mut app, "2");
    typ(&mut ui, &mut app, "3");
    // Quality cycles f16 -> q8_0 -> q5_0 -> f16: one step from q5_0 wraps to f16.
    assert_eq!(app.cfg.whisper.quantization, stt_core::WhisperQuant::F16);
}

#[test]
fn language_blank_is_none_and_text_is_lowercased() {
    let (mut app, _h) = test_app();
    let mut ui = ui();
    typ(&mut ui, &mut app, "2");
    typ(&mut ui, &mut app, "5");
    press(&mut ui, &mut app, KeyCode::Backspace);
    press(&mut ui, &mut app, KeyCode::Backspace);
    press(&mut ui, &mut app, KeyCode::Enter);
    assert_eq!(app.cfg.whisper.language, None);
    assert!(screen(&mut ui, &app).contains("auto-detect"));
    typ(&mut ui, &mut app, "5");
    typ(&mut ui, &mut app, " DE ");
    press(&mut ui, &mut app, KeyCode::Enter);
    assert_eq!(app.cfg.whisper.language.as_deref(), Some("de"));
}

#[test]
fn connections_validated() {
    let (mut app, _h) = test_app();
    let mut ui = ui();
    typ(&mut ui, &mut app, "2");
    typ(&mut ui, &mut app, "7");
    press(&mut ui, &mut app, KeyCode::Backspace);
    typ(&mut ui, &mut app, "99");
    press(&mut ui, &mut app, KeyCode::Enter);
    assert_eq!(app.cfg.whisper.download_connections, 8);
    assert_eq!(app.flash.as_ref().map(|f| f.kind), Some(FlashKind::Error));
    typ(&mut ui, &mut app, "7");
    press(&mut ui, &mut app, KeyCode::Backspace);
    typ(&mut ui, &mut app, "12");
    press(&mut ui, &mut app, KeyCode::Enter);
    assert_eq!(app.cfg.whisper.download_connections, 12);
}

#[test]
fn hf_token_stored_and_never_rendered() {
    let (mut app, h) = test_app();
    let mut ui = ui();
    typ(&mut ui, &mut app, "2");
    typ(&mut ui, &mut app, "8");
    let secret = "hf_SUPERSECRETTOKEN123";
    ui.on_paste(&mut app, secret);
    let s = screen(&mut ui, &app);
    assert!(!s.contains("SUPERSECRET"), "{s}");
    assert!(s.contains(&format!("({} characters)", secret.len())), "{s}");
    press(&mut ui, &mut app, KeyCode::Enter);
    tick_until_overlays(&mut ui, &mut app, 1);
    use stt_core::SecretStore;
    assert_eq!(
        h.secrets.get(HF_TOKEN_USER).unwrap().as_deref(),
        Some(secret)
    );
    let s = screen(&mut ui, &app);
    assert!(!s.contains("SUPERSECRET"), "{s}");
    assert!(s.contains("Hugging Face token") && s.contains("set"), "{s}");
    assert_eq!(
        app.flash.as_ref().map(|f| f.text.as_str()),
        Some("Token stored.")
    );
}

#[test]
fn download_now_shows_result() {
    let (mut app, _h) = test_app();
    let mut ui = ui();
    typ(&mut ui, &mut app, "2");
    typ(&mut ui, &mut app, "9");
    let s = tick_until_text(&mut ui, &mut app, "Model ready");
    assert!(s.contains("Model ready:"), "{s}");
}

#[test]
fn llm_enable_refused_until_configured() {
    let (mut app, _h) = test_app();
    let mut ui = ui();
    typ(&mut ui, &mut app, "3");
    typ(&mut ui, &mut app, "1");
    assert!(!app.cfg.llm.enabled);
    let f = app.flash.clone().unwrap();
    assert_eq!(f.kind, FlashKind::Warn);
    assert_eq!(f.text, "Choose a provider, server and model first.");
    app.cfg.llm.base_url = "http://localhost:1234/v1".into();
    app.cfg.llm.model = "qwen".into();
    typ(&mut ui, &mut app, "1");
    assert!(app.cfg.llm.enabled);
    assert!(screen(&mut ui, &app).contains("yes"));
}

#[test]
fn provider_change_sets_default_model_and_opens_server_picker() {
    let mut cfg = llm_cfg();
    cfg.llm.model = "old".into();
    let (mut app, _h) = test_app_with(cfg, FakeLlm::new("x"));
    let mut ui = ui();
    typ(&mut ui, &mut app, "3");
    typ(&mut ui, &mut app, "2");
    typ(&mut ui, &mut app, "3"); // Anthropic
    assert_eq!(app.cfg.llm.provider, Provider::Anthropic);
    assert_eq!(app.cfg.llm.model, Provider::Anthropic.default_model());
    assert_eq!(ui.overlay_count(), 1, "no server picker for cloud");
    // Back to the local provider: model cleared, server picker opens.
    typ(&mut ui, &mut app, "2");
    typ(&mut ui, &mut app, "2");
    assert_eq!(app.cfg.llm.provider, Provider::OpenAiCompat);
    assert_eq!(app.cfg.llm.model, "");
    assert_eq!(ui.overlay_count(), 2);
    assert!(screen(&mut ui, &app).contains("Custom URL"));
    typ(&mut ui, &mut app, "3"); // vLLM
    assert_eq!(app.cfg.llm.base_url, "http://localhost:8000/v1");
    assert_eq!(ui.overlay_count(), 1);
}

#[test]
fn custom_server_url() {
    let (mut app, _h) = test_app();
    let mut ui = ui();
    typ(&mut ui, &mut app, "3");
    typ(&mut ui, &mut app, "3");
    typ(&mut ui, &mut app, "5"); // Custom URL
    let s = screen(&mut ui, &app);
    assert!(s.contains("http://localhost:1234/v1"), "prefilled: {s}");
    press(&mut ui, &mut app, KeyCode::Enter);
    assert_eq!(app.cfg.llm.base_url, "http://localhost:1234/v1");
}

#[test]
fn model_list_comes_from_the_backend() {
    let (mut app, _h) = test_app_with(llm_cfg(), FakeLlm::new("x").models(&["alpha", "beta"]));
    let mut ui = ui();
    typ(&mut ui, &mut app, "3");
    typ(&mut ui, &mut app, "6");
    let s = tick_until_text(&mut ui, &mut app, "Type a model name");
    assert!(s.contains("alpha") && s.contains("beta"), "{s}");
    typ(&mut ui, &mut app, "2");
    assert_eq!(app.cfg.llm.model, "beta");
    assert_eq!(ui.overlay_count(), 1);
}

#[test]
fn model_can_be_typed() {
    let (mut app, _h) = test_app_with(llm_cfg(), FakeLlm::new("x").models(&["alpha"]));
    let mut ui = ui();
    typ(&mut ui, &mut app, "3");
    typ(&mut ui, &mut app, "6");
    tick_until_text(&mut ui, &mut app, "Type a model name");
    typ(&mut ui, &mut app, "2");
    typ(&mut ui, &mut app, "gemma");
    press(&mut ui, &mut app, KeyCode::Enter);
    assert_eq!(app.cfg.llm.model, "gemma");
}

#[test]
fn reasoning_and_toggles_and_ttl() {
    let (mut app, h) = test_app();
    let mut ui = ui();
    typ(&mut ui, &mut app, "3");
    typ(&mut ui, &mut app, "7");
    assert_eq!(app.cfg.llm.reasoning, ReasoningLevel::Off);
    let s = screen(&mut ui, &app);
    assert!(
        s.contains("◂ off ▸") && s.contains("ask the model not to reason"),
        "{s}"
    );
    press(&mut ui, &mut app, KeyCode::Left);
    assert_eq!(app.cfg.llm.reasoning, ReasoningLevel::Default);
    typ(&mut ui, &mut app, "8");
    assert!(app.cfg.llm.word_choice);
    typ(&mut ui, &mut app, "u");
    assert!(app.cfg.llm.unload_on_exit);
    // TTL: prefilled "2m"; replace with 90s.
    typ(&mut ui, &mut app, "9");
    press(&mut ui, &mut app, KeyCode::Backspace);
    press(&mut ui, &mut app, KeyCode::Backspace);
    typ(&mut ui, &mut app, "90s");
    press(&mut ui, &mut app, KeyCode::Enter);
    assert_eq!(app.cfg.llm.idle_ttl_seconds, 90);
    assert!(screen(&mut ui, &app).contains("90s"));
    typ(&mut ui, &mut app, "9");
    press(&mut ui, &mut app, KeyCode::Backspace);
    press(&mut ui, &mut app, KeyCode::Backspace);
    typ(&mut ui, &mut app, "soon");
    press(&mut ui, &mut app, KeyCode::Enter);
    assert_eq!(app.cfg.llm.idle_ttl_seconds, 90);
    assert_eq!(app.flash.as_ref().map(|f| f.kind), Some(FlashKind::Error));
    assert!(h.config.saves().len() >= 5);
}

#[test]
fn api_key_stored_not_shown_and_env_var_set() {
    let (mut app, h) = test_app();
    let mut ui = ui();
    typ(&mut ui, &mut app, "3");
    typ(&mut ui, &mut app, "4");
    ui.on_paste(&mut app, "sk-topsecret");
    assert!(!screen(&mut ui, &app).contains("topsecret"));
    press(&mut ui, &mut app, KeyCode::Enter);
    tick_until_overlays(&mut ui, &mut app, 1);
    use stt_core::SecretStore;
    assert_eq!(
        h.secrets.get("openai_compat").unwrap().as_deref(),
        Some("sk-topsecret")
    );
    let s = screen(&mut ui, &app);
    assert!(!s.contains("topsecret") && s.contains("set"), "{s}");
    typ(&mut ui, &mut app, "5");
    typ(&mut ui, &mut app, "MY_KEY");
    press(&mut ui, &mut app, KeyCode::Enter);
    assert_eq!(app.cfg.llm.api_key_env, "MY_KEY");
    assert!(screen(&mut ui, &app).contains("(env MY_KEY)"));
}

#[test]
fn test_connection_shows_result_and_marks_llm_used() {
    let (mut app, _h) = test_app_with(llm_cfg(), FakeLlm::new("Let's meet Wednesday at 3 pm."));
    let mut ui = ui();
    typ(&mut ui, &mut app, "3");
    typ(&mut ui, &mut app, "t");
    assert!(app.llm_used);
    let s = tick_until_text(&mut ui, &mut app, "Let's meet Wednesday");
    assert!(s.contains("Input: um so i think"), "{s}");
    assert!(s.contains("No reasoning"), "{s}");
    press(&mut ui, &mut app, KeyCode::Enter);
    assert_eq!(ui.overlay_count(), 1);
}

#[test]
fn test_connection_failure_is_shown() {
    let (mut app, _h) = test_app_with(llm_cfg(), FakeLlm::new("x").failing("boom"));
    let mut ui = ui();
    typ(&mut ui, &mut app, "3");
    typ(&mut ui, &mut app, "t");
    let s = tick_until_text(&mut ui, &mut app, "Test failed");
    assert!(s.contains("boom"), "{s}");
}

#[test]
fn custom_prompt_add_and_delete_resets_preset() {
    let (mut app, _h) = test_app();
    let mut ui = ui();
    typ(&mut ui, &mut app, "3");
    typ(&mut ui, &mut app, "c");
    typ(&mut ui, &mut app, "a");
    typ(&mut ui, &mut app, "pirate");
    press(&mut ui, &mut app, KeyCode::Enter);
    typ(&mut ui, &mut app, "Rewrite as a pirate.");
    press(&mut ui, &mut app, KeyCode::Enter);
    assert_eq!(
        app.cfg.llm.custom_prompts.get("pirate").map(String::as_str),
        Some("Rewrite as a pirate.")
    );
    let s = screen(&mut ui, &app);
    assert!(s.contains("pirate: Rewrite as a pirate."), "{s}");
    assert!(s.contains("› LLM › Custom prompts"), "{s}");

    app.cfg.llm.preset = "pirate".into();
    typ(&mut ui, &mut app, "d");
    typ(&mut ui, &mut app, "1");
    typ(&mut ui, &mut app, "y");
    assert!(app.cfg.llm.custom_prompts.is_empty());
    assert_eq!(app.cfg.llm.preset, "deramble");
    assert_eq!(ui.overlay_count(), 2, "still in custom prompts");
    press(&mut ui, &mut app, KeyCode::Esc);
    assert_eq!(ui.overlay_count(), 1);
}

#[test]
fn mic_pick_sets_device_and_reopens_mic() {
    let (mut app, _h) = test_app();
    settle(&mut app);
    let mut ui = ui();
    typ(&mut ui, &mut app, "1");
    assert!(screen(&mut ui, &app).contains("listening"));
    typ(&mut ui, &mut app, "3");
    assert!(screen(&mut ui, &app).contains("Test Mic"));
    typ(&mut ui, &mut app, "1");
    assert_eq!(app.cfg.audio.device_name.as_deref(), Some("Test Mic"));
    assert_eq!(app.mic.label(), "Test Mic");
    assert_eq!(ui.overlay_count(), 1);
}

#[test]
fn mic_calibration_row_opens_calibration() {
    let (mut app, _h) = test_app();
    let mut ui = ui();
    typ(&mut ui, &mut app, "1");
    typ(&mut ui, &mut app, "2");
    assert_eq!(ui.overlay_count(), 2);
}

#[test]
fn tab_and_ctrl_c_pass_through_on_the_tab() {
    let (mut app, _h) = test_app();
    let mut ui = ui();
    press(&mut ui, &mut app, KeyCode::Tab);
    assert_eq!(app.tab, crate::app::Tab::History);
    app.tab = crate::app::Tab::Main;
    ui.on_key(&mut app, ctrl('c'));
    assert!(app.quit);
}
