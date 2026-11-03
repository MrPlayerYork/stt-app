//! Config schema parity with `src/stt_app/config.py`. The files in `tests/data/` were written by the
//! Python app's `save_config` (tomli_w).

use stt_core::config::{AppConfig, LlmMode, Provider, ReasoningLevel, WhisperDevice};

const PY_DEFAULTS: &str = include_str!("data/python_defaults.toml");
const PY_CUSTOM: &str = include_str!("data/python_custom.toml");

#[test]
fn defaults_match_python() {
    let c = AppConfig::default();
    assert_eq!(c.history_size, 20);

    assert_eq!(c.audio.device_name, None);
    assert_eq!(c.audio.samplerate, 16000);

    assert_eq!(c.whisper.model, "large-v3-turbo");
    assert_eq!(c.whisper.device, WhisperDevice::Auto);
    assert_eq!(c.whisper.compute_type, "auto");
    assert_eq!(c.whisper.language.as_deref(), Some("en"));
    assert_eq!(c.whisper.models_dir, "");
    assert_eq!(c.whisper.download_connections, 8);

    let l = &c.llm;
    assert!(!l.enabled);
    assert_eq!(l.provider, Provider::OpenAiCompat);
    assert_eq!(l.model, "");
    assert_eq!(l.base_url, "");
    assert_eq!(l.api_key_env, "");
    assert_eq!(l.mode, LlmMode::Cleanup);
    assert_eq!(l.preset, "deramble");
    assert!(l.custom_prompts.is_empty());
    assert_eq!(l.idle_ttl_seconds, 120);
    assert!(!l.unload_on_exit);
    assert_eq!(l.reasoning, ReasoningLevel::Default);
    assert!(!l.word_choice);
}

#[test]
fn python_default_file_loads_as_default() {
    assert_eq!(
        AppConfig::from_toml_str(PY_DEFAULTS).unwrap(),
        AppConfig::default()
    );
}

#[test]
fn python_custom_file_loads() {
    let c = AppConfig::from_toml_str(PY_CUSTOM).unwrap();
    assert_eq!(c.history_size, 5);
    assert_eq!(c.audio.device_name.as_deref(), Some("USB Mic"));
    assert_eq!(c.whisper.model, "small");
    assert_eq!(c.whisper.device, WhisperDevice::Gpu);
    assert_eq!(c.whisper.language, None, "\"\" means auto-detect");
    assert!(c.llm.enabled);
    assert_eq!(c.llm.provider, Provider::Anthropic);
    assert_eq!(c.llm.mode, LlmMode::Prompt);
    assert_eq!(c.llm.preset, "mine");
    assert_eq!(c.llm.idle_ttl_seconds, 0);
    assert_eq!(c.llm.reasoning, ReasoningLevel::High);
    assert!(c.llm.word_choice);
    let prompts: Vec<_> = c
        .llm
        .custom_prompts
        .iter()
        .map(|(k, v)| (k.as_str(), v.as_str()))
        .collect();
    assert_eq!(
        prompts,
        [
            ("zeta", "Be terse."),
            ("mine", "Translate to French.\nKeep \"quotes\".")
        ],
        "custom prompts keep file order"
    );
}

/// What Rust writes has exactly the keys and values Python writes, so the Python app can read it,
/// plus only the additive Rust-only keys (`whisper.quantization`, `whisper.decoding`), which Python ignores.
#[test]
fn rust_output_matches_python_output() {
    for py in [PY_DEFAULTS, PY_CUSTOM] {
        let cfg = AppConfig::from_toml_str(py).unwrap();
        let mut ours: toml::Table = toml::from_str(&cfg.to_toml_string().unwrap()).unwrap();
        let whisper = ours["whisper"].as_table_mut().unwrap();
        assert_eq!(
            whisper.remove("quantization").unwrap().as_str(),
            Some("q5_0")
        );
        assert_eq!(whisper.remove("decoding").unwrap().as_str(), Some("beam"));
        let theirs: toml::Table = toml::from_str(py).unwrap();
        assert_eq!(ours, theirs);
    }
}

#[test]
fn round_trip_preserves_everything() {
    let mut c = AppConfig::default();
    c.audio.device_name = Some("Mic".into());
    c.whisper.language = None;
    c.llm.custom_prompts.insert("b".into(), "B".into());
    c.llm.custom_prompts.insert("a".into(), "A".into());
    let back = AppConfig::from_toml_str(&c.to_toml_string().unwrap()).unwrap();
    assert_eq!(back, c);
    assert_eq!(
        back.llm.custom_prompts.keys().collect::<Vec<_>>(),
        ["b", "a"]
    );
}

/// Python `test_unknown_keys_are_ignored`, plus missing sections/keys taking defaults.
#[test]
fn unknown_keys_ignored_and_missing_keys_default() {
    let c = AppConfig::from_toml_str(
        "history_size = 7\nmystery = 1\n\
         [audio]\nsamplerate = 16000\nbogus = \"x\"\n\
         [llm]\nenabled = true\nfuture_flag = false\n\
         [future_section]\nx = 1\n",
    )
    .unwrap();
    assert_eq!(c.history_size, 7);
    assert_eq!(c.audio, AppConfig::default().audio);
    assert_eq!(c.whisper, AppConfig::default().whisper);
    assert!(c.llm.enabled);
    assert_eq!(c.llm.preset, "deramble");

    assert_eq!(AppConfig::from_toml_str("").unwrap(), AppConfig::default());
}

#[test]
fn unknown_enum_values_fall_back_to_defaults() {
    let c = AppConfig::from_toml_str(
        "[whisper]\ndevice = \"tpu\"\n[llm]\nreasoning = \"extreme\"\nmode = \"weird\"\nprovider = \"nope\"\n",
    )
    .unwrap();
    assert_eq!(c.whisper.device, WhisperDevice::Auto);
    assert_eq!(c.llm.reasoning, ReasoningLevel::Default);
    assert_eq!(c.llm.mode, LlmMode::Cleanup);
    assert_eq!(c.llm.provider, Provider::OpenAiCompat);
}

#[test]
fn gpu_is_written_as_cuda_for_python() {
    let c = AppConfig::from_toml_str("[whisper]\ndevice = \"gpu\"\n").unwrap();
    assert_eq!(c.whisper.device, WhisperDevice::Gpu);
    assert!(c.to_toml_string().unwrap().contains("device = \"cuda\""));
}

#[test]
fn wrong_types_are_errors() {
    assert!(AppConfig::from_toml_str("history_size = \"lots\"").is_err());
}
