//! `load_config_recovering` against temp files (under `CARGO_TARGET_TMPDIR`, never the real config).

use std::path::PathBuf;

use stt_core::config::{AppConfig, Provider, WhisperDevice, load_config_from};
use stt_core::recover::{ConfigIssue, ConfigReport, load_config_recovering};

const PY_CUSTOM: &str = include_str!("data/python_custom.toml");
const PY_DEFAULTS: &str = include_str!("data/python_defaults.toml");

fn write(name: &str, content: &str) -> PathBuf {
    let dir = PathBuf::from(env!("CARGO_TARGET_TMPDIR"))
        .join(format!("recover-{name}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let p = dir.join("config.toml");
    std::fs::write(&p, content).unwrap();
    p
}

fn reset<'a>(r: &'a ConfigReport, key: &str) -> &'a ConfigIssue {
    r.reset
        .iter()
        .find(|i| i.key == key)
        .unwrap_or_else(|| panic!("no reset for {key}: {:?}", r.reset))
}

#[test]
fn clean_files_are_untouched() {
    for (n, text) in [("custom", PY_CUSTOM), ("defaults", PY_DEFAULTS)] {
        let p = write(&format!("clean-{n}"), text);
        let (cfg, r) = load_config_recovering(&p).unwrap();
        assert!(r.is_clean(), "{r:?}");
        assert!(r.backup.is_none());
        assert!(!r.kept.is_empty());
        assert_eq!(cfg, Some(AppConfig::from_toml_str(text).unwrap()));
        assert_eq!(std::fs::read_to_string(&p).unwrap(), text);
        assert_eq!(std::fs::read_dir(p.parent().unwrap()).unwrap().count(), 1);
    }
}

#[test]
fn missing_file_is_none() {
    let p = write("missing", "");
    std::fs::remove_file(&p).unwrap();
    let (cfg, r) = load_config_recovering(&p).unwrap();
    assert!(cfg.is_none() && r.is_clean() && r.kept.is_empty());
}

#[test]
fn bad_values_are_reset_and_rest_kept() {
    let text = r#"
history_size = "lots"
mystery = 1

[audio]
device_name = "USB Mic"
samplerate = "fast"

[whisper]
model = "gigantic"
device = "tpu"
quantization = "q2"
decoding = 3
download_connections = 99
language = "de"

[llm]
provider = "bing"
mode = "shout"
reasoning = "extreme"
idle_ttl_seconds = 30

[llm.custom_prompts]
ok = "fine"
bad = 5

[extra]
x = 1
"#;
    let p = write("bad", text);
    let (cfg, r) = load_config_recovering(&p).unwrap();
    let cfg = cfg.unwrap();
    assert_eq!(cfg.history_size, 20);
    assert_eq!(cfg.audio.device_name.as_deref(), Some("USB Mic"));
    assert_eq!(cfg.whisper.language.as_deref(), Some("de"));
    assert_eq!(cfg.llm.idle_ttl_seconds, 30);
    assert_eq!(cfg.llm.custom_prompts.len(), 1);
    let i = reset(&r, "history_size");
    assert_eq!(i.found, "\"lots\"");
    assert!(i.action.contains("20"));
    assert_eq!(reset(&r, "whisper.model").found, "\"gigantic\"");
    assert_eq!(reset(&r, "whisper.device").action, "reset to \"auto\"");
    reset(&r, "whisper.quantization");
    reset(&r, "whisper.decoding");
    assert_eq!(reset(&r, "whisper.download_connections").found, "99");
    reset(&r, "audio.samplerate");
    reset(&r, "llm.provider");
    reset(&r, "llm.mode");
    reset(&r, "llm.reasoning");
    reset(&r, "llm.custom_prompts.bad");
    assert_eq!(reset(&r, "mystery").action, "unknown key, dropped");
    assert_eq!(reset(&r, "extra").action, "unknown key, dropped");
    for k in [
        "audio.device_name",
        "whisper.language",
        "llm.idle_ttl_seconds",
        "llm.custom_prompts",
    ] {
        assert!(r.kept.iter().any(|x| x == k), "{k} not kept");
    }
    assert!(!r.kept.iter().any(|x| x == "history_size"));
    // Backup is the original; the rewritten file loads strictly and equals the recovered config.
    let b = r.backup.as_ref().unwrap();
    assert_eq!(std::fs::read_to_string(b).unwrap(), text);
    assert_eq!(load_config_from(&p).unwrap().unwrap(), cfg);
    assert!(load_config_recovering(&p).unwrap().1.is_clean());
}

#[test]
fn history_size_zero_and_aliases() {
    let p = write("zero", "history_size = 0\n[whisper]\ndevice = \"vulkan\"\n");
    let (cfg, r) = load_config_recovering(&p).unwrap();
    reset(&r, "history_size");
    assert_eq!(cfg.unwrap().whisper.device, WhisperDevice::Gpu);
    assert!(r.kept.iter().any(|k| k == "whisper.device"));
}

#[test]
fn llm_enabled_without_model_is_turned_off() {
    let p = write(
        "nomodel",
        "[llm]\nenabled = true\nprovider = \"anthropic\"\n",
    );
    let (cfg, r) = load_config_recovering(&p).unwrap();
    assert!(!cfg.unwrap().llm.enabled);
    assert_eq!(r.needs_value.len(), 1);
    assert_eq!(r.needs_value[0].key, "llm.model");
    assert_eq!(r.needs_value[0].action, "LLM turned off");
    assert!(r.backup.is_some());

    let p = write("nourl", "[llm]\nenabled = true\nmodel = \"m\"\n");
    let (cfg, r) = load_config_recovering(&p).unwrap();
    let cfg = cfg.unwrap();
    assert!(!cfg.llm.enabled);
    assert_eq!(cfg.llm.provider, Provider::OpenAiCompat);
    assert_eq!(r.needs_value[0].key, "llm.base_url");
}

#[test]
fn bad_preset_is_reset() {
    let p = write(
        "preset",
        "[llm]\npreset = \"nope\"\n[llm.custom_prompts]\nmine = \"x\"\n",
    );
    let (cfg, r) = load_config_recovering(&p).unwrap();
    assert_eq!(cfg.unwrap().llm.preset, "deramble");
    assert_eq!(r.needs_value[0].key, "llm.preset");
    assert_eq!(r.needs_value[0].found, "\"nope\"");
    for ok in ["concise", "mine"] {
        let p = write(
            "preset-ok",
            &format!("[llm]\npreset = \"{ok}\"\n[llm.custom_prompts]\nmine = \"x\"\n"),
        );
        assert!(load_config_recovering(&p).unwrap().1.is_clean());
    }
}

#[test]
fn broken_toml_salvages_other_sections() {
    let text = "history_size = 7\n[audio]\ndevice_name = \"Mic\"\n\n[whisper]\nmodel = \"small\"\nlanguage = \"fr\"\n\n[llm]\nmodel = \"unterminated\nprovider = \"google\"\nword_choice = true\n";
    let p = write("broken", text);
    let (cfg, r) = load_config_recovering(&p).unwrap();
    let cfg = cfg.unwrap();
    assert!(r.parse_error.is_some());
    assert!(!r.is_clean());
    assert_eq!(cfg.history_size, 7);
    assert_eq!(cfg.audio.device_name.as_deref(), Some("Mic"));
    assert_eq!(cfg.whisper.model, "small");
    assert_eq!(cfg.whisper.language.as_deref(), Some("fr"));
    assert_eq!(cfg.llm.provider, Provider::Google);
    assert!(cfg.llm.word_choice);
    assert_eq!(cfg.llm.model, "");
    assert_eq!(
        std::fs::read_to_string(r.backup.as_ref().unwrap()).unwrap(),
        text
    );
    assert_eq!(load_config_from(&p).unwrap().unwrap(), cfg);
}
