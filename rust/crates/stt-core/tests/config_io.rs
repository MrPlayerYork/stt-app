//! Config file load/save against a temp directory (never the real `~/.config/stt-app`).
//! Uses `CARGO_TARGET_TMPDIR`, so nothing is written outside the target dir.

use std::path::{Path, PathBuf};

use stt_core::config::{AppConfig, load_config_from, save_config_to};
use stt_core::error::ConfigError;

const PY_DEFAULTS: &str = include_str!("data/python_defaults.toml");
const PY_CUSTOM: &str = include_str!("data/python_custom.toml");

/// A fresh, empty directory under the target tmp dir, unique per test.
fn scratch_dir(name: &str) -> PathBuf {
    let dir = PathBuf::from(env!("CARGO_TARGET_TMPDIR"))
        .join(format!("config-io-{name}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

/// Names of leftover `.config-*.tmp` files in `dir`.
fn tmp_leftovers(dir: &Path) -> Vec<String> {
    std::fs::read_dir(dir)
        .unwrap()
        .map(|e| e.unwrap().file_name().to_string_lossy().into_owned())
        .filter(|n| n.starts_with(".config-") && n.ends_with(".tmp"))
        .collect()
}

fn custom_config() -> AppConfig {
    let mut c = AppConfig::from_toml_str(PY_CUSTOM).unwrap();
    c.audio.device_name = Some("Rust Mic".into());
    c
}

#[test]
fn save_then_load_round_trips() {
    let dir = scratch_dir("round-trip");
    let path = dir.join("config.toml");
    let cfg = custom_config();
    save_config_to(&cfg, &path).unwrap();
    assert_eq!(load_config_from(&path).unwrap(), Some(cfg));
    assert!(tmp_leftovers(&dir).is_empty());
}

#[test]
fn save_creates_missing_parent_dirs() {
    let dir = scratch_dir("mkdir");
    let path = dir.join("a").join("b").join("config.toml");
    save_config_to(&AppConfig::default(), &path).unwrap();
    assert_eq!(load_config_from(&path).unwrap(), Some(AppConfig::default()));
}

#[test]
fn save_replaces_existing_file() {
    let dir = scratch_dir("replace");
    let path = dir.join("config.toml");
    save_config_to(&AppConfig::default(), &path).unwrap();
    save_config_to(&custom_config(), &path).unwrap();
    assert_eq!(load_config_from(&path).unwrap(), Some(custom_config()));
    assert!(tmp_leftovers(&dir).is_empty());
}

#[cfg(unix)]
#[test]
fn saved_file_is_private_like_python() {
    use std::os::unix::fs::PermissionsExt;
    let dir = scratch_dir("mode");
    let path = dir.join("config.toml");
    save_config_to(&AppConfig::default(), &path).unwrap();
    let mode = std::fs::metadata(&path).unwrap().permissions().mode() & 0o777;
    assert_eq!(mode, 0o600);
}

/// Python's `save_config` output (committed fixtures) loads unchanged from a file on disk.
#[test]
fn python_written_files_load_from_disk() {
    let dir = scratch_dir("py-fixture");
    for (name, text) in [("defaults", PY_DEFAULTS), ("custom", PY_CUSTOM)] {
        let path = dir.join(format!("{name}.toml"));
        std::fs::write(&path, text).unwrap();
        let loaded = load_config_from(&path).unwrap().expect("file exists");
        assert_eq!(loaded, AppConfig::from_toml_str(text).unwrap());
    }
}

/// Python `load_config` returns None when there is no file (first run).
#[test]
fn missing_file_loads_as_none() {
    let dir = scratch_dir("missing");
    assert_eq!(load_config_from(&dir.join("nope.toml")).unwrap(), None);
    // A missing parent directory is also "no file", as Python's `exists()` is False there.
    assert_eq!(
        load_config_from(&dir.join("no-dir").join("config.toml")).unwrap(),
        None
    );
}

/// Invalid TOML is an error, not a silent default (Python raises `tomllib.TOMLDecodeError`).
#[test]
fn invalid_toml_is_a_parse_error() {
    let dir = scratch_dir("parse-error");
    let path = dir.join("config.toml");
    std::fs::write(&path, "[audio\ndevice_name = ").unwrap();
    assert!(matches!(
        load_config_from(&path),
        Err(ConfigError::Parse(_))
    ));
    // The bad file is left alone.
    assert_eq!(
        std::fs::read_to_string(&path).unwrap(),
        "[audio\ndevice_name = "
    );
}

/// Reading a path that is a directory is an I/O error, not "missing".
#[test]
fn unreadable_path_is_an_io_error() {
    let dir = scratch_dir("read-dir");
    assert!(matches!(
        load_config_from(&dir),
        Err(ConfigError::Io { .. })
    ));
}

/// A save that fails at the final rename removes its temp file and leaves nothing behind.
#[test]
fn failed_rename_cleans_up_temp_file() {
    let dir = scratch_dir("rename-fail");
    // A non-empty directory sitting at the target path makes `rename(file, path)` fail.
    let path = dir.join("config.toml");
    std::fs::create_dir_all(path.join("inner")).unwrap();
    assert!(matches!(
        save_config_to(&AppConfig::default(), &path),
        Err(ConfigError::Io { .. })
    ));
    assert!(tmp_leftovers(&dir).is_empty(), "temp file left behind");
    assert!(path.join("inner").is_dir(), "target directory untouched");
}

/// A save that fails while creating the temp file leaves the previous config intact.
#[cfg(unix)]
#[test]
fn failed_save_keeps_old_file_intact() {
    use std::os::unix::fs::PermissionsExt;
    let dir = scratch_dir("save-fail");
    let path = dir.join("config.toml");
    save_config_to(&custom_config(), &path).unwrap();
    let before = std::fs::read_to_string(&path).unwrap();

    // Read-only directory: creating the temp file fails. Root ignores permissions, so skip there.
    std::fs::set_permissions(&dir, std::fs::Permissions::from_mode(0o500)).unwrap();
    let probe = dir.join("probe");
    let is_root = std::fs::write(&probe, b"x").is_ok();
    let result = if is_root {
        None
    } else {
        Some(save_config_to(&AppConfig::default(), &path))
    };
    std::fs::set_permissions(&dir, std::fs::Permissions::from_mode(0o700)).unwrap();
    if is_root {
        let _ = std::fs::remove_file(&probe);
        eprintln!("skipped: running as root, directory permissions do not apply");
        return;
    }

    assert!(matches!(result, Some(Err(ConfigError::Io { .. }))));
    assert_eq!(std::fs::read_to_string(&path).unwrap(), before);
    assert!(tmp_leftovers(&dir).is_empty());
}

/// Default save has the same keys and values as Python's default save (plus the Rust-only quantization and decoding keys).
#[test]
fn saved_text_matches_python_key_layout() {
    let dir = scratch_dir("layout");
    let path = dir.join("config.toml");
    save_config_to(&AppConfig::default(), &path).unwrap();
    let written: toml::Table = toml::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
    let python: toml::Table = toml::from_str(PY_DEFAULTS).unwrap();
    let mut ours = written;
    ours["whisper"]
        .as_table_mut()
        .unwrap()
        .remove("quantization");
    ours["whisper"].as_table_mut().unwrap().remove("decoding");
    assert_eq!(ours, python);
}
