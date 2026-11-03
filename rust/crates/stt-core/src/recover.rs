//! Config recovery (decided 2026-11-02): when `config.toml` has bad values or does not parse, keep every
//! value that is still valid, reset the rest to defaults, and report what happened so the UI can warn at
//! startup ("Kept / Reset / Needs a value", UI_MOCKUPS.txt A9). The original file is backed up before the
//! recovered config is saved over it.
//!
//! **Owner: P3-4.** The types are the contract the UI (P3-3) codes against.

use std::path::{Path, PathBuf};

use indexmap::IndexMap;
use toml::{Table, Value};

use crate::config::{
    AppConfig, LlmMode, Provider, ReasoningLevel, WhisperDecoding, WhisperDevice, WhisperQuant,
    save_config_to,
};
use crate::error::ConfigError;
use crate::transcribe::WHISPER_MODELS;

/// One value that was not used as written.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ConfigIssue {
    /// Dotted key, e.g. `llm.reasoning`.
    pub key: String,
    /// The value found in the file, as written (`""` when missing).
    pub found: String,
    /// What the app did, e.g. `reset to default "default"`, `LLM turned off`.
    pub action: String,
}

/// What [`load_config_recovering`] found.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ConfigReport {
    /// The config file.
    pub path: PathBuf,
    /// Where the original was copied before the recovered config replaced it; `None` if nothing changed.
    pub backup: Option<PathBuf>,
    /// The file was not valid TOML at all (the message); recovery then salvages what it can.
    pub parse_error: Option<String>,
    /// Keys that loaded fine (dotted names).
    pub kept: Vec<String>,
    /// Values that were invalid and replaced by the default (unknown enum strings, wrong types, ...).
    pub reset: Vec<ConfigIssue>,
    /// Settings that are valid on their own but leave something unusable, e.g. LLM enabled with no
    /// model. The app applies a safe fallback (stated in `action`) and the user should set a value.
    pub needs_value: Vec<ConfigIssue>,
}

impl ConfigReport {
    /// True when there is nothing to tell the user.
    pub fn is_clean(&self) -> bool {
        self.parse_error.is_none() && self.reset.is_empty() && self.needs_value.is_empty()
    }
}

/// Built-in Prompt Mode preset keys. Hard-coded copy of the keys of `PRESETS` in
/// `stt-llm/src/prompts.rs` (stt-core must not depend on stt-llm); keep the two in sync.
const BUILTIN_PRESETS: &[&str] = &["deramble", "concise", "markdown"];

/// Load `path` leniently. `Ok((None, report))` when the file does not exist (first run). Only an
/// unreadable file (I/O error) is an error; bad content is recovered and reported.
///
/// Every key that is valid is kept; wrong types, unknown enum strings, out-of-range numbers and unknown
/// keys are reported in [`ConfigReport::reset`]; an unusable LLM setup or a dangling preset is reported in
/// [`ConfigReport::needs_value`]. If the file is not valid TOML, [`ConfigReport::parse_error`] is set and
/// whatever can be salvaged is used. When the report is not clean the original is copied to
/// `<path>.bak-<unix-seconds>` ([`ConfigReport::backup`]) and the recovered config is saved over it; a
/// clean file is never touched. The returned config is always the recovered one.
pub fn load_config_recovering(
    path: &Path,
) -> Result<(Option<AppConfig>, ConfigReport), ConfigError> {
    let mut report = ConfigReport {
        path: path.to_path_buf(),
        ..ConfigReport::default()
    };
    let io = |source| ConfigError::Io {
        path: path.to_path_buf(),
        source,
    };
    let text = match std::fs::read_to_string(path) {
        Ok(t) => t,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok((None, report)),
        Err(e) => return Err(io(e)),
    };

    let table = match toml::from_str::<Table>(&text) {
        Ok(t) => t,
        Err(e) => {
            report.parse_error = Some(e.to_string());
            salvage(&text)
        }
    };
    let cfg = check_table(&table, &mut report);

    if !report.is_clean() {
        let backup = backup_path(path);
        std::fs::copy(path, &backup).map_err(io)?;
        report.backup = Some(backup);
        save_config_to(&cfg, path)?;
    }
    Ok((Some(cfg), report))
}

/// `<path>.bak-<unix seconds>`, with a numeric suffix if that name is taken.
fn backup_path(path: &Path) -> PathBuf {
    let secs = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| d.as_secs());
    let mut name = path.as_os_str().to_owned();
    name.push(format!(".bak-{secs}"));
    let mut candidate = PathBuf::from(&name);
    let mut n = 1;
    while candidate.exists() {
        let mut next = name.clone();
        next.push(format!("-{n}"));
        candidate = PathBuf::from(next);
        n += 1;
    }
    candidate
}

/// Best-effort parse of a file that is not valid TOML: split into `[section]` blocks, parse each block
/// whole, and for a broken block fall back to one `key = value` line at a time.
fn salvage(text: &str) -> Table {
    let mut blocks: Vec<(String, Vec<&str>)> = vec![(String::new(), Vec::new())];
    for line in text.lines() {
        let t = line.trim();
        // Only a line that parses on its own as a table header starts a block.
        if t.starts_with('[') && !t.starts_with("[[") && toml::from_str::<Table>(t).is_ok() {
            blocks.push((format!("{t}\n"), Vec::new()));
        } else {
            blocks.last_mut().expect("never empty").1.push(line);
        }
    }
    let mut out = Table::new();
    for (header, lines) in blocks {
        let whole = format!("{header}{}", lines.join("\n"));
        if let Ok(t) = toml::from_str::<Table>(&whole) {
            merge(&mut out, t);
            continue;
        }
        if let Ok(t) = toml::from_str::<Table>(&header) {
            merge(&mut out, t);
        }
        for line in lines {
            if let Ok(t) = toml::from_str::<Table>(&format!("{header}{line}")) {
                merge(&mut out, t);
            }
        }
    }
    out
}

/// Deep-merge `src` into `dst` (tables merge, other values overwrite).
fn merge(dst: &mut Table, src: Table) {
    for (k, v) in src {
        match (dst.get_mut(&k), v) {
            (Some(Value::Table(d)), Value::Table(s)) => merge(d, s),
            (_, v) => {
                dst.insert(k, v);
            }
        }
    }
}

/// Collects kept/reset entries while walking one section.
struct Walk<'a> {
    /// The report being filled.
    report: &'a mut ConfigReport,
    /// Section prefix with trailing dot (`""` for the top level, `"llm."`, ...).
    prefix: &'static str,
}

impl Walk<'_> {
    fn kept(&mut self, key: &str) {
        self.report.kept.push(format!("{}{key}", self.prefix));
    }

    fn reset(&mut self, key: &str, found: &Value, default: &str) {
        self.report.reset.push(ConfigIssue {
            key: format!("{}{key}", self.prefix),
            found: found.to_string(),
            action: format!("reset to {default}"),
        });
    }

    /// Checked read of `key`: `get` converts the value or returns `None` if it is invalid.
    fn field<T>(
        &mut self,
        t: &Table,
        key: &str,
        slot: &mut T,
        default_text: &str,
        get: impl FnOnce(&Value) -> Option<T>,
    ) {
        let Some(v) = t.get(key) else { return };
        match get(v) {
            Some(x) => {
                *slot = x;
                self.kept(key);
            }
            None => self.reset(key, v, default_text),
        }
    }

    fn string(&mut self, t: &Table, key: &str, slot: &mut String) {
        let d = format!("{:?}", slot.as_str());
        self.field(t, key, slot, &d, |v| v.as_str().map(str::to_owned));
    }

    fn opt_string(&mut self, t: &Table, key: &str, slot: &mut Option<String>) {
        let d = format!("{:?}", slot.as_deref().unwrap_or(""));
        self.field(t, key, slot, &d, |v| {
            v.as_str().map(|s| (!s.is_empty()).then(|| s.to_owned()))
        });
    }

    fn boolean(&mut self, t: &Table, key: &str, slot: &mut bool) {
        let d = slot.to_string();
        self.field(t, key, slot, &d, Value::as_bool);
    }

    fn int<N: TryFrom<i64> + ToString + Copy>(
        &mut self,
        t: &Table,
        key: &str,
        slot: &mut N,
        ok: impl FnOnce(N) -> bool,
    ) {
        let d = slot.to_string();
        self.field(t, key, slot, &d, |v| {
            v.as_integer()
                .and_then(|i| N::try_from(i).ok())
                .filter(|n| ok(*n))
        });
    }

    fn enumeration<E: Copy + Default>(
        &mut self,
        t: &Table,
        key: &str,
        slot: &mut E,
        parse: fn(&str) -> Option<E>,
        name: fn(E) -> &'static str,
    ) {
        let d = format!("{:?}", name(E::default()));
        self.field(t, key, slot, &d, |v| v.as_str().and_then(parse));
    }

    /// Report keys of `t` that are not in `known`.
    fn unknown(&mut self, t: &Table, known: &[&str]) {
        for (k, v) in t {
            if !known.contains(&k.as_str()) {
                self.report.reset.push(ConfigIssue {
                    key: format!("{}{k}", self.prefix),
                    found: v.to_string(),
                    action: "unknown key, dropped".into(),
                });
            }
        }
    }
}

/// Fetch section `name` of `root` as a table; a non-table value is reported and treated as absent.
fn section<'a>(root: &'a Table, name: &str, report: &mut ConfigReport) -> Option<&'a Table> {
    match root.get(name)? {
        Value::Table(t) => Some(t),
        other => {
            report.reset.push(ConfigIssue {
                key: name.into(),
                found: other.to_string(),
                action: "not a section, reset to defaults".into(),
            });
            None
        }
    }
}

/// Run every key check over `root`, filling `report` and returning the recovered config.
fn check_table(root: &Table, report: &mut ConfigReport) -> AppConfig {
    let mut cfg = AppConfig::default();
    let empty = Table::new();

    {
        let mut w = Walk { report, prefix: "" };
        w.int(root, "history_size", &mut cfg.history_size, |n| n >= 1);
        w.unknown(root, &["history_size", "audio", "whisper", "llm"]);
    }

    let audio = section(root, "audio", report).unwrap_or(&empty);
    {
        let mut w = Walk {
            report,
            prefix: "audio.",
        };
        w.opt_string(audio, "device_name", &mut cfg.audio.device_name);
        w.int(audio, "samplerate", &mut cfg.audio.samplerate, |_| true);
        w.unknown(audio, &["device_name", "samplerate"]);
    }

    let wh = section(root, "whisper", report).unwrap_or(&empty);
    {
        let mut w = Walk {
            report,
            prefix: "whisper.",
        };
        let c = &mut cfg.whisper;
        let d = format!("{:?}", c.model);
        w.field(wh, "model", &mut c.model, &d, |v| {
            v.as_str()
                .filter(|s| WHISPER_MODELS.contains(s))
                .map(str::to_owned)
        });
        w.enumeration(
            wh,
            "device",
            &mut c.device,
            WhisperDevice::parse,
            WhisperDevice::as_str,
        );
        w.string(wh, "compute_type", &mut c.compute_type);
        w.opt_string(wh, "language", &mut c.language);
        w.string(wh, "models_dir", &mut c.models_dir);
        w.int(
            wh,
            "download_connections",
            &mut c.download_connections,
            |n| (1..=32).contains(&n),
        );
        w.enumeration(
            wh,
            "quantization",
            &mut c.quantization,
            WhisperQuant::parse,
            WhisperQuant::as_str,
        );
        w.enumeration(
            wh,
            "decoding",
            &mut c.decoding,
            WhisperDecoding::parse,
            WhisperDecoding::as_str,
        );
        w.unknown(
            wh,
            &[
                "model",
                "device",
                "compute_type",
                "language",
                "models_dir",
                "download_connections",
                "quantization",
                "decoding",
            ],
        );
    }

    let llm = section(root, "llm", report).unwrap_or(&empty);
    {
        let mut w = Walk {
            report,
            prefix: "llm.",
        };
        let c = &mut cfg.llm;
        w.boolean(llm, "enabled", &mut c.enabled);
        w.enumeration(
            llm,
            "provider",
            &mut c.provider,
            Provider::parse,
            Provider::as_str,
        );
        w.string(llm, "model", &mut c.model);
        w.string(llm, "base_url", &mut c.base_url);
        w.string(llm, "api_key_env", &mut c.api_key_env);
        w.enumeration(llm, "mode", &mut c.mode, LlmMode::parse, LlmMode::as_str);
        w.string(llm, "preset", &mut c.preset);
        w.int(llm, "idle_ttl_seconds", &mut c.idle_ttl_seconds, |_| true);
        w.boolean(llm, "unload_on_exit", &mut c.unload_on_exit);
        w.enumeration(
            llm,
            "reasoning",
            &mut c.reasoning,
            ReasoningLevel::parse,
            ReasoningLevel::as_str,
        );
        w.boolean(llm, "word_choice", &mut c.word_choice);
        check_custom_prompts(&mut w, llm, &mut c.custom_prompts);
        w.unknown(
            llm,
            &[
                "enabled",
                "provider",
                "model",
                "base_url",
                "api_key_env",
                "mode",
                "preset",
                "custom_prompts",
                "idle_ttl_seconds",
                "unload_on_exit",
                "reasoning",
                "word_choice",
            ],
        );
    }

    check_usable(&mut cfg, report);
    cfg
}

/// `[llm.custom_prompts]`: keep string entries, report the others.
fn check_custom_prompts(w: &mut Walk<'_>, llm: &Table, slot: &mut IndexMap<String, String>) {
    let Some(v) = llm.get("custom_prompts") else {
        return;
    };
    let Value::Table(t) = v else {
        w.reset("custom_prompts", v, "no custom prompts");
        return;
    };
    let mut bad = false;
    for (name, val) in t {
        match val.as_str() {
            Some(s) => {
                slot.insert(name.clone(), s.to_owned());
            }
            None => {
                bad = true;
                w.report.reset.push(ConfigIssue {
                    key: format!("llm.custom_prompts.{name}"),
                    found: val.to_string(),
                    action: "not text, dropped".into(),
                });
            }
        }
    }
    if !bad || !slot.is_empty() {
        w.kept("custom_prompts");
    }
}

/// Settings that parse but leave the app unusable: LLM enabled without what it needs, or a preset that
/// does not exist. Applies the safe fallback and records it in `needs_value`.
fn check_usable(cfg: &mut AppConfig, report: &mut ConfigReport) {
    let llm = &mut cfg.llm;
    if llm.enabled {
        let mut missing = Vec::new();
        if llm.model.trim().is_empty() {
            missing.push("llm.model");
        }
        if llm.provider == Provider::OpenAiCompat && llm.base_url.trim().is_empty() {
            missing.push("llm.base_url");
        }
        if !missing.is_empty() {
            llm.enabled = false;
            report.kept.retain(|k| k != "llm.enabled");
            for key in missing {
                report.needs_value.push(ConfigIssue {
                    key: key.into(),
                    found: String::new(),
                    action: "LLM turned off".into(),
                });
            }
        }
    }
    let known = BUILTIN_PRESETS.contains(&llm.preset.as_str())
        || llm.custom_prompts.contains_key(&llm.preset);
    if !known {
        report.kept.retain(|k| k != "llm.preset");
        report.needs_value.push(ConfigIssue {
            key: "llm.preset".into(),
            found: format!("{:?}", llm.preset),
            action: "reset to \"deramble\"".into(),
        });
        llm.preset = "deramble".into();
    }
}
