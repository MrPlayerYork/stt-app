//! Config recovery (decided 2026-11-02): when `config.toml` has bad values or does not parse, keep every
//! value that is still valid, reset the rest to defaults, and report what happened so the UI can warn at
//! startup ("Kept / Reset / Needs a value", UI_MOCKUPS.txt A9). The original file is backed up before the
//! recovered config is saved over it.
//!
//! **Owner: P3-4.** The types below are the contract the UI (P3-3) codes against; the function body is a
//! stub that only does a strict load.

use std::path::{Path, PathBuf};

use crate::config::{AppConfig, load_config_from};
use crate::error::ConfigError;

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

/// Load `path` leniently. `Ok((None, report))` when the file does not exist (first run). Only an
/// unreadable file (I/O error) is an error; bad content is recovered and reported.
///
/// STUB (P3-4 implements): currently a strict load with an empty report.
pub fn load_config_recovering(
    path: &Path,
) -> Result<(Option<AppConfig>, ConfigReport), ConfigError> {
    let cfg = load_config_from(path)?;
    Ok((
        cfg,
        ConfigReport {
            path: path.to_path_buf(),
            ..ConfigReport::default()
        },
    ))
}
