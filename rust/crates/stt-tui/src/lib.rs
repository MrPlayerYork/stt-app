//! The terminal UI: `ratatui` + `crossterm`, **full screen with tabs** (Main / History / Settings) in
//! the live-panel look. Design reference: `UI_MOCKUPS.txt` section D (repo root); behaviour reference:
//! the Python app's `src/stt_app/ui.py` and `calibrate.py`.
//!
//! Uses only `stt-core` traits for audio, transcription, LLM, clipboard and secrets ([`app::Services`]),
//! so everything is tested with `stt_core::fakes` (see `testkit`).
//!
//! Module ownership (Phase 3). Shared framework, written by the orchestrator; change only after a board
//! `note` and keep it backwards compatible: [`app`], [`screen`], [`runner`], [`theme`], [`widgets`],
//! [`task`], [`model`], [`mic`], `testkit`.
//! - P3-1: [`main_tab`], [`record`], [`history_tab`] (+ new private modules they need)
//! - P3-2: [`settings`] (and submodules under `settings/`)
//! - P3-3: [`calibrate`], [`first_run`], [`config_notice`], and the `stt` binary crate
#![warn(missing_docs)]

pub mod app;
pub mod mic;
pub mod model;
pub mod runner;
pub mod screen;
pub mod task;
pub mod theme;
pub mod widgets;

pub mod calibrate;
pub mod config_notice;
pub mod first_run;
pub mod history_tab;
pub mod main_tab;
pub mod reasoning_view;
pub mod record;
pub mod settings;

#[cfg(test)]
pub(crate) mod testkit;

pub use app::{App, ConfigStore, FileConfigStore, MemoryConfigStore, Services, Tab};
pub use runner::{Ui, run};
pub use screen::{Layer, Nav, Screen};
