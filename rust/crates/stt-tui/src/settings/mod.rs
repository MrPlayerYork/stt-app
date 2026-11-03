//! Settings tab (A6-A8): section list (Microphone, Whisper, LLM) with a help box; each section is a list of
//! rows with in-place ←→ choices, text editing and masked secrets.
//!
//! Every change goes into `app.cfg` and is saved at once (`app.save()`); blocking work (model list, test
//! connection, keyring writes, downloads) runs on a [`crate::task::Task`] and never blocks the UI.

use crossterm::event::KeyEvent;
use ratatui::Frame;
use ratatui::layout::Rect;
use stt_core::{Provider, WhisperDecoding};

use crate::app::App;
use crate::screen::{Hint, Nav, Screen};

mod llm;
mod mic;
mod popups;
mod prompts;
mod section;
mod ttl;
mod whisper;

#[cfg(test)]
mod tests;

pub use ttl::{parse_duration, ttl_label};

use section::{Row, Section, SectionScreen};

/// The Settings tab.
pub struct SettingsTab {
    inner: SectionScreen<Root>,
}

impl SettingsTab {
    /// Settings tab.
    pub fn new() -> Self {
        Self {
            inner: SectionScreen::tab(Root),
        }
    }
}

impl Default for SettingsTab {
    fn default() -> Self {
        Self::new()
    }
}

impl Screen for SettingsTab {
    fn render(&mut self, app: &App, frame: &mut Frame, area: Rect) {
        self.inner.render(app, frame, area);
    }

    fn on_key(&mut self, app: &mut App, key: KeyEvent) -> Nav {
        self.inner.on_key(app, key)
    }

    fn on_tick(&mut self, app: &mut App) -> Nav {
        self.inner.on_tick(app)
    }

    fn hints(&self, app: &App) -> Vec<Hint> {
        self.inner.hints(app)
    }
}

/// Save after changing `app.cfg`.
pub(crate) fn commit(app: &mut App) {
    app.save();
}

/// The next/previous element of `all` after `cur` (wraps).
pub(crate) fn cycle_in<T: Copy + PartialEq>(all: &[T], cur: T, dir: i32) -> T {
    let i = all.iter().position(|x| *x == cur).unwrap_or(0) as i32;
    all[(i + dir).rem_euclid(all.len() as i32) as usize]
}

/// "beam 5 (accurate)" / "greedy (fast)".
pub(crate) fn decoding_label(d: WhisperDecoding) -> &'static str {
    match d {
        WhisperDecoding::Beam => "beam 5 (accurate)",
        WhisperDecoding::Greedy => "greedy (fast)",
    }
}

/// Short provider name for summaries.
pub(crate) fn provider_short(p: Provider) -> &'static str {
    match p {
        Provider::OpenAi => "OpenAI",
        Provider::OpenAiCompat => "Local server",
        Provider::Anthropic => "Anthropic",
        Provider::Google => "Gemini",
    }
}

/// The section list (A6).
struct Root;

impl Section for Root {
    fn title(&self) -> String {
        String::new()
    }

    fn rows(&self, app: &App) -> Vec<Row> {
        let w = &app.cfg.whisper;
        let l = &app.cfg.llm;
        let mic = app
            .cfg
            .audio
            .device_name
            .clone()
            .unwrap_or_else(|| "(system default)".into());
        let whisper = format!(
            "{} · {} · {} · {} · {}",
            w.model,
            w.device,
            w.quantization,
            if w.decoding == WhisperDecoding::Beam {
                "beam 5"
            } else {
                "greedy"
            },
            w.language.as_deref().unwrap_or("auto"),
        );
        let mut llm = vec![
            if l.enabled { "on" } else { "off" }.to_string(),
            provider_short(l.provider).to_string(),
        ];
        if !l.model.is_empty() {
            llm.push(l.model.clone());
        }
        if l.enabled && l.reasoning != stt_core::ReasoningLevel::Default {
            llm.push(format!("reasoning {}", l.reasoning));
        }
        vec![
            Row::new('1', "Microphone", mic),
            Row::new('2', "Whisper", whisper),
            Row::new('3', "LLM", llm.join(" · ")),
        ]
    }

    fn help(&self, _app: &App, i: usize) -> String {
        match i {
            0 => "Which microphone to record from, and calibration to find the right one.",
            1 => "Model, device, quality/speed (beam 5 or greedy), language, downloads.",
            _ => "Optional cleanup or Prompt Mode rewriting by a local or cloud language model.",
        }
        .into()
    }

    fn activate(&mut self, app: &mut App, i: usize) -> Nav {
        match i {
            0 => Nav::Push(Box::new(SectionScreen::overlay(mic::MicSection))),
            1 => Nav::Push(Box::new(SectionScreen::overlay(
                whisper::WhisperSection::new(app),
            ))),
            _ => Nav::Push(Box::new(SectionScreen::overlay(llm::LlmSection))),
        }
    }

    fn enter_hint(&self) -> &'static str {
        "open"
    }
}
