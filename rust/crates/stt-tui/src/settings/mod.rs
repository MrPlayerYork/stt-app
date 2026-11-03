//! Settings tab (A6-A8): section list (Microphone, Whisper, LLM) with a help box; each section is a list of rows with in-place ←→ choices, text editing and masked secrets.
//!
//! **Owner: P3-2.** Stub from the Phase 3 skeleton; replace the body, keep the public entry point
//! `SettingsTab::new()` (other modules and the runner call it).

use crossterm::event::KeyEvent;
use ratatui::Frame;
use ratatui::layout::Rect;
use ratatui::widgets::Paragraph;

use crate::app::App;
use crate::screen::{Hint, Nav, Screen};

/// The Settings tab.
#[derive(Default)]
pub struct SettingsTab;

impl SettingsTab {
    /// Settings tab.
    pub fn new() -> Self {
        Self
    }
}

impl Screen for SettingsTab {
    fn render(&mut self, _app: &App, frame: &mut Frame, area: Rect) {
        frame.render_widget(Paragraph::new("Settings tab (not built yet)"), area);
    }

    fn on_key(&mut self, _app: &mut App, _key: KeyEvent) -> Nav {
        Nav::Ignored
    }

    fn hints(&self, _app: &App) -> Vec<Hint> {
        vec![]
    }
}
