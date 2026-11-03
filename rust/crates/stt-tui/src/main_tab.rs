//! Main tab (UI_MOCKUPS.txt D / A1-A4): live header (mic with hot-mic meter, Whisper status, mode), last result, recent dictations, and the single-key menu.
//!
//! **Owner: P3-1.** Stub from the Phase 3 skeleton; replace the body, keep the public entry point
//! `MainTab::new()` (other modules and the runner call it).

use crossterm::event::{KeyCode, KeyEvent};
use ratatui::Frame;
use ratatui::layout::Rect;
use ratatui::widgets::Paragraph;

use crate::app::App;
use crate::screen::{Hint, Nav, Screen};

/// The Main tab.
#[derive(Default)]
pub struct MainTab;

impl MainTab {
    /// Main tab.
    pub fn new() -> Self {
        Self
    }
}

impl Screen for MainTab {
    fn render(&mut self, _app: &App, frame: &mut Frame, area: Rect) {
        frame.render_widget(Paragraph::new("Main tab (not built yet)"), area);
    }

    fn on_key(&mut self, _app: &mut App, key: KeyEvent) -> Nav {
        match key.code {
            KeyCode::Char('q') => Nav::Quit,
            _ => Nav::Ignored,
        }
    }

    fn hints(&self, _app: &App) -> Vec<Hint> {
        vec![("q", "quit".into())]
    }
}
