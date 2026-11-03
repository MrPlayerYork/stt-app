//! Record flow (A3-A4): recording overlay with the red frame and big meter, then transcribe → LLM → clipboard on worker threads, with step lines.
//!
//! **Owner: P3-1.** Stub from the Phase 3 skeleton; replace the body, keep the public entry point
//! `RecordScreen::new(app)` (other modules and the runner call it).

use crossterm::event::{KeyCode, KeyEvent};
use ratatui::Frame;
use ratatui::layout::Rect;
use ratatui::widgets::Paragraph;

use crate::app::App;
use crate::screen::{Hint, Nav, Screen};

/// Recording + processing overlay, pushed by the Main tab.
pub struct RecordScreen;

impl RecordScreen {
    /// Start recording on the hot mic.
    pub fn new(_app: &mut App) -> Self {
        Self
    }
}

impl Screen for RecordScreen {
    fn render(&mut self, _app: &App, frame: &mut Frame, area: Rect) {
        frame.render_widget(Paragraph::new("Recording (not built yet)"), area);
    }

    fn on_key(&mut self, _app: &mut App, key: KeyEvent) -> Nav {
        match key.code {
            KeyCode::Esc => Nav::Pop,
            _ => Nav::Ignored,
        }
    }

    fn hints(&self, _app: &App) -> Vec<Hint> {
        vec![]
    }
}
