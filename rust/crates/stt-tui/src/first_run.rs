//! First-run welcome (Python `first_run`): explains that the model loads in the background, then opens calibration; if no mic is chosen, uses the default device.
//!
//! **Owner: P3-3.** Stub from the Phase 3 skeleton; replace the body, keep the public entry point
//! `FirstRunScreen::new()` (other modules and the runner call it).

use crossterm::event::{KeyCode, KeyEvent};
use ratatui::Frame;
use ratatui::layout::Rect;
use ratatui::widgets::Paragraph;

use crate::app::App;
use crate::screen::{Hint, Layer, Nav, Screen};

/// Full-screen welcome overlay shown when there is no config or no mic chosen.
#[derive(Default)]
pub struct FirstRunScreen;

impl FirstRunScreen {
    /// Welcome screen.
    pub fn new() -> Self {
        Self
    }
}

impl Screen for FirstRunScreen {
    fn render(&mut self, _app: &App, frame: &mut Frame, area: Rect) {
        frame.render_widget(Paragraph::new("Welcome (not built yet)"), area);
    }

    fn on_key(&mut self, _app: &mut App, key: KeyEvent) -> Nav {
        match key.code {
            KeyCode::Enter | KeyCode::Esc => Nav::Pop,
            _ => Nav::Ignored,
        }
    }

    fn hints(&self, _app: &App) -> Vec<Hint> {
        vec![]
    }

    fn layer(&self) -> Layer {
        Layer::Full
    }
}
