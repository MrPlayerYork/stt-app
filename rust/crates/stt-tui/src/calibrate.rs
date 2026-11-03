//! Calibration overlay (Python `calibrate.run_calibration`): record every mic at once with live per-mic meters, score with Whisper via `stt_calibrate::evaluate`, ranked table, pick / retry / cancel. On a pick it saves `audio.device_name` and reopens the hot mic.
//!
//! **Owner: P3-3.** Stub from the Phase 3 skeleton; replace the body, keep the public entry point
//! `CalibrationScreen::new(app)` (other modules and the runner call it).

use crossterm::event::{KeyCode, KeyEvent};
use ratatui::Frame;
use ratatui::layout::Rect;
use ratatui::widgets::Paragraph;

use crate::app::App;
use crate::screen::{Hint, Layer, Nav, Screen};

/// Full-screen calibration overlay. Pushed by Settings › Microphone and by first run.
pub struct CalibrationScreen;

impl CalibrationScreen {
    /// Prepare calibration (closes the hot mic; it is reopened when the overlay finishes).
    pub fn new(_app: &mut App) -> Self {
        Self
    }
}

impl Screen for CalibrationScreen {
    fn render(&mut self, _app: &App, frame: &mut Frame, area: Rect) {
        frame.render_widget(Paragraph::new("Calibration (not built yet)"), area);
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

    fn layer(&self) -> Layer {
        Layer::Full
    }
}
