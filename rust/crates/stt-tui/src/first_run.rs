//! First-run welcome (Python `first_run`): explains that the model loads in the background, then opens calibration; if no mic is chosen, uses the default device.

use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};
use ratatui::Frame;
use ratatui::layout::Rect;
use ratatui::style::Style;
use ratatui::text::{Line, Span};
use ratatui::widgets::{Paragraph, Wrap};

use crate::app::{App, FlashKind};
use crate::calibrate::CalibrationScreen;
use crate::screen::{Hint, Layer, Nav, Screen};
use crate::theme;

/// Full-screen welcome overlay shown when there is no config or no mic chosen.
#[derive(Default)]
pub struct FirstRunScreen;

impl FirstRunScreen {
    /// Welcome screen.
    pub fn new() -> Self {
        Self
    }

    fn skip(app: &mut App) -> Nav {
        app.flash(
            FlashKind::Warn,
            "No microphone chosen; using the default input device.",
        );
        // Writes the config so the next start is not a first run; device_name stays None.
        app.save();
        Nav::Pop
    }
}

impl Screen for FirstRunScreen {
    fn render(&mut self, _app: &App, frame: &mut Frame, area: Rect) {
        let lines = vec![
            Line::from(
                "Welcome! Let's pick your microphone. The Whisper model loads in the background while we calibrate.",
            ),
            Line::from(""),
            Line::from(vec![
                Span::styled("[Enter]", Style::new().fg(theme::KEY)),
                Span::raw(" calibrate   "),
                Span::styled("[Esc]", Style::new().fg(theme::KEY)),
                Span::raw(" skip (use the default input device)"),
            ]),
        ];
        frame.render_widget(
            Paragraph::new(lines)
                .wrap(Wrap { trim: false })
                .block(theme::panel("Speech to Text")),
            area,
        );
    }

    fn on_key(&mut self, app: &mut App, key: KeyEvent) -> Nav {
        if key.modifiers.contains(KeyModifiers::CONTROL) && key.code == KeyCode::Char('c') {
            return Self::skip(app);
        }
        match key.code {
            KeyCode::Enter => Nav::Replace(Box::new(CalibrationScreen::new(app))),
            KeyCode::Esc | KeyCode::Char('s') => Self::skip(app),
            _ => Nav::Stay,
        }
    }

    fn hints(&self, _app: &App) -> Vec<Hint> {
        vec![("Enter", "calibrate".into()), ("Esc", "skip".into())]
    }

    fn layer(&self) -> Layer {
        Layer::Full
    }

    fn title(&self, _app: &App) -> String {
        "Welcome".into()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testkit::{key, render_screen, test_app};

    #[test]
    fn enter_replaces_with_calibration() {
        let (mut app, _h) = test_app();
        let mut s = FirstRunScreen::new();
        let out = render_screen(&mut s, &app, 80, 24);
        assert!(out.contains("Welcome! Let's pick your microphone."));
        assert!(matches!(
            s.on_key(&mut app, key(KeyCode::Enter)),
            Nav::Replace(_)
        ));
    }

    #[test]
    fn esc_saves_and_warns() {
        let (mut app, h) = test_app();
        let mut s = FirstRunScreen::new();
        assert!(matches!(s.on_key(&mut app, key(KeyCode::Esc)), Nav::Pop));
        let saves = h.config.saves();
        assert_eq!(saves.len(), 1);
        assert_eq!(saves[0].audio.device_name, None);
        assert_eq!(app.flash.as_ref().unwrap().kind, FlashKind::Warn);
    }
}
