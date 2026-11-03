//! Startup warning for a recovered config (UI_MOCKUPS.txt A9): backup path, what was kept, what was
//! reset, what still needs a value. `[Enter]` continues, `[s]` opens the Settings tab.
//!
//! **Owner: P3-3.** Stub from the Phase 3 skeleton; keep `ConfigNoticeScreen::new(report)`.

use crossterm::event::{KeyCode, KeyEvent};
use ratatui::Frame;
use ratatui::layout::Rect;
use ratatui::widgets::Paragraph;
use stt_core::recover::ConfigReport;

use crate::app::{App, Tab};
use crate::screen::{Hint, Layer, Nav, Screen};

/// Full-screen notice overlay shown at startup when the config report is not clean.
pub struct ConfigNoticeScreen {
    report: ConfigReport,
}

impl ConfigNoticeScreen {
    /// Notice for `report`.
    pub fn new(report: ConfigReport) -> Self {
        Self { report }
    }
}

impl Screen for ConfigNoticeScreen {
    fn render(&mut self, _app: &App, frame: &mut Frame, area: Rect) {
        let text = format!(
            "{} had problems (not built yet)",
            self.report.path.display()
        );
        frame.render_widget(Paragraph::new(text), area);
    }

    fn on_key(&mut self, _app: &mut App, key: KeyEvent) -> Nav {
        match key.code {
            KeyCode::Enter | KeyCode::Esc => Nav::Pop,
            KeyCode::Char('s') => Nav::SwitchTab(Tab::Settings),
            _ => Nav::Stay,
        }
    }

    fn hints(&self, _app: &App) -> Vec<Hint> {
        vec![("Enter", "continue".into()), ("s", "open settings".into())]
    }

    fn layer(&self) -> Layer {
        Layer::Full
    }
}
