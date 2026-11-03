//! History tab (A5): newest-first list with a preview pane, copy on Enter / number keys, reasoning view.
//!
//! **Owner: P3-1.** Stub from the Phase 3 skeleton; replace the body, keep the public entry point
//! `HistoryTab::new()` (other modules and the runner call it).

use crossterm::event::KeyEvent;
use ratatui::Frame;
use ratatui::layout::Rect;
use ratatui::widgets::Paragraph;

use crate::app::App;
use crate::screen::{Hint, Nav, Screen};

/// The History tab.
#[derive(Default)]
pub struct HistoryTab;

impl HistoryTab {
    /// History tab.
    pub fn new() -> Self {
        Self
    }
}

impl Screen for HistoryTab {
    fn render(&mut self, _app: &App, frame: &mut Frame, area: Rect) {
        frame.render_widget(Paragraph::new("History tab (not built yet)"), area);
    }

    fn on_key(&mut self, _app: &mut App, _key: KeyEvent) -> Nav {
        Nav::Ignored
    }

    fn hints(&self, _app: &App) -> Vec<Hint> {
        vec![]
    }
}
