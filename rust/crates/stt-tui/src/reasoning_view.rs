//! Scrollable full-body view of an LLM's reasoning text (`[v]` on the Main and History tabs).

use crossterm::event::{KeyCode, KeyEvent};
use ratatui::Frame;
use ratatui::layout::Rect;
use ratatui::text::Line;
use ratatui::widgets::{Paragraph, Wrap};

use crate::app::App;
use crate::screen::{Hint, Nav, Screen};
use crate::theme;

/// Full overlay showing `text`; ↑↓/PgUp/PgDn/Home/End scroll, Esc closes.
pub struct ReasoningView {
    text: String,
    scroll: usize,
    page: usize,
    max_scroll: usize,
}

impl ReasoningView {
    /// View of `text`.
    pub fn new(text: impl Into<String>) -> Self {
        Self {
            text: text.into(),
            scroll: 0,
            page: 10,
            max_scroll: usize::MAX,
        }
    }

    fn wrapped_lines(&self, width: usize) -> usize {
        let w = width.max(1);
        self.text
            .lines()
            .map(|l| l.chars().count().max(1).div_ceil(w))
            .sum::<usize>()
            .max(1)
    }
}

impl Screen for ReasoningView {
    fn render(&mut self, _app: &App, frame: &mut Frame, area: Rect) {
        let block = theme::panel("Reasoning");
        let inner = block.inner(area);
        frame.render_widget(block, area);
        self.page = inner.height.max(1) as usize;
        // Over-estimate slightly (word wrap can add lines) so the end is always reachable.
        let total = self.wrapped_lines(inner.width as usize) + 2;
        self.max_scroll = total.saturating_sub(self.page);
        self.scroll = self.scroll.min(self.max_scroll);
        let lines: Vec<Line> = self
            .text
            .lines()
            .map(|l| Line::raw(l.to_string()))
            .collect();
        frame.render_widget(
            Paragraph::new(lines)
                .wrap(Wrap { trim: false })
                .scroll((self.scroll.min(u16::MAX as usize) as u16, 0)),
            inner,
        );
    }

    fn on_key(&mut self, _app: &mut App, key: KeyEvent) -> Nav {
        let page = self.page;
        match key.code {
            KeyCode::Esc | KeyCode::Char('q') | KeyCode::Char('v') => return Nav::Pop,
            KeyCode::Up | KeyCode::Char('k') => self.scroll = self.scroll.saturating_sub(1),
            KeyCode::Down | KeyCode::Char('j') => self.scroll = self.scroll.saturating_add(1),
            KeyCode::PageUp => self.scroll = self.scroll.saturating_sub(page),
            KeyCode::PageDown => self.scroll = self.scroll.saturating_add(page),
            KeyCode::Home => self.scroll = 0,
            KeyCode::End => self.scroll = usize::MAX,
            _ => return Nav::Ignored,
        }
        self.scroll = self.scroll.min(self.max_scroll);
        Nav::Stay
    }

    fn hints(&self, _app: &App) -> Vec<Hint> {
        vec![("↑↓/PgUp/PgDn", "scroll".into()), ("Esc", "back".into())]
    }

    fn title(&self, _app: &App) -> String {
        "Reasoning".into()
    }
}
