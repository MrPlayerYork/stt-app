//! Two small popups used by several sections: [`Busy`] (a spinner while a [`Task`] runs, then a
//! callback) and [`Info`] (a read-only message box).

use std::time::Instant;

use crossterm::event::{KeyCode, KeyEvent};
use ratatui::Frame;
use ratatui::layout::Rect;
use ratatui::text::{Line, Span};
use ratatui::widgets::{Clear, Paragraph, Wrap};

use crate::app::App;
use crate::screen::{Hint, Layer, Nav, Screen};
use crate::task::Task;
use crate::theme;

type ThenFn<T> = Box<dyn FnOnce(&mut App, T) -> Nav>;

/// "⠋ Fetching models…" while `f` runs on a worker thread; then `then` runs with the result. Esc cancels
/// (the result is dropped). If `then` returns a screen it replaces this popup, otherwise the popup closes.
pub(super) struct Busy<T> {
    title: String,
    task: Task<T>,
    since: Instant,
    then: Option<ThenFn<T>>,
}

impl<T: Send + 'static> Busy<T> {
    pub fn new(
        title: impl Into<String>,
        f: impl FnOnce() -> T + Send + 'static,
        then: impl FnOnce(&mut App, T) -> Nav + 'static,
    ) -> Self {
        Self {
            title: title.into(),
            task: Task::spawn(f),
            since: Instant::now(),
            then: Some(Box::new(then)),
        }
    }
}

impl<T> Screen for Busy<T> {
    fn render(&mut self, _app: &App, frame: &mut Frame, area: Rect) {
        frame.render_widget(Clear, area);
        let block = theme::panel("Working");
        let inner = block.inner(area);
        frame.render_widget(block, area);
        frame.render_widget(
            Paragraph::new(Line::from(vec![
                Span::raw(format!("{} ", theme::spinner(self.since))),
                Span::raw(self.title.clone()),
            ])),
            inner,
        );
    }

    fn on_key(&mut self, _app: &mut App, key: KeyEvent) -> Nav {
        if key.code == KeyCode::Esc {
            Nav::Pop
        } else {
            Nav::Stay
        }
    }

    fn on_tick(&mut self, app: &mut App) -> Nav {
        if let Some(v) = self.task.poll() {
            return match self.then.take().map(|f| f(app, v)) {
                Some(Nav::Push(s) | Nav::Replace(s)) => Nav::Replace(s),
                _ => Nav::Pop,
            };
        }
        if self.task.is_finished() {
            // The worker died without a result.
            return Nav::Pop;
        }
        Nav::Stay
    }

    fn hints(&self, _app: &App) -> Vec<Hint> {
        vec![("Esc", "cancel".into())]
    }

    fn layer(&self) -> Layer {
        Layer::Popup {
            width: (self.title.chars().count() as u16 + 8).max(30),
            height: 3,
        }
    }
}

/// A read-only message box; Enter or Esc closes it.
pub(super) struct Info {
    title: String,
    lines: Vec<Line<'static>>,
}

impl Info {
    pub fn new(title: impl Into<String>, lines: Vec<Line<'static>>) -> Self {
        Self {
            title: title.into(),
            lines,
        }
    }
}

const INFO_WIDTH: u16 = 76;

impl Screen for Info {
    fn render(&mut self, _app: &App, frame: &mut Frame, area: Rect) {
        frame.render_widget(Clear, area);
        let block = theme::panel(self.title.clone());
        let inner = block.inner(area);
        frame.render_widget(block, area);
        frame.render_widget(
            Paragraph::new(self.lines.clone()).wrap(Wrap { trim: false }),
            inner,
        );
    }

    fn on_key(&mut self, _app: &mut App, key: KeyEvent) -> Nav {
        match key.code {
            KeyCode::Enter | KeyCode::Esc => Nav::Pop,
            _ => Nav::Stay,
        }
    }

    fn hints(&self, _app: &App) -> Vec<Hint> {
        vec![("Enter", "close".into())]
    }

    fn layer(&self) -> Layer {
        let inner = (INFO_WIDTH - 4) as usize;
        let rows: usize = self
            .lines
            .iter()
            .map(|l| l.width().div_ceil(inner).max(1))
            .sum();
        Layer::Popup {
            width: INFO_WIDTH,
            height: rows as u16 + 2,
        }
    }
}
