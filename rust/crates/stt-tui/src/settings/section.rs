//! The shared row-list screen used by the section list and every section (A6-A8): numbered rows with a
//! `›` cursor, ←→ cycling for fixed choices, a status area and a help box for the selected row.

use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};
use ratatui::Frame;
use ratatui::layout::{Constraint, Layout, Rect};
use ratatui::style::Style;
use ratatui::text::{Line, Span};
use ratatui::widgets::{Paragraph, Wrap};

use crate::app::App;
use crate::screen::{Hint, Nav, Screen};
use crate::theme;

/// One displayed row.
pub(super) struct Row {
    /// Jump key (`1`-`9`, or a letter) shown before the label.
    pub key: char,
    /// Label column.
    pub label: &'static str,
    /// Current value (may be empty for action rows).
    pub value: String,
    /// Fixed-choice or toggle row: shown as `◂ value ▸` while selected, ←→ change it.
    pub choice: bool,
}

impl Row {
    pub fn new(key: char, label: &'static str, value: impl Into<String>) -> Self {
        Self {
            key,
            label,
            value: value.into(),
            choice: false,
        }
    }

    pub fn choice(mut self) -> Self {
        self.choice = true;
        self
    }
}

/// What a section provides; [`SectionScreen`] does the drawing and key handling.
pub(super) trait Section {
    /// Overlay title (the runner shows `› title`).
    fn title(&self) -> String;
    /// Rows, derived from `app` on every render.
    fn rows(&self, app: &App) -> Vec<Row>;
    /// Help text for row `i`.
    fn help(&self, app: &App, i: usize) -> String;
    /// Enter / jump key on row `i`.
    fn activate(&mut self, app: &mut App, i: usize) -> Nav;
    /// ←→ on row `i` (`dir` is -1 or 1).
    fn cycle(&mut self, _app: &mut App, _i: usize, _dir: i32) {}
    /// Extra lines between the rows and the help box (progress, results). `width` is the usable width.
    fn status(&self, _app: &App, _width: u16) -> Vec<Line<'static>> {
        Vec::new()
    }
    /// Poll background tasks.
    fn tick(&mut self, _app: &mut App) -> Nav {
        Nav::Stay
    }
    /// The section is closing (Esc).
    fn close(&mut self, _app: &mut App) {}
    /// Hint for Enter.
    fn enter_hint(&self) -> &'static str {
        "edit"
    }
}

/// Generic screen over a [`Section`]. As a tab (`is_tab`) it passes Esc/Tab on to the runner.
pub(super) struct SectionScreen<S: Section> {
    section: S,
    sel: usize,
    is_tab: bool,
}

impl<S: Section> SectionScreen<S> {
    pub fn overlay(section: S) -> Self {
        Self {
            section,
            sel: 0,
            is_tab: false,
        }
    }

    pub fn tab(section: S) -> Self {
        Self {
            section,
            sel: 0,
            is_tab: true,
        }
    }
}

const LABEL_W: usize = 28;

impl<S: Section> Screen for SectionScreen<S> {
    fn render(&mut self, app: &App, frame: &mut Frame, area: Rect) {
        let rows = self.section.rows(app);
        self.sel = self.sel.min(rows.len().saturating_sub(1));
        let status = self.section.status(app, area.width);
        let help = self.section.help(app, self.sel);
        let [list, status_area, help_area] = Layout::vertical([
            Constraint::Min(rows.len() as u16),
            Constraint::Length(status.len() as u16),
            Constraint::Length(4),
        ])
        .areas(area);

        let lines: Vec<Line> = rows
            .iter()
            .enumerate()
            .map(|(i, r)| {
                let on = i == self.sel;
                let value = if on && r.choice {
                    format!("◂ {} ▸", r.value)
                } else {
                    r.value.clone()
                };
                Line::from(vec![
                    Span::raw(if on { theme::CURSOR } else { "  " }),
                    Span::styled(format!("{} ", r.key), Style::new().fg(theme::KEY)),
                    Span::styled(
                        format!("{:<w$}", r.label, w = LABEL_W),
                        if on {
                            theme::selected()
                        } else {
                            theme::label()
                        },
                    ),
                    Span::raw(" "),
                    Span::raw(value),
                ])
            })
            .collect();
        frame.render_widget(Paragraph::new(lines), list);
        for i in 0..rows.len() {
            crate::click::add_row(list, i, i);
        }
        if !status.is_empty() {
            frame.render_widget(Paragraph::new(status), status_area);
        }
        let block = theme::panel("About this setting");
        let inner = block.inner(help_area);
        frame.render_widget(block, help_area);
        frame.render_widget(
            Paragraph::new(Line::styled(help, theme::dim())).wrap(Wrap { trim: false }),
            inner,
        );
    }

    fn on_key(&mut self, app: &mut App, key: KeyEvent) -> Nav {
        let ctrl = key.modifiers.contains(KeyModifiers::CONTROL);
        if key.code == KeyCode::Esc || (ctrl && key.code == KeyCode::Char('c')) {
            if self.is_tab {
                return Nav::Ignored;
            }
            self.section.close(app);
            return Nav::Pop;
        }
        let rows = self.section.rows(app);
        match key.code {
            KeyCode::Up | KeyCode::Char('k') if !ctrl => {
                self.sel = self.sel.saturating_sub(1);
                Nav::Stay
            }
            KeyCode::Down | KeyCode::Char('j') if !ctrl => {
                self.sel = (self.sel + 1).min(rows.len().saturating_sub(1));
                Nav::Stay
            }
            KeyCode::Left => {
                self.section.cycle(app, self.sel, -1);
                Nav::Stay
            }
            KeyCode::Right => {
                self.section.cycle(app, self.sel, 1);
                Nav::Stay
            }
            KeyCode::Enter => self.section.activate(app, self.sel),
            KeyCode::Char(c) if !ctrl => match rows.iter().position(|r| r.key == c) {
                Some(i) => {
                    self.sel = i;
                    self.section.activate(app, i)
                }
                None if self.is_tab => Nav::Ignored,
                None => Nav::Stay,
            },
            _ if self.is_tab => Nav::Ignored,
            _ => Nav::Stay,
        }
    }

    fn on_tick(&mut self, app: &mut App) -> Nav {
        self.section.tick(app)
    }

    /// Clicking a row selects it; a fixed-choice row cycles to the next value, any other row acts
    /// like Enter (opens its picker, prompt or sub-section).
    fn on_click(&mut self, app: &mut App, i: usize) -> Nav {
        let rows = self.section.rows(app);
        let Some(row) = rows.get(i) else {
            return Nav::Stay;
        };
        self.sel = i;
        if row.choice {
            self.section.cycle(app, i, 1);
            Nav::Stay
        } else {
            self.section.activate(app, i)
        }
    }

    fn hints(&self, app: &App) -> Vec<Hint> {
        let mut h = vec![("↑↓", "move".to_string())];
        if self.section.rows(app).iter().any(|r| r.choice) {
            h.push(("←→", "change".into()));
        }
        h.push(("Enter", self.section.enter_hint().into()));
        if !self.is_tab {
            h.push(("Esc", "back".into()));
        }
        h
    }

    fn title(&self, _app: &App) -> String {
        self.section.title()
    }
}
