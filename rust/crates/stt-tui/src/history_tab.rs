//! History tab (A5): newest-first list with a preview pane, copy on Enter / number keys, reasoning view.

use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};
use ratatui::Frame;
use ratatui::layout::{Constraint, Layout, Rect};
use ratatui::text::Line;
use ratatui::widgets::{Paragraph, Wrap};
use stt_core::ClipboardMethod;

use crate::app::{App, Tab};
use crate::main_tab::history_row;
use crate::reasoning_view::ReasoningView;
use crate::screen::{Hint, Nav, Screen};
use crate::task::Task;
use crate::theme;

/// The History tab.
#[derive(Default)]
pub struct HistoryTab {
    sel: usize,
    copying: Option<Task<Result<ClipboardMethod, String>>>,
}

impl HistoryTab {
    /// History tab.
    pub fn new() -> Self {
        Self::default()
    }

    fn copy(&mut self, app: &mut App, idx: usize) {
        let Some(item) = app.history.get(idx) else {
            return;
        };
        let text = item.final_text.clone();
        let clip = app.services.clipboard.clone();
        self.copying = Some(Task::spawn(move || {
            clip.copy(&text).map_err(|e| e.to_string())
        }));
    }

    fn selected_reasoning(&self, app: &App) -> Option<String> {
        app.history
            .get(self.sel)
            .map(|i| i.reasoning.clone())
            .filter(|r| !r.is_empty())
    }
}

impl Screen for HistoryTab {
    fn render(&mut self, app: &App, frame: &mut Frame, area: Rect) {
        if app.history.is_empty() {
            let r = theme::centered(area, area.width, 1);
            frame.render_widget(
                Paragraph::new(Line::styled("No history yet.", theme::dim())).centered(),
                r,
            );
            return;
        }
        self.sel = self.sel.min(app.history.len() - 1);
        let [list_a, prev_a] =
            Layout::vertical([Constraint::Min(4), Constraint::Percentage(40)]).areas(area);

        let block = theme::panel("History (newest first)");
        let inner = block.inner(list_a);
        frame.render_widget(block, list_a);
        let rows = inner.height.max(1) as usize;
        let top = self.sel.saturating_sub(rows - 1);
        let lines: Vec<Line> = app
            .history
            .iter()
            .enumerate()
            .skip(top)
            .take(rows)
            .map(|(i, it)| {
                let text = history_row(it, Some(i + 1), (inner.width as usize).saturating_sub(2));
                if i == self.sel {
                    Line::styled(format!("{}{text}", theme::CURSOR), theme::selected())
                } else {
                    Line::raw(format!("  {text}"))
                }
            })
            .collect();
        frame.render_widget(Paragraph::new(lines), inner);
        for (line, i) in (top..app.history.len()).take(rows).enumerate() {
            crate::click::add_row(inner, line, i);
        }

        let block = theme::panel("Preview");
        let inner = block.inner(prev_a);
        frame.render_widget(block, prev_a);
        if let Some(item) = app.history.get(self.sel) {
            frame.render_widget(
                Paragraph::new(item.final_text.clone()).wrap(Wrap { trim: false }),
                inner,
            );
        }
    }

    /// Clicking an item selects it (Enter or `[Enter/c] copy` then copies it).
    fn on_click(&mut self, app: &mut App, i: usize) -> Nav {
        if i < app.history.len() {
            self.sel = i;
        }
        Nav::Stay
    }

    fn on_key(&mut self, app: &mut App, key: KeyEvent) -> Nav {
        if key.modifiers.contains(KeyModifiers::CONTROL) {
            return Nav::Ignored;
        }
        let n = app.history.len();
        match key.code {
            KeyCode::Up | KeyCode::Char('k') => self.sel = self.sel.saturating_sub(1),
            KeyCode::Down | KeyCode::Char('j') => {
                self.sel = (self.sel + 1).min(n.saturating_sub(1));
            }
            KeyCode::Home => self.sel = 0,
            KeyCode::End => self.sel = n.saturating_sub(1),
            KeyCode::Enter | KeyCode::Char('c') => {
                let i = self.sel;
                self.copy(app, i);
            }
            KeyCode::Char(c @ '1'..='9') => {
                let i = c as usize - '1' as usize;
                if i < n {
                    self.sel = i;
                    self.copy(app, i);
                }
            }
            KeyCode::Char('v') => {
                return match self.selected_reasoning(app) {
                    Some(t) => Nav::Push(Box::new(ReasoningView::new(t))),
                    None => Nav::Stay,
                };
            }
            KeyCode::Esc => return Nav::SwitchTab(Tab::Main),
            _ => return Nav::Ignored,
        }
        Nav::Stay
    }

    fn on_tick(&mut self, app: &mut App) -> Nav {
        if let Some(task) = &mut self.copying
            && let Some(res) = task.poll()
        {
            self.copying = None;
            match res {
                Ok(m) => app.flash_ok(format!("Copied to clipboard ({m})")),
                Err(e) => app.flash_error(format!("Clipboard error: {e}")),
            }
        }
        Nav::Stay
    }

    fn hints(&self, app: &App) -> Vec<Hint> {
        if app.history.is_empty() {
            return Vec::new();
        }
        let mut h: Vec<Hint> = vec![
            ("↑↓", "move".into()),
            ("Enter/c", "copy".into()),
            ("1-9", "copy that number".into()),
        ];
        if self.selected_reasoning(app).is_some() {
            h.push(("v", "reasoning".into()));
        }
        h.push(("Esc", "back".into()));
        h
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::app::HistoryItem;
    use crate::testkit::*;

    fn item(final_text: &str, reasoning: &str) -> HistoryItem {
        HistoryItem {
            at: chrono::Local::now(),
            raw: "raw".into(),
            final_text: final_text.into(),
            mode: "Cleanup".into(),
            reasoning: reasoning.into(),
        }
    }

    fn pump(tab: &mut HistoryTab, app: &mut App) {
        for _ in 0..200 {
            tab.on_tick(app);
            if tab.copying.is_none() {
                return;
            }
            std::thread::sleep(std::time::Duration::from_millis(5));
        }
        panic!("copy did not finish");
    }

    #[test]
    fn empty_message() {
        let (app, _h) = test_app();
        let out = render_screen(&mut HistoryTab::new(), &app, 80, 24);
        assert!(out.contains("No history yet."), "{out}");
    }

    #[test]
    fn lists_and_previews_selected() {
        let (mut app, _h) = test_app();
        app.push_history(item("older text", ""));
        app.push_history(item("newest text", ""));
        let mut tab = HistoryTab::new();
        let out = render_screen(&mut tab, &app, 80, 24);
        assert!(out.contains("› 1"), "{out}");
        assert!(out.contains("Preview"), "{out}");
        tab.on_key(&mut app, ch('j'));
        assert_eq!(tab.sel, 1);
        tab.on_key(&mut app, ch('j'));
        assert_eq!(tab.sel, 1, "clamped");
    }

    #[test]
    fn enter_copies_selected() {
        let (mut app, h) = test_app();
        app.push_history(item("older text", ""));
        app.push_history(item("newest text", ""));
        let mut tab = HistoryTab::new();
        tab.on_key(&mut app, key(KeyCode::Down));
        tab.on_key(&mut app, key(KeyCode::Enter));
        pump(&mut tab, &mut app);
        assert_eq!(h.clipboard.copies(), vec!["older text".to_string()]);
        assert!(app.flash.as_ref().unwrap().text.starts_with("Copied"));
    }

    #[test]
    fn number_key_copies_that_item() {
        let (mut app, h) = test_app();
        app.push_history(item("first", ""));
        app.push_history(item("second", ""));
        let mut tab = HistoryTab::new();
        tab.on_key(&mut app, ch('2'));
        pump(&mut tab, &mut app);
        assert_eq!(h.clipboard.copies(), vec!["first".to_string()]);
        tab.on_key(&mut app, ch('9'));
        assert!(tab.copying.is_none(), "out of range is ignored");
    }

    #[test]
    fn v_only_with_reasoning() {
        let (mut app, _h) = test_app();
        app.push_history(item("a", ""));
        app.push_history(item("b", "thinking hard"));
        let mut tab = HistoryTab::new();
        assert!(tab.hints(&app).iter().any(|(k, _)| *k == "v"));
        assert!(matches!(tab.on_key(&mut app, ch('v')), Nav::Push(_)));
        tab.on_key(&mut app, ch('j'));
        assert!(!tab.hints(&app).iter().any(|(k, _)| *k == "v"));
        assert!(matches!(tab.on_key(&mut app, ch('v')), Nav::Stay));
    }
}
