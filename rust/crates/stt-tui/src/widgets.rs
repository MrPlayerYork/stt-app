//! Reusable inputs: a line editor ([`LineEdit`], optionally masked for secrets) and three popup overlays
//! built on it — [`PickList`], [`TextPrompt`] and [`Confirm`]. Each popup reports its result through a
//! callback `FnOnce(&mut App, ...) -> Nav` that runs after the popup closes itself, so the caller just
//! writes `app.cfg`, saves, and returns `Nav::Stay` (or pushes a follow-up popup).

use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};
use ratatui::Frame;
use ratatui::layout::Rect;
use ratatui::style::Style;
use ratatui::text::{Line, Span};
use ratatui::widgets::{Clear, Paragraph, Wrap};

use crate::app::App;
use crate::screen::{Hint, Layer, Nav, Screen};
use crate::theme;

/// Single-line text editing state. Paste-safe: [`LineEdit::paste`] inserts text verbatim (newlines
/// dropped), never interpreting it as keys.
#[derive(Debug, Clone, Default)]
pub struct LineEdit {
    chars: Vec<char>,
    cursor: usize,
    /// Show `●` per character instead of the text (API keys, tokens).
    pub masked: bool,
}

/// What a key did to a [`LineEdit`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EditResult {
    /// Text or cursor changed (or the key was swallowed).
    Edited,
    /// Enter.
    Submit,
    /// Esc.
    Cancel,
    /// Not an editing key (e.g. Up/Down, Tab): the owner may use it.
    Unhandled,
}

impl LineEdit {
    /// Editor holding `text`, cursor at the end.
    pub fn new(text: &str) -> Self {
        let chars: Vec<char> = text.chars().collect();
        Self {
            cursor: chars.len(),
            chars,
            masked: false,
        }
    }

    /// Masked editor (starts empty).
    pub fn masked() -> Self {
        Self {
            masked: true,
            ..Self::default()
        }
    }

    /// Current text.
    pub fn text(&self) -> String {
        self.chars.iter().collect()
    }

    /// Number of characters (shown for masked input: "(32 characters)").
    pub fn len(&self) -> usize {
        self.chars.len()
    }

    /// True when empty.
    pub fn is_empty(&self) -> bool {
        self.chars.is_empty()
    }

    /// Insert pasted text at the cursor (line breaks and control characters dropped).
    pub fn paste(&mut self, text: &str) {
        for c in text.chars().filter(|c| !c.is_control()) {
            self.chars.insert(self.cursor, c);
            self.cursor += 1;
        }
    }

    /// Apply a key.
    pub fn on_key(&mut self, key: KeyEvent) -> EditResult {
        let ctrl = key.modifiers.contains(KeyModifiers::CONTROL);
        match key.code {
            KeyCode::Enter => EditResult::Submit,
            KeyCode::Esc => EditResult::Cancel,
            KeyCode::Char('u') if ctrl => {
                self.chars.drain(..self.cursor);
                self.cursor = 0;
                EditResult::Edited
            }
            KeyCode::Char('a') if ctrl => {
                self.cursor = 0;
                EditResult::Edited
            }
            KeyCode::Char('e') if ctrl => {
                self.cursor = self.chars.len();
                EditResult::Edited
            }
            KeyCode::Char(_) if ctrl => EditResult::Unhandled,
            KeyCode::Char(c) => {
                self.chars.insert(self.cursor, c);
                self.cursor += 1;
                EditResult::Edited
            }
            KeyCode::Backspace => {
                if self.cursor > 0 {
                    self.cursor -= 1;
                    self.chars.remove(self.cursor);
                }
                EditResult::Edited
            }
            KeyCode::Delete => {
                if self.cursor < self.chars.len() {
                    self.chars.remove(self.cursor);
                }
                EditResult::Edited
            }
            KeyCode::Left => {
                self.cursor = self.cursor.saturating_sub(1);
                EditResult::Edited
            }
            KeyCode::Right => {
                self.cursor = (self.cursor + 1).min(self.chars.len());
                EditResult::Edited
            }
            KeyCode::Home => {
                self.cursor = 0;
                EditResult::Edited
            }
            KeyCode::End => {
                self.cursor = self.chars.len();
                EditResult::Edited
            }
            _ => EditResult::Unhandled,
        }
    }

    /// The line to draw: text (or `●`s) with a reversed-cell cursor.
    pub fn line(&self) -> Line<'static> {
        let shown: Vec<char> = if self.masked {
            vec!['●'; self.chars.len()]
        } else {
            self.chars.clone()
        };
        let before: String = shown[..self.cursor].iter().collect();
        let at: String = shown.get(self.cursor).map_or(" ".into(), |c| c.to_string());
        let after: String = shown
            .get(self.cursor + 1..)
            .map_or(String::new(), |s| s.iter().collect());
        Line::from(vec![
            Span::raw(before),
            Span::styled(at, theme::selected()),
            Span::raw(after),
        ])
    }
}

type PickFn = Box<dyn FnOnce(&mut App, usize) -> Nav>;
type TextFn = Box<dyn FnOnce(&mut App, String) -> Nav>;
type ConfirmFn = Box<dyn FnOnce(&mut App) -> Nav>;

fn popup_frame(frame: &mut Frame, area: Rect, title: &str) -> Rect {
    frame.render_widget(Clear, area);
    let block = theme::panel(title);
    let inner = block.inner(area);
    frame.render_widget(block, area);
    inner
}

/// Numbered list popup (Python `pick`). Keys: ↑↓/j/k move, Enter picks, 1-9 pick directly, Esc cancels.
/// The current value is marked "(current)".
pub struct PickList {
    title: String,
    options: Vec<String>,
    current: Option<usize>,
    sel: usize,
    help: String,
    on_pick: Option<PickFn>,
}

impl PickList {
    /// `on_pick(app, index)` runs after the popup closes; return what should happen next.
    pub fn new(
        title: impl Into<String>,
        options: Vec<String>,
        current: Option<usize>,
        on_pick: impl FnOnce(&mut App, usize) -> Nav + 'static,
    ) -> Self {
        Self {
            title: title.into(),
            sel: current.unwrap_or(0),
            options,
            current,
            help: String::new(),
            on_pick: Some(Box::new(on_pick)),
        }
    }

    /// A dim help line under the list.
    pub fn help(mut self, text: impl Into<String>) -> Self {
        self.help = text.into();
        self
    }

    /// Selected index (tests).
    pub fn selected(&self) -> usize {
        self.sel
    }

    fn pick(&mut self, app: &mut App, i: usize) -> Nav {
        match self.on_pick.take() {
            Some(f) => after_pop(f(app, i)),
            None => Nav::Pop,
        }
    }
}

/// Turn a callback's `Nav` into the popup's own result: the popup always closes first.
fn after_pop(next: Nav) -> Nav {
    match next {
        Nav::Push(s) | Nav::Replace(s) => Nav::Replace(s),
        Nav::SwitchTab(t) => Nav::SwitchTab(t),
        Nav::Quit => Nav::Quit,
        _ => Nav::Pop,
    }
}

impl Screen for PickList {
    fn render(&mut self, _app: &App, frame: &mut Frame, area: Rect) {
        let inner = popup_frame(frame, area, &self.title);
        let mut lines: Vec<Line> = self
            .options
            .iter()
            .enumerate()
            .map(|(i, o)| {
                let num = if i < 9 {
                    format!("{} ", i + 1)
                } else {
                    "  ".into()
                };
                let mut spans = vec![
                    Span::raw(if i == self.sel { theme::CURSOR } else { "  " }),
                    Span::styled(num, Style::new().fg(theme::KEY)),
                    Span::styled(
                        o.clone(),
                        if i == self.sel {
                            theme::selected()
                        } else {
                            Style::new()
                        },
                    ),
                ];
                if Some(i) == self.current {
                    spans.push(Span::styled(" (current)", Style::new().fg(theme::OK)));
                }
                Line::from(spans)
            })
            .collect();
        // Keep the selection visible in a short popup.
        let h = inner.height as usize;
        let skip = if self.help.is_empty() {
            h
        } else {
            h.saturating_sub(2)
        };
        let skip = self.sel.saturating_sub(skip.saturating_sub(1));
        lines.drain(..skip.min(lines.len()));
        let visible = if self.help.is_empty() {
            h
        } else {
            h.saturating_sub(2)
        };
        for (line, i) in (skip..self.options.len()).take(visible).enumerate() {
            crate::click::add_row(inner, line, i);
        }
        if !self.help.is_empty() {
            lines.push(Line::default());
            lines.push(Line::styled(self.help.clone(), theme::dim()));
        }
        frame.render_widget(Paragraph::new(lines), inner);
    }

    fn on_key(&mut self, app: &mut App, key: KeyEvent) -> Nav {
        match key.code {
            KeyCode::Up | KeyCode::Char('k') => {
                self.sel = self.sel.saturating_sub(1);
                Nav::Stay
            }
            KeyCode::Down | KeyCode::Char('j') => {
                self.sel = (self.sel + 1).min(self.options.len().saturating_sub(1));
                Nav::Stay
            }
            KeyCode::Enter if !self.options.is_empty() => self.pick(app, self.sel),
            KeyCode::Char(c @ '1'..='9') => {
                let i = c as usize - '1' as usize;
                if i < self.options.len() {
                    self.pick(app, i)
                } else {
                    Nav::Stay
                }
            }
            KeyCode::Esc => Nav::Pop,
            _ => Nav::Stay,
        }
    }

    fn on_click(&mut self, app: &mut App, i: usize) -> Nav {
        if i < self.options.len() {
            self.pick(app, i)
        } else {
            Nav::Stay
        }
    }

    fn hints(&self, _app: &App) -> Vec<Hint> {
        vec![
            ("↑↓", "move".into()),
            ("Enter", "choose".into()),
            ("1-9", "pick".into()),
            ("Esc", "cancel".into()),
        ]
    }

    fn layer(&self) -> Layer {
        let w = self
            .options
            .iter()
            .map(|o| o.chars().count())
            .max()
            .unwrap_or(10) as u16;
        let help = if self.help.is_empty() { 0 } else { 2 };
        Layer::Popup {
            width: (w + 18).max(self.help.chars().count() as u16 + 4).max(36),
            height: self.options.len() as u16 + 2 + help,
        }
    }
}

/// Text entry popup. Enter submits (the callback gets the text, possibly empty), Esc cancels. Masked
/// mode shows `●` per character and "(N characters)", never the value (Python `ask_secret`).
pub struct TextPrompt {
    title: String,
    help: String,
    edit: LineEdit,
    on_submit: Option<TextFn>,
}

impl TextPrompt {
    /// Prompt pre-filled with `initial`.
    pub fn new(
        title: impl Into<String>,
        initial: &str,
        on_submit: impl FnOnce(&mut App, String) -> Nav + 'static,
    ) -> Self {
        Self {
            title: title.into(),
            help: String::new(),
            edit: LineEdit::new(initial),
            on_submit: Some(Box::new(on_submit)),
        }
    }

    /// Masked prompt for secrets (starts empty).
    pub fn secret(
        title: impl Into<String>,
        on_submit: impl FnOnce(&mut App, String) -> Nav + 'static,
    ) -> Self {
        Self {
            title: title.into(),
            help: String::new(),
            edit: LineEdit::masked(),
            on_submit: Some(Box::new(on_submit)),
        }
    }

    /// A dim help line under the input.
    pub fn help(mut self, text: impl Into<String>) -> Self {
        self.help = text.into();
        self
    }

    /// Current text (tests).
    pub fn text(&self) -> String {
        self.edit.text()
    }
}

impl Screen for TextPrompt {
    fn render(&mut self, _app: &App, frame: &mut Frame, area: Rect) {
        let inner = popup_frame(frame, area, &self.title);
        let mut lines = vec![self.edit.line()];
        if self.edit.masked && !self.edit.is_empty() {
            lines.push(Line::styled(
                format!("({} characters)", self.edit.len()),
                theme::dim(),
            ));
        }
        if !self.help.is_empty() {
            lines.push(Line::styled(self.help.clone(), theme::dim()));
        }
        frame.render_widget(Paragraph::new(lines).wrap(Wrap { trim: false }), inner);
    }

    fn on_key(&mut self, app: &mut App, key: KeyEvent) -> Nav {
        match self.edit.on_key(key) {
            EditResult::Submit => match self.on_submit.take() {
                Some(f) => after_pop(f(app, self.edit.text())),
                None => Nav::Pop,
            },
            EditResult::Cancel => Nav::Pop,
            // Swallow everything else so Tab/q never leak to the runner while typing.
            EditResult::Edited | EditResult::Unhandled => Nav::Stay,
        }
    }

    fn on_paste(&mut self, _app: &mut App, text: &str) -> Nav {
        self.edit.paste(text);
        Nav::Stay
    }

    fn hints(&self, _app: &App) -> Vec<Hint> {
        vec![("Enter", "save".into()), ("Esc", "cancel".into())]
    }

    fn layer(&self) -> Layer {
        Layer::Popup {
            width: 72,
            height: 6,
        }
    }
}

/// Yes/no popup: `y` or Enter confirms, `n`/Esc cancels.
pub struct Confirm {
    question: String,
    on_yes: Option<ConfirmFn>,
}

impl Confirm {
    /// Ask `question`; `on_yes` runs after the popup closes.
    pub fn new(
        question: impl Into<String>,
        on_yes: impl FnOnce(&mut App) -> Nav + 'static,
    ) -> Self {
        Self {
            question: question.into(),
            on_yes: Some(Box::new(on_yes)),
        }
    }
}

impl Screen for Confirm {
    fn render(&mut self, _app: &App, frame: &mut Frame, area: Rect) {
        let inner = popup_frame(frame, area, "Confirm");
        // The y/n line sits after the (wrapped) question and a blank line.
        let q_lines = (self.question.chars().count() as u16).div_ceil(inner.width.max(1));
        crate::click::add_hints(
            inner.x,
            inner.y + q_lines + 1,
            &[("y", "yes".into()), ("n", "no".into())],
            false,
        );
        frame.render_widget(
            Paragraph::new(vec![
                Line::raw(self.question.clone()),
                Line::default(),
                Line::from(theme::hint_spans(&[
                    ("y", "yes".into()),
                    ("n", "no".into()),
                ])),
            ])
            .wrap(Wrap { trim: false }),
            inner,
        );
    }

    fn on_key(&mut self, app: &mut App, key: KeyEvent) -> Nav {
        match key.code {
            KeyCode::Char('y' | 'Y') | KeyCode::Enter => match self.on_yes.take() {
                Some(f) => after_pop(f(app)),
                None => Nav::Pop,
            },
            KeyCode::Char('n' | 'N') | KeyCode::Esc => Nav::Pop,
            _ => Nav::Stay,
        }
    }

    fn layer(&self) -> Layer {
        Layer::Popup {
            width: (self.question.chars().count() as u16 + 6).clamp(30, 72),
            height: 6,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testkit::{key, test_app};

    #[test]
    fn line_edit_paste_is_verbatim() {
        let mut e = LineEdit::masked();
        e.paste("sk-abc\nq\t");
        assert_eq!(e.text(), "sk-abcq");
        assert_eq!(e.on_key(key(KeyCode::Backspace)), EditResult::Edited);
        assert_eq!(e.text(), "sk-abc");
        let shown: String = e
            .line()
            .spans
            .iter()
            .map(|s| s.content.to_string())
            .collect();
        assert!(!shown.contains("sk"), "masked text never drawn: {shown}");
    }

    #[test]
    fn pick_list_number_key_runs_callback() {
        let (mut app, _h) = test_app();
        let mut p = PickList::new("Model", vec!["a".into(), "b".into()], Some(0), |app, i| {
            app.cfg.whisper.model = format!("picked {i}");
            Nav::Stay
        });
        assert!(matches!(
            p.on_key(&mut app, key(KeyCode::Char('2'))),
            Nav::Pop
        ));
        assert_eq!(app.cfg.whisper.model, "picked 1");
    }

    #[test]
    fn text_prompt_swallows_shortcuts_and_submits() {
        let (mut app, _h) = test_app();
        let mut t = TextPrompt::new("Lang", "", |app, s| {
            app.cfg.whisper.language = Some(s);
            Nav::Stay
        });
        for c in ['q', 'e', 'n'] {
            assert!(matches!(
                t.on_key(&mut app, key(KeyCode::Char(c))),
                Nav::Stay
            ));
        }
        assert!(matches!(t.on_key(&mut app, key(KeyCode::Tab)), Nav::Stay));
        assert!(matches!(t.on_key(&mut app, key(KeyCode::Enter)), Nav::Pop));
        assert_eq!(app.cfg.whisper.language.as_deref(), Some("qen"));
    }
}
