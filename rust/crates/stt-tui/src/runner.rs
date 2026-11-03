//! The full-screen event loop and the outer frame (UI_MOCKUPS.txt section D).
//!
//! [`Ui`] holds the three tabs and the overlay stack and does all key routing and drawing; it has no
//! terminal of its own, so tests drive it with [`ratatui::backend::TestBackend`]. [`run`] wraps it in
//! the real terminal (alternate screen, raw mode, bracketed paste).

use std::io;
use std::time::Duration;

use crossterm::event::{self, Event, KeyCode, KeyEvent, KeyEventKind, KeyModifiers};
use ratatui::Frame;
use ratatui::layout::Rect;
use ratatui::style::{Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, BorderType, Borders};

use crate::app::{App, FlashKind, Tab};
use crate::mic::MicStatus;
use crate::screen::{Layer, Nav, Screen};
use crate::theme;

/// How often the loop wakes up to redraw and tick (meter, spinner, background tasks).
pub const TICK: Duration = Duration::from_millis(50);

/// Tabs + overlays + routing.
pub struct Ui {
    tabs: [Box<dyn Screen>; 3],
    overlays: Vec<Box<dyn Screen>>,
}

impl Ui {
    /// The real tabs (Main, History, Settings).
    pub fn new() -> Self {
        Self::with_tabs([
            Box::new(crate::main_tab::MainTab::new()),
            Box::new(crate::history_tab::HistoryTab::new()),
            Box::new(crate::settings::SettingsTab::new()),
        ])
    }

    /// Custom tabs (tests).
    pub fn with_tabs(tabs: [Box<dyn Screen>; 3]) -> Self {
        Self {
            tabs,
            overlays: Vec::new(),
        }
    }

    /// Open an overlay (startup screens: first run, config notice).
    pub fn push(&mut self, s: Box<dyn Screen>) {
        self.overlays.push(s);
    }

    /// Number of open overlays (tests).
    pub fn overlay_count(&self) -> usize {
        self.overlays.len()
    }

    fn top(&mut self, app: &App) -> &mut Box<dyn Screen> {
        match self.overlays.last_mut() {
            Some(s) => s,
            None => &mut self.tabs[app.tab.index()],
        }
    }

    fn top_ref(&self, app: &App) -> &dyn Screen {
        match self.overlays.last() {
            Some(s) => s.as_ref(),
            None => self.tabs[app.tab.index()].as_ref(),
        }
    }

    fn apply(&mut self, app: &mut App, nav: Nav, from_overlay: bool) {
        match nav {
            Nav::Ignored | Nav::Stay => {}
            Nav::Push(s) => self.overlays.push(s),
            Nav::Pop => {
                if from_overlay {
                    self.overlays.pop();
                }
            }
            Nav::Replace(s) => {
                if from_overlay {
                    self.overlays.pop();
                }
                self.overlays.push(s);
            }
            Nav::SwitchTab(t) => {
                self.overlays.clear();
                app.tab = t;
            }
            Nav::Quit => app.quit = true,
        }
    }

    /// Route a key press.
    pub fn on_key(&mut self, app: &mut App, key: KeyEvent) {
        let from_overlay = !self.overlays.is_empty();
        let nav = self.top(app).on_key(app, key);
        if !matches!(nav, Nav::Ignored) {
            return self.apply(app, nav, from_overlay);
        }
        let ctrl_c =
            key.code == KeyCode::Char('c') && key.modifiers.contains(KeyModifiers::CONTROL);
        match key.code {
            _ if ctrl_c && from_overlay => {
                self.overlays.pop();
            }
            _ if ctrl_c => app.quit = true,
            KeyCode::Esc if from_overlay => {
                self.overlays.pop();
            }
            KeyCode::Tab if !from_overlay => app.tab = app.tab.next(),
            KeyCode::BackTab if !from_overlay => app.tab = app.tab.prev(),
            _ => {}
        }
    }

    /// Route pasted text to the top screen.
    pub fn on_paste(&mut self, app: &mut App, text: &str) {
        let from_overlay = !self.overlays.is_empty();
        let nav = self.top(app).on_paste(app, text);
        self.apply(app, nav, from_overlay);
    }

    /// Tick background state and the top screen.
    pub fn tick(&mut self, app: &mut App) {
        app.tick();
        let from_overlay = !self.overlays.is_empty();
        let nav = self.top(app).on_tick(app);
        self.apply(app, nav, from_overlay);
    }

    /// Draw everything.
    pub fn draw(&mut self, app: &App, frame: &mut Frame) {
        let area = frame.area();
        let recording = self.top_ref(app).is_recording(app);
        let color = if recording { theme::REC } else { theme::BORDER };

        let mut top: Vec<Span> = vec![Span::raw(" ")];
        if recording {
            let secs = match app.mic.status() {
                MicStatus::Recording { since } => since.elapsed().as_secs_f32(),
                _ => 0.0,
            };
            let rec = format!("● REC {}:{:04.1} ", (secs / 60.0) as u32, secs % 60.0);
            top.push(Span::styled(
                rec,
                Style::new().fg(theme::REC).add_modifier(Modifier::BOLD),
            ));
        } else {
            for (i, t) in Tab::ALL.iter().enumerate() {
                if i > 0 {
                    top.push(Span::styled("│", Style::new().fg(color)));
                }
                let style = if *t == app.tab {
                    Style::new().add_modifier(Modifier::BOLD | Modifier::REVERSED)
                } else {
                    theme::dim()
                };
                top.push(Span::styled(format!(" {} ", t.title()), style));
            }
            if let Some(o) = self
                .overlays
                .iter()
                .rev()
                .find(|o| o.layer() == Layer::Full)
            {
                let title = o.title(app);
                if !title.is_empty() {
                    top.push(Span::styled(format!(" › {title} "), theme::label()));
                }
            }
        }
        let right = match &app.flash {
            Some(f) => {
                let c = match f.kind {
                    FlashKind::Ok => theme::OK,
                    FlashKind::Warn => theme::WARN,
                    FlashKind::Error => theme::ERR,
                };
                Line::from(Span::styled(format!(" {} ", f.text), Style::new().fg(c)))
            }
            None => Line::styled(format!(" stt {} ", env!("CARGO_PKG_VERSION")), theme::dim()),
        };
        let mut bottom = vec![Span::raw(" ")];
        bottom.extend(theme::hint_spans(&self.top_ref(app).hints(app)));
        if self.overlays.is_empty() && !recording {
            bottom.extend(theme::hint_spans(&[("Tab", "switch".into())]));
        }
        let block = Block::new()
            .borders(Borders::ALL)
            .border_type(BorderType::Rounded)
            .border_style(Style::new().fg(color))
            .title_top(Line::from(top))
            .title_top(right.right_aligned())
            .title_bottom(Line::from(bottom));
        let body = block.inner(area);
        frame.render_widget(block, area);
        let body = Rect {
            x: body.x + 1,
            width: body.width.saturating_sub(2),
            ..body
        };

        let first_full = self.overlays.iter().rposition(|o| o.layer() == Layer::Full);
        if first_full.is_none() {
            self.tabs[app.tab.index()].render(app, frame, body);
        }
        for o in &mut self.overlays[first_full.unwrap_or(0)..] {
            let r = match o.layer() {
                Layer::Full => body,
                Layer::Popup { width, height } => theme::centered(body, width, height),
            };
            o.render(app, frame, r);
        }
    }
}

impl Default for Ui {
    fn default() -> Self {
        Self::new()
    }
}

/// Run the app full screen until it quits. `startup` overlays are shown first, first element on top
/// (e.g. a config notice, then first-run calibration). Restores the terminal on return and on panic.
pub fn run(app: &mut App, startup: Vec<Box<dyn Screen>>) -> io::Result<()> {
    let mut ui = Ui::new();
    for s in startup.into_iter().rev() {
        ui.push(s);
    }
    let mut terminal = ratatui::init();
    crossterm::execute!(io::stdout(), crossterm::event::EnableBracketedPaste)?;
    let res = (|| -> io::Result<()> {
        while !app.quit {
            terminal.draw(|f| ui.draw(app, f))?;
            if event::poll(TICK)? {
                match event::read()? {
                    Event::Key(k) if k.kind != KeyEventKind::Release => ui.on_key(app, k),
                    Event::Paste(text) => ui.on_paste(app, &text),
                    _ => {}
                }
            }
            ui.tick(app);
        }
        Ok(())
    })();
    let _ = crossterm::execute!(io::stdout(), crossterm::event::DisableBracketedPaste);
    ratatui::restore();
    res
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testkit::{ctrl, key, render_ui, test_app};

    #[test]
    fn tab_switches_and_ctrl_c_quits() {
        let (mut app, _h) = test_app();
        let mut ui = Ui::new();
        ui.on_key(&mut app, key(KeyCode::Tab));
        assert_eq!(app.tab, Tab::History);
        ui.on_key(&mut app, key(KeyCode::BackTab));
        assert_eq!(app.tab, Tab::Main);
        ui.on_key(&mut app, ctrl('c'));
        assert!(app.quit);
    }

    #[test]
    fn frame_shows_tabs_and_version() {
        let (app, _h) = test_app();
        let mut ui = Ui::new();
        let s = render_ui(&mut ui, &app, 80, 24);
        assert!(
            s.contains("Main") && s.contains("History") && s.contains("Settings"),
            "{s}"
        );
        assert!(s.contains("stt 0.1.0"), "{s}");
    }

    #[test]
    fn popup_esc_closes() {
        let (mut app, _h) = test_app();
        let mut ui = Ui::new();
        ui.push(Box::new(crate::widgets::Confirm::new("Sure?", |_| {
            Nav::Stay
        })));
        ui.on_key(&mut app, key(KeyCode::Esc));
        assert_eq!(ui.overlay_count(), 0);
    }
}
