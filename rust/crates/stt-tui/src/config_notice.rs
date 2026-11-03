//! Startup warning for a recovered config (UI_MOCKUPS.txt A9): backup path, what was kept, what was
//! reset, what still needs a value. `[Enter]` continues, `[s]` opens the Settings tab.

use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};
use ratatui::Frame;
use ratatui::layout::Rect;
use ratatui::style::{Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, BorderType, Borders, Padding, Paragraph, Wrap};
use stt_core::recover::{ConfigIssue, ConfigReport};

use crate::app::{App, Tab};
use crate::screen::{Hint, Layer, Nav, Screen};
use crate::theme;

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

fn issues(lines: &mut Vec<Line<'static>>, title: &str, items: &[ConfigIssue]) {
    if items.is_empty() {
        return;
    }
    lines.push(Line::from(""));
    lines.push(Line::styled(
        title.to_string(),
        Style::new().fg(theme::WARN).add_modifier(Modifier::BOLD),
    ));
    for i in items {
        let found = if i.found.is_empty() {
            "(missing)".to_string()
        } else {
            format!("\"{}\"", theme::ellipsize(&i.found, 30))
        };
        lines.push(Line::from(format!(
            "  {}  {} -> {}",
            i.key, found, i.action
        )));
    }
}

impl Screen for ConfigNoticeScreen {
    fn render(&mut self, _app: &App, frame: &mut Frame, area: Rect) {
        let r = &self.report;
        let mut lines = vec![Line::from(vec![
            Span::styled("Config  ", theme::label()),
            Span::raw(r.path.display().to_string()),
        ])];
        if let Some(b) = &r.backup {
            lines.push(Line::from(vec![
                Span::styled("Backup  ", theme::label()),
                Span::raw(b.display().to_string()),
            ]));
        }
        if let Some(e) = &r.parse_error {
            lines.push(Line::from(""));
            lines.push(Line::styled(
                format!("The file could not be parsed: {}", theme::ellipsize(e, 200)),
                Style::new().fg(theme::ERR),
            ));
        }
        lines.push(Line::from(""));
        let kept = Style::new().fg(theme::OK).add_modifier(Modifier::BOLD);
        if r.kept.is_empty() {
            lines.push(Line::from(vec![
                Span::styled("Kept  ", kept),
                Span::raw("nothing"),
            ]));
        } else {
            let width = (area.width as usize).saturating_sub(24).max(20);
            lines.push(Line::from(vec![
                Span::styled("Kept  ", kept),
                Span::raw(format!(
                    "{} values: {}",
                    r.kept.len(),
                    theme::ellipsize(&r.kept.join(", "), width)
                )),
            ]));
        }
        issues(&mut lines, "Reset to defaults", &r.reset);
        issues(&mut lines, "Needs a value", &r.needs_value);

        let block = Block::new()
            .borders(Borders::ALL)
            .border_type(BorderType::Rounded)
            .border_style(Style::new().fg(theme::WARN))
            .title(Span::styled(" Your config needed repairs ", theme::label()))
            .padding(Padding::horizontal(1));
        frame.render_widget(
            Paragraph::new(lines)
                .wrap(Wrap { trim: false })
                .block(block),
            area,
        );
    }

    fn on_key(&mut self, _app: &mut App, key: KeyEvent) -> Nav {
        if key.modifiers.contains(KeyModifiers::CONTROL) && key.code == KeyCode::Char('c') {
            return Nav::Pop;
        }
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

    fn title(&self, _app: &App) -> String {
        "Config".into()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testkit::{ch, key, render_screen, test_app};
    use std::path::PathBuf;

    fn report() -> ConfigReport {
        let issue = |k: &str, f: &str, a: &str| ConfigIssue {
            key: k.into(),
            found: f.into(),
            action: a.into(),
        };
        ConfigReport {
            path: PathBuf::from("/home/u/.config/stt-app/config.toml"),
            backup: Some(PathBuf::from("/home/u/.config/stt-app/config.toml.bak")),
            parse_error: None,
            kept: vec!["whisper.model".into(), "audio.device_name".into()],
            reset: vec![issue("llm.reasoning", "extreme", "reset to default")],
            needs_value: vec![issue("llm.model", "", "LLM turned off")],
        }
    }

    #[test]
    fn renders_sections_and_keys() {
        let (mut app, _h) = test_app();
        let mut s = ConfigNoticeScreen::new(report());
        let out = render_screen(&mut s, &app, 80, 24);
        for want in [
            "config.toml.bak",
            "Kept",
            "2 values",
            "Reset to defaults",
            "llm.reasoning",
            "extreme",
            "Needs a value",
            "(missing)",
            "LLM turned off",
        ] {
            assert!(out.contains(want), "missing {want}:\n{out}");
        }
        assert!(matches!(
            s.on_key(&mut app, ch('s')),
            Nav::SwitchTab(Tab::Settings)
        ));
        assert!(matches!(s.on_key(&mut app, key(KeyCode::Enter)), Nav::Pop));
    }
}
