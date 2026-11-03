//! Main tab (UI_MOCKUPS.txt D / A1-A2): live header (mic with hot-mic meter, Whisper status, mode),
//! last result, recent dictations, and the single-key menu.

use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};
use ratatui::Frame;
use ratatui::layout::{Constraint, Layout, Rect};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Paragraph, Wrap};
use stt_core::{ComputeDevice, WhisperDecoding};

use crate::app::{App, ModeKey};
use crate::mic::MicStatus;
use crate::model::{ModelStatus, WARMUP_HINT_AFTER};
use crate::reasoning_view::ReasoningView;
use crate::record::RecordScreen;
use crate::screen::{Hint, Nav, Screen};
use crate::theme;
use crate::widgets::PickList;

/// The Main tab.
#[derive(Default)]
pub struct MainTab;

impl MainTab {
    /// Main tab.
    pub fn new() -> Self {
        Self
    }
}

const LABEL_W: usize = 9;

fn label(name: &str) -> Span<'static> {
    Span::styled(format!("{name:<LABEL_W$}"), theme::label())
}

/// The Mic row (dot, device, meter, state word) for `width` cells.
pub(crate) fn mic_line(app: &App, width: u16) -> Line<'static> {
    let status = app.mic.status();
    let recording = matches!(status, MicStatus::Recording { .. });
    let listening = matches!(status, MicStatus::Listening);
    let (word, style) = match status {
        MicStatus::Opening => ("opening…".to_string(), theme::dim()),
        MicStatus::Listening => (
            "listening".to_string(),
            Style::new().fg(theme::OK).add_modifier(Modifier::DIM),
        ),
        MicStatus::Recording { .. } => (
            "RECORDING".to_string(),
            Style::new().fg(theme::REC).add_modifier(Modifier::BOLD),
        ),
        MicStatus::Error(e) => (theme::ellipsize(e, 40), Style::new().fg(theme::ERR)),
        MicStatus::Closed => ("closed".to_string(), theme::dim()),
    };
    let name = theme::ellipsize(app.mic.label(), 34);
    let name_w = name.chars().count().max(24);
    let fixed = LABEL_W + 2 + name_w + 1 + 2 + word.chars().count();
    let meter_w = (width as usize).saturating_sub(fixed).clamp(6, 20);
    let mut spans = vec![
        label("Mic"),
        theme::mic_dot(recording, listening),
        Span::raw(format!("{name:<name_w$} ")),
    ];
    spans.extend(theme::meter(app.mic.level(), meter_w).spans);
    spans.push(Span::raw("  "));
    spans.push(Span::styled(word, style));
    Line::from(spans)
}

fn whisper_line(app: &App) -> Line<'static> {
    let mut spans = vec![label("Whisper")];
    match app.model.status() {
        ModelStatus::Idle => {
            spans.push(theme::dot(Color::DarkGray));
            spans.push(Span::styled("not loaded", theme::dim()));
        }
        ModelStatus::Downloading {
            file,
            done,
            total,
            bytes_per_sec,
        } => {
            let frac = if total > 0 {
                (done as f64 / total as f64).clamp(0.0, 1.0)
            } else {
                0.0
            };
            let n = (frac * 12.0).round() as usize;
            spans.push(theme::dot(theme::WARN));
            spans.push(Span::styled(
                format!(
                    "downloading {}  {}{}  {:.0}%  {:.1} MB/s",
                    theme::ellipsize(&file, 40),
                    "█".repeat(n),
                    "░".repeat(12 - n),
                    frac * 100.0,
                    bytes_per_sec / 1_000_000.0
                ),
                Style::new().fg(theme::WARN),
            ));
        }
        ModelStatus::Loading { device, since } => {
            spans.push(theme::dot(theme::WARN));
            let text = if matches!(device, ComputeDevice::Gpu { .. })
                && since.elapsed() > WARMUP_HINT_AFTER
            {
                format!(
                    "warming up the GPU (one-time shader compile, ~20 s)…  {}  {} s",
                    theme::spinner(since),
                    since.elapsed().as_secs()
                )
            } else {
                format!("{} loading onto {device}…", theme::spinner(since))
            };
            spans.push(Span::styled(text, Style::new().fg(theme::WARN)));
        }
        ModelStatus::Fallback { reason } => {
            spans.push(theme::dot(theme::WARN));
            spans.push(Span::styled(
                format!("GPU failed, trying CPU: {}", theme::ellipsize(&reason, 50)),
                Style::new().fg(theme::WARN),
            ));
        }
        ModelStatus::Ready(info) => {
            spans.push(theme::dot(theme::OK));
            let dec = match app.cfg.whisper.decoding {
                WhisperDecoding::Beam => "beam 5",
                WhisperDecoding::Greedy => "greedy",
            };
            spans.push(Span::raw(format!("{} · {dec}", info.description())));
        }
        ModelStatus::Failed(msg) => {
            spans.push(theme::dot(theme::ERR));
            spans.push(Span::styled(
                theme::ellipsize(&msg, 70),
                Style::new().fg(theme::ERR),
            ));
        }
    }
    Line::from(spans)
}

/// The header rows: Mic, optional mic note, Whisper, Mode.
pub(crate) fn header_lines(app: &App, width: u16) -> Vec<Line<'static>> {
    let mut lines = vec![mic_line(app, width)];
    if let Some(note) = app.mic.note() {
        lines.push(Line::from(vec![
            Span::raw(" ".repeat(LABEL_W)),
            Span::styled(note.to_string(), Style::new().fg(theme::WARN)),
        ]));
    }
    lines.push(whisper_line(app));
    lines.push(Line::from(vec![
        label("Mode"),
        Span::raw(format!("{}{}", app.mode_label(), app.mode_extras())),
    ]));
    lines
}

fn wrapped_height(text: &str, width: u16) -> u16 {
    let w = width.max(1) as usize;
    text.lines()
        .map(|l| l.chars().count().max(1).div_ceil(w))
        .sum::<usize>()
        .max(1) as u16
}

/// "HH:MM  mode          text" row used by the Recent panel and the History list.
pub(crate) fn history_row(
    item: &crate::app::HistoryItem,
    num: Option<usize>,
    width: usize,
) -> String {
    let head = match num {
        Some(n) => format!("{n:<2} {}  {:<16} ", item.at.format("%H:%M"), item.mode),
        None => format!("{}  {:<16} ", item.at.format("%H:%M"), item.mode),
    };
    let rest = width.saturating_sub(head.chars().count());
    format!("{head}{}", theme::ellipsize(&item.final_text, rest))
}

impl Screen for MainTab {
    fn render(&mut self, app: &App, frame: &mut Frame, area: Rect) {
        let header = header_lines(app, area.width);
        let [head_a, rest] = Layout::vertical([
            Constraint::Length(header.len() as u16 + 1),
            Constraint::Min(0),
        ])
        .areas(area);
        frame.render_widget(Paragraph::new(header), head_a);

        let Some(newest) = app.history.front() else {
            let block = theme::panel("Recent");
            let inner = block.inner(rest);
            frame.render_widget(block, rest);
            frame.render_widget(
                Paragraph::new(Line::styled(
                    "Nothing yet. Press Enter to record.",
                    theme::dim(),
                )),
                inner,
            );
            return;
        };

        let inner_w = area.width.saturating_sub(4);
        let mut lines: Vec<Line> = newest
            .final_text
            .lines()
            .map(|l| Line::raw(l.to_string()))
            .collect();
        let mut h = wrapped_height(&newest.final_text, inner_w);
        if newest.raw != newest.final_text {
            let raw = format!("raw: {}", newest.raw.replace('\n', " "));
            h += wrapped_height(&raw, inner_w);
            lines.push(Line::styled(raw, theme::dim()));
        }
        if let Some(res) = &app.last_result {
            lines.push(theme::reasoning_line(res));
            h += 1;
        }
        let max_h = (rest.height / 2).max(3);
        let last_h = h.min(max_h) + 2;
        let [last_a, recent_a] =
            Layout::vertical([Constraint::Length(last_h), Constraint::Min(3)]).areas(rest);
        let block = theme::panel("Last result");
        let inner = block.inner(last_a);
        frame.render_widget(block, last_a);
        frame.render_widget(Paragraph::new(lines).wrap(Wrap { trim: false }), inner);

        let block = theme::panel("Recent");
        let inner = block.inner(recent_a);
        frame.render_widget(block, recent_a);
        let rows: Vec<Line> = app
            .history
            .iter()
            .take(inner.height as usize)
            .enumerate()
            .map(|(i, it)| Line::raw(history_row(it, Some(i + 1), inner.width as usize)))
            .collect();
        frame.render_widget(Paragraph::new(rows), inner);
    }

    fn on_key(&mut self, app: &mut App, key: KeyEvent) -> Nav {
        if key.modifiers.contains(KeyModifiers::CONTROL) {
            return Nav::Ignored;
        }
        match key.code {
            KeyCode::Enter | KeyCode::Char('r') => Nav::Push(Box::new(RecordScreen::new(app))),
            KeyCode::Char('m') => {
                cycle_mode(app);
                Nav::Stay
            }
            KeyCode::Char('p') if app.mode_key() == ModeKey::Prompt => choose_preset(app),
            KeyCode::Char('v') => match reasoning_text(app) {
                Some(t) => Nav::Push(Box::new(ReasoningView::new(t))),
                None => Nav::Stay,
            },
            KeyCode::Char('h') => Nav::SwitchTab(crate::app::Tab::History),
            KeyCode::Char('s') => Nav::SwitchTab(crate::app::Tab::Settings),
            KeyCode::Char('q') => Nav::Quit,
            _ => Nav::Ignored,
        }
    }

    fn hints(&self, app: &App) -> Vec<Hint> {
        let mut h: Vec<Hint> = vec![("Enter/r", "record".into()), ("m", "mode".into())];
        if app.mode_key() == ModeKey::Prompt {
            h.push(("p", "preset".into()));
        }
        if reasoning_text(app).is_some() {
            h.push(("v", "reasoning".into()));
        }
        h.push(("h", "history".into()));
        h.push(("s", "settings".into()));
        h.push(("q", "quit".into()));
        h
    }
}

fn reasoning_text(app: &App) -> Option<String> {
    app.last_result
        .as_ref()
        .filter(|r| !r.reasoning.is_empty())
        .map(|r| r.reasoning.clone())
}

/// Raw → Cleanup → Prompt → Raw (Python `cycle_mode`); refuses without a configured LLM.
pub(crate) fn cycle_mode(app: &mut App) {
    let next = match app.mode_key() {
        ModeKey::Raw => ModeKey::Cleanup,
        ModeKey::Cleanup => ModeKey::Prompt,
        ModeKey::Prompt => ModeKey::Raw,
    };
    if next != ModeKey::Raw && !app.llm_configured() {
        app.flash(
            crate::app::FlashKind::Warn,
            "Set up an LLM in Settings first.",
        );
        app.cfg.llm.enabled = false;
        app.save();
        return;
    }
    app.cfg.llm.enabled = next != ModeKey::Raw;
    match next {
        ModeKey::Cleanup => app.cfg.llm.mode = stt_core::LlmMode::Cleanup,
        ModeKey::Prompt => app.cfg.llm.mode = stt_core::LlmMode::Prompt,
        ModeKey::Raw => {}
    }
    app.save();
}

fn choose_preset(app: &App) -> Nav {
    let presets = stt_llm::all_presets(&app.cfg.llm);
    let current = presets.iter().position(|(k, _)| *k == app.cfg.llm.preset);
    let labels = presets.iter().map(|(_, l)| l.clone()).collect();
    Nav::Push(Box::new(PickList::new(
        "Preset",
        labels,
        current,
        move |app, i| {
            if let Some((key, _)) = presets.get(i) {
                app.cfg.llm.preset = key.clone();
                app.save();
            }
            Nav::Stay
        },
    )))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runner::Ui;
    use crate::testkit::*;
    use stt_core::AppConfig;
    use stt_core::LlmResult;
    use stt_core::fakes::FakeLlm;

    fn llm_cfg() -> AppConfig {
        let mut cfg = AppConfig::default();
        cfg.llm.model = "qwen".into();
        cfg.llm.base_url = "http://localhost:1234/v1".into();
        cfg
    }

    #[test]
    fn header_shows_listening_after_settle() {
        let (mut app, _h) = test_app();
        settle(&mut app);
        let out = render_screen(&mut MainTab::new(), &app, 80, 24);
        assert!(out.contains("(system default)"), "{out}");
        assert!(out.contains("listening"), "{out}");
        assert!(out.contains("beam 5"), "{out}");
        assert!(out.contains("Raw"), "{out}");
    }

    #[test]
    fn header_shows_mode_extras() {
        let mut cfg = llm_cfg();
        cfg.llm.enabled = true;
        cfg.llm.mode = stt_core::LlmMode::Prompt;
        cfg.llm.word_choice = true;
        cfg.llm.reasoning = stt_core::ReasoningLevel::Low;
        let (app, _h) = test_app_with(cfg, FakeLlm::new("x"));
        let out = render_screen(&mut MainTab::new(), &app, 80, 24);
        assert!(out.contains("reasoning: low"), "{out}");
        assert!(out.contains("word choice"), "{out}");
    }

    #[test]
    fn v_hint_only_with_reasoning_text() {
        let (mut app, _h) = test_app();
        let tab = MainTab::new();
        let has_v = |app: &App| tab.hints(app).iter().any(|(k, _)| *k == "v");
        assert!(!has_v(&app));
        app.last_result = Some(LlmResult {
            text: "x".into(),
            reasoned: Some(false),
            ..Default::default()
        });
        assert!(!has_v(&app));
        app.last_result = Some(LlmResult {
            text: "x".into(),
            reasoning: "because".into(),
            ..Default::default()
        });
        assert!(has_v(&app));
        let mut ui = Ui::new();
        ui.on_key(&mut app, ch('v'));
        assert_eq!(ui.overlay_count(), 1);
        assert!(render_ui(&mut ui, &app, 80, 24).contains("because"));
        ui.on_key(&mut app, key(KeyCode::Esc));
        assert_eq!(ui.overlay_count(), 0);
    }

    #[test]
    fn p_hint_only_in_prompt_mode() {
        let mut cfg = llm_cfg();
        cfg.llm.enabled = true;
        cfg.llm.mode = stt_core::LlmMode::Cleanup;
        let (mut app, _h) = test_app_with(cfg, FakeLlm::new("x"));
        let tab = MainTab::new();
        assert!(!tab.hints(&app).iter().any(|(k, _)| *k == "p"));
        app.cfg.llm.mode = stt_core::LlmMode::Prompt;
        assert!(tab.hints(&app).iter().any(|(k, _)| *k == "p"));
    }

    #[test]
    fn mode_cycling_refuses_without_llm_config() {
        let (mut app, _h) = test_app();
        let mut tab = MainTab::new();
        tab.on_key(&mut app, ch('m'));
        assert_eq!(app.mode_key(), ModeKey::Raw);
        assert!(app.flash.as_ref().unwrap().text.contains("Set up an LLM"));
    }

    #[test]
    fn mode_cycles_raw_cleanup_prompt_raw() {
        let (mut app, h) = test_app_with(llm_cfg(), FakeLlm::new("x"));
        let mut tab = MainTab::new();
        let mut seen = vec![];
        for _ in 0..3 {
            tab.on_key(&mut app, ch('m'));
            seen.push(app.mode_key());
        }
        assert_eq!(seen, vec![ModeKey::Cleanup, ModeKey::Prompt, ModeKey::Raw]);
        assert_eq!(h.config.saves().len(), 3);
    }

    #[test]
    fn preset_picker_saves_choice() {
        let mut cfg = llm_cfg();
        cfg.llm.enabled = true;
        cfg.llm.mode = stt_core::LlmMode::Prompt;
        let (mut app, _h) = test_app_with(cfg, FakeLlm::new("x"));
        let mut ui = Ui::new();
        ui.on_key(&mut app, ch('p'));
        assert_eq!(ui.overlay_count(), 1);
        ui.on_key(&mut app, ch('2'));
        assert_eq!(ui.overlay_count(), 0);
        let presets = stt_llm::all_presets(&app.cfg.llm);
        assert_eq!(app.cfg.llm.preset, presets[1].0);
    }

    #[test]
    fn last_result_and_recent_render() {
        let (mut app, _h) = test_app();
        app.push_history(crate::app::HistoryItem {
            at: chrono::Local::now(),
            raw: "um hello".into(),
            final_text: "Hello.".into(),
            mode: "Cleanup".into(),
            reasoning: String::new(),
        });
        let out = render_screen(&mut MainTab::new(), &app, 80, 24);
        assert!(out.contains("Last result"), "{out}");
        assert!(out.contains("raw: um hello"), "{out}");
        assert!(out.contains("Recent"), "{out}");
    }
}
