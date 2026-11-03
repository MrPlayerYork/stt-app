//! Calibration overlay (Python `calibrate.run_calibration`): record every mic at once with live per-mic meters, score with Whisper via `stt_calibrate::evaluate`, ranked table, pick / retry / cancel. On a pick it saves `audio.device_name` and reopens the hot mic.
//!
//! Flow: intro → (starting) → recording → evaluating → results. Everything that can block (opening
//! every device, stopping, Whisper) runs on a [`Task`] polled from `on_tick`.

use std::collections::BTreeMap;
use std::sync::{Arc, Mutex};
use std::time::Instant;

use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};
use ratatui::Frame;
use ratatui::layout::Rect;
use ratatui::style::{Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Paragraph, Wrap};
use stt_calibrate::{MicScore, TEST_PHRASE, evaluate, suggested};
use stt_core::audio::{DeviceId, InputDevice, MultiRecorder};
use stt_core::progress::{LoadEvent, LoadProgressFn};

use crate::app::App;
use crate::screen::{Hint, Layer, Nav, Screen};
use crate::task::Task;
use crate::theme;

type Started = (Box<dyn MultiRecorder>, BTreeMap<DeviceId, String>);

enum Step {
    /// Read the phrase, Enter to start.
    Intro,
    /// Opening every device (1-2 s on PulseAudio).
    Starting(Task<Started>),
    /// Recording all devices.
    Recording {
        rec: Box<dyn MultiRecorder>,
        failed: BTreeMap<DeviceId, String>,
        since: Instant,
    },
    /// Stopping and scoring.
    Evaluating {
        task: Task<Vec<MicScore>>,
        progress: Arc<Mutex<String>>,
        since: Instant,
    },
    /// Ranked table.
    Results {
        ranked: Vec<MicScore>,
        selected: usize,
    },
}

/// Full-screen calibration overlay. Pushed by Settings › Microphone and by first run.
pub struct CalibrationScreen {
    devices: Vec<InputDevice>,
    list_error: Option<String>,
    step: Step,
}

fn lock<T>(m: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    m.lock().unwrap_or_else(|e| e.into_inner())
}

impl CalibrationScreen {
    /// Prepare calibration (closes the hot mic; it is reopened when the overlay finishes).
    pub fn new(app: &mut App) -> Self {
        app.mic.close();
        let (devices, list_error) = match app.services.audio.list_input_devices() {
            Ok(d) if d.is_empty() => (d, Some("No input devices found.".to_string())),
            Ok(d) => (d, None),
            Err(e) => (Vec::new(), Some(format!("Could not list devices: {e}"))),
        };
        Self {
            devices,
            list_error,
            step: Step::Intro,
        }
    }

    fn finish(&mut self, app: &mut App) -> Nav {
        // Stop anything still running in the background.
        match std::mem::replace(&mut self.step, Step::Intro) {
            Step::Starting(task) => {
                std::thread::spawn(move || {
                    if let Some((mut rec, _)) = task.wait() {
                        rec.stop();
                    }
                });
            }
            Step::Recording { mut rec, .. } => {
                std::thread::spawn(move || {
                    rec.stop();
                });
            }
            _ => {}
        }
        app.open_mic();
        Nav::Pop
    }

    fn pick(&mut self, app: &mut App, i: usize) -> Nav {
        let Step::Results { ranked, .. } = &self.step else {
            return Nav::Stay;
        };
        let Some(s) = ranked.get(i) else {
            return Nav::Stay;
        };
        if let Some(err) = &s.error {
            let msg = format!("Can't use {}: {err}", s.device.name);
            app.flash(crate::app::FlashKind::Warn, msg);
            return Nav::Stay;
        }
        let name = s.device.name.clone();
        app.cfg.audio.device_name = Some(name.clone());
        app.save();
        app.open_mic();
        app.flash_ok(format!("Microphone set: {name}"));
        Nav::Pop
    }

    fn begin_stop(&mut self, app: &mut App) {
        let Step::Recording { rec, failed, .. } = std::mem::replace(&mut self.step, Step::Intro)
        else {
            return;
        };
        let transcriber = match app.model.transcriber() {
            Some(t) => t,
            None => {
                app.start_loading();
                match app.model.transcriber() {
                    Some(t) => t,
                    None => {
                        // Factory failed: report on every device through a message.
                        let msg = format!("{:?}", app.model.status());
                        app.flash_error(format!("Whisper model unavailable: {msg}"));
                        let mut rec = rec;
                        rec.stop();
                        self.step = Step::Intro;
                        return;
                    }
                }
            }
        };
        let progress = Arc::new(Mutex::new("Loading the Whisper model…".to_string()));
        let devices = self.devices.clone();
        let p_load = progress.clone();
        let p_tr = progress.clone();
        let task = Task::spawn(move || {
            let mut rec = rec;
            let audio = rec.stop();
            let on_load: LoadProgressFn = Arc::new(move |ev: &LoadEvent| {
                *lock(&p_load) = match ev {
                    LoadEvent::Download(d) => {
                        format!("Downloading {} {}%…", d.file, d.percent())
                    }
                    LoadEvent::Fallback { .. } => "GPU failed; loading on the CPU…".into(),
                    _ => "Loading the Whisper model…".into(),
                };
            });
            let mut on_tr = |n: usize, total: usize, name: &str| {
                *lock(&p_tr) = format!("Transcribing {n}/{total}: {name}");
            };
            evaluate(
                transcriber.as_ref(),
                &devices,
                &audio,
                &failed,
                Some(on_load),
                &mut on_tr,
            )
        });
        self.step = Step::Evaluating {
            task,
            progress,
            since: Instant::now(),
        };
    }
}

fn clock(secs: f64) -> String {
    let t = (secs * 10.0) as u64;
    format!("{}:{:02}.{}", t / 600, (t / 10) % 60, t % 10)
}

fn fit(s: &str, w: usize) -> String {
    let t = theme::ellipsize(s, w);
    format!("{t:<w$}")
}

impl Screen for CalibrationScreen {
    fn render(&mut self, _app: &App, frame: &mut Frame, area: Rect) {
        match &self.step {
            Step::Intro => {
                let mut lines = vec![
                    Line::from("Read this aloud:"),
                    Line::from(""),
                    Line::styled(
                        TEST_PHRASE,
                        Style::new().fg(theme::KEY).add_modifier(Modifier::BOLD),
                    ),
                    Line::from(""),
                ];
                if let Some(e) = &self.list_error {
                    lines.push(Line::styled(e.clone(), Style::new().fg(theme::ERR)));
                } else {
                    lines.push(Line::from(format!(
                        "All {} input devices record at the same time, then Whisper scores each one.",
                        self.devices.len()
                    )));
                    lines.push(Line::from(""));
                    lines.push(Line::from(vec![
                        Span::styled("[Enter]", Style::new().fg(theme::KEY)),
                        Span::raw(" start recording"),
                    ]));
                }
                frame.render_widget(
                    Paragraph::new(lines)
                        .wrap(Wrap { trim: false })
                        .block(theme::panel("Read this aloud")),
                    area,
                );
            }
            Step::Starting(_) => {
                frame.render_widget(
                    Paragraph::new("Opening the microphones…").block(theme::panel("Recording")),
                    area,
                );
            }
            Step::Recording { rec, failed, since } => {
                let levels = rec.levels();
                let name_w = self
                    .devices
                    .iter()
                    .map(|d| d.name.chars().count() + if d.is_monitor { 10 } else { 0 })
                    .max()
                    .unwrap_or(10)
                    .min((area.width as usize).saturating_sub(40).max(12));
                let mut lines = vec![
                    Line::from("Read the phrase aloud, then press Enter."),
                    Line::styled(
                        TEST_PHRASE,
                        Style::new().fg(theme::KEY).add_modifier(Modifier::BOLD),
                    ),
                    Line::from(""),
                ];
                for d in &self.devices {
                    let label = if d.is_monitor {
                        format!("{} (monitor)", d.name)
                    } else {
                        d.name.clone()
                    };
                    let mut spans = vec![Span::raw(fit(&label, name_w)), Span::raw("  ")];
                    if let Some(msg) = failed.get(&d.id) {
                        spans.push(Span::styled(
                            format!("failed to open: {msg}"),
                            Style::new().fg(theme::ERR),
                        ));
                    } else {
                        let lvl = levels.get(&d.id).copied().unwrap_or(0.0);
                        spans.extend(theme::meter(lvl, 24, true).spans);
                    }
                    lines.push(Line::from(spans));
                }
                let title = format!("Recording {}", clock(since.elapsed().as_secs_f64()));
                frame.render_widget(Paragraph::new(lines).block(theme::panel(title)), area);
            }
            Step::Evaluating {
                progress, since, ..
            } => {
                let msg = lock(progress).clone();
                frame.render_widget(
                    Paragraph::new(Line::from(vec![
                        Span::styled(
                            format!("{} ", theme::spinner(*since)),
                            Style::new().fg(theme::WARN),
                        ),
                        Span::raw(msg),
                    ]))
                    .block(theme::panel("Scoring")),
                    area,
                );
            }
            Step::Results { ranked, selected } => {
                let inner_h = area.height.saturating_sub(2) as usize;
                let any_ok = ranked.iter().any(|s| s.is_ok());
                let best = suggested(ranked).map(|s| s.device.name.clone());
                let name_w = ranked
                    .iter()
                    .map(|s| s.device.name.chars().count())
                    .max()
                    .unwrap_or(10)
                    .min((area.width as usize).saturating_sub(48).max(12));
                let mut lines = vec![Line::styled(
                    format!(
                        "    {:<name_w$} {:>4} {:>8} {:>7} {:>6}",
                        "Microphone", "Acc", "Level", "SNR", "Clip"
                    ),
                    theme::dim(),
                )];
                let reserved = 1 + if any_ok { 0 } else { 2 };
                let per_page = (inner_h.saturating_sub(reserved) / 2).max(1);
                let start = (selected + 1).saturating_sub(per_page);
                for (i, s) in ranked.iter().enumerate().skip(start).take(per_page) {
                    let cur = if i == *selected { theme::CURSOR } else { "  " };
                    let num = format!("{}", i + 1);
                    let row_style = if i == *selected {
                        theme::selected()
                    } else {
                        Style::new()
                    };
                    let stats = if s.is_ok() {
                        format!(
                            "{:>3.0}% {:>6.1}dB {:>5.1}dB {:>5.1}%",
                            s.accuracy * 100.0,
                            s.rms_db,
                            s.snr_db,
                            s.clip_frac * 100.0
                        )
                    } else {
                        format!("{:>4} {:>8} {:>7} {:>6}", "-", "-", "-", "-")
                    };
                    lines.push(Line::styled(
                        format!("{cur}{num:<2}{} {stats}", fit(&s.device.name, name_w)),
                        row_style,
                    ));
                    let width = (area.width as usize).saturating_sub(12);
                    let mut spans = vec![Span::raw("     ")];
                    if let Some(e) = &s.error {
                        spans.push(Span::styled(
                            theme::ellipsize(e, width),
                            Style::new().fg(theme::ERR),
                        ));
                    } else {
                        if best.as_deref() == Some(&s.device.name) {
                            spans.push(Span::styled(
                                "suggested · ",
                                Style::new().fg(theme::OK).add_modifier(Modifier::BOLD),
                            ));
                        }
                        spans.push(Span::styled(
                            format!(
                                "\"{}\"",
                                theme::ellipsize(&s.transcript, width.saturating_sub(14))
                            ),
                            theme::dim(),
                        ));
                    }
                    lines.push(Line::from(spans));
                }
                if !any_ok {
                    lines.push(Line::from(""));
                    lines.push(Line::styled(
                        "Every device failed or was silent.",
                        Style::new().fg(theme::ERR),
                    ));
                }
                frame.render_widget(
                    Paragraph::new(lines).block(theme::panel("Microphone ranking")),
                    area,
                );
            }
        }
    }

    fn on_key(&mut self, app: &mut App, key: KeyEvent) -> Nav {
        if key.modifiers.contains(KeyModifiers::CONTROL) && key.code == KeyCode::Char('c') {
            return self.finish(app);
        }
        match &mut self.step {
            Step::Intro => match key.code {
                KeyCode::Enter if self.list_error.is_none() => {
                    let rec_src = app.services.audio.multi_recorder(&self.devices);
                    self.step = Step::Starting(Task::spawn(move || {
                        let mut rec = rec_src;
                        let failed = rec.start();
                        (rec, failed)
                    }));
                    Nav::Stay
                }
                KeyCode::Esc | KeyCode::Char('q') => self.finish(app),
                _ => Nav::Stay,
            },
            Step::Starting(_) => match key.code {
                KeyCode::Esc => self.finish(app),
                _ => Nav::Stay,
            },
            Step::Recording { .. } => match key.code {
                KeyCode::Enter => {
                    self.begin_stop(app);
                    Nav::Stay
                }
                KeyCode::Esc => self.finish(app),
                _ => Nav::Stay,
            },
            Step::Evaluating { .. } => match key.code {
                KeyCode::Esc => self.finish(app),
                _ => Nav::Stay,
            },
            Step::Results { ranked, selected } => {
                let n = ranked.len();
                match key.code {
                    KeyCode::Up => {
                        *selected = selected.saturating_sub(1);
                        Nav::Stay
                    }
                    KeyCode::Down => {
                        *selected = (*selected + 1).min(n.saturating_sub(1));
                        Nav::Stay
                    }
                    KeyCode::Enter => {
                        let i = *selected;
                        self.pick(app, i)
                    }
                    KeyCode::Char(c @ '1'..='9') => {
                        let i = c as usize - '1' as usize;
                        if i < n { self.pick(app, i) } else { Nav::Stay }
                    }
                    KeyCode::Char('r') => {
                        self.step = Step::Intro;
                        Nav::Stay
                    }
                    KeyCode::Esc | KeyCode::Char('q') => self.finish(app),
                    _ => Nav::Stay,
                }
            }
        }
    }

    fn on_tick(&mut self, app: &mut App) -> Nav {
        match &mut self.step {
            Step::Starting(task) => {
                if let Some((rec, failed)) = task.poll() {
                    self.step = Step::Recording {
                        rec,
                        failed,
                        since: Instant::now(),
                    };
                } else if task.is_finished() {
                    // The worker died; back to the start.
                    app.flash_error("Could not start recording");
                    self.step = Step::Intro;
                }
            }
            Step::Evaluating { task, .. } => {
                if let Some(ranked) = task.poll() {
                    let selected = suggested(&ranked)
                        .and_then(|s| ranked.iter().position(|r| r.device.id == s.device.id))
                        .unwrap_or(0);
                    self.step = Step::Results { ranked, selected };
                } else if task.is_finished() {
                    app.flash_error("Calibration failed unexpectedly");
                    self.step = Step::Intro;
                }
            }
            _ => {}
        }
        Nav::Stay
    }

    fn hints(&self, _app: &App) -> Vec<Hint> {
        match &self.step {
            Step::Intro if self.list_error.is_none() => {
                vec![("Enter", "start".into()), ("Esc", "cancel".into())]
            }
            Step::Intro => vec![("Esc", "cancel".into())],
            Step::Starting(_) | Step::Evaluating { .. } => vec![("Esc", "cancel".into())],
            Step::Recording { .. } => vec![("Enter", "stop".into()), ("Esc", "cancel".into())],
            Step::Results { ranked, .. } => {
                let mut h = vec![("Enter", "use selected".into()), ("↑↓", "select".into())];
                if ranked.len() > 1 {
                    h.push(("1-9", "pick".into()));
                }
                h.push(("r", "retry".into()));
                h.push(("Esc", "cancel".into()));
                h
            }
        }
    }

    fn layer(&self) -> Layer {
        Layer::Full
    }

    fn title(&self, _app: &App) -> String {
        "Calibration".into()
    }

    fn is_recording(&self, _app: &App) -> bool {
        matches!(self.step, Step::Recording { .. })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::app::Services;
    use crate::mic::MicStatus;
    use crate::testkit::{Handles, ch, ctrl, key, render_screen, test_app};
    use std::time::Duration;
    use stt_core::fakes::FakeTranscriber;
    use stt_core::secrets::SecretStore;
    use stt_core::transcribe::{Transcriber, TranscriberFactory};
    use stt_core::{AppConfig, TranscribeError};

    const GOOD: usize = 16_000;

    struct PhraseFactory;
    impl TranscriberFactory for PhraseFactory {
        fn create(
            &self,
            _cfg: &AppConfig,
            _s: &dyn SecretStore,
        ) -> Result<Arc<dyn Transcriber>, TranscribeError> {
            Ok(Arc::new(FakeTranscriber::with_fn(|a| {
                Ok(if a.len() == GOOD {
                    TEST_PHRASE.to_string()
                } else {
                    "nonsense".into()
                })
            })))
        }
    }

    fn setup() -> (App, Handles) {
        let (mut app, mut h) = test_app();
        let tone: Vec<f32> = (0..GOOD)
            .map(|i| 0.3 * (i as f32 * 0.05).sin() * if i % 4000 < 2000 { 1.0 } else { 0.05 })
            .collect();
        h.audio = stt_core::fakes::FakeAudioSource::new()
            .device("Good Mic", false, tone)
            .device("Quiet Monitor", true, vec![0.0; 8000])
            .device("Broken Mic", false, vec![0.1; 100])
            .failing_open("Broken Mic")
            .level(0.2);
        app.services = Services {
            audio: Arc::new(h.audio.clone()),
            transcribers: Arc::new(PhraseFactory),
            ..app.services.clone()
        };
        (app, h)
    }

    fn pump(app: &mut App, s: &mut CalibrationScreen, until: impl Fn(&CalibrationScreen) -> bool) {
        for _ in 0..600 {
            app.tick();
            s.on_tick(app);
            if until(s) {
                return;
            }
            std::thread::sleep(Duration::from_millis(5));
        }
        panic!("timed out");
    }

    fn to_results(app: &mut App, s: &mut CalibrationScreen) {
        assert!(matches!(s.on_key(app, key(KeyCode::Enter)), Nav::Stay));
        pump(app, s, |s| matches!(s.step, Step::Recording { .. }));
        assert!(s.is_recording(app));
        s.on_key(app, key(KeyCode::Enter));
        pump(app, s, |s| matches!(s.step, Step::Results { .. }));
    }

    #[test]
    fn end_to_end_ranks_and_pick_saves() {
        let (mut app, h) = setup();
        app.open_mic();
        let mut s = CalibrationScreen::new(&mut app);
        assert_eq!(app.mic.status(), &MicStatus::Closed);
        let intro = render_screen(&mut s, &app, 80, 24);
        assert!(intro.contains("quick brown fox"));
        assert!(!s.is_recording(&app));
        assert!(matches!(s.on_key(&mut app, key(KeyCode::Enter)), Nav::Stay));
        pump(&mut app, &mut s, |s| {
            matches!(s.step, Step::Recording { .. })
        });
        let rec = render_screen(&mut s, &app, 80, 24);
        assert!(rec.contains("Good Mic") && rec.contains("(monitor)"));
        assert!(rec.contains("failed to open"));
        assert!(rec.contains('▮'));
        s.on_key(&mut app, key(KeyCode::Enter));
        pump(&mut app, &mut s, |s| matches!(s.step, Step::Results { .. }));
        let out = render_screen(&mut s, &app, 80, 24);
        assert!(out.contains("100%"), "{out}");
        assert!(out.contains("suggested"), "{out}");
        assert!(out.contains("silent (likely an output monitor)"), "{out}");
        assert!(out.contains("failed to open"), "{out}");
        let Step::Results { ranked, selected } = &s.step else {
            panic!()
        };
        assert_eq!(*selected, 0);
        assert_eq!(ranked[0].device.name, "Good Mic");

        assert!(matches!(s.on_key(&mut app, key(KeyCode::Enter)), Nav::Pop));
        let saves = h.config.saves();
        assert_eq!(
            saves.last().unwrap().audio.device_name.as_deref(),
            Some("Good Mic")
        );
        for _ in 0..400 {
            app.tick();
            if app.mic.status() == &MicStatus::Listening {
                break;
            }
            std::thread::sleep(Duration::from_millis(5));
        }
        assert_eq!(app.mic.status(), &MicStatus::Listening);
        assert!(
            app.flash
                .as_ref()
                .unwrap()
                .text
                .contains("Microphone set: Good Mic")
        );
    }

    #[test]
    fn picking_an_errored_device_is_refused_and_number_keys_pick() {
        let (mut app, h) = setup();
        let mut s = CalibrationScreen::new(&mut app);
        to_results(&mut app, &mut s);
        assert!(matches!(s.on_key(&mut app, key(KeyCode::Down)), Nav::Stay));
        assert!(matches!(s.on_key(&mut app, key(KeyCode::Enter)), Nav::Stay));
        assert!(h.config.saves().is_empty());
        assert!(matches!(s.on_key(&mut app, ch('1')), Nav::Pop));
        assert_eq!(h.config.saves().len(), 1);
    }

    #[test]
    fn retry_returns_to_intro_and_cancel_reopens_mic() {
        let (mut app, h) = setup();
        let mut s = CalibrationScreen::new(&mut app);
        to_results(&mut app, &mut s);
        s.on_key(&mut app, ch('r'));
        assert!(matches!(s.step, Step::Intro));
        assert!(matches!(s.on_key(&mut app, key(KeyCode::Esc)), Nav::Pop));
        assert_eq!(app.mic.status(), &MicStatus::Opening);
        assert!(h.config.saves().is_empty());
    }

    #[test]
    fn ctrl_c_cancels_while_recording() {
        let (mut app, _h) = setup();
        let mut s = CalibrationScreen::new(&mut app);
        assert!(matches!(s.on_key(&mut app, key(KeyCode::Enter)), Nav::Stay));
        pump(&mut app, &mut s, |s| {
            matches!(s.step, Step::Recording { .. })
        });
        assert!(matches!(s.on_key(&mut app, ctrl('c')), Nav::Pop));
        assert!(!s.is_recording(&app));
    }

    #[test]
    fn no_devices_shows_message() {
        let (mut app, mut h) = test_app();
        h.audio = stt_core::fakes::FakeAudioSource::new();
        app.services.audio = Arc::new(h.audio.clone());
        let mut s = CalibrationScreen::new(&mut app);
        assert!(render_screen(&mut s, &app, 80, 24).contains("No input devices found."));
        assert!(matches!(s.on_key(&mut app, key(KeyCode::Enter)), Nav::Stay));
        assert!(matches!(s.step, Step::Intro));
    }
}
