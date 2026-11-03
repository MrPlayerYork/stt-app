//! Record flow (A3-A4): recording overlay with the red frame and big meter, then transcribe → LLM →
//! clipboard on a worker thread, with step lines.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};
use ratatui::Frame;
use ratatui::layout::{Constraint, Layout, Rect};
use ratatui::style::{Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::Paragraph;
use stt_core::{AppConfig, Clipboard, ClipboardMethod, LlmResult, SecretStore, Transcriber};

use crate::app::{App, FlashKind, HistoryItem};
use crate::main_tab::{header_lines, mic_line};
use crate::mic::MicStatus;
use crate::screen::{Hint, Nav, Screen};
use crate::task::Task;
use crate::theme;

/// Where the worker is, for the step lines.
#[derive(Debug, Clone, Default)]
struct Progress {
    transcribed: Option<Duration>,
    llm_started: bool,
    llm_done: Option<Duration>,
    llm_error: Option<String>,
    copying: bool,
}

/// Everything the worker needs, cloned out of `App` (no `&App` in threads).
struct Job {
    audio: Vec<f32>,
    transcriber: Arc<dyn Transcriber>,
    cfg: AppConfig,
    llm: Arc<dyn stt_core::LlmFactory>,
    secrets: Arc<dyn SecretStore>,
    clipboard: Arc<dyn Clipboard>,
    mode_label: String,
    progress: Arc<Mutex<Progress>>,
    cancelled: Arc<AtomicBool>,
}

struct Done {
    raw: String,
    text: String,
    mode: String,
    result: Option<LlmResult>,
    llm_error: Option<String>,
    copy: Result<ClipboardMethod, String>,
}

enum Fail {
    Error(String),
    Empty,
}

fn lock<T>(m: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    m.lock().unwrap_or_else(|e| e.into_inner())
}

fn run_job(job: Job) -> Result<Done, Fail> {
    let t0 = Instant::now();
    let raw = job
        .transcriber
        .transcribe(&job.audio)
        .map_err(|e| Fail::Error(format!("Transcription failed: {e}")))?
        .trim()
        .to_string();
    lock(&job.progress).transcribed = Some(t0.elapsed());
    if raw.is_empty() {
        return Err(Fail::Empty);
    }
    let mut text = raw.clone();
    let mut mode = job.mode_label.clone();
    let mut result = None;
    let mut llm_error = None;
    if job.cfg.llm.enabled {
        lock(&job.progress).llm_started = true;
        let t1 = Instant::now();
        let key = stt_core::secrets::get_api_key(
            job.secrets.as_ref(),
            job.cfg.llm.provider.keyring_user(),
            &job.cfg.llm.api_key_env,
        );
        match stt_llm::process(job.llm.as_ref(), &job.cfg.llm, key.as_deref(), &raw) {
            Ok(r) => {
                text = r.text.clone();
                result = Some(r);
                lock(&job.progress).llm_done = Some(t1.elapsed());
            }
            Err(e) => {
                mode = "Raw (LLM failed)".into();
                let msg = e.to_string();
                lock(&job.progress).llm_error = Some(msg.clone());
                llm_error = Some(msg);
            }
        }
    }
    lock(&job.progress).copying = true;
    let copy = if job.cancelled.load(Ordering::SeqCst) {
        Err("cancelled".to_string())
    } else {
        job.clipboard.copy(&text).map_err(|e| e.to_string())
    };
    Ok(Done {
        raw,
        text,
        mode,
        result,
        llm_error,
        copy,
    })
}

enum Phase {
    /// Mic is (becoming) hot and keeping audio.
    Recording,
    /// Audio kept; waiting for the Whisper model.
    WaitingModel {
        audio: Vec<f32>,
        secs: f32,
        since: Instant,
    },
    /// Worker running.
    Working {
        task: Box<Task<Result<Done, Fail>>>,
        progress: Arc<Mutex<Progress>>,
        cancelled: Arc<AtomicBool>,
        secs: f32,
        started: Instant,
    },
    /// The mic could not start; flash and leave on the next tick.
    StartFailed(String),
}

/// Recording + processing overlay, pushed by the Main tab.
pub struct RecordScreen {
    phase: Phase,
}

impl RecordScreen {
    /// Start recording on the hot mic.
    pub fn new(app: &mut App) -> Self {
        let phase = match app.mic.start() {
            Ok(()) => Phase::Recording,
            Err(e) => {
                // Reopen so the next try can work.
                app.open_mic();
                Phase::StartFailed(format!("Could not start recording: {e}"))
            }
        };
        Self { phase }
    }

    fn stop_and_process(&mut self, app: &mut App) -> Nav {
        let audio = match app.mic.stop() {
            Ok(a) => a,
            Err(e) => {
                app.flash_error(format!("Recording failed: {e}"));
                return Nav::Pop;
            }
        };
        let secs = audio.len() as f32 / stt_core::SAMPLE_RATE as f32;
        self.phase = Phase::WaitingModel {
            audio,
            secs,
            since: Instant::now(),
        };
        self.try_start_worker(app)
    }

    fn try_start_worker(&mut self, app: &mut App) -> Nav {
        let Phase::WaitingModel { audio, secs, .. } = &mut self.phase else {
            return Nav::Stay;
        };
        if app.model.is_busy() {
            return Nav::Stay;
        }
        let Some(transcriber) = app.model.transcriber() else {
            app.flash_error(format!(
                "Whisper model is not available: {}",
                match app.model.status() {
                    crate::model::ModelStatus::Failed(m) => m,
                    _ => "not loaded".into(),
                }
            ));
            return Nav::Pop;
        };
        if app.cfg.llm.enabled {
            app.llm_used = true;
        }
        let progress = Arc::new(Mutex::new(Progress::default()));
        let cancelled = Arc::new(AtomicBool::new(false));
        let job = Job {
            audio: std::mem::take(audio),
            transcriber,
            cfg: app.cfg.clone(),
            llm: app.services.llm.clone(),
            secrets: app.services.secrets.clone(),
            clipboard: app.services.clipboard.clone(),
            mode_label: app.mode_label(),
            progress: progress.clone(),
            cancelled: cancelled.clone(),
        };
        let secs = *secs;
        self.phase = Phase::Working {
            task: Box::new(Task::spawn(move || run_job(job))),
            progress,
            cancelled,
            secs,
            started: Instant::now(),
        };
        Nav::Stay
    }

    fn finish(app: &mut App, done: Done) {
        let reasoning = done
            .result
            .as_ref()
            .map(|r| r.reasoning.clone())
            .unwrap_or_default();
        app.push_history(HistoryItem {
            at: chrono::Local::now(),
            raw: done.raw,
            final_text: done.text,
            mode: done.mode,
            reasoning,
        });
        app.last_result = done.result;
        match (&done.copy, &done.llm_error) {
            (Ok(m), None) => app.flash_ok(format!("Copied to clipboard ({m})")),
            (Ok(m), Some(e)) => app.flash(
                FlashKind::Warn,
                format!("✗ LLM error: {e} · copied the raw text instead ({m})"),
            ),
            (Err(e), _) => app.flash_error(format!("Clipboard error: {e}")),
        }
    }
}

fn step(mark: &str, color: ratatui::style::Color, text: String) -> Line<'static> {
    Line::from(vec![
        Span::styled(format!("{mark} "), Style::new().fg(color)),
        Span::raw(text),
    ])
}

fn secs_text(d: Duration) -> String {
    format!("{:.1} s", d.as_secs_f32())
}

impl Screen for RecordScreen {
    fn render(&mut self, app: &App, frame: &mut Frame, area: Rect) {
        match &self.phase {
            Phase::Recording => {
                let [head, _, meter, _, keys, _] = Layout::vertical([
                    Constraint::Length(1),
                    Constraint::Length(1),
                    Constraint::Length(1),
                    Constraint::Length(1),
                    Constraint::Length(1),
                    Constraint::Min(0),
                ])
                .areas(area);
                frame.render_widget(Paragraph::new(mic_line(app, area.width)), head);
                if matches!(app.mic.status(), MicStatus::Recording { .. }) {
                    let w = area.width.saturating_sub(2) as usize;
                    frame.render_widget(
                        Paragraph::new(theme::meter(app.mic.level(), w, true)),
                        meter,
                    );
                    let k = Style::new().fg(theme::KEY).add_modifier(Modifier::BOLD);
                    frame.render_widget(
                        Paragraph::new(Line::from(vec![
                            Span::styled("Enter", k),
                            Span::raw("  stop and transcribe      "),
                            Span::styled("Ctrl+C", k),
                            Span::raw("  cancel (throws it away)"),
                        ])),
                        keys,
                    );
                } else {
                    frame.render_widget(
                        Paragraph::new(Line::styled(
                            "opening the microphone…",
                            Style::new().fg(theme::WARN),
                        )),
                        meter,
                    );
                    frame.render_widget(
                        Paragraph::new(Line::from(vec![
                            Span::styled("Ctrl+C", Style::new().fg(theme::KEY)),
                            Span::raw("  cancel"),
                        ])),
                        keys,
                    );
                }
            }
            Phase::WaitingModel { secs, since, .. } => {
                let mut lines = header_lines(app, area.width);
                lines.push(Line::raw(""));
                lines.push(step("✓", theme::OK, format!("Recorded {secs:.1} s")));
                lines.push(step(
                    theme::spinner(*since),
                    theme::WARN,
                    "Waiting for the Whisper model…".into(),
                ));
                frame.render_widget(Paragraph::new(lines), area);
            }
            Phase::Working {
                progress,
                secs,
                started,
                ..
            } => {
                let p = lock(progress).clone();
                let spin = theme::spinner(*started);
                let mut lines = header_lines(app, area.width);
                lines.push(Line::raw(""));
                lines.push(step("✓", theme::OK, format!("Recorded {secs:.1} s")));
                match p.transcribed {
                    Some(d) => lines.push(step(
                        "✓",
                        theme::OK,
                        format!("Transcribed ({})", secs_text(d)),
                    )),
                    None => lines.push(step(spin, theme::WARN, "Transcribing…".into())),
                }
                if p.llm_started {
                    let model = &app.cfg.llm.model;
                    if let Some(e) = &p.llm_error {
                        lines.push(step(
                            "✗",
                            theme::ERR,
                            format!("LLM error: {e} · copying the raw text instead"),
                        ));
                    } else if let Some(d) = p.llm_done {
                        lines.push(step(
                            "✓",
                            theme::OK,
                            format!("Processed with the LLM ({})", secs_text(d)),
                        ));
                    } else {
                        lines.push(step(
                            spin,
                            theme::WARN,
                            format!("Processing with the LLM ({model})…"),
                        ));
                    }
                }
                if p.copying {
                    lines.push(step(spin, theme::WARN, "Copying to the clipboard…".into()));
                }
                frame.render_widget(Paragraph::new(lines), area);
            }
            Phase::StartFailed(msg) => {
                frame.render_widget(
                    Paragraph::new(Line::styled(msg.clone(), Style::new().fg(theme::ERR))),
                    area,
                );
            }
        }
    }

    fn on_key(&mut self, app: &mut App, key: KeyEvent) -> Nav {
        let ctrl_c =
            key.code == KeyCode::Char('c') && key.modifiers.contains(KeyModifiers::CONTROL);
        if ctrl_c {
            match &self.phase {
                Phase::Recording => {
                    app.mic.cancel();
                    app.flash(FlashKind::Warn, "Recording cancelled.");
                }
                Phase::Working { cancelled, .. } => {
                    cancelled.store(true, Ordering::SeqCst);
                    app.flash(FlashKind::Warn, "Cancelled");
                }
                _ => app.flash(FlashKind::Warn, "Cancelled"),
            }
            return Nav::Pop;
        }
        // Everything else is swallowed so the runner's Esc default can't leave the mic recording.
        if matches!(self.phase, Phase::Recording)
            && key.code == KeyCode::Enter
            && matches!(app.mic.status(), MicStatus::Recording { .. })
        {
            return self.stop_and_process(app);
        }
        Nav::Stay
    }

    fn on_tick(&mut self, app: &mut App) -> Nav {
        match &mut self.phase {
            Phase::StartFailed(msg) => {
                let msg = msg.clone();
                app.flash_error(msg);
                Nav::Pop
            }
            Phase::Recording => {
                if matches!(app.mic.status(), MicStatus::Error(_) | MicStatus::Closed) {
                    let why = match app.mic.status() {
                        MicStatus::Error(e) => e.clone(),
                        _ => "microphone closed".into(),
                    };
                    app.flash_error(format!("Could not start recording: {why}"));
                    app.open_mic();
                    return Nav::Pop;
                }
                Nav::Stay
            }
            Phase::WaitingModel { .. } => self.try_start_worker(app),
            Phase::Working { task, .. } => match task.poll() {
                None => {
                    if task.is_finished() {
                        app.flash_error("Processing failed unexpectedly.");
                        Nav::Pop
                    } else {
                        Nav::Stay
                    }
                }
                Some(Ok(done)) => {
                    Self::finish(app, done);
                    Nav::Pop
                }
                Some(Err(Fail::Empty)) => {
                    app.flash(FlashKind::Warn, "Heard nothing.");
                    Nav::Pop
                }
                Some(Err(Fail::Error(e))) => {
                    app.flash_error(e);
                    Nav::Pop
                }
            },
        }
    }

    fn hints(&self, _app: &App) -> Vec<Hint> {
        match self.phase {
            Phase::Recording => vec![("Enter", "stop".into()), ("Ctrl+C", "cancel".into())],
            _ => vec![("Ctrl+C", "cancel".into())],
        }
    }

    fn title(&self, _app: &App) -> String {
        match self.phase {
            Phase::Recording => "Record".into(),
            _ => "Working".into(),
        }
    }

    fn is_recording(&self, app: &App) -> bool {
        matches!(self.phase, Phase::Recording)
            && matches!(app.mic.status(), MicStatus::Recording { .. })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runner::Ui;
    use crate::testkit::*;
    use stt_core::fakes::{FakeClipboard, FakeLlm};

    fn llm_cfg() -> AppConfig {
        let mut cfg = AppConfig::default();
        cfg.llm.model = "qwen".into();
        cfg.llm.base_url = "http://localhost:1234/v1".into();
        cfg.llm.enabled = true;
        cfg
    }

    fn pump(ui: &mut Ui, app: &mut App) {
        for _ in 0..600 {
            ui.tick(app);
            if ui.overlay_count() == 0 {
                return;
            }
            std::thread::sleep(Duration::from_millis(5));
        }
        panic!("record flow did not finish");
    }

    fn start(app: &mut App) -> Ui {
        settle(app);
        let mut ui = Ui::new();
        ui.on_key(app, key(KeyCode::Enter));
        assert_eq!(ui.overlay_count(), 1);
        ui
    }

    #[test]
    fn raw_flow_end_to_end() {
        let (mut app, h) = test_app();
        let mut ui = start(&mut app);
        assert!(matches!(app.mic.status(), MicStatus::Recording { .. }));
        let shown = render_ui(&mut ui, &app, 80, 24);
        assert!(shown.contains("RECORDING"), "{shown}");
        assert!(shown.contains("stop and transcribe"), "{shown}");
        ui.on_key(&mut app, key(KeyCode::Enter));
        pump(&mut ui, &mut app);
        assert_eq!(h.clipboard.copies(), vec!["hello world".to_string()]);
        assert_eq!(app.history.len(), 1);
        assert_eq!(app.history[0].mode, "Raw");
        assert!(app.last_result.is_none());
        assert!(app.flash.as_ref().unwrap().text.starts_with("Copied"));
        assert!(!app.llm_used);
    }

    #[test]
    fn llm_flow_copies_llm_text() {
        let (mut app, h) = test_app_with(llm_cfg(), FakeLlm::new("Hello, world."));
        let mut ui = start(&mut app);
        ui.on_key(&mut app, key(KeyCode::Enter));
        pump(&mut ui, &mut app);
        assert_eq!(h.clipboard.copies(), vec!["Hello, world.".to_string()]);
        assert_eq!(app.history[0].raw, "hello world");
        assert_eq!(app.history[0].final_text, "Hello, world.");
        assert!(app.last_result.is_some());
        assert!(app.llm_used);
    }

    #[test]
    fn failing_llm_copies_raw() {
        let (mut app, h) = test_app_with(llm_cfg(), FakeLlm::new("x").failing("boom"));
        let mut ui = start(&mut app);
        ui.on_key(&mut app, key(KeyCode::Enter));
        pump(&mut ui, &mut app);
        assert_eq!(h.clipboard.copies(), vec!["hello world".to_string()]);
        assert_eq!(app.history[0].mode, "Raw (LLM failed)");
        assert!(app.last_result.is_none());
        let f = app.flash.as_ref().unwrap();
        assert_eq!(f.kind, FlashKind::Warn);
        assert!(f.text.contains("copied the raw text instead"), "{}", f.text);
    }

    #[test]
    fn ctrl_c_cancels_without_copying() {
        let (mut app, h) = test_app();
        let mut ui = start(&mut app);
        ui.on_key(&mut app, ctrl('c'));
        assert_eq!(ui.overlay_count(), 0);
        assert!(!app.quit, "Ctrl+C in the record screen must not quit");
        assert!(h.clipboard.copies().is_empty());
        assert!(app.history.is_empty());
        assert_eq!(app.mic.status(), &MicStatus::Listening);
        assert!(app.flash.as_ref().unwrap().text.contains("cancelled"));
    }

    #[test]
    fn is_recording_only_while_recording() {
        let (mut app, _h) = test_app();
        settle(&mut app);
        let mut r = RecordScreen::new(&mut app);
        assert!(r.is_recording(&app));
        let _ = r.on_key(&mut app, key(KeyCode::Enter));
        assert!(!r.is_recording(&app));
    }

    #[test]
    fn empty_transcript_says_heard_nothing() {
        let (mut app, h) = test_app();
        // Replace the transcriber factory text by clearing recorded audio: an empty device clip.
        let _ = &h;
        app.services.transcribers = Arc::new(stt_core::fakes::FakeTranscriberFactory::new("  "));
        let mut ui = start(&mut app);
        ui.on_key(&mut app, key(KeyCode::Enter));
        pump(&mut ui, &mut app);
        assert!(app.history.is_empty());
        assert_eq!(app.flash.as_ref().unwrap().text, "Heard nothing.");
    }

    #[test]
    fn clipboard_failure_is_reported() {
        let (mut app, _h) = test_app();
        app.services.clipboard = Arc::new(FakeClipboard::failing());
        let mut ui = start(&mut app);
        ui.on_key(&mut app, key(KeyCode::Enter));
        pump(&mut ui, &mut app);
        let f = app.flash.as_ref().unwrap();
        assert_eq!(f.kind, FlashKind::Error);
        assert_eq!(app.history.len(), 1);
    }

    #[test]
    fn mic_failure_pops_with_error() {
        let (mut app, _h) = test_app();
        app.services.audio = Arc::new(stt_core::fakes::FakeAudioSource::new().failing_open("Bad"));
        app.cfg.audio.device_name = Some("Bad".into());
        settle(&mut app);
        let mut ui = Ui::new();
        ui.on_key(&mut app, key(KeyCode::Enter));
        pump(&mut ui, &mut app);
        assert_eq!(app.flash.as_ref().unwrap().kind, FlashKind::Error);
    }
}
