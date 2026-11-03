//! Settings › Whisper (A7).

use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::Instant;

use ratatui::style::Style;
use ratatui::text::{Line, Span};
use stt_core::config::resolved_models_dir;
use stt_core::secrets::{HF_TOKEN_ENV, HF_TOKEN_USER, get_api_key, set_api_key};
use stt_core::transcribe::WHISPER_MODELS;
use stt_core::{DownloadProgressFn, WhisperDecoding, WhisperDevice, WhisperQuant};

use crate::app::App;
use crate::screen::Nav;
use crate::task::Task;
use crate::theme;
use crate::widgets::{PickList, TextPrompt};

use super::popups::Busy;
use super::section::{Row, Section};
use super::{commit, cycle_in, decoding_label};

/// The settings that need a model reload when they change.
type ReloadKey = (
    String,
    WhisperDevice,
    WhisperQuant,
    WhisperDecoding,
    Option<String>,
    String,
);

fn reload_key(app: &App) -> ReloadKey {
    let w = &app.cfg.whisper;
    (
        w.model.clone(),
        w.device,
        w.quantization,
        w.decoding,
        w.language.clone(),
        w.models_dir.clone(),
    )
}

#[derive(Default)]
struct Prog {
    file: String,
    done: u64,
    total: u64,
    last: Option<(Instant, u64)>,
    bps: f64,
}

struct Download {
    task: Task<Result<PathBuf, String>>,
    prog: Arc<Mutex<Prog>>,
    since: Instant,
}

pub(super) struct WhisperSection {
    initial: ReloadKey,
    download: Option<Download>,
    result: Option<Result<PathBuf, String>>,
}

impl WhisperSection {
    pub fn new(app: &App) -> Self {
        Self {
            initial: reload_key(app),
            download: None,
            result: None,
        }
    }

    fn start_download(&mut self, app: &mut App) {
        if self.download.is_some() {
            return;
        }
        // Use the loaded transcriber unless settings changed since (then it would fetch the old model).
        let t = if reload_key(app) == self.initial {
            app.model.transcriber()
        } else {
            None
        };
        let t = match t {
            Some(t) => Ok(t),
            None => app
                .services
                .transcribers
                .create(&app.cfg, app.services.secrets.as_ref()),
        };
        let t = match t {
            Ok(t) => t,
            Err(e) => {
                self.result = Some(Err(e.to_string()));
                return;
            }
        };
        let prog = Arc::new(Mutex::new(Prog::default()));
        let p = prog.clone();
        let cb: DownloadProgressFn = Arc::new(move |ev| {
            let mut g = p.lock().unwrap_or_else(|e| e.into_inner());
            let now = Instant::now();
            if let Some((t0, d0)) = g.last
                && ev.done >= d0
            {
                g.bps = (ev.done - d0) as f64 / now.duration_since(t0).as_secs_f64().max(1e-3);
            }
            g.last = Some((now, ev.done));
            g.file = ev.file.clone();
            g.done = ev.done;
            g.total = ev.total;
        });
        self.result = None;
        self.download = Some(Download {
            task: Task::spawn(move || t.fetch(Some(cb)).map_err(|e| e.to_string())),
            prog,
            since: Instant::now(),
        });
    }
}

fn mb(bytes: u64) -> f64 {
    bytes as f64 / 1_000_000.0
}

impl Section for WhisperSection {
    fn title(&self) -> String {
        "Whisper".into()
    }

    fn rows(&self, app: &App) -> Vec<Row> {
        let w = &app.cfg.whisper;
        let device = match w.device {
            WhisperDevice::Auto => "auto (GPU, falls back to CPU)",
            WhisperDevice::Gpu => "gpu (no fallback)",
            WhisperDevice::Cpu => "cpu",
        };
        let folder = match resolved_models_dir(&app.cfg) {
            Ok(p) => p.display().to_string(),
            Err(_) => "(unknown)".into(),
        } + if w.models_dir.trim().is_empty() {
            " (default)"
        } else {
            ""
        };
        let token =
            if get_api_key(app.services.secrets.as_ref(), HF_TOKEN_USER, HF_TOKEN_ENV).is_some() {
                "set"
            } else {
                "not set"
            };
        vec![
            Row::new('1', "Model", w.model.clone()),
            Row::new('2', "Device", device).choice(),
            Row::new('3', "Quality", w.quantization.as_str()).choice(),
            Row::new('4', "Decoding", decoding_label(w.decoding)).choice(),
            Row::new(
                '5',
                "Language",
                w.language.as_deref().unwrap_or("auto-detect"),
            ),
            Row::new('6', "Models folder", folder),
            Row::new(
                '7',
                "Download connections",
                w.download_connections.to_string(),
            ),
            Row::new('8', "Hugging Face token", token),
            Row::new('9', "Download / verify now", ""),
        ]
    }

    fn help(&self, _app: &App, i: usize) -> String {
        match i {
            0 => "Which Whisper model transcribes your speech. Bigger is more accurate but slower and needs more memory.",
            1 => "auto uses the GPU and falls back to the CPU if that fails. gpu never falls back. cpu is the slowest.",
            2 => "Weight precision. q5_0 is the smallest and gave the same text as f16 in tests; q8_0 sits between; f16 is the largest.",
            3 => "Beam 5 tries several word sequences and keeps the best (about 0.34 s for a 20 s clip). Greedy takes the first guess (about 0.23 s), a bit less accurate.",
            4 => "Language code such as en or de. Blank = auto-detect (slower, and can guess wrong on short clips).",
            5 => "Where model files are stored. Blank = the default folder. Changing it reloads the model (and may download it).",
            6 => "Parallel connections per downloaded file, 1 to 32.",
            7 => "Optional Hugging Face token for faster or gated downloads. Kept in the system keyring; never shown.",
            _ => "Downloads the current model file now, or verifies it is complete. An interrupted download resumes.",
        }
        .into()
    }

    fn activate(&mut self, app: &mut App, i: usize) -> Nav {
        match i {
            0 => {
                let models: Vec<String> = WHISPER_MODELS.iter().map(|m| m.to_string()).collect();
                let cur = WHISPER_MODELS
                    .iter()
                    .position(|m| *m == app.cfg.whisper.model);
                Nav::Push(Box::new(PickList::new("Model", models, cur, |app, i| {
                    app.cfg.whisper.model = WHISPER_MODELS[i].to_string();
                    commit(app);
                    Nav::Stay
                })))
            }
            1..=3 => {
                self.cycle(app, i, 1);
                Nav::Stay
            }
            4 => {
                let cur = app.cfg.whisper.language.clone().unwrap_or_default();
                Nav::Push(Box::new(
                    TextPrompt::new("Language code", &cur, |app, s| {
                        let s = s.trim().to_lowercase();
                        app.cfg.whisper.language = (!s.is_empty()).then_some(s);
                        commit(app);
                        Nav::Stay
                    })
                    .help("Blank = auto-detect."),
                ))
            }
            5 => {
                let cur = app.cfg.whisper.models_dir.clone();
                Nav::Push(Box::new(
                    TextPrompt::new("Models folder", &cur, |app, s| {
                        app.cfg.whisper.models_dir = s.trim().to_string();
                        commit(app);
                        Nav::Stay
                    })
                    .help("Blank = the default folder."),
                ))
            }
            6 => {
                let cur = app.cfg.whisper.download_connections.to_string();
                Nav::Push(Box::new(TextPrompt::new(
                    "Download connections (1-32)",
                    &cur,
                    |app, s| {
                        match s.trim().parse::<u32>() {
                            Ok(n) if (1..=32).contains(&n) => {
                                app.cfg.whisper.download_connections = n;
                                commit(app);
                            }
                            _ if s.trim().is_empty() => {}
                            _ => app.flash_error("Enter a number from 1 to 32."),
                        }
                        Nav::Stay
                    },
                )))
            }
            7 => Nav::Push(Box::new(
                TextPrompt::secret("Hugging Face token", |_app, token| {
                    if token.is_empty() {
                        return Nav::Stay;
                    }
                    let store = _app.services.secrets.clone();
                    Nav::Push(Box::new(Busy::new(
                        "Storing the token…",
                        move || set_api_key(store.as_ref(), HF_TOKEN_USER, &token),
                        |app, res| {
                            match res {
                                Ok(()) => app.flash_ok("Token stored."),
                                Err(e) => app.flash_error(format!(
                                    "Could not store the token in the keyring: {e}"
                                )),
                            }
                            Nav::Stay
                        },
                    )))
                })
                .help("Paste, then Enter. Enter alone keeps the current token."),
            )),
            _ => {
                self.start_download(app);
                Nav::Stay
            }
        }
    }

    fn cycle(&mut self, app: &mut App, i: usize, dir: i32) {
        let w = &mut app.cfg.whisper;
        match i {
            1 => w.device = cycle_in(WhisperDevice::ALL, w.device, dir),
            2 => w.quantization = cycle_in(WhisperQuant::ALL, w.quantization, dir),
            3 => w.decoding = cycle_in(WhisperDecoding::ALL, w.decoding, dir),
            _ => return,
        }
        commit(app);
    }

    fn status(&self, _app: &App, width: u16) -> Vec<Line<'static>> {
        if let Some(d) = &self.download {
            let p = d.prog.lock().unwrap_or_else(|e| e.into_inner());
            let name = if p.file.is_empty() {
                "checking…".to_string()
            } else {
                p.file.clone()
            };
            let frac = if p.total > 0 {
                (p.done as f64 / p.total as f64).clamp(0.0, 1.0)
            } else {
                0.0
            };
            let bar_w = (width as usize).saturating_sub(48).clamp(8, 30);
            let on = (frac * bar_w as f64).round() as usize;
            return vec![
                Line::from(vec![
                    Span::raw(format!("{} ", theme::spinner(d.since))),
                    Span::raw(name),
                ]),
                Line::from(vec![
                    Span::styled("█".repeat(on), Style::new().fg(theme::OK)),
                    Span::styled("░".repeat(bar_w - on), theme::dim()),
                    Span::raw(format!(
                        " {:>3.0}%  {:.0}/{:.0} MB  {:.1} MB/s",
                        frac * 100.0,
                        mb(p.done),
                        mb(p.total),
                        p.bps / 1_000_000.0
                    )),
                ]),
            ];
        }
        match &self.result {
            Some(Ok(path)) => vec![Line::styled(
                format!("Model ready: {}", path.display()),
                Style::new().fg(theme::OK),
            )],
            Some(Err(e)) => vec![Line::from(vec![
                Span::styled(format!("Download failed: {e}"), Style::new().fg(theme::ERR)),
                Span::styled(" (run it again to resume)", theme::dim()),
            ])],
            None => Vec::new(),
        }
    }

    fn tick(&mut self, _app: &mut App) -> Nav {
        if let Some(d) = &mut self.download
            && let Some(res) = d.task.poll()
        {
            self.result = Some(res);
            self.download = None;
        } else if self.download.as_ref().is_some_and(|d| d.task.is_finished()) {
            self.result = Some(Err("download stopped unexpectedly".into()));
            self.download = None;
        }
        Nav::Stay
    }

    fn close(&mut self, app: &mut App) {
        if reload_key(app) != self.initial {
            self.initial = reload_key(app);
            app.start_loading();
            app.flash_ok("Reloading the model in the background.");
        }
    }
}
