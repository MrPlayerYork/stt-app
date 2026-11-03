//! `stt` binary: wires the concrete implementations (stt-audio, stt-whisper, stt-llm, stt-clipboard,
//! keyring secrets) into `stt-tui`, checks that stdin/stdout are a TTY, and runs the startup screens
//! (config notice, first run), then the full-screen UI. After the terminal is restored it prints this
//! session's dictations.

use std::io::IsTerminal;
use std::process::ExitCode;
use std::sync::Arc;

use anyhow::Context;
use stt_core::config::config_path;
use stt_core::recover::load_config_recovering;
use stt_tui::config_notice::ConfigNoticeScreen;
use stt_tui::first_run::FirstRunScreen;
use stt_tui::{App, FileConfigStore, Screen, Services};

fn main() -> ExitCode {
    if std::env::args()
        .skip(1)
        .any(|a| a == "--version" || a == "-V")
    {
        println!("stt {}", env!("CARGO_PKG_VERSION"));
        return ExitCode::SUCCESS;
    }
    if !std::io::stdin().is_terminal() || !std::io::stdout().is_terminal() {
        eprintln!("stt needs an interactive terminal (stdin is not a TTY).");
        return ExitCode::FAILURE;
    }
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("stt: {e:#}");
            ExitCode::FAILURE
        }
    }
}

fn run() -> anyhow::Result<()> {
    let path = config_path().context("could not find the config directory")?;
    let (cfg, report) = load_config_recovering(&path)
        .with_context(|| format!("could not read {}", path.display()))?;
    let first_start = cfg.is_none();
    let cfg = cfg.unwrap_or_default();
    let no_device = cfg.audio.device_name.is_none();

    let services = Services {
        audio: Arc::new(stt_audio::CpalAudioSource::new()),
        transcribers: Arc::new(stt_whisper::whisper_factory()),
        llm: Arc::new(stt_llm::HttpLlmFactory),
        clipboard: Arc::new(stt_clipboard::SystemClipboard::from_env()),
        secrets: Arc::new(stt_core::KeyringStore),
        config: Arc::new(FileConfigStore { path }),
    };
    let mut app = App::new(cfg, services);
    app.start_loading();
    app.open_mic();

    // First in the list is the top overlay: the config notice shows before the welcome screen.
    let mut startup: Vec<Box<dyn Screen>> = Vec::new();
    if !report.is_clean() {
        startup.push(Box::new(ConfigNoticeScreen::new(report)));
    }
    if first_start || no_device {
        startup.push(Box::new(FirstRunScreen::new()));
    }

    let result = stt_tui::run(&mut app, startup);

    // The terminal is restored now; print the session to the normal screen.
    print_session(&app);
    if let Some(warning) = app.shutdown() {
        eprintln!("\x1b[33m{warning}\x1b[0m");
    }
    println!("Bye.");
    result.context("terminal error")
}

fn print_session(app: &App) {
    if app.session_log.is_empty() {
        return;
    }
    println!("This session:");
    for item in &app.session_log {
        println!(
            "{}  {}  {}",
            item.at.format("%H:%M"),
            item.mode,
            item.final_text
        );
    }
}
