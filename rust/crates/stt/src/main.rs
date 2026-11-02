//! `stt` binary: wires the concrete implementations (stt-audio, stt-whisper, stt-llm, stt-clipboard,
//! keyring secrets) into `stt-tui`, checks that stdin/stdout are a TTY, and runs first-run logic
//! (no config or no microphone chosen → calibration).
//!
//! Stub until Phase 3 (P3-3).

fn main() -> anyhow::Result<()> {
    let cfg = stt_core::AppConfig::default();
    println!(
        "stt {} (skeleton): Rust rewrite in progress; default model {}",
        env!("CARGO_PKG_VERSION"),
        cfg.whisper.model
    );
    Ok(())
}
