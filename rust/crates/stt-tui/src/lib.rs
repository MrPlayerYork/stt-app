//! The terminal UI: `ratatui` + `crossterm` with an inline viewport (not full-screen).
//!
//! Owns (Phase 3): app state, main screen with live header and single-key menu, push-to-record flow with
//! level meter (Enter stops, Ctrl+C cancels), history with re-copy, settings screens (mic, whisper, LLM
//! incl. reasoning / word choice / TTL, custom prompts), masked secret input (`*` echo, paste-safe), and
//! calibration screens. Uses only `stt-core` traits for audio, transcription, LLM, clipboard and secrets,
//! so it is tested with `stt_core::fakes`.
