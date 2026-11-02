//! Microphone calibration.
//!
//! Owns (P2-3): pure scoring ported from `src/stt_app/calibrate.py` (transcript accuracy against the test
//! phrase, RMS / SNR / clipping analysis, `combine`, `rank`, silent-monitor marking) with the tests from
//! `tests/test_calibrate.py`, plus a UI-independent flow driver over the [`stt_core::AudioSource`],
//! [`stt_core::MultiRecorder`] and [`stt_core::Transcriber`] traits. The screens live in `stt-tui`.
