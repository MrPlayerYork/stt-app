//! Whisper speech-to-text via `whisper-rs` (whisper.cpp) with the Vulkan backend.
//!
//! Owns (P2-8): loading a ggml model with GPU→CPU fallback and warm-up, transcription of mono 16 kHz
//! `f32`, the model name → ggml file map (`ggml-large-v3-turbo.bin`, `-q5_0`/`-q8_0` variants from HF
//! `ggerganov/whisper.cpp`), fetching via `stt-download`, and progress hooks. Implements
//! [`stt_core::Transcriber`] and [`stt_core::TranscriberFactory`].
//!
//! This is the ONLY crate that compiles whisper.cpp (feature `whisper`, off while this is a stub), and
//! only the `stt` binary may depend on it. Building it needs the `heavy` board slot; GPU tests are
//! `#[ignore]` and run in the `gpu` slot.
