//! Microphone access via `cpal` (ALSA / PipeWire).
//!
//! Owns (P2-5): `list_input_devices` with monitor detection and deduplication (one entry per real source,
//! like `src/stt_app/audio.py` preferring the JACK list), a [`stt_core::Recorder`] with a live level meter,
//! a [`stt_core::MultiRecorder`] capturing every device at once for calibration, and `to_mono_16k`
//! (downmix + resample to 16 kHz). Implements [`stt_core::AudioSource`].
//!
//! `cpal::Stream` is `!Send`; keep each stream on its own thread (see [`stt_core::audio`]).
