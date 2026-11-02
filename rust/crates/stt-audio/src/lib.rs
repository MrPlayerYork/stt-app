//! Microphone access via `cpal`.
//!
//! Implements [`stt_core::AudioSource`] as [`CpalAudioSource`]: input listing with monitor detection and
//! dedupe (one entry per real source, like `src/stt_app/audio.py` preferring the JACK list; here the
//! PulseAudio host served by PipeWire plays that role, ALSA is the fallback), a [`stt_core::Recorder`] with
//! a live level meter, a [`stt_core::MultiRecorder`] capturing every device at once for calibration, and
//! conversion to mono 16 kHz `f32` with a band-limited resampler.
//!
//! `cpal::Stream` is `!Send`: every stream lives on its own thread (see `capture.rs`); stopping or dropping
//! a recorder always joins that thread and closes the device.

mod capture;
pub mod convert;
pub mod matching;
mod source;

pub use convert::{downmix, peak, resample, rms, to_mono_16k};
pub use matching::{is_monitor_name, match_saved_device, normalize_name};
pub use source::{CpalAudioSource, duration_secs, peak_rms};
