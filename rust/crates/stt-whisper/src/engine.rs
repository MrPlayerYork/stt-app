//! The real [`Backend`]: whisper.cpp through `whisper-rs` (Vulkan or CPU). Only built with the
//! `whisper` feature.
//!
//! Decoding mirrors the Python app (faster-whisper defaults): beam search with 5 beams, configured
//! language (or auto), temperature fallback, no-speech suppression, and Silero VAD (whisper.cpp's
//! built-in VAD stage) so silence gives an empty string.

use std::path::Path;

use stt_core::config::WhisperDecoding;
use stt_core::transcribe::ComputeDevice;
use whisper_rs::{
    FullParams, SamplingStrategy, WhisperContext, WhisperContextParameters, WhisperState,
    WhisperVadParams,
};

use crate::transcriber::{Backend, DeviceKind, Session};

/// Beams used by default (faster-whisper's `beam_size`).
pub const BEAM_SIZE: i32 = 5;
/// Segments whose no-speech probability is above this are dropped by whisper.cpp when also low-confidence.
const NO_SPEECH_THOLD: f32 = 0.6;
/// whisper.cpp wants at least about a second of audio; shorter clips are zero-padded to this.
const MIN_SAMPLES: usize = 16_000 + 1_600;

/// whisper-rs engine settings.
#[derive(Debug, Clone)]
pub struct WhisperRsBackend {
    /// Beams used when `whisper.decoding` is beam; `1` makes that greedy too. Greedy decoding always uses 1.
    pub beam_size: i32,
    /// Threads for CPU inference; `0` = pick automatically (see [`default_cpu_threads`]).
    pub cpu_threads: i32,
}

impl Default for WhisperRsBackend {
    fn default() -> Self {
        Self {
            beam_size: BEAM_SIZE,
            cpu_threads: 0,
        }
    }
}

/// CPU thread count when the user does not choose: half the hardware threads, clamped to 4..=8.
pub fn default_cpu_threads() -> i32 {
    let n = std::thread::available_parallelism().map_or(4, |n| n.get());
    (n / 2).clamp(4, 8) as i32
}

impl Backend for WhisperRsBackend {
    fn wants_vad(&self) -> bool {
        true
    }

    fn open(
        &self,
        path: &Path,
        vad: Option<&Path>,
        device: DeviceKind,
        language: Option<&str>,
        decoding: WhisperDecoding,
    ) -> Result<Box<dyn Session>, String> {
        // ggml/whisper.cpp print to stderr by default, which would corrupt the inline TUI. With neither
        // the `log` nor the `tracing` backend enabled this discards everything.
        whisper_rs::install_logging_hooks();

        let (compute, threads) = match device {
            DeviceKind::Gpu => (gpu_device()?, 4),
            DeviceKind::Cpu => (
                ComputeDevice::Cpu,
                if self.cpu_threads > 0 {
                    self.cpu_threads
                } else {
                    default_cpu_threads()
                },
            ),
        };
        let mut cp = WhisperContextParameters::default();
        cp.use_gpu(device == DeviceKind::Gpu);
        let ctx = WhisperContext::new_with_params(path, cp)
            .map_err(|e| format!("could not load {}: {e}", path.display()))?;
        let state = ctx
            .create_state()
            .map_err(|e| format!("could not create inference state: {e}"))?;
        Ok(Box::new(RsSession {
            _ctx: ctx,
            state,
            compute,
            vad: vad.map(|p| p.to_string_lossy().into_owned()),
            language: language.map(str::to_owned),
            beam_size: match decoding {
                WhisperDecoding::Beam => self.beam_size,
                WhisperDecoding::Greedy => 1,
            },
            threads,
        }))
    }
}

/// The Vulkan device whisper.cpp will use (index 0), or an error when there is no real GPU.
/// whisper.cpp itself silently falls back to the CPU when it finds no device, so check first.
fn gpu_device() -> Result<ComputeDevice, String> {
    let devices = whisper_rs::vulkan::list_devices();
    let Some(first) = devices.first() else {
        return Err("no Vulkan device found".into());
    };
    let lower = first.name.to_lowercase();
    if lower.contains("llvmpipe") || lower.contains("lavapipe") || lower.contains("swiftshader") {
        return Err(format!(
            "only a software Vulkan device found ({})",
            first.name
        ));
    }
    Ok(ComputeDevice::Gpu {
        backend: "Vulkan".into(),
        name: Some(first.name.clone()),
    })
}

struct RsSession {
    _ctx: WhisperContext,
    state: WhisperState,
    compute: ComputeDevice,
    vad: Option<String>,
    language: Option<String>,
    beam_size: i32,
    threads: i32,
}

impl RsSession {
    fn run(&mut self, audio: &[f32], use_vad: bool) -> Result<String, String> {
        let strategy = if self.beam_size > 1 {
            SamplingStrategy::BeamSearch {
                beam_size: self.beam_size,
                patience: -1.0,
            }
        } else {
            SamplingStrategy::Greedy { best_of: 1 }
        };
        let mut p = FullParams::new(strategy);
        p.set_language(self.language.as_deref());
        p.set_n_threads(self.threads);
        p.set_translate(false);
        p.set_no_context(true);
        p.set_suppress_blank(true);
        p.set_suppress_nst(true);
        p.set_no_speech_thold(NO_SPEECH_THOLD);
        p.set_print_progress(false);
        p.set_print_realtime(false);
        p.set_print_special(false);
        p.set_print_timestamps(false);
        if use_vad && let Some(v) = &self.vad {
            p.set_vad_model_path(Some(v));
            p.set_vad_params(WhisperVadParams::default());
            p.enable_vad(true);
        }

        let padded;
        let data = if audio.len() < MIN_SAMPLES {
            let mut v = audio.to_vec();
            v.resize(MIN_SAMPLES, 0.0);
            padded = v;
            &padded[..]
        } else {
            audio
        };
        self.state.full(p, data).map_err(|e| e.to_string())?;

        let mut parts = Vec::new();
        for i in 0..self.state.full_n_segments() {
            let Some(seg) = self.state.get_segment(i) else {
                continue;
            };
            let text = seg.to_str_lossy().map_err(|e| e.to_string())?;
            let text = text.trim();
            if !text.chars().any(char::is_alphanumeric) || is_non_speech_marker(text) {
                continue;
            }
            parts.push(text.to_owned());
        }
        Ok(parts.join(" ").trim().to_owned())
    }
}

/// `[BLANK_AUDIO]`, `(silence)`, `[Music]` ...: whisper's markers for "nothing said".
fn is_non_speech_marker(t: &str) -> bool {
    (t.starts_with('[') && t.ends_with(']')) || (t.starts_with('(') && t.ends_with(')'))
}

impl Session for RsSession {
    fn device(&self) -> ComputeDevice {
        self.compute.clone()
    }

    fn warm_up(&mut self) -> Result<(), String> {
        // No VAD here: it would drop the silence before the encoder/decoder ever run, and the point
        // is to run them once (compiles the Vulkan shaders, about 16 s the first time on this machine).
        self.run(&vec![0.0; 16_000], false).map(|_| ())
    }

    fn transcribe(&mut self, audio: &[f32]) -> Result<String, String> {
        if audio.iter().all(|s| s.abs() < 1e-6) {
            return Ok(String::new());
        }
        self.run(audio, true)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn markers() {
        assert!(is_non_speech_marker("[BLANK_AUDIO]"));
        assert!(is_non_speech_marker("(silence)"));
        assert!(!is_non_speech_marker("Hello [there]"));
    }

    #[test]
    fn thread_default_in_range() {
        assert!((4..=8).contains(&default_cpu_threads()));
    }
}
