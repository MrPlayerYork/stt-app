//! Timing helper: `cargo run --release --features whisper --example bench -- <gpu|cpu> <beam> <runs> [threads]`.
//!
//! Env: `STT_MODELS_DIR` (default the shared models dir), `STT_AUDIO` (16 kHz mono wav), `STT_MODEL`,
//! `STT_QUANT`. Prints the load+warm-up time and each transcribe time. Run in the `gpu` slot.

use std::sync::Arc;
use std::time::Instant;

use stt_core::config::{WhisperConfig, WhisperDecoding, WhisperDevice, WhisperQuant};
use stt_core::transcribe::Transcriber;
use stt_whisper::engine::WhisperRsBackend;
use stt_whisper::transcriber::{Settings, SharedTranscriber, WhisperTranscriber, default_fetcher};

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let device = a.get(1).map_or("gpu", String::as_str);
    let beam: i32 = a.get(2).and_then(|s| s.parse().ok()).unwrap_or(5);
    let runs: usize = a.get(3).and_then(|s| s.parse().ok()).unwrap_or(3);
    let threads: i32 = a.get(4).and_then(|s| s.parse().ok()).unwrap_or(0);

    let dir = std::env::var("STT_MODELS_DIR")
        .unwrap_or_else(|_| "<data-drive>/stt-models".into());
    let audio_path = std::env::var("STT_AUDIO").expect("set STT_AUDIO to a 16 kHz mono wav");
    let cfg = WhisperConfig {
        model: std::env::var("STT_MODEL").unwrap_or_else(|_| "large-v3-turbo".into()),
        quantization: std::env::var("STT_QUANT")
            .ok()
            .and_then(|q| WhisperQuant::parse(&q))
            .unwrap_or(WhisperQuant::Q5_0),
        device: if device == "cpu" {
            WhisperDevice::Cpu
        } else {
            WhisperDevice::Gpu
        },
        decoding: if beam == 1 {
            WhisperDecoding::Greedy
        } else {
            WhisperDecoding::Beam
        },
        ..WhisperConfig::default()
    };
    let settings = Settings::from_config(&cfg, dir.into(), None);
    let backend = WhisperRsBackend {
        beam_size: beam,
        cpu_threads: threads,
    };
    let t: Arc<dyn Transcriber> = Arc::new(SharedTranscriber::new(WhisperTranscriber::new(
        settings,
        backend,
        default_fetcher(),
    )));

    let mut rd = hound::WavReader::open(&audio_path).expect("open wav");
    let audio: Vec<f32> = rd
        .samples::<i16>()
        .map(|s| f32::from(s.unwrap()) / 32768.0)
        .collect();

    let t0 = Instant::now();
    let info = t.load(None).expect("load");
    println!(
        "LOAD_WARMUP_S {:.3} ({})",
        t0.elapsed().as_secs_f64(),
        info.description()
    );
    let mut best = f64::MAX;
    let mut text = String::new();
    for i in 0..runs {
        let t0 = Instant::now();
        text = t.transcribe(&audio).expect("transcribe");
        let dt = t0.elapsed().as_secs_f64();
        println!("RUN {} TRANSCRIBE_S {dt:.3}", i + 1);
        best = best.min(dt);
    }
    println!("BEST_S {best:.3}");
    println!("TEXT: {text}");
}
