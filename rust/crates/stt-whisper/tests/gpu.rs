//! Real-engine tests. `#[ignore]`: run in the `gpu` board slot with
//! `cargo test -p stt-whisper --features whisper -- --ignored`.
//!
//! Models come from `STT_MODELS_DIR` (default the shared `<data-drive>/stt-models`),
//! audio from `STT_AUDIO` (default `reference/gate.wav` in the main checkout). Tests skip when missing.
#![cfg(feature = "whisper")]

use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use stt_core::config::{AppConfig, WhisperDevice, WhisperQuant};
use stt_core::fakes::MemorySecretStore;
use stt_core::progress::LoadEvent;
use stt_core::transcribe::{ComputeDevice, Transcriber, TranscriberFactory};

const MAIN: &str = "<repo>";

fn models_dir() -> PathBuf {
    std::env::var("STT_MODELS_DIR")
        .unwrap_or_else(|_| "<data-drive>/stt-models".into())
        .into()
}

fn gate_audio() -> Option<Vec<f32>> {
    let p = std::env::var("STT_AUDIO").unwrap_or_else(|_| format!("{MAIN}/reference/gate.wav"));
    let mut rd = hound::WavReader::open(&p).ok()?;
    Some(
        rd.samples::<i16>()
            .map(|s| f32::from(s.unwrap()) / 32768.0)
            .collect(),
    )
}

fn make(quant: WhisperQuant, device: WhisperDevice) -> Option<Arc<dyn Transcriber>> {
    let mut cfg = AppConfig::default();
    cfg.whisper.models_dir = models_dir().to_string_lossy().into_owned();
    cfg.whisper.quantization = quant;
    cfg.whisper.device = device;
    let file = stt_whisper::models::resolve(&cfg.whisper.model, quant)
        .unwrap()
        .file;
    if !models_dir().join(file).exists() {
        eprintln!("skipping: {file} not in {}", models_dir().display());
        return None;
    }
    Some(
        stt_whisper::whisper_factory()
            .create(&cfg, &MemorySecretStore::new())
            .unwrap(),
    )
}

fn norm(s: &str) -> String {
    s.to_lowercase()
        .chars()
        .filter(|c| c.is_alphanumeric() || c.is_whitespace())
        .collect::<String>()
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
}

#[test]
#[ignore = "needs the GPU slot, models and audio"]
fn gpu_q5_0_loads_on_vulkan_and_transcribes() {
    let (Some(t), Some(audio)) = (make(WhisperQuant::Q5_0, WhisperDevice::Gpu), gate_audio())
    else {
        return;
    };
    let events = Arc::new(Mutex::new(Vec::new()));
    let e2 = events.clone();
    let info = t
        .load(Some(Arc::new(move |e| e2.lock().unwrap().push(e.clone()))))
        .unwrap();
    match &info.device {
        ComputeDevice::Gpu { backend, name } => {
            assert_eq!(backend, "Vulkan");
            let n = name.clone().unwrap_or_default();
            assert!(!n.to_lowercase().contains("llvmpipe"), "{n}");
            eprintln!("device: {n}");
        }
        d => panic!("expected GPU, got {d:?}"),
    }
    assert_eq!(info.detail, "q5_0");
    assert!(matches!(
        events.lock().unwrap().last(),
        Some(LoadEvent::Ready(_))
    ));

    let text = norm(&t.transcribe(&audio).unwrap());
    eprintln!("text: {text}");
    assert!(text.contains("this is a test"), "{text}");
    assert!(text.contains("shape or form"), "{text}");

    // Silence and near-silence give an empty string.
    assert_eq!(t.transcribe(&vec![0.0; 16_000 * 3]).unwrap(), "");
    let quiet: Vec<f32> = (0..16_000 * 3)
        .map(|i| ((i * 7919 % 1000) as f32 / 1000.0 - 0.5) * 0.002)
        .collect();
    assert_eq!(t.transcribe(&quiet).unwrap(), "");
    // Very short audio does not error.
    t.transcribe(&audio[..800]).unwrap();

    t.unload();
    assert!(t.info().is_none());
}

#[test]
#[ignore = "needs the GPU slot, models and audio"]
fn gpu_f16_matches_q5_0_text() {
    let (Some(a), Some(b), Some(audio)) = (
        make(WhisperQuant::Q5_0, WhisperDevice::Gpu),
        make(WhisperQuant::F16, WhisperDevice::Gpu),
        gate_audio(),
    ) else {
        return;
    };
    assert_eq!(
        norm(&a.transcribe(&audio).unwrap()),
        norm(&b.transcribe(&audio).unwrap())
    );
}

#[test]
#[ignore = "needs the GPU slot, models and audio"]
fn cpu_device_runs_on_cpu() {
    let (Some(t), Some(audio)) = (make(WhisperQuant::Q5_0, WhisperDevice::Cpu), gate_audio())
    else {
        return;
    };
    let info = t.load(None).unwrap();
    assert_eq!(info.device, ComputeDevice::Cpu);
    let text = norm(&t.transcribe(&audio[..16_000 * 5]).unwrap());
    assert!(text.contains("this is a test"), "{text}");
}
