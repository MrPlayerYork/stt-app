//! Whisper speech-to-text via `whisper-rs` (whisper.cpp) with the Vulkan backend.
//!
//! - [`models`]: (`whisper.model` x `whisper.quantization`) -> Hugging Face repo and ggml file.
//! - [`transcriber`]: the [`stt_core::Transcriber`] logic (fetch via `stt-download`, concurrency-safe
//!   load, GPU -> CPU fallback, warm-up, progress events) over an inference [`transcriber::Backend`].
//!   Unit-tested with a fake backend, no whisper.cpp needed.
//! - `engine` (feature `whisper`): the real backend. This is the ONLY crate that compiles whisper.cpp
//!   (feature off by default; the `stt` binary enables it). Building with it needs the `heavy` board
//!   slot; GPU tests are `#[ignore]` and run in the `gpu` slot.
//!
//! With the feature on, use [`WhisperFactory`] as the binary's [`stt_core::TranscriberFactory`].

pub mod models;
pub mod transcriber;

#[cfg(feature = "whisper")]
pub mod engine;

use std::sync::Arc;

use stt_core::config::{AppConfig, resolved_models_dir};
use stt_core::error::TranscribeError;
use stt_core::secrets::{HF_TOKEN_ENV, HF_TOKEN_USER, SecretStore, get_api_key};
use stt_core::transcribe::{Transcriber, TranscriberFactory};

use transcriber::{Backend, Fetcher, Settings, SharedTranscriber, WhisperTranscriber};

/// [`TranscriberFactory`] over any backend (see [`WhisperFactory`] for the real one).
pub struct GenericFactory<B> {
    make_backend: Box<dyn Fn() -> B + Send + Sync>,
    fetcher: Fetcher,
}

impl<B: Backend> GenericFactory<B> {
    /// Factory with a custom backend constructor and downloader.
    pub fn new(make_backend: impl Fn() -> B + Send + Sync + 'static, fetcher: Fetcher) -> Self {
        Self {
            make_backend: Box::new(make_backend),
            fetcher,
        }
    }
}

impl<B: Backend> TranscriberFactory for GenericFactory<B> {
    fn create(
        &self,
        cfg: &AppConfig,
        secrets: &dyn SecretStore,
    ) -> Result<Arc<dyn Transcriber>, TranscribeError> {
        let models_dir = resolved_models_dir(cfg).map_err(|e| TranscribeError::Load {
            model: cfg.whisper.model.clone(),
            message: e.to_string(),
        })?;
        let token = get_api_key(secrets, HF_TOKEN_USER, HF_TOKEN_ENV);
        let settings = Settings::from_config(&cfg.whisper, models_dir, token);
        Ok(Arc::new(SharedTranscriber::new(WhisperTranscriber::new(
            settings,
            (self.make_backend)(),
            self.fetcher.clone(),
        ))))
    }
}

/// The production factory: whisper.cpp engine, real downloader.
#[cfg(feature = "whisper")]
pub type WhisperFactory = GenericFactory<engine::WhisperRsBackend>;

/// Production factory with default decoding settings.
#[cfg(feature = "whisper")]
pub fn whisper_factory() -> WhisperFactory {
    GenericFactory::new(
        engine::WhisperRsBackend::default,
        transcriber::default_fetcher(),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use stt_core::config::WhisperDecoding;
    use stt_core::fakes::MemorySecretStore;
    use transcriber::{DeviceKind, Session};

    struct Nop;
    impl Backend for Nop {
        fn wants_vad(&self) -> bool {
            false
        }
        fn open(
            &self,
            _: &std::path::Path,
            _: Option<&std::path::Path>,
            _: DeviceKind,
            _: Option<&str>,
            _: WhisperDecoding,
        ) -> Result<Box<dyn Session>, String> {
            Err("unused".into())
        }
    }

    #[test]
    fn factory_creates_without_loading() {
        let f = GenericFactory::new(
            || Nop,
            Arc::new(|_, _, _, _, _, _| panic!("no download on create")),
        );
        let mut cfg = AppConfig::default();
        cfg.whisper.models_dir = "/nonexistent/models".into();
        let t = f.create(&cfg, &MemorySecretStore::new()).unwrap();
        assert_eq!(t.model_name(), "large-v3-turbo");
        assert!(t.info().is_none());
        assert_eq!(t.transcribe(&[]).unwrap(), "");
    }

    #[test]
    fn decoding_setting_reaches_the_backend() {
        use std::sync::Mutex;
        struct Spy(Arc<Mutex<Vec<WhisperDecoding>>>);
        impl Backend for Spy {
            fn wants_vad(&self) -> bool {
                false
            }
            fn open(
                &self,
                _: &std::path::Path,
                _: Option<&std::path::Path>,
                _: DeviceKind,
                _: Option<&str>,
                decoding: WhisperDecoding,
            ) -> Result<Box<dyn Session>, String> {
                self.0.lock().unwrap().push(decoding);
                Err("stop".into())
            }
        }
        let dir = std::env::temp_dir().join(format!("stt-whisper-decoding-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("ggml-large-v3-turbo-q5_0.bin"), b"x").unwrap();
        let seen = Arc::new(Mutex::new(Vec::new()));
        let seen2 = seen.clone();
        let f = GenericFactory::new(
            move || Spy(seen2.clone()),
            Arc::new(|_, _, _, _, _, _| panic!("no download")),
        );
        for decoding in [WhisperDecoding::Greedy, WhisperDecoding::Beam] {
            let mut cfg = AppConfig::default();
            cfg.whisper.models_dir = dir.to_string_lossy().into_owned();
            cfg.whisper.device = stt_core::config::WhisperDevice::Cpu;
            cfg.whisper.decoding = decoding;
            let t = f.create(&cfg, &MemorySecretStore::new()).unwrap();
            let _ = t.load(None);
        }
        let _ = std::fs::remove_dir_all(&dir);
        assert_eq!(
            *seen.lock().unwrap(),
            [WhisperDecoding::Greedy, WhisperDecoding::Beam]
        );
    }
}
