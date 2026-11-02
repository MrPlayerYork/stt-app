//! Backend-independent transcriber: file fetching, concurrency-safe loading, GPU -> CPU fallback,
//! warm-up and progress events. The actual inference engine sits behind [`Backend`] so all of this is
//! unit-tested without compiling whisper.cpp.

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, MutexGuard};

use stt_core::config::{WhisperConfig, WhisperDevice, WhisperQuant};
use stt_core::error::{DownloadError, TranscribeError};
use stt_core::progress::{DownloadProgressFn, LoadEvent, LoadProgressFn};
use stt_core::transcribe::{ComputeDevice, LoadInfo, Transcriber};

use crate::models::{self, ModelFile};

/// Which device a [`Backend`] is asked to use.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeviceKind {
    /// Vulkan GPU.
    Gpu,
    /// CPU.
    Cpu,
}

/// Inference engine factory (whisper.cpp in production, a fake in tests).
pub trait Backend: Send + Sync + 'static {
    /// Whether this engine can use the Silero VAD model (so the transcriber fetches it).
    fn wants_vad(&self) -> bool;

    /// Load the model onto `device`. `vad` is the VAD model path if it is available. Must fail (not
    /// silently use another device) when the GPU is requested but unusable. Does not warm up.
    fn open(
        &self,
        path: &Path,
        vad: Option<&Path>,
        device: DeviceKind,
        language: Option<&str>,
    ) -> Result<Box<dyn Session>, String>;
}

/// A loaded model.
pub trait Session: Send {
    /// Where it runs.
    fn device(&self) -> ComputeDevice;
    /// Run a short inference that triggers one-time setup (shader compilation).
    fn warm_up(&mut self) -> Result<(), String>;
    /// Transcribe non-empty mono 16 kHz audio.
    fn transcribe(&mut self, audio: &[f32]) -> Result<String, String>;
}

/// Downloads `file` of Hugging Face repo `repo` to `dest` (`stt_download::download_file` in production).
pub type Fetcher = Arc<
    dyn Fn(
            &str,
            &str,
            &Path,
            Option<&str>,
            u32,
            Option<DownloadProgressFn>,
        ) -> Result<(), DownloadError>
        + Send
        + Sync,
>;

/// The real downloader.
pub fn default_fetcher() -> Fetcher {
    Arc::new(|repo, file, dest, token, connections, on_progress| {
        let url = stt_download::hf_resolve_url(repo, file);
        stt_download::download_file(&url, dest, token, connections, on_progress)
    })
}

/// Settings captured when the transcriber is created.
#[derive(Debug, Clone)]
pub struct Settings {
    /// Model name (`whisper.model`).
    pub model: String,
    /// Weight variant wanted.
    pub quant: WhisperQuant,
    /// Device preference.
    pub device: WhisperDevice,
    /// Language code, `None` = auto.
    pub language: Option<String>,
    /// Where model files live.
    pub models_dir: PathBuf,
    /// Parallel download connections.
    pub connections: u32,
    /// Hugging Face token.
    pub token: Option<String>,
}

impl Settings {
    /// Build from the config section.
    pub fn from_config(cfg: &WhisperConfig, models_dir: PathBuf, token: Option<String>) -> Self {
        Self {
            model: cfg.model.clone(),
            quant: cfg.quantization,
            device: cfg.device,
            language: cfg.language.clone(),
            models_dir,
            connections: cfg.download_connections.max(1),
            token,
        }
    }
}

struct Listener {
    id: u64,
    cb: LoadProgressFn,
}

/// [`Transcriber`] over any [`Backend`].
pub struct WhisperTranscriber<B: Backend> {
    settings: Settings,
    backend: B,
    fetcher: Fetcher,
    /// Held for the whole of a load; serialises loaders.
    load_lock: Mutex<()>,
    /// Loaded model and its info.
    loaded: Mutex<Option<(Box<dyn Session>, LoadInfo)>>,
    /// Callers currently in `load`, so one loader's events reach all of them.
    listeners: Mutex<Vec<Listener>>,
    /// Latest in-flight event, replayed to late joiners.
    last_event: Mutex<Option<LoadEvent>>,
    next_id: AtomicU64,
}

fn lock<T>(m: &Mutex<T>) -> MutexGuard<'_, T> {
    m.lock().unwrap_or_else(|e| e.into_inner())
}

impl<B: Backend> WhisperTranscriber<B> {
    /// Create (does nothing else; no file access).
    pub fn new(settings: Settings, backend: B, fetcher: Fetcher) -> Self {
        Self {
            settings,
            backend,
            fetcher,
            load_lock: Mutex::new(()),
            loaded: Mutex::new(None),
            listeners: Mutex::new(Vec::new()),
            last_event: Mutex::new(None),
            next_id: AtomicU64::new(0),
        }
    }

    fn model_file(&self) -> Result<ModelFile, TranscribeError> {
        models::resolve(&self.settings.model, self.settings.quant).ok_or_else(|| {
            TranscribeError::Load {
                model: self.settings.model.clone(),
                message: format!("unknown model '{}'", self.settings.model),
            }
        })
    }

    fn broadcast(&self, ev: LoadEvent) {
        *lock(&self.last_event) = Some(ev.clone());
        // Clone the callbacks so none runs under the lock (a callback may be slow).
        let cbs: Vec<LoadProgressFn> = lock(&self.listeners)
            .iter()
            .map(|l| l.cb.clone())
            .collect();
        for cb in cbs {
            cb(&ev);
        }
    }

    fn dl_err(&self, source: DownloadError) -> TranscribeError {
        TranscribeError::Download {
            model: self.settings.model.clone(),
            source,
        }
    }

    /// Ensure the model file is local; downloads with `on_progress` when missing.
    fn ensure_model(
        &self,
        mf: &ModelFile,
        on_progress: Option<DownloadProgressFn>,
    ) -> Result<PathBuf, TranscribeError> {
        let dest = self.settings.models_dir.join(mf.file);
        if dest.exists() {
            return Ok(dest);
        }
        (self.fetcher)(
            mf.repo,
            mf.file,
            &dest,
            self.settings.token.as_deref(),
            self.settings.connections,
            on_progress,
        )
        .map_err(|e| self.dl_err(e))?;
        Ok(dest)
    }

    /// The small VAD model; best effort (`None` if unavailable, decoding then runs without VAD).
    fn ensure_vad(&self) -> Option<PathBuf> {
        if !self.backend.wants_vad() {
            return None;
        }
        let dest = self.settings.models_dir.join(models::VAD_FILE);
        if dest.exists() {
            return Some(dest);
        }
        let r = (self.fetcher)(
            models::VAD_REPO,
            models::VAD_FILE,
            &dest,
            self.settings.token.as_deref(),
            1,
            None,
        );
        (r.is_ok() && dest.exists()).then_some(dest)
    }
}

impl<B: Backend> WhisperTranscriber<B> {
    fn do_load_with(self: &Arc<Self>) -> Result<LoadInfo, TranscribeError> {
        let mf = self.model_file()?;
        let this = Arc::clone(self);
        let dl: DownloadProgressFn =
            Arc::new(move |p| this.broadcast(LoadEvent::Download(p.clone())));
        let path = self.ensure_model(&mf, Some(dl))?;
        let vad = self.ensure_vad();

        let attempts: &[DeviceKind] = match self.settings.device {
            WhisperDevice::Auto => &[DeviceKind::Gpu, DeviceKind::Cpu],
            WhisperDevice::Gpu => &[DeviceKind::Gpu],
            WhisperDevice::Cpu => &[DeviceKind::Cpu],
        };
        let mut errors: Vec<String> = Vec::new();
        let mut fallback_reason: Option<String> = None;
        for (i, kind) in attempts.iter().enumerate() {
            let announce = match kind {
                DeviceKind::Gpu => ComputeDevice::Gpu {
                    backend: "Vulkan".into(),
                    name: None,
                },
                DeviceKind::Cpu => ComputeDevice::Cpu,
            };
            self.broadcast(LoadEvent::Loading { device: announce });
            let label = if *kind == DeviceKind::Gpu { "GPU" } else { "CPU" };
            let result = self
                .backend
                .open(&path, vad.as_deref(), *kind, self.settings.language.as_deref())
                .and_then(|mut s| s.warm_up().map(|()| s));
            match result {
                Ok(session) => {
                    let info = LoadInfo {
                        model: self.settings.model.clone(),
                        path: path.clone(),
                        device: session.device(),
                        detail: mf.detail(),
                        fallback_reason: fallback_reason.clone(),
                    };
                    *lock(&self.loaded) = Some((session, info.clone()));
                    return Ok(info);
                }
                Err(e) => {
                    errors.push(format!("{label}: {e}"));
                    if i + 1 < attempts.len() {
                        fallback_reason = Some(e.clone());
                        self.broadcast(LoadEvent::Fallback { reason: e });
                    }
                }
            }
        }
        Err(TranscribeError::Load {
            model: self.settings.model.clone(),
            message: errors.join(" | "),
        })
    }
}

/// `Arc` wrapper so the load closure can broadcast; this is what the factory hands out.
pub struct SharedTranscriber<B: Backend>(Arc<WhisperTranscriber<B>>);

impl<B: Backend> SharedTranscriber<B> {
    /// Wrap a transcriber.
    pub fn new(t: WhisperTranscriber<B>) -> Self {
        Self(Arc::new(t))
    }
}

impl<B: Backend> Transcriber for SharedTranscriber<B> {
    fn model_name(&self) -> &str {
        &self.0.settings.model
    }

    fn fetch(&self, on_progress: Option<DownloadProgressFn>) -> Result<PathBuf, TranscribeError> {
        let mf = self.0.model_file()?;
        let p = self.0.ensure_model(&mf, on_progress)?;
        let _ = self.0.ensure_vad();
        Ok(p)
    }

    fn load(&self, on_progress: Option<LoadProgressFn>) -> Result<LoadInfo, TranscribeError> {
        let t = &self.0;
        // Register before taking the load lock so waiters see the loader's events.
        let id = t.next_id.fetch_add(1, Ordering::Relaxed);
        let got_ready = Arc::new(AtomicBool::new(false));
        if let Some(cb) = &on_progress {
            let flag = got_ready.clone();
            let inner = cb.clone();
            let wrapped: LoadProgressFn = Arc::new(move |ev| {
                if matches!(ev, LoadEvent::Ready(_)) {
                    flag.store(true, Ordering::Relaxed);
                }
                inner(ev);
            });
            let replay = lock(&t.last_event).clone();
            lock(&t.listeners).push(Listener {
                id,
                cb: wrapped.clone(),
            });
            if let Some(ev) = replay {
                wrapped(&ev);
            }
        }
        let guard = lock(&t.load_lock);
        let existing = lock(&t.loaded).as_ref().map(|(_, i)| i.clone());
        let result = match existing {
            Some(info) => Ok(info),
            None => {
                *lock(&t.last_event) = None;
                let r = t.do_load_with();
                if let Ok(info) = &r {
                    t.broadcast(LoadEvent::Ready(info.clone()));
                }
                r
            }
        };
        *lock(&t.last_event) = None;
        drop(guard);
        lock(&t.listeners).retain(|l| l.id != id);
        if let (Ok(info), Some(cb)) = (&result, &on_progress)
            && !got_ready.load(Ordering::Relaxed)
        {
            cb(&LoadEvent::Ready(info.clone()));
        }
        result
    }

    fn info(&self) -> Option<LoadInfo> {
        lock(&self.0.loaded).as_ref().map(|(_, i)| i.clone())
    }

    fn transcribe(&self, audio: &[f32]) -> Result<String, TranscribeError> {
        if audio.is_empty() {
            return Ok(String::new());
        }
        self.load(None)?;
        let mut g = lock(&self.0.loaded);
        let Some((session, _)) = g.as_mut() else {
            // Unloaded between load and here; load again.
            drop(g);
            return self.transcribe(audio);
        };
        session
            .transcribe(audio)
            .map_err(TranscribeError::Transcribe)
    }

    fn unload(&self) {
        // Wait for any in-flight load so it can't re-populate right after.
        let _g = lock(&self.0.load_lock);
        *lock(&self.0.loaded) = None;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::AtomicUsize;
    use std::time::Duration;

    #[derive(Default)]
    struct FakeBackend {
        gpu_error: Option<&'static str>,
        cpu_error: Option<&'static str>,
        opens: Arc<AtomicUsize>,
        runs: Arc<AtomicUsize>,
        delay_ms: u64,
    }

    struct FakeSession(DeviceKind, Arc<AtomicUsize>);

    impl Session for FakeSession {
        fn device(&self) -> ComputeDevice {
            match self.0 {
                DeviceKind::Gpu => ComputeDevice::Gpu {
                    backend: "Vulkan".into(),
                    name: Some("Fake GPU".into()),
                },
                DeviceKind::Cpu => ComputeDevice::Cpu,
            }
        }
        fn warm_up(&mut self) -> Result<(), String> {
            Ok(())
        }
        fn transcribe(&mut self, audio: &[f32]) -> Result<String, String> {
            self.1.fetch_add(1, Ordering::SeqCst);
            Ok(format!("{} samples", audio.len()))
        }
    }

    impl Backend for FakeBackend {
        fn wants_vad(&self) -> bool {
            false
        }
        fn open(
            &self,
            _: &Path,
            _: Option<&Path>,
            device: DeviceKind,
            _: Option<&str>,
        ) -> Result<Box<dyn Session>, String> {
            self.opens.fetch_add(1, Ordering::SeqCst);
            std::thread::sleep(Duration::from_millis(self.delay_ms));
            let err = match device {
                DeviceKind::Gpu => self.gpu_error,
                DeviceKind::Cpu => self.cpu_error,
            };
            match err {
                Some(e) => Err(e.into()),
                None => Ok(Box::new(FakeSession(device, self.runs.clone()))),
            }
        }
    }

    fn temp_models_dir(tag: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("stt-whisper-test-{}-{tag}", std::process::id()));
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    fn no_fetch() -> Fetcher {
        Arc::new(|_, _, _, _, _, _| panic!("fetcher must not be called"))
    }

    fn make(
        tag: &str,
        device: WhisperDevice,
        backend: FakeBackend,
        fetcher: Fetcher,
        create_file: bool,
    ) -> (SharedTranscriber<FakeBackend>, PathBuf) {
        let dir = temp_models_dir(tag);
        if create_file {
            std::fs::write(dir.join("ggml-large-v3-turbo-q5_0.bin"), b"x").unwrap();
        }
        let s = Settings {
            model: "large-v3-turbo".into(),
            quant: WhisperQuant::Q5_0,
            device,
            language: Some("en".into()),
            models_dir: dir.clone(),
            connections: 2,
            token: None,
        };
        (
            SharedTranscriber::new(WhisperTranscriber::new(s, backend, fetcher)),
            dir,
        )
    }

    fn collect() -> (LoadProgressFn, Arc<Mutex<Vec<LoadEvent>>>) {
        let v = Arc::new(Mutex::new(Vec::new()));
        let v2 = v.clone();
        (Arc::new(move |e| v2.lock().unwrap().push(e.clone())), v)
    }

    #[test]
    fn empty_audio_does_not_load() {
        let (t, dir) = make("empty", WhisperDevice::Auto, FakeBackend::default(), no_fetch(), true);
        assert_eq!(t.transcribe(&[]).unwrap(), "");
        assert!(t.info().is_none());
        std::fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn loads_on_gpu_and_transcribes() {
        let (t, dir) = make("gpu", WhisperDevice::Auto, FakeBackend::default(), no_fetch(), true);
        let (cb, evs) = collect();
        let info = t.load(Some(cb)).unwrap();
        assert!(matches!(info.device, ComputeDevice::Gpu { .. }));
        assert_eq!(info.detail, "q5_0");
        assert_eq!(info.fallback_reason, None);
        let evs = evs.lock().unwrap();
        assert!(matches!(evs[0], LoadEvent::Loading { .. }));
        assert!(matches!(evs.last(), Some(LoadEvent::Ready(_))));
        assert_eq!(evs.len(), 2);
        assert_eq!(t.transcribe(&[0.0; 10]).unwrap(), "10 samples");
        t.unload();
        assert!(t.info().is_none());
        std::fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn gpu_failure_falls_back_to_cpu() {
        let b = FakeBackend {
            gpu_error: Some("no vulkan"),
            ..Default::default()
        };
        let (t, dir) = make("fb", WhisperDevice::Auto, b, no_fetch(), true);
        let (cb, evs) = collect();
        let info = t.load(Some(cb)).unwrap();
        assert_eq!(info.device, ComputeDevice::Cpu);
        assert_eq!(info.fallback_reason.as_deref(), Some("no vulkan"));
        let evs = evs.lock().unwrap();
        assert!(matches!(evs[1], LoadEvent::Fallback { .. }));
        assert!(matches!(evs[2], LoadEvent::Loading { device: ComputeDevice::Cpu }));
        std::fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn forced_gpu_does_not_fall_back_and_cpu_only_skips_gpu() {
        let b = FakeBackend {
            gpu_error: Some("boom"),
            ..Default::default()
        };
        let (t, d1) = make("forced", WhisperDevice::Gpu, b, no_fetch(), true);
        let err = t.load(None).unwrap_err().to_string();
        assert!(err.contains("GPU: boom"), "{err}");

        let b = FakeBackend {
            gpu_error: Some("boom"),
            ..Default::default()
        };
        let opens = b.opens.clone();
        let (t, d2) = make("cpuonly", WhisperDevice::Cpu, b, no_fetch(), true);
        assert_eq!(t.load(None).unwrap().device, ComputeDevice::Cpu);
        assert_eq!(opens.load(Ordering::SeqCst), 1);
        std::fs::remove_dir_all(d1).ok();
        std::fs::remove_dir_all(d2).ok();
    }

    #[test]
    fn both_devices_failing_lists_every_attempt() {
        let b = FakeBackend {
            gpu_error: Some("g"),
            cpu_error: Some("c"),
            ..Default::default()
        };
        let (t, dir) = make("both", WhisperDevice::Auto, b, no_fetch(), true);
        let err = t.load(None).unwrap_err().to_string();
        assert!(err.contains("GPU: g | CPU: c"), "{err}");
        std::fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn concurrent_loads_open_once_and_all_get_ready() {
        let b = FakeBackend {
            delay_ms: 150,
            ..Default::default()
        };
        let opens = b.opens.clone();
        let (t, dir) = make("conc", WhisperDevice::Auto, b, no_fetch(), true);
        let t = Arc::new(t);
        let mut hs = Vec::new();
        for i in 0..4 {
            let t = t.clone();
            hs.push(std::thread::spawn(move || {
                std::thread::sleep(Duration::from_millis(i * 20));
                let (cb, evs) = collect();
                let info = t.load(Some(cb)).unwrap();
                let evs = evs.lock().unwrap().clone();
                (info, evs)
            }));
        }
        for h in hs {
            let (info, evs) = h.join().unwrap();
            assert!(matches!(info.device, ComputeDevice::Gpu { .. }));
            assert!(matches!(evs.last(), Some(LoadEvent::Ready(_))), "{evs:?}");
            assert!(
                evs.iter().any(|e| matches!(e, LoadEvent::Loading { .. })),
                "late joiner should still see Loading: {evs:?}"
            );
        }
        assert_eq!(opens.load(Ordering::SeqCst), 1);
        std::fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn missing_file_is_downloaded_with_progress() {
        let calls = Arc::new(AtomicUsize::new(0));
        let c2 = calls.clone();
        let fetcher: Fetcher = Arc::new(move |repo, file, dest, _, conns, _| {
            assert_eq!((repo, file), ("ggerganov/whisper.cpp", "ggml-large-v3-turbo-q5_0.bin"));
            assert_eq!(conns, 2);
            c2.fetch_add(1, Ordering::SeqCst);
            std::fs::write(dest, b"x").unwrap();
            Ok(())
        });
        let (t, dir) = make("dl", WhisperDevice::Auto, FakeBackend::default(), fetcher, false);
        let p = t.fetch(None).unwrap();
        assert!(p.ends_with("ggml-large-v3-turbo-q5_0.bin"));
        assert_eq!(calls.load(Ordering::SeqCst), 1);
        t.load(None).unwrap();
        assert_eq!(calls.load(Ordering::SeqCst), 1, "existing file is reused");
        std::fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn download_failure_maps_to_download_error() {
        let fetcher: Fetcher = Arc::new(|_, _, _, _, _, _| {
            Err(DownloadError::Network("offline".into()))
        });
        let (t, dir) = make("dlfail", WhisperDevice::Auto, FakeBackend::default(), fetcher, false);
        assert!(matches!(t.load(None), Err(TranscribeError::Download { .. })));
        std::fs::remove_dir_all(dir).ok();
    }
}
