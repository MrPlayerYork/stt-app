//! In-memory fakes of every `stt-core` trait, for other crates' tests (enable the `fakes` feature in
//! `[dev-dependencies]`). They never touch hardware, the network or the real keyring, and they record
//! their calls so tests can assert on them.
//!
//! All fakes are cheap to clone where shared state matters (`Arc` inside), so a test can keep a handle
//! while the code under test owns a boxed copy.

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use crate::audio::{AudioSource, DeviceId, InputDevice, MultiRecorder, Recorder};
use crate::clipboard::{Clipboard, ClipboardMethod};
use crate::config::{AppConfig, LlmConfig};
use crate::error::{AudioError, ClipboardError, LlmError, SecretError, TranscribeError};
use crate::llm::{LlmBackend, LlmFactory, LlmResult, ServerKind};
use crate::progress::{DownloadProgress, DownloadProgressFn, LoadEvent, LoadProgressFn};
use crate::secrets::SecretStore;
use crate::transcribe::{ComputeDevice, LoadInfo, Transcriber, TranscriberFactory};

fn lock<T>(m: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    m.lock().unwrap_or_else(|e| e.into_inner())
}

// ------------------------------------------------------------------ Transcriber

type Responder = dyn Fn(&[f32]) -> Result<String, String> + Send + Sync;

/// Fake [`Transcriber`]: "loads" instantly and answers every non-empty clip via a responder function.
pub struct FakeTranscriber {
    model: String,
    responder: Box<Responder>,
    load_error: Option<String>,
    gpu_failure: Option<String>,
    loaded: Mutex<Option<LoadInfo>>,
    loads: AtomicUsize,
    calls: Mutex<Vec<usize>>,
}

impl FakeTranscriber {
    /// Answers every non-empty clip with `text`.
    pub fn new(text: impl Into<String>) -> Self {
        let text = text.into();
        Self::with_fn(move |_| Ok(text.clone()))
    }

    /// Answers with `f(audio)`; `Err(msg)` becomes [`TranscribeError::Transcribe`]. Useful for
    /// calibration tests that need a different transcript per device (e.g. keyed on clip length).
    pub fn with_fn(f: impl Fn(&[f32]) -> Result<String, String> + Send + Sync + 'static) -> Self {
        Self {
            model: "large-v3-turbo".into(),
            responder: Box::new(f),
            load_error: None,
            gpu_failure: None,
            loaded: Mutex::new(None),
            loads: AtomicUsize::new(0),
            calls: Mutex::new(Vec::new()),
        }
    }

    /// Set the model name.
    pub fn model(mut self, name: impl Into<String>) -> Self {
        self.model = name.into();
        self
    }

    /// Make every `load` (and so every `transcribe`) fail with [`TranscribeError::Load`].
    pub fn failing_load(mut self, message: impl Into<String>) -> Self {
        self.load_error = Some(message.into());
        self
    }

    /// Simulate a GPU failure: `load` emits `Fallback` and ends up on the CPU.
    pub fn gpu_fails(mut self, reason: impl Into<String>) -> Self {
        self.gpu_failure = Some(reason.into());
        self
    }

    /// Lengths (in samples) of every clip passed to `transcribe`, in order.
    pub fn calls(&self) -> Vec<usize> {
        lock(&self.calls).clone()
    }

    /// How many times a load actually happened (not counting already-loaded returns).
    pub fn load_count(&self) -> usize {
        self.loads.load(Ordering::SeqCst)
    }

    fn path(&self) -> PathBuf {
        PathBuf::from(format!("fake-models/ggml-{}.bin", self.model))
    }
}

impl Transcriber for FakeTranscriber {
    fn model_name(&self) -> &str {
        &self.model
    }

    fn fetch(&self, on_progress: Option<DownloadProgressFn>) -> Result<PathBuf, TranscribeError> {
        if let Some(cb) = on_progress {
            cb(&DownloadProgress {
                file: format!("ggml-{}.bin", self.model),
                done: 100,
                total: 100,
            });
        }
        Ok(self.path())
    }

    fn load(&self, on_progress: Option<LoadProgressFn>) -> Result<LoadInfo, TranscribeError> {
        let mut loaded = lock(&self.loaded);
        if let Some(info) = loaded.as_ref() {
            return Ok(info.clone());
        }
        let emit = |e: LoadEvent| {
            if let Some(cb) = &on_progress {
                cb(&e);
            }
        };
        if let Some(message) = &self.load_error {
            return Err(TranscribeError::Load {
                model: self.model.clone(),
                message: message.clone(),
            });
        }
        let path = self.path();
        emit(LoadEvent::Download(DownloadProgress {
            file: format!("ggml-{}.bin", self.model),
            done: 100,
            total: 100,
        }));
        let gpu = ComputeDevice::Gpu {
            backend: "Fake".into(),
            name: None,
        };
        emit(LoadEvent::Loading {
            device: gpu.clone(),
        });
        let device = match &self.gpu_failure {
            Some(reason) => {
                emit(LoadEvent::Fallback {
                    reason: reason.clone(),
                });
                emit(LoadEvent::Loading {
                    device: ComputeDevice::Cpu,
                });
                ComputeDevice::Cpu
            }
            None => gpu,
        };
        let info = LoadInfo {
            model: self.model.clone(),
            path,
            device,
            detail: String::new(),
            fallback_reason: self.gpu_failure.clone(),
        };
        self.loads.fetch_add(1, Ordering::SeqCst);
        emit(LoadEvent::Ready(info.clone()));
        *loaded = Some(info.clone());
        Ok(info)
    }

    fn info(&self) -> Option<LoadInfo> {
        lock(&self.loaded).clone()
    }

    fn transcribe(&self, audio: &[f32]) -> Result<String, TranscribeError> {
        if audio.is_empty() {
            return Ok(String::new());
        }
        self.load(None)?;
        lock(&self.calls).push(audio.len());
        (self.responder)(audio)
            .map(|t| t.trim().to_string())
            .map_err(TranscribeError::Transcribe)
    }

    fn unload(&self) {
        *lock(&self.loaded) = None;
    }
}

/// Fake [`TranscriberFactory`]: creates a [`FakeTranscriber`] answering `text` for the configured model,
/// and remembers every one it created.
pub struct FakeTranscriberFactory {
    text: String,
    created: Mutex<Vec<Arc<FakeTranscriber>>>,
}

impl FakeTranscriberFactory {
    /// Every created transcriber answers `text`.
    pub fn new(text: impl Into<String>) -> Self {
        Self {
            text: text.into(),
            created: Mutex::new(Vec::new()),
        }
    }

    /// Every transcriber created so far, oldest first.
    pub fn created(&self) -> Vec<Arc<FakeTranscriber>> {
        lock(&self.created).clone()
    }
}

impl TranscriberFactory for FakeTranscriberFactory {
    fn create(
        &self,
        cfg: &AppConfig,
        _secrets: &dyn SecretStore,
    ) -> Result<Arc<dyn Transcriber>, TranscribeError> {
        let t = Arc::new(FakeTranscriber::new(self.text.clone()).model(cfg.whisper.model.clone()));
        lock(&self.created).push(t.clone());
        Ok(t)
    }
}

// ------------------------------------------------------------------ LLM

#[derive(Default)]
struct FakeLlmState {
    result: LlmResult,
    failure: Option<String>,
    models: Vec<String>,
    server_kind: ServerKind,
    calls: Vec<(String, String)>,
    unloads: usize,
}

/// Fake [`LlmBackend`]: returns a fixed [`LlmResult`] (or a fixed failure) and records every call.
/// Clones share state.
#[derive(Clone, Default)]
pub struct FakeLlm {
    state: Arc<Mutex<FakeLlmState>>,
}

impl FakeLlm {
    /// Answers every completion with `text` (no reasoning, `reasoned: Some(false)`).
    pub fn new(text: impl Into<String>) -> Self {
        Self::with_result(LlmResult {
            text: text.into(),
            reasoned: Some(false),
            ..LlmResult::default()
        })
    }

    /// Answers every completion with `result`.
    pub fn with_result(result: LlmResult) -> Self {
        let llm = Self::default();
        lock(&llm.state).result = result;
        llm
    }

    /// Make every call fail with [`LlmError::Connection`] carrying `message`.
    pub fn failing(self, message: impl Into<String>) -> Self {
        lock(&self.state).failure = Some(message.into());
        self
    }

    /// Set the list returned by `list_models`.
    pub fn models(self, models: &[&str]) -> Self {
        lock(&self.state).models = models.iter().map(|m| m.to_string()).collect();
        self
    }

    /// Set the reported server kind.
    pub fn server_kind(self, kind: ServerKind) -> Self {
        lock(&self.state).server_kind = kind;
        self
    }

    /// Every `(system_prompt, text)` passed to `complete`, in order.
    pub fn calls(&self) -> Vec<(String, String)> {
        lock(&self.state).calls.clone()
    }

    /// How many times `unload` was called.
    pub fn unload_count(&self) -> usize {
        lock(&self.state).unloads
    }

    fn fail(&self) -> Result<(), LlmError> {
        match &lock(&self.state).failure {
            Some(m) => Err(LlmError::Connection {
                url: "fake".into(),
                message: m.clone(),
            }),
            None => Ok(()),
        }
    }
}

impl LlmBackend for FakeLlm {
    fn complete(&self, system_prompt: &str, text: &str) -> Result<LlmResult, LlmError> {
        lock(&self.state)
            .calls
            .push((system_prompt.to_string(), text.to_string()));
        self.fail()?;
        Ok(lock(&self.state).result.clone())
    }

    fn list_models(&self) -> Result<Vec<String>, LlmError> {
        self.fail()?;
        Ok(lock(&self.state).models.clone())
    }

    fn unload(&self) -> Result<(), LlmError> {
        lock(&self.state).unloads += 1;
        self.fail()
    }

    fn server_kind(&self) -> ServerKind {
        lock(&self.state).server_kind
    }
}

/// Fake [`LlmFactory`]: hands out clones of one [`FakeLlm`], applying the same key / base-URL checks as
/// Python's `make_backend`. Records each config it was asked for.
#[derive(Clone, Default)]
pub struct FakeLlmFactory {
    /// The backend every `create` returns a clone of (keep a handle to inspect calls).
    pub llm: FakeLlm,
    requests: Arc<Mutex<Vec<LlmConfig>>>,
}

impl FakeLlmFactory {
    /// A factory that returns clones of `llm`.
    pub fn new(llm: FakeLlm) -> Self {
        Self {
            llm,
            requests: Arc::default(),
        }
    }

    /// Every config passed to `create`, in order.
    pub fn requests(&self) -> Vec<LlmConfig> {
        lock(&self.requests).clone()
    }
}

impl LlmFactory for FakeLlmFactory {
    fn create(
        &self,
        cfg: &LlmConfig,
        api_key: Option<&str>,
    ) -> Result<Box<dyn LlmBackend>, LlmError> {
        lock(&self.requests).push(cfg.clone());
        if cfg.provider.needs_api_key() && api_key.is_none_or(str::is_empty) {
            let env = if cfg.api_key_env.is_empty() {
                "?"
            } else {
                cfg.api_key_env.as_str()
            };
            return Err(LlmError::NoApiKey {
                provider: cfg.provider.label().into(),
                env: env.into(),
            });
        }
        if !cfg.provider.needs_api_key() && cfg.base_url.is_empty() {
            return Err(LlmError::NoBaseUrl);
        }
        Ok(Box::new(self.llm.clone()))
    }
}

// ------------------------------------------------------------------ Audio

#[derive(Default)]
struct FakeAudioState {
    devices: Vec<InputDevice>,
    clips: BTreeMap<DeviceId, Vec<f32>>,
    fail_open: BTreeSet<DeviceId>,
    level: f32,
}

/// Fake [`AudioSource`]: a fixed device list; each device "records" a preset clip.
/// Clones share state.
#[derive(Clone, Default)]
pub struct FakeAudioSource {
    state: Arc<Mutex<FakeAudioState>>,
}

impl FakeAudioSource {
    /// No devices yet; add some with [`Self::device`].
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a device that returns `clip` when a recording stops. The id equals the name.
    pub fn device(self, name: &str, is_monitor: bool, clip: Vec<f32>) -> Self {
        let id = DeviceId(name.to_string());
        {
            let mut s = lock(&self.state);
            s.devices.push(InputDevice {
                id: id.clone(),
                name: name.to_string(),
                host: "fake".into(),
                channels: 1,
                default_sample_rate: crate::SAMPLE_RATE,
                is_monitor,
            });
            s.clips.insert(id, clip);
        }
        self
    }

    /// Make opening `name` fail (single and multi recorders).
    pub fn failing_open(self, name: &str) -> Self {
        lock(&self.state)
            .fail_open
            .insert(DeviceId(name.to_string()));
        self
    }

    /// Level every running recorder reports (default 0.0).
    pub fn level(self, level: f32) -> Self {
        lock(&self.state).level = level;
        self
    }
}

impl AudioSource for FakeAudioSource {
    fn list_input_devices(&self) -> Result<Vec<InputDevice>, AudioError> {
        Ok(lock(&self.state).devices.clone())
    }

    fn recorder(&self, device: Option<&DeviceId>) -> Result<Box<dyn Recorder>, AudioError> {
        let s = lock(&self.state);
        let id = match device {
            Some(id) => id.clone(),
            None => s.devices.first().ok_or(AudioError::NoDevices)?.id.clone(),
        };
        if !s.clips.contains_key(&id) {
            return Err(AudioError::DeviceNotFound(id.0));
        }
        Ok(Box::new(FakeRecorder {
            source: self.clone(),
            id,
            recording: false,
        }))
    }

    fn multi_recorder(&self, devices: &[InputDevice]) -> Box<dyn MultiRecorder> {
        Box::new(FakeMultiRecorder {
            source: self.clone(),
            devices: devices.iter().map(|d| d.id.clone()).collect(),
            running: BTreeSet::new(),
        })
    }
}

/// Recorder handed out by [`FakeAudioSource::recorder`].
pub struct FakeRecorder {
    source: FakeAudioSource,
    id: DeviceId,
    recording: bool,
}

impl Recorder for FakeRecorder {
    fn start(&mut self) -> Result<(), AudioError> {
        if lock(&self.source.state).fail_open.contains(&self.id) {
            return Err(AudioError::Open {
                device: self.id.0.clone(),
                message: "fake failure".into(),
            });
        }
        self.recording = true;
        Ok(())
    }

    fn stop(&mut self) -> Result<Vec<f32>, AudioError> {
        if !std::mem::take(&mut self.recording) {
            return Ok(Vec::new());
        }
        Ok(lock(&self.source.state)
            .clips
            .get(&self.id)
            .cloned()
            .unwrap_or_default())
    }

    fn level(&self) -> f32 {
        if self.recording {
            lock(&self.source.state).level
        } else {
            0.0
        }
    }

    fn is_recording(&self) -> bool {
        self.recording
    }
}

/// Multi-recorder handed out by [`FakeAudioSource::multi_recorder`].
pub struct FakeMultiRecorder {
    source: FakeAudioSource,
    devices: Vec<DeviceId>,
    running: BTreeSet<DeviceId>,
}

impl MultiRecorder for FakeMultiRecorder {
    fn start(&mut self) -> BTreeMap<DeviceId, String> {
        let s = lock(&self.source.state);
        let mut errors = BTreeMap::new();
        for id in &self.devices {
            if self.running.contains(id) {
                continue;
            }
            if s.fail_open.contains(id) || !s.clips.contains_key(id) {
                errors.insert(id.clone(), "fake failure".to_string());
            } else {
                self.running.insert(id.clone());
            }
        }
        errors
    }

    fn stop(&mut self) -> BTreeMap<DeviceId, Vec<f32>> {
        let s = lock(&self.source.state);
        std::mem::take(&mut self.running)
            .into_iter()
            .map(|id| {
                let clip = s.clips.get(&id).cloned().unwrap_or_default();
                (id, clip)
            })
            .collect()
    }

    fn levels(&self) -> BTreeMap<DeviceId, f32> {
        let level = lock(&self.source.state).level;
        self.running.iter().map(|id| (id.clone(), level)).collect()
    }
}

// ------------------------------------------------------------------ Clipboard

/// Fake [`Clipboard`]: remembers every copy. Clones share state.
#[derive(Clone)]
pub struct FakeClipboard {
    copies: Arc<Mutex<Vec<String>>>,
    method: ClipboardMethod,
    fail: bool,
}

impl Default for FakeClipboard {
    fn default() -> Self {
        Self {
            copies: Arc::default(),
            method: ClipboardMethod::WlCopy,
            fail: false,
        }
    }
}

impl FakeClipboard {
    /// Succeeds, reporting `wl-copy`.
    pub fn new() -> Self {
        Self::default()
    }

    /// Every copy fails with [`ClipboardError::NoTool`].
    pub fn failing() -> Self {
        Self {
            fail: true,
            ..Self::default()
        }
    }

    /// Everything copied so far, oldest first.
    pub fn copies(&self) -> Vec<String> {
        lock(&self.copies).clone()
    }

    /// The current clipboard contents (last copy).
    pub fn contents(&self) -> Option<String> {
        lock(&self.copies).last().cloned()
    }
}

impl Clipboard for FakeClipboard {
    fn copy(&self, text: &str) -> Result<ClipboardMethod, ClipboardError> {
        if self.fail {
            return Err(ClipboardError::NoTool);
        }
        lock(&self.copies).push(text.to_string());
        Ok(self.method)
    }
}

// ------------------------------------------------------------------ Secrets

/// In-memory [`SecretStore`] (never touches the real keyring).
#[derive(Debug, Default)]
pub struct MemorySecretStore {
    entries: Mutex<HashMap<String, String>>,
}

impl MemorySecretStore {
    /// Empty store.
    pub fn new() -> Self {
        Self::default()
    }

    /// Store pre-filled with `(user, secret)` pairs.
    pub fn with(entries: &[(&str, &str)]) -> Self {
        let store = Self::default();
        lock(&store.entries).extend(entries.iter().map(|(u, s)| (u.to_string(), s.to_string())));
        store
    }
}

impl SecretStore for MemorySecretStore {
    fn get(&self, user: &str) -> Result<Option<String>, SecretError> {
        Ok(lock(&self.entries).get(user).cloned())
    }

    fn set(&self, user: &str, secret: &str) -> Result<(), SecretError> {
        lock(&self.entries).insert(user.to_string(), secret.to_string());
        Ok(())
    }

    fn delete(&self, user: &str) -> Result<(), SecretError> {
        lock(&self.entries).remove(user);
        Ok(())
    }
}
