//! The Whisper model in the background (Python `ModelLoader`): created from the settings, loaded on a
//! worker thread at startup and whenever whisper settings change, with a live status for the header.

use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use stt_core::{
    AppConfig, ComputeDevice, LoadEvent, LoadInfo, LoadProgressFn, SecretStore, Transcriber,
    TranscriberFactory,
};

use crate::task::Task;

/// After this long in [`ModelStatus::Loading`] on a GPU, the header explains the one-time shader compile.
pub const WARMUP_HINT_AFTER: Duration = Duration::from_secs(4);

/// What the header shows for Whisper.
#[derive(Debug, Clone, PartialEq)]
pub enum ModelStatus {
    /// No transcriber yet (only before the first [`ModelLoader::start`]).
    Idle,
    /// Downloading the model file.
    Downloading {
        /// File name.
        file: String,
        /// Bytes so far.
        done: u64,
        /// Total bytes.
        total: u64,
        /// Recent speed in bytes per second.
        bytes_per_sec: f64,
    },
    /// Loading onto `device` (includes the warm-up inference; on a GPU the first run after a build or
    /// driver update also compiles shaders, about 20 s).
    Loading {
        /// Device being tried.
        device: ComputeDevice,
        /// When this attempt started (for the "warming up" hint after [`WARMUP_HINT_AFTER`]).
        since: Instant,
    },
    /// The GPU failed; trying the CPU.
    Fallback {
        /// Why.
        reason: String,
    },
    /// Ready.
    Ready(LoadInfo),
    /// Loading failed.
    Failed(String),
}

impl ModelStatus {
    /// True while the status can still change (download/load in progress).
    pub fn is_busy(&self) -> bool {
        matches!(
            self,
            Self::Downloading { .. } | Self::Loading { .. } | Self::Fallback { .. }
        )
    }
}

#[derive(Default)]
struct Speed {
    last: Option<(Instant, u64)>,
}

/// Owns the current [`Transcriber`] and its background load.
pub struct ModelLoader {
    transcriber: Option<Arc<dyn Transcriber>>,
    status: Arc<Mutex<ModelStatus>>,
    load: Option<Task<Result<LoadInfo, String>>>,
}

impl Default for ModelLoader {
    fn default() -> Self {
        Self {
            transcriber: None,
            status: Arc::new(Mutex::new(ModelStatus::Idle)),
            load: None,
        }
    }
}

fn lock<T>(m: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    m.lock().unwrap_or_else(|e| e.into_inner())
}

impl ModelLoader {
    /// (Re)create the transcriber from `cfg` and start loading it in the background. Unloads the previous
    /// one first. Called at startup and after any whisper setting changes.
    pub fn start(
        &mut self,
        factory: &dyn TranscriberFactory,
        cfg: &AppConfig,
        secrets: &dyn SecretStore,
    ) {
        if let Some(old) = self.transcriber.take() {
            old.unload();
        }
        // A fresh status cell, so a still-running old load can't overwrite the new status.
        self.status = Arc::new(Mutex::new(ModelStatus::Loading {
            device: ComputeDevice::Cpu,
            since: Instant::now(),
        }));
        let t = match factory.create(cfg, secrets) {
            Ok(t) => t,
            Err(e) => {
                *lock(&self.status) = ModelStatus::Failed(e.to_string());
                self.load = None;
                return;
            }
        };
        self.transcriber = Some(t.clone());
        let status = self.status.clone();
        let speed = Arc::new(Mutex::new(Speed::default()));
        let on_progress: LoadProgressFn = Arc::new(move |ev: &LoadEvent| {
            let next = match ev {
                LoadEvent::Download(p) => {
                    let now = Instant::now();
                    let mut sp = lock(&speed);
                    let bps = match sp.last {
                        Some((t0, d0)) if p.done >= d0 => {
                            (p.done - d0) as f64 / now.duration_since(t0).as_secs_f64().max(1e-3)
                        }
                        _ => 0.0,
                    };
                    sp.last = Some((now, p.done));
                    ModelStatus::Downloading {
                        file: p.file.clone(),
                        done: p.done,
                        total: p.total,
                        bytes_per_sec: bps,
                    }
                }
                LoadEvent::Loading { device } => ModelStatus::Loading {
                    device: device.clone(),
                    since: Instant::now(),
                },
                LoadEvent::Fallback { reason } => ModelStatus::Fallback {
                    reason: reason.clone(),
                },
                LoadEvent::Ready(info) => ModelStatus::Ready(info.clone()),
            };
            *lock(&status) = next;
        });
        self.load = Some(Task::spawn(move || {
            t.load(Some(on_progress)).map_err(|e| e.to_string())
        }));
    }

    /// Pick up the finished load (call every tick).
    pub fn tick(&mut self) {
        if let Some(task) = &mut self.load
            && let Some(res) = task.poll()
        {
            *lock(&self.status) = match res {
                Ok(info) => ModelStatus::Ready(info),
                Err(e) => ModelStatus::Failed(e),
            };
            self.load = None;
        }
    }

    /// Current status (cheap clone).
    pub fn status(&self) -> ModelStatus {
        lock(&self.status).clone()
    }

    /// True while downloading/loading.
    pub fn is_busy(&self) -> bool {
        self.load.is_some()
    }

    /// The current transcriber, if one was created. Its `transcribe` loads (or waits for the in-flight
    /// load) by itself, so the record flow can call it on a worker thread even while busy.
    pub fn transcriber(&self) -> Option<Arc<dyn Transcriber>> {
        self.transcriber.clone()
    }

    /// Free the model (on quit).
    pub fn unload(&mut self) {
        if let Some(t) = &self.transcriber {
            t.unload();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use stt_core::fakes::{FakeTranscriberFactory, MemorySecretStore};

    #[test]
    fn loads_in_background_and_reports_ready() {
        let f = FakeTranscriberFactory::new("hi");
        let mut m = ModelLoader::default();
        m.start(&f, &AppConfig::default(), &MemorySecretStore::new());
        for _ in 0..200 {
            m.tick();
            if !m.is_busy() {
                break;
            }
            std::thread::sleep(Duration::from_millis(5));
        }
        assert!(
            matches!(m.status(), ModelStatus::Ready(_)),
            "{:?}",
            m.status()
        );
        assert!(m.transcriber().is_some());
    }
}
