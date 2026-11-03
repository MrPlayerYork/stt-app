//! One capture stream on its own thread. `cpal::Stream` is `!Send`, so the thread builds, owns and drops
//! the stream; the rest of the program talks to it through shared state and a stop channel.

use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::sync::mpsc::{self, RecvTimeoutError, Sender};
use std::sync::{Arc, Mutex};
use std::thread::{self, JoinHandle};
use std::time::Duration;

use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{FromSample, SizedSample};
use stt_core::AudioError;

use crate::convert::to_mono_16k;
use crate::source::preferred_hosts;

/// Python `MAX_CHANNELS`: open at most this many channels, then average them.
const MAX_CHANNELS: u16 = 2;
/// How long to wait for the backend to open a stream before giving up.
const OPEN_TIMEOUT: Duration = Duration::from_secs(10);

/// Which device to open.
#[derive(Debug, Clone)]
pub(crate) enum Target {
    /// The system default input.
    Default,
    /// A cpal device id string (`"pulseaudio:alsa_input...."`).
    Id(String),
}

#[derive(Default)]
struct Shared {
    /// Whether the callback keeps samples (recording) or only measures the level (listening).
    keep: AtomicBool,
    /// Mono samples at the device's native rate.
    samples: Mutex<Vec<f32>>,
    /// Most recent block RMS, as `f32` bits.
    level: AtomicU32,
    /// First error reported by the stream while running.
    error: Mutex<Option<String>>,
}

/// A running capture. Dropping it stops the stream, joins the thread and closes the device.
pub(crate) struct Capture {
    shared: Arc<Shared>,
    rate: u32,
    stop_tx: Option<Sender<()>>,
    thread: Option<JoinHandle<()>>,
}

/// A capture whose device is still opening on its thread.
pub(crate) struct Pending {
    shared: Arc<Shared>,
    ready_rx: mpsc::Receiver<Result<u32, String>>,
    stop_tx: Sender<()>,
    thread: JoinHandle<()>,
    label: String,
}

impl Pending {
    /// Wait for the device to open (up to [`OPEN_TIMEOUT`]).
    pub(crate) fn wait(self) -> Result<Capture, AudioError> {
        let Pending {
            shared,
            ready_rx,
            stop_tx,
            thread,
            label,
        } = self;
        let open_err = |message: String| AudioError::Open {
            device: label.clone(),
            message,
        };
        match ready_rx.recv_timeout(OPEN_TIMEOUT) {
            Ok(Ok(rate)) => Ok(Capture {
                shared,
                rate,
                stop_tx: Some(stop_tx),
                thread: Some(thread),
            }),
            Ok(Err(message)) => {
                let _ = thread.join();
                Err(open_err(message))
            }
            Err(RecvTimeoutError::Timeout) => {
                // The thread exits by itself as soon as the stop sender is dropped.
                drop(stop_tx);
                Err(open_err("timed out opening the device".into()))
            }
            Err(RecvTimeoutError::Disconnected) => {
                let _ = thread.join();
                Err(open_err("audio thread ended unexpectedly".into()))
            }
        }
    }
}

impl Capture {
    /// Open `target` and start capturing. `label` is used in error messages.
    pub(crate) fn open(target: Target, label: &str, keep: bool) -> Result<Self, AudioError> {
        Self::spawn(target, label, keep)?.wait()
    }

    /// Start opening on a new thread without waiting; call [`Pending::wait`] for the result. Lets several
    /// devices open in parallel so they start capturing at nearly the same moment.
    pub(crate) fn spawn(target: Target, label: &str, keep: bool) -> Result<Pending, AudioError> {
        let shared = Arc::new(Shared::default());
        shared.keep.store(keep, Ordering::Relaxed);
        let (ready_tx, ready_rx) = mpsc::channel::<Result<u32, String>>();
        let (stop_tx, stop_rx) = mpsc::channel::<()>();
        let thread_shared = Arc::clone(&shared);
        let thread = thread::Builder::new()
            .name("stt-audio-capture".into())
            .spawn(move || {
                let stream = match build_stream(&target, &thread_shared) {
                    Ok((stream, rate)) => {
                        let _ = ready_tx.send(Ok(rate));
                        stream
                    }
                    Err(e) => {
                        let _ = ready_tx.send(Err(e));
                        return;
                    }
                };
                // Blocks until stop is requested or the Capture (and its sender) is dropped.
                let _ = stop_rx.recv();
                let _ = stream.pause();
                drop(stream);
            })
            .map_err(|e| AudioError::Backend(format!("could not start audio thread: {e}")))?;

        Ok(Pending {
            shared,
            ready_rx,
            stop_tx,
            thread,
            label: label.to_string(),
        })
    }

    /// Live level (linear RMS of the latest block).
    pub(crate) fn level(&self) -> f32 {
        f32::from_bits(self.shared.level.load(Ordering::Relaxed))
    }

    /// Begin keeping samples: drops anything buffered so the recording holds only audio after this call.
    pub(crate) fn begin_keep(&self) {
        let mut samples = lock(&self.shared.samples);
        samples.clear();
        self.shared.keep.store(true, Ordering::Relaxed);
    }

    /// Stop keeping samples and return what was kept as mono 16 kHz; the stream keeps running.
    pub(crate) fn take(&self) -> Result<Vec<f32>, AudioError> {
        let samples = {
            let mut guard = lock(&self.shared.samples);
            self.shared.keep.store(false, Ordering::Relaxed);
            std::mem::take(&mut *guard)
        };
        if samples.is_empty() {
            if let Some(err) = lock(&self.shared.error).take() {
                return Err(AudioError::Stream(err));
            }
            return Ok(Vec::new());
        }
        Ok(to_mono_16k(&samples, 1, self.rate))
    }

    /// Stop the stream, close the device and return everything captured as mono 16 kHz.
    pub(crate) fn finish(mut self) -> Result<Vec<f32>, AudioError> {
        self.shutdown();
        let samples = std::mem::take(&mut *lock(&self.shared.samples));
        if samples.is_empty() {
            if let Some(err) = lock(&self.shared.error).take() {
                return Err(AudioError::Stream(err));
            }
            return Ok(Vec::new());
        }
        Ok(to_mono_16k(&samples, 1, self.rate))
    }

    fn shutdown(&mut self) {
        if let Some(tx) = self.stop_tx.take() {
            let _ = tx.send(());
        }
        if let Some(t) = self.thread.take() {
            let _ = t.join();
        }
        self.shared.level.store(0, Ordering::Relaxed);
    }
}

impl Drop for Capture {
    fn drop(&mut self) {
        self.shutdown();
    }
}

fn lock<T>(m: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    m.lock().unwrap_or_else(std::sync::PoisonError::into_inner)
}

/// Handle one callback block of mono samples: update the level, and buffer the samples only while keeping.
fn ingest(shared: &Shared, mono: impl Iterator<Item = f32>) {
    let mut sum_sq = 0.0f32;
    let mut count = 0usize;
    let mut samples = shared
        .keep
        .load(Ordering::Relaxed)
        .then(|| lock(&shared.samples));
    for m in mono {
        sum_sq += m * m;
        count += 1;
        if let Some(buf) = samples.as_mut() {
            buf.push(m);
        }
    }
    drop(samples);
    let level = if count == 0 {
        0.0
    } else {
        (sum_sq / count as f32).sqrt().min(1.0)
    };
    shared.level.store(level.to_bits(), Ordering::Relaxed);
}

fn resolve_device(target: &Target) -> Result<cpal::Device, String> {
    match target {
        Target::Id(id) => {
            let did: cpal::DeviceId = id.parse().map_err(|e| format!("{e}"))?;
            let host = cpal::host_from_id(did.host()).map_err(|e| format!("{e}"))?;
            host.device_by_id(&did)
                .ok_or_else(|| format!("device {id} is no longer available"))
        }
        Target::Default => {
            for id in preferred_hosts() {
                if let Ok(host) = cpal::host_from_id(id)
                    && let Some(d) = host.default_input_device()
                {
                    return Ok(d);
                }
            }
            Err("no default input device".into())
        }
    }
}

fn build_stream(target: &Target, shared: &Arc<Shared>) -> Result<(cpal::Stream, u32), String> {
    let device = resolve_device(target)?;
    let default = device
        .default_input_config()
        .map_err(|e| format!("no usable input config: {e}"))?;
    let channels = default.channels().clamp(1, MAX_CHANNELS);
    let rate = default.sample_rate();
    let config = cpal::StreamConfig {
        channels,
        sample_rate: rate,
        buffer_size: cpal::BufferSize::Default,
    };
    let stream = match default.sample_format() {
        cpal::SampleFormat::F32 => typed::<f32>(&device, &config, shared),
        cpal::SampleFormat::I16 => typed::<i16>(&device, &config, shared),
        cpal::SampleFormat::U16 => typed::<u16>(&device, &config, shared),
        cpal::SampleFormat::I32 => typed::<i32>(&device, &config, shared),
        cpal::SampleFormat::U32 => typed::<u32>(&device, &config, shared),
        cpal::SampleFormat::F64 => typed::<f64>(&device, &config, shared),
        cpal::SampleFormat::I8 => typed::<i8>(&device, &config, shared),
        cpal::SampleFormat::U8 => typed::<u8>(&device, &config, shared),
        other => return Err(format!("unsupported sample format {other:?}")),
    }?;
    stream.play().map_err(|e| format!("could not start: {e}"))?;
    Ok((stream, rate))
}

fn typed<T>(
    device: &cpal::Device,
    config: &cpal::StreamConfig,
    shared: &Arc<Shared>,
) -> Result<cpal::Stream, String>
where
    T: SizedSample + Send + 'static,
    f32: FromSample<T>,
{
    let channels = usize::from(config.channels);
    let data_shared = Arc::clone(shared);
    let err_shared = Arc::clone(shared);
    device
        .build_input_stream(
            *config,
            move |data: &[T], _| {
                ingest(
                    &data_shared,
                    data.chunks_exact(channels).map(|frame| {
                        let sum: f32 = frame
                            .iter()
                            .map(|&s| <f32 as FromSample<T>>::from_sample_(s))
                            .sum();
                        sum / channels as f32
                    }),
                );
            },
            move |e| {
                let mut slot = lock(&err_shared.error);
                if slot.is_none() {
                    *slot = Some(e.to_string());
                }
            },
            None,
        )
        .map_err(|e| format!("{e}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fake_capture() -> Capture {
        Capture {
            shared: Arc::new(Shared::default()),
            rate: 16_000,
            stop_tx: None,
            thread: None,
        }
    }

    #[test]
    fn listening_discards_samples_but_reports_level() {
        let c = fake_capture();
        ingest(&c.shared, std::iter::repeat_n(0.5, 1600));
        assert!(lock(&c.shared.samples).is_empty());
        assert!((c.level() - 0.5).abs() < 1e-6);
    }

    #[test]
    fn recording_holds_only_audio_after_begin_keep() {
        let c = fake_capture();
        ingest(&c.shared, std::iter::repeat_n(0.1, 800));
        c.begin_keep();
        ingest(&c.shared, std::iter::repeat_n(0.2, 16_000));
        let out = c.take().unwrap();
        assert_eq!(out.len(), 16_000);
        assert!(out.iter().all(|&x| (x - 0.2).abs() < 1e-6));
        // After take the capture is listening again: nothing accumulates.
        ingest(&c.shared, std::iter::repeat_n(0.3, 800));
        assert!(lock(&c.shared.samples).is_empty());
    }

    #[test]
    fn begin_keep_clears_stale_buffer() {
        let c = fake_capture();
        c.shared.keep.store(true, Ordering::Relaxed);
        ingest(&c.shared, std::iter::repeat_n(0.9, 500));
        c.begin_keep();
        assert!(lock(&c.shared.samples).is_empty());
    }
}
