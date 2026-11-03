//! [`CpalAudioSource`]: device listing, single recorder and multi-device recorder over cpal.

use std::collections::BTreeMap;

use cpal::HostId;
use cpal::traits::{DeviceTrait, HostTrait};
use stt_core::{
    AudioError, AudioSource, DeviceId, InputDevice, MultiRecorder, Recorder, SAMPLE_RATE,
};

use crate::capture::{Capture, Target};
use crate::convert::peak;
use crate::matching::{alsa_rank, dedupe_by, is_monitor_name};

/// Available hosts, best first. PulseAudio (served by pipewire-pulse) gives one friendly-named entry per
/// real PipeWire source and opens reliably, the same role JACK played for the Python app. Raw ALSA is the
/// fallback.
pub(crate) fn preferred_hosts() -> Vec<HostId> {
    let mut hosts = cpal::available_hosts();
    hosts.sort_by_key(|h| match h.to_string().as_str() {
        "pipewire" => 0,
        "pulseaudio" => 1,
        "jack" => 2,
        "alsa" => 3,
        _ => 4,
    });
    hosts
}

/// The real, cpal-backed microphone source.
#[derive(Debug, Default, Clone, Copy)]
pub struct CpalAudioSource;

impl CpalAudioSource {
    /// Create the source (stateless; devices are enumerated on demand).
    pub fn new() -> Self {
        Self
    }
}

fn list_host(id: HostId) -> Vec<InputDevice> {
    let Ok(host) = cpal::host_from_id(id) else {
        return Vec::new();
    };
    let Ok(devices) = host.input_devices() else {
        return Vec::new();
    };
    let is_alsa = id.to_string() == "alsa";
    let mut found: Vec<(u8, InputDevice)> = Vec::new();
    for d in devices {
        let (Ok(desc), Ok(cid)) = (d.description(), d.id()) else {
            continue;
        };
        let cfg = d.default_input_config().ok();
        let rank = if is_alsa {
            // Skip busy raw nodes, null and generic aliases; keep one entry per card name.
            match (cfg.as_ref(), alsa_rank(desc.driver().unwrap_or(""))) {
                (Some(_), Some(r)) => r,
                _ => continue,
            }
        } else {
            0
        };
        let name = desc.name().to_string();
        found.push((
            rank,
            InputDevice {
                id: DeviceId(cid.to_string()),
                is_monitor: is_monitor_name(&name) || cid.id().ends_with(".monitor"),
                name,
                host: id.to_string(),
                channels: cfg
                    .as_ref()
                    .map_or(1, cpal::SupportedStreamConfig::channels),
                default_sample_rate: cfg.as_ref().map_or(48_000, |c| c.sample_rate()),
            },
        ));
    }
    found.sort_by_key(|(r, _)| *r);
    dedupe_by(found, |(_, d)| d.name.clone())
        .into_iter()
        .map(|(_, d)| d)
        .collect()
}

impl AudioSource for CpalAudioSource {
    fn list_input_devices(&self) -> Result<Vec<InputDevice>, AudioError> {
        // Like Python's "JACK list or everything": the first host that yields devices wins.
        for id in preferred_hosts() {
            let devs = list_host(id);
            if !devs.is_empty() {
                return Ok(devs);
            }
        }
        Err(AudioError::NoDevices)
    }

    fn recorder(&self, device: Option<&DeviceId>) -> Result<Box<dyn Recorder>, AudioError> {
        let (target, label) = match device {
            None => (Target::Default, "default input".to_string()),
            Some(id) => {
                if id.0.parse::<cpal::DeviceId>().is_err() {
                    return Err(AudioError::DeviceNotFound(id.0.clone()));
                }
                (Target::Id(id.0.clone()), id.0.clone())
            }
        };
        Ok(Box::new(CpalRecorder {
            target,
            label,
            capture: None,
            hot: false,
            recording: false,
        }))
    }

    fn multi_recorder(&self, devices: &[InputDevice]) -> Box<dyn MultiRecorder> {
        Box::new(CpalMultiRecorder {
            devices: devices.to_vec(),
            running: BTreeMap::new(),
        })
    }
}

struct CpalRecorder {
    target: Target,
    label: String,
    capture: Option<Capture>,
    /// Stream was opened with `open` (hot mic): it stays open after `stop`.
    hot: bool,
    recording: bool,
}

impl Recorder for CpalRecorder {
    fn open(&mut self) -> Result<(), AudioError> {
        if self.capture.is_none() {
            self.capture = Some(Capture::open(self.target.clone(), &self.label, false)?);
        }
        self.hot = true;
        Ok(())
    }

    fn close(&mut self) {
        self.capture = None;
        self.hot = false;
        self.recording = false;
    }

    fn is_open(&self) -> bool {
        self.capture.is_some()
    }

    fn start(&mut self) -> Result<(), AudioError> {
        if self.recording {
            return Ok(());
        }
        match &self.capture {
            Some(c) => c.begin_keep(),
            None => {
                self.capture = Some(Capture::open(self.target.clone(), &self.label, true)?);
            }
        }
        self.recording = true;
        Ok(())
    }

    fn stop(&mut self) -> Result<Vec<f32>, AudioError> {
        if !self.recording {
            return Ok(Vec::new());
        }
        self.recording = false;
        if self.hot {
            self.capture.as_ref().map_or(Ok(Vec::new()), Capture::take)
        } else {
            self.capture.take().map_or(Ok(Vec::new()), Capture::finish)
        }
    }

    fn level(&self) -> f32 {
        self.capture.as_ref().map_or(0.0, Capture::level)
    }

    fn is_recording(&self) -> bool {
        self.recording
    }
}

struct CpalMultiRecorder {
    devices: Vec<InputDevice>,
    running: BTreeMap<DeviceId, Capture>,
}

impl MultiRecorder for CpalMultiRecorder {
    fn start(&mut self) -> BTreeMap<DeviceId, String> {
        let mut errors = BTreeMap::new();
        // Open all devices in parallel so they start together (opening one can take a second or two).
        let mut pending = Vec::new();
        for d in &self.devices {
            if self.running.contains_key(&d.id) {
                continue;
            }
            match Capture::spawn(Target::Id(d.id.0.clone()), &d.name, true) {
                Ok(p) => pending.push((d.id.clone(), p)),
                Err(e) => {
                    errors.insert(d.id.clone(), e.to_string());
                }
            }
        }
        for (id, p) in pending {
            match p.wait() {
                Ok(c) => {
                    self.running.insert(id, c);
                }
                Err(e) => {
                    errors.insert(id, e.to_string());
                }
            }
        }
        errors
    }

    fn stop(&mut self) -> BTreeMap<DeviceId, Vec<f32>> {
        std::mem::take(&mut self.running)
            .into_iter()
            .map(|(id, c)| (id, c.finish().unwrap_or_default()))
            .collect()
    }

    fn levels(&self) -> BTreeMap<DeviceId, f32> {
        self.running
            .iter()
            .map(|(id, c)| (id.clone(), c.level()))
            .collect()
    }
}

/// Seconds of audio in a 16 kHz buffer (helper for smoke tests and logging).
pub fn duration_secs(samples: &[f32]) -> f32 {
    samples.len() as f32 / SAMPLE_RATE as f32
}

/// Peak and RMS of a buffer, for diagnostics that must never expose the audio itself.
pub fn peak_rms(samples: &[f32]) -> (f32, f32) {
    (peak(samples), crate::convert::rms(samples))
}
