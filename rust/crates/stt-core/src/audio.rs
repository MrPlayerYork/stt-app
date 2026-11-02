//! Microphone contracts. The real implementation is `stt-audio` (cpal); tests use
//! `fakes::FakeAudioSource`.
//!
//! All captured audio leaves a recorder as mono `f32` at [`crate::SAMPLE_RATE`] (16 kHz), already downmixed
//! and resampled (Python `to_mono_16k`).
//!
//! Threading note for implementers: `cpal::Stream` is `!Send`, but [`Recorder`] and [`MultiRecorder`] must
//! be `Send` (the UI may move them to a worker thread). Keep each stream on a dedicated thread that owns
//! it, and talk to it via channels / shared atomics.

use std::collections::BTreeMap;

use crate::error::AudioError;

/// Stable identifier of an input device within one [`AudioSource`]. For cpal this is the device name
/// (plus host if names collide); it is only meaningful to the source that produced it.
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct DeviceId(pub String);

impl std::fmt::Display for DeviceId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

/// One input device as listed for the user.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct InputDevice {
    /// Identifier to open it with.
    pub id: DeviceId,
    /// Friendly name; this is what `audio.device_name` stores.
    pub name: String,
    /// Host/API name, e.g. "ALSA", "JACK" (informational).
    pub host: String,
    /// Maximum input channels (recording uses at most 2, then downmixes).
    pub channels: u16,
    /// Native sample rate the device is opened at, in Hz.
    pub default_sample_rate: u32,
    /// True when this is an output monitor ("Monitor of ..."), not a real microphone. Calibration still
    /// records monitors but marks a silent one as "likely an output monitor".
    pub is_monitor: bool,
}

/// Python `find_device`: exact name match first, then the first case-insensitive substring match.
pub fn find_device<'a>(devices: &'a [InputDevice], name: &str) -> Option<&'a InputDevice> {
    if let Some(d) = devices.iter().find(|d| d.name == name) {
        return Some(d);
    }
    let low = name.to_lowercase();
    devices
        .iter()
        .find(|d| d.name.to_lowercase().contains(&low))
}

/// Lists and opens input devices.
pub trait AudioSource: Send + Sync {
    /// Input devices, deduplicated: one entry per real source (no raw `hw:*` duplicates of the same
    /// PipeWire node), monitors flagged with [`InputDevice::is_monitor`].
    fn list_input_devices(&self) -> Result<Vec<InputDevice>, AudioError>;

    /// A recorder for `device` (`None` = the system default input). Does not start capturing.
    fn recorder(&self, device: Option<&DeviceId>) -> Result<Box<dyn Recorder>, AudioError>;

    /// A recorder that captures all `devices` at once (calibration). Does not start capturing; per-device
    /// open failures are reported by [`MultiRecorder::start`].
    fn multi_recorder(&self, devices: &[InputDevice]) -> Box<dyn MultiRecorder>;
}

/// Records one device (push-to-record).
pub trait Recorder: Send {
    /// Start capturing. Calling it while already recording is a no-op.
    fn start(&mut self) -> Result<(), AudioError>;

    /// Stop and return everything captured since `start`, as mono 16 kHz `f32`. Returns an empty buffer if
    /// nothing was captured or it was not recording. To cancel (Ctrl+C), call this and drop the result.
    fn stop(&mut self) -> Result<Vec<f32>, AudioError>;

    /// Live input level for the meter: linear RMS of the most recent block, `0.0..=1.0`; 0 when idle.
    /// Cheap and non-blocking (called on every UI redraw).
    fn level(&self) -> f32;

    /// True between `start` and `stop`.
    fn is_recording(&self) -> bool;
}

/// Records several devices simultaneously (Python `MultiRecorder`).
pub trait MultiRecorder: Send {
    /// Open and start every device not already running. Returns the devices that failed to open, with an
    /// error message each; the rest keep recording.
    fn start(&mut self) -> BTreeMap<DeviceId, String>;

    /// Stop all streams and return each device's mono 16 kHz `f32` audio.
    fn stop(&mut self) -> BTreeMap<DeviceId, Vec<f32>>;

    /// Live level (linear RMS, `0.0..=1.0`) of each running device.
    fn levels(&self) -> BTreeMap<DeviceId, f32>;
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dev(name: &str) -> InputDevice {
        InputDevice {
            id: DeviceId(name.into()),
            name: name.into(),
            host: "test".into(),
            channels: 1,
            default_sample_rate: 48_000,
            is_monitor: false,
        }
    }

    #[test]
    fn find_device_exact_then_substring() {
        let devs = [dev("USB Mic Pro"), dev("USB Mic"), dev("Webcam")];
        assert_eq!(find_device(&devs, "USB Mic").unwrap().name, "USB Mic");
        assert_eq!(find_device(&devs, "webcam").unwrap().name, "Webcam");
        assert_eq!(find_device(&devs, "usb").unwrap().name, "USB Mic Pro");
        assert!(find_device(&devs, "nope").is_none());
    }
}
