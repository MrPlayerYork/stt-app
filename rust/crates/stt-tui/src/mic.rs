//! The hot mic (decided 2026-11-02): the chosen microphone is opened once in the background and kept
//! open ("listening": live level, nothing kept), so pressing record starts instantly. The header must
//! always show [`MicStatus`], and recording must look clearly different from listening.

use std::sync::Arc;
use std::time::Instant;

use stt_core::{AudioError, AudioSource, Recorder, find_device};

use crate::task::Task;

/// Mic state for the header and the record flow.
#[derive(Debug, Clone, PartialEq)]
pub enum MicStatus {
    /// Not open (before startup, during calibration, after an error was cleared).
    Closed,
    /// Opening in the background (1-2 s on PulseAudio).
    Opening,
    /// Open and showing levels, keeping nothing.
    Listening,
    /// Keeping audio since `since`.
    Recording {
        /// When recording started.
        since: Instant,
    },
    /// Could not open the device.
    Error(String),
}

type Opened = Result<Box<dyn Recorder>, AudioError>;

/// Owns the hot recorder.
pub struct Mic {
    status: MicStatus,
    rec: Option<Box<dyn Recorder>>,
    opening: Option<Task<Opened>>,
    /// Start recording as soon as the device is open (record pressed while opening).
    start_when_open: bool,
    /// Name shown in the header: the configured name, or "(system default)".
    label: String,
    /// Why the configured mic was not used, e.g. "'USB Mic' not found; using the default device".
    note: Option<String>,
    /// Software gain (`audio.gain`) applied to the level and the recorded samples.
    gain: f32,
}

impl Default for Mic {
    fn default() -> Self {
        Self {
            status: MicStatus::Closed,
            rec: None,
            opening: None,
            start_when_open: false,
            label: "(system default)".into(),
            note: None,
            gain: 1.0,
        }
    }
}

impl Mic {
    /// Close any open stream and open `device_name` (`None` = system default) in the background. A name
    /// that matches no device falls back to the default device and sets [`Self::note`].
    pub fn open(&mut self, source: Arc<dyn AudioSource>, device_name: Option<&str>) {
        self.close();
        self.note = None;
        self.label = device_name.unwrap_or("(system default)").to_string();
        let mut id = None;
        if let Some(name) = device_name {
            match source.list_input_devices() {
                Ok(devs) => match find_device(&devs, name) {
                    Some(d) => {
                        self.label = d.name.clone();
                        id = Some(d.id.clone());
                    }
                    None => {
                        self.note = Some(format!(
                            "Microphone '{name}' not found; using the default device"
                        ));
                    }
                },
                Err(e) => self.note = Some(format!("Could not list devices: {e}")),
            }
        }
        self.status = MicStatus::Opening;
        self.opening = Some(Task::spawn(move || {
            let mut rec = source.recorder(id.as_ref())?;
            rec.open()?;
            Ok(rec)
        }));
    }

    /// Pick up a finished open; start a pending recording (call every tick).
    pub fn tick(&mut self) {
        let Some(task) = &mut self.opening else {
            return;
        };
        let Some(res) = task.poll() else {
            return;
        };
        self.opening = None;
        match res {
            Ok(rec) => {
                self.rec = Some(rec);
                self.status = MicStatus::Listening;
                if std::mem::take(&mut self.start_when_open)
                    && let Err(e) = self.start()
                {
                    self.status = MicStatus::Error(e.to_string());
                }
            }
            Err(e) => {
                self.start_when_open = false;
                self.status = MicStatus::Error(e.to_string());
            }
        }
    }

    /// Start keeping audio. While still opening, recording starts as soon as the device is open (the
    /// status stays `Opening` until then). Errors if the mic failed to open or is closed.
    pub fn start(&mut self) -> Result<(), AudioError> {
        if self.opening.is_some() {
            self.start_when_open = true;
            return Ok(());
        }
        let rec = self.rec.as_mut().ok_or_else(|| AudioError::Open {
            device: self.label.clone(),
            message: match &self.status {
                MicStatus::Error(e) => e.clone(),
                _ => "microphone is not open".into(),
            },
        })?;
        rec.start()?;
        self.status = MicStatus::Recording {
            since: Instant::now(),
        };
        Ok(())
    }

    /// Stop keeping audio and return it; the stream stays open (listening).
    pub fn stop(&mut self) -> Result<Vec<f32>, AudioError> {
        self.start_when_open = false;
        let Some(rec) = self.rec.as_mut() else {
            return Ok(Vec::new());
        };
        let audio = rec.stop();
        if matches!(self.status, MicStatus::Recording { .. }) {
            self.status = MicStatus::Listening;
        }
        let g = self.gain;
        audio.map(|mut a| {
            if g != 1.0 {
                a.iter_mut().for_each(|s| *s = (*s * g).clamp(-1.0, 1.0));
            }
            a
        })
    }

    /// Set the software gain (`audio.gain`, clamped to the allowed range). Takes effect at once.
    pub fn set_gain(&mut self, gain: f32) {
        self.gain = if gain.is_finite() {
            gain.clamp(stt_core::GAIN_MIN, stt_core::GAIN_MAX)
        } else {
            1.0
        };
    }

    /// The software gain in use.
    pub fn gain(&self) -> f32 {
        self.gain
    }

    /// Stop and throw the audio away (Ctrl+C).
    pub fn cancel(&mut self) {
        let _ = self.stop();
    }

    /// Close the stream (calibration needs the devices; quit).
    pub fn close(&mut self) {
        if let Some(mut rec) = self.rec.take() {
            rec.close();
        }
        // A still-opening task finishes on its own; dropping it drops its recorder when done.
        self.opening = None;
        self.start_when_open = false;
        self.status = MicStatus::Closed;
    }

    /// Current status.
    pub fn status(&self) -> &MicStatus {
        &self.status
    }

    /// Live level `0.0..=1.0` (0 when closed).
    pub fn level(&self) -> f32 {
        self.rec
            .as_ref()
            .map_or(0.0, |r| (r.level() * self.gain).min(1.0))
    }

    /// True while recording (or about to, once the device opens).
    pub fn is_recording(&self) -> bool {
        matches!(self.status, MicStatus::Recording { .. }) || self.start_when_open
    }

    /// Device name for the header.
    pub fn label(&self) -> &str {
        &self.label
    }

    /// Why the configured device was not used, if it wasn't.
    pub fn note(&self) -> Option<&str> {
        self.note.as_deref()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use stt_core::fakes::FakeAudioSource;

    fn settle(m: &mut Mic) {
        for _ in 0..200 {
            m.tick();
            if !matches!(m.status(), MicStatus::Opening) {
                return;
            }
            std::thread::sleep(std::time::Duration::from_millis(5));
        }
    }

    #[test]
    fn opens_listens_records_and_keeps_listening() {
        let src = Arc::new(
            FakeAudioSource::new()
                .device("Mic A", false, vec![0.5; 160])
                .level(0.3),
        );
        let mut m = Mic::default();
        m.open(src, Some("mic a"));
        settle(&mut m);
        assert_eq!(m.status(), &MicStatus::Listening);
        assert_eq!(m.label(), "Mic A");
        assert!(m.level() > 0.0, "hot mic shows levels while listening");
        m.start().unwrap();
        assert!(m.is_recording());
        assert_eq!(m.stop().unwrap().len(), 160);
        assert_eq!(m.status(), &MicStatus::Listening);
    }

    #[test]
    fn gain_scales_level_and_samples() {
        let src = Arc::new(
            FakeAudioSource::new()
                .device("Mic A", false, vec![0.8, -0.2])
                .level(0.4),
        );
        let mut m = Mic::default();
        m.set_gain(0.5);
        m.open(src, None);
        settle(&mut m);
        assert!((m.level() - 0.2).abs() < 1e-6);
        m.start().unwrap();
        assert_eq!(m.stop().unwrap(), vec![0.4, -0.1]);
        m.set_gain(100.0);
        assert_eq!(m.gain(), stt_core::GAIN_MAX);
    }

    #[test]
    fn record_pressed_while_opening_starts_when_open() {
        let src = Arc::new(FakeAudioSource::new().device("Mic A", false, vec![0.1; 10]));
        let mut m = Mic::default();
        m.open(src, None);
        m.start().unwrap();
        assert!(m.is_recording());
        settle(&mut m);
        assert!(matches!(m.status(), MicStatus::Recording { .. }));
    }

    #[test]
    fn unknown_name_falls_back_with_note() {
        let src = Arc::new(FakeAudioSource::new().device("Mic A", false, vec![]));
        let mut m = Mic::default();
        m.open(src, Some("Nope"));
        settle(&mut m);
        assert!(m.note().unwrap().contains("not found"));
        assert_eq!(m.status(), &MicStatus::Listening);
    }
}
