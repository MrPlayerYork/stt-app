//! Settings › Microphone (A7-style rows).

use crate::app::App;
use crate::mic::MicStatus;
use crate::screen::Nav;
use crate::widgets::PickList;

use super::commit;
use super::section::{Row, Section};

pub(super) struct MicSection;

fn state(app: &App) -> String {
    match app.mic.status() {
        MicStatus::Closed => "not open".into(),
        MicStatus::Opening => "opening…".into(),
        MicStatus::Listening => "listening".into(),
        MicStatus::Recording { .. } => "recording".into(),
        MicStatus::Error(e) => format!("error: {e}"),
    }
}

impl Section for MicSection {
    fn title(&self) -> String {
        "Microphone".into()
    }

    fn rows(&self, app: &App) -> Vec<Row> {
        let name = app
            .cfg
            .audio
            .device_name
            .clone()
            .unwrap_or_else(|| "(system default)".into());
        vec![
            Row::new('1', "Current device", format!("{name} · {}", state(app))),
            Row::new('2', "Run calibration", ""),
            Row::new('3', "Pick from a list", ""),
        ]
    }

    fn help(&self, app: &App, i: usize) -> String {
        match i {
            0 => match app.mic.note() {
                Some(n) => n.to_string(),
                None => "The microphone in use. The stream stays open while the app runs, so recording starts instantly.".into(),
            },
            1 => "Records a short phrase on every input device and picks the one that hears you best.".into(),
            _ => "Choose the microphone yourself from the list of input devices.".into(),
        }
    }

    fn activate(&mut self, app: &mut App, i: usize) -> Nav {
        match i {
            0 => Nav::Stay,
            1 => Nav::Push(Box::new(crate::calibrate::CalibrationScreen::new(app))),
            _ => pick_device(app),
        }
    }

    fn enter_hint(&self) -> &'static str {
        "open"
    }
}

fn pick_device(app: &mut App) -> Nav {
    let devs = match app.services.audio.list_input_devices() {
        Ok(d) => d,
        Err(e) => {
            app.flash_error(format!("Could not list devices: {e}"));
            return Nav::Stay;
        }
    };
    if devs.is_empty() {
        app.flash_error("No input devices found.");
        return Nav::Stay;
    }
    let labels: Vec<String> = devs
        .iter()
        .map(|d| {
            if d.is_monitor {
                format!("{} (monitor)", d.name)
            } else {
                d.name.clone()
            }
        })
        .collect();
    let current = app
        .cfg
        .audio
        .device_name
        .as_ref()
        .and_then(|n| devs.iter().position(|d| &d.name == n));
    let names: Vec<String> = devs.into_iter().map(|d| d.name).collect();
    Nav::Push(Box::new(PickList::new(
        "Device",
        labels,
        current,
        move |app, i| {
            app.cfg.audio.device_name = Some(names[i].clone());
            commit(app);
            app.open_mic();
            app.flash_ok(format!("Microphone: {}", names[i]));
            Nav::Stay
        },
    )))
}
