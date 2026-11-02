//! Matching a saved `audio.device_name` (written by the Python app from sounddevice / PortAudio) to the
//! devices cpal lists now, plus the monitor and dedupe helpers used by the listing.

use std::collections::HashSet;

use stt_core::InputDevice;

/// PortAudio host-API suffixes that can end a displayed name (`"<name>, ALSA"`).
const HOST_SUFFIXES: [&str; 6] = [
    ", jack audio connection kit",
    ", alsa",
    ", pulseaudio",
    ", pipewire",
    ", oss",
    ", jack",
];

/// True for output monitors ("Monitor of ...", PulseAudio `*.monitor`).
pub fn is_monitor_name(name: &str) -> bool {
    let low = name.to_lowercase();
    low.starts_with("monitor of ") || low.ends_with(".monitor") || low.ends_with(" monitor")
}

/// Normalise a device name for comparison: lowercase, drop a PortAudio host suffix and `(hw:X,Y)`
/// markers, turn punctuation into spaces and collapse whitespace.
pub fn normalize_name(name: &str) -> String {
    let mut s = name.trim().to_lowercase();
    for suf in HOST_SUFFIXES {
        if let Some(stripped) = s.strip_suffix(suf) {
            s = stripped.trim_end().to_string();
            break;
        }
    }
    // Remove "(hw:0,1)" / "(plughw:2,0)" style markers anywhere.
    while let Some(start) = s.find("(hw:").or_else(|| s.find("(plughw:")) {
        match s[start..].find(')') {
            Some(end) => s.replace_range(start..start + end + 1, " "),
            None => break,
        }
    }
    let cleaned: String = s
        .chars()
        .map(|c| if c.is_alphanumeric() { c } else { ' ' })
        .collect();
    cleaned.split_whitespace().collect::<Vec<_>>().join(" ")
}

/// Find the device a saved name refers to. Order: exact name, then equal after [`normalize_name`], then
/// normalised substring either way (preferring the closest length and real mics over monitors). Returns
/// `None` when nothing plausibly matches, so the app re-runs calibration.
pub fn match_saved_device<'a>(
    devices: &'a [InputDevice],
    saved_name: &str,
) -> Option<&'a InputDevice> {
    if saved_name.trim().is_empty() {
        return None;
    }
    if let Some(d) = devices.iter().find(|d| d.name == saved_name) {
        return Some(d);
    }
    let want = normalize_name(saved_name);
    if want.is_empty() {
        return None;
    }
    if let Some(d) = devices.iter().find(|d| normalize_name(&d.name) == want) {
        return Some(d);
    }
    if want.len() < 4 {
        return None; // too short to substring-match safely
    }
    let wants_monitor = is_monitor_name(saved_name);
    devices
        .iter()
        .filter(|d| wants_monitor || !d.is_monitor)
        .filter_map(|d| {
            let have = normalize_name(&d.name);
            (have.contains(&want) || want.contains(&have) && have.len() >= 4)
                .then(|| (have.len().abs_diff(want.len()), d))
        })
        .min_by_key(|(diff, _)| *diff)
        .map(|(_, d)| d)
}

/// Keep the first device per key, preserving order (dedupe helper for the listing).
pub fn dedupe_by<T, K: std::hash::Hash + Eq>(items: Vec<T>, key: impl Fn(&T) -> K) -> Vec<T> {
    let mut seen = HashSet::new();
    items.into_iter().filter(|i| seen.insert(key(i))).collect()
}

/// ALSA fallback only: lower is better. Raw `hw:`/`front:` nodes and the generic aliases are the
/// duplicates Python avoided by using the JACK list; `plughw:` converts formats so it opens reliably.
pub fn alsa_rank(driver: &str) -> Option<u8> {
    let kind = driver.split(':').next().unwrap_or(driver);
    match kind {
        "null" | "default" | "pipewire" | "sysdefault" if !driver.contains("CARD=") => None,
        "plughw" => Some(0),
        "sysdefault" => Some(1),
        "front" => Some(2),
        "hw" => Some(3),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use stt_core::DeviceId;

    fn dev(name: &str) -> InputDevice {
        InputDevice {
            id: DeviceId(format!("pulseaudio:{name}")),
            name: name.into(),
            host: "pulseaudio".into(),
            channels: 2,
            default_sample_rate: 48_000,
            is_monitor: is_monitor_name(name),
        }
    }

    /// The device names cpal's PulseAudio host reports on the author's machine.
    fn real_cpal() -> Vec<InputDevice> {
        [
            "HD Pro Webcam C920 Analog Stereo",
            "Blue Snowball Analog Stereo",
            "Monitor of Astro A50 Game",
            "Monitor of Astro A50 Chat",
            "Astro A50 Chat",
            "Monitor of Starship/Matisse HD Audio Controller Analog Stereo",
            "Starship/Matisse HD Audio Controller Analog Stereo",
            "Monitor of AD104 High Definition Audio Controller Digital Stereo (HDMI) [U32R59x]",
        ]
        .iter()
        .map(|n| dev(n))
        .collect()
    }

    fn name_of(d: Option<&InputDevice>) -> Option<&str> {
        d.map(|d| d.name.as_str())
    }

    #[test]
    fn exact_match_with_real_portaudio_jack_names() {
        let devs = real_cpal();
        for saved in [
            "Blue Snowball Analog Stereo",
            "HD Pro Webcam C920 Analog Stereo",
            "Astro A50 Chat",
            "Starship/Matisse HD Audio Controller Analog Stereo",
        ] {
            assert_eq!(name_of(match_saved_device(&devs, saved)), Some(saved));
        }
    }

    #[test]
    fn strips_portaudio_host_suffix() {
        let devs = real_cpal();
        assert_eq!(
            name_of(match_saved_device(
                &devs,
                "Blue Snowball Analog Stereo, JACK Audio Connection Kit"
            )),
            Some("Blue Snowball Analog Stereo")
        );
    }

    #[test]
    fn case_and_punctuation_insensitive() {
        let devs = real_cpal();
        assert_eq!(
            name_of(match_saved_device(&devs, "blue  snowball analog-stereo")),
            Some("Blue Snowball Analog Stereo")
        );
    }

    #[test]
    fn substring_prefers_real_mic_over_monitor() {
        let devs = real_cpal();
        // "Astro A50 Chat" is both a mic and inside the monitor's name; the mic wins even when partial.
        assert_eq!(
            name_of(match_saved_device(&devs, "Astro A50 Cha")),
            Some("Astro A50 Chat")
        );
        assert_eq!(
            name_of(match_saved_device(&devs, "Snowball")),
            Some("Blue Snowball Analog Stereo")
        );
    }

    #[test]
    fn monitor_only_when_saved_was_a_monitor() {
        let devs = real_cpal();
        assert_eq!(
            name_of(match_saved_device(&devs, "Monitor of Astro A50 Game")),
            Some("Monitor of Astro A50 Game")
        );
        // A saved real mic that is gone must not silently become its output monitor.
        let only_monitor: Vec<_> = devs.into_iter().filter(|d| d.is_monitor).collect();
        assert!(match_saved_device(&only_monitor, "Astro A50 Game").is_none());
    }

    #[test]
    fn alsa_hw_names_from_python_do_not_match_pulse_devices() {
        let devs = real_cpal();
        // Saved from PortAudio's ALSA list (no equivalent friendly PipeWire name): re-calibrate.
        for saved in [
            "Astro A50: USB Audio #1 (hw:0,1)",
            "HD-Audio Generic: ALC1220 Alt Analog (hw:2,2)",
            "pipewire",
            "default",
        ] {
            assert!(match_saved_device(&devs, saved).is_none(), "{saved}");
        }
    }

    #[test]
    fn alsa_hw_marker_is_stripped_when_names_agree() {
        let alsa = vec![dev("Astro A50: USB Audio #1")];
        assert_eq!(
            name_of(match_saved_device(
                &alsa,
                "Astro A50: USB Audio #1 (hw:0,1), ALSA"
            )),
            Some("Astro A50: USB Audio #1")
        );
    }

    #[test]
    fn empty_and_unknown_return_none() {
        let devs = real_cpal();
        assert!(match_saved_device(&devs, "").is_none());
        assert!(match_saved_device(&devs, "  ").is_none());
        assert!(match_saved_device(&devs, "Rode NT-USB").is_none());
        assert!(match_saved_device(&[], "Blue Snowball").is_none());
    }

    #[test]
    fn monitor_detection() {
        assert!(is_monitor_name("Monitor of Astro A50 Game"));
        assert!(is_monitor_name(
            "alsa_output.pci-0000.analog-stereo.monitor"
        ));
        assert!(!is_monitor_name("Blue Snowball Analog Stereo"));
    }

    #[test]
    fn dedupe_keeps_first_per_key() {
        let v = dedupe_by(vec![("a", 1), ("b", 2), ("a", 3)], |x| x.0);
        assert_eq!(v, vec![("a", 1), ("b", 2)]);
    }

    #[test]
    fn alsa_ranking_drops_aliases_and_prefers_plughw() {
        assert_eq!(alsa_rank("null"), None);
        assert_eq!(alsa_rank("pipewire"), None);
        assert_eq!(alsa_rank("default"), None);
        assert_eq!(alsa_rank("sysdefault"), None);
        assert_eq!(alsa_rank("plughw:CARD=0,DEV=1"), Some(0));
        assert_eq!(alsa_rank("sysdefault:CARD=A50"), Some(1));
        assert_eq!(alsa_rank("hw:CARD=0,DEV=1"), Some(3));
    }
}
