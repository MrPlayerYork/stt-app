//! Microphone calibration.
//!
//! Pure scoring ported from `src/stt_app/calibrate.py`: transcript accuracy against the test phrase
//! ([`score_transcript`], a port of `difflib.SequenceMatcher.ratio` including its autojunk heuristic),
//! signal analysis ([`analyze_audio`]), the combined score ([`combine`]), ranking ([`rank`]) and the
//! silent-monitor check. [`evaluate`] is the UI-independent part of the flow: given the recorded audio it
//! classifies each device, loads and runs the transcriber, scores and ranks. The recording itself and the
//! screens live in the UI (`stt-tui`, Phase 3).

use std::cmp::Ordering;
use std::collections::{BTreeMap, HashMap};
use std::hash::Hash;

use stt_core::audio::{DeviceId, InputDevice};
use stt_core::progress::LoadProgressFn;
use stt_core::transcribe::Transcriber;

/// The sentence the user reads aloud (Python `TEST_PHRASE`).
pub const TEST_PHRASE: &str =
    "The quick brown fox jumps over the lazy dog while seven bright purple whales sing quietly.";

/// Level reported for signals with no energy (dB).
pub const FLOOR_DB: f64 = -120.0;
/// Below this speech level a device is treated as silent (dB).
pub const SILENT_DB: f64 = -60.0;
/// Below this level a score is penalised for being too quiet (dB).
pub const QUIET_DB: f64 = -45.0;
/// Analysis frame length in samples: 20 ms at 16 kHz.
pub const FRAME: usize = 320;
/// Score given to devices that failed or were silent, so they always rank last.
pub const ERROR_SCORE: f64 = -1e9;

/// Error message for a device whose level is below [`SILENT_DB`].
pub const SILENT_ERROR: &str = "silent (likely an output monitor)";

/// Signal metrics of one recording (Python `analyze_audio` result).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Signal {
    /// Speech level in dBFS (loudest 20 % of frames), or [`FLOOR_DB`] for silence.
    pub rms_db: f64,
    /// Loudest 20 % of frames against quietest 20 %, in dB, capped at 90.
    pub snr_db: f64,
    /// Fraction of samples at or above 0.99 in magnitude.
    pub clip_frac: f64,
}

/// One device's calibration result (Python `MicScore`).
#[derive(Debug, Clone, PartialEq)]
pub struct MicScore {
    /// The device that was recorded.
    pub device: InputDevice,
    /// What the transcriber heard; empty when not transcribed.
    pub transcript: String,
    /// Word-level similarity to [`TEST_PHRASE`], `0.0..=1.0`.
    pub accuracy: f64,
    /// Speech level in dBFS.
    pub rms_db: f64,
    /// Signal-to-noise ratio in dB.
    pub snr_db: f64,
    /// Fraction of clipped samples.
    pub clip_frac: f64,
    /// Combined score (see [`combine`]), or [`ERROR_SCORE`] when `error` is set.
    pub score: f64,
    /// Why this device could not be scored, if it could not.
    pub error: Option<String>,
}

impl MicScore {
    /// True when the device was recorded and scored (no error).
    pub fn is_ok(&self) -> bool {
        self.error.is_none()
    }

    fn errored(device: InputDevice, signal: Signal, error: String) -> Self {
        Self {
            device,
            transcript: String::new(),
            accuracy: 0.0,
            rms_db: signal.rms_db,
            snr_db: signal.snr_db,
            clip_frac: signal.clip_frac,
            score: ERROR_SCORE,
            error: Some(error),
        }
    }
}

/// Lowercased words with punctuation dropped and apostrophes removed (Python `_words`).
fn words(text: &str) -> Vec<String> {
    let cleaned: String = text
        .to_lowercase()
        .chars()
        .filter_map(|c| {
            if c == '\'' {
                None
            } else if c.is_alphanumeric() || c == '_' || c.is_whitespace() {
                Some(c)
            } else {
                Some(' ')
            }
        })
        .collect();
    cleaned.split_whitespace().map(str::to_owned).collect()
}

/// Word-level accuracy of `got` against `expected`, `0.0..=1.0` (Python `score_transcript`).
pub fn score_transcript(expected: &str, got: &str) -> f64 {
    let a = words(expected);
    let b = words(got);
    if a.is_empty() || b.is_empty() {
        return if a == b { 1.0 } else { 0.0 };
    }
    sequence_ratio(&a, &b)
}

/// `difflib.SequenceMatcher(None, a, b).ratio()`: `2 * matches / (len(a) + len(b))`.
fn sequence_ratio<T: Eq + Hash>(a: &[T], b: &[T]) -> f64 {
    let total = a.len() + b.len();
    if total == 0 {
        return 1.0;
    }
    2.0 * matching_total(a, b) as f64 / total as f64
}

/// Total size of the matching blocks difflib finds between `a` and `b`, including its autojunk rule
/// (for `b` of 200 or more items, elements occurring more than `len(b) / 100 + 1` times are ignored for
/// seeding matches, but still extended into them).
fn matching_total<T: Eq + Hash>(a: &[T], b: &[T]) -> usize {
    let mut b2j: HashMap<&T, Vec<usize>> = HashMap::new();
    for (j, x) in b.iter().enumerate() {
        b2j.entry(x).or_default().push(j);
    }
    if b.len() >= 200 {
        let ntest = b.len() / 100 + 1;
        b2j.retain(|_, idxs| idxs.len() <= ntest);
    }

    // difflib `find_longest_match` with no junk elements.
    let find = |alo: usize, ahi: usize, blo: usize, bhi: usize| -> (usize, usize, usize) {
        let (mut besti, mut bestj, mut bestsize) = (alo, blo, 0);
        let mut j2len: HashMap<usize, usize> = HashMap::new();
        for (i, item) in a.iter().enumerate().take(ahi).skip(alo) {
            let mut newj2len: HashMap<usize, usize> = HashMap::new();
            if let Some(js) = b2j.get(item) {
                for &j in js {
                    if j < blo {
                        continue;
                    }
                    if j >= bhi {
                        break;
                    }
                    let prev = j
                        .checked_sub(1)
                        .and_then(|p| j2len.get(&p))
                        .copied()
                        .unwrap_or(0);
                    let k = prev + 1;
                    newj2len.insert(j, k);
                    if k > bestsize {
                        besti = i + 1 - k;
                        bestj = j + 1 - k;
                        bestsize = k;
                    }
                }
            }
            j2len = newj2len;
        }
        while besti > alo && bestj > blo && a[besti - 1] == b[bestj - 1] {
            besti -= 1;
            bestj -= 1;
            bestsize += 1;
        }
        while besti + bestsize < ahi
            && bestj + bestsize < bhi
            && a[besti + bestsize] == b[bestj + bestsize]
        {
            bestsize += 1;
        }
        (besti, bestj, bestsize)
    };

    let mut total = 0;
    let mut queue = vec![(0, a.len(), 0, b.len())];
    while let Some((alo, ahi, blo, bhi)) = queue.pop() {
        let (i, j, k) = find(alo, ahi, blo, bhi);
        if k > 0 {
            total += k;
            if alo < i && blo < j {
                queue.push((alo, i, blo, j));
            }
            if i + k < ahi && j + k < bhi {
                queue.push((i + k, ahi, j + k, bhi));
            }
        }
    }
    total
}

/// dBFS of a linear amplitude, floored at [`FLOOR_DB`] (Python `_db`).
fn db(x: f64) -> f64 {
    if x > 0.0 {
        (20.0 * x.log10()).max(FLOOR_DB)
    } else {
        FLOOR_DB
    }
}

fn mean(v: &[f64]) -> f64 {
    v.iter().sum::<f64>() / v.len() as f64
}

/// Level, SNR and clipping of one recording (Python `analyze_audio`). Empty or silent input gives
/// `Signal { rms_db: -120, snr_db: 0, clip_frac: 0 }`; the clip fraction is still measured for silence.
pub fn analyze_audio(audio: &[f32]) -> Signal {
    if audio.is_empty() {
        return Signal {
            rms_db: FLOOR_DB,
            snr_db: 0.0,
            clip_frac: 0.0,
        };
    }
    let x: Vec<f64> = audio.iter().map(|&s| f64::from(s)).collect();
    let clip_frac = x.iter().filter(|v| v.abs() >= 0.99).count() as f64 / x.len() as f64;
    let n = x.len() / FRAME;
    if n < 2 {
        let ms = x.iter().map(|v| v * v).sum::<f64>() / x.len() as f64;
        return Signal {
            rms_db: db(ms.sqrt()),
            snr_db: 0.0,
            clip_frac,
        };
    }
    let mut energies: Vec<f64> = x[..n * FRAME]
        .chunks_exact(FRAME)
        .map(|frame| frame.iter().map(|v| v * v).sum::<f64>() / FRAME as f64)
        .collect();
    energies.sort_by(f64::total_cmp);
    let k = (n / 5).max(1);
    let quiet = mean(&energies[..k]);
    let loud = mean(&energies[n - k..]);
    if loud <= 0.0 {
        return Signal {
            rms_db: FLOOR_DB,
            snr_db: 0.0,
            clip_frac,
        };
    }
    let snr = 10.0 * (loud / quiet.max(1e-12)).log10();
    Signal {
        rms_db: db(loud.sqrt()),
        snr_db: snr.min(90.0),
        clip_frac,
    }
}

/// Combine accuracy and signal metrics into one score (Python `combine`). Higher is better.
pub fn combine(accuracy: f64, rms_db: f64, snr_db: f64, clip_frac: f64) -> f64 {
    let mut score = 100.0 * accuracy + 0.5 * snr_db.clamp(0.0, 40.0);
    score -= 200.0 * clip_frac;
    if rms_db < QUIET_DB {
        score -= (QUIET_DB - rms_db.max(FLOOR_DB)) * 0.5;
    }
    score
}

/// Sort best first; devices with an error go last, in their original order (Python `rank`).
pub fn rank(mut scores: Vec<MicScore>) -> Vec<MicScore> {
    scores.sort_by(|a, b| {
        a.error
            .is_some()
            .cmp(&b.error.is_some())
            .then_with(|| b.score.partial_cmp(&a.score).unwrap_or(Ordering::Equal))
    });
    scores
}

/// The device to suggest: the first entry of a ranked list that has no error.
pub fn suggested(ranked: &[MicScore]) -> Option<&MicScore> {
    ranked.iter().find(|s| s.is_ok())
}

/// Score every recorded device (the UI-independent part of Python `_evaluate`).
///
/// `audio` holds each device's recording, `failed` the devices that did not open (from
/// `MultiRecorder::start`). Devices that failed or are silent get an error entry without being
/// transcribed. If any device needs transcribing, the model is loaded once with `on_load`; if the load
/// fails every such device gets a `model load failed` error. `on_transcribe(n, total, name)` is called
/// before each transcription. The result is ranked with [`rank`].
pub fn evaluate(
    transcriber: &dyn Transcriber,
    devices: &[InputDevice],
    audio: &BTreeMap<DeviceId, Vec<f32>>,
    failed: &BTreeMap<DeviceId, String>,
    on_load: Option<LoadProgressFn>,
    on_transcribe: &mut dyn FnMut(usize, usize, &str),
) -> Vec<MicScore> {
    let mut results = Vec::new();
    let mut todo: Vec<(&InputDevice, &[f32], Signal)> = Vec::new();
    for device in devices {
        if let Some(message) = failed.get(&device.id) {
            let signal = Signal {
                rms_db: FLOOR_DB,
                snr_db: 0.0,
                clip_frac: 0.0,
            };
            let error = format!("failed to open: {message}");
            results.push(MicScore::errored(device.clone(), signal, error));
            continue;
        }
        let clip: &[f32] = audio.get(&device.id).map_or(&[], Vec::as_slice);
        let signal = analyze_audio(clip);
        if signal.rms_db < SILENT_DB {
            results.push(MicScore::errored(
                device.clone(),
                signal,
                SILENT_ERROR.to_owned(),
            ));
        } else {
            todo.push((device, clip, signal));
        }
    }

    if !todo.is_empty()
        && let Err(e) = transcriber.load(on_load)
    {
        for (device, _, signal) in todo.drain(..) {
            let error = format!("model load failed: {e}");
            results.push(MicScore::errored(device.clone(), signal, error));
        }
    }

    let total = todo.len();
    for (n, (device, clip, signal)) in todo.into_iter().enumerate() {
        on_transcribe(n + 1, total, &device.name);
        match transcriber.transcribe(clip) {
            Ok(text) => {
                let accuracy = score_transcript(TEST_PHRASE, &text);
                let score = combine(accuracy, signal.rms_db, signal.snr_db, signal.clip_frac);
                results.push(MicScore {
                    device: device.clone(),
                    transcript: text,
                    accuracy,
                    rms_db: signal.rms_db,
                    snr_db: signal.snr_db,
                    clip_frac: signal.clip_frac,
                    score,
                    error: None,
                });
            }
            Err(e) => {
                let error = format!("transcription error: {e}");
                results.push(MicScore::errored(device.clone(), signal, error));
            }
        }
    }
    rank(results)
}

#[cfg(test)]
mod tests {
    use super::*;
    use stt_core::audio::{AudioSource, MultiRecorder};
    use stt_core::fakes::{FakeAudioSource, FakeTranscriber};

    const T: &str = "The quick brown fox jumps over the lazy dog.";

    fn sine(amp: f64, n: usize) -> Vec<f32> {
        (0..n)
            .map(|i| (amp * (2.0 * std::f64::consts::PI * 440.0 * i as f64 / 16000.0).sin()) as f32)
            .collect()
    }

    fn input(name: &str) -> InputDevice {
        InputDevice {
            id: DeviceId(name.into()),
            name: name.into(),
            host: "JACK".into(),
            channels: 1,
            default_sample_rate: 48000,
            is_monitor: false,
        }
    }

    fn ms(name: &str, score: f64, error: Option<&str>) -> MicScore {
        MicScore {
            device: input(name),
            transcript: String::new(),
            accuracy: 0.0,
            rms_db: -20.0,
            snr_db: 10.0,
            clip_frac: 0.0,
            score,
            error: error.map(str::to_owned),
        }
    }

    #[test]
    fn score_identical_and_case_punct() {
        assert_eq!(score_transcript(T, T), 1.0);
        assert_eq!(
            score_transcript(T, "the QUICK brown fox, jumps over the lazy dog"),
            1.0
        );
    }

    #[test]
    fn score_partial_and_empty() {
        let s = score_transcript(T, "the quick brown cat");
        assert!(0.0 < s && s < 1.0);
        assert_eq!(score_transcript(T, ""), 0.0);
    }

    #[test]
    fn score_matches_python_difflib() {
        // Reference values from Python's score_transcript (difflib, including autojunk).
        let partial = "the quick brown cat jumps over the lazy dog while seven bright purple wales sing quietly";
        assert!((score_transcript(TEST_PHRASE, partial) - 0.875).abs() < 1e-12);
        assert_eq!(score_transcript("don't stop", "dont stop!"), 1.0);
        // 245 words: autojunk ignores the frequent words, so this is far below the naive LCS ratio.
        let long = "the quick brown fox ".repeat(60) + "jumps over the lazy dog";
        assert!((score_transcript(T, &long) - 0.070_866_141_732_283_46).abs() < 1e-12);
        let spam = "the ".repeat(250);
        assert!((score_transcript(TEST_PHRASE, &spam) - 0.007_518_796_992_481_203).abs() < 1e-12);
    }

    #[test]
    fn analyze_silence_and_empty() {
        let floor = Signal {
            rms_db: -120.0,
            snr_db: 0.0,
            clip_frac: 0.0,
        };
        assert_eq!(analyze_audio(&[]), floor);
        assert_eq!(analyze_audio(&vec![0.0; 16000]), floor);
    }

    #[test]
    fn analyze_sine_levels() {
        let s = analyze_audio(&sine(0.5, 16000));
        assert!(-10.0 < s.rms_db && s.rms_db < -8.0);
        assert_eq!(s.clip_frac, 0.0);
        assert!(analyze_audio(&sine(0.05, 16000)).rms_db < s.rms_db);
    }

    #[test]
    fn analyze_clipped_and_snr() {
        assert_eq!(analyze_audio(&vec![1.0; 16000]).clip_frac, 1.0);
        let mut x = sine(0.5, 16000);
        x.extend(std::iter::repeat_n(0.001_f32, 32000));
        assert!(analyze_audio(&x).snr_db > 20.0);
    }

    #[test]
    fn combine_monotonic() {
        assert!(combine(0.9, -20.0, 20.0, 0.0) > combine(0.5, -20.0, 20.0, 0.0));
        assert!(combine(0.9, -20.0, 30.0, 0.0) > combine(0.9, -20.0, 10.0, 0.0));
        assert!(combine(0.9, -20.0, 20.0, 0.2) < combine(0.9, -20.0, 20.0, 0.0));
        assert!(combine(0.9, -55.0, 20.0, 0.0) < combine(0.9, -20.0, 20.0, 0.0));
    }

    #[test]
    fn rank_errors_last() {
        let r = rank(vec![
            ms("d0", 5.0, None),
            ms("d1", 99.0, Some("silent")),
            ms("d2", 50.0, None),
            ms("d3", ERROR_SCORE, Some("x")),
        ]);
        let names: Vec<&str> = r.iter().map(|s| s.device.name.as_str()).collect();
        assert_eq!(&names[..2], ["d2", "d0"]);
        assert!(r[2..].iter().all(|s| s.error.is_some()));
        assert_eq!(suggested(&r).map(|s| s.device.name.as_str()), Some("d2"));
    }

    /// Records every device of `source` the way the UI does between "start" and "Enter".
    type Recording = (
        Vec<InputDevice>,
        BTreeMap<DeviceId, Vec<f32>>,
        BTreeMap<DeviceId, String>,
    );

    fn record(source: &FakeAudioSource) -> Recording {
        let devices = source.list_input_devices().unwrap();
        let mut rec: Box<dyn MultiRecorder> = source.multi_recorder(&devices);
        let failed = rec.start();
        let audio = rec.stop();
        (devices, audio, failed)
    }

    fn run(transcriber: &dyn Transcriber, source: &FakeAudioSource) -> Vec<MicScore> {
        let (devices, audio, failed) = record(source);
        evaluate(
            transcriber,
            &devices,
            &audio,
            &failed,
            None,
            &mut |_, _, _| {},
        )
    }

    #[test]
    fn evaluate_ranks_good_mic_first() {
        let source = FakeAudioSource::new()
            .device("Good Mic", false, sine(0.5, 16000))
            .device("Silent Monitor", true, vec![0.0; 16000])
            .device("Broken", false, sine(0.5, 16000))
            .failing_open("Broken");
        let transcriber = FakeTranscriber::new(TEST_PHRASE);
        let out = run(&transcriber, &source);

        let names: Vec<&str> = out.iter().map(|s| s.device.name.as_str()).collect();
        assert_eq!(names, ["Good Mic", "Silent Monitor", "Broken"]);
        assert!(out[0].is_ok());
        assert_eq!(out[0].accuracy, 1.0);
        assert_eq!(out[0].transcript, TEST_PHRASE);
        assert_eq!(out[1].error.as_deref(), Some(SILENT_ERROR));
        assert_eq!(out[1].score, ERROR_SCORE);
        assert!(
            out[2]
                .error
                .as_deref()
                .unwrap()
                .starts_with("failed to open: ")
        );
        // Only the audible device that opened is transcribed.
        assert_eq!(transcriber.calls().len(), 1);
    }

    #[test]
    fn evaluate_reports_model_load_failure() {
        let source = FakeAudioSource::new().device("Good Mic", false, sine(0.5, 16000));
        let transcriber = FakeTranscriber::new(TEST_PHRASE).failing_load("no model");
        let out = run(&transcriber, &source);
        assert_eq!(out.len(), 1);
        let error = out[0].error.as_deref().unwrap();
        assert!(error.starts_with("model load failed: "), "{error}");
        assert!(error.contains("no model"));
        assert_eq!(out[0].score, ERROR_SCORE);
    }

    #[test]
    fn evaluate_reports_transcription_error_per_device() {
        let source = FakeAudioSource::new()
            .device("Good Mic", false, sine(0.5, 16000))
            .device("Flaky Mic", false, sine(0.3, 16000));
        // Answer the phrase for the louder clip only; fail the other one.
        let transcriber = FakeTranscriber::with_fn(|audio| {
            if audio.iter().any(|s| s.abs() > 0.4) {
                Ok(TEST_PHRASE.into())
            } else {
                Err("boom".into())
            }
        });
        let out = run(&transcriber, &source);
        assert_eq!(out.len(), 2);
        assert_eq!(out[0].device.name, "Good Mic");
        assert!(out[0].is_ok());
        assert_eq!(out[1].device.name, "Flaky Mic");
        let error = out[1].error.as_deref().unwrap();
        assert!(error.starts_with("transcription error: "), "{error}");
    }
}
