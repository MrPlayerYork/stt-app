//! Pure sample-format maths: downmix, resample to 16 kHz, level. No audio hardware involved.
//! Ports `to_mono_16k` and `_rms` from `src/stt_app/audio.py`.

use std::f64::consts::PI;

use stt_core::SAMPLE_RATE;

/// Zero crossings of the sinc kernel on each side (at the lower of the two rates).
const ZEROS: f64 = 16.0;
/// Kernel table resolution: sub-sample steps per input sample.
const STEPS: usize = 512;

/// Average interleaved frames of `channels` samples into one mono sample each.
/// A trailing partial frame is ignored. `channels == 0` is treated as 1.
pub fn downmix(interleaved: &[f32], channels: usize) -> Vec<f32> {
    let ch = channels.max(1);
    if ch == 1 {
        return interleaved.to_vec();
    }
    interleaved
        .chunks_exact(ch)
        .map(|f| f.iter().sum::<f32>() / ch as f32)
        .collect()
}

/// Linear RMS of a block, clamped to `0.0..=1.0` (Python `_rms`). Empty block is 0.
pub fn rms(block: &[f32]) -> f32 {
    if block.is_empty() {
        return 0.0;
    }
    let sum: f64 = block.iter().map(|&x| f64::from(x) * f64::from(x)).sum();
    ((sum / block.len() as f64).sqrt() as f32).min(1.0)
}

/// Peak absolute value of a buffer.
pub fn peak(block: &[f32]) -> f32 {
    block.iter().fold(0.0_f32, |m, &x| m.max(x.abs()))
}

/// Windowed-sinc (Blackman) resample of mono audio from `rate` Hz to `to` Hz. Low-pass filters when
/// downsampling (the Python version used plain linear interpolation, which aliases). Edges are handled
/// by renormalising the kernel over the samples that exist, so DC level is preserved.
pub fn resample(mono: &[f32], rate: u32, to: u32) -> Vec<f32> {
    if mono.is_empty() || rate == 0 || to == 0 {
        return Vec::new();
    }
    if rate == to {
        return mono.to_vec();
    }
    let ratio = f64::from(to) / f64::from(rate);
    let out_len = ((mono.len() as f64 * ratio).round() as usize).max(1);
    // Cutoff as a fraction of the input rate's Nyquist; kernel half-width in input samples.
    let cutoff = ratio.min(1.0);
    let half = ZEROS / cutoff;
    let table = kernel_table(cutoff, half);

    let step = 1.0 / ratio; // input samples per output sample
    let mut out = Vec::with_capacity(out_len);
    for m in 0..out_len {
        let center = m as f64 * step;
        let lo = (center - half).ceil().max(0.0) as usize;
        let hi = ((center + half).floor() as usize).min(mono.len() - 1);
        let mut acc = 0.0_f64;
        let mut wsum = 0.0_f64;
        if lo <= hi {
            for (i, &s) in mono.iter().enumerate().take(hi + 1).skip(lo) {
                let w = table_lookup(&table, (i as f64 - center).abs());
                acc += w * f64::from(s);
                wsum += w;
            }
        }
        out.push(if wsum.abs() > 1e-9 {
            (acc / wsum) as f32
        } else {
            0.0
        });
    }
    out
}

fn kernel_table(cutoff: f64, half: f64) -> Vec<f64> {
    let n = (half * STEPS as f64).ceil() as usize + 2;
    (0..n)
        .map(|k| {
            let t = k as f64 / STEPS as f64;
            if t >= half {
                return 0.0;
            }
            let x = PI * cutoff * t;
            let sinc = if x.abs() < 1e-12 { 1.0 } else { x.sin() / x };
            let u = t / half; // 0..1
            let window = 0.42 + 0.5 * (PI * u).cos() + 0.08 * (2.0 * PI * u).cos();
            cutoff * sinc * window
        })
        .collect()
}

fn table_lookup(table: &[f64], t: f64) -> f64 {
    let p = t * STEPS as f64;
    let i = p as usize;
    if i + 1 >= table.len() {
        return 0.0;
    }
    let frac = p - i as f64;
    table[i] * (1.0 - frac) + table[i + 1] * frac
}

/// Downmix interleaved audio to mono and resample to [`SAMPLE_RATE`].
pub fn to_mono_16k(interleaved: &[f32], channels: usize, rate: u32) -> Vec<f32> {
    resample(&downmix(interleaved, channels), rate, SAMPLE_RATE)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sine(freq: f64, rate: u32, secs: f64) -> Vec<f32> {
        (0..(f64::from(rate) * secs) as usize)
            .map(|i| (2.0 * PI * freq * i as f64 / f64::from(rate)).sin() as f32)
            .collect()
    }

    #[test]
    fn output_length_for_rates() {
        for rate in [48_000_u32, 16_000, 44_100, 32_000, 8_000] {
            let out = to_mono_16k(&sine(440.0, rate, 1.0), 1, rate);
            assert!(
                out.len().abs_diff(16_000) <= 1,
                "rate {rate} -> {}",
                out.len()
            );
        }
    }

    #[test]
    fn stereo_half_second_length() {
        let out = to_mono_16k(&vec![0.0; 48_000], 2, 48_000);
        assert!(out.len().abs_diff(8_000) <= 1);
    }

    #[test]
    fn passthrough_16k_mono() {
        let x: Vec<f32> = (0..1600).map(|i| -1.0 + 2.0 * i as f32 / 1599.0).collect();
        assert_eq!(to_mono_16k(&x, 1, 16_000), x);
    }

    #[test]
    fn length_scaled_from_48k() {
        assert_eq!(to_mono_16k(&vec![0.0; 4800], 1, 48_000).len(), 1600);
    }

    #[test]
    fn stereo_averaged() {
        let mut st = Vec::new();
        for _ in 0..1600 {
            st.extend([1.0_f32, 0.0]);
        }
        let out = to_mono_16k(&st, 2, 16_000);
        assert_eq!(out.len(), 1600);
        assert!(out.iter().all(|&v| (v - 0.5).abs() < 1e-6));
    }

    #[test]
    fn stereo_mixes_per_sample() {
        let out = to_mono_16k(&[0.2, 0.4, -1.0, 1.0, 0.6, 0.0], 2, 16_000);
        for (a, b) in out.iter().zip([0.3, 0.0, 0.3]) {
            assert!((a - b).abs() < 1e-6);
        }
    }

    #[test]
    fn empty_input_is_empty() {
        assert!(to_mono_16k(&[], 2, 48_000).is_empty());
    }

    #[test]
    fn dc_preserved_including_edges() {
        let out = to_mono_16k(&vec![0.5; 48_000], 1, 48_000);
        assert!(out.iter().all(|&v| (v - 0.5).abs() < 1e-4));
    }

    #[test]
    fn in_band_tone_keeps_amplitude() {
        let out = to_mono_16k(&sine(1000.0, 48_000, 1.0), 1, 48_000);
        let mid = &out[2000..14_000];
        assert!(
            (rms(mid) - std::f32::consts::FRAC_1_SQRT_2).abs() < 0.01,
            "rms {}",
            rms(mid)
        );
    }

    #[test]
    fn out_of_band_tone_is_filtered_not_aliased() {
        // 12 kHz is above the 8 kHz target Nyquist: linear interpolation would alias it, we must not.
        let out = to_mono_16k(&sine(12_000.0, 48_000, 1.0), 1, 48_000);
        assert!(rms(&out[2000..14_000]) < 0.01);
    }

    #[test]
    fn upsample_8k_tone() {
        let out = to_mono_16k(&sine(500.0, 8_000, 1.0), 1, 8_000);
        assert!(out.len().abs_diff(16_000) <= 1);
        assert!((rms(&out[2000..14_000]) - std::f32::consts::FRAC_1_SQRT_2).abs() < 0.02);
    }

    #[test]
    fn rms_values() {
        assert_eq!(rms(&[]), 0.0);
        assert!((rms(&[0.5, -0.5, 0.5, -0.5]) - 0.5).abs() < 1e-6);
        assert_eq!(rms(&[3.0, -3.0]), 1.0); // clamped
        assert_eq!(peak(&[0.1, -0.7, 0.3]), 0.7);
    }

    #[test]
    fn downmix_ignores_partial_frame() {
        assert_eq!(downmix(&[1.0, 3.0, 5.0], 2), vec![2.0]);
        assert_eq!(downmix(&[1.0, 2.0], 0), vec![1.0, 2.0]);
    }
}
