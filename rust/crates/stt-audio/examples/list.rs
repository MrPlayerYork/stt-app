//! Smoke test: lists devices, records 2 s from the default input, then from all inputs at once.
//! Prints only device names and peak/RMS numbers; audio is never saved or played.
use std::time::Duration;
use stt_audio::{CpalAudioSource, duration_secs, peak_rms};
use stt_core::AudioSource;

fn main() {
    let src = CpalAudioSource::new();
    let devs = src.list_input_devices().expect("list");
    for d in &devs {
        println!(
            "dev: {:?} host={} ch={} rate={} monitor={}",
            d.name, d.host, d.channels, d.default_sample_rate, d.is_monitor
        );
    }
    let mut rec = src.recorder(None).expect("recorder");
    rec.start().expect("start");
    std::thread::sleep(Duration::from_millis(1000));
    println!("default level mid-capture: {:.4}", rec.level());
    std::thread::sleep(Duration::from_millis(1000));
    let a = rec.stop().expect("stop");
    let (p, r) = peak_rms(&a);
    println!(
        "default: {:.2}s peak={p:.4} rms={r:.4} recording={}",
        duration_secs(&a),
        rec.is_recording()
    );

    let mut multi = src.multi_recorder(&devs);
    let errs = multi.start();
    for (id, e) in &errs {
        println!("open failed: {id}: {e}");
    }
    std::thread::sleep(Duration::from_secs(2));
    let lv = multi.levels();
    let out = multi.stop();
    for d in &devs {
        if let Some(a) = out.get(&d.id) {
            let (p, r) = peak_rms(a);
            println!(
                "{:?}: {:.2}s peak={p:.4} rms={r:.4} lastlevel={:.4}",
                d.name,
                duration_secs(a),
                lv.get(&d.id).copied().unwrap_or(0.0)
            );
        }
    }
}
