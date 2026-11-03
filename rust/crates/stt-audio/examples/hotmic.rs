//! Hot-mic smoke test: open, listen, record twice, report timings. Prints only numbers.
use std::time::{Duration, Instant};
use stt_audio::CpalAudioSource;
use stt_core::AudioSource;

fn main() {
    let src = CpalAudioSource::new();
    let mut cold = src.recorder(None).expect("recorder");
    let t = Instant::now();
    cold.start().expect("cold start");
    println!("cold start (no open): {:?}", t.elapsed());
    std::thread::sleep(Duration::from_secs(1));
    let a = cold.stop().expect("stop");
    println!("cold stop: {} samples, open={}", a.len(), cold.is_open());

    let mut rec = src.recorder(None).expect("recorder");
    let t = Instant::now();
    rec.open().expect("open");
    println!("open: {:?} open={}", t.elapsed(), rec.is_open());
    std::thread::sleep(Duration::from_secs(1));
    println!("listening level: {:.4}", rec.level());
    // Meter refresh: how often the level value changes (one change per callback block).
    let (mut changes, mut last) = (0u32, rec.level());
    let t = Instant::now();
    while t.elapsed() < Duration::from_secs(2) {
        let l = rec.level();
        if l != last {
            changes += 1;
            last = l;
        }
        std::thread::sleep(Duration::from_millis(2));
    }
    println!("meter updates: {:.1}/s", changes as f32 / 2.0);
    for i in 0..2 {
        let t = Instant::now();
        rec.start().expect("start");
        println!("hot start #{i}: {:?}", t.elapsed());
        std::thread::sleep(Duration::from_secs(1));
        let a = rec.stop().expect("stop");
        println!("stop: {} samples, open={}", a.len(), rec.is_open());
    }
    rec.close();
    println!("closed: open={} level={}", rec.is_open(), rec.level());
}
