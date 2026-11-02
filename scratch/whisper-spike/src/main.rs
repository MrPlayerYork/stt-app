use std::time::Instant;
use whisper_rs::{FullParams, SamplingStrategy, WhisperContext, WhisperContextParameters};

fn main() {
    let a: Vec<String> = std::env::args().collect();
    if a.len() < 4 {
        eprintln!("usage: whisper-spike <model.bin> <audio.wav> --gpu|--cpu [lang=en] [runs=3]");
        std::process::exit(2);
    }
    let gpu = a[3] == "--gpu";
    let lang = a.get(4).cloned().unwrap_or_else(|| "en".into());
    let runs: usize = a.get(5).and_then(|s| s.parse().ok()).unwrap_or(3);

    let mut rd = hound::WavReader::open(&a[2]).expect("open wav");
    let spec = rd.spec();
    assert_eq!(spec.sample_rate, 16000);
    assert_eq!(spec.channels, 1);
    let audio: Vec<f32> = rd.samples::<i16>().map(|s| s.unwrap() as f32 / 32768.0).collect();
    println!("audio: {:.2}s, {} samples", audio.len() as f32 / 16000.0, audio.len());

    let t = Instant::now();
    let mut cp = WhisperContextParameters::default();
    cp.use_gpu(gpu);
    let ctx = WhisperContext::new_with_params(&a[1], cp).expect("load model");
    let mut state = ctx.create_state().expect("state");
    println!("LOAD_S {:.3} (gpu={})", t.elapsed().as_secs_f64(), gpu);

    let mut best = f64::MAX;
    let mut text = String::new();
    for i in 0..runs {
        let mut p = FullParams::new(SamplingStrategy::Greedy { best_of: 1 });
        p.set_language(Some(&lang));
        p.set_print_progress(false);
        p.set_print_realtime(false);
        p.set_print_special(false);
        p.set_print_timestamps(false);
        let t = Instant::now();
        state.full(p, &audio).expect("full");
        let dt = t.elapsed().as_secs_f64();
        let n = state.full_n_segments();
        text = (0..n).map(|i| state.get_segment(i).unwrap().to_str().unwrap().to_string()).collect::<Vec<_>>().join(" ").trim().to_string();
        println!("RUN {} TRANSCRIBE_S {:.3}", i + 1, dt);
        best = best.min(dt);
    }
    println!("BEST_S {:.3}", best);
    println!("TEXT: {}", text);
}
