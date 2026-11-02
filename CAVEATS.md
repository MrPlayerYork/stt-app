# Rust Rewrite: Open Caveats

Things to come back to. Each one is an open question or a known gap, not a blocker. Tick it off (or delete it)
once it's decided. Context: [RUST_PLAN.md](RUST_PLAN.md); gate numbers: `scratch/whisper-spike/RESULTS.md`.

## From the Whisper gate test (P1-b, 2026-11-01, verdict GO)
- [ ] **Cold start on first GPU run: 16.6 s.** Probably a one-time Vulkan shader compile; warm runs took 0.22 s.
  Untested: after a reboot, after a driver update, or with the shader cache cleared. The app needs a warm-up
  transcription at load (the Python app already does a 0.5 s silence warm-up) and a visible "warming up" state.
  Worth measuring once P2-8 exists.
- [ ] **Decoding settings weren't identical.** Rust used greedy decoding with no VAD; the Python app uses beam 5 with
  `vad_filter=True`. Re-compare speed once `stt-whisper` uses the same settings (beam search will cost some time).
- [ ] **CPU fallback is about 1x real time** (21.8 s for 20 s of audio, default 4 threads). Thread count not tuned;
  the 3700X has 16 threads.
- [ ] **Small sample:** one 20 s clip, one speaker. Try a longer clip, names/numbers/jargon, and maybe a noisy one
  before calling the rewrite done.
- [ ] **Model default:** turbo q5_0 used about half the VRAM of f16 (1.1 GB vs 2.1 GB) with identical text. Probably the
  default, but that needs a config decision (see the P1-a question below).
- [x] Build workaround (whisper-rs 0.16, brew Vulkan headers, rpath): now in AGENTS.md, "Building whisper.cpp with Vulkan".

## From the workspace skeleton (P1-a): decide before or during Phase 2
- [x] **Keyring compatibility (P2-1):** verified 2026-11-01. Rust (`keyring` 4.2, Secret Service over zbus) reads
  the same entries as Python: `openai_compat` present and identical (hash compared, value never shown); the
  other four absent in both.
- [x] **Model quantization setting:** decided 2026-11-01: new key `whisper.quantization` (f16 / q8_0 / q5_0),
  default q5_0. P2-8 maps missing variants to the nearest one. The `stt` binary enabling the `whisper` feature is P3-3's job.
- [ ] **Microphone names (P2-5):** the mic name saved by the Python app may not match cpal's device names, so a saved
  mic could go missing after switching. Needs a matching rule (or re-calibration on mismatch).
- [ ] **Config leniency:** unknown `device` / `provider` / `mode` / `reasoning` values now silently fall back to the
  default (Python failed instead). Fine, or should the app warn?
- [ ] `whisper.device = "cuda"` is still what gets written (so the Python app can read the file); `"gpu"`/`"vulkan"`
  are accepted on read. Revisit once the Python app is retired.
