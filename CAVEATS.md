# Rust Rewrite: Open Caveats

Things to come back to. Each one is an open question or a known gap, not a blocker. Tick it off (or delete it)
once it's decided. Context: [RUST_PLAN.md](RUST_PLAN.md); gate numbers: `scratch/whisper-spike/RESULTS.md`.

## From the Whisper gate test (P1-b, 2026-11-01, verdict GO)
- [ ] **Cold start: about 20 s once, then 0.6 s.** The first GPU load+warm-up after a fresh build took 16.6 s (P1-b) /
  20.5 s (P2-8): a one-time Vulkan shader compile. The next fresh process took 0.64 s, so the shader cache
  persists across runs. Untested: after a reboot or a driver update. stt-whisper already warms up at load; Phase 3
  still needs a visible "warming up…" state for the slow case.
- [x] **Speed with matched settings (P2-8):** stt-whisper now uses beam 5 + Silero VAD like the Python app. Warm
  transcribe of the 20 s clip, q5_0 GPU: **beam 5 = 0.34 s** (greedy 0.23 s) vs faster-whisper 0.245 s. So about
  1.4x slower on transcription with equal settings (well inside the gate's 2x), but loads about 4x faster. Decide
  whether to keep beam 5 or offer greedy as a "fast" option.
  **Decided 2026-11-02:** default beam 5; Whisper settings offer greedy as a "fast" option (P3-2).
- [ ] **CPU fallback:** beam 5, 20 s clip: 4 threads 23.7 s, 8 threads about 13.8 s, 12 threads 13.6 s (machine was busy
  with other agents, so noisy). Default is now 8 threads, about 0.7x real time. Fine as a fallback; slow as a main path.
- [ ] **VAD model:** Silero VAD (`ggml-silero-v5.1.2.bin`, 885 KB from `ggml-org/whisper-vad`) is fetched next to
  the Whisper model; if that fails, decoding runs without VAD. Check that the app shows it in download progress.
- [ ] **Small sample:** one 20 s clip, one speaker. Try a longer clip, names/numbers/jargon, and maybe a noisy one
  before calling the rewrite done.
- [x] **Model default:** q5_0 (same text as f16, about half the VRAM: 1.1 GB vs 2.1 GB). See the quantization item below.
- [x] Build workaround (whisper-rs 0.16, brew Vulkan headers, rpath): now in AGENTS.md, "Building whisper.cpp with Vulkan".

## From the workspace skeleton (P1-a): decide before or during Phase 2
- [x] **Keyring compatibility (P2-1):** verified 2026-11-01. Rust (`keyring` 4.2, Secret Service over zbus) reads
  the same entries as Python: `openai_compat` present and identical (hash compared, value never shown); the
  other four absent in both.
- [x] **Model quantization setting:** decided 2026-11-01: new key `whisper.quantization` (f16 / q8_0 / q5_0),
  default q5_0. P2-8 maps missing variants to the nearest one. The `stt` binary enabling the `whisper` feature is P3-3's job.
- [x] **Microphone names (P2-5):** solved 2026-11-01. cpal's PulseAudio host (served by pipewire-pulse) reports the
  same names the Python app saved (e.g. "Blue Snowball Analog Stereo"). `stt_audio::match_saved_device` matches exact →
  normalised → substring; no match → re-calibrate. Details: `scratch/device-naming.md`. Confirm with your real saved mic in Phase 4.
- [x] **Config leniency:** unknown `device` / `provider` / `mode` / `reasoning` values now silently fall back to the
  default (Python failed instead). Fine, or should the app warn?
  **Decided 2026-11-02:** warn on startup. Also try recovering a partial config: keep every valid value, reset the
  rest, and show what was recovered and what still needs a value (back up the original file first).
- [ ] `whisper.device = "cuda"` is still what gets written (so the Python app can read the file); `"gpu"`/`"vulkan"`
  are accepted on read. Revisit once the Python app is retired.

## From the Phase 2 crates
- [ ] **Stalled download can hang (stt-download):** `ureq` only has a whole-body timeout, no per-read timeout, so a
  connection that stops sending mid-file could hang that download. Fix: a read-idle watchdog per connection, or a
  socket read timeout. Matters for multi-GB models on flaky Wi-Fi.
- [ ] **LLM error wording (stt-llm):** `LlmError` has no free-text variant, so some messages read differently from
  Python (e.g. "HTTP 401: Authentication failed…" instead of Python's text; timeouts and connection failures use
  generic variants). Cosmetic. Fix if wanted: add `LlmError::Other(String)` and use the Python strings.
- [x] **Record start latency (stt-audio):** opening a PulseAudio stream takes about 1-2 s per device, so the first
  push-to-record could miss your first word. Likely fix in Phase 3: open the mic stream once at startup and keep it
  warm (or pre-open when the main screen shows). Check in Phase 4 testing.
  **Decided 2026-11-02:** hot mic is OK: open the stream at startup and keep it warm, but the UI must always show
  that the mic is open, and show clearly (distinct from that) when audio is actually being recorded.
  **Done (P3-5):** start on a real mic took 2.12 s without the hot mic, 4-5 µs with it (open itself 2.04 s, once).
  Confirm in Phase 4 that the first word is no longer cut off.
- [ ] **Unknown download size (stt-download):** now streams without resume support instead of failing like Python.
  Probably fine (HF always sends a size).

## Cleanup before GitHub push / local archive (after the rewrite)
- [ ] **Out-of-tree files** used during development (fine for now; gather or document them before archiving):
  shared models at `<data-drive>/stt-models` (shared across worktrees on purpose), the sccache
  cache (`<data-drive>/.sccache`, now set in the gitignored repo-root `.cargo/config.toml`). Agent worktrees and
  merged branches were removed on 2026-11-02.
- [ ] **App file locations:** paths plan from 2026-11-02 (shader cache + models next to the exe or set in config,
  portable config next to the exe, install to a chosen folder instead of `~/.cargo/bin`). Not started.
- [ ] **Release binary rpath** points at `/home/linuxbrew/.linuxbrew/lib` (Vulkan build workaround). Fine on this
  machine; revisit if the binary is ever shared or moved.
  A build from distro packages (README) has no rpath at all.

## From the pre-push checks (2026-11-02)
- [x] History rewritten: noreply author email, machine paths replaced by `<repo>` / `<data-drive>`, `reference/`
  never in history. Backup bundle of the old history: `.backup/pre-rewrite-20261102.bundle` (gitignored).
- [x] Clean build proven in a fresh Fedora distrobox using only the README's packages, with no Cargo config:
  release build, `cargo test --workspace` (279 passed), and the CPU-only `--no-default-features` build.
- [ ] **Parallel GPU loads can segfault** (found 2026-11-03): the GPU tests crash when they load models in
  parallel threads, with both the brew build and the portable Ubuntu build (`--test-threads=1` always passes, and
  the GPU is used). This is most likely what the earlier "GPU inside distrobox `--nvidia`" failure was. The app can
  hit it: `ModelLoader::start` starts a new load while an old one may still be running (a Whisper setting
  changed mid-load), and `load_lock` is per transcriber. Fix: a process-wide load lock in stt-whisper. Listed as
  a known issue in the v1.0.0 release notes.
- [ ] **Leftover to delete by hand:** `<data-drive>/stt-distrobox-home` (about 7.9 GB, the container's home with the
  clean-build checkout). Claude's safety check blocks deleting top-level folders of a drive.
- [x] **Pushed 2026-11-03:** https://github.com/MrPlayerYork/stt-app (public; master and all 3 tags).
- [x] **v1.0.0 release:** `stt-1.0.0-linux-x86_64.tar.xz` (+ sha256), built in an Ubuntu 24.04 distrobox with
  the LTO release profile: system libs only, no rpath, needs glibc 2.39+. Its GPU tests passed on this host.

## From Phase 3 (UI + binary, 2026-11-02)
- [ ] **Cancel while processing:** Ctrl+C after Enter stops the copy, but the running transcription / LLM call
  finishes in the background (result dropped). Fine unless a long LLM call ties up the local server.
- [ ] **Default mic name:** with no mic chosen the header says "(system default)" rather than the real device name.
- [ ] **Config salvage:** a multi-line value (e.g. a long custom prompt) inside a section that fails to parse can't
  be rescued; single-line values are.
- [ ] **Python-written config with LLM on but no model/base URL** gets flagged, backed up and rewritten with the LLM
  off (the Python app then sees it off too). Keep, or only warn?
- [ ] **LLM settings key:** "Unload when quitting" is on `u` (digits run out after 9).
- [ ] **Nothing has been run in a real terminal yet** (tests render at 80x24). Phase 4: Konsole, real mics, LLMs.
