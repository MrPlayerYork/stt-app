# Rust Rewrite Plan

## Context
The Python app (`src/stt_app/`, milestone `reasoning-working`) works and is feature-complete. Startup is already
fast (0.23 s for all imports) and the heavy work is native code, so embedding Python in a Rust binary would gain
nothing. Goal of the rewrite: **one native binary, instant startup, no Python/venv/CUDA wheels (~2.7 GB today)**,
with full feature parity. The Python app stays in the repo as the working reference until the Rust app replaces it.

Sub-agent rules, resource limits and the shared board are in **[AGENTS.md](AGENTS.md)**. Read it before launching anything.

## Decisions
| Area | Choice | Why |
|---|---|---|
| Speech-to-text | `whisper-rs` (whisper.cpp) with the **Vulkan** feature | GPU without a CUDA toolkit (Bazzite: no system installs). CPU fallback built in. |
| Models | whisper.cpp ggml files from HF `ggerganov/whisper.cpp` (e.g. `ggml-large-v3-turbo.bin`, `-q5_0`/`-q8_0` variants) | Native format; quantized variants fit next to LM Studio in 12 GB VRAM. |
| HTTP | `ureq` (blocking) + `serde_json` | No async runtime: smaller build, simpler code. Threads + channels where concurrency is needed. |
| LLMs | Raw HTTPS to OpenAI / OpenAI-compatible, Anthropic, Google | API-only (user decision). Same features: model list, reasoning levels + capture, TTL, unload. |
| Audio | `cpal` (ALSA/PipeWire) | Device list, multi-device simultaneous capture for calibration. Resample to 16 kHz mono f32. |
| UI | `ratatui` + `crossterm` with an **inline viewport** (not full-screen) | Matches the current Rich feel; live redraw replaces the "Live only when busy" hack. |
| Clipboard | `wl-copy` → `xclip` → `xsel` subprocess, same as Python | `wl-copy` forks and keeps serving after exit, which libraries get wrong on Wayland. |
| Secrets | `keyring` crate, service `stt-app`, username = provider | **Existing keys carry over** from the Python app. |
| Config | Same file `~/.config/stt-app/config.toml`, same schema (`serde` + `toml`) | **Existing settings carry over.** New Rust-only keys are additive with defaults. |
| Errors / CLI | `thiserror` in libs, `anyhow` in the binary | Conventional. |

## Workspace layout (`rust/`, cargo workspace, binary name `stt`)
```
rust/
  Cargo.toml              # workspace; shared deps + versions in [workspace.dependencies]
  .cargo/config.toml      # [build] jobs = 2  (resource rule; heavy slot overrides with CARGO_BUILD_JOBS)
  crates/
    stt-core/        config (TOML schema, load/save atomic), secrets (keyring+env), shared types & TRAITS:
                     Transcriber, Recorder/AudioSource, LlmBackend, LlmResult, Progress callbacks
    stt-audio/       cpal: list_input_devices (dedupe monitors), Recorder (level meter), MultiRecorder, to_mono_16k
    stt-whisper/     whisper-rs + Vulkan: load (GPU→CPU fallback), transcribe, model name → ggml file map
                     ** the ONLY crate that compiles whisper.cpp — heavy build, see AGENTS.md **
    stt-download/    multi-connection ranged download with resume (.part + .progress.json), HF file listing,
                     progress callback; port of reference/hf_downloader.py + src/stt_app/download.py
    stt-llm/         providers (openai/openai_compat, anthropic, google), prompts + presets + word choice,
                     reasoning levels + capture (<think> stripping), server detection, TTL, unload
    stt-clipboard/   copy(text) -> method
    stt-calibrate/   pure scoring (transcript accuracy, rms/snr/clipping, combine, rank) + flow driver over traits
    stt-tui/         ratatui app: main screen, record flow, history, settings, calibration screens, masked input
    stt/             binary: wires concrete impls into stt-tui, TTY check, first-run logic
```
**Key rule:** everything except `stt-whisper` and the `stt` binary depends only on traits in `stt-core`, so agents
working on UI, LLM, audio, etc. never compile whisper.cpp. UI/calibration tests use fake Transcribers.

## Parity checklist (the definition of done)
Python tests in `tests/` are the behavioural spec; port each to a Rust test in the matching crate.
- [ ] TTY check; first run → calibration; re-run calibration while no mic is chosen
- [ ] Calibration: record all mics at once, live per-mic levels, silent monitors marked, Whisper accuracy scoring, ranked table, pick/retry/cancel
- [ ] Push-to-record with live level meter; Enter stops, Ctrl+C cancels
- [ ] Whisper model setting (large-v3, large-v3-turbo, distil-large-v3, medium, small), device auto/gpu/cpu, language; GPU→CPU fallback
- [ ] Fast model download: multi-connection, resume, progress in header + settings, HF token, reuse existing files
- [ ] Clipboard **replaces** each time; history with re-copy
- [ ] Modes Raw / Cleanup / Prompt; presets (deramble, concise, markdown) + custom prompts; word choice (Prompt only)
- [ ] LLM providers: OpenAI, OpenAI-compatible (LM Studio/Ollama/vLLM/llama-server presets), Anthropic, Google; model list; test connection; API key in keyring or env var; LLM failure → copy raw text
- [ ] Reasoning level (default/off/low/medium/high) per provider; reasoning line + view; `<think>` never reaches clipboard
- [ ] Local server detection; idle TTL (LM Studio `ttl`, Ollama `keep_alive`); unload on exit (opt-in)
- [ ] Single-key menus; masked secret entry with `*` echo, paste-safe
- [ ] Reads the existing Python config + keyring entries unchanged

## Phases
Model tiers: **H** = haiku, **S** = sonnet, **O** = opus. "Heavy" = needs the `heavy` board slot.

### Phase 0 — Prerequisites (orchestrator + user, before any agent)
1. `/merge` the milestone branches into `master`; branch `rust-rewrite` from it (docs are committed there already).
2. Vulkan build deps are **already installed via brew** (`vulkan-headers`, `vulkan-loader`, `shaderc`/`glslc`,
   `glslang`, `spirv-*`, plus `cmake`, `clang`). The system `/usr/bin/pkg-config` doesn't search brew, so builds need
   `PKG_CONFIG_PATH=/home/linuxbrew/.linuxbrew/lib/pkgconfig` (put it in `rust/.cargo/config.toml` `[env]`).
   `sccache` 0.19.0 and `ninja` 1.13.2 are installed via brew (done 2026-11-01). P1-a wires them into
   `rust/.cargo/config.toml`: `[build] rustc-wrapper = "sccache"`, and `[env]` `SCCACHE_DIR` on <data-drive>
   (e.g. `<data-drive>/.sccache`, same disk as the worktrees), `CMAKE_GENERATOR = "Ninja"`,
   `CMAKE_C_COMPILER_LAUNCHER`/`CMAKE_CXX_COMPILER_LAUNCHER = "sccache"` so whisper.cpp's C++ is cached too.
3. Verify: `PKG_CONFIG_PATH=... pkg-config --modversion vulkan` (1.4.363), `glslc --version`, `cargo --version` (rustc 1.97.1).

### Phase 1 — Skeleton + Whisper gate (2 agents, parallel)
| ID | Task | Tier | Notes |
|---|---|---|---|
| P1-a | Workspace skeleton: all crates as stubs, `[workspace.dependencies]`, `.cargo/config.toml`, **all trait/type contracts in `stt-core`** with doc comments, CI-style `cargo check --workspace` (without stt-whisper) | O | Contracts are what makes Phase 2 parallel. Opus because mistakes here multiply. |
| P1-b | **Gate spike** in `scratch/whisper-spike/` (standalone crate): whisper-rs + Vulkan, load `ggml-large-v3-turbo` (and q5_0), transcribe a fixed WAV recorded from the Python app; compare text + speed vs faster-whisper on the same WAV; measure VRAM | S, heavy+gpu+download | **Go/no-go for the whole rewrite.** User reviews the numbers. |

**Gate:** if whisper.cpp/Vulkan is clearly worse (accuracy or >2x slower), stop and decide with the user (options: whisper-rs CUDA via a distrobox build, or keep Python for STT).

### Phase 2 — Crates (up to 8 agents, parallel, one worktree each)
| ID | Crate / task | Tier | Resources |
|---|---|---|---|
| P2-1 | `stt-core` config + secrets: schema identical to `src/stt_app/config.py`, atomic save, round-trip tests incl. reading a real Python-written config | H | light |
| P2-2 | `stt-clipboard` | H | light |
| P2-3 | `stt-calibrate` pure scoring + tests ported from `tests/test_calibrate.py` | H | light |
| P2-4 | `stt-download` (+ tests with a local fake HTTP server, like `tests/test_download.py`) | S | light; real-download check needs `download` |
| P2-5 | `stt-audio` (cpal; device dedupe like `audio.py`; multi-capture) | S | light; short real-mic smoke test |
| P2-6 | `stt-llm` part 1: openai + openai_compat backends, server detection, TTL/unload, reasoning capture, `<think>` stripping | S | light |
| P2-7 | `stt-llm` part 2: anthropic + google backends, prompts/presets/word choice | S | light (coordinate module boundaries with P2-6 via the board `note`) |
| P2-8 | `stt-whisper` (productionise the spike: model map, GPU→CPU fallback, progress hooks, `#[ignore]` GPU tests) | S | **heavy** + gpu |

### Phase 3 — UI + binary (3 agents, parallel on separate modules of `stt-tui`)
| ID | Task | Tier |
|---|---|---|
| P3-1 | App state + main screen (inline viewport, live header, single-key menu) + record flow + history | S |
| P3-2 | Settings screens (mic, whisper, LLM incl. reasoning/word choice/TTL, custom prompts) + masked secret input | S |
| P3-3 | Calibration screens + first-run + `stt` binary wiring | S |
The orchestrator defines the `App` state struct and screen module boundaries in the P3 prompts before launch.

### Phase 4 — Integration + parity (orchestrator, 1–2 agents)
1. Merge Phase 3 branches into `rust-rewrite`; `cargo test --workspace`; release build (heavy slot).
2. P4-a (S): walk the parity checklist against both apps; fix gaps.
3. User tests the binary in Konsole on real mics and LLMs → milestone.
4. P4-b (H): `README` section + install (`cargo install --path rust/crates/stt` → `~/.cargo/bin/stt`), and remove the Python app only when the user says so.

## Risks
- **whisper.cpp Vulkan accuracy/speed** vs faster-whisper → the Phase 1 gate.
- **VRAM**: LM Studio often holds ~10 GB; use quantized ggml by default and fall back to CPU cleanly.
- **cpal + PipeWire device naming** differs from PortAudio's JACK list → calibration must still dedupe monitors.
- **LM Studio `reasoning_effort`** is undocumented (same caveat as Python) → keep the 400-retry behaviour.
- **Parallel builds on a home PC** → AGENTS.md slots, `jobs = 2`, sccache.

## Verification
- Every crate: `cargo test -p <crate>` (light). Workspace: `cargo test --workspace` in the heavy slot.
- GPU tests are `#[ignore]` and run only in the `gpu` slot: `cargo test -p stt-whisper -- --ignored`.
- End to end: user runs `stt` in Konsole: first run calibration, dictation in all three modes, LLM with reasoning, TTL unload observed in `nvidia-smi`.
