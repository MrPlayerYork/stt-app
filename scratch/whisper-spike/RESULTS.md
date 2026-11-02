# whisper.cpp (Vulkan) vs faster-whisper spike (P1-b)

Audio: `reference/gate.wav`, 20.20 s, 16 kHz mono s16, language `en`. Raw logs: `out_*.log`.

## Results

| Engine / model | load s | transcribe s (run1 / run2 / run3) | best s | peak VRAM delta MiB | WER |
|---|---|---|---|---|---|
| whisper.cpp Vulkan, turbo q5_0 | 0.80 | 16.63 / 0.221 / 0.221 | 0.221 | 1083 | 0.098 |
| whisper.cpp Vulkan, turbo f16 | 0.98 | 1.00 / 0.238 / 0.230 | 0.230 | 2087 | 0.098 |
| whisper.cpp CPU (4 threads default), turbo q5_0 | 0.45 | 26.7 / 22.0 / 21.8 | 21.8 | 3 | 0.098 |
| faster-whisper 1.x CUDA int8_float16, large-v3-turbo (app config) | 3.43 | 0.305 / 0.245 / 0.253 | 0.245 | 1261 | 0.098 |

VRAM baseline was 2456 MiB (LM Studio etc.) before every run; the delta is peak minus baseline, sampled every 0.5 s.
Vulkan device: `ggml_vulkan: 0 = NVIDIA GeForce RTX 4070 SUPER (NVIDIA) ... matrix cores: NV_coopmat2`; `using Vulkan0 backend`. It was the real GPU, not llvmpipe.

## Accuracy

WER 4 edits / 41 words for all four, normalized (lowercase, no punctuation). The errors are the same everywhere and come from the
user's reference not matching what was actually said, not from the engines:
reference "audio is works" vs heard "audio works"; "so we can" vs "so that we can"; "helps" vs "becomes useful".

- Reference: This is a test. I am making sure that the audio is works correctly on both models so we can compare them. I don't know if it will work, but I hope this audio helps in some way, shape, or form.
- whisper.cpp (q5_0 and f16, identical): This is a test. I am making sure that the audio works correctly on both models so that we can compare them. I don't know if it will work, but I hope this audio becomes useful in some way, shape, or form.
- faster-whisper: identical wording to the above.

## Caveats (be honest about these)

- Settings are not identical: the Rust side uses greedy decoding with no VAD; faster-whisper uses the app's path (beam 5, `vad_filter=True`). Speeds are still comparable on this clip (0.22 vs 0.25 s best).
- **Cold-start**: the first Vulkan run of the q5_0 model took 16.6 s (shader/pipeline compilation, no cache yet). The f16 run right after took 1.0 s on its first run, so a driver-side cache seems to help. The real app should do a warm-up transcription at load (the Python app already does a 0.5 s silence warm-up) and I did not test a cold boot or a cleared shader cache. Plan on a visible "warming up" state.
- CPU fallback is slow (about 1x real time, 4 threads default; thread count was not tuned).
- One 20 s clip, one speaker: it is a gate check, not a benchmark.
- ggml models were fetched with single-connection `curl -C -` in the download slot before the orchestrator asked for the app's multi-connection downloader. Both files match the Hugging Face API sizes exactly (1624555275 and 574041195 bytes), so I did not re-download. No download speed was recorded.

## Environment

Fedora/Bazzite, rustc 1.97.1, RTX 4070 SUPER 12 GB (driver via `nvidia-smi`), Vulkan loader/headers 1.4.363 from brew, glslc from brew, cmake 4.4.2, Ninja, sccache.
`whisper-rs 0.16.0` (whisper-rs-sys 0.15.0, feature `vulkan`) and `hound 3`.
`whisper-rs 0.14` (sys 0.13.1) did **not** build: bindgen layout assertions failed (`size_of::<_IO_FILE>() - 216`) with clang 22. 0.16 works.

Build fixes needed (all in `.cargo/config.toml`):
- Cargo reads `.cargo/config.toml` relative to the cwd, so build from `scratch/whisper-spike/` (not with `--manifest-path` from the repo root).
- Brew does not link vulkan-headers into its prefix, so a `vk-sdk/` dir (gitignored) is symlinked: `vk-sdk/include -> .../Cellar/vulkan-headers/1.4.363.0/include`, `vk-sdk/lib -> /home/linuxbrew/.linuxbrew/lib`, and `VULKAN_SDK`, `CMAKE_INCLUDE_PATH`, `CMAKE_LIBRARY_PATH` point at them.
- `rustflags` adds `-L` and an rpath for `/home/linuxbrew/.linuxbrew/lib` so `-lvulkan` links and runs. If CMake has a stale `NOTFOUND` cache, delete `target/release/build/whisper-rs-sys-*/out/build`.
- Recreate the symlinks: `mkdir vk-sdk && ln -s /home/linuxbrew/.linuxbrew/Cellar/vulkan-headers/1.4.363.0/include vk-sdk/include && ln -s /home/linuxbrew/.linuxbrew/lib vk-sdk/lib`.

## Commands

```bash
B=<worktree>/tools/agent-board.sh; MAIN=<repo>
cd scratch/whisper-spike
$B run heavy P1-b "whisper-spike build" -- env PATH=$HOME/.cargo/bin:/home/linuxbrew/.linuxbrew/bin:/usr/bin:/bin CARGO_BUILD_JOBS=8 cargo build --release
# from the worktree root:
$B run gpu P1-b "turbo q5 gpu" -- scratch/whisper-spike/run_rust.sh ggml-large-v3-turbo-q5_0.bin --gpu    # also: ggml-large-v3-turbo.bin --gpu, q5_0 --cpu
$B run gpu P1-b "faster-whisper turbo" -- scratch/whisper-spike/vram_wrap.sh uv run --project $MAIN python -I scratch/whisper-spike/fw_bench.py $MAIN/reference/gate.wav
uv run --project $MAIN python -I scratch/whisper-spike/wer.py scratch/whisper-spike/out_*.log
```

Models: `<data-drive>/stt-models/ggml-large-v3-turbo{,-q5_0}.bin` (from `huggingface.co/ggerganov/whisper.cpp`).
The first build took about 4 minutes of the heavy slot in total across attempts (sccache was active once run from the right directory).

## Verdict

**Go.** Vulkan best warm time (0.22-0.23 s) is on par with faster-whisper (0.245 s), not anywhere near 2x slower, and accuracy is identical. Open item: cold-start shader compile (about 16 s once) needs a warm-up/cache strategy.
