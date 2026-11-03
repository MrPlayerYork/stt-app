# stt: push-to-record speech-to-text for the terminal

Press Enter, talk, and press Enter again. Your speech is transcribed locally with Whisper and copied to the
clipboard, ready to paste anywhere. Optionally, a language model cleans up the text first or rewrites it to a
prompt of your choice ("Prompt Mode").

It runs full screen in your terminal, with three tabs:
- **Main:** the live status of the mic, the Whisper model and the current mode.
- **History:** your recent dictations; click or press Enter to copy one again.
- **Settings:** the mic, Whisper and LLM options.

The keyboard drives everything, and the mouse works too: click tabs, key hints and settings rows.

- **Local Whisper** (whisper.cpp) on your GPU through Vulkan, or on the CPU. The models are `large-v3-turbo`
  (default), `large-v3`, `distil-large-v3`, `medium` and `small`, downloaded on first use.
- **Hot mic:** the microphone stays open (shown as *listening*), so recording starts instantly. *RECORDING*
  turns the whole frame red.
- **Mic calibration:** records a test phrase on every input device at once, ranks them by accuracy, level, SNR
  and clipping, and sets an in-app gain so your speech lands in the meter's green zone.
- **Optional LLM pass:** use an OpenAI-compatible local server (LM Studio, Ollama, …), OpenAI, Anthropic or
  Google Gemini. It can show the model's reasoning when the provider exposes it. API keys are stored in your
  system keyring, never in the config file.
- **Plain TOML config** at `~/.config/stt-app/config.toml`. A broken config is backed up, and every valid value
  in it is kept. You then get a report of what was kept, what was reset, and what still needs a value.

Linux only for now (tested on Fedora with KDE Plasma, Wayland, PipeWire and an NVIDIA GPU).

## Build from source (Rust, recommended)

You need Rust 1.88 or newer (`rustup` or your distro's package), a C/C++ toolchain with CMake and Clang (for
whisper.cpp), the ALSA headers, and for the GPU build the Vulkan headers, the Vulkan loader and `glslc`.

**Fedora**
```sh
sudo dnf install rust cargo gcc-c++ cmake clang-devel pkgconf-pkg-config alsa-lib-devel \
    vulkan-headers vulkan-loader-devel glslc git
```

**Ubuntu / Debian**
```sh
sudo apt install build-essential cmake clang libclang-dev pkg-config libasound2-dev \
    libvulkan-dev glslc git
```
(Get Rust from https://rustup.rs if your distro's `cargo` is older than 1.88.)

**Arch**
```sh
sudo pacman -S rust base-devel cmake clang pkgconf alsa-lib vulkan-headers vulkan-icd-loader shaderc git
```

Then build:
```sh
git clone https://github.com/MrPlayerYork/stt-app.git
cd stt-app/rust
cargo build --release -p stt          # GPU (Vulkan) + CPU
./target/release/stt
```

**CPU only** (no Vulkan packages needed):
```sh
cargo build --release -p stt --no-default-features
```

Vulkan from Homebrew instead of distro packages: see `rust/cargo-config.example.toml`.

### At run time
- **A Vulkan driver** for the GPU, e.g. the NVIDIA driver or Mesa. The first GPU start after a build or driver
  update compiles shaders, which takes about 20 s once.
- **A clipboard tool:** `wl-clipboard` on Wayland, or `xclip`/`xsel` on X11.
- **A Secret Service keyring** (KWallet, GNOME Keyring, …), only if you store LLM API keys.
- **About 600 MB of disk** for the default Whisper model (`large-v3-turbo`, q5_0 quantized). Models go to `~/.local/share/stt-app/models` unless
  you choose a folder in Settings › Whisper.

## First run
1. A welcome screen offers **microphone calibration**: read the phrase on screen, then pick the suggested mic.
2. The Whisper model downloads (with progress) and loads in the background.
3. On the Main tab, press **Enter** (or `r`) to record and **Enter** to stop. The text is copied to the clipboard.
   `m` switches the mode (Raw / LLM cleanup / Prompt Mode), `h` opens History, `s` opens Settings, `q` quits.
   **Ctrl+C** cancels a recording.

Mouse capture means Konsole and similar terminals need **Shift+drag** to select text while the app runs.

## The Python version

`src/stt_app/` is the original Python app (Rich UI, faster-whisper). The Rust app was rewritten from it, and it
is kept as the reference implementation. Both read and write the same config file.

```sh
uv run stt-app        # run it (needs a real terminal)
uv run pytest -q      # its tests
```

## How this was built

The Rust rewrite was done in phases with parallel AI coding agents, coordinated through a shared resource board
on a single home PC:
- `RUST_PLAN.md`: the plan and phases.
- `AGENTS.md`: the rules every agent followed.
- `CAVEATS.md`: open questions and the decisions made along the way.
- `UI_MOCKUPS.txt`: the UI layout options, including the chosen one.
- `scratch/whisper-spike/`: the GPU benchmark that decided the go/no-go for the rewrite.

These files refer to the original machine's folders as `<repo>` and `<data-drive>`.

Rust tests (from `rust/`): `cargo test --workspace`. GPU tests:
`cargo test -p stt-whisper --features vulkan --test gpu -- --ignored`, with `STT_MODELS_DIR` and `STT_AUDIO`
set to a models folder and a 16 kHz mono WAV.

## License

MIT, see [LICENSE](LICENSE).
