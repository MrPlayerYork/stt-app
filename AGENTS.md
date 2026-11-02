# Running Sub-Agents on This Project

This is a **home PC**, not a server: Ryzen 7 3700X (16 threads), 31 GB RAM, RTX 4070 SUPER 12 GB (LM Studio often
holds most of the VRAM), Bazzite (atomic Fedora). The user is working on it at the same time. Every agent shares
it, so every agent follows the rules below. The plan they serve is [RUST_PLAN.md](RUST_PLAN.md).

## Part 1 — For the orchestrator (main session)

### Limits
- **At most 10 agents at once** (user's limit). Count running ones before launching more.
- Pick the cheapest model that will get it right (the user is paying in usage):
  - **haiku** — mechanical, well-specified work: ports of pure functions, tests from an existing spec, small crates.
  - **sonnet** — the default for real implementation: I/O, protocols, UI, anything needing judgement.
  - **opus** — only where mistakes multiply: shared contracts/architecture (P1-a), hard debugging that sonnet failed.
- Always pass `model` explicitly. (`~/.claude/settings.json` sets `CLAUDE_CODE_SUBAGENT_MODEL=sonnet` as the fallback.)

### Launching
- One worktree per agent (`isolation: "worktree"`), `run_in_background: true`, parallel launches in one message.
- Integration branch for the rewrite: **`rust-rewrite`**. Worktrees may be created from an older commit, so every
  prompt starts with: `git merge --ff-only rust-rewrite` and a check that the expected commit is present.
- Give each agent a self-contained prompt: the files/crate it **owns** (and must not leave), the contracts it codes
  against (paste them; don't make it explore), the tests that define done, and the report format below.
- Paste the **Agent rules** section (Part 2) into every prompt, or tell the agent to read `AGENTS.md` Part 2 first.

### Integrating
- Merge finished branches into `rust-rewrite` with `--no-ff`, run `cargo test --workspace` in the heavy slot,
  fix shims/glue yourself if small, otherwise send the fix back to the owning agent (SendMessage) or a new one.
- Milestones (`*-working` tags) only after the user confirms something works (see global CLAUDE.md).

### Watching resources
- `tools/agent-board.sh status` shows slot holders, load, free RAM and VRAM, and recent board messages.
  Check it before launching a wave and when agents seem stuck.

## Part 2 — Agent rules (every sub-agent follows these)

### Environment
- Rust: `cargo` from `~/.cargo/bin`. Python reference app: run only via `uv run ...` (never bare `python`/`pip`).
- **Never** `dnf`/`rpm-ostree`/`sudo`. Missing tools → stop and report; the orchestrator asks the user about `brew`.
- Scratch work goes in `scratch/` inside your worktree, never `/tmp`. Delete it when done.
- Stay inside the crate/files you were assigned. Don't edit other crates, `Cargo.lock` conflicts aside.
- No network except: crates.io via cargo, Hugging Face model downloads (in the `download` slot), and docs lookups.
  Never call paid LLM APIs; never load/unload models on the user's running LM Studio/Ollama.

### The shared board (`tools/agent-board.sh`)
One board for all agents, stored in the **main checkout** (`.agents/`, gitignored), so it's shared across worktrees.
Use the script path from your worktree (`tools/agent-board.sh`); it always writes to the shared location.
Use your agent label (e.g. `P2-4`) as `AGENT`.

| Slot | Capacity | Claim it for |
|---|---|---|
| `heavy` | 1 | Anything compiling whisper.cpp (`stt-whisper`, the `stt` binary, `--workspace` builds), `--release` builds |
| `gpu` | 1 | Any test or run that does Whisper inference on the GPU |
| `download` | 1 | Any download over ~200 MB (Whisper models) |

```bash
B=tools/agent-board.sh
$B status                                   # look before doing anything expensive
$B run heavy P2-8 "stt-whisper vulkan build" -- env CARGO_BUILD_JOBS=8 cargo build -p stt-whisper
$B run gpu   P2-8 "turbo q5 gpu test"        -- cargo test -p stt-whisper -- --ignored
$B note P2-6 "stt-llm: I own src/openai*.rs, P2-7 owns anthropic/google"   # short coordination messages
$B wait-load                                # block until load < 12 and >= 4 GB RAM free
```
- `run` claims (waiting up to 30 min if held), runs the command, and **always releases**, even on failure.
  Prefer it over manual `claim`/`release`. If you must `claim`, `release` in the same step you finish.
- Claims expire automatically (heavy 60 min, gpu 20, download 45) if an agent dies; the takeover is logged.
- Post a `note` when you start something long, change a shared contract, or are blocked. Keep notes one line.

### Builds on a shared PC
- Light work (your own crate only): `$B wait-load` first, then `cargo build/test -p <your-crate>`. The workspace
  config sets `jobs = 2`; don't override it outside the heavy slot (10 agents × 2 jobs already fills 16 threads).
- Never `cargo build --workspace` or `--release` outside the `heavy` slot. Inside it, `CARGO_BUILD_JOBS=8` is the max.
- Don't depend on `stt-whisper` from any other crate; use the traits in `stt-core` and fakes in tests.
- Shared model files: download into the shared models dir the orchestrator gives you (never per-worktree), and
  check whether the file already exists before claiming `download`. whisper.cpp ggml files go in
  `<data-drive>/stt-models`.
- **Downloads use the multi-connection downloader, never single-connection `curl`/`wget`.** Until `stt-download`
  is ported, use the Python app's `download_file()` (a port of `reference/hf_downloader.py`: 8 connections,
  resume via `.part` + `.progress.json`) through the main checkout's venv, in the `download` slot:
  `uv run --project <main checkout> python -I -c "from pathlib import Path; from stt_app.download import download_file; download_file('<url>', Path('<dest>'), connections=8)"`.
  It skips any existing `dest`, so delete partial files left by other tools first. Don't run
  `reference/hf_downloader.py` itself (it writes a config into `reference/` and nests files by publisher).
- Don't run long-lived processes; stop anything you start. Use `timeout` on test runs that could hang.

### Building whisper.cpp with Vulkan (`stt-whisper`, the `stt` binary, `scratch/whisper-spike`)
Proven in P1-b (`scratch/whisper-spike/RESULTS.md`). Without these fixes the build fails or links wrong:
- Use **`whisper-rs` 0.16+** (feature `vulkan`). 0.14 fails with clang 22 (bindgen `_IO_FILE` layout assertions).
- Brew doesn't link `vulkan-headers` into its prefix, so give CMake a fake SDK dir. Create it once per worktree, next
  to the `.cargo/` dir that uses it (it's gitignored; add `vk-sdk/` to `.gitignore` if it isn't there yet):
  `mkdir -p vk-sdk && ln -sfn /home/linuxbrew/.linuxbrew/opt/vulkan-headers/include vk-sdk/include && ln -sfn /home/linuxbrew/.linuxbrew/lib vk-sdk/lib`
  (`opt/` is brew's version-independent path, so this survives brew upgrades.)
- Add to that `.cargo/config.toml` (on top of the jobs/sccache/Ninja settings already there):
  ```toml
  [build]
  rustflags = ["-L", "/home/linuxbrew/.linuxbrew/lib", "-C", "link-arg=-Wl,-rpath,/home/linuxbrew/.linuxbrew/lib"]
  [env]
  VULKAN_SDK = { value = "vk-sdk", relative = true }
  CMAKE_INCLUDE_PATH = "/home/linuxbrew/.linuxbrew/opt/vulkan-headers/include"
  CMAKE_LIBRARY_PATH = "/home/linuxbrew/.linuxbrew/lib"
  ```
  The rpath makes `-lvulkan` link and load at run time.
- Cargo only reads `.cargo/config.toml` from the cwd upward: run cargo **from inside `rust/`** (or the spike dir), never
  with `--manifest-path` from the repo root, or none of this config (or sccache) applies.
- A failed first try can leave a stale CMake `NOTFOUND` cache: delete `target/*/build/whisper-rs-sys-*/out/build`, then retry.
- Check the run log shows `ggml_vulkan: 0 = NVIDIA GeForce RTX 4070 SUPER` and `using Vulkan0 backend`, not `llvmpipe`.
- The first GPU run after a fresh build compiles shaders (about 16 s once); time warm runs, not that one.

### Verifying and reporting
- Done = your crate's tests pass (`cargo test -p <crate>`), `cargo clippy -p <crate> --all-targets` is clean, `cargo fmt -p <crate>`
  has been run (only on your crate), plus any real-world
  check your prompt asks for (in the right slot).
- Commit on your branch: short message, blank line, `Co-Authored-By: <your model> <noreply@anthropic.com>`.
- Final report, **under 12 lines**: branch, commit, what's done, how it was verified, contract changes, open issues.
  The report is all the orchestrator reads, so make it complete and don't pad it.
