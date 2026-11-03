# STT App

Push-to-record speech-to-text terminal app: Whisper transcription → clipboard, with optional LLM cleanup/Prompt Mode.
The Python app (`src/stt_app/`) is complete (milestone `reasoning-working`) and is the reference for a planned
**Rust rewrite**: see [RUST_PLAN.md](RUST_PLAN.md) for the plan and [AGENTS.md](AGENTS.md) for how sub-agents run
(model tiers, worktrees, max 10 at once, shared resource board `tools/agent-board.sh`, home-PC limits).

Test command: `uv run pytest -q`
Run the app (real terminal only): `uv run stt-app`

- Python via `uv` only. User settings live in `~/.config/stt-app/config.toml`; API keys in the system keyring.
- `reference/` is the user's material: gitignored, never commit it.
- Rewrite status: Phases 0-2 done (2026-11-01; gate passed, all 8 crates merged and green). Phase 3 done (2026-11-02): full-screen
  tabbed UI (UI_MOCKUPS.txt section D), all P3 branches merged, workspace green, `stt` builds. Next: Phase 4 (user tests
  `rust/target/debug/stt` in Konsole; parity checklist). **v1.0.1 (2026-11-03)**: fix for a segfault when models opened on the GPU in parallel. **v1.0.0 (2026-11-02)**: user tested the Rust app
  (fresh first-run setup, LLM reasoning), plus live 20 Hz meter with green/yellow/red zones, in-app
  `audio.gain` set by calibration, centred recording screen, mouse support. Next: cleanup before push/archive
  (CAVEATS.md).
  Open questions to settle along the way: [CAVEATS.md](CAVEATS.md).
- Rust test command (from `rust/`, heavy slot): `cargo test --workspace --no-fail-fast`; GPU (gpu slot, needs
  `rust/vk-sdk` symlinks per AGENTS.md): `cargo test -p stt-whisper --features vulkan --test gpu -- --ignored`
