# STT App

Push-to-record speech-to-text terminal app: Whisper transcription → clipboard, with optional LLM cleanup/Prompt Mode.
The Python app (`src/stt_app/`) is complete (milestone `reasoning-working`) and is the reference for a planned
**Rust rewrite**: see [RUST_PLAN.md](RUST_PLAN.md) for the plan and [AGENTS.md](AGENTS.md) for how sub-agents run
(model tiers, worktrees, max 10 at once, shared resource board `tools/agent-board.sh`, home-PC limits).

Test command: `uv run pytest -q`
Run the app (real terminal only): `uv run stt-app`

- Python via `uv` only. User settings live in `~/.config/stt-app/config.toml`; API keys in the system keyring.
- `reference/` is the user's material: gitignored, never commit it.
- Rewrite status: Phases 0-1 done (2026-11-01; Whisper/Vulkan gate passed, user greenlit). Next: RUST_PLAN.md Phase 2.
  Open questions to settle along the way: [CAVEATS.md](CAVEATS.md).
- Rust test command (from `rust/`, heavy slot): `cargo test --workspace`
