//! Clipboard copy through a subprocess: `wl-copy` (when `WAYLAND_DISPLAY` is set) → `xclip` → `xsel`,
//! like `src/stt_app/clipboard.py`. `wl-copy` forks a server that inherits our pipes, so stderr must go to
//! a file, not a pipe. Each copy replaces the clipboard.
//!
//! Owns (P2-2): `copy(text) -> ClipboardMethod`, implementing [`stt_core::Clipboard`].
