//! Clipboard contract. The real implementation is `stt-clipboard` (subprocess `wl-copy` → `xclip` →
//! `xsel`, like Python); tests use `fakes::FakeClipboard`.

use crate::error::ClipboardError;

/// Which tool performed the copy (shown to the user).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ClipboardMethod {
    /// `wl-copy` (used when `WAYLAND_DISPLAY` is set and it is installed).
    WlCopy,
    /// `xclip -selection clipboard`.
    Xclip,
    /// `xsel --clipboard --input`.
    Xsel,
}

impl ClipboardMethod {
    /// Command name, e.g. "wl-copy".
    pub fn command(self) -> &'static str {
        match self {
            Self::WlCopy => "wl-copy",
            Self::Xclip => "xclip",
            Self::Xsel => "xsel",
        }
    }
}

impl std::fmt::Display for ClipboardMethod {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.command())
    }
}

/// Copies text to the system clipboard.
pub trait Clipboard: Send + Sync {
    /// Replace the clipboard contents with `text`; return the tool used.
    fn copy(&self, text: &str) -> Result<ClipboardMethod, ClipboardError>;
}
