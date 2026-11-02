//! Clipboard copy through a subprocess: `wl-copy` (when `WAYLAND_DISPLAY` is set) → `xclip` → `xsel`,
//! like `src/stt_app/clipboard.py`. `wl-copy` forks a server that inherits our pipes, so stderr goes to
//! a file (never a pipe) and stdout is discarded; we only wait for the tool to exit, never for EOF.
//! Each copy replaces the clipboard.
//!
//! Owns (P2-2): [`SystemClipboard`], implementing [`stt_core::Clipboard`].

use std::fs::{self, File, OpenOptions};
use std::io::{self, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::atomic::{AtomicU64, Ordering};
use std::thread;
use std::time::{Duration, Instant};

use stt_core::{Clipboard, ClipboardError, ClipboardMethod};

/// How long a clipboard tool may run before it is killed (the Python app uses 5 s).
pub const DEFAULT_TIMEOUT: Duration = Duration::from_secs(5);

/// Copies text by running `wl-copy`, `xclip` or `xsel`.
///
/// Tool choice (same as Python): `wl-copy` if `wayland` is set and it is found in the search dirs,
/// else `xclip`, else `xsel`. The search dirs replace `PATH` so tests can use fake tools.
#[derive(Debug, Clone)]
pub struct SystemClipboard {
    wayland: bool,
    search_dirs: Vec<PathBuf>,
    timeout: Duration,
}

impl SystemClipboard {
    /// Use `WAYLAND_DISPLAY` (non-empty) and `PATH` from the environment.
    pub fn from_env() -> Self {
        let wayland = std::env::var_os("WAYLAND_DISPLAY").is_some_and(|v| !v.is_empty());
        let search_dirs = std::env::var_os("PATH")
            .map(|p| std::env::split_paths(&p).collect())
            .unwrap_or_default();
        Self::new(wayland, search_dirs)
    }

    /// Explicit settings: whether Wayland is in use, and the directories to search for tools.
    pub fn new(wayland: bool, search_dirs: Vec<PathBuf>) -> Self {
        Self {
            wayland,
            search_dirs,
            timeout: DEFAULT_TIMEOUT,
        }
    }

    /// Override the tool timeout (default [`DEFAULT_TIMEOUT`]).
    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout = timeout;
        self
    }

    /// First executable named `name` in the search dirs.
    fn find(&self, name: &str) -> Option<PathBuf> {
        self.search_dirs
            .iter()
            .map(|dir| dir.join(name))
            .find(|p| is_executable(p))
    }

    /// Pick the tool to use, with its resolved path.
    fn pick(&self) -> Option<(ClipboardMethod, PathBuf)> {
        if self.wayland
            && let Some(path) = self.find("wl-copy")
        {
            return Some((ClipboardMethod::WlCopy, path));
        }
        if let Some(path) = self.find("xclip") {
            return Some((ClipboardMethod::Xclip, path));
        }
        self.find("xsel").map(|path| (ClipboardMethod::Xsel, path))
    }

    /// Run `method` with `text` on stdin, waiting up to the timeout for it to exit.
    fn run(&self, method: ClipboardMethod, path: &Path, text: &str) -> Result<(), ClipboardError> {
        let tool = method.command();
        let failed = |message: String| ClipboardError::Failed {
            tool: tool.to_owned(),
            message,
        };

        let stderr = stderr_capture().map_err(|e| failed(e.to_string()))?;
        let stderr_child = stderr.try_clone().map_err(|e| failed(e.to_string()))?;
        let mut child = Command::new(path)
            .args(tool_args(method))
            .stdin(Stdio::piped())
            .stdout(Stdio::null())
            .stderr(Stdio::from(stderr_child))
            .spawn()
            .map_err(|e| failed(e.to_string()))?;

        // Feed stdin on a thread so a tool that never reads cannot block past the timeout. Dropping
        // the handle at the end of the thread closes the pipe, which is the EOF the tool waits for.
        if let Some(mut stdin) = child.stdin.take() {
            let input = text.as_bytes().to_vec();
            thread::spawn(move || {
                let _ = stdin.write_all(&input);
            });
        }

        let deadline = Instant::now() + self.timeout;
        let status = loop {
            match child.try_wait() {
                Ok(Some(status)) => break status,
                Ok(None) if Instant::now() >= deadline => {
                    let _ = child.kill();
                    let _ = child.wait();
                    return Err(ClipboardError::Timeout {
                        tool: tool.to_owned(),
                    });
                }
                Ok(None) => thread::sleep(Duration::from_millis(10)),
                Err(e) => return Err(failed(e.to_string())),
            }
        };
        if status.success() {
            return Ok(());
        }

        let mut stderr = stderr;
        let mut message = Vec::new();
        if stderr.seek(SeekFrom::Start(0)).is_ok() {
            let _ = stderr.read_to_end(&mut message);
        }
        let msg = String::from_utf8_lossy(&message).trim().to_owned();
        Err(failed(if msg.is_empty() {
            status.to_string()
        } else {
            msg
        }))
    }
}

impl Clipboard for SystemClipboard {
    fn copy(&self, text: &str) -> Result<ClipboardMethod, ClipboardError> {
        let (method, path) = self.pick().ok_or(ClipboardError::NoTool)?;
        self.run(method, &path, text)?;
        Ok(method)
    }
}

/// Arguments each tool gets (same as the Python app).
fn tool_args(method: ClipboardMethod) -> &'static [&'static str] {
    match method {
        ClipboardMethod::WlCopy => &[],
        ClipboardMethod::Xclip => &["-selection", "clipboard"],
        ClipboardMethod::Xsel => &["--clipboard", "--input"],
    }
}

fn is_executable(path: &Path) -> bool {
    use std::os::unix::fs::PermissionsExt;
    fs::metadata(path).is_ok_and(|m| m.is_file() && m.permissions().mode() & 0o111 != 0)
}

/// A file for a tool's stderr that is already unlinked, so nothing is left behind even on a crash.
fn stderr_capture() -> io::Result<File> {
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let n = COUNTER.fetch_add(1, Ordering::Relaxed);
    let path = std::env::temp_dir().join(format!("stt-clipboard-{}-{n}.err", std::process::id()));
    let file = OpenOptions::new()
        .read(true)
        .write(true)
        .create_new(true)
        .open(&path)?;
    let _ = fs::remove_file(&path);
    Ok(file)
}
