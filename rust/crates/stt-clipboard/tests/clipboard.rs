//! Tool selection and error handling for `SystemClipboard`, with fake tools (never the real clipboard).
//!
//! Each test writes small shell scripts into its own directory under `CARGO_TARGET_TMPDIR`, the only
//! search dir given to the clipboard. The fakes record their argv and stdin so tests can check them.

use std::fs;
use std::os::unix::fs::PermissionsExt;
use std::path::PathBuf;
use std::sync::{Mutex, MutexGuard};
use std::time::{Duration, Instant};

use stt_clipboard::SystemClipboard;
use stt_core::{Clipboard, ClipboardError, ClipboardMethod};

/// Tests run one at a time. Writing a fake tool while another test thread forks lets the child inherit
/// the script's write fd, and exec then fails with ETXTBSY ("Text file busy"). The suite takes ~0.3 s.
static SERIAL: Mutex<()> = Mutex::new(());

struct Tools {
    dir: PathBuf,
    _serial: MutexGuard<'static, ()>,
}

impl Tools {
    fn new(name: &str) -> Self {
        let serial = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
        let dir = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join(format!("clipboard-{name}"));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        Self {
            dir,
            _serial: serial,
        }
    }

    /// A fake tool that records its argv (one per line) and stdin, then runs `body`.
    fn fake(&self, tool: &str, body: &str) {
        let d = self.dir.display();
        let script = format!(
            "#!/bin/sh\nfor a in \"$@\"; do printf '%s\\n' \"$a\"; done > '{d}/{tool}.args'\n\
             cat > '{d}/{tool}.stdin'\n{body}\n"
        );
        self.write_exec(tool, &script);
    }

    fn write_exec(&self, tool: &str, content: &str) {
        let path = self.dir.join(tool);
        fs::write(&path, content).unwrap();
        fs::set_permissions(&path, fs::Permissions::from_mode(0o755)).unwrap();
    }

    fn was_called(&self, tool: &str) -> bool {
        self.dir.join(format!("{tool}.args")).exists()
    }

    fn args(&self, tool: &str) -> Vec<String> {
        fs::read_to_string(self.dir.join(format!("{tool}.args")))
            .unwrap()
            .lines()
            .map(str::to_owned)
            .collect()
    }

    fn stdin(&self, tool: &str) -> String {
        fs::read_to_string(self.dir.join(format!("{tool}.stdin"))).unwrap()
    }

    fn clipboard(&self, wayland: bool) -> SystemClipboard {
        SystemClipboard::new(wayland, vec![self.dir.clone()])
    }
}

fn failed_tool(err: &ClipboardError) -> Option<(&str, &str)> {
    match err {
        ClipboardError::Failed { tool, message } => Some((tool, message)),
        _ => None,
    }
}

#[test]
fn prefers_wl_copy_on_wayland() {
    let t = Tools::new("prefers-wl-copy");
    for tool in ["wl-copy", "xclip", "xsel"] {
        t.fake(tool, "");
    }
    let method = t.clipboard(true).copy("hi").unwrap();
    assert_eq!(method, ClipboardMethod::WlCopy);
    assert!(t.args("wl-copy").is_empty());
    assert_eq!(t.stdin("wl-copy"), "hi");
    assert!(!t.was_called("xclip"));
}

#[test]
fn wayland_without_wl_copy_falls_back_to_xclip() {
    let t = Tools::new("wayland-fallback");
    t.fake("xclip", "");
    t.fake("xsel", "");
    let method = t.clipboard(true).copy("x").unwrap();
    assert_eq!(method, ClipboardMethod::Xclip);
    assert_eq!(t.args("xclip"), ["-selection", "clipboard"]);
}

#[test]
fn x11_prefers_xclip_over_xsel() {
    let t = Tools::new("x11-order");
    for tool in ["wl-copy", "xclip", "xsel"] {
        t.fake(tool, "");
    }
    // Not Wayland, so wl-copy is skipped even though it is installed.
    let method = t.clipboard(false).copy("x").unwrap();
    assert_eq!(method, ClipboardMethod::Xclip);
    assert_eq!(t.args("xclip"), ["-selection", "clipboard"]);
    assert!(!t.was_called("wl-copy"));
}

#[test]
fn falls_back_to_xsel() {
    let t = Tools::new("xsel-only");
    t.fake("xsel", "");
    let method = t.clipboard(false).copy("x").unwrap();
    assert_eq!(method, ClipboardMethod::Xsel);
    assert_eq!(t.args("xsel"), ["--clipboard", "--input"]);
    assert_eq!(t.stdin("xsel"), "x");
}

#[test]
fn wl_copy_ignored_without_wayland() {
    let t = Tools::new("wl-copy-ignored");
    t.fake("wl-copy", "");
    let err = t.clipboard(false).copy("x").unwrap_err();
    assert!(matches!(err, ClipboardError::NoTool));
    assert!(!t.was_called("wl-copy"));
}

#[test]
fn no_tool_is_no_tool_error_and_runs_nothing() {
    let t = Tools::new("no-tool");
    let err = t.clipboard(true).copy("x").unwrap_err();
    assert!(matches!(err, ClipboardError::NoTool));
    assert!(err.to_string().to_lowercase().contains("no clipboard tool"));
    assert!(fs::read_dir(&t.dir).unwrap().next().is_none());
}

#[test]
fn non_executable_file_is_not_a_tool() {
    let t = Tools::new("not-executable");
    fs::write(t.dir.join("xclip"), "#!/bin/sh\nexit 0\n").unwrap();
    fs::set_permissions(t.dir.join("xclip"), fs::Permissions::from_mode(0o644)).unwrap();
    assert!(matches!(
        t.clipboard(false).copy("x"),
        Err(ClipboardError::NoTool)
    ));
}

#[test]
fn nonzero_exit_reports_stderr() {
    let t = Tools::new("nonzero");
    t.fake("xclip", "echo 'cannot open display' >&2\nexit 1");
    let err = t.clipboard(false).copy("x").unwrap_err();
    assert_eq!(failed_tool(&err), Some(("xclip", "cannot open display")));
    assert_eq!(err.to_string(), "xclip failed: cannot open display");
}

#[test]
fn nonzero_exit_without_stderr_reports_status() {
    let t = Tools::new("nonzero-silent");
    t.fake("xsel", "exit 3");
    let err = t.clipboard(false).copy("x").unwrap_err();
    let (tool, message) = failed_tool(&err).unwrap();
    assert_eq!(tool, "xsel");
    assert!(message.contains('3'), "message was {message:?}");
}

#[test]
fn timeout_kills_tool_and_raises_timeout() {
    let t = Tools::new("timeout");
    t.fake("xsel", "exec sleep 10");
    let start = Instant::now();
    let err = t
        .clipboard(false)
        .with_timeout(Duration::from_millis(300))
        .copy("x")
        .unwrap_err();
    assert!(
        matches!(&err, ClipboardError::Timeout { tool } if tool == "xsel"),
        "got {err:?}"
    );
    assert_eq!(err.to_string(), "xsel timed out");
    assert!(start.elapsed() < Duration::from_secs(5));
}

#[test]
fn spawn_failure_raises_failed() {
    let t = Tools::new("spawn-failure");
    // Executable bit set, but the interpreter does not exist, so exec fails like Python's OSError.
    t.write_exec("xclip", "#!/nonexistent/interpreter\n");
    let err = t.clipboard(false).copy("x").unwrap_err();
    let (tool, _) = failed_tool(&err).expect("expected Failed");
    assert_eq!(tool, "xclip");
}

#[test]
fn wl_copy_returns_while_forked_server_keeps_running() {
    let t = Tools::new("wl-copy-fork");
    // Like real wl-copy: read stdin to EOF, fork a server that keeps the inherited fds open, then exit.
    t.fake("wl-copy", "(sleep 3) &\nexit 0");
    let start = Instant::now();
    let method = t.clipboard(true).copy("forked").unwrap();
    assert_eq!(method, ClipboardMethod::WlCopy);
    assert!(
        start.elapsed() < Duration::from_secs(2),
        "copy waited for the forked server"
    );
    assert_eq!(t.stdin("wl-copy"), "forked");
}

#[test]
fn large_text_is_delivered_in_full() {
    let t = Tools::new("large");
    t.fake("xclip", "");
    let text = "abcdefghij\n".repeat(50_000); // well past the 64 KiB pipe buffer
    t.clipboard(false).copy(&text).unwrap();
    assert_eq!(t.stdin("xclip"), text);
}

#[test]
fn each_copy_runs_the_tool_with_only_its_text() {
    let t = Tools::new("replace");
    t.fake("xclip", "");
    let cb = t.clipboard(false);
    cb.copy("one").unwrap();
    cb.copy("two").unwrap();
    assert_eq!(t.stdin("xclip"), "two");
}

#[test]
#[ignore = "touches the real clipboard; run by hand only"]
fn real_clipboard_round_trip() {
    let cb = SystemClipboard::from_env();
    cb.copy("stt-clipboard test").unwrap();
}
