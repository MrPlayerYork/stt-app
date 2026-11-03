//! Test helpers shared by every module's tests: an [`App`] wired to `stt_core::fakes`, key builders, and
//! rendering a screen or the whole UI to a plain string.
#![allow(dead_code)] // helpers for the Phase 3 modules' tests

use std::sync::Arc;
use std::time::Duration;

use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};
use ratatui::Terminal;
use ratatui::backend::TestBackend;
use ratatui::layout::Rect;
use stt_core::AppConfig;
use stt_core::fakes::{
    FakeAudioSource, FakeClipboard, FakeLlm, FakeLlmFactory, FakeTranscriberFactory,
    MemorySecretStore,
};

use crate::app::{App, MemoryConfigStore, Services};
use crate::runner::Ui;
use crate::screen::Screen;

/// Handles to the fakes behind a test [`App`], for assertions.
pub struct Handles {
    /// Microphones ("Test Mic", returns 1 s of 0.1, level 0.2).
    pub audio: FakeAudioSource,
    /// Transcriber factory (every transcriber answers "hello world").
    pub transcribers: Arc<FakeTranscriberFactory>,
    /// LLM factory (answers "Hello, world.").
    pub llm: FakeLlmFactory,
    /// Clipboard.
    pub clipboard: FakeClipboard,
    /// Keyring.
    pub secrets: Arc<MemorySecretStore>,
    /// Saved configs.
    pub config: MemoryConfigStore,
}

/// An app with default config and fresh fakes (model and mic not started).
pub fn test_app() -> (App, Handles) {
    test_app_with(AppConfig::default(), FakeLlm::new("Hello, world."))
}

/// An app with `cfg` and `llm`.
pub fn test_app_with(cfg: AppConfig, llm: FakeLlm) -> (App, Handles) {
    let h = Handles {
        audio: FakeAudioSource::new()
            .device("Test Mic", false, vec![0.1; 16_000])
            .level(0.2),
        transcribers: Arc::new(FakeTranscriberFactory::new("hello world")),
        llm: FakeLlmFactory::new(llm),
        clipboard: FakeClipboard::new(),
        secrets: Arc::new(MemorySecretStore::new()),
        config: MemoryConfigStore::default(),
    };
    let services = Services {
        audio: Arc::new(h.audio.clone()),
        transcribers: h.transcribers.clone(),
        llm: Arc::new(h.llm.clone()),
        clipboard: Arc::new(h.clipboard.clone()),
        secrets: h.secrets.clone(),
        config: Arc::new(h.config.clone()),
    };
    (App::new(cfg, services), h)
}

/// Start the model and mic and tick until both settle (up to ~2 s).
pub fn settle(app: &mut App) {
    if app.model.transcriber().is_none() {
        app.start_loading();
    }
    if matches!(app.mic.status(), crate::mic::MicStatus::Closed) {
        app.open_mic();
    }
    for _ in 0..400 {
        app.tick();
        if !app.model.is_busy() && !matches!(app.mic.status(), crate::mic::MicStatus::Opening) {
            return;
        }
        std::thread::sleep(Duration::from_millis(5));
    }
}

/// A plain key press.
pub fn key(code: KeyCode) -> KeyEvent {
    KeyEvent::new(code, KeyModifiers::NONE)
}

/// A character key.
pub fn ch(c: char) -> KeyEvent {
    key(KeyCode::Char(c))
}

/// Ctrl + character.
pub fn ctrl(c: char) -> KeyEvent {
    KeyEvent::new(KeyCode::Char(c), KeyModifiers::CONTROL)
}

fn buffer_text(t: &Terminal<TestBackend>) -> String {
    let buf = t.backend().buffer();
    let mut out = String::new();
    for y in 0..buf.area.height {
        for x in 0..buf.area.width {
            out.push_str(buf[(x, y)].symbol());
        }
        out.push('\n');
    }
    out
}

/// Render one screen into a `w` x `h` area and return the text (one line per row).
pub fn render_screen(screen: &mut dyn Screen, app: &App, w: u16, h: u16) -> String {
    let mut t = Terminal::new(TestBackend::new(w, h)).expect("test terminal");
    t.draw(|f| screen.render(app, f, Rect::new(0, 0, w, h)))
        .expect("draw");
    buffer_text(&t)
}

/// Render the whole UI (frame, tab, overlays) and return the text.
pub fn render_ui(ui: &mut Ui, app: &App, w: u16, h: u16) -> String {
    let mut t = Terminal::new(TestBackend::new(w, h)).expect("test terminal");
    t.draw(|f| ui.draw(app, f)).expect("draw");
    buffer_text(&t)
}
