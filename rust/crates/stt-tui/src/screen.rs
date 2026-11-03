//! The screen contract every tab and overlay implements, and the navigation it returns.
//!
//! Layout (UI_MOCKUPS.txt section D): the runner draws one full-screen rounded frame with the tab bar in
//! the top border and key hints in the bottom border. Inside it draws the active tab, then any overlays
//! (stack, top one gets the keys). An overlay is either full-content (replaces the tab body: recording,
//! calibration, first run, config notice, reasoning view) or a centred popup (pickers, prompts, confirm).

use crossterm::event::KeyEvent;
use ratatui::Frame;
use ratatui::layout::Rect;

use crate::app::{App, Tab};

/// What a key/tick handler wants next.
pub enum Nav {
    /// Not handled: the runner applies its defaults (Tab/Shift+Tab switch tabs when no overlay is open;
    /// Esc pops an overlay; Ctrl+C pops an overlay, or quits from a tab).
    Ignored,
    /// Handled; stay.
    Stay,
    /// Open an overlay on top.
    Push(Box<dyn Screen>),
    /// Close this overlay (no-op for a tab).
    Pop,
    /// Close this overlay and open another in its place.
    Replace(Box<dyn Screen>),
    /// Close every overlay and switch to a tab.
    SwitchTab(Tab),
    /// Leave the app.
    Quit,
}

impl std::fmt::Debug for Nav {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let s = match self {
            Self::Ignored => "Ignored",
            Self::Stay => "Stay",
            Self::Push(_) => "Push",
            Self::Pop => "Pop",
            Self::Replace(_) => "Replace",
            Self::SwitchTab(_) => "SwitchTab",
            Self::Quit => "Quit",
        };
        f.write_str(s)
    }
}

/// How an overlay is placed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Layer {
    /// Replaces the tab body (inside the frame).
    Full,
    /// Centred box of at most `width` x `height` cells over the tab body.
    Popup {
        /// Max width.
        width: u16,
        /// Max height.
        height: u16,
    },
}

/// One key hint for the bottom border, e.g. `("Enter", "record")` → `[Enter] record`.
pub type Hint = (&'static str, String);

/// A tab or overlay.
pub trait Screen {
    /// Draw into `area` (already inside the frame; for a popup, the popup's own area, cleared).
    /// Derive content from `app`; `&mut self` only so screens can update scroll offsets etc.
    fn render(&mut self, app: &App, frame: &mut Frame, area: Rect);

    /// Handle a key press (repeats included, releases filtered out by the runner).
    fn on_key(&mut self, app: &mut App, key: KeyEvent) -> Nav;

    /// Handle pasted text (bracketed paste). Default: ignored. Text inputs must insert it verbatim so a
    /// pasted API key never triggers shortcuts.
    fn on_paste(&mut self, _app: &mut App, _text: &str) -> Nav {
        Nav::Ignored
    }

    /// Called about every 50 ms for the top overlay (or the active tab when there is none), after
    /// `App::tick`. Poll your [`crate::task::Task`]s here.
    fn on_tick(&mut self, _app: &mut App) -> Nav {
        Nav::Stay
    }

    /// Key hints for the bottom border.
    fn hints(&self, _app: &App) -> Vec<Hint> {
        Vec::new()
    }

    /// Overlay placement (ignored for tabs).
    fn layer(&self) -> Layer {
        Layer::Full
    }

    /// Title for an overlay (popup border title, or the frame's breadcrumb for a full overlay).
    fn title(&self, _app: &App) -> String {
        String::new()
    }

    /// True while this screen is recording: the runner turns the whole frame red and shows
    /// "● REC m:ss.t" in the top border.
    fn is_recording(&self, _app: &App) -> bool {
        false
    }
}
