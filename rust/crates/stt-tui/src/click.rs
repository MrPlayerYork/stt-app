//! Mouse support: a click map rebuilt on every draw. Whatever is drawn as clickable registers its
//! rectangle here; a left click finds the topmost hit and the runner turns it into the same action the
//! keyboard would take, so mouse and keys never disagree.
//!
//! The map lives in a thread-local because drawing happens on the UI thread only and passing it through
//! every `render` would touch every screen. Only the top screen's regions are kept: the runner calls
//! [`begin_screen`] before rendering each screen, which drops the regions of the screens below.

use std::cell::RefCell;

use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};
use ratatui::layout::{Position, Rect};

use crate::app::Tab;

/// What a click does.
#[derive(Debug, Clone, PartialEq)]
pub enum Click {
    /// Press this key (routed exactly like a key press).
    Key(KeyEvent),
    /// Switch to this tab.
    Tab(Tab),
    /// Screen-defined action, passed to the top screen's [`crate::screen::Screen::on_click`]
    /// (e.g. a row index).
    Tag(usize),
}

#[derive(Default)]
struct Map {
    /// Frame regions (tabs, bottom-border hints).
    frame: Vec<(Rect, Click)>,
    /// The top screen's regions.
    screen: Vec<(Rect, Click)>,
}

thread_local! {
    static MAP: RefCell<Map> = RefCell::new(Map::default());
}

/// Start a new frame: forget every region.
pub fn clear() {
    MAP.with(|m| *m.borrow_mut() = Map::default());
}

/// A screen is about to render: drop the regions of the screen drawn before it (it is covered).
pub fn begin_screen() {
    MAP.with(|m| m.borrow_mut().screen.clear());
}

/// Register a frame region (runner only).
pub(crate) fn add_frame(area: Rect, click: Click) {
    MAP.with(|m| m.borrow_mut().frame.push((area, click)));
}

/// Register a clickable region of the screen being rendered.
pub fn add(area: Rect, click: Click) {
    if area.width > 0 && area.height > 0 {
        MAP.with(|m| m.borrow_mut().screen.push((area, click)));
    }
}

/// Register row `i` of a list drawn from `area.y` (one line per row) as `Click::Tag(tag)`, if visible.
pub fn add_row(area: Rect, line: usize, tag: usize) {
    if line < area.height as usize {
        add(
            Rect {
                y: area.y + line as u16,
                height: 1,
                ..area
            },
            Click::Tag(tag),
        );
    }
}

/// The topmost region under (`x`, `y`): the screen's own regions first, then the frame's, latest
/// registered first.
pub fn hit(x: u16, y: u16) -> Option<Click> {
    let p = Position { x, y };
    MAP.with(|m| {
        let m = m.borrow();
        m.screen
            .iter()
            .rev()
            .chain(m.frame.iter().rev())
            .find(|(r, _)| r.contains(p))
            .map(|(_, c)| c.clone())
    })
}

/// The key a hint label stands for: `"Enter"`, `"Enter/c"` (first form), `"Esc"`, `"Ctrl+C"`, `"Tab"`,
/// or one character. `None` for ranges like `"↑↓"` or `"1-9"`, which aren't buttons.
pub fn key_for_hint(label: &str) -> Option<KeyEvent> {
    let first = label.split('/').next().unwrap_or(label);
    let plain = |code| Some(KeyEvent::new(code, KeyModifiers::NONE));
    match first {
        "Enter" => plain(KeyCode::Enter),
        "Esc" => plain(KeyCode::Esc),
        "Tab" => plain(KeyCode::Tab),
        "Space" => plain(KeyCode::Char(' ')),
        "PgUp" => plain(KeyCode::PageUp),
        "PgDn" => plain(KeyCode::PageDown),
        _ => {
            if let Some(c) = first.strip_prefix("Ctrl+") {
                let mut ch = c.chars();
                return match (ch.next(), ch.next()) {
                    (Some(c), None) => Some(KeyEvent::new(
                        KeyCode::Char(c.to_ascii_lowercase()),
                        KeyModifiers::CONTROL,
                    )),
                    _ => None,
                };
            }
            let mut ch = first.chars();
            match (ch.next(), ch.next()) {
                (Some(c), None) if c.is_ascii_alphanumeric() => plain(KeyCode::Char(c)),
                _ => None,
            }
        }
    }
}

/// Register the `[k] label  ` spans of [`crate::theme::hint_spans`] drawn on row `y` starting at `x`
/// (the frame calls this for the bottom border; screens can use it for inline key lines). Returns the
/// x after the last hint.
pub fn add_hints(x: u16, y: u16, hints: &[(&str, String)], frame_level: bool) -> u16 {
    let mut x = x;
    for (k, l) in hints {
        // "[k]" + " label" is the button; the two trailing spaces are not.
        let w = (k.chars().count() + 2 + 1 + l.chars().count()) as u16;
        if let Some(key) = key_for_hint(k) {
            let r = Rect::new(x, y, w, 1);
            if frame_level {
                add_frame(r, Click::Key(key));
            } else {
                add(r, Click::Key(key));
            }
        }
        x += w + 2;
    }
    x
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hint_labels_map_to_keys() {
        assert_eq!(key_for_hint("Enter/c").unwrap().code, KeyCode::Enter);
        assert_eq!(key_for_hint("q").unwrap().code, KeyCode::Char('q'));
        let c = key_for_hint("Ctrl+C").unwrap();
        assert_eq!(
            (c.code, c.modifiers),
            (KeyCode::Char('c'), KeyModifiers::CONTROL)
        );
        assert!(key_for_hint("↑↓").is_none());
        assert!(key_for_hint("1-9").is_none());
    }

    #[test]
    fn top_screen_wins_and_covered_screens_drop_out() {
        clear();
        add_frame(Rect::new(0, 0, 10, 1), Click::Tab(Tab::Main));
        begin_screen();
        add(Rect::new(0, 5, 10, 1), Click::Tag(1));
        begin_screen(); // a popup covers the first screen
        add(Rect::new(0, 0, 10, 1), Click::Tag(2));
        assert_eq!(hit(3, 5), None);
        assert_eq!(hit(3, 0), Some(Click::Tag(2)));
        clear();
        assert_eq!(hit(3, 0), None);
    }
}
