//! The shared look (UI_MOCKUPS.txt sections A and D): colours, rounded panels, the level meter, status
//! dots, the spinner and layout helpers. Every screen uses these so the app reads as one thing.

use std::time::Instant;

use ratatui::layout::{Constraint, Flex, Layout, Rect};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, BorderType, Borders, Padding};

/// Frame and panel borders.
pub const BORDER: Color = Color::Blue;
/// Recording: frame, title, mic dot.
pub const REC: Color = Color::Red;
/// Ready / ok / copied.
pub const OK: Color = Color::Green;
/// Busy / warnings.
pub const WARN: Color = Color::Yellow;
/// Errors.
pub const ERR: Color = Color::Red;
/// Key letters in hints and menus.
pub const KEY: Color = Color::Cyan;

/// Labels column ("Mic", "Whisper", ...).
pub fn label() -> Style {
    Style::new().add_modifier(Modifier::BOLD)
}

/// Secondary text (raw transcript, help, reasoning line).
pub fn dim() -> Style {
    Style::new().add_modifier(Modifier::DIM)
}

/// The selected row in lists (with a leading `›`).
pub fn selected() -> Style {
    Style::new().add_modifier(Modifier::REVERSED)
}

/// Marker drawn before the selected row.
pub const CURSOR: &str = "› ";

/// A rounded panel with a title, like the mockups' inner boxes.
pub fn panel(title: impl Into<String>) -> Block<'static> {
    Block::new()
        .borders(Borders::ALL)
        .border_type(BorderType::Rounded)
        .border_style(Style::new().fg(BORDER))
        .title(Span::styled(format!(" {} ", title.into()), label()))
        .padding(Padding::horizontal(1))
}

/// Status dot: `●` in `color`.
pub fn dot(color: Color) -> Span<'static> {
    Span::styled("● ", Style::new().fg(color))
}

/// Hot-mic dot: `◉` (listening, dim green), `●` red when recording, `◌` while opening.
pub fn mic_dot(recording: bool, open: bool) -> Span<'static> {
    match (recording, open) {
        (true, _) => Span::styled("● ", Style::new().fg(REC).add_modifier(Modifier::BOLD)),
        (false, true) => Span::styled("◉ ", Style::new().fg(OK).add_modifier(Modifier::DIM)),
        (false, false) => Span::styled("◌ ", dim()),
    }
}

/// Level meter `▮▮▮▯▯▯` of `width` cells. Speech RMS is small, so the level is scaled by 6 first
/// (as in Python). Red while recording, green otherwise.
pub fn meter(level: f32, width: usize, recording: bool) -> Line<'static> {
    let n = ((level * 6.0).clamp(0.0, 1.0) * width as f32).round() as usize;
    let on = Style::new().fg(if recording { REC } else { OK });
    Line::from(vec![
        Span::styled("▮".repeat(n), on),
        Span::styled("▯".repeat(width - n), dim()),
    ])
}

/// Braille spinner frame for "busy" lines, advancing about 10 times per second.
pub fn spinner(since: Instant) -> &'static str {
    const FRAMES: [&str; 10] = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"];
    FRAMES[(since.elapsed().as_millis() / 100) as usize % FRAMES.len()]
}

/// `[k] label  [k] label` spans for the bottom border.
pub fn hint_spans(hints: &[(&str, String)]) -> Vec<Span<'static>> {
    let mut out = Vec::new();
    for (k, l) in hints {
        out.push(Span::styled(format!("[{k}]"), Style::new().fg(KEY)));
        out.push(Span::raw(format!(" {l}  ")));
    }
    out
}

/// A centred rect of at most `w` x `h` inside `area`.
pub fn centered(area: Rect, w: u16, h: u16) -> Rect {
    let [v] = Layout::vertical([Constraint::Length(h.min(area.height))])
        .flex(Flex::Center)
        .areas(area);
    let [r] = Layout::horizontal([Constraint::Length(w.min(area.width))])
        .flex(Flex::Center)
        .areas(v);
    r
}

/// Truncate to `max` chars with `…`.
pub fn ellipsize(s: &str, max: usize) -> String {
    let one_line = s.replace('\n', " ");
    if one_line.chars().count() <= max {
        return one_line;
    }
    let mut t: String = one_line.chars().take(max.saturating_sub(1)).collect();
    t.push('…');
    t
}

/// One dim status line about the reasoning behind an LLM result (Python `reasoning_line`), e.g.
/// `Reasoning: 1,234 tokens · [v] view`. Used by the Main tab, History and Settings › Test connection.
pub fn reasoning_text(res: &stt_core::LlmResult) -> String {
    let mut line = match (res.reasoning.is_empty(), res.reasoning_tokens, res.reasoned) {
        (false, Some(n), _) => format!("Reasoning: {} tokens · [v] view", thousands(n)),
        (false, None, _) => "Reasoning: captured · [v] view".to_string(),
        (true, Some(n), _) if n > 0 => {
            format!(
                "Reasoning: {} tokens (text not exposed by provider)",
                thousands(n)
            )
        }
        (true, _, Some(false)) => "No reasoning".to_string(),
        _ => "Reasoning: unknown".to_string(),
    };
    if !res.note.is_empty() {
        line.push_str(&format!(" · {}", res.note));
    }
    line
}

/// [`reasoning_text`] as a dim line.
pub fn reasoning_line(res: &stt_core::LlmResult) -> Line<'static> {
    Line::styled(reasoning_text(res), dim())
}

/// `1234567` → `1,234,567`.
pub fn thousands(n: u64) -> String {
    let s = n.to_string();
    let mut out = String::new();
    for (i, c) in s.chars().enumerate() {
        if i > 0 && (s.len() - i).is_multiple_of(3) {
            out.push(',');
        }
        out.push(c);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn res(
        reasoning: &str,
        tokens: Option<u64>,
        reasoned: Option<bool>,
        note: &str,
    ) -> stt_core::LlmResult {
        stt_core::LlmResult {
            text: "x".into(),
            reasoning: reasoning.into(),
            reasoning_tokens: tokens,
            reasoned,
            note: note.into(),
        }
    }

    /// Port of Python `test_ui_reasoning.test_line_variants`.
    #[test]
    fn reasoning_line_variants() {
        assert_eq!(
            reasoning_text(&res("because", Some(1234), Some(true), "")),
            "Reasoning: 1,234 tokens · [v] view"
        );
        assert_eq!(
            reasoning_text(&res("", Some(1234), Some(true), "")),
            "Reasoning: 1,234 tokens (text not exposed by provider)"
        );
        assert_eq!(
            reasoning_text(&res("", None, Some(false), "")),
            "No reasoning"
        );
        assert_eq!(
            reasoning_text(&res("", None, None, "")),
            "Reasoning: unknown"
        );
        assert!(reasoning_text(&res("", None, Some(false), "ignored")).ends_with("· ignored"));
        assert_eq!(thousands(1_234_567), "1,234,567");
    }

    #[test]
    fn meter_fills_and_clamps() {
        let w = |l: Line| {
            l.spans
                .iter()
                .map(|s| s.content.chars().count())
                .sum::<usize>()
        };
        assert_eq!(w(meter(0.0, 10, false)), 10);
        assert_eq!(w(meter(1.0, 10, true)), 10);
        assert_eq!(meter(1.0, 10, true).spans[0].content.chars().count(), 10);
    }

    #[test]
    fn ellipsize_cuts() {
        assert_eq!(ellipsize("abcdef", 4), "abc…");
        assert_eq!(ellipsize("a\nb", 4), "a b");
    }
}
