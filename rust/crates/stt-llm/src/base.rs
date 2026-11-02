//! Shared constants and `<think>` handling (Python `llm/base.py`). Owner: P2-6. The constants are fixed
//! here so P2-7 can use them.

use std::time::Duration;

/// Request timeout for every provider (Python `TIMEOUT_S`).
pub const TIMEOUT: Duration = Duration::from_secs(60);

/// Sampling temperature for every provider (Python `TEMPERATURE`).
pub const TEMPERATURE: f32 = 0.2;

/// Find the first valid `<think>` / `<thinking>` (or, with `close`, `</think>` / `</thinking>`) tag at or
/// after `from` in the lowercased text → `(start, end)`.
fn find_tag(lc: &str, from: usize, close: bool) -> Option<(usize, usize)> {
    let pat = if close { "</think" } else { "<think" };
    let mut pos = from;
    while let Some(i) = lc.get(pos..)?.find(pat) {
        let start = pos + i;
        let rest = &lc[start + pat.len()..];
        if rest.starts_with('>') {
            return Some((start, start + pat.len() + 1));
        }
        if rest.starts_with("ing>") {
            return Some((start, start + pat.len() + 4));
        }
        pos = start + pat.len();
    }
    None
}

/// Split inline `<think>…</think>` / `<thinking>…</thinking>` blocks out of `content` →
/// `(answer, reasoning)`, handling an unclosed opening tag and a stray closing tag like Python
/// `split_think`.
pub fn split_think(content: &str) -> (String, String) {
    let mut reasoning: Vec<String> = Vec::new();

    // Closed blocks, case-insensitive. ASCII lowercasing keeps byte offsets identical.
    let lc = content.to_ascii_lowercase();
    let mut out = String::new();
    let mut last = 0;
    let mut pos = 0;
    while let Some((s, e)) = find_tag(&lc, pos, false) {
        let Some((cs, ce)) = find_tag(&lc, e, true) else {
            break;
        };
        out.push_str(&content[last..s]);
        reasoning.push(content[e..cs].trim().to_string());
        last = ce;
        pos = ce;
    }
    out.push_str(&content[last..]);

    let lc = out.to_ascii_lowercase();
    let lead = out.len() - out.trim_start().len();
    if let Some((0, e)) = find_tag(&lc[lead..], 0, false) {
        // Unterminated leading <think>: everything after it is reasoning.
        reasoning.push(out[lead + e..].trim().to_string());
        out.clear();
    } else if let Some((cs, ce)) = find_tag(&lc, 0, true) {
        // Lone </think> with no opening tag (servers that strip the opening tag).
        if !lc[..cs].contains("<think") {
            reasoning.push(out[..cs].trim().to_string());
            out = out[ce..].to_string();
        }
    }
    let joined = reasoning
        .into_iter()
        .filter(|r| !r.is_empty())
        .collect::<Vec<_>>()
        .join("\n\n");
    (out.trim().to_string(), joined)
}

#[cfg(test)]
mod tests {
    use super::split_think;

    fn t(raw: &str, answer: &str, thought: &str) {
        assert_eq!(
            split_think(raw),
            (answer.to_string(), thought.to_string()),
            "{raw:?}"
        );
    }

    #[test]
    fn python_cases() {
        t("<think>plan it</think>\nHello.", "Hello.", "plan it");
        t("<think>never ends", "", "never ends");
        t("just text", "just text", "");
        t("plan it</think>Hello.", "Hello.", "plan it");
        t("<think>a</think>X<think>b</think>Y", "XY", "a\n\nb");
    }

    #[test]
    fn edge_cases() {
        t("<THINKING>Loud</Thinking> ok ", "ok", "Loud");
        t("  <think>lead", "", "lead");
        t("<think></think>Hi", "Hi", "");
        t("answer <think>tail", "answer <think>tail", "");
        t("x <think>y</think> z <think>open", "x  z <think>open", "y");
        t("<thinker>no</thinker>", "<thinker>no</thinker>", "");
        t("é<think>ü</think>ö", "éö", "ü");
        t("", "", "");
    }
}
