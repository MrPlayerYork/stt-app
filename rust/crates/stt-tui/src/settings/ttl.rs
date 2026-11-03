//! Idle-TTL text helpers (ports of Python `parse_duration` and `ttl_label`).

/// `"120"`, `"90s"`, `"2m"`, `"1h"` → seconds; `None` if unparseable, negative or not finite.
pub fn parse_duration(text: &str) -> Option<u64> {
    let t = text.trim().to_lowercase();
    let (num, mult) = match t.chars().last() {
        Some('s') => (&t[..t.len() - 1], 1.0),
        Some('m') => (&t[..t.len() - 1], 60.0),
        Some('h') => (&t[..t.len() - 1], 3600.0),
        _ => (t.as_str(), 1.0),
    };
    let v: f64 = num.trim().parse().ok()?;
    (v.is_finite() && v >= 0.0).then_some((v * mult) as u64)
}

/// `0` → "server default"; whole minutes → "2m"; otherwise "90s".
pub fn ttl_label(secs: u64) -> String {
    if secs == 0 {
        "server default".into()
    } else if secs.is_multiple_of(60) {
        format!("{}m", secs / 60)
    } else {
        format!("{secs}s")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_duration_cases() {
        assert_eq!(parse_duration("120"), Some(120));
        assert_eq!(parse_duration("90s"), Some(90));
        assert_eq!(parse_duration("2m"), Some(120));
        assert_eq!(parse_duration(" 1H "), Some(3600));
        assert_eq!(parse_duration("0"), Some(0));
        assert_eq!(parse_duration("1.5m"), Some(90));
        assert_eq!(parse_duration("-5"), None);
        assert_eq!(parse_duration("abc"), None);
        assert_eq!(parse_duration(""), None);
        assert_eq!(parse_duration("m"), None);
        assert_eq!(parse_duration("inf"), None);
    }

    #[test]
    fn ttl_label_cases() {
        assert_eq!(ttl_label(0), "server default");
        assert_eq!(ttl_label(120), "2m");
        assert_eq!(ttl_label(90), "90s");
    }
}
