//! Blocking HTTP helpers on top of `ureq`: manual redirects so the bearer token can be dropped
//! when a redirect leaves the original host (Hugging Face redirects to a presigned CDN URL).

use std::io::Read;
use std::time::Duration;

use stt_core::DownloadError;
use ureq::Agent;

const MAX_REDIRECTS: usize = 10;

pub(crate) fn agent() -> Agent {
    Agent::new_with_config(
        Agent::config_builder()
            .http_status_as_error(false)
            .max_redirects(0)
            .max_redirects_will_error(false)
            .timeout_connect(Some(Duration::from_secs(30)))
            .timeout_recv_response(Some(Duration::from_secs(60)))
            .user_agent("stt-app/0.1")
            .build(),
    )
}

/// `host:port` part of a URL (what Python compared as `netloc`).
fn authority(url: &str) -> &str {
    let rest = url.split_once("://").map_or(url, |(_, r)| r);
    rest.split(['/', '?', '#']).next().unwrap_or(rest)
}

fn resolve_location(base: &str, loc: &str) -> String {
    if loc.contains("://") {
        return loc.to_string();
    }
    let (scheme, rest) = base.split_once("://").unwrap_or(("http", base));
    let auth = authority(base);
    if let Some(stripped) = loc.strip_prefix("//") {
        return format!("{scheme}://{stripped}");
    }
    if loc.starts_with('/') {
        return format!("{scheme}://{auth}{loc}");
    }
    let path = rest[auth.len()..].split(['?', '#']).next().unwrap_or("");
    let dir = path.rsplit_once('/').map_or("", |(d, _)| d);
    format!("{scheme}://{auth}{dir}/{loc}")
}

/// A successful (2xx) response.
pub(crate) struct Reply {
    pub status: u16,
    pub headers: ureq::http::HeaderMap,
    body: ureq::Body,
}

impl Reply {
    pub fn header(&self, name: &str) -> Option<&str> {
        self.headers.get(name).and_then(|v| v.to_str().ok())
    }
    pub fn into_reader(self) -> impl Read {
        self.body.into_reader()
    }
}

fn hint(status: u16) -> &'static str {
    if status == 404 {
        "not found"
    } else {
        "the repo may be gated, add a Hugging Face token"
    }
}

/// GET or HEAD `url`, following redirects. The token is sent only to the original host.
/// 401/403/404 become [`DownloadError::Http`]; other failures are retryable [`DownloadError::Network`].
pub(crate) fn send(
    agent: &Agent,
    head: bool,
    url: &str,
    token: Option<&str>,
    range: Option<(u64, u64)>,
) -> Result<Reply, DownloadError> {
    let origin = authority(url).to_string();
    let mut cur = url.to_string();
    for _ in 0..=MAX_REDIRECTS {
        let mut rb = if head {
            agent.head(&cur)
        } else {
            agent.get(&cur)
        };
        rb = rb.header("Accept-Encoding", "identity");
        if let Some(t) = token.filter(|_| authority(&cur) == origin) {
            rb = rb.header("Authorization", format!("Bearer {t}"));
        }
        if let Some((a, b)) = range {
            rb = rb.header("Range", format!("bytes={a}-{b}"));
        }
        let resp = rb
            .call()
            .map_err(|e| DownloadError::Network(format!("{cur}: {e}")))?;
        let status = resp.status().as_u16();
        match status {
            200..=299 => {
                let (parts, body) = resp.into_parts();
                return Ok(Reply {
                    status,
                    headers: parts.headers,
                    body,
                });
            }
            301 | 302 | 303 | 307 | 308 => {
                let loc = resp
                    .headers()
                    .get("location")
                    .and_then(|v| v.to_str().ok())
                    .ok_or_else(|| {
                        DownloadError::Network(format!("{cur}: redirect without Location"))
                    })?;
                cur = resolve_location(&cur, loc);
            }
            401 | 403 | 404 => {
                return Err(DownloadError::Http {
                    status,
                    url: cur,
                    message: hint(status).to_string(),
                });
            }
            _ => return Err(DownloadError::Network(format!("HTTP {status} for {cur}"))),
        }
    }
    Err(DownloadError::Network(format!("{url}: too many redirects")))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn locations() {
        assert_eq!(authority("http://a:1/x/y?z"), "a:1");
        assert_eq!(resolve_location("http://a:1/x/y", "/q"), "http://a:1/q");
        assert_eq!(resolve_location("http://a:1/x/y", "z"), "http://a:1/x/z");
        assert_eq!(resolve_location("http://a/x", "https://b/c"), "https://b/c");
    }
}
