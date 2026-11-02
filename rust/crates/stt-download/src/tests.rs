//! Port of `tests/test_download.py` against a local fake HTTP server (std `TcpListener`, no real network).

use std::collections::{HashMap, HashSet};
use std::io::{Read, Write};
use std::net::{Shutdown, TcpListener, TcpStream};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Duration;

use stt_core::{DownloadError, DownloadProgress, DownloadProgressFn};

use crate::engine::{download, list_repo_files_at};
use crate::{Options, hf_resolve_url};

const REPO: &str = "acme/whisper-test";

fn big() -> Vec<u8> {
    let mut x: u32 = 12345;
    (0..10 * 1024 * 1024)
        .map(|_| {
            x = x.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            (x >> 24) as u8
        })
        .collect()
}

fn fast() -> Options {
    Options {
        retry_backoff: Duration::ZERO,
        ..Options::default()
    }
}

type Req = (String, String, Option<(u64, u64)>, Option<String>);

#[derive(Default)]
struct Cfg {
    files: HashMap<String, Vec<u8>>,
    no_ranges: bool,
    hide_length: bool,
    /// Paths whose first ranged GET is cut short.
    fail_once: HashSet<String>,
    /// Ranged GETs starting at this byte are always cut short.
    fail_start: Option<u64>,
    require_token: Option<String>,
    redirects: HashMap<String, String>,
    /// `(method, path, range, authorization)`
    log: Vec<Req>,
}

struct Server {
    base: String,
    cfg: Arc<Mutex<Cfg>>,
}

impl Server {
    fn start(cfg: Cfg) -> Server {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let base = format!("http://127.0.0.1:{}", listener.local_addr().unwrap().port());
        let cfg = Arc::new(Mutex::new(cfg));
        let shared = cfg.clone();
        thread::spawn(move || {
            for conn in listener.incoming().flatten() {
                let cfg = shared.clone();
                thread::spawn(move || handle(conn, &cfg));
            }
        });
        Server { base, cfg }
    }
    fn with_files(files: &[(&str, Vec<u8>)]) -> Server {
        Server::start(Cfg {
            files: files
                .iter()
                .map(|(n, d)| (format!("/{n}"), d.clone()))
                .collect(),
            ..Cfg::default()
        })
    }
    fn url(&self, name: &str) -> String {
        format!("{}/{name}", self.base)
    }
    fn log(&self) -> Vec<Req> {
        self.cfg.lock().unwrap().log.clone()
    }
}

fn handle(mut s: TcpStream, cfg: &Mutex<Cfg>) {
    let mut buf = Vec::new();
    let mut byte = [0u8; 1];
    while !buf.ends_with(b"\r\n\r\n") {
        if s.read(&mut byte).unwrap_or(0) == 0 {
            return;
        }
        buf.push(byte[0]);
    }
    let text = String::from_utf8_lossy(&buf).into_owned();
    let mut lines = text.lines();
    let mut first = lines.next().unwrap_or("").split(' ');
    let (method, path) = (
        first.next().unwrap_or("").to_string(),
        first.next().unwrap_or("").to_string(),
    );
    let header = |name: &str| {
        text.lines().find_map(|l| {
            l.split_once(": ")
                .filter(|(k, _)| k.eq_ignore_ascii_case(name))
                .map(|(_, v)| v.to_string())
        })
    };
    let range = header("range").and_then(|r| {
        let (a, b) = r.strip_prefix("bytes=")?.split_once('-')?;
        Some((a.parse::<u64>().ok()?, b.parse::<u64>().ok()?))
    });
    let auth = header("authorization");

    let mut c = cfg.lock().unwrap();
    c.log
        .push((method.clone(), path.clone(), range, auth.clone()));
    if let Some(tok) = &c.require_token
        && auth.as_deref() != Some(&format!("Bearer {tok}"))
    {
        let _ = s.write_all(
            b"HTTP/1.1 401 Unauthorized\r\nContent-Length: 0\r\nConnection: close\r\n\r\n",
        );
        return;
    }
    if let Some(to) = c.redirects.get(&path) {
        let _ = s.write_all(format!("HTTP/1.1 302 Found\r\nLocation: {to}\r\nContent-Length: 0\r\nConnection: close\r\n\r\n").as_bytes());
        return;
    }
    let Some(data) = c.files.get(&path).cloned() else {
        let _ = s
            .write_all(b"HTTP/1.1 404 Not Found\r\nContent-Length: 0\r\nConnection: close\r\n\r\n");
        return;
    };
    let ranged = range.filter(|_| !c.no_ranges);
    let (status, body, extra) = match ranged {
        Some((a, b)) => {
            let b = b.min(data.len() as u64 - 1);
            (
                "206 Partial Content",
                data[a as usize..=b as usize].to_vec(),
                format!("Content-Range: bytes {a}-{b}/{}\r\n", data.len()),
            )
        }
        None => ("200 OK", data, String::new()),
    };
    let cut = method == "GET"
        && body.len() > 1024
        && (c.fail_once.remove(&path) && ranged.is_some()
            || c.fail_start
                .is_some_and(|f| ranged.is_some_and(|r| r.0 >= f && r.0 < f + 2 * 1024 * 1024)));
    let accept = if c.no_ranges {
        ""
    } else {
        "Accept-Ranges: bytes\r\n"
    };
    let length = if c.hide_length {
        String::new()
    } else {
        format!("Content-Length: {}\r\n", body.len())
    };
    let hide = c.hide_length;
    drop(c);
    let head = format!("HTTP/1.1 {status}\r\n{length}{accept}{extra}Connection: close\r\n\r\n");
    let _ = s.write_all(head.as_bytes());
    if method == "GET" {
        let _ = s.write_all(&body[..if cut { body.len() / 3 } else { body.len() }]);
    }
    let _ = s.flush();
    if cut || hide {
        let _ = s.shutdown(Shutdown::Both);
    }
}

struct TmpDir(PathBuf);
impl TmpDir {
    fn new(tag: &str) -> TmpDir {
        let p = std::env::temp_dir().join(format!("stt-download-{}-{tag}", std::process::id()));
        let _ = std::fs::remove_dir_all(&p);
        std::fs::create_dir_all(&p).unwrap();
        TmpDir(p)
    }
}
impl Drop for TmpDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

fn recorder() -> (DownloadProgressFn, Arc<Mutex<Vec<DownloadProgress>>>) {
    let seen = Arc::new(Mutex::new(Vec::new()));
    let s2 = seen.clone();
    (
        Arc::new(move |p: &DownloadProgress| s2.lock().unwrap().push(p.clone())),
        seen,
    )
}

fn names(dir: &Path) -> Vec<String> {
    let mut v: Vec<_> = std::fs::read_dir(dir)
        .unwrap()
        .map(|e| e.unwrap().file_name().to_string_lossy().into_owned())
        .collect();
    v.sort();
    v
}

#[test]
fn multi_connection_download_identical_with_progress() {
    let data = big();
    let srv = Server::with_files(&[("model.bin", data.clone())]);
    let tmp = TmpDir::new("multi");
    let dest = tmp.0.join("model.bin");
    let (cb, seen) = recorder();
    download(&fast(), &srv.url("model.bin"), &dest, None, 4, Some(cb)).unwrap();
    assert_eq!(std::fs::read(&dest).unwrap(), data);
    assert_eq!(names(&tmp.0), ["model.bin"]);
    let seen = seen.lock().unwrap();
    let last = seen.last().unwrap();
    assert_eq!(
        (last.done, last.total, last.file.as_str()),
        (data.len() as u64, data.len() as u64, "model.bin")
    );
    assert!(seen.windows(2).all(|w| w[0].done <= w[1].done));
    let ranged_gets = srv
        .log()
        .iter()
        .filter(|r| r.0 == "GET" && r.2.is_some())
        .count();
    assert_eq!(ranged_gets, 4);
}

#[test]
fn small_file_single_connection_and_creates_parent() {
    let srv = Server::with_files(&[("config.json", b"{\"a\": 1}".to_vec())]);
    let tmp = TmpDir::new("small");
    let dest = tmp.0.join("sub/dir/config.json");
    download(&fast(), &srv.url("config.json"), &dest, None, 8, None).unwrap();
    assert_eq!(std::fs::read(&dest).unwrap(), b"{\"a\": 1}");
    assert_eq!(srv.log().iter().filter(|r| r.0 == "GET").count(), 1);
}

#[test]
fn existing_dest_returns_immediately() {
    let tmp = TmpDir::new("exists");
    let dest = tmp.0.join("m.bin");
    std::fs::write(&dest, b"x").unwrap();
    download(
        &fast(),
        "http://127.0.0.1:1/unreachable",
        &dest,
        None,
        4,
        None,
    )
    .unwrap();
    assert_eq!(std::fs::read(&dest).unwrap(), b"x");
}

/// Lay out a `.part` + sidecar as the Python app would: 2 chunks done, 1000 bytes of the third.
fn seed_python_partial(dir: &Path, data: &[u8], n: usize) -> u64 {
    let size = data.len() / n;
    let mut part = vec![0u8; data.len()];
    let mut state = Vec::new();
    for i in 0..n {
        let (a, b) = (
            i * size,
            if i == n - 1 {
                data.len() - 1
            } else {
                (i + 1) * size - 1
            },
        );
        let got = if i < 2 {
            b - a + 1
        } else if i == 2 {
            1000
        } else {
            0
        };
        part[a..a + got].copy_from_slice(&data[a..a + got]);
        // Python's json.dumps spacing.
        state.push(format!(
            "{{\"downloaded\": {got}, \"done\": {}}}",
            if i < 2 { "true" } else { "false" }
        ));
    }
    std::fs::write(dir.join("model.bin.part"), part).unwrap();
    std::fs::write(
        dir.join("model.bin.progress.json"),
        format!("[{}]", state.join(", ")),
    )
    .unwrap();
    size as u64
}

#[test]
fn resume_from_python_format_sidecar() {
    let data = big();
    let srv = Server::with_files(&[("model.bin", data.clone())]);
    let tmp = TmpDir::new("resume");
    let size = seed_python_partial(&tmp.0, &data, 4);
    let (cb, seen) = recorder();
    let dest = tmp.0.join("model.bin");
    download(&fast(), &srv.url("model.bin"), &dest, None, 4, Some(cb)).unwrap();
    assert_eq!(std::fs::read(&dest).unwrap(), data);
    assert!(
        seen.lock().unwrap()[0].done >= size * 2,
        "counter starts from saved progress"
    );
    let starts: Vec<u64> = srv
        .log()
        .iter()
        .filter(|r| r.0 == "GET")
        .filter_map(|r| r.2.map(|x| x.0))
        .collect();
    assert!(
        !starts.contains(&0),
        "finished chunk 0 must not be re-fetched: {starts:?}"
    );
    assert!(starts.contains(&(2 * size + 1000)));
    assert_eq!(names(&tmp.0), ["model.bin"]);
}

#[test]
fn interrupted_run_leaves_python_readable_sidecar_then_resumes() {
    let data = big();
    let srv = Server::with_files(&[("model.bin", data.clone())]);
    let size = (data.len() / 4) as u64;
    srv.cfg.lock().unwrap().fail_start = Some(2 * size);
    let tmp = TmpDir::new("sidecar");
    let dest = tmp.0.join("model.bin");
    let opts = Options {
        max_retries: 2,
        ..fast()
    };
    let err = download(&opts, &srv.url("model.bin"), &dest, None, 4, None).unwrap_err();
    assert!(matches!(err, DownloadError::Network(_)), "{err}");
    assert!(!dest.exists());
    // Same shape Python reads: list of {"downloaded": int, "done": bool}.
    let v: serde_json::Value =
        serde_json::from_slice(&std::fs::read(tmp.0.join("model.bin.progress.json")).unwrap())
            .unwrap();
    let arr = v.as_array().unwrap();
    assert_eq!(arr.len(), 4);
    assert!(
        arr.iter()
            .all(|c| c["downloaded"].is_u64() && c["done"].is_boolean())
    );
    assert!(arr[2]["downloaded"].as_u64().unwrap() > 0);
    srv.cfg.lock().unwrap().fail_start = None;
    download(&fast(), &srv.url("model.bin"), &dest, None, 4, None).unwrap();
    assert_eq!(std::fs::read(&dest).unwrap(), data);
    assert_eq!(names(&tmp.0), ["model.bin"]);
}

#[test]
fn no_range_support_falls_back_to_single_stream() {
    let data = big();
    let srv = Server::start(Cfg {
        files: HashMap::from([("/model.bin".to_string(), data.clone())]),
        no_ranges: true,
        ..Cfg::default()
    });
    let tmp = TmpDir::new("norange");
    let dest = tmp.0.join("model.bin");
    download(&fast(), &srv.url("model.bin"), &dest, None, 4, None).unwrap();
    assert_eq!(std::fs::read(&dest).unwrap(), data);
    assert!(
        srv.log()
            .iter()
            .filter(|r| r.0 == "GET" && r.2.is_none())
            .count()
            >= 1
    );
}

#[test]
fn unknown_size_streams() {
    let data = vec![7u8; 3 * 1024 * 1024];
    let srv = Server::start(Cfg {
        files: HashMap::from([("/m.bin".to_string(), data.clone())]),
        no_ranges: true,
        hide_length: true,
        ..Cfg::default()
    });
    let tmp = TmpDir::new("unknown");
    let dest = tmp.0.join("m.bin");
    let (cb, seen) = recorder();
    download(&fast(), &srv.url("m.bin"), &dest, None, 4, Some(cb)).unwrap();
    assert_eq!(std::fs::read(&dest).unwrap(), data);
    let last = seen.lock().unwrap().last().cloned().unwrap();
    assert_eq!(
        (last.done, last.total),
        (data.len() as u64, data.len() as u64)
    );
}

#[test]
fn retry_after_dropped_connection() {
    let data = big();
    let srv = Server::with_files(&[("model.bin", data.clone())]);
    srv.cfg
        .lock()
        .unwrap()
        .fail_once
        .insert("/model.bin".into());
    let tmp = TmpDir::new("retry");
    let dest = tmp.0.join("model.bin");
    download(&fast(), &srv.url("model.bin"), &dest, None, 1, None).unwrap();
    assert_eq!(std::fs::read(&dest).unwrap(), data);
}

#[test]
fn bearer_token_sent_but_dropped_on_cross_host_redirect() {
    let data = vec![1u8; 4096];
    let cdn = Server::with_files(&[("blob", data.clone())]);
    let hub = Server::start(Cfg {
        require_token: Some("hf_secret".into()),
        redirects: HashMap::from([("/model.bin".to_string(), cdn.url("blob"))]),
        ..Cfg::default()
    });
    let tmp = TmpDir::new("redirect");
    let dest = tmp.0.join("model.bin");
    download(
        &fast(),
        &hub.url("model.bin"),
        &dest,
        Some("hf_secret"),
        2,
        None,
    )
    .unwrap();
    assert_eq!(std::fs::read(&dest).unwrap(), data);
    assert!(
        hub.log()
            .iter()
            .all(|r| r.3.as_deref() == Some("Bearer hf_secret"))
    );
    let cdn_log = cdn.log();
    assert!(!cdn_log.is_empty());
    assert!(
        cdn_log.iter().all(|r| r.3.is_none()),
        "token leaked to another host: {cdn_log:?}"
    );
}

#[test]
fn unauthorized_is_fatal_with_gated_hint() {
    let srv = Server::start(Cfg {
        files: HashMap::from([("/m.bin".to_string(), vec![0u8; 10])]),
        require_token: Some("tok".into()),
        ..Cfg::default()
    });
    let tmp = TmpDir::new("401");
    let err = download(
        &fast(),
        &srv.url("m.bin"),
        &tmp.0.join("m.bin"),
        None,
        2,
        None,
    )
    .unwrap_err();
    match &err {
        DownloadError::Http {
            status, message, ..
        } => {
            assert_eq!(*status, 401);
            assert!(message.contains("gated") && message.contains("token"));
        }
        other => panic!("{other:?}"),
    }
    assert_eq!(srv.log().len(), 1, "fatal errors are not retried");
}

#[test]
fn not_found_is_fatal_and_leaves_nothing() {
    let srv = Server::with_files(&[]);
    let tmp = TmpDir::new("404");
    let err = download(
        &fast(),
        &srv.url("nope.bin"),
        &tmp.0.join("nope.bin"),
        None,
        2,
        None,
    )
    .unwrap_err();
    assert!(
        matches!(err, DownloadError::Http { status: 404, .. }),
        "{err}"
    );
    assert_eq!(srv.log().len(), 1);
    assert!(names(&tmp.0).is_empty());
}

#[test]
fn lists_repo_files() {
    let body = "{\"siblings\": [{\"rfilename\": \"ggml-tiny.en.bin\"}, {\"rfilename\": \"models/a b.bin\"}]}".to_string();
    let srv = Server::with_files(&[(&format!("api/models/{REPO}"), body.into_bytes())]);
    let files = list_repo_files_at(&format!("{}/api/models", srv.base), REPO, None).unwrap();
    assert_eq!(files, ["ggml-tiny.en.bin", "models/a b.bin"]);
    let err =
        list_repo_files_at(&format!("{}/api/models", srv.base), "acme/missing", None).unwrap_err();
    assert!(matches!(err, DownloadError::Http { status: 404, .. }));
}

#[test]
fn resolve_url_encodes_segments() {
    assert_eq!(
        hf_resolve_url("ggerganov/whisper.cpp", "ggml-tiny.en.bin"),
        "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-tiny.en.bin"
    );
    assert_eq!(
        hf_resolve_url("a/b", "sub dir/ü#?.bin"),
        "https://huggingface.co/a/b/resolve/main/sub%20dir/%C3%BC%23%3F.bin"
    );
}

/// Manual real-network check: `STT_DL_URL=... STT_DL_DEST=... cargo test -p stt-download -- --ignored --nocapture`.
#[test]
#[ignore = "real network; set STT_DL_URL and STT_DL_DEST"]
fn real_download() {
    let (Ok(url), Ok(dest)) = (std::env::var("STT_DL_URL"), std::env::var("STT_DL_DEST")) else {
        return;
    };
    let t = std::time::Instant::now();
    let cb: DownloadProgressFn =
        Arc::new(|p| eprintln!("{} {}/{} ({}%)", p.file, p.done, p.total, p.percent()));
    crate::download_file(&url, Path::new(&dest), None, 8, Some(cb)).unwrap();
    let size = std::fs::metadata(&dest).unwrap().len();
    let secs = t.elapsed().as_secs_f64();
    eprintln!(
        "size {size} in {secs:.1}s = {:.1} MB/s",
        size as f64 / 1e6 / secs
    );
}
