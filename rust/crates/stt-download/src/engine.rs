//! The download engine: size probe, chunk plan, worker threads, sidecar, final rename.

use std::fs::{self, File, OpenOptions};
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread;
use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};
use stt_core::{DownloadError, DownloadProgress, DownloadProgressFn};

use crate::Options;
use crate::http::{self, Reply};

const BLOCK_SIZE: usize = 1024 * 1024;
/// Below this one connection is enough.
const SMALL_FILE_THRESHOLD: u64 = 8 * 1024 * 1024;

/// One sidecar entry; serialises as Python's `{"downloaded": n, "done": bool}`.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct Chunk {
    downloaded: u64,
    done: bool,
}

fn io_err(path: &Path) -> impl FnOnce(std::io::Error) -> DownloadError + '_ {
    move |source| DownloadError::Io {
        path: path.to_path_buf(),
        source,
    }
}

fn is_fatal(e: &DownloadError) -> bool {
    matches!(e, DownloadError::Http { .. } | DownloadError::Io { .. })
}

fn file_name_of(p: &Path) -> String {
    p.file_name()
        .map_or_else(String::new, |n| n.to_string_lossy().into_owned())
}

pub(crate) fn list_repo_files_at(
    api: &str,
    repo: &str,
    token: Option<&str>,
) -> Result<Vec<String>, DownloadError> {
    #[derive(Deserialize)]
    struct Sibling {
        rfilename: String,
    }
    #[derive(Deserialize)]
    struct Info {
        #[serde(default)]
        siblings: Vec<Sibling>,
    }
    let list_err = |e: &dyn std::fmt::Display| {
        DownloadError::Network(format!("Could not list files of {repo}: {e}"))
    };
    let agent = http::agent();
    let reply = http::send(&agent, false, &format!("{api}/{repo}"), token, None)?;
    let mut body = Vec::new();
    reply
        .into_reader()
        .read_to_end(&mut body)
        .map_err(|e| list_err(&e))?;
    let info: Info = serde_json::from_slice(&body).map_err(|e| list_err(&e))?;
    Ok(info.siblings.into_iter().map(|s| s.rfilename).collect())
}

/// `(total_size, supports_range)`; HEAD first, then a 1-byte ranged GET. Size 0 means unknown.
fn remote_info(
    agent: &ureq::Agent,
    url: &str,
    token: Option<&str>,
) -> Result<(u64, bool), DownloadError> {
    match http::send(agent, true, url, token, None) {
        Ok(r) => {
            let total = r
                .header("content-length")
                .and_then(|v| v.parse().ok())
                .unwrap_or(0);
            if total > 0 {
                return Ok((total, r.header("accept-ranges") == Some("bytes")));
            }
        }
        Err(e @ DownloadError::Http { .. }) => return Err(e),
        Err(_) => {} // some mirrors reject HEAD
    }
    let r = http::send(agent, false, url, token, Some((0, 0)))?;
    let cr = r.header("content-range");
    let total = match cr.and_then(|c| c.rsplit('/').next()) {
        Some(t) => t.parse().unwrap_or(0),
        None => r
            .header("content-length")
            .and_then(|v| v.parse().ok())
            .unwrap_or(0),
    };
    Ok((total, r.status == 206 && cr.is_some()))
}

struct JobState {
    chunks: Vec<Chunk>,
    done: u64,
    last_save: Option<Instant>,
    last_report: Option<Instant>,
}

/// Shared state of one file download: chunk progress, throttled sidecar saves and reports.
struct Job<'a> {
    opts: &'a Options,
    ranges: Vec<(u64, u64)>,
    ranged: bool,
    total: u64,
    name: String,
    progress_path: PathBuf,
    on_progress: Option<DownloadProgressFn>,
    state: Mutex<JobState>,
    stop: AtomicBool,
}

impl Job<'_> {
    fn lock(&self) -> std::sync::MutexGuard<'_, JobState> {
        self.state.lock().unwrap_or_else(|p| p.into_inner())
    }

    fn save(&self, st: &mut JobState) {
        let tmp = self.progress_path.with_extension("json.tmp");
        if let Ok(json) = serde_json::to_vec(&st.chunks)
            && fs::write(&tmp, json).is_ok()
        {
            let _ = fs::rename(&tmp, &self.progress_path);
        }
        st.last_save = Some(Instant::now());
    }

    fn report(&self, st: &JobState) {
        if let Some(cb) = &self.on_progress {
            cb(&DownloadProgress {
                file: self.name.clone(),
                done: st.done,
                total: self.total,
            });
        }
    }

    fn add(&self, idx: usize, n: u64) {
        let mut st = self.lock();
        st.chunks[idx].downloaded += n;
        st.done += n;
        let now = Instant::now();
        if st
            .last_save
            .is_none_or(|t| now - t >= self.opts.save_interval)
        {
            self.save(&mut st);
        }
        if st
            .last_report
            .is_none_or(|t| now - t >= self.opts.report_interval)
        {
            st.last_report = Some(now);
            self.report(&st);
        }
    }

    fn finish_chunk(&self, idx: usize) {
        let mut st = self.lock();
        st.chunks[idx].done = true;
        self.save(&mut st);
    }

    /// Forget a chunk's bytes (a plain stream always restarts at byte 0).
    fn reset_chunk(&self, idx: usize) {
        let mut st = self.lock();
        let had = std::mem::take(&mut st.chunks[idx].downloaded);
        st.done -= had;
    }

    fn flush(&self) {
        let mut st = self.lock();
        self.save(&mut st);
        self.report(&st);
    }

    fn sleep_backoff(&self, attempt: u32) {
        let mut left = self.opts.retry_backoff * attempt;
        let step = Duration::from_millis(50);
        while !left.is_zero() && !self.stop.load(Ordering::Relaxed) {
            let s = left.min(step);
            thread::sleep(s);
            left -= s;
        }
    }
}

fn run_chunk(
    job: &Job,
    agent: &ureq::Agent,
    idx: usize,
    url: &str,
    token: Option<&str>,
    part: &Path,
) -> Result<(), DownloadError> {
    let (start, end) = job.ranges[idx];
    if job.lock().chunks[idx].done {
        return Ok(());
    }
    for attempt in 1..=job.opts.max_retries {
        if job.stop.load(Ordering::Relaxed) {
            return Ok(());
        }
        if !job.ranged {
            job.reset_chunk(idx);
        }
        let pos = start + job.lock().chunks[idx].downloaded;
        if pos > end {
            job.finish_chunk(idx);
            return Ok(());
        }
        let result = (|| -> Result<(), DownloadError> {
            let range = job.ranged.then_some((pos, end));
            let reply = http::send(agent, false, url, token, range)?;
            if job.ranged && reply.status != 206 && pos != 0 {
                return Err(DownloadError::Network(
                    "server ignored the Range header".into(),
                ));
            }
            let mut f = OpenOptions::new()
                .write(true)
                .open(part)
                .map_err(io_err(part))?;
            f.seek(SeekFrom::Start(pos)).map_err(io_err(part))?;
            copy_chunk(job, idx, reply, &mut f, end - pos + 1, part)
        })();
        match result {
            Ok(()) => {
                if start + job.lock().chunks[idx].downloaded > end {
                    job.finish_chunk(idx);
                    return Ok(());
                }
                if job.stop.load(Ordering::Relaxed) {
                    return Ok(());
                }
                if attempt == job.opts.max_retries {
                    return Err(DownloadError::Network("connection closed early".into()));
                }
            }
            Err(e) if is_fatal(&e) => return Err(e),
            Err(e) => {
                if attempt == job.opts.max_retries {
                    return Err(e);
                }
            }
        }
        job.sleep_backoff(attempt);
    }
    Ok(())
}

/// Copy at most `limit` bytes of the body into `f`, counting progress as bytes land.
fn copy_chunk(
    job: &Job,
    idx: usize,
    reply: Reply,
    f: &mut File,
    limit: u64,
    part: &Path,
) -> Result<(), DownloadError> {
    let mut reader = reply.into_reader();
    let mut buf = vec![0u8; BLOCK_SIZE];
    let mut left = limit;
    while left > 0 && !job.stop.load(Ordering::Relaxed) {
        let want = buf.len().min(usize::try_from(left).unwrap_or(usize::MAX));
        let n = match reader.read(&mut buf[..want]) {
            Ok(0) => break,
            Ok(n) => n,
            Err(e) => return Err(DownloadError::Network(e.to_string())),
        };
        f.write_all(&buf[..n]).map_err(io_err(part))?;
        job.add(idx, n as u64);
        left -= n as u64;
    }
    Ok(())
}

fn load_state(path: &Path, ranges: &[(u64, u64)]) -> Vec<Chunk> {
    let fresh = || {
        ranges
            .iter()
            .map(|_| Chunk {
                downloaded: 0,
                done: false,
            })
            .collect()
    };
    let Ok(text) = fs::read(path) else {
        return fresh();
    };
    match serde_json::from_slice::<Vec<Chunk>>(&text) {
        Ok(saved)
            if saved.len() == ranges.len()
                && saved
                    .iter()
                    .zip(ranges)
                    .all(|(s, (b, e))| s.downloaded <= e - b + 1) =>
        {
            saved
        }
        _ => fresh(),
    }
}

/// Single stream, no sidecar: used when the server reports no size at all.
fn stream_unknown_size(
    opts: &Options,
    agent: &ureq::Agent,
    url: &str,
    dest: &Path,
    part: &Path,
    token: Option<&str>,
    on_progress: Option<DownloadProgressFn>,
) -> Result<(), DownloadError> {
    let name = file_name_of(dest);
    let report = |done: u64, total: u64| {
        if let Some(cb) = &on_progress {
            cb(&DownloadProgress {
                file: name.clone(),
                done,
                total,
            });
        }
    };
    let mut last_err = None;
    for attempt in 1..=opts.max_retries {
        let outcome = (|| -> Result<u64, DownloadError> {
            let reply = http::send(agent, false, url, token, None)?;
            let total: u64 = reply
                .header("content-length")
                .and_then(|v| v.parse().ok())
                .unwrap_or(0);
            let mut f = File::create(part).map_err(io_err(part))?;
            let mut reader = reply.into_reader();
            let mut buf = vec![0u8; BLOCK_SIZE];
            let (mut done, mut last) = (0u64, Instant::now());
            loop {
                let n = reader
                    .read(&mut buf)
                    .map_err(|e| DownloadError::Network(e.to_string()))?;
                if n == 0 {
                    break;
                }
                f.write_all(&buf[..n]).map_err(io_err(part))?;
                done += n as u64;
                if last.elapsed() >= opts.report_interval {
                    last = Instant::now();
                    report(done, total.max(done));
                }
            }
            if total > 0 && done != total {
                return Err(DownloadError::Network("connection closed early".into()));
            }
            Ok(done)
        })();
        match outcome {
            Ok(done) => {
                report(done, done);
                return fs::rename(part, dest).map_err(io_err(dest));
            }
            Err(e) if is_fatal(&e) => return Err(e),
            Err(e) => {
                last_err = Some(e);
                thread::sleep(opts.retry_backoff * attempt);
            }
        }
    }
    Err(DownloadError::Network(format!(
        "{name}: {}",
        last_err.map_or_else(String::new, |e| e.to_string())
    )))
}

pub(crate) fn download(
    opts: &Options,
    url: &str,
    dest: &Path,
    token: Option<&str>,
    connections: u32,
    on_progress: Option<DownloadProgressFn>,
) -> Result<(), DownloadError> {
    if dest.exists() {
        return Ok(());
    }
    if let Some(parent) = dest.parent().filter(|p| !p.as_os_str().is_empty()) {
        fs::create_dir_all(parent).map_err(io_err(parent))?;
    }
    let file_name = file_name_of(dest);
    let part = dest.with_file_name(format!("{file_name}.part"));
    let progress_path = dest.with_file_name(format!("{file_name}.progress.json"));
    let agent = http::agent();
    let (total, ranged) = remote_info(&agent, url, token).map_err(|e| match e {
        DownloadError::Network(m) => DownloadError::Network(format!("{file_name}: {m}")),
        other => other,
    })?;
    if total == 0 {
        return stream_unknown_size(opts, &agent, url, dest, &part, token, on_progress);
    }

    if fs::metadata(&part).map_or(true, |m| m.len() != total) {
        let f = File::create(&part).map_err(io_err(&part))?;
        f.set_len(total).map_err(io_err(&part))?;
        let _ = fs::remove_file(&progress_path);
    }

    let n = if ranged && total >= SMALL_FILE_THRESHOLD {
        u64::from(connections.max(1))
    } else {
        1
    };
    let size = total / n;
    let ranges: Vec<(u64, u64)> = (0..n)
        .map(|i| {
            (
                i * size,
                if i == n - 1 {
                    total - 1
                } else {
                    (i + 1) * size - 1
                },
            )
        })
        .collect();
    let chunks = load_state(&progress_path, &ranges);
    let done = chunks.iter().map(|c| c.downloaded).sum();
    let job = Job {
        opts,
        ranges,
        ranged,
        total,
        name: file_name.clone(),
        progress_path: progress_path.clone(),
        on_progress,
        state: Mutex::new(JobState {
            chunks,
            done,
            last_save: None,
            last_report: None,
        }),
        stop: AtomicBool::new(false),
    };
    job.report(&job.lock());

    let errors: Mutex<Vec<DownloadError>> = Mutex::new(Vec::new());
    thread::scope(|s| {
        for idx in 0..n as usize {
            let (job, agent, errors, part) = (&job, &agent, &errors, &part);
            s.spawn(move || {
                if let Err(e) = run_chunk(job, agent, idx, url, token, part) {
                    errors.lock().unwrap_or_else(|p| p.into_inner()).push(e);
                    job.stop.store(true, Ordering::Relaxed);
                }
            });
        }
    });
    job.flush();
    let mut errors = errors.into_inner().unwrap_or_else(|p| p.into_inner());
    if !errors.is_empty() {
        let e = errors.remove(0);
        return Err(if is_fatal(&e) {
            e
        } else {
            DownloadError::Network(format!(
                "Download of {file_name} failed (it will resume next time): {e}"
            ))
        });
    }
    if !job.lock().chunks.iter().all(|c| c.done) {
        return Err(DownloadError::Interrupted(file_name));
    }
    fs::rename(&part, dest).map_err(io_err(dest))?;
    let _ = fs::remove_file(&progress_path);
    Ok(())
}
