"""Multi-connection, resumable Hugging Face model downloader (stdlib only)."""
from __future__ import annotations

import fnmatch
import http.client
import json
import os
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable
from pathlib import Path

HF_API = "https://huggingface.co/api/models"  # + /<repo_id>
HF_RESOLVE = "https://huggingface.co"  # + /<repo_id>/resolve/main/<file>

# Same files faster_whisper.utils.download_model fetches.
ALLOW_PATTERNS = ["config.json", "preprocessor_config.json", "model.bin", "tokenizer.json", "vocabulary.*"]

BLOCK_SIZE = 1024 * 1024
MAX_RETRIES = 5
RETRY_BACKOFF = 3.0  # seconds, multiplied by the attempt number
SMALL_FILE_THRESHOLD = 8 * 1024 * 1024  # below this one connection is enough
SAVE_INTERVAL = 0.5
REPORT_INTERVAL = 0.2

ProgressFn = Callable[[int, int, str], None]


class DownloadError(Exception):
    pass


class _FatalHTTP(DownloadError):
    """HTTP status that retrying will not fix."""


class _NoAuthOnRedirect(urllib.request.HTTPRedirectHandler):
    """Drop the bearer token when a redirect leaves the original host (CDN links are presigned)."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        new = super().redirect_request(req, fp, code, msg, headers, newurl)
        if new is not None and urllib.parse.urlparse(newurl).netloc != urllib.parse.urlparse(req.full_url).netloc:
            new.headers.pop("Authorization", None)
        return new


_opener = urllib.request.build_opener(_NoAuthOnRedirect)


def _request(url: str, token: str | None, byte_range: tuple[int, int] | None = None, method: str = "GET"):
    req = urllib.request.Request(url, method=method)
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    if byte_range:
        req.add_header("Range", f"bytes={byte_range[0]}-{byte_range[1]}")
    return req


def _open(req, timeout: float):
    try:
        return _opener.open(req, timeout=timeout)
    except urllib.error.HTTPError as e:
        if e.code in (401, 403):
            raise _FatalHTTP(f"HTTP {e.code} for {req.full_url}; the repo may be gated, add a Hugging Face token") from e
        if e.code == 404:
            raise _FatalHTTP(f"HTTP 404 (not found): {req.full_url}") from e
        raise


def repo_for(name: str) -> str:
    if "/" in name:
        return name
    from faster_whisper.utils import _MODELS

    try:
        return _MODELS[name]
    except KeyError:
        raise DownloadError(f"Unknown model '{name}'") from None


def _repo_files(repo: str, token: str | None) -> list[str]:
    try:
        with _open(_request(f"{HF_API}/{repo}", token), 30) as resp:
            info = json.loads(resp.read().decode("utf-8"))
    except DownloadError:
        raise
    except (OSError, ValueError, http.client.HTTPException) as e:
        raise DownloadError(f"Could not list files of {repo}: {e}") from e
    files = [s["rfilename"] for s in info.get("siblings", [])]
    return [f for f in files if any(fnmatch.fnmatch(f, p) for p in ALLOW_PATTERNS)]


def _remote_info(url: str, token: str | None) -> tuple[int, bool]:
    """Return (total_size, supports_range); HEAD first, then a 1-byte ranged GET."""
    try:
        with _open(_request(url, token, method="HEAD"), 30) as resp:
            return int(resp.headers.get("Content-Length", 0)), resp.headers.get("Accept-Ranges", "") == "bytes"
    except _FatalHTTP:
        raise
    except OSError:
        pass  # some mirrors reject HEAD
    with _open(_request(url, token, byte_range=(0, 0)), 30) as resp:
        cr = resp.headers.get("Content-Range", "")
        size = int(cr.split("/")[-1]) if "/" in cr else int(resp.headers.get("Content-Length", 0))
        return size, bool(cr)


class _Job:
    """Shared state of one file download: chunk progress, throttled sidecar saves and reports."""

    def __init__(self, chunks: list[tuple[int, int]], state: list[dict], total: int, name: str,
                 progress_path: Path, on_progress: ProgressFn | None):
        self.chunks, self.state, self.total, self.name = chunks, state, total, name
        self.progress_path, self.on_progress = progress_path, on_progress
        self.lock = threading.Lock()
        self.stop = threading.Event()
        self.done = sum(c["downloaded"] for c in state)
        self._last_save = self._last_report = 0.0

    def _save(self) -> None:  # call with the lock held
        tmp = self.progress_path.with_name(self.progress_path.name + ".tmp")
        tmp.write_text(json.dumps(self.state))
        os.replace(tmp, self.progress_path)
        self._last_save = time.monotonic()

    def add(self, idx: int, n: int) -> None:
        with self.lock:
            self.state[idx]["downloaded"] += n
            self.done += n
            now = time.monotonic()
            if now - self._last_save >= SAVE_INTERVAL:
                self._save()
            if self.on_progress and now - self._last_report >= REPORT_INTERVAL:
                self._last_report = now
                self.on_progress(self.done, self.total, self.name)

    def finish_chunk(self, idx: int) -> None:
        with self.lock:
            self.state[idx]["done"] = True
            self._save()

    def flush(self) -> None:
        with self.lock:
            self._save()
            if self.on_progress:
                self.on_progress(self.done, self.total, self.name)


def _run_chunk(job: _Job, idx: int, url: str, token: str | None, part: Path) -> None:
    start, end = job.chunks[idx]
    st = job.state[idx]
    if st["done"]:
        return
    for attempt in range(1, MAX_RETRIES + 1):
        if job.stop.is_set():
            return
        pos = start + st["downloaded"]
        if pos > end:
            job.finish_chunk(idx)
            return
        try:
            with _open(_request(url, token, byte_range=(pos, end)), 60) as resp, part.open("r+b") as f:
                if resp.status != 206 and pos != 0:
                    raise OSError("server ignored the Range header")
                f.seek(pos)
                while not job.stop.is_set():
                    block = resp.read(BLOCK_SIZE)
                    if not block:
                        break
                    f.write(block)
                    job.add(idx, len(block))
            if start + st["downloaded"] <= end:
                if job.stop.is_set():
                    return
                raise OSError("connection closed early")
            job.finish_chunk(idx)
            return
        except (OSError, http.client.HTTPException) as e:
            if attempt == MAX_RETRIES:
                raise DownloadError(f"{job.name}: {e}") from e
            job.stop.wait(RETRY_BACKOFF * attempt)


def _load_state(path: Path, chunks: list[tuple[int, int]]) -> list[dict]:
    try:
        saved = json.loads(path.read_text())
        if len(saved) == len(chunks) and all(
            0 <= s["downloaded"] <= e - b + 1 for s, (b, e) in zip(saved, chunks)
        ):
            return [{"downloaded": s["downloaded"], "done": bool(s["done"])} for s in saved]
    except (OSError, ValueError, KeyError, TypeError):
        pass
    return [{"downloaded": 0, "done": False} for _ in chunks]


def download_file(url: str, dest: Path, token: str | None = None, connections: int = 8,
                  on_progress: ProgressFn | None = None) -> None:
    """Download url to dest (resumable via dest.part + dest.progress.json). No-op if dest exists."""
    if dest.exists():
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    progress_path = dest.with_name(dest.name + ".progress.json")
    try:
        total, ranged = _remote_info(url, token)
    except DownloadError:
        raise
    except (OSError, ValueError, http.client.HTTPException) as e:
        raise DownloadError(f"{dest.name}: {e}") from e
    if not total:
        raise DownloadError(f"Could not determine the size of {dest.name}")

    fresh = not part.exists() or part.stat().st_size != total
    if fresh:
        with part.open("wb") as f:
            f.truncate(total)
        progress_path.unlink(missing_ok=True)

    n = max(1, connections) if ranged and total >= SMALL_FILE_THRESHOLD else 1
    size = total // n
    chunks = [(i * size, total - 1 if i == n - 1 else (i + 1) * size - 1) for i in range(n)]
    job = _Job(chunks, _load_state(progress_path, chunks), total, dest.name, progress_path, on_progress)

    errors: list[Exception] = []

    def worker(idx: int) -> None:
        try:
            _run_chunk(job, idx, url, token, part)
        except Exception as e:  # noqa: BLE001 - reported to the caller after join
            errors.append(e)
            job.stop.set()

    threads = [threading.Thread(target=worker, args=(i,), daemon=True) for i in range(n)]
    for t in threads:
        t.start()
    try:
        for t in threads:
            t.join()
    except BaseException:
        job.stop.set()
        raise
    job.flush()
    if errors:
        raise DownloadError(f"Download of {dest.name} failed (it will resume next time): {errors[0]}")
    if not all(c["done"] for c in job.state):
        raise DownloadError(f"Download of {dest.name} was interrupted")
    os.replace(part, dest)
    progress_path.unlink(missing_ok=True)


def _is_complete(directory: Path) -> bool:
    """Offline check: core files present and nothing half-downloaded."""
    if not directory.is_dir() or any(directory.glob("*.part")):
        return False
    has = lambda pat: any(directory.glob(pat))  # noqa: E731
    return has("model.bin") and has("config.json") and (has("tokenizer.json") or has("vocabulary.*"))


_locks: dict[str, threading.Lock] = {}
_locks_guard = threading.Lock()


def _model_lock(key: str) -> threading.Lock:
    with _locks_guard:
        return _locks.setdefault(key, threading.Lock())


def ensure_model(name: str, models_dir: Path, token: str | None = None, connections: int = 8,
                 on_progress: ProgressFn | None = None) -> Path:
    """Return a local directory holding the model, downloading it first if needed."""
    if Path(name).is_dir():
        return Path(name)
    with _model_lock(name):
        try:
            from faster_whisper.utils import download_model

            return Path(download_model(name, local_files_only=True))
        except Exception:  # not in the HF cache
            pass
        repo = repo_for(name)
        publisher, _, model = repo.partition("/")
        target = Path(models_dir) / publisher / (model or "misc")
        if _is_complete(target):
            return target
        for filename in _repo_files(repo, token):
            quoted = urllib.parse.quote(filename)
            download_file(f"{HF_RESOLVE}/{repo}/resolve/main/{quoted}", target / filename,
                          token, connections, on_progress)
        if not _is_complete(target):
            raise DownloadError(f"{repo} is missing expected files after download")
        return target
