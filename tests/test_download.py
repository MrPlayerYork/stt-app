"""Tests for stt_app.download against a local range-capable HTTP server (no real network)."""
from __future__ import annotations

import json
import os
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from stt_app import download as dl

REPO = "acme/faster-whisper-test"
BIG = os.urandom(10 * 1024 * 1024)
FILES = {
    "model.bin": BIG,
    "config.json": b'{"a": 1}',
    "tokenizer.json": b"{}",
    "vocabulary.txt": b"hello\n",
    "README.md": b"ignored",
}


class Handler(BaseHTTPRequestHandler):
    fail_once: set[str] = set()  # paths whose first ranged GET is cut short
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):
        pass

    def _send(self, body: bytes, status=200, headers=None, send_body=True):
        self.send_response(status)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Accept-Ranges", "bytes")
        for k, v in (headers or {}).items():
            self.send_header(k, v)
        self.end_headers()
        if send_body:
            self.wfile.write(body)

    def _route(self, send_body: bool):
        if self.path == f"/api/models/{REPO}":
            body = json.dumps({"siblings": [{"rfilename": n} for n in FILES]}).encode()
            return self._send(body, send_body=send_body)
        prefix = f"/{REPO}/resolve/main/"
        name = self.path[len(prefix):] if self.path.startswith(prefix) else ""
        data = FILES.get(name)
        if data is None:
            return self._send(b"", 404, send_body=False)
        rng = self.headers.get("Range")
        if not rng:
            return self._send(data, send_body=send_body)
        a, b = rng.removeprefix("bytes=").split("-")
        a, b = int(a), min(int(b), len(data) - 1)
        body = data[a:b + 1]
        hdrs = {"Content-Range": f"bytes {a}-{b}/{len(data)}"}
        if self.path in Handler.fail_once and len(body) > 1024:
            Handler.fail_once.discard(self.path)
            self.send_response(206)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Content-Range", hdrs["Content-Range"])
            self.end_headers()
            self.wfile.write(body[: len(body) // 3])
            self.wfile.flush()
            self.close_connection = True
            return
        self._send(body, 206, hdrs, send_body)

    def do_GET(self):
        self._route(True)

    def do_HEAD(self):
        self._route(False)


@pytest.fixture
def server(monkeypatch):
    Handler.fail_once = set()
    srv = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{srv.server_address[1]}"
    monkeypatch.setattr(dl, "HF_API", f"{base}/api/models")
    monkeypatch.setattr(dl, "HF_RESOLVE", base)
    monkeypatch.setattr(dl, "RETRY_BACKOFF", 0.0)
    yield base
    srv.shutdown()
    srv.server_close()


def test_multi_connection_download_identical(server, tmp_path):
    seen = []
    dest = tmp_path / "model.bin"
    dl.download_file(f"{server}/{REPO}/resolve/main/model.bin", dest, connections=4,
                     on_progress=lambda d, t, n: seen.append((d, t, n)))
    assert dest.read_bytes() == BIG
    assert not list(tmp_path.glob("model.bin.*"))
    assert seen[-1] == (len(BIG), len(BIG), "model.bin")


def test_small_file_single_connection(server, tmp_path):
    dest = tmp_path / "config.json"
    dl.download_file(f"{server}/{REPO}/resolve/main/config.json", dest, connections=8)
    assert dest.read_bytes() == FILES["config.json"]


def test_resume_from_sidecar(server, tmp_path):
    dest = tmp_path / "model.bin"
    part = tmp_path / "model.bin.part"
    n = 4
    size = len(BIG) // n
    chunks = [(i * size, len(BIG) - 1 if i == n - 1 else (i + 1) * size - 1) for i in range(n)]
    data = bytearray(len(BIG))
    state = []
    for i, (a, b) in enumerate(chunks):
        got = (b - a + 1) if i < 2 else (1000 if i == 2 else 0)
        data[a:a + got] = BIG[a:a + got]
        state.append({"downloaded": got, "done": i < 2})
    part.write_bytes(bytes(data))
    (tmp_path / "model.bin.progress.json").write_text(json.dumps(state))
    seen = []
    dl.download_file(f"{server}/{REPO}/resolve/main/model.bin", dest, connections=n,
                     on_progress=lambda d, t, name: seen.append(d))
    assert dest.read_bytes() == BIG
    assert seen[0] >= size * 2  # counter started from the saved progress


def test_retry_after_dropped_connection(server, tmp_path):
    Handler.fail_once = {f"/{REPO}/resolve/main/model.bin"}
    dest = tmp_path / "model.bin"
    dl.download_file(f"{server}/{REPO}/resolve/main/model.bin", dest, connections=1)
    assert dest.read_bytes() == BIG


def test_failure_keeps_part_and_raises(server, tmp_path):
    with pytest.raises(dl.DownloadError):
        dl.download_file(f"{server}/{REPO}/resolve/main/nope.bin", tmp_path / "nope.bin")


def test_ensure_model_downloads_filtered_files(server, tmp_path, monkeypatch):
    monkeypatch.setattr("faster_whisper.utils.download_model",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("not cached")))
    path = dl.ensure_model(REPO, tmp_path, connections=4)
    assert path == tmp_path / "acme" / "faster-whisper-test"
    assert (path / "model.bin").read_bytes() == BIG
    assert not (path / "README.md").exists()
    assert {p.name for p in path.iterdir()} == {"model.bin", "config.json", "tokenizer.json", "vocabulary.txt"}
    # complete directory is reused without touching the network
    monkeypatch.setattr(dl, "HF_API", "http://127.0.0.1:1/unreachable")
    assert dl.ensure_model(REPO, tmp_path) == path


def test_ensure_model_returns_existing_directory(tmp_path):
    assert dl.ensure_model(str(tmp_path), tmp_path / "elsewhere") == tmp_path


def test_repo_for():
    assert dl.repo_for("large-v3-turbo") == "mobiuslabsgmbh/faster-whisper-large-v3-turbo"
    assert dl.repo_for("tiny") == "Systran/faster-whisper-tiny"
    assert dl.repo_for("me/custom-model") == "me/custom-model"
    with pytest.raises(dl.DownloadError):
        dl.repo_for("no-such-model")
