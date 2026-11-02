"""TTL / unload behaviour of the OpenAI-compatible backend against fake local servers."""
from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from stt_app.config import LLMConfig
from stt_app.llm import LLMError, make_backend, process
from stt_app.llm.openai_compat import detect_server
from stt_app.ui import parse_duration, ttl_label

CHAT = {
    "id": "x", "object": "chat.completion", "created": 0, "model": "m",
    "choices": [{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": "ok"}}],
}


def make_server(kind: str):
    log: list[tuple[str, str, dict | None]] = []

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _send(self, code, body, ctype="application/json"):
            data = body if isinstance(body, bytes) else json.dumps(body).encode()
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            log.append(("GET", self.path, None))
            if kind == "ollama" and self.path == "/":
                return self._send(200, b"Ollama is running", "text/plain")
            if kind == "ollama" and self.path == "/api/version":
                return self._send(200, {"version": "0.1"})
            if kind == "lmstudio" and self.path == "/api/v0/models":
                return self._send(401, {"error": {"message": "An LM Studio API token is required"}})
            self._send(404, {"error": "nope"})

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
            log.append(("POST", self.path, body))
            if self.path == "/v1/chat/completions":
                return self._send(200, CHAT)
            self._send(200, {})

    srv = ThreadingHTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, f"http://127.0.0.1:{srv.server_address[1]}/v1", log


@pytest.fixture
def server():
    servers = []

    def start(kind):
        srv, url, log = make_server(kind)
        servers.append(srv)
        return url, log

    yield start
    for s in servers:
        s.shutdown()
        s.server_close()


def cfg_for(url, **kw):
    return LLMConfig(provider="openai_compat", model="m", base_url=url, **kw)


def chat_bodies(log):
    return [b for m, p, b in log if p == "/v1/chat/completions"]


def test_detection(server):
    for kind in ("ollama", "lmstudio", "other"):
        url, _ = server(kind)
        assert detect_server(url) == kind


def test_lmstudio_sends_ttl(server):
    url, log = server("lmstudio")
    assert process(cfg_for(url, idle_ttl_seconds=120), "tok", "hi") == "ok"
    assert chat_bodies(log)[0]["ttl"] == 120


def test_ollama_sets_keep_alive_natively(server):
    url, log = server("ollama")
    process(cfg_for(url, idle_ttl_seconds=90), None, "hi")
    assert "ttl" not in chat_bodies(log)[0]
    assert ("POST", "/api/generate", {"model": "m", "keep_alive": 90}) in log


def test_other_sends_nothing_extra(server):
    url, log = server("other")
    process(cfg_for(url, idle_ttl_seconds=120), None, "hi")
    assert set(chat_bodies(log)[0]) == {"model", "messages", "temperature"}
    assert [p for m, p, b in log if m == "POST"] == ["/v1/chat/completions"]


def test_ttl_zero_sends_nothing_extra(server):
    url, log = server("lmstudio")
    process(cfg_for(url, idle_ttl_seconds=0), "tok", "hi")
    assert "ttl" not in chat_bodies(log)[0]


def test_unload_endpoints(server):
    url, log = server("lmstudio")
    make_backend(cfg_for(url), "tok").unload()
    assert ("POST", "/api/v1/models/unload", {"instance_id": "m"}) in log
    url, log = server("ollama")
    make_backend(cfg_for(url), None).unload()
    assert ("POST", "/api/generate", {"model": "m", "keep_alive": 0}) in log
    url, log = server("other")
    make_backend(cfg_for(url), None).unload()
    assert not [e for e in log if e[0] == "POST"]


def test_unload_failure_raises_llmerror():
    with pytest.raises(LLMError):
        b = make_backend(cfg_for("http://127.0.0.1:9/v1/"), None)
        b._kind = "lmstudio"
        b.unload()


def test_detection_failure_still_completes(server, monkeypatch):
    url, log = server("other")
    import stt_app.llm.openai_compat as oc

    def boom(*a, **k):
        raise RuntimeError("probe broke")

    monkeypatch.setattr(oc, "_http", boom)
    assert detect_server(url) == "other"
    assert process(cfg_for(url, idle_ttl_seconds=120), None, "hi") == "ok"


def test_detection_cached(server):
    url, log = server("ollama")
    b = make_backend(cfg_for(url, idle_ttl_seconds=60), None)
    b.complete("s", "a")
    b.complete("s", "b")
    assert len([e for e in log if e[:2] == ("GET", "/")]) == 1


def test_cloud_unload_noop():
    make_backend(LLMConfig(provider="anthropic"), "k").unload()


def test_duration_helpers():
    assert parse_duration("2m") == 120 and parse_duration("90") == 90 and parse_duration("1h") == 3600
    assert parse_duration("0") == 0 and parse_duration("x") is None and parse_duration("-5") is None
    assert ttl_label(0) == "server default" and ttl_label(120) == "2m" and ttl_label(90) == "90s"
