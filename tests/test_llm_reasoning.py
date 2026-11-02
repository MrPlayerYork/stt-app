"""Reasoning controls and result parsing for all LLM backends (no network, no real servers)."""
from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace as NS

import anthropic
import httpx
import pytest
from google.genai import errors as gerrors
from google.genai import types as gtypes

from stt_app.llm import LLMResult, REASONING_LEVELS
from stt_app.llm.anthropic_backend import AnthropicBackend
from stt_app.llm.base import split_think
from stt_app.llm.google_backend import GoogleBackend
from stt_app.llm.openai_compat import OpenAICompatBackend


def test_levels_and_dataclass():
    assert REASONING_LEVELS == ("default", "off", "low", "medium", "high")
    r = LLMResult("x")
    assert (r.reasoning, r.reasoning_tokens, r.reasoned, r.note) == ("", None, None, "")


@pytest.mark.parametrize("raw,answer,thought", [
    ("<think>plan it</think>\nHello.", "Hello.", "plan it"),
    ("<think>never ends", "", "never ends"),
    ("just text", "just text", ""),
    ("plan it</think>Hello.", "Hello.", "plan it"),
    ("<think>a</think>X<think>b</think>Y", "XY", "a\n\nb"),
])
def test_split_think(raw, answer, thought):
    assert split_think(raw) == (answer, thought)


# ---- fake OpenAI-compatible server ----

def make_server(handler_fn):
    log: list[dict] = []

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_GET(self):
            self.send_response(404)
            self.send_header("Content-Length", "0")
            self.end_headers()

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
            log.append(body)
            code, payload = handler_fn(body)
            data = json.dumps(payload).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

    srv = ThreadingHTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, f"http://127.0.0.1:{srv.server_address[1]}/v1", log


def chat(message: dict, usage: dict | None = None) -> dict:
    d = {"id": "x", "object": "chat.completion", "created": 0, "model": "m",
         "choices": [{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", **message}}]}
    if usage:
        d["usage"] = usage
    return d


@pytest.fixture
def serve():
    servers = []

    def start(fn):
        srv, url, log = make_server(fn)
        servers.append(srv)
        return url, log

    yield start
    for s in servers:
        s.shutdown()
        s.server_close()


def backend(url, reasoning="default"):
    return OpenAICompatBackend("m", None, url, reasoning=reasoning)


def test_reasoning_content_field(serve):
    url, log = serve(lambda b: (200, chat({"content": "Hi.", "reasoning_content": "thought hard"})))
    r = backend(url).complete("s", "t")
    assert (r.text, r.reasoning, r.reasoned) == ("Hi.", "thought hard", True)
    assert "reasoning_effort" not in log[0]


def test_reasoning_field_ollama_style(serve):
    url, _ = serve(lambda b: (200, chat({"content": "Hi.", "reasoning": "ollama thoughts"})))
    r = backend(url).complete("s", "t")
    assert (r.text, r.reasoning) == ("Hi.", "ollama thoughts")


def test_inline_think(serve):
    url, _ = serve(lambda b: (200, chat({"content": "<think>hmm</think>\n\nHi."})))
    r = backend(url).complete("s", "t")
    assert (r.text, r.reasoning, r.reasoned) == ("Hi.", "hmm", True)


def test_unterminated_inline_think(serve):
    url, _ = serve(lambda b: (200, chat({"content": "<think>cut off"})))
    r = backend(url).complete("s", "t")
    assert (r.text, r.reasoning) == ("", "cut off")


def test_reasoning_tokens_usage(serve):
    usage = {"prompt_tokens": 1, "completion_tokens": 9, "total_tokens": 10,
             "completion_tokens_details": {"reasoning_tokens": 7}}
    url, _ = serve(lambda b: (200, chat({"content": "Hi."}, usage)))
    r = backend(url).complete("s", "t")
    assert (r.reasoning_tokens, r.reasoned, r.reasoning) == (7, True, "")
    usage["completion_tokens_details"]["reasoning_tokens"] = 0
    r = backend(url).complete("s", "t")
    assert (r.reasoning_tokens, r.reasoned) == (0, False)


def test_unknown_when_no_signal(serve):
    url, _ = serve(lambda b: (200, chat({"content": "Hi."})))
    r = backend(url).complete("s", "t")
    assert r.reasoned is None and r.reasoning_tokens is None


@pytest.mark.parametrize("level,sent", [("default", None), ("off", "none"), ("low", "low"),
                                        ("medium", "medium"), ("high", "high")])
def test_reasoning_effort_only_when_not_default(serve, level, sent):
    url, log = serve(lambda b: (200, chat({"content": "ok"})))
    backend(url, level).complete("s", "t")
    assert log[0].get("reasoning_effort") == sent
    assert ("reasoning_effort" in log[0]) == (level != "default")


def test_400_retries_without_param(serve):
    def fn(b):
        if "reasoning_effort" in b:
            return 400, {"error": {"message": "Unknown parameter: reasoning_effort", "type": "invalid_request_error"}}
        return 200, chat({"content": "ok"})

    url, log = serve(fn)
    r = backend(url, "high").complete("s", "t")
    assert r.text == "ok" and r.note == "server ignored reasoning setting"
    assert "reasoning_effort" in log[0] and all("reasoning_effort" not in b for b in log[1:])


def test_unrelated_400_still_raises(serve):
    from stt_app.llm import LLMError

    url, _ = serve(lambda b: (400, {"error": {"message": "context too long"}}))
    with pytest.raises(LLMError):
        backend(url, "high").complete("s", "t")


# ---- Anthropic ----

def test_anthropic_parse_thinking_blocks():
    resp = NS(
        content=[NS(type="thinking", thinking="step 1"), NS(type="text", text="Final."),
                 NS(type="redacted_thinking")],
        usage=NS(output_tokens_details=NS(thinking_tokens=12)),
    )
    r = AnthropicBackend._parse(resp)
    assert (r.text, r.reasoning, r.reasoning_tokens, r.reasoned) == ("Final.", "step 1", 12, True)


def test_anthropic_parse_no_thinking_unknown():
    resp = NS(content=[NS(type="text", text="A")], usage=NS(output_tokens_details=None))
    r = AnthropicBackend._parse(resp)
    assert (r.text, r.reasoned, r.reasoning_tokens) == ("A", None, None)


def _anthropic_bad_request(msg):
    req = httpx.Request("POST", "https://x")
    return anthropic.BadRequestError(msg, response=httpx.Response(400, request=req), body=None)


def _patched(b, fn):
    b._client = NS(messages=NS(create=fn))
    return b


def test_anthropic_params_per_level():
    seen = []
    ok = NS(content=[NS(type="text", text="ok")], usage=None)

    def create(**kw):
        seen.append(kw)
        return ok

    AnthropicBackend("claude-haiku-5-5", "k", "default")
    _patched(AnthropicBackend("m", "k", "default"), create).complete("s", "t")
    assert "thinking" not in seen[-1] and seen[-1]["temperature"] == 0.2
    _patched(AnthropicBackend("m", "k", "off"), create).complete("s", "t")
    assert seen[-1]["thinking"] == {"type": "disabled"}
    _patched(AnthropicBackend("m", "k", "medium"), create).complete("s", "t")
    assert seen[-1]["thinking"]["type"] == "adaptive" and seen[-1]["output_config"] == {"effort": "medium"}
    assert "temperature" not in seen[-1]


def test_anthropic_falls_back_to_budget_then_none():
    seen = []

    def create(**kw):
        seen.append(kw)
        t = kw.get("thinking", {}).get("type")
        if t == "adaptive":
            raise _anthropic_bad_request("adaptive thinking is not supported on this model")
        return NS(content=[NS(type="text", text="ok")], usage=None)

    r = _patched(AnthropicBackend("old", "k", "high"), create).complete("s", "t")
    assert r.text == "ok" and r.note == ""
    last = seen[-1]
    assert last["thinking"]["type"] == "enabled" and last["max_tokens"] > last["thinking"]["budget_tokens"]

    def always(**kw):
        if "thinking" in kw:
            raise _anthropic_bad_request("thinking is not allowed")
        return NS(content=[NS(type="text", text="ok")], usage=None)

    r = _patched(AnthropicBackend("old", "k", "low"), always).complete("s", "t")
    assert r.note == "model rejected the reasoning setting"


def test_anthropic_drops_temperature_on_400():
    seen = []

    def create(**kw):
        seen.append(kw)
        if "temperature" in kw:
            raise _anthropic_bad_request("temperature is not supported on this model")
        return NS(content=[NS(type="text", text="ok")], usage=None)

    r = _patched(AnthropicBackend("m", "k"), create).complete("s", "t")
    assert r.text == "ok" and "temperature" not in seen[-1]


# ---- Google ----

def test_google_parse_thought_parts():
    resp = gtypes.GenerateContentResponse(
        candidates=[gtypes.Candidate(content=gtypes.Content(parts=[
            gtypes.Part(text="considering", thought=True), gtypes.Part(text="Answer.")]))],
        usage_metadata=gtypes.GenerateContentResponseUsageMetadata(thoughts_token_count=33),
    )
    r = GoogleBackend._parse(resp)
    assert (r.text, r.reasoning, r.reasoning_tokens, r.reasoned) == ("Answer.", "considering", 33, True)


def test_google_parse_no_thoughts():
    resp = gtypes.GenerateContentResponse(
        candidates=[gtypes.Candidate(content=gtypes.Content(parts=[gtypes.Part(text="A")]))],
        usage_metadata=gtypes.GenerateContentResponseUsageMetadata(prompt_token_count=3),
    )
    r = GoogleBackend._parse(resp)
    assert (r.text, r.reasoned) == ("A", False)


@pytest.mark.parametrize("model,level,field,value", [
    ("gemini-2.5-flash", "off", "thinking_budget", 0),
    ("gemini-2.5-pro", "off", "thinking_budget", 128),
    ("gemini-2.5-flash", "high", "thinking_budget", 24576),
    ("gemini-3-pro-preview", "low", "thinking_level", gtypes.ThinkingLevel.LOW),
    ("gemini-3-flash", "off", "thinking_level", gtypes.ThinkingLevel.MINIMAL),
])
def test_google_thinking_config(model, level, field, value):
    cfg = GoogleBackend.__new__(GoogleBackend)
    cfg.model, cfg.reasoning = model, level
    assert getattr(cfg._thinking(), field) == value
    cfg.reasoning = "default"
    assert cfg._thinking() is None


def test_google_retry_without_thinking_on_400():
    b = GoogleBackend.__new__(GoogleBackend)
    b.model, b.reasoning = "gemini-2.5-pro", "off"
    calls = []

    def gen(model, contents, config):
        calls.append(config.thinking_config)
        if config.thinking_config is not None:
            raise gerrors.ClientError(400, {"error": {"message": "thinking budget not supported"}})
        return gtypes.GenerateContentResponse(
            candidates=[gtypes.Candidate(content=gtypes.Content(parts=[gtypes.Part(text="ok")]))])

    b._client = NS(models=NS(generate_content=gen))
    r = b.complete("s", "t")
    assert r.text == "ok" and r.note == "model rejected the reasoning setting"
    assert calls[0] is not None and calls[1] is None
