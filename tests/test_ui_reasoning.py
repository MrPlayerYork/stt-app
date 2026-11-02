from rich.console import Console

from stt_app import ui
from stt_app.config import AppConfig
from stt_app.ui import LLMResult, _as_result, reasoning_line


def render(res):
    c = Console(record=True, width=120)
    c.print(reasoning_line(res))
    return c.export_text().strip()


def test_line_variants():
    assert render(LLMResult("x", "because", 1234, True)) == "Reasoning: 1,234 tokens · [v] view"
    assert render(LLMResult("x", "", 1234, True)) == "Reasoning: 1,234 tokens (text not exposed by provider)"
    assert render(LLMResult("x", reasoned=False)) == "No reasoning"
    assert render(LLMResult("x")) == "Reasoning: unknown"
    assert render(LLMResult("x", reasoned=False, note="ignored")).endswith("· ignored")


def test_as_result():
    r = _as_result("hello")
    assert isinstance(r, LLMResult) and r.text == "hello" and r.reasoning == ""
    orig = LLMResult("a", "b")
    assert _as_result(orig) is orig


def make_app(**llm):
    cfg = AppConfig()
    cfg.llm.enabled = True
    for k, v in llm.items():
        setattr(cfg.llm, k, v)
    return ui.App(cfg, Console(record=True, width=120))


def header_text(app):
    c = Console(record=True, width=120)
    c.print(app.header())
    return c.export_text()


def test_header_extras():
    assert "reasoning" not in header_text(make_app())
    assert "· reasoning: high" in header_text(make_app(reasoning="high"))
    app = make_app(mode="prompt", word_choice=True)
    assert "· word choice" in header_text(app)
    assert "word choice" not in header_text(make_app(mode="cleanup", word_choice=True))
    off = make_app(reasoning="high")
    off.cfg.llm.enabled = False
    assert "reasoning:" not in header_text(off)


def test_menu_offers_v_only_with_reasoning_text(monkeypatch):
    seen = []

    def fake_read_key(timeout=None):
        return "q"

    monkeypatch.setattr(ui.keys, "read_key", fake_read_key)
    monkeypatch.setattr(ui.keys, "cbreak", lambda *a, **k: __import__("contextlib").nullcontext())

    app = make_app()
    app.last_result = LLMResult("x", reasoning_tokens=5, reasoned=True)
    assert app.main_screen() == "q"
    assert "[v] reasoning" not in app.console.export_text()

    app = make_app()
    app.last_result = LLMResult("x", "thinking...", 5, True)
    assert app.main_screen() == "q"
    assert "[v] reasoning" in app.console.export_text()
    seen.append(1)


def test_view_reasoning_prints_panel(monkeypatch):
    app = make_app()
    monkeypatch.setattr(app, "pause", lambda: None)
    app.last_result = LLMResult("x", "step one", 3, True)
    app.view_reasoning()
    out = app.console.export_text()
    assert "Reasoning" in out and "step one" in out
