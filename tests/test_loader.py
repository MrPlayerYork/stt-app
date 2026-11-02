import threading

from stt_app import ui
from stt_app.config import AppConfig


class FakeTranscriber:
    description = None
    gate = threading.Event()
    fail = False

    def __init__(self, *a, **kw):
        pass

    def load(self):
        self.gate.wait(2)
        if self.fail:
            raise RuntimeError("boom")
        self.description = "fake on cpu"


def make(monkeypatch, fail=False):
    monkeypatch.setattr(ui, "Transcriber", FakeTranscriber)
    monkeypatch.setattr(ui, "get_api_key", lambda *a, **k: None)
    FakeTranscriber.gate = threading.Event()
    FakeTranscriber.fail = fail
    return ui.ModelLoader(AppConfig())


def test_busy_until_loaded(monkeypatch):
    loader = make(monkeypatch)
    assert loader.busy
    assert str(loader.status) == "loading..."
    FakeTranscriber.gate.set()
    assert loader.done.wait(2)
    assert not loader.busy
    assert str(loader.status) == "fake on cpu"


def test_not_busy_after_error(monkeypatch):
    loader = make(monkeypatch, fail=True)
    FakeTranscriber.gate.set()
    assert loader.done.wait(2)
    assert not loader.busy
    assert "boom" in str(loader.status)
