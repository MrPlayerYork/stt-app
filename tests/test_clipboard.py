"""Tests for stt_app.clipboard.copy tool selection and error handling (no real clipboard)."""
from __future__ import annotations

import subprocess

import pytest

from stt_app import clipboard
from stt_app.clipboard import ClipboardError, copy


@pytest.fixture
def env(monkeypatch):
    monkeypatch.delenv("WAYLAND_DISPLAY", raising=False)
    return monkeypatch


def _fake_which(available: set[str]):
    return lambda name: f"/usr/bin/{name}" if name in available else None


def _record_run(monkeypatch):
    calls: list[dict] = []

    def fake_run(cmd, **kwargs):
        calls.append({"cmd": cmd, "input": kwargs.get("input")})
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(clipboard.subprocess, "run", fake_run)
    return calls


def test_prefers_wl_copy_on_wayland(env):
    env.setenv("WAYLAND_DISPLAY", "wayland-0")
    env.setattr(clipboard.shutil, "which", _fake_which({"wl-copy", "xclip", "xsel"}))
    calls = _record_run(env)
    assert copy("hi") == "wl-copy"
    assert calls[0]["cmd"] == ["wl-copy"]
    assert calls[0]["input"] == b"hi"


def test_wayland_without_wl_copy_falls_back_to_xclip(env):
    env.setenv("WAYLAND_DISPLAY", "wayland-0")
    env.setattr(clipboard.shutil, "which", _fake_which({"xclip", "xsel"}))
    calls = _record_run(env)
    assert copy("x") == "xclip"
    assert calls[0]["cmd"] == ["xclip", "-selection", "clipboard"]


def test_x11_prefers_xclip_over_xsel(env):
    env.setattr(clipboard.shutil, "which", _fake_which({"wl-copy", "xclip", "xsel"}))
    calls = _record_run(env)
    assert copy("x") == "xclip"  # no WAYLAND_DISPLAY, so wl-copy is skipped
    assert calls[0]["cmd"] == ["xclip", "-selection", "clipboard"]


def test_falls_back_to_xsel(env):
    env.setattr(clipboard.shutil, "which", _fake_which({"xsel"}))
    calls = _record_run(env)
    assert copy("x") == "xsel"
    assert calls[0]["cmd"] == ["xsel", "--clipboard", "--input"]


def test_wl_copy_ignored_without_wayland_display(env):
    env.setattr(clipboard.shutil, "which", _fake_which({"wl-copy"}))
    _record_run(env)
    with pytest.raises(ClipboardError):
        copy("x")


def test_no_tool_raises_clipboard_error(env):
    env.setattr(clipboard.shutil, "which", _fake_which(set()))
    calls = _record_run(env)
    with pytest.raises(ClipboardError, match="No clipboard tool"):
        copy("x")
    assert calls == []


def test_nonzero_exit_raises_clipboard_error(env):
    env.setattr(clipboard.shutil, "which", _fake_which({"xclip"}))

    def failing_run(cmd, **kwargs):
        kwargs["stderr"].write(b"cannot open display\n")
        raise subprocess.CalledProcessError(1, cmd)

    env.setattr(clipboard.subprocess, "run", failing_run)
    with pytest.raises(ClipboardError, match="xclip failed.*cannot open display"):
        copy("x")


def test_timeout_raises_clipboard_error(env):
    env.setattr(clipboard.shutil, "which", _fake_which({"xsel"}))

    def timeout_run(cmd, **kwargs):
        raise subprocess.TimeoutExpired(cmd, 5)

    env.setattr(clipboard.subprocess, "run", timeout_run)
    with pytest.raises(ClipboardError, match="xsel timed out"):
        copy("x")


def test_missing_binary_oserror_raises_clipboard_error(env):
    env.setattr(clipboard.shutil, "which", _fake_which({"xclip"}))

    def oserr_run(cmd, **kwargs):
        raise FileNotFoundError("gone")

    env.setattr(clipboard.subprocess, "run", oserr_run)
    with pytest.raises(ClipboardError, match="xclip failed"):
        copy("x")
