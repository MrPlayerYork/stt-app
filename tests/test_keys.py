import os
import termios

import pytest

from stt_app import keys


@pytest.fixture
def pty_pair():
    master, slave = os.openpty()
    yield master, slave
    os.close(master)
    os.close(slave)


def test_reads_one_key(pty_pair):
    master, slave = pty_pair
    with keys.cbreak(slave):  # without cbreak the line discipline would hold the byte back
        os.write(master, b"q")
        assert keys.read_key(1, slave) == "q"


def test_enter_is_newline(pty_pair):
    master, slave = pty_pair
    with keys.cbreak(slave):
        os.write(master, b"\r")
        assert keys.read_key(1, slave) == "\n"


def test_timeout_returns_none(pty_pair):
    _, slave = pty_pair
    with keys.cbreak(slave):
        assert keys.read_key(0.05, slave) is None


def test_ctrl_d_raises_eof(pty_pair):
    master, slave = pty_pair
    with keys.cbreak(slave):
        os.write(master, b"\x04")
        with pytest.raises(EOFError):
            keys.read_key(1, slave)


def test_cbreak_restores_mode_on_error(pty_pair):
    _, slave = pty_pair
    before = termios.tcgetattr(slave)
    with pytest.raises(RuntimeError), keys.cbreak(slave):
        assert termios.tcgetattr(slave) != before
        raise RuntimeError
    assert termios.tcgetattr(slave) == before


def test_cbreak_on_non_tty_is_noop():
    r, w = os.pipe()
    try:
        with keys.cbreak(r):
            pass
    finally:
        os.close(r)
        os.close(w)



class _Out:
    def __init__(self):
        self.text = ""

    def write(self, s):
        self.text += s

    def flush(self):
        pass


# Bytes are written after entering cbreak so the pty line discipline doesn't pre-process them.
# Ctrl+C isn't testable here: with ISIG on, the pty turns it into a signal instead of a byte.
def test_secret_paste_is_masked(pty_pair):
    master, slave = pty_pair
    out = _Out()
    with keys.cbreak(slave):
        os.write(master, b"\x1b[200~sk-abc123\x1b[201~\r")  # bracketed paste
        assert keys.read_secret(slave, out) == "sk-abc123"
    assert out.text == "*" * 9 + "\n"


def test_secret_backspace_and_clear(pty_pair):
    master, slave = pty_pair
    with keys.cbreak(slave):
        os.write(master, b"junk\x15ab\x7fc\r")
        assert keys.read_secret(slave, _Out()) == "ac"
