"""Single-keypress terminal input (POSIX): cbreak mode plus a select-based reader."""
from __future__ import annotations

import contextlib
import os
import re
import select
import sys
import termios
import tty
from collections.abc import Iterator


@contextlib.contextmanager
def cbreak(fd: int | None = None, flush: bool = True) -> Iterator[None]:
    """No line buffering and no echo (Ctrl+C still raises); the old mode is always restored.

    flush=True discards pending type-ahead when entering (stray keys pressed during a long task).
    """
    fd = sys.stdin.fileno() if fd is None else fd
    try:
        old = termios.tcgetattr(fd)
    except termios.error:  # not a tty
        yield
        return
    try:
        tty.setcbreak(fd, termios.TCSAFLUSH if flush else termios.TCSANOW)
        yield
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)


def read_key(timeout: float | None = None, fd: int | None = None) -> str | None:
    """Return the next key ("\\n" for Enter, "\\x1b" for escape sequences), or None on timeout.

    Raises EOFError on end of input or Ctrl+D.
    """
    fd = sys.stdin.fileno() if fd is None else fd
    if not select.select([fd], [], [], timeout)[0]:
        return None
    data = os.read(fd, 32)  # a whole escape sequence arrives in one read; extra bytes are dropped
    if not data:
        raise EOFError
    key = data.decode("utf-8", "ignore")[:1] or "\x00"
    if key == "\x04":
        raise EOFError
    return "\n" if key == "\r" else key


_ESCAPE_SEQ = re.compile(r"\x1b\[[0-9;?]*[ -/]*[@-~]|\x1b.")  # CSI sequences (incl. bracketed paste) and Alt+key


def read_secret(fd: int | None = None, out=None) -> str:
    """Read a line without showing it, echoing one '*' per character so pastes are visible.

    Handles Backspace and Ctrl+U; Enter finishes; Ctrl+C raises KeyboardInterrupt; Ctrl+D on an
    empty line raises EOFError.
    """
    fd = sys.stdin.fileno() if fd is None else fd
    out = sys.stdout if out is None else out
    chars: list[str] = []

    def echo(s: str) -> None:
        out.write(s)
        out.flush()

    with cbreak(fd, flush=False):  # keep a paste that lands before cbreak is applied
        while True:
            data = os.read(fd, 4096)  # a paste usually arrives in one read
            if not data:
                raise EOFError
            for ch in _ESCAPE_SEQ.sub("", data.decode("utf-8", "ignore")):
                if ch in "\r\n":
                    echo("\n")
                    return "".join(chars)
                if ch == "\x03":
                    echo("\n")
                    raise KeyboardInterrupt
                if ch == "\x04" and not chars:
                    echo("\n")
                    raise EOFError
                if ch in "\x7f\x08":
                    if chars:
                        chars.pop()
                        echo("\b \b")
                elif ch == "\x15":
                    echo("\b \b" * len(chars))
                    chars.clear()
                elif ch.isprintable():
                    chars.append(ch)
                    echo("*")
