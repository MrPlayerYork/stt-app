"""Single-keypress terminal input (POSIX): cbreak mode plus a select-based reader."""
from __future__ import annotations

import contextlib
import os
import select
import sys
import termios
import tty
from collections.abc import Iterator


@contextlib.contextmanager
def cbreak(fd: int | None = None) -> Iterator[None]:
    """No line buffering and no echo (Ctrl+C still raises); the old mode is always restored."""
    fd = sys.stdin.fileno() if fd is None else fd
    try:
        old = termios.tcgetattr(fd)
    except termios.error:  # not a tty
        yield
        return
    try:
        tty.setcbreak(fd)
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
