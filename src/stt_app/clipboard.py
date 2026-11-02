"""Copy text to the system clipboard (wl-copy, xclip or xsel)."""
from __future__ import annotations

import os
import shutil
import subprocess
import tempfile


class ClipboardError(Exception):
    pass


def copy(text: str) -> str:
    """Copy text; return the tool used."""
    if os.environ.get("WAYLAND_DISPLAY") and shutil.which("wl-copy"):
        cmd = ["wl-copy"]
    elif shutil.which("xclip"):
        cmd = ["xclip", "-selection", "clipboard"]
    elif shutil.which("xsel"):
        cmd = ["xsel", "--clipboard", "--input"]
    else:
        raise ClipboardError("No clipboard tool found: install wl-clipboard (Wayland), xclip or xsel.")
    try:
        # wl-copy forks a server that inherits our pipes, so a PIPE would never hit EOF; use a file for stderr.
        with tempfile.TemporaryFile() as err:
            try:
                subprocess.run(cmd, input=text.encode(), check=True, timeout=5,
                               stdout=subprocess.DEVNULL, stderr=err)
            except subprocess.CalledProcessError as e:
                err.seek(0)
                e.stderr = err.read()
                raise
    except subprocess.TimeoutExpired as e:
        raise ClipboardError(f"{cmd[0]} timed out") from e
    except (subprocess.CalledProcessError, OSError) as e:
        err = getattr(e, "stderr", b"") or b""
        raise ClipboardError(f"{cmd[0]} failed: {err.decode(errors='replace').strip() or e}") from e
    return cmd[0]
