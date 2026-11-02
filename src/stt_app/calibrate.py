"""First-run microphone calibration: record all inputs, transcribe, score, pick the best."""
from __future__ import annotations

import difflib
import math
import re
import threading
from dataclasses import dataclass

import numpy as np
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.prompt import Prompt
from rich.table import Table

from .audio import InputDevice, MultiRecorder, list_input_devices
from .transcribe import Transcriber

TEST_PHRASE = (
    "The quick brown fox jumps over the lazy dog while seven bright purple whales sing quietly."
)

FLOOR_DB = -120.0
SILENT_DB = -60.0
QUIET_DB = -45.0
FRAME = 320  # 20 ms at 16 kHz


@dataclass
class MicScore:
    device: InputDevice
    transcript: str
    accuracy: float
    rms_db: float
    snr_db: float
    clip_frac: float
    score: float
    error: str | None = None


def _words(text: str) -> list[str]:
    return re.sub(r"[^\w\s']", " ", text.lower()).replace("'", "").split()


def score_transcript(expected: str, got: str) -> float:
    a, b = _words(expected), _words(got)
    if not a or not b:
        return 1.0 if a == b else 0.0
    return float(difflib.SequenceMatcher(None, a, b).ratio())


def _db(x: float) -> float:
    return max(FLOOR_DB, 20.0 * math.log10(x)) if x > 0 else FLOOR_DB


def analyze_audio(audio: np.ndarray) -> tuple[float, float, float]:
    """Return (rms_db, snr_db, clip_frac). Silent/empty input gives (-120, 0, 0)."""
    x = np.asarray(audio, dtype=np.float64).ravel()
    if x.size == 0:
        return FLOOR_DB, 0.0, 0.0
    clip = float(np.mean(np.abs(x) >= 0.99))
    n = x.size // FRAME
    if n < 2:
        return _db(float(np.sqrt(np.mean(x**2)))), 0.0, clip
    e = np.mean(x[: n * FRAME].reshape(n, FRAME) ** 2, axis=1)
    k = max(1, n // 5)
    s = np.sort(e)
    loud, quiet = float(np.mean(s[-k:])), float(np.mean(s[:k]))
    if loud <= 0:
        return FLOOR_DB, 0.0, clip
    rms_db = _db(math.sqrt(loud))  # speech level = loudest frames
    snr = 10.0 * math.log10(loud / max(quiet, 1e-12))
    return rms_db, float(min(snr, 90.0)), clip


def combine(accuracy: float, rms_db: float, snr_db: float, clip_frac: float) -> float:
    score = 100.0 * accuracy + 0.5 * min(max(snr_db, 0.0), 40.0)
    score -= 200.0 * clip_frac
    if rms_db < QUIET_DB:
        score -= (QUIET_DB - max(rms_db, FLOOR_DB)) * 0.5
    return float(score)


def rank(scores: list[MicScore]) -> list[MicScore]:
    return sorted(scores, key=lambda s: (s.error is not None, -s.score))


# ---------------------------------------------------------------- UI

def _bar(level: float, width: int = 30) -> str:
    db = _db(level)
    frac = min(1.0, max(0.0, (db + 60.0) / 60.0))
    n = int(round(frac * width))
    colour = "red" if db > -3 else "green" if db > -35 else "yellow"
    return f"[{colour}]{'█' * n}[/]{'░' * (width - n)}"


def _level_table(devices: list[InputDevice], levels: dict[int, float], failed: dict[int, str]) -> Table:
    t = Table(title="Recording... read the phrase", expand=False)
    t.add_column("Device")
    t.add_column("Level")
    for d in devices:
        if d.index in failed:
            t.add_row(d.name, "[red]failed to open[/]")
        else:
            t.add_row(d.name, _bar(levels.get(d.index, 0.0)))
    return t


def _snippet(text: str, n: int = 50) -> str:
    return text if len(text) <= n else text[: n - 1] + "…"


def results_table(ranked: list[MicScore]) -> Table:
    t = Table(title="Microphone ranking")
    t.add_column("#", justify="right")
    t.add_column("Device")
    t.add_column("Accuracy", justify="right")
    t.add_column("Level dB", justify="right")
    t.add_column("SNR dB", justify="right")
    t.add_column("Transcript")
    best = next((s for s in ranked if s.error is None), None)
    for i, s in enumerate(ranked, 1):
        if s.error:
            t.add_row(str(i), s.device.name, "-", "-", "-", f"[red]{s.error}[/]")
            continue
        mark = " [bold green](suggested)[/]" if s is best else ""
        t.add_row(str(i), s.device.name + mark, f"{s.accuracy * 100:.0f}%",
                  f"{s.rms_db:.1f}", f"{s.snr_db:.1f}", _snippet(s.transcript))
    return t


def _record(console: Console, devices: list[InputDevice]):
    rec = MultiRecorder(devices)
    failed = rec.start()
    stop = threading.Event()
    try:
        with Live(_level_table(devices, rec.levels(), failed), console=console,
                  auto_refresh=False) as live:
            def pump() -> None:
                while not stop.is_set():
                    live.update(_level_table(devices, rec.levels(), failed), refresh=True)
                    stop.wait(0.1)
            th = threading.Thread(target=pump, daemon=True)
            th.start()
            try:
                console.input("[bold]Press Enter when you finish the phrase[/] ")
            finally:
                stop.set()
                th.join(timeout=1)
    except (KeyboardInterrupt, EOFError):
        rec.stop()
        return None
    return rec.stop(), failed


def _evaluate(console: Console, transcriber: Transcriber, devices: list[InputDevice],
              audio: dict[int, np.ndarray], failed: dict[int, str]) -> list[MicScore]:
    results: list[MicScore] = []
    todo: list[tuple[InputDevice, np.ndarray, float, float, float]] = []
    for d in devices:
        if d.index in failed:
            results.append(MicScore(d, "", 0.0, FLOOR_DB, 0.0, 0.0, -1e9, f"failed to open: {failed[d.index]}"))
            continue
        a = audio.get(d.index, np.zeros(0, dtype=np.float32))
        rms, snr, clip = analyze_audio(a)
        if rms < SILENT_DB:
            results.append(MicScore(d, "", 0.0, rms, snr, clip, -1e9,
                                    "silent (likely an output monitor)"))
        else:
            todo.append((d, a, rms, snr, clip))
    if todo:
        with console.status("Loading speech model (first use may download it)...") as status:
            def show(done: int, total: int, name: str) -> None:
                if done < total:
                    status.update(f"Downloading {name} {done * 100 // max(total, 1)}%...")
                else:
                    status.update("Loading speech model...")
            try:
                transcriber.load(show)
            except Exception as e:
                for d, _, rms, snr, clip in todo:
                    results.append(MicScore(d, "", 0.0, rms, snr, clip, -1e9, f"model load failed: {e}"))
                todo = []
        for n, (d, a, rms, snr, clip) in enumerate(todo, 1):
            with console.status(f"Transcribing {n}/{len(todo)}: {d.name}"):
                try:
                    text = transcriber.transcribe(a)
                except Exception as e:
                    results.append(MicScore(d, "", 0.0, rms, snr, clip, -1e9, f"transcription error: {e}"))
                    continue
            acc = score_transcript(TEST_PHRASE, text)
            results.append(MicScore(d, text, acc, rms, snr, clip, combine(acc, rms, snr, clip)))
    return rank(results)


def run_calibration(console: Console, transcriber: Transcriber) -> str | None:
    """Interactive mic selection. Returns chosen device name, or None if cancelled."""
    while True:
        devices = list_input_devices()
        if not devices:
            console.print("[red]No input devices found.[/]")
            return None
        console.print(Panel(f"[bold cyan]{TEST_PHRASE}[/]", title="Read this aloud", padding=(1, 2)))
        try:
            console.input("[bold]Press Enter to start recording[/] ")
        except (KeyboardInterrupt, EOFError):
            return None
        rec = _record(console, devices)
        if rec is None:
            console.print("[yellow]Cancelled.[/]")
            return None
        audio, failed = rec
        for d in devices:
            if d.index in failed:
                console.print(f"[yellow]Could not open {d.name}: {failed[d.index]}[/]")
        ranked = _evaluate(console, transcriber, devices, audio, failed)
        console.print(results_table(ranked))
        if not any(s.error is None for s in ranked):
            console.print("[red]Every device failed or was silent.[/]")
            choices, default = ["r", "q"], "r"
            prompt = "[r]etry or [q]uit"
        else:
            choices = [str(i) for i, s in enumerate(ranked, 1) if s.error is None] + ["r", "q"]
            default = "1"
            prompt = "Pick a number, [r]etry, or [q]uit"
        try:
            ans = Prompt.ask(prompt, console=console, choices=choices, default=default)
        except (KeyboardInterrupt, EOFError):
            return None
        if ans == "q":
            return None
        if ans == "r":
            continue
        return ranked[int(ans) - 1].device.name
