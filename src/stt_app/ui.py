"""Rich-based terminal UI: simple menus, no full-screen TUI."""
from __future__ import annotations

import contextlib
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass

from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.progress import BarColumn, DownloadColumn, Progress, TextColumn, TransferSpeedColumn
from rich.prompt import Confirm, Prompt
from rich.table import Table
from rich.text import Text

from .audio import Recorder, find_device, list_input_devices
from .clipboard import ClipboardError, copy
from .config import AppConfig, get_api_key, load_config, resolved_models_dir, save_config, set_api_key
from .llm import (
    DEFAULT_MODELS,
    LOCAL_PRESETS,
    PROVIDERS,
    LLMError,
    all_presets,
    make_backend,
    process,
)
from . import keys
from .download import DownloadError, ensure_model
from .transcribe import MODELS, Transcriber, TranscriberError

MODE_ORDER = ["raw", "cleanup", "prompt"]
SAMPLE_TEXT = "um so i think we should uh meet on tuesday no wait wednesday at three pm"


@dataclass
class HistoryItem:
    raw: str
    final: str
    mode: str


class ModelLoader:
    """Loads a Transcriber in a background thread and tracks its status."""

    def __init__(self, cfg: AppConfig):
        w = cfg.whisper
        self.transcriber = Transcriber(
            w.model, w.device, w.compute_type, w.language, models_dir=resolved_models_dir(cfg),
            token=get_api_key("huggingface", "HF_TOKEN"), connections=w.download_connections,
            on_progress=self._on_progress)
        self.error: str | None = None
        self._sample: tuple[float, int] | None = None  # (time, bytes) of the previous callback
        self._dl: tuple[str, int, int, float, float] | None = None  # name, done, total, speed, time
        self.done = threading.Event()
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self) -> None:
        try:
            self.transcriber.load()
        except TranscriberError as e:
            self.error = str(e)
        except Exception as e:  # never let the loader thread die silently
            self.error = f"{type(e).__name__}: {e}"
        finally:
            self.done.set()

    def _on_progress(self, done: int, total: int, name: str) -> None:
        now = time.monotonic()
        prev = self._sample
        speed = (done - prev[1]) / max(now - prev[0], 1e-3) if prev and done >= prev[1] else 0.0
        self._sample = (now, done)
        self._dl = (name, done, total, speed, now)

    @property
    def busy(self) -> bool:
        """True while downloading or loading, i.e. while the status text can still change."""
        return not self.done.is_set()

    @property
    def status(self) -> Text:
        if not self.done.is_set():
            dl = self._dl
            if dl and dl[1] < dl[2] and time.monotonic() - dl[4] < 5:
                name, done, total, speed, _ = dl
                return Text(f"downloading {name} {done * 100 // max(total, 1)}% ({speed / 1e6:.1f} MB/s)",
                            style="yellow")
            return Text("loading...", style="yellow")
        if self.error:
            return Text(self.error[:160], style="red")
        return Text(self.transcriber.description or "ready", style="green")


class App:
    def __init__(self, cfg: AppConfig, console: Console | None = None):
        self.cfg = cfg
        self.console = console or Console()
        self.history: deque[HistoryItem] = deque(maxlen=max(1, cfg.history_size))
        self.last: HistoryItem | None = None
        self.loader: ModelLoader | None = None

    # ---- helpers -------------------------------------------------------

    def save(self) -> None:
        try:
            save_config(self.cfg)
        except OSError as e:
            self.console.print(f"[red]Could not save config: {e}[/]")

    def start_loading(self) -> None:
        if self.loader is not None:
            self.loader.transcriber.unload()
        self.loader = ModelLoader(self.cfg)

    def mode_key(self) -> str:
        llm = self.cfg.llm
        return "raw" if not llm.enabled else ("prompt" if llm.mode == "prompt" else "cleanup")

    def mode_label(self) -> str:
        key = self.mode_key()
        if key == "prompt":
            presets = all_presets(self.cfg.llm)
            return f"Prompt: {presets.get(self.cfg.llm.preset, self.cfg.llm.preset)}"
        return key.capitalize()

    def llm_configured(self) -> bool:
        llm = self.cfg.llm
        return bool(llm.model) and (llm.provider != "openai_compat" or bool(llm.base_url))

    def api_key(self) -> str | None:
        return get_api_key(self.cfg.llm.provider, self.cfg.llm.api_key_env)

    def ask(self, prompt: str, **kw) -> str:
        return Prompt.ask(prompt, console=self.console, **kw)

    def pause(self) -> None:
        self.ask("Press Enter to continue", default="", show_default=False)

    def pick(self, title: str, options: list[str], current: str | None = None) -> int | None:
        """Numbered pick; returns the index, or None on blank/invalid input."""
        for i, opt in enumerate(options, 1):
            mark = " (current)" if opt == current else ""
            self.console.print(Text.assemble((f"  {i} ", "cyan"), opt, (mark, "green")))
        raw = self.ask(f"{title} (number, Enter to cancel)", default="", show_default=False)
        if raw.isdigit() and 1 <= int(raw) <= len(options):
            return int(raw) - 1
        return None

    # ---- main screen ---------------------------------------------------

    def header(self) -> Panel:
        t = Table.grid(padding=(0, 2))
        t.add_row(Text("Microphone", style="bold"), Text(self.cfg.audio.device_name or "(system default)"))
        t.add_row(Text("Whisper", style="bold"), self.loader.status if self.loader else Text("-"))
        t.add_row(Text("Mode", style="bold"), Text(self.mode_label()))
        return Panel(t, title="[bold]Speech to Text[/]", border_style="blue")

    def result_panels(self, item: HistoryItem) -> list[Panel]:
        panels = [Panel(Text(item.raw), title="Raw transcript", border_style="dim")]
        if item.final != item.raw:
            panels.append(Panel(Text(item.final), title=f"Processed ({item.mode})", border_style="green"))
        return panels

    def show_result(self, item: HistoryItem) -> None:
        self.console.print(*self.result_panels(item))

    def choose(self, label: str, choices: list[str], default: str) -> str:
        """Single-keypress pick; Enter selects the default (which may be "")."""
        self.console.print(f"{label}: ", end="")
        with keys.cbreak():
            while True:
                key = keys.read_key(None)
                key = default if key == "\n" else key.lower()
                if key == default or key in choices:
                    break
        self.console.print(key)
        return key

    def main_screen(self) -> str:
        """Show the main screen and return the chosen key; the header updates live while loading."""
        choices = ["r", "m", "h", "s", "q"] + (["p"] if self.mode_key() == "prompt" else [])
        menu = Text("[r]ecord  [m]ode  " + ("[p]reset  " if "p" in choices else "")
                    + "[h]istory  [s]ettings  [q]uit", style="bold")

        def view() -> Group:
            parts = [self.header()]
            if self.last:
                parts += [Text("Last result:", style="dim"), *self.result_panels(self.last)]
            return Group(*parts, menu)

        loader = self.loader
        live: Live | None = None
        with keys.cbreak():
            try:
                if loader and loader.busy:
                    live = Live(console=self.console, refresh_per_second=4, get_renderable=view)
                    live.start()
                else:
                    self.console.print(view())
                while True:
                    key = keys.read_key(0.25 if live else None)
                    if live and not loader.busy:
                        live.stop()  # leaves the final state on screen
                        live = None
                    if key == "\n":
                        return "r"
                    if key and key.lower() in choices:
                        return key.lower()
            finally:
                if live:
                    live.stop()

    def run(self) -> None:
        self.start_loading()
        try:
            while True:
                self.console.clear()
                choice = self.main_screen()
                if choice == "q":
                    break
                {"r": self.record, "m": self.cycle_mode, "p": self.choose_preset,
                 "h": self.show_history, "s": self.settings}[choice]()
        except (KeyboardInterrupt, EOFError):
            self.console.print()
        self.console.print("Bye.")

    # ---- recording -----------------------------------------------------

    def resolve_device(self) -> int | None:
        name = self.cfg.audio.device_name
        if name:
            dev = find_device(name)
            if dev:
                return dev.index
            self.console.print(f"[yellow]Microphone '{name}' not found; using the default device.[/]")
        else:
            self.console.print("[yellow]No microphone configured; using the default device.[/]")
        return None

    def capture(self):
        """Record until Enter; return audio, or None if cancelled or failed."""
        try:
            rec = Recorder(self.resolve_device())
            rec.start()
        except Exception as e:  # sounddevice.PortAudioError, ValueError
            self.console.print(f"[red]Could not open the microphone: {e}[/]")
            self.pause()
            return None
        t0 = time.monotonic()

        def view():
            n = int(min(1.0, rec.level * 6) * 30)  # speech RMS is small, so scale up
            bar = Text("█" * n, style="green") + Text("░" * (30 - n), style="dim")
            return Group(Text.assemble(("● REC ", "bold red"), f"{time.monotonic() - t0:5.1f}s  ", bar),
                         Text("Press Enter to stop, Ctrl+C to cancel", style="dim"))

        try:
            with Live(console=self.console, refresh_per_second=10, get_renderable=view):
                input()
        except (KeyboardInterrupt, EOFError):
            rec.stop()
            self.console.print("[yellow]Recording cancelled.[/]")
            time.sleep(0.8)
            return None
        return rec.stop()

    def record(self) -> None:
        audio = self.capture()
        if audio is None:
            return
        try:
            text = self.transcribe(audio)
        except TranscriberError as e:
            self.console.print(f"[red]{e}[/]")
            self.pause()
            return
        if not text:
            self.console.print("[yellow]Heard nothing.[/]")
            self.pause()
            return
        final, label = self.postprocess(text)
        item = HistoryItem(text, final, label)
        self.history.appendleft(item)
        self.last = item
        self.copy_text(final)
        self.show_result(item)
        self.pause()

    def transcribe(self, audio) -> str:
        loader = self.loader
        if not loader.done.is_set():
            with self.console.status("Waiting for the Whisper model..."):
                loader.done.wait()
        if loader.error:
            raise TranscriberError(loader.error)
        with self.console.status("Transcribing..."):
            return loader.transcriber.transcribe(audio)

    def postprocess(self, text: str) -> tuple[str, str]:
        label = self.mode_label()
        if not self.cfg.llm.enabled:
            return text, label
        try:
            with self.console.status("Processing with the LLM..."):
                return process(self.cfg.llm, self.api_key(), text), label
        except LLMError as e:
            self.console.print(f"[red]LLM error: {e}[/] [dim](using the raw text)[/]")
            return text, "Raw (LLM failed)"

    def copy_text(self, text: str) -> None:
        try:
            copy(text)
            self.console.print("[green]Copied to clipboard.[/]")
        except ClipboardError as e:
            self.console.print(f"[red]{e}[/]")

    # ---- mode / preset / history ----------------------------------------

    def cycle_mode(self) -> None:
        llm = self.cfg.llm
        nxt = MODE_ORDER[(MODE_ORDER.index(self.mode_key()) + 1) % 3]
        if nxt != "raw" and not self.llm_configured():
            self.console.print("[yellow]Set up an LLM in Settings first.[/]")
            llm.enabled = False
            self.save()
            self.pause()
            return
        llm.enabled = nxt != "raw"
        if nxt != "raw":
            llm.mode = nxt
        self.save()

    def choose_preset(self) -> None:
        presets = all_presets(self.cfg.llm)
        keys = list(presets)
        idx = self.pick("Preset", [presets[k] for k in keys], presets.get(self.cfg.llm.preset))
        if idx is not None:
            self.cfg.llm.preset = keys[idx]
            self.save()

    def show_history(self) -> None:
        if not self.history:
            self.console.print("[dim]No history yet.[/]")
            self.pause()
            return
        while True:
            t = Table(title="History (newest first)")
            t.add_column("#", style="cyan")
            t.add_column("Mode")
            t.add_column("Text", overflow="ellipsis", no_wrap=True, max_width=70)
            for i, h in enumerate(self.history, 1):
                t.add_row(str(i), h.mode, Text(h.final.replace("\n", " ")))
            self.console.print(t)
            raw = self.ask("Number to copy, Enter to go back", default="", show_default=False)
            if not raw:
                return
            if raw.isdigit() and 1 <= int(raw) <= len(self.history):
                self.copy_text(self.history[int(raw) - 1].final)
            else:
                self.console.print("[red]Not a valid number.[/]")

    # ---- settings ------------------------------------------------------

    def settings(self) -> None:
        actions = {"1": self.set_microphone, "2": self.set_whisper, "3": self.set_llm}
        while True:
            self.console.print(Panel("1  Microphone\n2  Whisper\n3  LLM\nb  Back", title="Settings"))
            c = self.choose("Settings", [*actions, "b"], "b")
            if c == "b":
                return
            actions[c]()

    def set_microphone(self) -> None:
        self.console.print("1  Run calibration\n2  Pick from a list")
        c = self.choose("Microphone", ["1", "2"], "")
        if c == "1":
            self.calibrate()
        elif c == "2":
            try:
                devs = list_input_devices()
            except Exception as e:  # sounddevice.PortAudioError
                self.console.print(f"[red]Could not list devices: {e}[/]")
                return
            if not devs:
                self.console.print("[red]No input devices found.[/]")
                return
            idx = self.pick("Device", [d.name for d in devs], self.cfg.audio.device_name)
            if idx is not None:
                self.cfg.audio.device_name = devs[idx].name
                self.save()

    def calibrate(self) -> str | None:
        from .calibrate import run_calibration

        if self.loader is None:
            self.start_loading()
        try:
            name = run_calibration(self.console, self.loader.transcriber)
        except (TranscriberError, OSError, ValueError) as e:
            self.console.print(f"[red]Calibration failed: {e}[/]")
            return None
        if name:
            self.cfg.audio.device_name = name
            self.save()
        return name

    def set_whisper(self) -> None:
        actions = {"1": self.set_model_options, "2": self.set_models_dir, "3": self.set_connections,
                   "4": self.set_hf_token, "5": self.download_now}
        while True:
            w = self.cfg.whisper
            t = Table.grid(padding=(0, 2))
            for k, label, val in [
                ("1", "Model / device / language", f"{w.model}, {w.device}, {w.language or 'auto'}"),
                ("2", "Models folder", str(resolved_models_dir(self.cfg)) + ("" if w.models_dir else " (default)")),
                ("3", "Download connections", str(w.download_connections)),
                ("4", "Hugging Face token", "set" if get_api_key("huggingface", "HF_TOKEN") else "not set"),
                ("5", "Download/verify current model now", ""),
                ("b", "Back", ""),
            ]:
                t.add_row(Text(k, style="cyan"), label, Text(val))
            self.console.print(Panel(t, title="Whisper settings"))
            c = self.choose("Whisper", [*actions, "b"], "b")
            if c == "b":
                return
            actions[c]()

    def set_models_dir(self) -> None:
        w = self.cfg.whisper
        new = self.ask("Models folder (blank = default, '-' keeps)", default="-", show_default=False).strip()
        if new != "-" and new != w.models_dir:
            w.models_dir = new
            self.save()
            self.start_loading()
            self.console.print("[dim]Reloading the model in the background.[/]")

    def set_connections(self) -> None:
        raw = self.ask("Connections per file (1-32, Enter keeps)", default="", show_default=False).strip()
        if raw.isdigit() and 1 <= int(raw) <= 32:
            self.cfg.whisper.download_connections = int(raw)
            self.save()
        elif raw:
            self.console.print("[red]Enter a number from 1 to 32.[/]")

    def set_hf_token(self) -> None:
        key = self.ask("Hugging Face token (Enter to keep)", password=True, default="", show_default=False)
        if key.strip():
            try:
                set_api_key("huggingface", key.strip())
                self.console.print("[green]Token stored.[/]")
            except Exception as e:  # keyring backends raise many types
                self.console.print(f"[red]Could not store the token in the keyring: {e}[/]")

    def download_now(self) -> None:
        w = self.cfg.whisper
        bar = Progress(TextColumn("{task.description}"), BarColumn(), DownloadColumn(), TransferSpeedColumn(),
                       console=self.console)
        task = bar.add_task("checking...", total=None)

        def show(done: int, total: int, name: str) -> None:
            bar.update(task, description=name, completed=done, total=total)

        try:
            with bar:
                path = ensure_model(w.model, resolved_models_dir(self.cfg), get_api_key("huggingface", "HF_TOKEN"),
                                    w.download_connections, show)
            self.console.print(f"[green]Model ready:[/] {path}")
        except DownloadError as e:
            self.console.print(f"[red]Download failed: {e}[/] [dim](run it again to resume)[/]")

    def set_model_options(self) -> None:
        w = self.cfg.whisper
        before = (w.model, w.device, w.language)
        idx = self.pick("Model", MODELS, w.model)
        if idx is not None:
            w.model = MODELS[idx]
        devices = ["auto", "cuda", "cpu"]
        idx = self.pick("Device", devices, w.device)
        if idx is not None:
            w.device = devices[idx]
        lang = self.ask(f"Language code (blank = auto-detect, '-' keeps '{w.language or 'auto'}')",
                        default="-", show_default=False)
        if lang != "-":
            w.language = lang.strip().lower() or None
        if (w.model, w.device, w.language) != before:
            self.save()
            self.start_loading()
            self.console.print("[dim]Reloading the model in the background.[/]")

    def set_llm(self) -> None:
        llm = self.cfg.llm
        actions = {"1": self.toggle_llm, "2": self.set_provider, "3": self.set_base_url,
                   "4": self.set_api_key, "5": self.set_model, "6": self.test_llm, "7": self.custom_prompts}
        while True:
            key_state = "set" if self.api_key() else "not set"
            t = Table.grid(padding=(0, 2))
            for k, label, val in [
                ("1", "Enabled", "yes" if llm.enabled else "no"),
                ("2", "Provider", PROVIDERS.get(llm.provider, llm.provider)),
                ("3", "Base URL", llm.base_url or "-"),
                ("4", "API key", key_state + (f" (env {llm.api_key_env})" if llm.api_key_env else "")),
                ("5", "Model", llm.model or "-"),
                ("6", "Test connection", ""),
                ("7", "Custom prompts", f"{len(llm.custom_prompts)} defined"),
                ("b", "Back", ""),
            ]:
                t.add_row(Text(k, style="cyan"), label, Text(val))
            self.console.print(Panel(t, title="LLM settings"))
            c = self.choose("LLM", [*actions, "b"], "b")
            if c == "b":
                return
            actions[c]()

    def toggle_llm(self) -> None:
        llm = self.cfg.llm
        if not llm.enabled and not self.llm_configured():
            self.console.print("[yellow]Choose a provider, server and model first.[/]")
            return
        llm.enabled = not llm.enabled
        self.save()

    def set_provider(self) -> None:
        keys = list(PROVIDERS)
        idx = self.pick("Provider", [PROVIDERS[k] for k in keys], PROVIDERS.get(self.cfg.llm.provider))
        if idx is None or keys[idx] == self.cfg.llm.provider:
            return
        self.cfg.llm.provider = keys[idx]
        self.cfg.llm.model = DEFAULT_MODELS.get(keys[idx], "")
        self.save()
        if keys[idx] == "openai_compat":
            self.set_base_url()

    def set_base_url(self) -> None:
        if self.cfg.llm.provider != "openai_compat":
            self.console.print("[dim]Only used by the local / OpenAI-compatible provider.[/]")
            return
        names = [*LOCAL_PRESETS, "Custom URL"]
        idx = self.pick("Server", [f"{n} ({LOCAL_PRESETS[n]})" if n in LOCAL_PRESETS else n for n in names])
        if idx is None:
            return
        if names[idx] in LOCAL_PRESETS:
            self.cfg.llm.base_url = LOCAL_PRESETS[names[idx]]
        else:
            url = self.ask("Base URL", default=self.cfg.llm.base_url or "http://localhost:1234/v1")
            self.cfg.llm.base_url = url.strip()
        self.save()

    def set_api_key(self) -> None:
        llm = self.cfg.llm
        self.console.print(f"Key currently {'set' if self.api_key() else 'not set'} for {PROVIDERS[llm.provider]}.")
        key = self.ask("New API key (Enter to keep)", password=True, default="", show_default=False)
        if key.strip():
            try:
                set_api_key(llm.provider, key.strip())
                self.console.print("[green]Key stored.[/]")
            except Exception as e:  # keyring backends raise many types
                self.console.print(f"[red]Could not store the key in the keyring: {e}[/]")
        env = self.ask("Env var holding the key (blank clears, '-' keeps)", default="-", show_default=False)
        if env != "-":
            llm.api_key_env = env.strip()
            self.save()

    def set_model(self) -> None:
        llm = self.cfg.llm
        models: list[str] = []
        try:
            with self.console.status("Fetching models..."):
                models = make_backend(llm, self.api_key()).list_models()
        except LLMError as e:
            self.console.print(f"[red]Could not list models: {e}[/]")
        if models:
            shown = models[:40]
            idx = self.pick("Model (Enter to type one instead)", shown, llm.model)
            if idx is not None:
                llm.model = shown[idx]
                self.save()
                return
        name = self.ask("Model name", default=llm.model or DEFAULT_MODELS.get(llm.provider, ""))
        if name.strip():
            llm.model = name.strip()
            self.save()

    def test_llm(self) -> None:
        self.console.print(Text(f"Input: {SAMPLE_TEXT}"))
        try:
            with self.console.status("Testing..."):
                out = process(self.cfg.llm, self.api_key(), SAMPLE_TEXT)
            self.console.print(Panel(Text(out or "(empty response)"), title="Result", border_style="green"))
        except LLMError as e:
            self.console.print(f"[red]Test failed: {e}[/]")

    def custom_prompts(self) -> None:
        prompts = self.cfg.llm.custom_prompts
        for n, p in prompts.items():
            self.console.print(Text.assemble((n, "cyan"), ": ", p[:70] + ("..." if len(p) > 70 else "")))
        c = self.choose("add/edit (a), delete (d), Enter to go back", ["a", "d"], "")
        if c == "a":
            name = self.ask("Prompt name").strip()
            if not name:
                return
            text = self.ask("System prompt", default=prompts.get(name, "")).strip()
            if text:
                prompts[name] = text
                self.save()
        elif c == "d" and prompts:
            names = list(prompts)
            idx = self.pick("Delete which", names)
            if idx is not None and Confirm.ask(f"Delete '{names[idx]}'?", console=self.console):
                del prompts[names[idx]]
                if self.cfg.llm.preset == names[idx]:
                    self.cfg.llm.preset = "deramble"
                self.save()


def first_run(app: App) -> None:
    app.console.print(Panel("Welcome! Let's pick your microphone. The Whisper model loads in the "
                            "background while we calibrate.", title="Speech to Text", border_style="blue"))
    app.start_loading()
    name = None
    with contextlib.suppress(KeyboardInterrupt):
        try:
            name = app.calibrate()
        except ImportError as e:
            app.console.print(f"[red]Calibration unavailable: {e}[/]")
    if not name:
        app.console.print("[yellow]No microphone chosen; using the default input device.[/]")
    app.save()


def main() -> None:
    console = Console()
    if not sys.stdin.isatty():
        console.print("[red]stt-app needs an interactive terminal (stdin is not a TTY).[/]")
        sys.exit(1)
    try:
        cfg = load_config()
        app = App(cfg or AppConfig(), console)
        # Also re-run when a previous first run ended without choosing a mic.
        if cfg is None or cfg.audio.device_name is None:
            first_run(app)
        app.run()
    except KeyboardInterrupt:
        console.print("\nBye.")
