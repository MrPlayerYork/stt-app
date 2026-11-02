"""Microphone discovery and recording via sounddevice."""
from __future__ import annotations

import threading
from dataclasses import dataclass

import numpy as np
import sounddevice as sd

TARGET_RATE = 16000
MAX_CHANNELS = 2


@dataclass(frozen=True)
class InputDevice:
    index: int
    name: str
    hostapi: str
    channels: int
    default_samplerate: float


def list_input_devices() -> list[InputDevice]:
    """Input devices, preferring the JACK host API (pipewire-jack)."""
    # Under PipeWire, ALSA exposes raw hw:* nodes (often with unsupported rates) plus generic
    # "default"/"pipewire"; the JACK API shows one friendly-named entry per real PipeWire source
    # and opens reliably at 48 kHz. Use it when present, else fall back to every host API.
    apis = sd.query_hostapis()
    devs = [
        InputDevice(i, d["name"], apis[d["hostapi"]]["name"], int(d["max_input_channels"]),
                    float(d["default_samplerate"]))
        for i, d in enumerate(sd.query_devices())
        if d["max_input_channels"] > 0
    ]
    jack = [d for d in devs if "jack" in d.hostapi.lower()]
    return jack or devs


def find_device(name: str) -> InputDevice | None:
    """Exact name match first, then case-insensitive substring."""
    devs = list_input_devices()
    for d in devs:
        if d.name == name:
            return d
    low = name.lower()
    return next((d for d in devs if low in d.name.lower()), None)


def to_mono_16k(data: np.ndarray, samplerate: float) -> np.ndarray:
    """Downmix to mono float32 and resample to 16 kHz."""
    x = np.asarray(data, dtype=np.float32)
    if x.ndim > 1:
        x = x.mean(axis=1)
    if x.size == 0:
        return np.zeros(0, dtype=np.float32)
    if int(samplerate) != TARGET_RATE:
        n = max(1, round(x.size * TARGET_RATE / samplerate))
        x = np.interp(np.linspace(0, x.size - 1, n), np.arange(x.size), x).astype(np.float32)
    return np.ascontiguousarray(x, dtype=np.float32)


def _rms(block: np.ndarray) -> float:
    return float(min(1.0, np.sqrt(np.mean(np.square(block, dtype=np.float64))))) if block.size else 0.0


class _Capture:
    """One InputStream accumulating blocks."""

    def __init__(self, device: int | None):
        info = sd.query_devices(device, "input")
        self.device = device
        self.samplerate = float(info["default_samplerate"])
        self.channels = max(1, min(MAX_CHANNELS, int(info["max_input_channels"])))
        self._blocks: list[np.ndarray] = []
        self._lock = threading.Lock()
        self._level = 0.0
        self._stream: sd.InputStream | None = None

    def _callback(self, indata, frames, time_info, status) -> None:
        block = indata.copy()
        with self._lock:
            self._blocks.append(block)
            self._level = _rms(block)

    def open(self) -> None:
        self._stream = sd.InputStream(
            device=self.device, channels=self.channels, samplerate=self.samplerate,
            dtype="float32", callback=self._callback,
        )
        self._stream.start()

    def close(self) -> np.ndarray:
        stream, self._stream = self._stream, None
        if stream is not None:
            stream.stop()
            stream.close()
        with self._lock:
            blocks, self._blocks, self._level = self._blocks, [], 0.0
        if not blocks:
            return np.zeros(0, dtype=np.float32)
        return to_mono_16k(np.concatenate(blocks), self.samplerate)

    @property
    def level(self) -> float:
        return self._level

    @property
    def active(self) -> bool:
        return self._stream is not None


class Recorder:
    """Records one input device; stop() returns mono 16 kHz float32."""

    def __init__(self, device: int | None = None):
        self._cap = _Capture(device)

    def start(self) -> None:
        if not self._cap.active:
            self._cap.open()

    def stop(self) -> np.ndarray:
        return self._cap.close()

    @property
    def level(self) -> float:
        return self._cap.level

    @property
    def recording(self) -> bool:
        return self._cap.active


class MultiRecorder:
    """Records several input devices simultaneously (for mic calibration)."""

    def __init__(self, devices: list[InputDevice]):
        self._caps: dict[int, _Capture] = {}
        self._devices = devices

    def start(self) -> dict[int, str]:
        errors: dict[int, str] = {}
        for d in self._devices:
            if d.index in self._caps:
                continue
            try:
                cap = _Capture(d.index)
                cap.open()
                self._caps[d.index] = cap
            except Exception as e:  # sd.PortAudioError, ValueError, ...
                errors[d.index] = str(e)
        return errors

    def stop(self) -> dict[int, np.ndarray]:
        caps, self._caps = self._caps, {}
        return {i: c.close() for i, c in caps.items()}

    def levels(self) -> dict[int, float]:
        return {i: c.level for i, c in self._caps.items()}
