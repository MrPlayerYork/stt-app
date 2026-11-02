"""faster-whisper transcription with CUDA preloading and CPU fallback."""
from __future__ import annotations

import ctypes
import glob
import importlib.util
import os

import numpy as np

MODELS = ["large-v3", "large-v3-turbo", "distil-large-v3", "medium", "small"]

_preloaded = False


class TranscriberError(Exception):
    pass


def _preload_cuda_libs() -> None:
    """Load pip-wheel CUDA libs globally; LD_LIBRARY_PATH is too late for a running process."""
    global _preloaded
    if _preloaded:
        return
    _preloaded = True
    spec = importlib.util.find_spec("nvidia")
    roots = list(spec.submodule_search_locations or []) if spec else []
    # Order matters: cublasLt before cublas, cudnn_graph before the other cudnn libs.
    patterns = [
        ("cuda_nvrtc", ["libnvrtc-builtins.so*", "libnvrtc.so*"]),
        ("cublas", ["libcublasLt.so*", "libcublas.so*"]),
        ("cudnn", ["libcudnn_graph.so*", "libcudnn_ops.so*", "libcudnn_cnn.so*", "libcudnn_adv.so*",
                   "libcudnn_engines_precompiled.so*", "libcudnn_engines_runtime_compiled.so*",
                   "libcudnn_heuristic.so*", "libcudnn.so*"]),
    ]
    for root in roots:
        for sub, pats in patterns:
            for pat in pats:
                for path in sorted(glob.glob(os.path.join(root, sub, "lib", pat))):
                    try:
                        ctypes.CDLL(path, mode=ctypes.RTLD_GLOBAL)
                    except OSError:
                        pass


class Transcriber:
    def __init__(self, model: str = "large-v3-turbo", device: str = "auto",
                 compute_type: str = "auto", language: str | None = "en"):
        self.model_name = model
        self.device = device
        self.compute_type = compute_type
        self.language = language
        self._model = None
        self._description: str | None = None

    @property
    def description(self) -> str | None:
        return self._description

    def _attempts(self) -> list[tuple[str, str]]:
        def ct(dev: str) -> str:
            if self.compute_type != "auto":
                return self.compute_type
            return "int8_float16" if dev == "cuda" else "int8"
        if self.device in ("auto", "cuda"):
            return [("cuda", ct("cuda")), ("cpu", "int8")]
        return [(self.device, ct(self.device))]

    def load(self) -> str:
        if self._model is not None and self._description:
            return self._description
        _preload_cuda_libs()
        try:
            from faster_whisper import WhisperModel
        except ImportError as e:
            raise TranscriberError(f"faster-whisper is not available: {e}") from e
        errors: list[str] = []
        for dev, ctype in self._attempts():
            try:
                model = WhisperModel(self.model_name, device=dev, compute_type=ctype)
                # CUDA problems can surface only on first inference, so warm up here.
                silence = np.zeros(TARGET_WARMUP, dtype=np.float32)
                list(model.transcribe(silence, language=self.language, vad_filter=False)[0])
            except Exception as e:
                errors.append(f"{dev}/{ctype}: {type(e).__name__}: {e}")
                self._model = None
                continue
            self._model = model
            self._description = f"{self.model_name} on {dev} ({ctype})"
            return self._description
        raise TranscriberError("Could not load model '%s': %s" % (self.model_name, " | ".join(errors)))

    def transcribe(self, audio: np.ndarray) -> str:
        if audio is None or audio.size == 0:
            return ""
        self.load()
        try:
            segments, _ = self._model.transcribe(
                np.ascontiguousarray(audio, dtype=np.float32),
                language=self.language, vad_filter=True,
            )
            return " ".join(s.text.strip() for s in segments).strip()
        except Exception as e:
            raise TranscriberError(f"Transcription failed: {e}") from e

    def unload(self) -> None:
        self._model = None
        self._description = None
        import gc
        gc.collect()


TARGET_WARMUP = 8000  # 0.5 s at 16 kHz
