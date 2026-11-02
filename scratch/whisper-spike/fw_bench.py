"""faster-whisper bench using the STT app's own Transcriber setup. Run:
uv run --project $MAIN python -I scratch/whisper-spike/fw_bench.py <wav>
"""
import sys
import time
import wave

import numpy as np

from stt_app.config import load_config, resolved_models_dir
from stt_app.transcribe import Transcriber, _preload_cuda_libs

app = load_config()
cfg = app.whisper
with wave.open(sys.argv[1]) as w:
    audio = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32) / 32768.0
print(f"audio: {len(audio)/16000:.2f}s; config model={cfg.model} device={cfg.device} compute={cfg.compute_type} lang={cfg.language}")

t = time.perf_counter()
tr = Transcriber(cfg.model, cfg.device, cfg.compute_type, cfg.language,
                 models_dir=resolved_models_dir(app), connections=cfg.download_connections)
desc = tr.load()  # includes the app's 0.5 s silence warm-up
print(f"LOAD_S {time.perf_counter()-t:.3f} ({desc})")

best = 1e9
text = ""
for i in range(3):
    t = time.perf_counter()
    text = tr.transcribe(audio)  # app's own path: vad_filter=True, default beam 5
    dt = time.perf_counter() - t
    print(f"RUN {i+1} TRANSCRIBE_S {dt:.3f}")
    best = min(best, dt)
print(f"BEST_S {best:.3f}")
print("TEXT:", text)
