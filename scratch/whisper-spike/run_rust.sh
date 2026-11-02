#!/usr/bin/env bash
# usage: run_rust.sh <model-file-name> --gpu|--cpu   (run inside the gpu board slot)
here=$(cd "$(dirname "$0")" && pwd)
M=<data-drive>/stt-models
W=${WAV:-<repo>/reference/gate.wav}
"$here/vram_wrap.sh" "$here/target/release/whisper-spike" "$M/$1" "$W" "$2" en 3 2>&1
