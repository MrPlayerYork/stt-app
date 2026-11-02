#!/usr/bin/env bash
# usage: vram_wrap.sh <cmd...>  -- runs cmd, samples VRAM every 0.5 s, prints baseline/peak/delta (MiB)
q() { nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1; }
log=$(mktemp)
base=$(q)
( while true; do q >> "$log"; sleep 0.5; done ) &
sp=$!
"$@"; rc=$?
kill $sp 2>/dev/null; wait $sp 2>/dev/null
peak=$(sort -n "$log" | tail -1); peak=${peak:-$base}
echo "VRAM_MIB baseline=$base peak=$peak delta=$((peak-base))"
rm -f "$log"
exit $rc
