#!/usr/bin/env bash
# Shared resource board for parallel agents on a home PC (see AGENTS.md).
# Slots are claim files; the log is append-only. Lives in the MAIN checkout so every worktree sees it.
set -euo pipefail

# The main checkout is the parent of git's common dir, the same from every worktree.
MAIN_DIR="$(cd "$(git -C "$(dirname "$0")" rev-parse --path-format=absolute --git-common-dir)/.." && pwd)"
BOARD_DIR="${STT_BOARD_DIR:-$MAIN_DIR/.agents}"
LOG="$BOARD_DIR/board.log"
SLOTS="$BOARD_DIR/slots"
mkdir -p "$SLOTS"

# Seconds after which an unreleased claim counts as stale (agent died or forgot to release).
declare -A TTL=([heavy]=3600 [gpu]=1200 [download]=2700)
LOAD_MAX="${STT_LOAD_MAX:-12}"   # 1-min load average ceiling (16 threads, leave room for the user)
MEM_MIN_GB="${STT_MEM_MIN_GB:-4}"

log() { printf '%s\t%s\t%s\n' "$(date '+%F %T')" "$1" "$2" >> "$LOG"; }

slot_file() {
    [[ -n "${TTL[$1]:-}" ]] || { echo "unknown slot '$1' (heavy|gpu|download)" >&2; exit 2; }
    echo "$SLOTS/$1"
}

try_claim() {  # slot agent note
    local f; f=$(slot_file "$1")
    if [[ -f $f ]] && (( $(date +%s) - $(stat -c %Y "$f") > TTL[$1] )); then
        log "$2" "took over stale $1 slot from: $(head -c 200 "$f")"
        rm -f "$f"
    fi
    ( set -o noclobber; printf '%s | %s | since %s\n' "$2" "$3" "$(date '+%T')" > "$f" ) 2>/dev/null || return 1
    log "$2" "claimed $1: $3"
}

claim() {  # slot agent note [max_wait_s]
    local waited=0 max="${4:-1800}"
    until try_claim "$1" "$2" "$3"; do
        (( waited == 0 )) && echo "waiting for $1 slot, held by: $(cat "$(slot_file "$1")" 2>/dev/null)"
        (( waited >= max )) && { echo "gave up waiting for $1 after ${max}s" >&2; log "$2" "timed out waiting for $1"; exit 3; }
        sleep 15; waited=$((waited + 15))
    done
}

release() {  # slot agent
    local f; f=$(slot_file "$1")
    if [[ -f $f && $(cut -d' ' -f1 "$f") == "$2" ]]; then
        rm -f "$f"; log "$2" "released $1"
    fi
}

load_ok() {
    local load mem_kb
    load=$(cut -d' ' -f1 /proc/loadavg)
    mem_kb=$(awk '/MemAvailable/ {print $2}' /proc/meminfo)
    if awk -v l="$load" -v m="$LOAD_MAX" 'BEGIN {exit !(l < m)}' && (( mem_kb > MEM_MIN_GB * 1024 * 1024 )); then
        return 0
    fi
    echo "busy: load $load (max $LOAD_MAX), $((mem_kb / 1024 / 1024)) GB free (min $MEM_MIN_GB)"
    return 1
}

status() {
    echo "== slots"
    for s in heavy gpu download; do
        printf '  %-9s %s\n' "$s" "$(cat "$SLOTS/$s" 2>/dev/null || echo free)"
    done
    echo "== machine"
    echo "  load $(cut -d' ' -f1-3 /proc/loadavg) / 16 threads, $(awk '/MemAvailable/ {printf "%.1f", $2/1048576}' /proc/meminfo) GB RAM free"
    command -v nvidia-smi >/dev/null && \
        echo "  VRAM free: $(nvidia-smi --query-gpu=memory.free --format=csv,noheader)"
    echo "== recent"
    tail -n "${1:-15}" "$LOG" 2>/dev/null | sed 's/^/  /' || true
}

cmd="${1:-status}"; shift || true
case "$cmd" in
    status)  status "$@" ;;
    claim)   claim "$@" ;;
    release) release "$@" ;;
    note)    log "$1" "${*:2}" ;;
    load-ok) load_ok ;;
    wait-load)
        until load_ok >/dev/null; do sleep 20; done ;;
    run)  # slot agent note -- command...
        slot=$1 agent=$2 note=$3; shift 3; [[ ${1:-} == -- ]] && shift
        claim "$slot" "$agent" "$note"
        trap 'release "$slot" "$agent"' INT TERM EXIT
        "$@" ;;
    *) echo "usage: agent-board.sh status|claim SLOT AGENT NOTE [MAX_WAIT]|release SLOT AGENT|run SLOT AGENT NOTE -- CMD...|note AGENT MSG|load-ok|wait-load" >&2
       exit 2 ;;
esac
