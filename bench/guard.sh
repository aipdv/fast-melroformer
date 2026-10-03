#!/usr/bin/env bash
# Run one heavy job with a RAM watchdog. The 8 GB Mac froze once (2026-10-03) when jobs overlapped.
# Usage: bench/guard.sh <cmd...>   Kills the job if system free memory < MIN_FREE % (default 8).
set -u
MIN_FREE=${MIN_FREE:-8}
if pgrep -f "fast-melroformer/.venv/bin/python" >/dev/null; then echo "guard: another job is running, refusing" >&2; exit 3; fi
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=${PYTORCH_MPS_HIGH_WATERMARK_RATIO:-0.7}  # hard cap on the MPS pool, as a share of the GPU working set
export PYTORCH_MPS_LOW_WATERMARK_RATIO=${PYTORCH_MPS_LOW_WATERMARK_RATIO:-0.5}
"$@" &
pid=$!
while kill -0 $pid 2>/dev/null; do
  free=$(memory_pressure | awk -F': ' '/free percentage/ {gsub("%","",$2); print $2}')
  if [ "${free:-100}" -lt "$MIN_FREE" ]; then
    echo "guard: free memory ${free}% < ${MIN_FREE}%, killing $pid" >&2
    pkill -TERM -P $pid; kill -TERM $pid; sleep 2; kill -9 $pid 2>/dev/null; pkill -9 -P $pid 2>/dev/null
    exit 4
  fi
  sleep 1
done
wait $pid
