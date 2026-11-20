#!/usr/bin/env bash
# Wait until a served model answers, then report how long it took.
#
#   export VLLM_HOST=user@host
#   ./wait_ready.sh              # blocks until /health returns 200
#   VLLM_TIMEOUT=7200 ./wait_ready.sh
set -euo pipefail

: "${VLLM_HOST:?set VLLM_HOST=user@host (never commit it)}"
PORT="${VLLM_PORT:-8000}"
TIMEOUT="${VLLM_TIMEOUT:-10800}"
INTERVAL="${VLLM_POLL_INTERVAL:-30}"

start=$(date +%s)
while :; do
    if ssh -o ConnectTimeout=10 "${VLLM_HOST}" \
        "curl -sf -m 10 http://localhost:${PORT}/health" >/dev/null 2>&1; then
        echo ">> ${VLLM_HOST}: ready after $(( $(date +%s) - start ))s"
        exit 0
    fi
    if [ $(( $(date +%s) - start )) -ge "${TIMEOUT}" ]; then
        echo ">> ${VLLM_HOST}: not ready after ${TIMEOUT}s" >&2
        exit 1
    fi
    sleep "${INTERVAL}"
done
