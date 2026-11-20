#!/usr/bin/env bash
# Run bench_serving.py against a remote endpoint (copies it over, stdlib only).
#
#   export VLLM_HOST=user@host
#   ./bench_serving.sh                       # concurrency 1 4 8
#   VLLM_CONCURRENCY="1 16 32" ./bench_serving.sh
set -euo pipefail

: "${VLLM_HOST:?set VLLM_HOST=user@host (never commit it)}"
PORT="${VLLM_PORT:-8000}"
MODEL="${VLLM_MODEL:-deepseek-ai/DeepSeek-V4.1-Flash-1017}"
CONC="${VLLM_CONCURRENCY:-1 4 8}"
MAX_TOKENS="${VLLM_MAX_TOKENS:-256}"

here="$(cd "$(dirname "$0")" && pwd)"
scp -q "${here}/bench_serving.py" "${VLLM_HOST}:/tmp/bench_serving.py"
ssh "${VLLM_HOST}" "python3 /tmp/bench_serving.py \
    --base-url http://localhost:${PORT} --model ${MODEL} \
    --concurrency ${CONC} --max-tokens ${MAX_TOKENS}"
