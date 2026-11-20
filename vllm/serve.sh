#!/usr/bin/env bash
# Serve DeepSeek-V4.1-Flash with vLLM on a Grace-Blackwell station.
#
# The host never appears in this file: pass it in the environment.
#
#   export VLLM_HOST=user@host
#   ./serve.sh                 # start detached on GPU 0
#   ./serve.sh --gpu 1         # pin to nvidia-smi index 1 (multi-GPU chassis)
#   ./serve.sh --foreground    # run attached, logs on the terminal
#
# Overridable: VLLM_MODEL, VLLM_PORT, VLLM_IMAGE, VLLM_NAME, VLLM_EXTRA_ARGS.
set -euo pipefail

: "${VLLM_HOST:?set VLLM_HOST=user@host (never commit it)}"

MODEL="${VLLM_MODEL:-deepseek-ai/DeepSeek-V4.1-Flash-1017}"
PORT="${VLLM_PORT:-8000}"
IMAGE="${VLLM_IMAGE:-vllm/vllm-openai:latest}"
NAME="${VLLM_NAME:-deepseek-v41-flash}"
GPU="${VLLM_GPU:-0}"
# The first start downloads ~475 GiB.  The stock 3600 s is measured from
# process start, so a cold cache kills the frontend mid-download.
READY_TIMEOUT="${VLLM_ENGINE_READY_TIMEOUT_S:-14400}"
# moe_backend='auto' benchmarks the candidates at startup and picks the
# fastest.  The choice is not stable across machines: on one station the
# autotuner picked deep_gemm and the first forward pass died in
# fp8_einsum (layout.hpp:51, t.dim() == N), while flashinfer_trtllm ran
# fine.  Pin it so every station serves the same kernels.
MOE_BACKEND="${VLLM_MOE_BACKEND:-flashinfer_trtllm}"
DETACH=1

while [ $# -gt 0 ]; do
    case "$1" in
        --gpu) GPU="$2"; shift 2 ;;
        --foreground) DETACH=0; shift ;;
        --name) NAME="$2"; shift 2 ;;
        *) echo "unknown argument: $1" >&2; exit 2 ;;
    esac
done

if [ "${DETACH}" = "1" ]; then
    MODE="--detach --name ${NAME}"
else
    MODE="--rm --name ${NAME}"
fi

echo ">> ${VLLM_HOST}: ${IMAGE} ${MODEL} on GPU ${GPU}, port ${PORT}"

# The command is piped in as a script: the JSON arguments contain braces and
# quotes that a nested ssh command line would mangle.
ssh "${VLLM_HOST}" bash -s <<REMOTE
set -euo pipefail
docker rm -f ${NAME} >/dev/null 2>&1 || true
docker run ${MODE} \\
  --gpus "device=${GPU}" \\
  --privileged --ipc=host -p ${PORT}:8000 \\
  -v "\${HOME}/.cache/huggingface:/root/.cache/huggingface" \\
  -e VLLM_ENGINE_READY_TIMEOUT_S=${READY_TIMEOUT} \\
  -e VLLM_USE_RUST_FRONTEND=1 \\
  ${IMAGE} ${MODEL} \\
  --tokenizer-mode deepseek_v41 \\
  --offload-backend uva \\
  --cpu-offload-gb 60 \\
  --cpu-offload-params routed_experts.w13_weight routed_experts.w2_weight \\
  --engram-config '{"cpu_offload":true}' \\
  --gpu-memory-utilization 0.97 \\
  --max-model-len 1048576 \\
  --max-num-batched-tokens 8192 \\
  --long-prefill-token-threshold 6144 \\
  --compilation-config '{"cudagraph_capture_sizes":[1,2,4,6,8,12,16,18,24,32,40,48,64,96,128]}' \\
  --tensor-parallel-size 1 \\
  --attention-config '{"backend":"FLASHMLA_MEGA_ATTN_DSV41"}' \\
  --kv-cache-dtype fp8_ds_mla \\
  --max-num-seqs 24 \\
  --tool-call-parser deepseek_v41 \\
  --enable-auto-tool-choice \\
  --reasoning-parser deepseek_v41 \\
  --mm-encoder-tp-mode data \\
  --moe-backend ${MOE_BACKEND} \\
  --speculative-config '{"method":"dspark","num_speculative_tokens":5,"num_speculative_tokens_per_batch_size":[[1,4,5],[5,24,1]]}' \\
  ${VLLM_EXTRA_ARGS:-}
REMOTE
