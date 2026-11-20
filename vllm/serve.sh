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
# moe_backend stays 'auto': vLLM benchmarks the candidates at startup and
# keeps the fastest.  Set VLLM_MOE_BACKEND to force one (e.g. deep_gemm,
# flashinfer_trtllm) when comparing kernels rather than machines.
MOE_BACKEND="${VLLM_MOE_BACKEND:-}"
MOE_ARG=""
if [ -n "${MOE_BACKEND}" ]; then MOE_ARG="--moe-backend ${MOE_BACKEND}"; fi
# FlashInfer autotune profiles every GEMM at every token bucket during
# warmup: hundreds of shapes, ~1 min each, so a cold start spends hours
# there.  Off by default for bring-up; set VLLM_FLASHINFER_AUTOTUNE=1 for
# final performance numbers.
FI_AUTOTUNE="${VLLM_FLASHINFER_AUTOTUNE:-0}"
# The checkpoint leaves ~10 GiB of the 249 GiB card for everything else, so
# the upstream 1M context does not fit: one max-length request needs 1.89 GiB
# of KV cache and only 0.62 GiB is left after the model, activations and
# CUDA graphs.  256k needs 0.47 GiB and fits with room to spare (17.5 GiB of
# KV cache).  Raise VLLM_MAX_MODEL_LEN once the autotune is on: it leaves
# ~24 GiB of KV cache and the full 1M works.
MAX_BATCHED="${VLLM_MAX_BATCHED_TOKENS:-4096}"
MEM_UTIL="${VLLM_GPU_MEM_UTIL:-0.99}"
MAX_LEN="${VLLM_MAX_MODEL_LEN:-262144}"
FI_ARG=""
if [ "${FI_AUTOTUNE}" = "0" ]; then FI_ARG="--no-enable-flashinfer-autotune"; fi
# No --privileged: it mounts every /dev/nvidia* node and voids the
# --gpus device= restriction.  On a chassis with a second card the
# container then sees two devices, vLLM detects the wrong compute
# capability (12.0 instead of 10.3) and rejects the sm_100 kernels.
# --ipc=host is all the server needs.
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
  --ipc=host -p ${PORT}:8000 \\
  -v "\${HOME}/.cache/huggingface:/root/.cache/huggingface" \\
  -v "\${HOME}/.cache/flashinfer:/root/.cache/flashinfer" \\
  -e VLLM_ENGINE_READY_TIMEOUT_S=${READY_TIMEOUT} \\
  -e VLLM_USE_RUST_FRONTEND=1 \\
  ${IMAGE} ${MODEL} \\
  --tokenizer-mode deepseek_v41 \\
  --offload-backend uva \\
  --cpu-offload-gb 60 \\
  --cpu-offload-params routed_experts.w13_weight routed_experts.w2_weight \\
  --engram-config '{"cpu_offload":true}' \\
  --gpu-memory-utilization ${MEM_UTIL} \\
  --max-model-len ${MAX_LEN} \\
  --max-num-batched-tokens ${MAX_BATCHED} \\
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
  ${MOE_ARG} ${FI_ARG} \\
  --speculative-config '{"method":"dspark","num_speculative_tokens":5,"num_speculative_tokens_per_batch_size":[[1,4,5],[5,24,1]]}' \\
  ${VLLM_EXTRA_ARGS:-}
REMOTE
