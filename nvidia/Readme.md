```bash

docker run --runtime nvidia --gpus all \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    -p 8000:8000 \
    --ipc=host \
    lmsysorg/sglang:latest python3 -m sglang.launch_server --model neuralmagic/Meta-Llama-3.1-405B-Instruct-FP8 --tp 8 --enable-mixed-chunk --context-length 10000 --mem-fraction-static 0.86 --stream-interval 6 --kv-cache-dtype fp8_e5m2 --disable-cuda-graph --host 0.0.0.0 --port 8000

```

## vLLM single-GPU benchmark

The named Docker
volume keeps downloaded model weights across runs.
This uses a pre-quantized 4-bit 70B checkpoint to stress one large data-center
GPU. The default workload processes 10,000 requests with 512 generated tokens
each, which is intended to run for roughly one hour on one large data-center GPU.
Exact duration depends on the GPU; adjust `--num-prompts` after the first run.

The benchmark writes these files to `vllm-results`:

- `vllm-throughput-<timestamp>.log`: full terminal output
- `vllm-throughput-<timestamp>.json`: vLLM throughput statistics
- `vllm-gpu-<timestamp>.csv`: sampled GPU utilization, memory, power, and temperature

```bash
mkdir -p vllm-results

docker run --rm --runtime nvidia --gpus '"device=0"' \
    --ipc=host \
    -v vllm-hf-cache:/root/.cache/huggingface \
    -v "$PWD/vllm-results:/results" \
    --entrypoint bash \
    vllm/vllm-openai:latest \
    -lc '
set -euo pipefail

ts=$(date -u +%Y%m%dT%H%M%SZ)
gpu_stats="/results/vllm-gpu-${ts}.csv"
run_log="/results/vllm-throughput-${ts}.log"
run_json="/results/vllm-throughput-${ts}.json"

gpu_logger=
if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi \
        --query-gpu=timestamp,index,name,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw,temperature.gpu \
        --format=csv \
        -l 5 > "${gpu_stats}" &
    gpu_logger=$!
fi

cleanup() {
    if [ -n "${gpu_logger}" ]; then
        kill "${gpu_logger}" 2>/dev/null || true
    fi
}
trap cleanup EXIT

vllm bench throughput \
    --model unsloth/Llama-3.3-70B-Instruct-bnb-4bit \
    --backend vllm \
    --dataset-name random \
    --random-input-len 1024 \
    --random-output-len 512 \
    --num-prompts 10000 \
    --num-warmups 32 \
    --max-model-len 2048 \
    --dtype bfloat16 \
    --output-json "${run_json}" 2>&1 | tee "${run_log}"
'
```

For a closer one-hour target after a first run, set `--num-prompts` to about
`3600 * requests_per_second` from the JSON result.

Due to inkompitability issues in some cases the grub params needs to be edited.

```bash
sudo nano /etc/default/grub
```

then put to the default linux params:

```bash
GRUB_CMDLINE_LINUX_DEFAULT="quiet splash nokaslr"
```

followed by:
```bash
sudo update-grub
sudo reboot now
```
