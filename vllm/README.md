# Serving DeepSeek-V4.1-Flash on Grace-Blackwell stations

`serve.sh` starts the vLLM OpenAI server for `deepseek-ai/DeepSeek-V4.1-Flash-1017`
on a DGX Station GB300. The host is never stored in this directory — it comes
from `VLLM_HOST`.

```bash
export VLLM_HOST=user@host          # never committed
./serve.sh --gpu 1                  # start detached, pinned to nvidia-smi index 1
./wait_ready.sh                     # block until /health answers
./smoke.sh                          # one real chat completion
./bench_serving.sh                  # TTFT + decode rate at concurrency 1/4/8
```

## Why the GPU is pinned

`--gpus all` hands the container every card in the chassis. On a station that
mixes a GB300 with an RTX PRO 6000 that means device 0 is the small card — and
the two cannot even share a CUDA context (HMM vs ATS addressing), so vLLM
fails outright. `serve.sh` therefore always passes `--gpus device=<index>`.
Find the right index with `nvidia-smi -L`.

## What the flags do

| Flag | Effect |
|---|---|
| `--offload-backend uva` | Experts live in host memory and are read over NVLink-C2C (coherent, ~450 GB/s each way) instead of being copied |
| `--cpu-offload-gb 60` | 60 GiB of weights stay on the host |
| `--cpu-offload-params routed_experts.*` | Only the routed experts are offloaded — the dense part stays on the GPU |
| `--engram-config {"cpu_offload":true}` | Engram tables (2 × 94 GiB) pinned in host memory |
| `--kv-cache-dtype fp8_ds_mla` | DeepSeek's FP8 MLA KV format |
| `--attention-config FLASHMLA_MEGA_ATTN_DSV41` | Sparse MLA backend for this architecture |
| `--speculative-config dspark` | 5 speculative tokens, fewer at larger batch sizes |
| `--max-model-len 1048576` | 1M context |

## Sizing

The checkpoint is ~475 GiB across 48 shards, so the first start downloads for
a while (the HF cache is bind-mounted from `~/.cache/huggingface`). Two engram
tables of 94 GiB each plus 61 GiB of offloaded parameters live in host RAM —
budget ~400 GiB RSS on a 744 GiB machine.

Unauthenticated Hub requests are rate-limited. For a fleet, download once and
copy the cache to the other machines instead of pulling 475 GiB per host:

```bash
rsync -a --info=progress2 ~/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-V4.1-Flash-1017 \
      other-host:~/.cache/huggingface/hub/
```

## Measuring it

`bench_serving.py` streams real requests and reports time-to-first-token,
per-stream decode rate and aggregate output tokens/s. Stdlib only, so it runs
on the GPU host without installing anything:

```bash
VLLM_CONCURRENCY="1 8 24" ./bench_serving.sh
```

## Checking on it

```bash
ssh "$VLLM_HOST" 'docker logs --tail 50 deepseek-v41-flash'
ssh "$VLLM_HOST" 'docker stats --no-stream deepseek-v41-flash'
ssh "$VLLM_HOST" 'curl -s localhost:8000/v1/models'
```
