# DeepSeek-V4.1-Flash-1017 on 8× B300 (single-node TEP)

Runbook for serving `deepseek-ai/DeepSeek-V4.1-Flash-1017` with vLLM on one
HGX B300 node (8× B300 SXM6, 268 GB HBM each), following the
[vLLM recipe](https://recipes.vllm.ai/deepseek-ai/DeepSeek-V4.1-Flash?hardware=b300&strategy=single_node_tep).

Everything that writes — weights, virtualenv, HF cache, JIT/autotune caches —
lives on a dedicated NVMe partition mounted at `/models`. The host in question
boots a live image whose root is a RAM overlay, so nothing outside `/models`
survives a power cycle.

## 1. Storage layout

The node has six 7 TB NVMe drives. Only the first is used; the rest stay
unformatted for later.

```bash
parted -s /dev/nvme0n1 mklabel gpt
parted -s /dev/nvme0n1 mkpart primary ext4 0% 100%
mkfs.ext4 -q -L models /dev/nvme0n1p1
mkdir -p /models && mount /dev/nvme0n1p1 /models
```

`/etc/fstab` is pointless on a live image — remount by hand after any restart.

| Path | Contents | Size |
| --- | --- | --- |
| `/models/DeepSeek-V4.1-Flash-1017` | weights, 57 files | 476 GiB |
| `/models/venv` | vLLM 0.34.0 + CUDA 13 wheels | 9.0 GB |
| `/models/hfvenv` | `huggingface_hub[hf_transfer]` for downloads | 120 MB |
| `/models/.hf` | HF hub cache | — |
| `/models/.vllm` | vLLM compile cache (`VLLM_CACHE_ROOT`) | 4.5 MB |
| `/models/.cache/flashinfer` | FlashInfer cubins + JIT (`FLASHINFER_WORKSPACE_BASE`) | 191 MB |
| `/models/tmp` | `TMPDIR` | — |

## 2. Prerequisites the recipe does not mention

The recipe assumes a normal install. Two things are missing on a bare driver
image and both fail *late* — after the weights are already on the GPUs.

**CUDA toolkit.** DeepGEMM and FlashInfer JIT-compile kernels at load time and
need `nvcc`:

```bash
apt-get install -y cuda-toolkit-13-0
ln -sfn /usr/local/cuda-13.0 /usr/local/cuda
```

Without it the first failure is
`deep_jit/backend/cuda/backend.hpp:200: not home_path.empty() and
std::filesystem::exists(home_path)`.

**`ninja` on `PATH`.** vLLM spawns one worker per GPU; those subprocesses
inherit `PATH` from the launcher, not from the venv. Calling
`/models/venv/bin/vllm` directly leaves `ninja` off `PATH` and every worker
dies with `FileNotFoundError: [Errno 2] No such file or directory: 'ninja'`.
Install it into the venv *and* export the venv's `bin`:

```bash
uv pip install --python /models/venv/bin/python ninja
export PATH=/models/venv/bin:$PATH
```

## 3. Download the weights

`huggingface_hub` now transfers over Xet; the old `HF_HUB_ENABLE_HF_TRANSFER`
switch is deprecated and ignored. Use the Xet knob instead. Sustained
throughput on this link was ~106 MB/s, ~80 min for 476 GiB.

```bash
uv venv --python 3.12 /models/hfvenv
uv pip install --python /models/hfvenv/bin/python huggingface_hub

export HF_HOME=/models/.hf HF_XET_HIGH_PERFORMANCE=1
/models/hfvenv/bin/hf download deepseek-ai/DeepSeek-V4.1-Flash-1017 \
  --local-dir /models/DeepSeek-V4.1-Flash-1017
```

The two 101.5 GB expert shards dominate the transfer; the progress bar counts
files, so it sits at 82/84 for the last half hour.

## 4. Install vLLM

```bash
uv venv --python 3.12 /models/venv
UV_CACHE_DIR=/models/.uvcache uv pip install --python /models/venv/bin/python vllm==0.34.0
```

0.34.0 carries native `deepseek_v41` support — tokenizer mode, config,
quantization and the DSpark speculator are all in-tree:

```bash
grep -rl deepseek_v41 /models/venv/lib/python3.12/site-packages/vllm/ | head
grep -rl dspark       /models/venv/lib/python3.12/site-packages/vllm/ | head
```

## 5. Serve

`/models/serve-dsv41.sh`:

```bash
#!/usr/bin/env bash
set -euo pipefail

export PATH=/models/venv/bin:$PATH
export CUDA_HOME=/usr/local/cuda
export VLLM_ENGINE_READY_TIMEOUT_S=3600
export VLLM_USE_RUST_FRONTEND=1
export HF_HOME=/models/.hf
export VLLM_CACHE_ROOT=/models/.vllm
export FLASHINFER_WORKSPACE_BASE=/models
export TMPDIR=/models/tmp
export HF_HUB_OFFLINE=1

exec /models/venv/bin/vllm serve /models/DeepSeek-V4.1-Flash-1017 \
  --served-model-name deepseek-v4.1-flash \
  --tokenizer-mode deepseek_v41 \
  --enable-expert-parallel \
  --tensor-parallel-size 8 \
  --tool-call-parser deepseek_v41 \
  --enable-auto-tool-choice \
  --reasoning-parser deepseek_v41 \
  --mm-encoder-tp-mode data \
  --enable-ep-weight-filter \
  --max-num-batched-tokens 8192 \
  --max-num-seqs 256 \
  --gpu-memory-utilization 0.95 \
  --speculative-config '{"method":"dspark","num_speculative_tokens":5,"draft_sample_method":"probabilistic","rejection_sample_method":"block","enable_adaptive_verification":true}' \
  "$@"
```

Launch detached so it outlives the SSH session:

```bash
cd /models && setsid nohup ./serve-dsv41.sh > /models/serve.log 2>&1 < /dev/null &
```

`VLLM_ENGINE_READY_TIMEOUT_S=3600` is not optional: the first start spends
~33 min in JIT compilation and kernel autotune before the engine reports ready.

## 6. What the first start does

Observed on a cold cache, process start to `/health` returning 200: **~40 min**.

| Marker in the log | Wall clock |
| --- | --- |
| process start | 14:52 |
| `Loading model from scratch...` | 14:57 |
| JIT `fp4_quantization_103`, `trtllm_mnnvl_comm`, `fused_moe_trtllm_sm100`, `sampling` | 14:57 – 15:09 |
| `Available KV cache memory: 206.67 GiB` | 15:09 |
| JIT `trtllm_utils`, then autotune `mxfp8_gemm` (21 buckets) | 15:12 – 15:17 |
| autotune `trtllm_fp4_block_scale_moe` (21 buckets, ~28 s each) | 15:17 – 15:28 |
| CUDA graph capture (54 piecewise graphs) + KV alloc | 15:28 – 15:32 |
| `init engine ... took 1998.77 s`, Rust frontend listening | 15:32 |

Both caches are persistent, so a restart with the same `VLLM_CACHE_ROOT` and
`FLASHINFER_WORKSPACE_BASE` skips most of this.

Harmless noise in the log:

- `[MNNVL] FABRIC handles failed on this host ... CUDA_ERROR_NOT_PERMITTED` —
  FlashInfer probes multi-node NVLink fabric handles, finds no IMEX channel,
  and falls back to POSIX file descriptors. Expected on a single node.
- `Ignoring incompatible flashinfer-cubin package` — the PyPI cubin wheel lags
  `flashinfer-python`; cubins are fetched on demand into the workspace instead.
- `No available shared memory broadcast block found in 60 seconds` — the engine
  core waiting on workers that are compiling. Normal during first start.

## 7. Measured result

```
GPU KV cache size: 127,409,364 tokens
Maximum concurrency for 1,048,576 tokens per request: 121.51x
Available KV cache memory: 206.67 GiB
```

Per GPU: 260 GB of 275 GB resident, 33–36 °C idle, ~240 W idle.

| Test | Result |
| --- | --- |
| single stream, 512 out | **242 tok/s** |
| 32 concurrent, 256 out each | **2311 tok/s** aggregate |
| `17*19` | `323` in 148 ms |
| thinking, effort 25 | 41 reasoning tokens, correct answer |
| tool call | `get_weather({"city":"Berlin"})`, `finish_reason=tool_calls` |
| vision | transcribed `HGX B300 - 8729 W` from a PNG, 194 prompt tokens |

## 8. Client notes

Thinking is **on at effort 50** when neither key is set, so a small
`max_tokens` returns an empty `content` with `finish_reason=length`. Pin it:

```json
{"chat_template_kwargs": {"thinking": false}}
```

or give the budget room:

```json
{"chat_template_kwargs": {"thinking": true, "reasoning_effort": 25}}
```

`reasoning_effort` accepts `low`/`high`/`xhigh`/`max` (25/50/75/100) or an
integer 1–100. `minimal` and `medium` are rejected.

## 9. Operating notes

- **No reboot.** The root filesystem is a RAM overlay; `/models` is the only
  durable location. A restart of the host loses the CUDA toolkit, the venv's
  `PATH` wiring and everything else outside `/models` — re-run §1–§4.
- **Stop the server** with `pkill -f serve-dsv41.sh`, not `pkill -f "vllm serve"`:
  the latter also matches the SSH command line that issued it.
- **Memory headroom.** `gpu_memory_utilization=0.95` caps the instance at
  254.3 GiB and leaves 7.86 GiB free per device. The extensible KV cache is
  sized after warmup, so 1.0 would also work on a dedicated node.
- **`--enable-ep-weight-filter`** skips experts that do not belong to the local
  EP rank. With 48 of 384 experts per rank it cuts load time substantially.
- **Engram tables** (196.6 B parameters, ~183 GiB) are offloaded to pinned host
  memory, 11.8 GiB per rank. Budget ~95 GiB of host RAM for them.
