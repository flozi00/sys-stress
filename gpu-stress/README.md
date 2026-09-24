# GPU stress test

A single, vendor neutral GPU burn / soak test that stresses **both** the
memory subsystem (memory bandwidth) and the **computation cores** (tensor +
FP32 ALUs) of every visible GPU.

The default `torch` backend does the heavy work with pure PyTorch ops
(BF16 GEMMs on the tensor/matrix cores fused with FP32 ALU work and device
memory copies), replayed from CUDA graphs on several streams per GPU — no
compiler toolchain needed, so it runs anywhere a GPU PyTorch build runs and
reaches the card power limit (proven **~1290/1300 W on NVIDIA GB300**).
Optional [Triton](https://github.com/triton-lang/triton) and
[Helion](https://github.com/pytorch/helion) backends keep the exact same
test portable across NVIDIA (CUDA) and AMD (ROCm) GPUs where a working
Triton/gcc toolchain exists. This replaces the old CUDA-only `gpu_burn`
implementation.

## What it does

Each iteration replays CUDA graphs of fused BF16 matrix multiplies, FP32
elementwise work and device-to-device copies on every selected GPU —
driving tensor cores, ALUs and HBM together. Reports estimated **TFLOP/s**
and **GB/s**. (The legacy `triton` backend instead runs a register bound
FMA loop + streaming copy in two phases; `helion` runs an autotuned
matmul instead of the FMA loop.)

The compute phase is **self checking**: the op chain is a pure function of
its inputs, so all workers are seeded with identical inputs and every later
iteration is compared against a reference checksum. Any mismatch (caused by
overheating, an unstable overclock or faulty hardware) is counted and logged
as an error, just like the classic `gpu_burn`.

## Metrics explained

| Metric | Unit | What it measures |
|--------|------|------------------|
| **compute** | TFLOP/s | Tera floating-point operations per second (GEMM + ALU FLOPs per graph replay; triton backend: 4 FLOPs per element per FMA iteration). |
| **mem** | GB/s | Achieved device memory bandwidth (read + write) from the in-graph device copies (triton backend: streaming copy kernel). |
| **errors** | count | Number of checksum mismatches detected by the self-check. 0 = clean run. |
| **iterations** | count | How many complete graph-replay cycles completed during the run. |

Progress is logged to stdout at `--log-interval` seconds (default 10) and to a
full-system SMI snapshot (temperature, power, utilisation) at
`--monitor-interval` seconds (default 30).

## Maximum power & maximum throughput

The default backend is the **FP8 burn** (`auto` → `fp8` on Hopper/Blackwell,
else `torch`): CUDA-graph replay of `_scaled_mm` FP8 GEMMs mixed 5:1 with BF16
GEMMs plus FP32→BF16 conversions and an HBM copy phase, on several streams
per GPU. At startup it **autotunes**:

- the GEMM matrix dimension — it sweeps every candidate that fits the
  free-VRAM budget (16384…2048), captures a short CUDA graph per candidate
  and keeps the empirically fastest one (the same idea as Helion's matmul
  autotuning, at the shape level), and
- the memory-phase buffer size, scaled to the VRAM left over.

Measured on NVIDIA GB300 (DGX Station, 1300 W): autotune selects 8192²
(2791 TFLOP/s in-tune), sustains **~2600 TFLOP/s at 1294 W ≈ 99.6 % of the
power limit** with 0 errors; a pure `_scaled_mm` probe on the same host
reaches 3588 TFLOP/s (76 % of the 5 PF FP8 dense datasheet figure at the
1300 W/77 °C operating point).

Tune it with `--burn-streams`, `--burn-replays`, `--mem-fraction`.
Buffers auto-shrink to fit free VRAM (it also runs alongside a loaded
inference server — verified at ~1000 TFLOP/s in 5 GB of free VRAM).

```bash
python3 gpu_stress.py --duration 2h            # autotuned fp8 burn
python3 gpu_stress.py --duration 2h --compute-backend torch   # bf16 burn
```

### Legacy backends: triton / helion (optional)

The backend is selected with `--compute-backend {auto,torch,triton,helion}`
(default `auto` = `torch`). `triton` runs the original FMA-loop + copy
kernels; `helion` autotunes a matmul compute phase instead of the FMA loop.
Both need a working Triton/gcc toolchain (and `pip install helion` for the
latter; autotuning may take several minutes on the first run). Helion works
on NVIDIA (CUDA) and AMD (ROCm) alike.

```bash
pip install helion
python3 gpu_stress.py --duration 2h --compute-backend helion
```

## Quick start (Docker)

### NVIDIA

```bash
cd gpu-stress
docker build -f Dockerfile.nvidia -t gpu-stress .

# stress all GPUs for 2 hours, writing logs to ./stress-logs
docker run --rm --gpus all \
    -v "$PWD/stress-logs:/workspace/gpu-stress/stress-logs" \
    gpu-stress --duration 2h
```

### AMD

```bash
cd gpu-stress
docker build -f Dockerfile.rocm -t gpu-stress-rocm .

docker run --rm \
    --device=/dev/kfd --device=/dev/dri \
    --security-opt seccomp=unconfined --group-add video \
    -v "$PWD/stress-logs:/workspace/gpu-stress/stress-logs" \
    gpu-stress-rocm --duration 2h
```

> **AMD driver note:** The host needs the `amdgpu` kernel driver loaded and
> `/dev/kfd` + `/dev/dri` present. Run `amd/install.sh` to install the
> AMDGPU DKMS driver + ROCm and build the image. The `amdgpu.dc=0` modprobe
> option is set by the install script to avoid a display-core divide-by-zero
> on headless cards — a reboot is required after the first install.

## Quick start (bare metal)

Requires a GPU enabled PyTorch install (see `requirements.txt`). Triton is
only needed for the legacy `triton`/`helion` backends.

```bash
python3 gpu_stress.py --duration 30m
```

## Duration / time configuration

The `--duration` / `-t` flag controls how long the stress test runs.  It accepts
a bare number (seconds) or a value with unit suffixes:

| Example | Meaning |
|--------|---------|
| `--duration 120` | 120 seconds |
| `--duration 30m` | 30 minutes |
| `--duration 2h` | 2 hours |
| `--duration 3d` | 3 days |
| `--duration 1h30m` | 1 hour 30 minutes (compound) |

The test runs until the duration elapses, then reports a summary and exits with
status `0` (all GPUs passed) or non-zero (mismatches or worker failures
detected).

## Options

```
-t, --duration          How long to run. Seconds or units: 90, 30m, 2h, 3d, 1h30m (default: 120)
-m, --mem-fraction      Fraction of free GPU memory to allocate (default: 0.8)
    --compute-backend   Compute kernel backend: auto, torch, fp8, triton, helion (default: auto = fp8 when _scaled_mm works, else torch)
    --burn-streams      Parallel CUDA streams per GPU for the torch/fp8 backends (default: 4)
    --burn-dim          Matrix dim for the torch backend, auto-shrunk to fit VRAM (default: 8192). fp8 backend autotunes the dim
    --burn-replays      GEMM+ALU+copy replays per CUDA graph (default: 20)
    --compute-iters     Inner FMA-loop iterations per launch, higher = more compute bound. Triton backend only (default: 2048)
    --block-size        Triton block size in elements. Triton backend only (default: 1024)
    --devices           Comma separated GPU indices, or 'all' (default: all)
    --log-interval      Seconds between per-GPU progress lines (default: 10)
    --monitor-interval  Seconds between full-system nvidia-smi/rocm-smi snapshots (default: 30)
    --log-dir           Directory for the run log file (default: stress-logs)
    --no-error-check    Disable the compute self-check
```

## Examples

```bash
# 60-second smoke test
python3 gpu_stress.py --duration 60

# Overnight soak test on all GPUs
python3 gpu_stress.py --duration 12h

# Multi-day cooling validation on GPUs 0 and 1
python3 gpu_stress.py --duration 3d --devices 0,1

# Maximum compute pressure (torch backend, alongside a loaded server)
python3 gpu_stress.py --duration 1h --mem-fraction 0.9 --burn-streams 6

# Legacy triton backend with maximum compute pressure
python3 gpu_stress.py --duration 1h --compute-backend triton --compute-iters 4096 --mem-fraction 0.9

# Pure compute, skip the self-check (slightly faster)
python3 gpu_stress.py --duration 1h --no-error-check
```

The process exits with status `0` when every GPU passes, and non-zero if any
compute mismatch or worker failure was detected.