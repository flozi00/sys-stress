# Triton GPU stress test

A single, vendor neutral GPU burn / soak test that stresses **both** the
memory subsystem (memory bandwidth) and the **computation cores** (FP32 ALUs)
of every visible GPU.

The heavy work is done by [Triton](https://github.com/triton-lang/triton)
kernels executed through PyTorch, so the **exact same test runs on NVIDIA
(CUDA) and AMD (ROCm) GPUs** — no CUDA-only code required. This replaces the old
CUDA-only `gpu_burn` implementation.

## What it does

Each iteration runs two phases on every selected GPU:

- **compute** — a long, register bound fused-multiply-add loop that saturates
  the FP32 ALUs.  Values are clamped to `[−2, 2]` every iteration to prevent
  float32 overflow, keeping the self-check meaningful over long runs.  Reports
  estimated **TFLOP/s** (tera-operations per second).
- **memory** — a streaming copy over large buffers that saturates the device
  memory bus.  Reports the achieved **GB/s** (gigabytes per second of read +
  write bandwidth).

The compute phase is **self checking**: the first iteration's result is kept
as a reference and every later iteration is compared against it via a
checksum.  Any mismatch (caused by overheating, an unstable overclock or faulty
hardware) is counted and logged as an error, just like the classic `gpu_burn`.

## Metrics explained

| Metric | Unit | What it measures |
|--------|------|------------------|
| **compute** | TFLOP/s | Tera floating-point operations per second from the FMA loop. 4 FLOPs per element per inner iteration. |
| **mem** | GB/s | Achieved device memory bandwidth (read + write) from the streaming copy kernel. |
| **errors** | count | Number of differing elements detected by the self-check. 0 = clean run. |
| **iterations** | count | How many complete compute + memory cycles completed during the run. |

Progress is logged to stdout at `--log-interval` seconds (default 10) and to a
full-system SMI snapshot (temperature, power, utilisation) at
`--monitor-interval` seconds (default 30).

## Maximum power draw with Helion (optional)

For the highest possible load and power consumption, install
[Helion](https://github.com/pytorch/helion). When available, the compute phase
uses an **autotuned matrix multiply** instead of the FP32 FMA loop. A GEMM drives
the tensor/matrix cores together with the memory subsystem, which is the most
effective way to push a GPU to its power limit, and Helion autotunes the kernel
(this may take several minutes on the first run, depending on the GPU and matrix
size) so it reaches peak throughput on both NVIDIA and AMD.

```bash
pip install helion
python3 gpu_stress.py --duration 2h --compute-backend helion
```

The backend is selected with `--compute-backend {auto,triton,helion}` (default
`auto`, which uses Helion when available and otherwise falls back to the
Triton FMA kernel). Helion works on NVIDIA (CUDA) and AMD (ROCm) alike.

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

Requires a GPU enabled PyTorch + Triton install (see `requirements.txt`).

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
    --compute-backend   Compute kernel backend: auto, triton, helion (default: auto)
    --compute-iters     Inner FMA-loop iterations per launch, higher = more compute bound. Triton backend only (default: 2048)
    --block-size        Triton block size in elements (default: 1024)
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

# Maximum compute pressure
python3 gpu_stress.py --duration 1h --compute-iters 4096 --mem-fraction 0.9

# Pure compute, skip the self-check (slightly faster)
python3 gpu_stress.py --duration 1h --no-error-check
```

The process exits with status `0` when every GPU passes, and non-zero if any
compute mismatch or worker failure was detected.