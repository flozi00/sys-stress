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
  the FP32 ALUs and reports estimated **TFLOP/s**.
- **memory** — a streaming copy over large buffers that saturates the device
  memory bus and reports the achieved **GB/s**.

The compute phase is **self checking**: the first iteration's result is kept as a
reference and every later iteration is compared against it. Any mismatch (caused
by overheating, an unstable overclock or faulty hardware) is counted and logged
as an error, just like the classic `gpu_burn`.

The test is designed to keep the GPUs at ~100% load for a **configurable**
amount of time — from minutes to several **days** — to validate cooling and long
term stability. Progress is streamed to stdout **and** to a timestamped log file
so a long soak test leaves behind a nice, self contained record.

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

## Quick start (bare metal)

Requires a GPU enabled PyTorch + Triton install (see `requirements.txt`).

```bash
python3 gpu_stress.py --duration 30m
```

## Options

```
-t, --duration          How long to run. Seconds or units: 90, 30m, 2h, 3d, 1h30m (default: 120)
-m, --mem-fraction      Fraction of free GPU memory to allocate (default: 0.8)
    --compute-iters     Inner FMA-loop iterations per launch, higher = more compute bound (default: 2048)
    --block-size        Triton block size in elements (default: 1024)
    --devices           Comma separated GPU indices, or 'all' (default: all)
    --log-interval      Seconds between per-GPU progress lines (default: 10)
    --monitor-interval  Seconds between full-system nvidia-smi/rocm-smi snapshots (default: 30)
    --log-dir           Directory for the run log file (default: stress-logs)
    --no-error-check    Disable the compute self-check
```

## Examples

```bash
# Overnight soak test on all GPUs
python3 gpu_stress.py --duration 12h

# Multi-day cooling validation on GPUs 0 and 1
python3 gpu_stress.py --duration 3d --devices 0,1

# Maximum compute pressure
python3 gpu_stress.py --duration 1h --compute-iters 4096 --mem-fraction 0.9
```

The process exits with status `0` when every GPU passes, and non-zero if any
compute mismatch or worker failure was detected.
