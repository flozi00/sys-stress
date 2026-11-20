# System Stress tests


## Getting started

`git clone https://github.com/flozi00/sys-stress.git`

`cd sys-stress`

### Nvidia

1. `cd nvidia`
2. `sh install.sh`
3. `nvidia-smi`
4. `docker run --rm --gpus all gpu-stress --duration 2h`

### AMD

1. `cd amd`
2. `sudo bash install.sh`
3. Reboot (required after the first install so the `amdgpu.dc=0` modprobe
   option takes effect and the DKMS driver loads cleanly).
4. `rocm-smi`
5. `docker run --rm --device=/dev/kfd --device=/dev/dri --security-opt seccomp=unconfined --group-add video gpu-stress-rocm --duration 2h`

### GPU stress test (NVIDIA + AMD)

A single Triton based stress test that hammers both memory bandwidth and the
computation cores of every GPU, works on NVIDIA (CUDA) and AMD (ROCm), and is
designed for multi-hour / multi-day cooling and stability soak tests with a
configurable runtime and a nice system-stress log.

Each iteration measures two metrics per GPU:

| Metric | Unit | What it measures |
|--------|------|------------------|
| **compute** | TFLOP/s | FMA-loop throughput (FP32). 4 FLOPs per element per inner iteration. |
| **mem** | GB/s | Device memory bandwidth (read + write) from the streaming copy kernel. |

The compute phase is self checking — any mismatch under load (overheating,
unstable overclock, faulty hardware) is counted and logged as an error.

See [`gpu-stress/README.md`](gpu-stress/README.md) for full options, the
duration / time format (`30m`, `2h`, `3d`, …), and bare-metal instructions.

### Hardware benchmarks (FLOPS, bandwidth, interconnect)

`gpu-stress` answers "does it stay stable for days?".  [`bench/`](bench/README.md)
answers "how fast is it, really?" — HBM/GDDR bandwidth (read, write, copy,
triad), GEMM peak per precision (fp32, tf32, bf16, fp16, fp8, fp4), host↔device
bandwidth (NVLink-C2C or PCIe), P2P copies, CPU STREAM, and the power/clock
point each number was measured at.

```bash
export SYS_STRESS_HOST=user@host   # never committed
cd bench
./bootstrap_remote.sh              # one-time: uv + venv + torch
./run_remote.sh all                # full report into ./results/
./run_remote.sh --pdf all          # same, plus a PDF report
```

The target host is read from `SYS_STRESS_HOST` only; no hostnames, IPs or
credentials live in this repository.