# System Stress tests


## Getting started

`git clone https://github.com/flozi00/sys-stress.git`

`cd sys-stress`

### Nvidia

1. `cd nvidia`
2. `sh install.sh`
3. `nvidia-smi`
4. `docker run --rm --gpus all gpu-stress --duration 2h`

### GPU stress test (NVIDIA + AMD)

A single Triton based stress test that hammers both memory bandwidth and the
computation cores of every GPU, works on NVIDIA (CUDA) and AMD (ROCm), and is
designed for multi-hour / multi-day cooling and stability soak tests with a
configurable runtime and a nice system-stress log.

See [`gpu-stress/README.md`](gpu-stress/README.md) for details.
