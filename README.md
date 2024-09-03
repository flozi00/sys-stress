# System Stress tests


## Getting started

`git clone https://github.com/flozi00/sys-stress.git`

`cd sys-stress`

### Nvidia

1. `cd nvidia`
2. `sh install.sh`
3. `nvidia-smi`
4. `docker run --rm --gpus all gpu_burn ./gpu_burn 10`
