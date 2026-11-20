#!/usr/bin/env bash
# One-time (idempotent) setup of a PyTorch environment on a remote GPU host.
#
#   export SYS_STRESS_HOST=user@host
#   ./bootstrap_remote.sh
#
# Installs uv into ~/.local/bin (no root needed), creates ~/bench-venv and
# installs the CUDA 13 PyTorch build.  Safe to re-run.
set -euo pipefail

: "${SYS_STRESS_HOST:?Set SYS_STRESS_HOST to the ssh target (user@host). It is intentionally not stored in the repo.}"
TORCH_INDEX="${SYS_STRESS_TORCH_INDEX:-https://download.pytorch.org/whl/cu130}"

ssh "${SYS_STRESS_HOST}" bash -s <<EOF
set -euo pipefail
export PATH="\$HOME/.local/bin:\$PATH"
if ! command -v uv >/dev/null 2>&1; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="\$HOME/.local/bin:\$PATH"
fi
uv --version
if [ ! -x "\$HOME/bench-venv/bin/python" ]; then
    uv venv "\$HOME/bench-venv" --python 3.12
fi
uv pip install --python "\$HOME/bench-venv/bin/python" torch --index-url ${TORCH_INDEX}
# nvidia-ml-py lives on PyPI, not in the torch wheel index.
uv pip install --python "\$HOME/bench-venv/bin/python" nvidia-ml-py numpy
"\$HOME/bench-venv/bin/python" - <<'PY'
import torch
print("torch", torch.__version__, "cuda", torch.version.cuda,
      "devices", torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    print(" ", i, torch.cuda.get_device_name(i))
PY
EOF
