#!/usr/bin/env bash
# Run the benchmark suite on a remote GPU host.
#
# The target is NEVER stored in this repository.  Export it for the session:
#
#   export SYS_STRESS_HOST=user@host        # ssh target
#   export SYS_STRESS_PYTHON=~/bench-venv/bin/python   # optional
#
# Then:
#
#   ./run_remote.sh all --json results.json
#   ./run_remote.sh compute --gpu 1 --dtype bf16 fp8
#
# Results are copied back into ./results/ (git-ignored).
set -euo pipefail

: "${SYS_STRESS_HOST:?Set SYS_STRESS_HOST to the ssh target (user@host). It is intentionally not stored in the repo.}"
REMOTE_DIR="${SYS_STRESS_REMOTE_DIR:-sys-stress-bench}"
REMOTE_PYTHON="${SYS_STRESS_PYTHON:-~/bench-venv/bin/python}"
LOCAL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
OUT_DIR="${LOCAL_DIR}/results"
mkdir -p "${OUT_DIR}"

echo ">> syncing ${LOCAL_DIR} -> ${SYS_STRESS_HOST}:${REMOTE_DIR}"
rsync -az --delete \
    --exclude results --exclude __pycache__ --exclude '*.pyc' \
    "${LOCAL_DIR}/" "${SYS_STRESS_HOST}:${REMOTE_DIR}/"

echo ">> running: bench.py $*"
ssh "${SYS_STRESS_HOST}" \
    "cd ${REMOTE_DIR} && ${REMOTE_PYTHON} bench.py $* --json results-${STAMP}.json" \
    | tee "${OUT_DIR}/bench-${STAMP}.log"

echo ">> fetching JSON report"
scp -q "${SYS_STRESS_HOST}:${REMOTE_DIR}/results-${STAMP}.json" \
    "${OUT_DIR}/bench-${STAMP}.json"
echo ">> done: ${OUT_DIR}/bench-${STAMP}.json"
