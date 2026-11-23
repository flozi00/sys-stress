# Stressing all 8 GPUs of a DGX / HGX node

How to put **every** GPU of an 8-GPU NVSwitch baseboard (DGX B300/B200, HGX
B300/B200/H100, …) under full load, and how to prove that the run actually
exhausted the cards — full tensor-core throughput, full HBM bandwidth and the
full power budget — instead of idling at 60 % and calling it a pass.

The short version:

```bash
# on the node, with the venv from bench/bootstrap_remote.sh
python3 gpu_stress.py --duration 2h --burn-dim 16384 --min-power-fraction 0.95
```

`--burn-dim 16384` is what takes the node from "98 % of the power budget" to
"99.2 %": the per-card autotune maximises TFLOP/s at boost clocks and can pick
a small GEMM that then draws 2–3 % less power. See §4.

Everything below explains what has to be true before that command can work,
what the numbers should look like, and what to do when they do not.

---

## 1. One process per GPU (not one process with 8 threads)

`gpu_stress.py` starts **one worker process per GPU** as soon as more than one
device is selected. This is not a style choice:

* CUDA **graph capture is not safe across threads of one process**. With eight
  GPUs in one process, seven of eight workers died in `cuStreamBeginCapture`
  with `CUDA_ERROR_ILLEGAL_STATE` (401) while the eighth ran fine — the run
  "passed" with one card stressed and seven idle.
* A crashed worker (Xid, OOM, driver reset) only takes down its own GPU, not
  the whole soak test.
* Every card gets its own CUDA context, which is also what avoids the mixed
  addressing-mode failures on heterogeneous nodes.

`--single-process` still exists for debugging a single card, but a multi-GPU
run must not use it.

## 2. Prerequisites: the fabric must be up (error 802)

On an NVSwitch baseboard the GPUs register with the NVLink fabric at boot.
**Until that registration completes, every CUDA call fails with
`CUDA_ERROR_SYSTEM_NOT_READY` (802)** — `nvidia-smi` looks perfectly healthy,
`torch.cuda.device_count()` even returns 8, but the first allocation dies:

```
RuntimeError: Unexpected error from cudaGetDeviceCount() ...
Error 802: system not yet initialized
```

Check the fabric state first — this is the single most common reason a fresh
DGX/HGX node cannot run any GPU workload:

```bash
nvidia-smi -q | grep -A3 "Fabric$"
#   Fabric
#       State   : Completed      <-- must be Completed
#       Status  : Success        <-- must be Success
```

`State: In Progress` means the Fabric Manager is not running (or is missing a
dependency). On B300/B200 baseboards the FM additionally needs the **NVLink
Subnet Manager (NVLSM)** and the `ib_umad` kernel module — without them FM
exits and the fabric never completes.

### Installing the fabric stack (Ubuntu 24.04, driver 580.x)

```bash
# 1. Fabric Manager, matching the driver branch exactly
apt-get install -y nvidia-fabricmanager-580 libibumad3 infiniband-diags

# 2. NVLSM (not in the Ubuntu archive - it lives in NVIDIA's CUDA repo)
curl -fsSL -O https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb
dpkg -i cuda-keyring_1.1-1_all.deb && apt-get update
apt-get install -y --no-install-recommends nvlsm

# 3. The NVSwitch control plane runs over the CX7 bridge's InfiniBand stack
modprobe ib_umad
echo ib_umad >> /etc/modules-load.d/nvidia-fabric.conf

# 4. Start it
systemctl enable --now nvidia-fabricmanager
```

Verify: `systemctl is-active nvidia-fabricmanager` → `active`, and the fabric
state above → `Completed / Success`. The FM log is `/var/log/fabricmanager.log`.

> **Version alignment matters.** Fabric Manager, NVLSM and the kernel driver
> must come from the same branch (580.178.04 here). A mismatch either refuses
> to start or leaves the fabric half-configured.

### Other pre-flight settings

```bash
nvidia-smi -pm 1                  # persistence mode: no driver unload between runs
nvidia-smi --query-gpu=index,name,power.limit,power.default_limit,ecc.mode.current \
           --format=csv
nvidia-smi -q -d POWER | grep -A6 "Module Power Readings"   # GB200/GB300 only
```

* **ECC on** (default) costs a few percent of bandwidth; that is the number to
  report, not the ECC-off one.
* **Module power budget**: on Grace-Blackwell the GPU shares one budget with
  the CPU and memory. If the *module* limit sits below its default, the GPU
  stops early and looks 10-15 % slow on every tensor-core metric. Compare
  `Default Power Limit` with `Current Power Limit` before blaming the card.
* **Power limits** are per card (1100 W on the B300 SXM6 AC measured here).
  Do not raise them for a burn-in — the point is to validate the *shipped*
  configuration.

## 3. Run it

### Bare metal / venv

```bash
cd gpu-stress
python3 gpu_stress.py --duration 2h --min-power-fraction 0.95
```

### Docker

```bash
docker run --rm --gpus all -v "$PWD/stress-logs:/workspace/gpu-stress/stress-logs" \
    gpu-stress --duration 2h --min-power-fraction 0.95
```

`--gpus all` is what makes all eight cards visible; the container needs no
`--privileged` and no host IPC.

### What the flags do

| Flag | Why it matters for an 8-GPU burn-in |
|------|-------------------------------------|
| `--duration 2h` | wall-clock budget; `30m`, `3d`, `1h30m` also work |
| `--min-power-fraction 0.95` | **fails the run** if a card's sustained compute-phase power stays below 95 % of its limit — this is what turns "it ran" into "it was actually stressed" |
| `--mem-fraction 0.8` | share of free VRAM for the buffers; lower it when a server is already resident |
| `--burn-streams 4` | parallel CUDA streams per GPU; more streams = more in-flight work |
| `--burn-dim 0` | `0` = autotune the GEMM dim per card. Set it explicitly (e.g. `16384`) to force a large GEMM: the autotune optimises TFLOP/s at boost clocks, and a small dim that wins there can leave 2–3 % of the power budget unused |
| `--gemm-mix 5:1` | fp8:bf16 GEMM ratio; `6:0` is pure FP8 and reaches ~16 % more TFLOP/s at the same power |
| `--devices all` | default; `--devices 0,3` for a subset |

## 4. What "full load" looks like

The default backend is the FP8 burn: CUDA-graph replays of `_scaled_mm` FP8
GEMMs mixed 5:1 with BF16 GEMMs, plus FP32→BF16 conversions and an HBM copy
phase, on several streams per GPU. It autotunes the GEMM dimension (unless
`--burn-dim` pins it) and the memory buffer size at startup, then reports per
GPU:

```
[GPU 0] backend=fp8 | iterations=401 | compute=2286 TFLOP/s sustained (best 2380) | mem=6062 GB/s sustained (best 6112) | errors=0
    power: sustained 1090 W of 1100 W (99.1%) | peak 1101 W | 71 C | SM 1149 MHz | util 100%
    bandwidth: 78.7% of the 7700 GB/s datasheet HBM peak
...
system: sustained GPU power 8729 W of 8800 W (99.2%) across 8 GPUs
```

Reference numbers measured on an **HGX B300 (8× B300 SXM6 AC, 1100 W cap,
driver 580.178.04, CUDA 13.0, torch 2.14.1+cu130)**, 5-minute runs,
`--gemm-mix 5:1`:

| Metric | Autotuned dim (default) | `--burn-dim 16384` |
|--------|-------------------------|--------------------|
| Sustained power per GPU | 1064–1088 W (96.7–98.9 %) | **1090–1092 W (99.1–99.3 %)** |
| Node power, 8 GPUs | 8647–8690 W of 8800 W (98.3–98.8 %) | **8729 W of 8800 W (99.2 %)** |
| Peak power per GPU | 1070–1100 W | 1096–1111 W |
| FP8+BF16 GEMM throughput | 2243–2368 TFLOP/s sustained | 2286–2366 TFLOP/s sustained |
| HBM copy bandwidth | 6094–6258 GB/s (79.1–81.3 %) | 6042–6102 GB/s (78.5–79.2 %) |
| SM clock under load | 1.14–1.49 GHz | 1.15–1.19 GHz |
| Temperature | 62–71 °C | 65–72 °C |
| Compute errors | 0 | 0 |

The autotuned column is the honest default (each card picks its own fastest
GEMM size, which is what you want when the goal is throughput). The forced-dim
column is what you want when the goal is **maximum power**: every card then
sits at 99 % of its limit instead of 97–99 %, and the spread across the eight
cards collapses to 2 W.

With `--gemm-mix 6:0` (pure FP8, no BF16 GEMMs) the same run reaches
**2645–2781 TFLOP/s sustained** (3279–3409 best) at the same power —
**~16 % more FLOPS for free**, because a BF16 GEMM delivers half the FLOPs of
an FP8 GEMM on the same tensor cores. Use `5:1` to exercise both instruction
paths, `6:0` when the number on the report should be the maximum the card can
do.

How to read it:

* **Power is the binding constraint.** A B300 at its 1100 W limit runs the SMs
  at ~1.2 GHz instead of the 2.03 GHz boost clock. That is correct behaviour,
  not throttling: the datasheet FLOPS figures are quoted at a higher power
  point than this SKU's cap. A card that reaches 99 % of its limit *is* fully
  loaded.
* **Bandwidth is measured on a copy**, so ~80 % of the datasheet HBM peak is
  the practical ceiling (a pure read or write stream gets closer to 90 %; a
  read+write copy cannot). 6.1 TB/s on a 7.7 TB/s part is healthy — a
  16 GiB-buffer copy probe on the same card tops out at 6.56 TB/s.
* **The memory phase is deliberately short** (a few ms per iteration) so the
  card spends almost all of its time in the power-hungry GEMM phase. The
  reported power is the *compute-phase* average, with the first 5 s of ramp
  dropped — averaging the ramp in is what makes a card at its limit look like
  it only reached 60 %.
* **`errors=0`** is the self-check: the captured op chain is a pure function of
  its inputs, so any divergence under load (overheating, unstable clock, bad
  HBM) is counted. A non-zero count fails the run.

## 5. Bandwidth: HBM *and* NVLink

"Full bandwidth" on an 8-GPU baseboard means two different things, and a burn-in
that only measures one of them can miss a broken fabric:

**HBM (per card)** — measured by `gpu_stress.py` itself in the copy phase of
every iteration, reported as a percentage of the datasheet peak. ~80 % of peak
is the ceiling for a read+write copy; a card that sits at 60 % has a problem
(too-small buffers, another process, ECC storm).

**NVLink / NVSwitch (between cards)** — *not* exercised by the burn, because a
compute burn keeps each card in its own HBM. Measure it separately with the
P2P benchmark, which copies between every ordered GPU pair:

```bash
python3 bench/bench.py p2p --devices all      # 512 MiB per direction, 3 s per pair
```

What to expect on a healthy NVSwitch baseboard:

| Generation | Per-GPU NVLink | P2P copy, one direction | All 8 GPUs at once |
|------------|----------------|-------------------------|--------------------|
| B300 / B200 (NVLink 5) | 1.8 TB/s bidirectional | ~760 GB/s measured | ~6 TB/s aggregate |
| H100 / H200 (NVLink 4) | 900 GB/s bidirectional | ~400–450 GB/s | ~3.5 TB/s aggregate |

Measured on the HGX B300 used for the numbers above: **56 ordered pairs, min
758.4 / median 759.9 / max 761.2 GB/s** — a 3 GB/s spread across all pairs,
i.e. every link runs at the same speed. That is 84 % of the 900 GB/s
per-direction peak, which is the normal copy efficiency.

A pair that reports PCIe-like numbers (~25–60 GB/s) is not on the fabric: check
`nvidia-smi nvlink -s` (link state and speed per link) and `nvidia-smi topo -m`
(every pair must read `NV18`/`NV12`, not `SYS` or `PIX`). On a healthy B300 all
18 links per GPU report `53.125 GB/s` and `NVLink is active`; on H100 the same
check reads `25.000 GB/s`.

> The fabric check in §2 proves the *control plane* came up. The P2P numbers
> prove the *data plane* actually runs at speed — a node can pass §2 and still
> have a degraded link.

## 6. Acceptance checklist

A run counts as "the node was fully stressed" only when all of these hold:

| # | Check | Threshold |
|---|-------|-----------|
| 1 | Every GPU reports iterations | 8 of 8, none `FAILED` |
| 2 | Sustained compute-phase power | ≥ 95 % of the card's limit (gate: `--min-power-fraction 0.95`) |
| 3 | Node power | ≈ 8 × limit (8729 W of 8800 W measured) |
| 4 | HBM copy bandwidth | ≥ 75 % of the datasheet peak |
| 5 | Compute errors | 0 on every card |
| 6 | Temperature | stable over the run, no clock collapse |
| 7 | P2P between all pairs | NVLink-class numbers (§5): ≥ 700 GB/s on NVLink 5, ≥ 380 GB/s on NVLink 4 |
| 8 | Exit code | `0` |

Items 1–6 and 8 come from the `gpu_stress.py` summary; item 7 from
`bench.py p2p`. Keep the log file — it is the evidence.

## 7. Evidence in the PDF report

The burn-in numbers belong in the acceptance document, not just in a terminal
scrollback. `gpu_stress.py --json` writes the run summary, and `report.py
--stress` renders it as a dedicated **full-load proof** page inside the
benchmark PDF:

```bash
# 1. on the node: burn-in with a machine-readable summary
python3 gpu_stress.py --duration 2h --burn-dim 16384 \
        --min-power-fraction 0.95 --json stress.json

# 2. locally: benchmark report + burn-in proof in one PDF
python3 bench/report.py bench/results/bench-<stamp>.json \
        -o report.pdf --stress stress.json
```

The page carries a PASSED/FAILED verdict, the configuration that produced the
numbers (backend, GEMM mix, GEMM size, streams, power gate), a per-GPU table of
sustained power with its share of the limit, a second table of sustained
TFLOP/s and HBM bandwidth with its share of the datasheet peak, the node total,
and the acceptance criteria with their outcome. Every figure is read from the
JSON — nothing is retyped, so the PDF cannot drift from the measurement.

`--stress` is repeatable: pass several JSONs (e.g. one per node of a delivery)
and each gets its own page.

## 8. Troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| `Error 802: system not yet initialized` | fabric not registered | start Fabric Manager + NVLSM, `modprobe ib_umad`, check `Fabric State: Completed` |
| `CUDA_ERROR_ILLEGAL_STATE` in `cuStreamBeginCapture` | several GPUs captured from threads of one process | drop `--single-process` (one process per GPU is the default) |
| Only some GPUs report iterations | worker crashed early | read its `FAILED:` line in the summary; the other GPUs keep running |
| Sustained power far below the limit | module power budget, `nvidia-smi -pl` cap, ECC/thermal, or a server already using the card | compare `Default` vs `Current` power limit, check `--mem-fraction` |
| Bandwidth well below ~75 % of HBM peak | buffers too small (low `--mem-fraction`), or another process on the card | raise `--mem-fraction`, stop the other workload |
| Temperature climbing over hours | cooling/airflow | that is what the soak test is for; watch `temp_max_c` in the summary |
| `power sampling: unavailable` | `nvidia-ml-py` not installed | `pip install nvidia-ml-py` (the bench venv already has it) |

## 9. Long soaks

```bash
nohup python3 gpu_stress.py --duration 3d --min-power-fraction 0.95 \
      --log-dir stress-logs > /dev/null 2>&1 &
```

* One log file per run (`stress-logs/gpu-stress-<UTC>.log`) contains the
  parent's summary, the periodic `nvidia-smi` snapshots and every worker line.
* Exit code `0` = every GPU passed, `1` = a compute error, a crashed worker or
  a failed power gate, `2` = no GPU visible to PyTorch.
* The self-check runs every 8 iterations, so a card that starts producing wrong
  results after 40 hours is caught within seconds.

## 10. Related

* `bench/` measures *how fast* the node is (FLOPS per precision, HBM read /
  write / copy / triad, P2P, host↔device, CPU STREAM) and renders a PDF report.
  Use it for acceptance numbers; use `gpu-stress` for the soak.
* `bench/README.md` documents the datasheet-vs-dense trap and the power/clock
  point every number was measured at.
