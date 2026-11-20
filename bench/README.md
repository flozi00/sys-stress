# Hardware benchmark suite (FLOPS, bandwidth, interconnect)

Measures what a GPU node actually delivers — not what the datasheet promises:

| Area | What is measured |
|------|------------------|
| **HBM / GDDR** | read, write, copy and triad (2R+1W) bandwidth in GB/s |
| **Compute** | GEMM throughput in TFLOP/s for fp32, tf32, bf16, fp16, fp8 (e4m3) and fp4 (nvfp4) |
| **Host ↔ device** | pinned-memory H2D / D2H bandwidth (NVLink-C2C on Grace systems, PCIe otherwise) |
| **Device ↔ device** | P2P copy bandwidth between every GPU pair |
| **CPU** | STREAM triad bandwidth and fp32 BLAS GFLOPS |
| **Power** | average/max power, temperature, SM clock and utilisation sampled during the run |

Every case is warmed up, repeated, and reported as *best* plus *median* (best
is the least noisy estimate on a shared machine).  Achieved values are printed
next to the datasheet peak where NVIDIA publishes one, so a report immediately
shows whether a card is healthy or throttled.

## Requirements

* Linux with an NVIDIA driver and `nvidia-smi`
* PyTorch with CUDA (the bootstrap script installs the CUDA 13 build)
* No root, no Docker, no compiler toolchain

## Quick start

The target host is **never** stored in this repository.  Export it per session:

```bash
export SYS_STRESS_HOST=user@host          # ssh target
./bootstrap_remote.sh                     # one-time: uv + venv + torch
./run_remote.sh all                       # full suite, JSON into ./results/
./run_remote.sh --pdf all                 # same, plus a PDF report
```

`run_remote.sh` rsyncs this directory to `~/sys-stress-bench` on the target,
runs `bench.py` there and copies the JSON report back into `./results/`
(git-ignored).

Optional environment variables:

| Variable | Default | Meaning |
|----------|---------|---------|
| `SYS_STRESS_HOST` | *(required)* | ssh target, e.g. `user@host` |
| `SYS_STRESS_REMOTE_DIR` | `sys-stress-bench` | working directory on the target |
| `SYS_STRESS_PYTHON` | `~/bench-venv/bin/python` | interpreter on the target |
| `SYS_STRESS_TORCH_INDEX` | `https://download.pytorch.org/whl/cu130` | wheel index for bootstrap |

## PDF report

`report.py` turns a JSON result into a fixed-layout document (A4) that can be
sent to a customer: cover block, host table, cross-GPU summary, one page per
GPU with achieved-vs-datasheet bars, observations, methodology and sources.

```bash
python3 report.py results/bench-*.json -o report.pdf \
    --title "DGX Station — GPU Benchmark" \
    --subtitle "Leistungsmessung FLOPS, Speicherbandbreite und Interconnect" \
    --operator "primeLine Solutions GmbH"
```

Or let the runner do it:

```bash
export SYS_STRESS_REPORT_TITLE="DGX Station — GPU Benchmark"
export SYS_STRESS_REPORT_SUBTITLE="Kunde / Projekt"
export SYS_STRESS_REPORT_OPERATOR="primeLine Solutions GmbH"
./run_remote.sh --pdf all
```

Options: `--lang de|en` (default `de`), `--notes "…"` for a free-text
paragraph, `--html-only` to skip PDF conversion.  The HTML is always written
next to the PDF.

PDF conversion uses **WeasyPrint** (real page numbers via CSS paged media),
then **headless Chrome**, then **wkhtmltopdf** — whichever is installed.  No
Python dependencies beyond the standard library.

The report derives its findings from the data, not from a template: it states
the share of datasheet peak per metric, whether the card hit its power limit,
the sustained clock versus the maximum, and the peak temperature.  Missing
datasheet figures (NVIDIA publishes no dense FP8/BF16/TF32 numbers for the
RTX PRO 6000 Max-Q) are shown as "kein Dense-Wert" rather than guessed.

## Direct usage on the machine

```bash
python3 bench.py info                       # devices, topology, driver
python3 bench.py hbm                        # memory bandwidth only
python3 bench.py compute --dtype bf16 fp8   # selected precisions
python3 bench.py interconnect               # host <-> device
python3 bench.py p2p                        # device <-> device
python3 bench.py cpu                        # host CPU
python3 bench.py all --json report.json     # everything
```

Useful flags: `--gpu 0 1` (nvidia-smi indices), `--matrix 16384` (GEMM
dimension), `--hbm-gib 8` (buffer size per stream), `--iters 10` (timed
repetitions).

## One process per GPU (why)

On systems that mix GPU addressing modes — for example a Grace-Blackwell
`GB300` running in **ATS** mode next to a discrete `RTX PRO 6000` in **HMM**
mode — the two cards cannot share a CUDA context: `cudaGetDeviceCount()`
returns `cudaErrorSystemNotReady` (802) as soon as both are visible to one
process.  `bench.py` therefore runs every GPU in its own child process pinned
with `CUDA_VISIBLE_DEVICES=<uuid>`, and reports P2P as skipped when the pair
cannot be opened together.  The same quirk also means a plain
`torch.cuda.device_count()` may under-report the installed GPUs; the suite
enumerates via `nvidia-smi` instead.

## Datasheet references

Peaks used for the percentage column (dense, i.e. non-sparse):

* **Blackwell Ultra / HGX B300** — [NVIDIA Blackwell Ultra datasheet](https://resources.nvidia.com/en-us-blackwell-architecture/blackwell-ultra-datasheet),
  "Individual Blackwell Ultra GPU Specifications", HGX B300 column
  (270 GB HBM3E, 7.7 TB/s): FP4 14 PFLOPS, FP8 4.5 PFLOPS, FP16/BF16
  2.25 PFLOPS, TF32 1.1 PFLOPS, FP32 75 TFLOPS, FP64 1.2 TFLOPS.
  The datasheet prints FP8/BF16/TF32 as sparse values with the note
  "dense is ½ sparse spec shown".
* **RTX PRO 6000 Blackwell Max-Q Workstation Edition** — [NVIDIA workstation datasheet](https://www.nvidia.com/content/dam/en-zz/Solutions/products/workstations/professional-desktop-gpus/rtx-pro-6000-max-q/workstation-datasheet-blackwell-rtx-pro-6000-max-q-nvidia-us-5349650-web.pdf):
  96 GB GDDR7, 1792 GB/s, 3511 AI TOPS FP4 *with sparsity* (→ 1755 dense),
  300 W.  NVIDIA publishes no dense FP8/BF16/TF32 figures for this SKU, so
  those rows are reported without a percentage.

## Two traps that make a healthy card look slow

**Power averaging over the ramp.** A GB300 needs about a second to go from
idle to its sustained clock. Averaging that ramp into a 3 s phase understates
power by 40 % or more — it made a card sitting at its 1300 W limit report
550 W. `PowerSampler.since()` therefore drops the first second of every phase
(`skip_seconds=1.0`) and samples at 0.25 s.

**Module power budget.** On Grace-Blackwell the GPU shares a power budget with
the CPU and memory. `nvidia-smi` reports the GPU limit (1300 W) *and* a module
limit; if the platform caps the module below its default (e.g. 1250 W instead
of 1600 W), the GPU stops at ~1125 W and looks 12 % slower than an identical
card in another chassis. The suite records `module_power_limit_w` /
`module_power_default_w` and the report calls it out. Same VBIOS, same board
part number, different platform setting.

## Reading the numbers

* **HBM copy/triad** should land at 75–90 % of the datasheet bandwidth with
  buffers of 8 GiB.  Much lower means ECC overhead, a throttled memory clock
  or a shared card.
* **BF16/FP8 GEMM** at 16384³ typically reaches 70–80 % of dense peak with
  cuBLAS.  FP4 with block scales is lower (50–60 %) because the scaling
  factors cost extra bandwidth.
* **FP32** is the CUDA-core peak and should be close to 100 % — it is the
  cleanest health check for clock throttling.
* **Power** shows whether the workload actually reached the limit.  A card at
  40 % of its power limit while reporting 40 % of peak FLOPS is power-capped,
  not broken.
* Phases run for `--seconds` (default 3 s) each, so the reported clocks are
  **sustained** clocks, not the boost clock a 5 ms burst would show.  On a
  power-limited card that alone explains a 30–50 % gap to the datasheet.

## Reference results

Measured on a DGX Station (driver 595.99.02, torch 2.14.1+cu130, 72-core
Neoverse-V2, 744 GiB RAM), `bench.py all` with defaults:

| Metric | RTX PRO 6000 Max-Q (300 W) | GB300 (1300 W) |
|--------|---------------------------:|---------------:|
| HBM read | 1,638 GB/s (91 %) | 6,116 GB/s (79 %) |
| HBM write | 1,636 GB/s (91 %) | 7,088 GB/s (92 %) |
| HBM copy | 1,463 GB/s (82 %) | 6,436 GB/s (84 %) |
| HBM triad | 1,517 GB/s (85 %) | 6,549 GB/s (85 %) |
| fp32 GEMM | 50.4 TFLOP/s | 72.6 TFLOP/s (97 %) |
| tf32 GEMM | 123.4 TFLOP/s | 797 TFLOP/s (73 %) |
| bf16 GEMM | 264.0 TFLOP/s | 1,521 TFLOP/s (68 %) |
| fp16 GEMM | 257.2 TFLOP/s | 1,433 TFLOP/s (64 %) |
| fp8 GEMM | 504.7 TFLOP/s | 2,861 TFLOP/s (64 %) |
| fp4 GEMM | 958.5 TFLOP/s (55 %) | 6,568 TFLOP/s (47 %) |
| H2D (pinned) | 55.1 GB/s (PCIe 5.0 x16) | 388.5 GB/s (NVLink-C2C) |
| D2H (pinned) | 57.2 GB/s | 380.4 GB/s |
| CPU triad | 203.7 GB/s (72 threads) | — |
| CPU GEMM | 2,424 GFLOP/s | — |

Both cards are power-limited under sustained tensor load: the GB300 settles at
~1.3–1.4 GHz (max 2.07 GHz) and ~1.1 kW of its 1.3 kW budget, the RTX at
~1.4 GHz (max 3.09 GHz) and its full 300 W.  That is why the tensor rows sit
below the boost-clock datasheet peaks while fp32 (which does not hit the power
limit) reaches 97 %.
