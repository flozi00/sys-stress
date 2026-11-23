#!/usr/bin/env python3
"""GPU stress test.

A single, vendor neutral stress test that hammers both the memory subsystem
(memory bandwidth) and the arithmetic units (tensor cores + computation cores)
of every visible GPU. The default `torch` backend does the heavy lifting with
pure PyTorch ops (BF16 GEMMs fused with FP32 ALU work and device copies)
replayed from CUDA graphs, so it runs anywhere a GPU PyTorch build runs with
no compiler toolchain. Optional Triton / Helion backends keep the exact same
test portable across NVIDIA (CUDA) and AMD (ROCm) where Triton is available.

The goal is to keep the GPUs at (or close to) 100% load for a configurable
amount of time -- from a couple of minutes up to several days -- so that the
cooling solution and the long term stability of the cards can be validated.

Every iteration replays CUDA graphs of fused BF16 GEMMs, FP32 elementwise work
and device-to-device copies (default `torch` backend), or -- with the legacy
`triton` backend -- performs:

* a **compute** phase: a long fused-multiply-add loop kept in registers to
  saturate the FP32 ALUs and report an estimated TFLOP/s figure, and
* a **memory** phase: a streaming copy over large buffers to saturate the
  device memory bus and report the achieved bandwidth in GB/s.

The compute phase is also self checking: the op chain is a pure function of its
inputs, so all workers are seeded with identical inputs and every iteration is
compared against a reference checksum.  A
mismatch means the GPU produced a wrong result under load (overheating, unstable
overclock, faulty hardware, ...) and is counted and logged as an error, exactly
like the classic ``gpu_burn`` tool does.

Progress is written both to stdout and to a timestamped log file so that a long
running soak test leaves behind a nice, self contained record of the run.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import logging
import os
import platform
import shutil
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field

try:
    import torch
except Exception as exc:  # pragma: no cover - import guard
    sys.stderr.write(
        "Failed to import torch. Install a GPU enabled PyTorch "
        "build (CUDA for NVIDIA, ROCm for AMD).\n"
        f"Original error: {exc}\n"
    )
    raise

try:
    import triton
    import triton.language as tl

    _TRITON_AVAILABLE = True
except Exception:  # pragma: no cover - environment dependent
    triton = None
    tl = None
    _TRITON_AVAILABLE = False


LOGGER = logging.getLogger("gpu_stress")


# Helion is optional. When available it is used to autotune a matrix-multiply
# based compute phase, which drives the tensor/matrix cores much harder than a
# plain FP32 ALU loop and therefore reaches maximum GPU power draw. If it cannot
# be imported (not installed, or unsupported build) we transparently fall back
# to the pure Triton kernels below.
try:  # pragma: no cover - environment dependent
    import helion
    import helion.language as hl

    _HELION_AVAILABLE = True
except Exception:  # pragma: no cover - environment dependent
    helion = None
    hl = None
    _HELION_AVAILABLE = False


# NVML (the `nvidia-ml-py` package) gives cheap, high resolution power, clock
# and temperature samples.  It is optional: without it the burn still runs, it
# just cannot report the power point it reached -- and "did every card sit at
# its power limit?" is the whole point of a burn-in.
try:  # pragma: no cover - environment dependent
    import pynvml as _nvml

    _NVML_AVAILABLE = True
except Exception:  # pragma: no cover - environment dependent
    _nvml = None
    _NVML_AVAILABLE = False


# --------------------------------------------------------------------------- #
# Triton kernels (optional - only when triton imports AND compiles)
# --------------------------------------------------------------------------- #
if _TRITON_AVAILABLE:

    @triton.jit
    def _compute_kernel(x_ptr, out_ptr, n_elements, n_iters, BLOCK_SIZE: tl.constexpr):
        """Register bound fused-multiply-add loop to saturate the FP32 ALUs.

        Each inner iteration issues two FMA-like operations (4 FLOPs total) that
        depend on the previous result, which keeps the pipeline busy without
        touching memory.  Values are clamped to ``[-2, 2]`` every iteration to
        prevent float32 overflow (with 2048 iterations the unclamped recurrence
        would overflow to Inf/NaN, making the self-check meaningless).
        """
        pid = tl.program_id(axis=0).to(tl.int64)
        offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        x = tl.load(x_ptr + offsets, mask=mask, other=0.0)

        a = x
        b = x + 1.0
        for _ in range(n_iters):
            a = a * b + b
            b = b * a + a
            a = tl.minimum(tl.maximum(a, -2.0), 2.0)
            b = tl.minimum(tl.maximum(b, -2.0), 2.0)
        tl.store(out_ptr + offsets, a + b, mask=mask)

    @triton.jit
    def _copy_kernel(src_ptr, dst_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
        """Streaming copy used to measure/stress memory bandwidth (read + write)."""
        pid = tl.program_id(axis=0).to(tl.int64)
        offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        val = tl.load(src_ptr + offsets, mask=mask)
        tl.store(dst_ptr + offsets, val, mask=mask)


# FLOPs performed per input element in a single `_compute_kernel` launch.
# 2 FMA operations per inner iteration, each FMA counts as 2 FLOPs.
_FLOPS_PER_ELEM_PER_ITER = 4


# --------------------------------------------------------------------------- #
# Helion kernel (optional, autotuned matmul)
# --------------------------------------------------------------------------- #
if _HELION_AVAILABLE:  # pragma: no cover - requires a GPU + helion

    @helion.kernel()
    def _helion_matmul(
        x: "torch.Tensor", y: "torch.Tensor", out: "torch.Tensor"
    ) -> "torch.Tensor":
        """Autotuned matmul used to drive the matrix cores at maximum power.

        Helion searches hundreds of Triton implementations on the first call and
        keeps the fastest one for the running hardware, so the same source
        reaches peak throughput on both NVIDIA and AMD GPUs. The output buffer is
        passed in and written in place to avoid reallocating it every iteration.
        """
        m, k = x.size()
        _, n = y.size()
        for tile_m, tile_n in hl.tile([m, n]):
            acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
            for tile_k in hl.tile(k):
                acc = torch.addmm(acc, x[tile_m, tile_k], y[tile_k, tile_n])
            out[tile_m, tile_n] = acc
        return out


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def parse_duration(value: str) -> float:
    """Parse a human friendly duration into seconds.

    Accepts a bare number (seconds) or a value with a unit suffix, e.g.
    ``120`` -> 120s, ``30m`` -> 1800s, ``2h`` -> 7200s, ``3d`` -> 259200s.
    Compound values such as ``1h30m`` are also supported.
    """
    value = value.strip().lower()
    if not value:
        raise argparse.ArgumentTypeError("duration must not be empty")

    units = {"d": 86400.0, "h": 3600.0, "m": 60.0, "s": 1.0}

    # Bare number -> seconds.
    try:
        return float(value)
    except ValueError:
        pass

    total = 0.0
    number = ""
    matched = False
    for ch in value:
        if ch.isdigit() or ch == ".":
            number += ch
        elif ch in units:
            if not number:
                raise argparse.ArgumentTypeError(f"invalid duration: {value!r}")
            total += float(number) * units[ch]
            number = ""
            matched = True
        else:
            raise argparse.ArgumentTypeError(f"invalid duration: {value!r}")

    if number:  # trailing number without unit is treated as seconds
        total += float(number)
        matched = True
    if not matched:
        raise argparse.ArgumentTypeError(f"invalid duration: {value!r}")
    return total


def format_duration(seconds: float) -> str:
    seconds = int(seconds)
    days, rem = divmod(seconds, 86400)
    hours, rem = divmod(rem, 3600)
    minutes, secs = divmod(rem, 60)
    parts = []
    if days:
        parts.append(f"{days}d")
    if hours or days:
        parts.append(f"{hours}h")
    if minutes or hours or days:
        parts.append(f"{minutes}m")
    parts.append(f"{secs}s")
    return "".join(parts)


def detect_vendor() -> str:
    """Return 'AMD', 'NVIDIA' or 'Unknown' based on the torch build/device."""
    if getattr(torch.version, "hip", None):
        return "AMD"
    if getattr(torch.version, "cuda", None):
        return "NVIDIA"
    return "Unknown"


def _smi_command() -> list[str] | None:
    """Return an SMI command (as argv list) able to emit a CSV line per GPU."""
    if shutil.which("nvidia-smi"):
        return [
            "nvidia-smi",
            "--query-gpu=index,name,utilization.gpu,memory.used,memory.total,"
            "power.draw,temperature.gpu,clocks.sm",
            "--format=csv,noheader,nounits",
        ]
    if shutil.which("rocm-smi"):
        return ["rocm-smi", "--showuse", "--showtemp", "--showpower", "--csv"]
    return None


def sample_smi() -> str | None:
    """Return a one-line-per-GPU SMI snapshot, or None if unavailable."""
    cmd = _smi_command()
    if not cmd:
        return None
    try:
        out = subprocess.run(
            cmd, capture_output=True, text=True, timeout=15, check=False
        )
    except Exception:  # pragma: no cover - environment dependent
        return None
    text = out.stdout.strip()
    return text or None


# Published HBM bandwidth per part, in GB/s.  Only used to print the achieved
# copy bandwidth as a share of the datasheet peak -- a card that copies at
# 60 % of its HBM peak is either throttled or not being stressed hard enough.
# Anything not listed is reported as an absolute number without a percentage.
HBM_PEAK_GBS: dict[str, float] = {
    "B300": 7700.0,
    "GB300": 7700.0,
    "B200": 8000.0,
    "GB200": 8000.0,
    "H200": 4800.0,
    "H100": 3350.0,
    "A100": 2039.0,
    "MI325X": 6000.0,
    "MI300X": 5300.0,
    "RTX PRO 6000": 1792.0,
    "RTX 5090": 1792.0,
    "RTX 4090": 1008.0,
}


def hbm_peak_gbs(name: str) -> float:
    """Datasheet HBM bandwidth for a device name, or 0.0 when unknown."""
    for key, value in HBM_PEAK_GBS.items():
        if key.lower() in name.lower():
            return value
    return 0.0


class PowerSampler(threading.Thread):
    """Sample one GPU's power, clock and temperature while the burn runs.

    The burn loop labels each phase ("compute" / "mem"), so the summary can
    report the power of the *compute* phase -- the phase that has to sit at
    the card's power limit -- separately from the memory phase and the idle
    ramp.  Averaging the ramp into the number is what makes a card at its
    limit look like it only reached 60 %.
    """

    def __init__(self, device: int, interval: float = 0.25):
        super().__init__(daemon=True)
        self.device = device
        self.interval = interval
        self.phase = "ramp"
        self.samples: list[tuple[float, str, float, int, int, int]] = []
        self.power_limit_w = 0.0
        self._stop_event = threading.Event()
        self._t0 = time.time()
        self._handle = None
        nvml = _nvml
        if nvml is not None:
            try:
                nvml.nvmlInit()
                self._handle = nvml.nvmlDeviceGetHandleByIndex(device)
                self.power_limit_w = (
                    nvml.nvmlDeviceGetPowerManagementLimit(self._handle) / 1000.0
                )
            except Exception:  # pragma: no cover - environment dependent
                self._handle = None

    @property
    def available(self) -> bool:
        return self._handle is not None

    def reset(self) -> None:
        """Drop the setup/autotune samples and restart the ramp clock."""
        self.samples.clear()
        self._t0 = time.time()
        self.phase = "ramp"

    def run(self) -> None:  # pragma: no cover - timing dependent
        nvml = _nvml
        if self._handle is None or nvml is None:
            return
        while not self._stop_event.is_set():
            try:
                self.samples.append(
                    (
                        time.time(),
                        self.phase,
                        nvml.nvmlDeviceGetPowerUsage(self._handle) / 1000.0,
                        nvml.nvmlDeviceGetClockInfo(self._handle, nvml.NVML_CLOCK_SM),
                        nvml.nvmlDeviceGetTemperature(
                            self._handle, nvml.NVML_TEMPERATURE_GPU
                        ),
                        nvml.nvmlDeviceGetUtilizationRates(self._handle).gpu,
                    )
                )
            except Exception:  # pragma: no cover - environment dependent
                pass
            self._stop_event.wait(self.interval)

    def stop(self) -> None:
        self._stop_event.set()

    def summary(self, skip_seconds: float = 5.0) -> dict:
        """Sustained power/clock/temperature, ramp excluded."""
        if not self.samples:
            return {}
        usable = [s for s in self.samples if s[0] - self._t0 >= skip_seconds]
        if not usable:
            usable = list(self.samples)
        compute = [s for s in usable if s[1] == "compute"] or usable

        def mean(values: list[float]) -> float:
            return sum(values) / len(values) if values else 0.0

        return {
            "power_mean_w": round(mean([s[2] for s in usable]), 1),
            "power_compute_w": round(mean([s[2] for s in compute]), 1),
            "power_max_w": round(max(s[2] for s in usable), 1),
            "power_limit_w": round(self.power_limit_w, 1),
            "clock_sm_mhz": round(mean([float(s[3]) for s in compute]), 0),
            "temp_max_c": max(s[4] for s in usable),
            "util_pct": round(mean([float(s[5]) for s in compute]), 1),
            "samples": len(usable),
        }


# --------------------------------------------------------------------------- #
# Per-GPU worker
# --------------------------------------------------------------------------- #
@dataclass
class GpuResult:
    device: int
    name: str = ""
    iterations: int = 0
    errors: int = 0
    best_tflops: float = 0.0
    last_tflops: float = 0.0
    best_bandwidth: float = 0.0
    last_bandwidth: float = 0.0
    sum_tflops: float = 0.0
    sum_bandwidth: float = 0.0
    backend: str = "triton"
    failed: bool = False
    message: str = ""
    # Power point reached during the run (NVML, ramp excluded).
    power_mean_w: float = 0.0
    power_compute_w: float = 0.0
    power_max_w: float = 0.0
    power_limit_w: float = 0.0
    clock_sm_mhz: float = 0.0
    temp_max_c: float = 0.0
    util_pct: float = 0.0
    hbm_peak_gbs: float = 0.0

    def mean_tflops(self) -> float:
        """Sustained (per-iteration average) compute throughput."""
        return self.sum_tflops / self.iterations if self.iterations else 0.0

    def mean_bandwidth(self) -> float:
        """Sustained (per-iteration average) memory bandwidth."""
        return self.sum_bandwidth / self.iterations if self.iterations else 0.0

    def power_fraction(self) -> float:
        """Sustained compute-phase power as a share of the card's limit."""
        if self.power_limit_w <= 0:
            return 0.0
        return self.power_compute_w / self.power_limit_w

    def bandwidth_fraction(self) -> float:
        """Sustained copy bandwidth as a share of the datasheet HBM peak."""
        if self.hbm_peak_gbs <= 0:
            return 0.0
        return self.mean_bandwidth() / self.hbm_peak_gbs


@dataclass
class StressConfig:
    duration: float
    mem_fraction: float
    compute_iters: int
    block_size: int
    error_check: bool
    log_interval: float
    compute_backend: str = "auto"
    devices: list[int] = field(default_factory=list)
    burn_streams: int = 4
    burn_dim: int = 0  # 0 = autotune (fp8) / 8192 (torch)
    burn_replays: int = 20
    gemm_mix: str = "5:1"


def parse_gemm_mix(value: str) -> str:
    """Validate an ``fp8:bf16`` GEMM ratio such as ``5:1`` or ``6:0``."""
    parts = value.replace(" ", "").split(":")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError(f"invalid gemm mix: {value!r} (expected fp8:bf16)")
    try:
        fp8, bf16 = int(parts[0]), int(parts[1])
    except ValueError:
        raise argparse.ArgumentTypeError(f"invalid gemm mix: {value!r} (expected fp8:bf16)")
    if fp8 < 0 or bf16 < 0 or fp8 + bf16 < 1:
        raise argparse.ArgumentTypeError(f"invalid gemm mix: {value!r} (need at least one GEMM)")
    return f"{fp8}:{bf16}"


def gemm_mix_counts(value: str) -> tuple[int, int]:
    """Return ``(fp8_gemms, bf16_gemms)`` for a validated mix string."""
    fp8, bf16 = value.split(":")
    return int(fp8), int(bf16)


def resolve_backend(requested: str) -> str:
    """Resolve the effective compute backend from the requested value."""
    requested = requested.strip().lower()
    if requested == "torch":
        return "torch"
    if requested == "helion":
        if not _HELION_AVAILABLE or not _TRITON_AVAILABLE:
            LOGGER.warning(
                "Helion backend requested but helion/triton is not importable; "
                "falling back to the pure-torch burn backend."
            )
            return "torch"
        return "helion"
    if requested == "triton":
        if not _TRITON_AVAILABLE:
            LOGGER.warning(
                "Triton backend requested but triton is not importable; "
                "falling back to the pure-torch burn backend."
            )
            return "torch"
        return "triton"
    if requested == "fp8":
        return "fp8"
    # "auto": the FP8 torch burn if the hardware supports _scaled_mm,
    # otherwise the BF16 torch burn. Both need no Triton/gcc toolchain
    # and reach the card power limit.
    if torch.cuda.is_available():
        try:
            _probe = torch.zeros(64, 64, device="cuda", dtype=torch.float32).to(
                torch.float8_e4m3fn
            )
            _probe_t = _probe.t().contiguous().t()
            _s = torch.tensor(1.0, device="cuda")
            torch._scaled_mm(_probe, _probe_t, scale_a=_s, scale_b=_s)
            return "fp8"
        except Exception:
            pass
    return "torch"


def _has_fp8_scaled_mm(device) -> bool:
    """True if torch._scaled_mm fp8 works on this device/build."""
    try:
        probe = torch.zeros(32, 32, device=device, dtype=torch.float32).to(
            torch.float8_e4m3fn
        )
        probe_t = probe.t().contiguous().t()
        s = torch.tensor(1.0, device=device)
        out = torch._scaled_mm(probe, probe_t, scale_a=s, scale_b=s)
        del probe, probe_t, s, out
        return True
    except Exception:
        return False


def _make_fp8_buffers(dev, n: int) -> tuple:
    """One stream's worth of fp8-burn buffers (18*n^2 bytes)."""
    fa = torch.randn(n, n, device=dev).to(torch.float8_e4m3fn)
    fb = torch.randn(n, n, device=dev).to(torch.float8_e4m3fn).t().contiguous().t()
    # zeros, not empty: with a pure-fp8 or pure-bf16 mix one of the two
    # outputs is never written, and the self-check must not compare
    # uninitialised memory (NaN != NaN would fake a compute error).
    fc = torch.zeros(n, n, device=dev, dtype=torch.bfloat16)
    fa_f32 = torch.randn(n, n, device=dev, dtype=torch.float32)
    fb_f32 = torch.randn(n, n, device=dev, dtype=torch.float32)
    sa = torch.randn(n, n, device=dev, dtype=torch.bfloat16)
    sb = torch.randn(n, n, device=dev, dtype=torch.bfloat16)
    sc = torch.zeros(n, n, device=dev, dtype=torch.bfloat16)
    return fa, fb, fc, fa_f32, fb_f32, sa, sb, sc


def _prepare_fp8_buffers(dev, n: int, n_streams: int) -> list:
    """Allocate ``n_streams`` buffer sets, shrinking ``n`` until they fit."""
    for _ in range(6):
        try:
            return [_make_fp8_buffers(dev, n) for _ in range(n_streams)]
        except torch.OutOfMemoryError:
            torch.cuda.empty_cache()
            n = max(1024, (n * 3 // 4 // 256) * 256)
    raise torch.OutOfMemoryError("could not fit fp8 burn buffers in VRAM")


def _autotune_fp8_dim(
    device: int,
    dev,
    cfg: StressConfig,
    n_streams: int,
) -> tuple[int, list]:
    """Autotune the matrix dim for the fp8 burn: sweep candidate dims that
    fit the free-VRAM budget, benchmark each with a short CUDA-graph replay
    and return (best_n, prepared_buffers) for the winner.

    This mirrors what Helion does for its matmul kernel, but at the GEMM
    shape level: the fastest achievable TFLOP/s on this silicon is found
    empirically instead of assuming one fixed size.
    """
    free_bytes, _ = torch.cuda.mem_get_info(device)
    budget = int(free_bytes * max(0.05, min(cfg.mem_fraction, 0.95)))
    # Per stream the buffers are n^2-sized: fa/fb fp8 (2*n^2 B) + fc bf16
    # (2*n^2) + fa_f32/fb_f32 (8*n^2) + sa/sb/sc bf16 (6*n^2) = 18*n^2
    # bytes, plus cuBLAS workspace headroom.
    max_n = int((budget / (n_streams * 18.0)) ** 0.5 // 64 * 64)
    candidates = [
        c
        for c in (16384, 12288, 8192, 7168, 6144, 5120, 4096, 3072, 2048)
        if c <= max_n
    ]
    if not candidates:
        candidates = [min(1024, max_n)]
    torch.manual_seed(1234)
    torch.cuda.manual_seed_all(1234)
    s = torch.tensor(1.0, device=dev)
    one = torch.ones((), device=dev)
    n_fp8, n_bf16 = gemm_mix_counts(cfg.gemm_mix)


    best_n, best_tflops, best_bufs = -1, -1.0, None
    with torch.no_grad():
        for cand in candidates:
            try:
                bufs = [_make_fp8_buffers(dev, cand) for _ in range(n_streams)]
                torch.cuda.synchronize(device)
                # Warmup so cuBLAS picks its fastest algorithm before capture.
                for (fa, fb, fc, fa_f32, fb_f32, sa, sb, sc) in bufs:
                    for _ in range(3):
                        torch._scaled_mm(fa, fb, scale_a=s, scale_b=s, out_dtype=torch.bfloat16)
                        torch.mm(sa, sb, out=sc)
                    fc.copy_(fa_f32.to(torch.bfloat16))
                torch.cuda.synchronize(device)
                # Capture one repeat unit: n_fp8 fp8 GEMMs + n_bf16 bf16 GEMMs
                # (drives both tensor-core paths, filling the wave tail).
                reps = max(2, min(8, cfg.burn_replays))
                graphs = []
                for (fa, fb, fc, fa_f32, fb_f32, sa, sb, sc) in bufs:
                    st = torch.cuda.Stream(device=device)
                    g = torch.cuda.CUDAGraph()
                    with torch.cuda.stream(st), torch.cuda.graph(g):
                        for i in range(reps):
                            for _ in range(n_fp8):
                                torch._scaled_mm(fa, fb, scale_a=s, scale_b=s, out_dtype=torch.bfloat16)
                            for _ in range(n_bf16):
                                torch.mm(sa, sb, out=sc)
                            if i % 4 == 3:
                                fa_f32.to(torch.bfloat16)
                                fb_f32.to(torch.bfloat16)
                    graphs.append((st, g))
                # Bench: 3 timed replays, all streams.
                torch.cuda.synchronize(device)
                t0 = time.perf_counter()
                for _ in range(3):
                    for st, g in graphs:
                        with torch.cuda.stream(st):
                            g.replay()
                    torch.cuda.synchronize(device)
                ms = (time.perf_counter() - t0) / 3 * 1000.0
                # FLOPs per graph: reps*(n_fp8 + n_bf16) gemms of 2n^3.
                flops = reps * (n_fp8 + n_bf16) * 2.0 * cand**3 * n_streams
                tflops = flops / (ms / 1000.0) / 1e12
                LOGGER.info(
                    "[GPU %d] autotune fp8 dim=%d: %.0f TFLOP/s (budget max n=%d)",
                    device, cand, tflops, max_n,
                )
                # Candidates are tried largest-first. A smaller dim must beat
                # the incumbent by 3 % to win: at boost clocks a small GEMM can
                # be marginally faster, but at the power limit it draws less
                # power, which is the opposite of what a burn-in wants.
                if tflops > best_tflops * 1.03:
                    if best_bufs is not None:
                        del best_bufs
                    best_n, best_tflops, best_bufs = cand, tflops, bufs
                else:
                    del bufs
                for st, g in graphs:
                    del g
                torch.cuda.empty_cache()
            except torch.OutOfMemoryError:
                del bufs
                torch.cuda.empty_cache()
                continue
    if best_bufs is None:
        raise torch.OutOfMemoryError("autotune could not fit any fp8 candidate")
    del s, one
    return best_n, best_bufs


def _run_fp8_burn(
    device: int,
    cfg: StressConfig,
    deadline: float,
    result: GpuResult,
    stop: threading.Event,
    sampler: PowerSampler,
):
    """FP8 tensor-core burn (max TFLOP/s on modern NVIDIA silicon).

    Uses torch._scaled_mm (cuBLASLt fp8) GEMMs mixed 5:1 with BF16 GEMMs so
    both fp8 and bf16 tensor-core pipelines stay busy, plus periodic fp32→
    bf16 conversions for HBM traffic. The matrix dim is autotuned at start
    (see _autotune_fp8_dim). No Triton/gcc needed; requires a GPU + torch
    build with working _scaled_mm (Hopper/Blackwell), else falls back to the
    bf16 torch burn.
    """
    result.backend = "fp8"
    try:
        torch.cuda.set_device(device)
        dev = torch.device(f"cuda:{device}")
        if not _has_fp8_scaled_mm(dev):
            LOGGER.warning(
                "[GPU %d] fp8 backend selected but _scaled_mm unavailable; "
                "falling back to bf16 torch burn",
                device,
            )
            _run_torch_burn(device, cfg, deadline, result, stop, sampler)
            return
        n_streams = max(1, cfg.burn_streams)
        reps = max(2, min(8, cfg.burn_replays))
        n_fp8, n_bf16 = gemm_mix_counts(cfg.gemm_mix)

        if cfg.burn_dim > 0:
            # Explicit dim: skip the autotune (a power-focused burn wants a
            # large GEMM, not the fastest one at boost clocks).
            n = max(1024, cfg.burn_dim)
            streams_bufs = _prepare_fp8_buffers(dev, n, n_streams)
            LOGGER.info("[GPU %d] fp8 dim forced to %d (autotune skipped)",
                        device, n)
        else:
            n, streams_bufs = _autotune_fp8_dim(device, dev, cfg, n_streams)
        s = torch.tensor(1.0, device=dev)

        # Autotuned memory phase: a dedicated alternating-buffers copy sweep
        # (src->dst, alt->src) keeps read+write channels saturated. Buffers
        # sized from the VRAM left after the GEMM buffers, capped.
        free_bytes, _ = torch.cuda.mem_get_info(device)
        mem_budget = int(free_bytes * max(0.05, min(cfg.mem_fraction, 0.95)) * 0.5)
        mem_elems = max(1, mem_budget // (3 * 4))
        mem_elems = min(mem_elems, (4 * 2**30) // 4)  # cap 4 GiB per buffer
        if cfg.mem_fraction >= 0.15 and mem_elems > 1024:
            ms_src = torch.randn(mem_elems, device=dev, dtype=torch.float32)
            ms_dst = torch.empty_like(ms_src)
            ms_alt = torch.randn(mem_elems, device=dev, dtype=torch.float32)
            mem_bytes = mem_elems * 4
        else:
            ms_src = ms_dst = ms_alt = None
            mem_bytes = 0
        torch.cuda.synchronize(device)

        # Warmup again with the winning buffers (freed the graphs above).
        with torch.no_grad():
            for (fa, fb, fc, fa_f32, fb_f32, sa, sb, sc) in streams_bufs:
                for _ in range(3):
                    torch._scaled_mm(fa, fb, scale_a=s, scale_b=s, out_dtype=torch.bfloat16)
                    torch.mm(sa, sb, out=sc)
            torch.cuda.synchronize(device)

            graphs = []
            for (fa, fb, fc, fa_f32, fb_f32, sa, sb, sc) in streams_bufs:
                st = torch.cuda.Stream(device=device)
                g = torch.cuda.CUDAGraph()
                with torch.cuda.stream(st), torch.cuda.graph(g):
                    for i in range(reps):
                        for _ in range(n_fp8):
                            torch._scaled_mm(fa, fb, scale_a=s, scale_b=s, out_dtype=torch.bfloat16)
                        for _ in range(n_bf16):
                            torch.mm(sa, sb, out=sc)
                        if i % 4 == 3:
                            fa_f32.to(torch.bfloat16)
                            fb_f32.to(torch.bfloat16)
                graphs.append((st, g))

            # Reference checksums from one replay: the captured chain is a
            # pure function of its fixed inputs, so any later divergence is
            # a real compute error.
            for st, g in graphs:
                with torch.cuda.stream(st):
                    g.replay()
            torch.cuda.synchronize(device)
            ref_sums = [
                (bufs[2].float().sum().item(), bufs[6].float().sum().item())
                for bufs in streams_bufs
            ]

        flops_per_round = reps * (n_fp8 + n_bf16) * 2.0 * n * n * n * n_streams
        # Approx HBM+L2 bytes per round: each fp8/bf16 GEMM unit reads fa/fb
        # (fp8) and sa/sb (bf16) once per gemm while fc/sc/sa/sb rewrite in
        # place => ~ (2*n_fp8 + 2*(n_fp8+n_bf16) + conv 3/4th iter) * n^2.
        bytes_per_round = (
            reps * (2.0 * n_fp8 + 2.0 * (n_fp8 + n_bf16) + 3.0) * n * n * n_streams
        )
        LOGGER.info(
            "[GPU %d] %s | backend=fp8 | %dx (%dxfp8+%dxbf16) gemm %dx%d + "
            "fp32->bf16 conversions | %d streams",
            device, torch.cuda.get_device_name(device), reps, n_fp8, n_bf16, n, n, n_streams,
        )

        last_log = time.time()
        local_iters = 0
        sampler.reset()
        with torch.no_grad():
            while time.time() < deadline and not stop.is_set():
                # Device-wide sync timing (CUDA events don't track the
                # non-blocking per-stream graphs). Phases are timed
                # separately so each metric reflects its own subsystem,
                # not an amortized blend.
                torch.cuda.synchronize(device)
                sampler.phase = "compute"
                t_start = time.perf_counter()
                for st, g in graphs:
                    with torch.cuda.stream(st):
                        g.replay()
                torch.cuda.synchronize(device)
                t_gemm = time.perf_counter()
                # Memory phase: alternating copies saturate read+write HBM.
                if ms_src is not None:
                    sampler.phase = "mem"
                    ms_dst.copy_(ms_src)
                    ms_src.copy_(ms_alt)
                    torch.cuda.synchronize(device)
                    t_mem = time.perf_counter()
                    gemm_ms = max((t_gemm - t_start) * 1000.0, 1e-3)
                    mem_ms = max((t_mem - t_gemm) * 1000.0, 1e-3)
                    # 2 copies x read+write of the buffer bytes.
                    bw = (2.0 * 2.0 * mem_bytes) / (mem_ms / 1000.0) / 1e9
                else:
                    t_mem = time.perf_counter()
                    gemm_ms = max((t_gemm - t_start) * 1000.0, 1e-3)
                    mem_ms = 0.0
                    bw = bytes_per_round / (gemm_ms / 1000.0) / 1e9
                tflops = flops_per_round / (gemm_ms / 1000.0) / 1e12
                result.last_tflops = tflops
                result.best_tflops = max(result.best_tflops, tflops)
                result.sum_tflops += tflops
                result.last_bandwidth = bw
                result.best_bandwidth = max(result.best_bandwidth, bw)
                result.sum_bandwidth += bw
                local_iters += 1
                result.iterations = local_iters

                if cfg.error_check and (local_iters & 7) == 0:
                    for i, bufs in enumerate(streams_bufs):
                        cs = (bufs[2].float().sum().item(), bufs[6].float().sum().item())
                        if cs != ref_sums[i]:
                            result.errors += 1
                            LOGGER.error(
                                "[GPU %d] computation mismatch on stream %d!",
                                device, i,
                            )
                            ref_sums[i] = cs

                now = time.time()
                if now - last_log >= cfg.log_interval:
                    remaining = max(0.0, deadline - now)
                    LOGGER.info(
                        "[GPU %d] iter=%d | compute=%.1f TFLOP/s | mem=%.1f GB/s | "
                        "errors=%d | remaining=%s",
                        device, result.iterations, tflops, bw, result.errors,
                        format_duration(remaining),
                    )
                    last_log = now
    except Exception as exc:  # pragma: no cover - hardware dependent
        result.failed = True
        result.message = str(exc)
        LOGGER.exception("[GPU %d] worker crashed: %s", device, exc)


def _run_torch_burn(
    device: int,
    cfg: StressConfig,
    deadline: float,
    result: GpuResult,
    stop: threading.Event,
    sampler: PowerSampler,
):
    """Pure-torch max-power burn (no Triton/gcc needed).

    Each worker replays CUDA-graph-captured BF16 GEMMs (tensor cores) fused
    with FP32 addcmul ALU work and D2D copies (HBM) in a tight loop, one
    CUDA stream per torch thread. A fixed input produces a fixed output, so
    the self-check compares the graph output against a CPU-side reference
    computed once with the same op sequence on identical inputs.
    """
    result.backend = "torch"
    try:
        torch.cuda.set_device(device)
        dev = torch.device(f"cuda:{device}")
        n_streams = max(1, cfg.burn_streams)
        n = max(1024, cfg.burn_dim or 8192)
        replays = max(1, cfg.burn_replays)

        # Size the matrices to the free-VRAM budget. Per stream we hold
        # 3x bf16 (n,n) + 4x fp32 vectors of n*n. Shrink n until it fits,
        # then allocate defensively: on OOM, release everything, shrink
        # further and retry (free VRAM can shift under us, e.g. KV cache).
        streams_bufs: list = []
        graphs: list = []
        ref_sums: list = []
        ref_z: list = []
        for attempt in range(6):
            per_stream = 3 * (n * n * 2) + 4 * (n * n * 4)
            need = int(per_stream * n_streams * 1.3)
            free_bytes, _ = torch.cuda.mem_get_info(device)
            budget = int(free_bytes * max(0.05, min(cfg.mem_fraction, 0.95)))
            while need > budget and n > 1024:
                n = max(1024, (n * 3 // 4 // 256) * 256)
                per_stream = 3 * (n * n * 2) + 4 * (n * n * 4)
                need = int(per_stream * n_streams * 1.3)
            try:
                streams_bufs = []
                for _ in range(n_streams):
                    a = torch.randn(n, n, device=dev, dtype=torch.bfloat16)
                    b = torch.randn(n, n, device=dev, dtype=torch.bfloat16)
                    c = torch.empty(n, n, device=dev, dtype=torch.bfloat16)
                    x = torch.randn(n * n, device=dev, dtype=torch.float32)
                    y = torch.randn(n * n, device=dev, dtype=torch.float32)
                    z = torch.empty(n * n, device=dev, dtype=torch.float32)
                    m1 = torch.randn(n * n, device=dev, dtype=torch.float32)
                    m2 = torch.empty(n * n, device=dev, dtype=torch.float32)
                    streams_bufs.append((a, b, c, x, y, z, m1, m2))
                torch.cuda.synchronize()

                # Warmup so cublas picks the fastest algorithm before capture.
                for a, b, c, x, y, z, m1, m2 in streams_bufs:
                    for _ in range(5):
                        torch.mm(a, b, out=c)
                        torch.addcmul(y, x, y, out=z)
                    torch.cuda.synchronize()

                graphs = []
                for a, b, c, x, y, z, m1, m2 in streams_bufs:
                    s = torch.cuda.Stream(device=device)
                    g = torch.cuda.CUDAGraph()
                    with torch.cuda.stream(s), torch.cuda.graph(g):
                        for _ in range(replays):
                            torch.mm(a, b, out=c)
                            torch.addcmul(y, x, y, out=z)
                            z.clamp_(-2.0, 2.0)
                            m2.copy_(m1)
                    graphs.append((s, g))

                # Reference checksums: the captured op chain is a pure function
                # of its inputs (mm reads a/b, addcmul reads x/y; outputs c/z
                # are rewritten every replay), so reseed every stream with
                # IDENTICAL inputs and compute the reference once from those
                # same values. Any later divergence is a real compute error.
                torch.manual_seed(1234)
                torch.cuda.manual_seed_all(1234)
                seed_a = torch.randn(n, n, device=dev, dtype=torch.bfloat16)
                seed_b = torch.randn(n, n, device=dev, dtype=torch.bfloat16)
                seed_x = torch.randn(n * n, device=dev, dtype=torch.float32)
                seed_y = torch.randn(n * n, device=dev, dtype=torch.float32)
                seed_m = torch.randn(n * n, device=dev, dtype=torch.float32)
                for a, b, c, x, y, z, m1, m2 in streams_bufs:
                    a.copy_(seed_a)
                    b.copy_(seed_b)
                    x.copy_(seed_x)
                    y.copy_(seed_y)
                    m1.copy_(seed_m)
                torch.cuda.synchronize(device)
                with torch.no_grad():
                    chk_c = torch.mm(seed_a, seed_b)
                    chk_z = torch.addcmul(seed_y, seed_x, seed_y)
                    chk_z.clamp_(-2.0, 2.0)
                    ref_sums = [chk_c.float().sum().item()] * n_streams
                    ref_z = [chk_z.sum().item()] * n_streams
                    del chk_c, chk_z
                del seed_a, seed_b, seed_x, seed_y, seed_m
                break
            except torch.OutOfMemoryError:
                del streams_bufs
                del graphs
                streams_bufs = []
                graphs = []
                torch.cuda.empty_cache()
                n = max(1024, (n * 3 // 4 // 256) * 256)
                if n <= 1024 and attempt >= 2 and n_streams > 1:
                    n_streams -= 1
                    n = max(1024, cfg.burn_dim or 8192)
                LOGGER.warning(
                    "[GPU %d] OOM during setup, retrying smaller (n=%d, streams=%d)",
                    device,
                    n,
                    n_streams,
                )
        else:
            raise torch.OutOfMemoryError("could not fit burn buffers in VRAM")
        if not graphs:
            raise torch.OutOfMemoryError("could not fit burn buffers in VRAM")

        flops_per_graph = replays * (2.0 * n * n * n + 2.0 * n * n)
        copy_bytes_per_graph = replays * (2 * n * n * 4)
        LOGGER.info(
            "[GPU %d] %s | backend=torch | %dx bf16-gemm %dx%d + fp32-alu + d2d "
            "| %d streams x %d replays/graph",
            device,
            torch.cuda.get_device_name(device),
            replays,
            n,
            n,
            n_streams,
            replays,
        )

        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        last_log = time.time()
        local_iters = 0
        sampler.reset()
        sampler.phase = "compute"
        while time.time() < deadline and not stop.is_set():
            # Wall-clock timing around a device-wide sync: CUDA events
            # recorded on the default stream do NOT track work enqueued on
            # the per-stream graphs (torch streams are non-blocking), so
            # event timing here would measure ~0ms and, worse, the loop
            # would outrun the GPU and checksums would read in-flight data.
            torch.cuda.synchronize(device)
            t_start = time.perf_counter()
            for s, g in graphs:
                with torch.cuda.stream(s):
                    g.replay()
            torch.cuda.synchronize(device)
            ms = max((time.perf_counter() - t_start) * 1000.0, 1e-3)
            tflops = (flops_per_graph * n_streams) / (ms / 1000.0) / 1e12
            bw = (copy_bytes_per_graph * n_streams) / (ms / 1000.0) / 1e9
            result.last_tflops = tflops
            result.best_tflops = max(result.best_tflops, tflops)
            result.sum_tflops += tflops
            result.last_bandwidth = bw
            result.best_bandwidth = max(result.best_bandwidth, bw)
            result.sum_bandwidth += bw
            local_iters += 1
            result.iterations = local_iters

            if cfg.error_check and (local_iters & 15) == 0:
                for i, (a, b, c, x, y, z, m1, m2) in enumerate(streams_bufs):
                    cs = c.float().sum().item()
                    zs = z.sum().item()
                    if cs != ref_sums[i] or zs != ref_z[i]:
                        result.errors += 1
                        LOGGER.error(
                            "[GPU %d] computation mismatch on stream %d!",
                            device,
                            i,
                        )
                        ref_sums[i] = cs
                        ref_z[i] = zs

            now = time.time()
            if now - last_log >= cfg.log_interval:
                remaining = max(0.0, deadline - now)
                LOGGER.info(
                    "[GPU %d] iter=%d | compute=%.1f TFLOP/s | mem=%.1f GB/s | "
                    "errors=%d | remaining=%s",
                    device,
                    result.iterations,
                    tflops,
                    bw,
                    result.errors,
                    format_duration(remaining),
                )
                last_log = now
    except Exception as exc:  # pragma: no cover - hardware dependent
        result.failed = True
        result.message = str(exc)
        LOGGER.exception("[GPU %d] worker crashed: %s", device, exc)


def _apply_power_summary(result: GpuResult, sampler: PowerSampler) -> None:
    """Copy the sampler's sustained numbers into the result record."""
    summary = sampler.summary()
    if not summary:
        return
    result.power_mean_w = summary["power_mean_w"]
    result.power_compute_w = summary["power_compute_w"]
    result.power_max_w = summary["power_max_w"]
    result.power_limit_w = summary["power_limit_w"]
    result.clock_sm_mhz = summary["clock_sm_mhz"]
    result.temp_max_c = summary["temp_max_c"]
    result.util_pct = summary["util_pct"]
    try:
        result.hbm_peak_gbs = hbm_peak_gbs(torch.cuda.get_device_name(result.device))
    except Exception:  # pragma: no cover - context may be broken after a crash
        result.hbm_peak_gbs = 0.0


def _run_on_device(
    device: int,
    cfg: StressConfig,
    deadline: float,
    result: GpuResult,
    stop: threading.Event,
):
    """Continuously stress a single GPU until ``deadline`` or ``stop``."""
    sampler = PowerSampler(device)
    sampler.start()
    try:
        _stress_device(device, cfg, deadline, result, stop, sampler)
    finally:
        sampler.stop()
        sampler.join(timeout=2)
        _apply_power_summary(result, sampler)


def _stress_device(
    device: int,
    cfg: StressConfig,
    deadline: float,
    result: GpuResult,
    stop: threading.Event,
    sampler: PowerSampler,
):
    try:
        result.name = torch.cuda.get_device_name(device)
    except Exception:  # noqa: BLE001 - cosmetic only
        result.name = ""
    backend = resolve_backend(cfg.compute_backend)
    if backend == "torch":
        _run_torch_burn(device, cfg, deadline, result, stop, sampler)
        return
    if backend == "fp8":
        _run_fp8_burn(device, cfg, deadline, result, stop, sampler)
        return
    if not _TRITON_AVAILABLE:
        result.failed = True
        result.message = "triton backend selected but triton is not importable"
        LOGGER.error("[GPU %d] %s", device, result.message)
        return
    result.backend = backend
    try:
        torch.cuda.set_device(device)
        dev = torch.device(f"cuda:{device}")
        dtype = torch.float32

        free_bytes, total_bytes = torch.cuda.mem_get_info(device)
        budget = int(free_bytes * cfg.mem_fraction)
        # Split the budget between the memory phase (src, dst) and the compute
        # phase, so both can run without exhausting device memory.
        mem_budget = budget // 2
        compute_budget = budget - mem_budget

        # Memory phase buffers: 2 buffers, 4 bytes per fp32 element.
        mem_n = max(1, mem_budget // (2 * 4))
        mem_n = max(cfg.block_size, (mem_n // cfg.block_size) * cfg.block_size)
        src = torch.randn(mem_n, device=dev, dtype=dtype)
        dst = torch.empty_like(src)
        mem_grid = (triton.cdiv(mem_n, cfg.block_size),)
        buffer_bytes = mem_n * 4

        # Compute phase buffers depend on the backend.  When error checking is
        # enabled we also need a `reference` buffer (same size as `out`), so we
        # reserve room for it to avoid OOM on the first iteration's clone().
        if backend == "helion":
            # matmul buffers a (n,n), b (n,n), c (n,n): 3 buffers, plus an
            # optional reference clone of c.
            n_compute_bufs = 3 + (1 if cfg.error_check else 0)
            n = int((compute_budget / (n_compute_bufs * 4)) ** 0.5)
            n = max(256, (n // 256) * 256)
            mat_a = torch.randn(n, n, device=dev, dtype=dtype)
            mat_b = torch.randn(n, n, device=dev, dtype=dtype)
            mat_c = torch.empty(n, n, device=dev, dtype=dtype)
            matmul_flops = 2.0 * n * n * n
            compute_desc = f"matmul {n}x{n}"
        else:
            # FMA-loop buffers x, out: 2 buffers, plus an optional reference
            # clone of out.
            n_compute_bufs = 2 + (1 if cfg.error_check else 0)
            comp_n = max(1, compute_budget // (n_compute_bufs * 4))
            comp_n = max(cfg.block_size, (comp_n // cfg.block_size) * cfg.block_size)
            x = torch.randn(comp_n, device=dev, dtype=dtype)
            out = torch.empty_like(x)
            comp_grid = (triton.cdiv(comp_n, cfg.block_size),)
            fma_flops = comp_n * cfg.compute_iters * _FLOPS_PER_ELEM_PER_ITER
            compute_desc = f"fma {comp_n // 1_000_000}M elements x{cfg.compute_iters}"

        LOGGER.info(
            "[GPU %d] %s | backend=%s | compute=%s | mem buffers: %d M elements",
            device,
            torch.cuda.get_device_name(device),
            backend,
            compute_desc,
            mem_n // 1_000_000,
        )

        reference = None
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        last_log = time.time()
        sampler.reset()

        while time.time() < deadline and not stop.is_set():
            # --- compute phase -------------------------------------------- #
            sampler.phase = "compute"
            start_event.record()
            if backend == "helion":
                out = _helion_matmul(mat_a, mat_b, mat_c)
                flops = matmul_flops
            else:
                _compute_kernel[comp_grid](
                    x, out, comp_n, cfg.compute_iters, BLOCK_SIZE=cfg.block_size
                )
                flops = fma_flops
            end_event.record()
            end_event.synchronize()
            compute_ms = start_event.elapsed_time(end_event)
            tflops = flops / (compute_ms / 1000.0) / 1e12
            result.last_tflops = tflops
            result.best_tflops = max(result.best_tflops, tflops)
            result.sum_tflops += tflops

            # --- self check ----------------------------------------------- #
            # Compare against the reference using a checksum to avoid allocating
            # large temporary tensors (out != reference would double VRAM use).
            if cfg.error_check:
                if reference is None:
                    reference = out.clone()
                else:
                    ref_sum = reference.sum().item()
                    out_sum = out.sum().item()
                    if ref_sum != out_sum:
                        # Slow path: count mismatches in chunks to bound memory.
                        mism = 0
                        chunk = max(1, comp_n // 8) if backend != "helion" else 0
                        if backend == "helion":
                            mism = int(~torch.isclose(out, reference)).sum().item()
                        else:
                            for i in range(0, comp_n, chunk):
                                sl = slice(i, min(i + chunk, comp_n))
                                mism += int((out[sl] != reference[sl]).sum().item())
                        result.errors += mism
                        LOGGER.error(
                            "[GPU %d] computation mismatch: %d differing elements!",
                            device,
                            mism,
                        )

            # --- memory phase --------------------------------------------- #
            sampler.phase = "mem"
            start_event.record()
            _copy_kernel[mem_grid](src, dst, mem_n, BLOCK_SIZE=cfg.block_size)
            end_event.record()
            end_event.synchronize()
            copy_ms = start_event.elapsed_time(end_event)
            # read + write = 2x the buffer.
            bandwidth = (2 * buffer_bytes) / (copy_ms / 1000.0) / 1e9
            result.last_bandwidth = bandwidth
            result.best_bandwidth = max(result.best_bandwidth, bandwidth)
            result.sum_bandwidth += bandwidth

            result.iterations += 1

            now = time.time()
            if now - last_log >= cfg.log_interval:
                remaining = max(0.0, deadline - now)
                LOGGER.info(
                    "[GPU %d] iter=%d | compute=%.1f TFLOP/s | mem=%.1f GB/s | "
                    "errors=%d | remaining=%s",
                    device,
                    result.iterations,
                    tflops,
                    bandwidth,
                    result.errors,
                    format_duration(remaining),
                )
                last_log = now
    except Exception as exc:  # pragma: no cover - hardware dependent
        result.failed = True
        result.message = str(exc)
        LOGGER.exception("[GPU %d] worker crashed: %s", device, exc)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def _monitor(deadline: float, interval: float, stop: threading.Event):
    """Periodically log an SMI snapshot of the whole system."""
    while not stop.is_set() and time.time() < deadline:
        snapshot = sample_smi()
        if snapshot:
            LOGGER.info("system stress snapshot:\n%s", snapshot)
        stop.wait(interval)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="GPU stress test for NVIDIA and AMD GPUs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-t",
        "--duration",
        type=parse_duration,
        default="120",
        help="How long to run. Accepts seconds or units, e.g. 90, 30m, 2h, 3d, 1h30m.",
    )
    parser.add_argument(
        "-m",
        "--mem-fraction",
        type=float,
        default=0.8,
        help="Fraction of free GPU memory to allocate for the buffers (0-0.95).",
    )
    parser.add_argument(
        "--compute-backend",
        type=str,
        default="auto",
        choices=["auto", "torch", "fp8", "triton", "helion"],
        help="Compute kernel backend. 'fp8' burns fp8+bf16 tensor cores via "
        "_scaled_mm with an autotuned GEMM size (max TFLOP/s on "
        "Hopper/Blackwell); 'auto' picks fp8 when available, else 'torch'. "
        "'torch' is the bf16 CUDA-graph burn; 'triton'/'helion' need a "
        "working Triton/gcc toolchain.",
    )
    parser.add_argument(
        "--compute-iters",
        type=int,
        default=2048,
        help="Inner FMA-loop iterations per compute kernel launch (higher = more compute bound). Triton backend only.",
    )
    parser.add_argument(
        "--block-size",
        type=int,
        default=1024,
        help="Triton block size (elements per program). Triton backend only.",
    )
    parser.add_argument(
        "--burn-streams",
        type=int,
        default=4,
        help="Parallel CUDA streams per GPU for the torch burn backend.",
    )
    parser.add_argument(
        "--burn-dim",
        type=int,
        default=0,
        help="Force the GEMM matrix dim instead of autotuning it (0 = autotune "
             "for the fp8 backend, 8192 for the torch backend). A large dim "
             "draws more power; a small one can be marginally faster at boost "
             "clocks but leaves 2-3%% of the power budget unused.",
    )
    parser.add_argument(
        "--burn-replays",
        type=int,
        default=20,
        help="GEMM+ALU+copy replays captured per CUDA graph (torch backend).",
    )
    parser.add_argument(
        "--gemm-mix",
        type=parse_gemm_mix,
        default="5:1",
        help="fp8:bf16 GEMM ratio per replay unit for the fp8 backend. '5:1' "
        "drives both tensor-core paths; '6:0' is pure fp8 and reaches ~15%% "
        "more TFLOP/s at the same power.",
    )
    parser.add_argument(
        "--devices",
        type=str,
        default="all",
        help="Comma separated GPU indices to stress, or 'all'.",
    )
    parser.add_argument(
        "--log-interval",
        type=float,
        default=10.0,
        help="Seconds between per-GPU progress log lines.",
    )
    parser.add_argument(
        "--monitor-interval",
        type=float,
        default=30.0,
        help="Seconds between full-system SMI snapshots.",
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default="stress-logs",
        help="Directory where the run log file is written.",
    )
    parser.add_argument(
        "--no-error-check",
        action="store_true",
        help="Disable the compute self-check (slightly faster, no fault detection).",
    )
    parser.add_argument(
        "--single-process",
        action="store_true",
        help="Stress every GPU from threads of one process instead of one "
        "process per GPU. Only useful for debugging: CUDA graph capture is "
        "not safe across threads, so multi-GPU runs must use one process "
        "per GPU (the default).",
    )
    parser.add_argument(
        "--min-power-fraction",
        type=float,
        default=0.0,
        help="Fail the run when a GPU's sustained compute-phase power stays "
        "below this fraction of its power limit (e.g. 0.95 for a burn-in "
        "acceptance run). 0 disables the check.",
    )
    parser.add_argument(
        "--json",
        metavar="PATH",
        help="Write the run summary as JSON (per-GPU power, throughput, "
             "bandwidth, errors). report.py --stress embeds it in the PDF.",
    )
    parser.add_argument(
        "--worker-json",
        type=str,
        default="",
        help="Internal: write this worker's result as JSON to the given path.",
    )
    parser.add_argument(
        "--no-log-file",
        action="store_true",
        help="Log to stdout only (used by the per-GPU worker processes).",
    )
    parser.add_argument(
        "--no-monitor",
        action="store_true",
        help="Disable the periodic full-system SMI snapshot.",
    )
    return parser


def _configure_logging(log_dir: str, no_log_file: bool = False) -> str:
    LOGGER.setLevel(logging.INFO)
    LOGGER.handlers.clear()
    # Worker processes are piped through the parent, which adds the timestamp:
    # they emit "LEVEL message" only, so the parent can re-emit at that level.
    fmt = logging.Formatter(
        "%(levelname)-7s %(message)s" if no_log_file else "%(asctime)s %(levelname)-7s %(message)s"
    )

    stream = logging.StreamHandler(sys.stdout)
    stream.setFormatter(fmt)
    LOGGER.addHandler(stream)

    if no_log_file:
        return ""
    os.makedirs(log_dir, exist_ok=True)
    ts = _dt.datetime.now(_dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    log_path = os.path.join(log_dir, f"gpu-stress-{ts}.log")
    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(fmt)
    LOGGER.addHandler(file_handler)
    return log_path


def _pump_worker_output(proc: subprocess.Popen, device: int) -> None:
    """Re-emit a worker's stdout through the parent logger.

    The worker processes log to stdout only; piping them through the parent
    keeps a single, complete run log instead of one file per GPU.
    """
    if proc.stdout is None:  # pragma: no cover - defensive
        return
    levels = {"INFO": logging.INFO, "WARNING": logging.WARNING, "ERROR": logging.ERROR}
    for raw in proc.stdout:
        line = raw.rstrip()
        if not line:
            continue
        level_name, _, message = line.partition(" ")
        LOGGER.log(levels.get(level_name, logging.INFO), "%s", message.strip() or line)


def _format_gpu_summary(r: GpuResult) -> str:
    """One result block per GPU: throughput, then the power point it ran at."""
    lines = [
        f"[GPU {r.device}] backend={r.backend} | iterations={r.iterations} | "
        f"compute={r.mean_tflops():.0f} TFLOP/s sustained "
        f"(best {r.best_tflops:.0f}) | "
        f"mem={r.mean_bandwidth():.0f} GB/s sustained "
        f"(best {r.best_bandwidth:.0f}) | errors={r.errors}"
    ]
    if r.power_limit_w > 0:
        lines.append(
            f"    power: sustained {r.power_compute_w:.0f} W of "
            f"{r.power_limit_w:.0f} W ({100.0 * r.power_fraction():.1f}%) | "
            f"peak {r.power_max_w:.0f} W | {r.temp_max_c:.0f} C | "
            f"SM {r.clock_sm_mhz:.0f} MHz | util {r.util_pct:.0f}%"
        )
    if r.hbm_peak_gbs > 0:
        lines.append(
            f"    bandwidth: {100.0 * r.bandwidth_fraction():.1f}% of the "
            f"{r.hbm_peak_gbs:.0f} GB/s datasheet HBM peak"
        )
    if r.failed:
        lines.append(f"    FAILED: {r.message}")
    return "\n".join(lines)


def _write_json(
    path: str,
    results: dict[int, GpuResult],
    devices: list[int],
    elapsed: float,
    cfg: "StressConfig",
    args,
) -> None:
    """Write the run summary as JSON — the machine-readable evidence that the
    PDF report embeds (``report.py --stress``)."""
    import json
    import socket

    gpus = []
    for d in devices:
        r = results[d]
        gpus.append({
            "device": d,
            "name": r.name,
            "backend": r.backend,
            "iterations": r.iterations,
            "tflops_sustained": round(r.mean_tflops(), 1),
            "tflops_best": round(r.best_tflops, 1),
            "bandwidth_sustained_gbs": round(r.mean_bandwidth(), 1),
            "bandwidth_best_gbs": round(r.best_bandwidth, 1),
            "hbm_peak_gbs": round(r.hbm_peak_gbs, 1),
            "bandwidth_fraction": round(r.bandwidth_fraction(), 4),
            "power_mean_w": round(r.power_compute_w, 1),
            "power_max_w": round(r.power_max_w, 1),
            "power_limit_w": round(r.power_limit_w, 1),
            "power_fraction": round(r.power_fraction(), 4),
            "clock_sm_mhz": round(r.clock_sm_mhz, 1),
            "temp_max_c": round(r.temp_max_c, 1),
            "util_pct": round(r.util_pct, 1),
            "errors": r.errors,
            "failed": r.failed,
            "message": r.message,
        })

    powered = [r for r in results.values() if r.power_limit_w > 0]
    total_w = sum(r.power_compute_w for r in powered)
    total_limit = sum(r.power_limit_w for r in powered)
    payload = {
        "timestamp": _dt.datetime.now().astimezone().isoformat(timespec="seconds"),
        "hostname": socket.gethostname(),
        "duration_s": round(elapsed, 1),
        "config": {
            "backend": cfg.compute_backend,
            "gemm_mix": cfg.gemm_mix,
            "burn_dim": cfg.burn_dim,
            "burn_streams": cfg.burn_streams,
            "burn_replays": cfg.burn_replays,
            "mem_fraction": cfg.mem_fraction,
            "min_power_fraction": args.min_power_fraction,
        },
        "gpus": gpus,
        "system": {
            "gpu_count": len(powered),
            "power_mean_w": round(total_w, 1),
            "power_limit_w": round(total_limit, 1),
            "power_fraction": round(total_w / total_limit, 4) if total_limit else 0.0,
            "errors": sum(r.errors for r in results.values()),
            "passed": not any(r.failed or r.errors for r in results.values()),
        },
    }
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    LOGGER.info("json summary        : %s", path)


def _report(
    results: dict[int, GpuResult],
    devices: list[int],
    elapsed: float,
    log_path: str,
    min_power_fraction: float,
) -> int:
    """Print the final summary, apply the power gate and return the exit code."""
    if min_power_fraction > 0:
        for d in devices:
            r = results[d]
            if r.power_limit_w <= 0:
                LOGGER.warning(
                    "[GPU %d] power gate requested but NVML reported no power "
                    "limit; cannot verify the power point",
                    d,
                )
                continue
            if r.power_fraction() < min_power_fraction:
                r.failed = True
                r.message = (
                    f"sustained power {r.power_compute_w:.0f} W is "
                    f"{100.0 * r.power_fraction():.1f}% of the "
                    f"{r.power_limit_w:.0f} W limit, below the required "
                    f"{100.0 * min_power_fraction:.0f}%"
                )

    total_errors = sum(r.errors for r in results.values())
    any_failed = any(r.failed for r in results.values())

    LOGGER.info("=" * 72)
    LOGGER.info("Stress test finished after %s", format_duration(elapsed))
    for d in devices:
        LOGGER.info("%s", _format_gpu_summary(results[d]))

    powered = [r for r in results.values() if r.power_limit_w > 0]
    if powered:
        total_w = sum(r.power_compute_w for r in powered)
        total_limit = sum(r.power_limit_w for r in powered)
        LOGGER.info(
            "system: sustained GPU power %.0f W of %.0f W (%.1f%%) across %d GPUs",
            total_w,
            total_limit,
            100.0 * total_w / total_limit if total_limit else 0.0,
            len(powered),
        )
    LOGGER.info("total compute errors: %d", total_errors)
    if log_path:
        LOGGER.info("log file            : %s", log_path)
    LOGGER.info("=" * 72)

    if any_failed or total_errors > 0:
        LOGGER.error("GPU stress test detected problems.")
        return 1
    LOGGER.info("All GPUs passed the stress test.")
    return 0


def _run_workers(args, devices: list[int], log_path: str) -> int:
    """Run one child process per GPU and aggregate their results.

    CUDA graph capture is not safe across threads of one process: with eight
    GPUs in one process, seven of eight workers died in
    ``cuStreamBeginCapture`` with ``CUDA_ERROR_ILLEGAL_STATE``.  One process
    per GPU gives every card its own CUDA context, isolates a crashed worker
    from the rest and is what a multi-GPU burn-in needs anyway.
    """
    import json
    import tempfile

    tmpdir = tempfile.mkdtemp(prefix="gpu-stress-")
    script = os.path.abspath(__file__)
    procs: list[tuple[int, str, subprocess.Popen]] = []
    for d in devices:
        out = os.path.join(tmpdir, f"gpu{d}.json")
        cmd = [
            sys.executable,
            script,
            "--duration",
            f"{args.duration:.0f}",
            "--devices",
            str(d),
            "--worker-json",
            out,
            "--no-log-file",
            "--no-monitor",
            "--mem-fraction",
            str(args.mem_fraction),
            "--compute-backend",
            args.compute_backend,
            "--compute-iters",
            str(args.compute_iters),
            "--block-size",
            str(args.block_size),
            "--burn-streams",
            str(args.burn_streams),
            "--burn-dim",
            str(args.burn_dim),
            "--burn-replays",
            str(args.burn_replays),
            "--gemm-mix",
            args.gemm_mix,
            "--log-interval",
            str(args.log_interval),
        ]
        if args.no_error_check:
            cmd.append("--no-error-check")
        procs.append(
            (
                d,
                out,
                subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                ),
            )
        )

    LOGGER.info(
        "started %d worker processes (one per GPU): %s",
        len(procs),
        ", ".join(f"GPU {d} pid {p.pid}" for d, _, p in procs),
    )

    pumps = [
        threading.Thread(target=_pump_worker_output, args=(proc, d), daemon=True)
        for d, _, proc in procs
    ]
    for pump in pumps:
        pump.start()

    start = time.time()
    stop = threading.Event()
    monitor = None
    if not args.no_monitor:
        monitor = threading.Thread(
            target=_monitor,
            args=(start + args.duration, args.monitor_interval, stop),
            daemon=True,
        )
        monitor.start()

    try:
        for d, _, proc in procs:
            rc = proc.wait()
            if rc != 0:
                LOGGER.error("worker for GPU %d exited with status %d", d, rc)
    except KeyboardInterrupt:  # pragma: no cover - interactive
        LOGGER.warning("Interrupted by user, stopping workers ...")
        for _, _, proc in procs:
            proc.terminate()
        for _, _, proc in procs:
            proc.wait()
    finally:
        stop.set()
        if monitor is not None:
            monitor.join(timeout=2)
        for pump in pumps:
            pump.join(timeout=5)

    results: dict[int, GpuResult] = {}
    for d, out, _ in procs:
        try:
            with open(out, encoding="utf-8") as fh:
                results[d] = GpuResult(**json.load(fh))
        except Exception as exc:  # pragma: no cover - worker crash
            results[d] = GpuResult(
                device=d, failed=True, message=f"no result from worker: {exc}"
            )
    shutil.rmtree(tmpdir, ignore_errors=True)

    elapsed = time.time() - start
    code = _report(results, devices, elapsed, log_path, args.min_power_fraction)
    if args.json:
        cfg = StressConfig(
            duration=args.duration,
            mem_fraction=args.mem_fraction,
            compute_iters=args.compute_iters,
            block_size=args.block_size,
            error_check=not args.no_error_check,
            log_interval=args.log_interval,
            compute_backend=args.compute_backend,
            devices=devices,
            burn_streams=args.burn_streams,
            burn_dim=args.burn_dim,
            burn_replays=args.burn_replays,
            gemm_mix=args.gemm_mix,
        )
        _write_json(args.json, results, devices, elapsed, cfg, args)
    return code


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    log_path = _configure_logging(args.log_dir, args.no_log_file)

    if not torch.cuda.is_available():
        LOGGER.error(
            "No GPU visible to PyTorch. On AMD make sure this is a ROCm build; "
            "on NVIDIA make sure this is a CUDA build and the driver is loaded."
        )
        return 2

    mem_fraction = max(0.05, min(args.mem_fraction, 0.95))

    if args.devices.strip().lower() == "all":
        devices = list(range(torch.cuda.device_count()))
    else:
        devices = [int(d) for d in args.devices.split(",") if d.strip() != ""]

    # Multi-GPU: one process per GPU (see _run_workers for why).
    if len(devices) > 1 and not args.single_process:
        return _run_workers(args, devices, log_path)

    cfg = StressConfig(
        duration=args.duration,
        mem_fraction=mem_fraction,
        compute_iters=args.compute_iters,
        block_size=args.block_size,
        error_check=not args.no_error_check,
        log_interval=args.log_interval,
        compute_backend=args.compute_backend,
        devices=devices,
        burn_streams=args.burn_streams,
        burn_dim=args.burn_dim,
        burn_replays=args.burn_replays,
        gemm_mix=args.gemm_mix,
    )

    effective_backend = resolve_backend(cfg.compute_backend)
    vendor = detect_vendor()
    LOGGER.info("=" * 72)
    LOGGER.info("GPU stress test")
    LOGGER.info("host           : %s", platform.node())
    LOGGER.info("vendor         : %s", vendor)
    LOGGER.info("torch          : %s", torch.__version__)
    LOGGER.info(
        "triton         : %s",
        getattr(triton, "__version__", "unavailable") if _TRITON_AVAILABLE else "unavailable",
    )
    LOGGER.info(
        "helion         : %s",
        getattr(helion, "__version__", "unknown") if _HELION_AVAILABLE else "not installed",
    )
    LOGGER.info(
        "cuda/hip       : %s",
        getattr(torch.version, "cuda", None) or getattr(torch.version, "hip", None),
    )
    LOGGER.info("gpus           : %s", devices)
    LOGGER.info("duration       : %s", format_duration(cfg.duration))
    LOGGER.info("mem fraction   : %.2f", cfg.mem_fraction)
    LOGGER.info("compute backend: %s (requested: %s)", effective_backend, cfg.compute_backend)
    LOGGER.info("error checking : %s", cfg.error_check)
    LOGGER.info("power sampling : %s", "NVML" if _NVML_AVAILABLE else "unavailable")
    LOGGER.info("log file       : %s", log_path or "(stdout only)")
    LOGGER.info("=" * 72)

    start = time.time()
    deadline = start + cfg.duration

    results = {d: GpuResult(device=d) for d in devices}
    stop = threading.Event()
    monitor = None
    if not args.no_monitor:
        monitor = threading.Thread(
            target=_monitor, args=(deadline, args.monitor_interval, stop), daemon=True
        )
        monitor.start()

    threads = []
    for d in devices:
        th = threading.Thread(
            target=_run_on_device,
            args=(d, cfg, deadline, results[d], stop),
            daemon=True,
        )
        th.start()
        threads.append(th)

    try:
        for th in threads:
            th.join()
    except KeyboardInterrupt:  # pragma: no cover - interactive
        LOGGER.warning("Interrupted by user, stopping ...")
        stop.set()  # signal workers to stop on their next loop check
        for th in threads:
            th.join()

    stop.set()
    if monitor is not None:
        monitor.join(timeout=2)

    elapsed = time.time() - start

    if args.worker_json:
        # Worker mode: hand the raw result to the parent, which owns the
        # summary (and the power gate) for the whole node.
        import json

        result = results[devices[0]]
        with open(args.worker_json, "w", encoding="utf-8") as fh:
            json.dump(result.__dict__, fh)
        return 1 if (result.failed or result.errors) else 0

    code = _report(results, devices, elapsed, log_path, args.min_power_fraction)
    if args.json:
        _write_json(args.json, results, devices, elapsed, cfg, args)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
