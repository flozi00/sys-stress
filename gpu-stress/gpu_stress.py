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


# --------------------------------------------------------------------------- #
# Per-GPU worker
# --------------------------------------------------------------------------- #
@dataclass
class GpuResult:
    device: int
    iterations: int = 0
    errors: int = 0
    best_tflops: float = 0.0
    last_tflops: float = 0.0
    best_bandwidth: float = 0.0
    last_bandwidth: float = 0.0
    backend: str = "triton"
    failed: bool = False
    message: str = ""


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
    burn_dim: int = 8192
    burn_replays: int = 20


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

    def make_buffers(n):
        fa = torch.randn(n, n, device=dev).to(torch.float8_e4m3fn)
        fb = torch.randn(n, n, device=dev).to(torch.float8_e4m3fn).t().contiguous().t()
        fc = torch.empty(n, n, device=dev, dtype=torch.bfloat16)
        fa_f32 = torch.randn(n, n, device=dev, dtype=torch.float32)
        fb_f32 = torch.randn(n, n, device=dev, dtype=torch.float32)
        sa = torch.randn(n, n, device=dev, dtype=torch.bfloat16)
        sb = torch.randn(n, n, device=dev, dtype=torch.bfloat16)
        sc = torch.empty(n, n, device=dev, dtype=torch.bfloat16)
        return fa, fb, fc, fa_f32, fb_f32, sa, sb, sc

    best_n, best_tflops, best_bufs = -1, -1.0, None
    with torch.no_grad():
        for cand in candidates:
            try:
                bufs = [make_buffers(cand) for _ in range(n_streams)]
                torch.cuda.synchronize(device)
                # Warmup so cuBLAS picks its fastest algorithm before capture.
                for (fa, fb, fc, fa_f32, fb_f32, sa, sb, sc) in bufs:
                    for _ in range(3):
                        torch._scaled_mm(fa, fb, scale_a=s, scale_b=s, out_dtype=torch.bfloat16)
                        torch.mm(sa, sb, out=sc)
                    fc.copy_(fa_f32.to(torch.bfloat16))
                torch.cuda.synchronize(device)
                # Capture one repeat unit: 5 fp8 GEMMs + 1 bf16 GEMM (drives
                # both fp8 and bf16 tensor-core paths, filling wave tail).
                reps = max(2, min(8, cfg.burn_replays))
                graphs = []
                for (fa, fb, fc, fa_f32, fb_f32, sa, sb, sc) in bufs:
                    st = torch.cuda.Stream(device=device)
                    g = torch.cuda.CUDAGraph()
                    with torch.cuda.stream(st), torch.cuda.graph(g):
                        for i in range(reps):
                            for _ in range(5):
                                torch._scaled_mm(fa, fb, scale_a=s, scale_b=s, out_dtype=torch.bfloat16)
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
                # FLOPs per graph: reps*(5 fp8 + 1 bf16) gemms of 2n^3.
                flops = reps * 6 * 2.0 * cand**3 * n_streams
                tflops = flops / (ms / 1000.0) / 1e12
                LOGGER.info(
                    "[GPU %d] autotune fp8 dim=%d: %.0f TFLOP/s (budget max n=%d)",
                    device, cand, tflops, max_n,
                )
                if tflops > best_tflops:
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
            _run_torch_burn(device, cfg, deadline, result, stop)
            return
        n_streams = max(1, cfg.burn_streams)
        reps = max(2, min(8, cfg.burn_replays))

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
                        for _ in range(5):
                            torch._scaled_mm(fa, fb, scale_a=s, scale_b=s, out_dtype=torch.bfloat16)
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

        flops_per_round = reps * 6 * 2.0 * n * n * n * n_streams
        # Approx HBM+L2 bytes per round: each 5xfp8+1xbf16 unit reads fa/fb
        # (fp8) and sa/sb (bf16) once per gemm while fc/sc/sa/sb rewrite in
        # place => ~ (5*2 + 6*2/round + conv 12/4th iter) * n^2.
        bytes_per_round = reps * (10.0 + 12.0 + 3.0) * n * n * n_streams
        LOGGER.info(
            "[GPU %d] %s | backend=fp8 | %dx (5xfp8+1xbf16) gemm %dx%d + "
            "fp32->bf16 conversions | %d streams",
            device, torch.cuda.get_device_name(device), reps, n, n, n_streams,
        )

        last_log = time.time()
        local_iters = 0
        with torch.no_grad():
            while time.time() < deadline and not stop.is_set():
                # Device-wide sync timing (CUDA events don't track the
                # non-blocking per-stream graphs). Phases are timed
                # separately so each metric reflects its own subsystem,
                # not an amortized blend.
                torch.cuda.synchronize(device)
                t_start = time.perf_counter()
                for st, g in graphs:
                    with torch.cuda.stream(st):
                        g.replay()
                torch.cuda.synchronize(device)
                t_gemm = time.perf_counter()
                # Memory phase: alternating copies saturate read+write HBM.
                if ms_src is not None:
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
                result.last_bandwidth = bw
                result.best_bandwidth = max(result.best_bandwidth, bw)
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
        n = max(1024, cfg.burn_dim)
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
                    n = max(1024, cfg.burn_dim)
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
            result.last_bandwidth = bw
            result.best_bandwidth = max(result.best_bandwidth, bw)
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


def _run_on_device(
    device: int,
    cfg: StressConfig,
    deadline: float,
    result: GpuResult,
    stop: threading.Event,
):
    """Continuously stress a single GPU until ``deadline`` or ``stop``."""
    backend = resolve_backend(cfg.compute_backend)
    if backend == "torch":
        _run_torch_burn(device, cfg, deadline, result, stop)
        return
    if backend == "fp8":
        _run_fp8_burn(device, cfg, deadline, result, stop)
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

        while time.time() < deadline and not stop.is_set():
            # --- compute phase -------------------------------------------- #
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
            start_event.record()
            _copy_kernel[mem_grid](src, dst, mem_n, BLOCK_SIZE=cfg.block_size)
            end_event.record()
            end_event.synchronize()
            copy_ms = start_event.elapsed_time(end_event)
            # read + write = 2x the buffer.
            bandwidth = (2 * buffer_bytes) / (copy_ms / 1000.0) / 1e9
            result.last_bandwidth = bandwidth
            result.best_bandwidth = max(result.best_bandwidth, bandwidth)

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
        default=8192,
        help="Matrix dim for the torch burn backend (auto-shrunk to fit VRAM).",
    )
    parser.add_argument(
        "--burn-replays",
        type=int,
        default=20,
        help="GEMM+ALU+copy replays captured per CUDA graph (torch backend).",
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
    return parser


def _configure_logging(log_dir: str) -> str:
    os.makedirs(log_dir, exist_ok=True)
    ts = _dt.datetime.now(_dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    log_path = os.path.join(log_dir, f"gpu-stress-{ts}.log")

    LOGGER.setLevel(logging.INFO)
    LOGGER.handlers.clear()
    fmt = logging.Formatter("%(asctime)s %(levelname)-7s %(message)s")

    stream = logging.StreamHandler(sys.stdout)
    stream.setFormatter(fmt)
    LOGGER.addHandler(stream)

    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(fmt)
    LOGGER.addHandler(file_handler)
    return log_path


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    log_path = _configure_logging(args.log_dir)

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
    LOGGER.info("log file       : %s", log_path)
    LOGGER.info("=" * 72)

    start = time.time()
    deadline = start + cfg.duration

    results = {d: GpuResult(device=d) for d in devices}
    stop = threading.Event()
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
    monitor.join(timeout=2)

    elapsed = time.time() - start
    total_errors = sum(r.errors for r in results.values())
    any_failed = any(r.failed for r in results.values())

    LOGGER.info("=" * 72)
    LOGGER.info("Stress test finished after %s", format_duration(elapsed))
    for d in devices:
        r = results[d]
        LOGGER.info(
            "[GPU %d] backend=%s | iterations=%d | best compute=%.1f TFLOP/s | "
            "best mem=%.1f GB/s | errors=%d%s",
            d,
            r.backend,
            r.iterations,
            r.best_tflops,
            r.best_bandwidth,
            r.errors,
            f" | FAILED: {r.message}" if r.failed else "",
        )
    LOGGER.info("total compute errors: %d", total_errors)
    LOGGER.info("log file            : %s", log_path)
    LOGGER.info("=" * 72)

    if any_failed or total_errors > 0:
        LOGGER.error("GPU stress test detected problems.")
        return 1
    LOGGER.info("All GPUs passed the stress test.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
