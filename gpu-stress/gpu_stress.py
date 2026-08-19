#!/usr/bin/env python3
"""Triton based GPU stress test.

A single, vendor neutral stress test that hammers both the memory subsystem
(memory bandwidth) and the arithmetic units (computation cores) of every
visible GPU.  Because the heavy lifting is done by `Triton <https://github.com/
triton-lang/triton>`_ kernels executed through PyTorch, the exact same test runs
on NVIDIA (CUDA) and AMD (ROCm) GPUs.

The goal is to keep the GPUs at (or close to) 100% load for a configurable
amount of time -- from a couple of minutes up to several days -- so that the
cooling solution and the long term stability of the cards can be validated.

Every iteration performs:

* a **compute** phase: a long fused-multiply-add loop kept in registers to
  saturate the FP32 ALUs and report an estimated TFLOP/s figure, and
* a **memory** phase: a streaming copy over large buffers to saturate the
  device memory bus and report the achieved bandwidth in GB/s.

The compute phase is also self checking: the result of the first iteration is
kept as a reference and every subsequent iteration is compared against it.  A
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
    import triton
    import triton.language as tl
except Exception as exc:  # pragma: no cover - import guard
    sys.stderr.write(
        "Failed to import torch/triton. Install a GPU enabled PyTorch + Triton "
        "build (CUDA for NVIDIA, ROCm for AMD).\n"
        f"Original error: {exc}\n"
    )
    raise


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
# Triton kernels
# --------------------------------------------------------------------------- #
@triton.jit
def _compute_kernel(x_ptr, out_ptr, n_elements, n_iters, BLOCK_SIZE: tl.constexpr):
    """Register bound fused-multiply-add loop to saturate the FP32 ALUs.

    Each inner iteration issues two FMA-like operations (4 FLOPs total) that
    depend on the previous result, which keeps the pipeline busy without
    touching memory.
    """
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)

    a = x
    b = x + 1.0
    for _ in range(n_iters):
        a = a * b + b
        b = b * a + a
    tl.store(out_ptr + offsets, a + b, mask=mask)


@triton.jit
def _copy_kernel(src_ptr, dst_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    """Streaming copy used to measure/stress memory bandwidth (read + write)."""
    pid = tl.program_id(axis=0)
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
    def _helion_matmul(x: "torch.Tensor", y: "torch.Tensor") -> "torch.Tensor":
        """Autotuned matmul used to drive the matrix cores at maximum power.

        Helion searches hundreds of Triton implementations on the first call and
        keeps the fastest one for the running hardware, so the same source
        reaches peak throughput on both NVIDIA and AMD GPUs.
        """
        m, k = x.size()
        k2, n = y.size()
        out = torch.empty([m, n], dtype=torch.float32, device=x.device)
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


def resolve_backend(requested: str) -> str:
    """Resolve the effective compute backend from the requested value."""
    requested = requested.strip().lower()
    if requested == "helion":
        if not _HELION_AVAILABLE:
            LOGGER.warning(
                "Helion backend requested but helion is not importable; "
                "falling back to the Triton compute kernel."
            )
            return "triton"
        return "helion"
    if requested == "auto":
        return "helion" if _HELION_AVAILABLE else "triton"
    return "triton"


def _run_on_device(
    device: int,
    cfg: StressConfig,
    deadline: float,
    result: GpuResult,
    stop: threading.Event,
):
    """Continuously stress a single GPU until ``deadline`` or ``stop``."""
    backend = resolve_backend(cfg.compute_backend)
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

        # Compute phase buffers depend on the backend.
        if backend == "helion":
            # matmul buffers a (n,n), b (n,n), c (n,n): 3 buffers.
            n = int((compute_budget / (3 * 4)) ** 0.5)
            n = max(256, (n // 256) * 256)
            mat_a = torch.randn(n, n, device=dev, dtype=dtype)
            mat_b = torch.randn(n, n, device=dev, dtype=dtype)
            matmul_flops = 2.0 * n * n * n
            compute_desc = f"matmul {n}x{n}"
        else:
            # FMA-loop buffers x, out: 2 buffers.
            comp_n = max(1, compute_budget // (2 * 4))
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
                out = _helion_matmul(mat_a, mat_b)
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
            if cfg.error_check:
                if reference is None:
                    reference = out.clone()
                elif not torch.equal(out, reference):
                    mism = int((out != reference).sum().item())
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
        description="Triton based GPU stress test for NVIDIA and AMD GPUs.",
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
        choices=["auto", "triton", "helion"],
        help="Compute kernel backend. 'helion' autotunes a matmul that pushes the "
        "matrix cores to maximum power draw; 'auto' uses helion when available, "
        "otherwise the Triton FMA loop.",
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
        help="Triton block size (elements per program).",
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
    )

    effective_backend = resolve_backend(cfg.compute_backend)
    vendor = detect_vendor()
    LOGGER.info("=" * 72)
    LOGGER.info("Triton GPU stress test")
    LOGGER.info("host           : %s", platform.node())
    LOGGER.info("vendor         : %s", vendor)
    LOGGER.info("torch          : %s", torch.__version__)
    LOGGER.info("triton         : %s", getattr(triton, "__version__", "unknown"))
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
