#!/usr/bin/env python3
"""Hardware benchmark suite: FLOPS, memory bandwidth, interconnect.

Measures, per GPU:

* **HBM/GDDR bandwidth** -- read, write, copy and triad (2R+1W) in GB/s.
* **Compute peak** -- GEMM throughput in TFLOP/s for fp64, fp32, tf32, bf16,
  fp16, fp8 (e4m3) and fp4 (nvfp4), i.e. CUDA cores and every tensor-core
  precision the card supports.
* **Host <-> device bandwidth** -- NVLink-C2C on Grace-Blackwell systems,
  PCIe otherwise (pinned memory, both directions).
* **Device <-> device bandwidth** -- P2P copies between every GPU pair.

and on the CPU: STREAM triad bandwidth plus BLAS GFLOPS.

Every measurement is repeated, the fastest run is reported (least noise from
other tenants) and the median is kept alongside.  Power, clocks and
temperature are sampled in the background so the report shows whether a run
actually reached the card's power limit.

This file contains no hostnames, IPs or credentials.  The remote runner takes
the target from the ``SYS_STRESS_HOST`` environment variable; see
``run_remote.sh``.

Usage::

    python3 bench.py info
    python3 bench.py hbm --gpu 0
    python3 bench.py compute --dtype bf16 fp8
    python3 bench.py interconnect
    python3 bench.py all --json results.json
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import re
import statistics
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable

try:
    import torch
except Exception as exc:  # pragma: no cover - import guard
    sys.stderr.write(
        "Failed to import torch. Install a CUDA enabled PyTorch build, e.g.\n"
        "  uv pip install torch --index-url https://download.pytorch.org/whl/cu130\n"
        f"Original error: {exc}\n"
    )
    raise


# --------------------------------------------------------------------------- #
# Reference (datasheet) peaks
# --------------------------------------------------------------------------- #
# Dense (non-sparse) figures.  Sources:
#   * NVIDIA Blackwell Ultra datasheet, "Individual Blackwell Ultra GPU
#     Specifications", HGX B300 column (270 GB HBM3E, 7.7 TB/s).  The
#     datasheet lists FP8/BF16/TF32 as sparse with "dense is 1/2 sparse".
#   * NVIDIA RTX PRO 6000 Blackwell Max-Q workstation datasheet
#     (3511 AI TOPS FP4 *with sparsity* -> 1755 dense, 1792 GB/s, 300 W).
# Only published numbers are listed; anything missing is reported as an
# absolute measurement without a percentage.
DATASHEET: dict[str, dict[str, float]] = {
    "GB300": {
        "hbm_gbs": 7700.0,
        "fp4_tflops": 14000.0,
        "fp8_tflops": 4500.0,
        "bf16_tflops": 2250.0,
        "fp16_tflops": 2250.0,
        "tf32_tflops": 1100.0,
        "fp32_tflops": 75.0,
        "fp64_tflops": 1.2,
    },
    "RTX PRO 6000": {
        "hbm_gbs": 1792.0,
        "fp4_tflops": 1755.0,  # 3511 AI TOPS with sparsity / 2
    },
}


def datasheet_for(name: str) -> dict[str, float]:
    """Return the datasheet entry whose key is a substring of the GPU name."""
    for key, values in DATASHEET.items():
        if key.lower() in name.lower():
            return values
    return {}


# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #
def human_bytes(n: float) -> str:
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if abs(n) < 1024.0 or unit == "TiB":
            return f"{n:.1f} {unit}"
        n /= 1024.0
    return f"{n:.1f} TiB"


def pct(value: float | None, peak: float | None) -> str:
    if not value or not peak:
        return ""
    return f"{100.0 * value / peak:5.1f}%"


def _timed_loop(fn: Callable[[], Any], min_seconds: float = 3.0,
                repeats: int = 3) -> tuple[float, float, int]:
    """Time ``fn`` over a fixed wall-clock budget.

    A single GEMM at 16384^3 finishes in ~5 ms, far too short for a stable
    number or for power sampling.  The loop is therefore calibrated once, then
    run for ``min_seconds`` split into ``repeats`` blocks; the fastest block
    wins.  Returns ``(best_per_iter, median_per_iter, iters_per_block)`` in
    seconds.
    """
    for _ in range(2):
        fn()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    fn()
    end.record()
    torch.cuda.synchronize()
    single = max(start.elapsed_time(end) / 1000.0, 1e-6)
    iters = max(1, int((min_seconds / repeats) / single))

    target = min_seconds / repeats
    samples: list[float] = []
    for _ in range(repeats):
        start.record()
        for _ in range(iters):
            fn()
        end.record()
        torch.cuda.synchronize()
        elapsed = start.elapsed_time(end) / 1000.0
        samples.append(elapsed / iters)
        # The single-iteration calibration runs at boost clock with a cold L2,
        # so it can be 2-3x faster than the sustained rate.  Without this the
        # block overruns its budget, the phase drags on and the power average
        # picks up idle tail.  Re-derive the iteration count from what the
        # block actually took.
        if elapsed > 1.5 * target:
            iters = max(1, int(iters * target / elapsed))
    return min(samples), statistics.median(samples), iters


# --------------------------------------------------------------------------- #
# Background power / clock sampler
# --------------------------------------------------------------------------- #
try:
    import pynvml  # nvidia-ml-py: fast, non-blocking power/clock queries
    _NVML_AVAILABLE = True
except Exception:  # pragma: no cover - optional dependency
    pynvml = None
    _NVML_AVAILABLE = False


class PowerSampler:
    """Poll power/clock/temperature in a thread and keep a per-GPU series.

    Uses NVML when ``nvidia-ml-py`` is installed: spawning ``nvidia-smi``
    blocks for seconds while the GPU is saturated, which silently drops
    samples exactly when they matter.  Filtering is done by GPU UUID, not by
    index, because inside a child process ``CUDA_VISIBLE_DEVICES`` renumbers
    the CUDA index while nvidia-smi keeps reporting the global one.
    """

    FIELDS = "uuid,power.draw,temperature.gpu,clocks.sm,utilization.gpu"

    def __init__(self, uuid: str | None = None, interval: float = 0.25) -> None:
        self.uuid = uuid
        self.interval = interval
        self.samples: dict[str, list[float]] = {
            "power": [], "temp": [], "clock": [], "util": [], "t": []
        }
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._handle = None
        if _NVML_AVAILABLE and uuid:
            assert pynvml is not None
            try:
                pynvml.nvmlInit()
                self._handle = pynvml.nvmlDeviceGetHandleByUUID(uuid.encode())
            except Exception:
                self._handle = None

    def _sample_nvml(self) -> tuple[float, float, float, float] | None:
        assert pynvml is not None
        try:
            power = pynvml.nvmlDeviceGetPowerUsage(self._handle) / 1000.0
            temp = float(pynvml.nvmlDeviceGetTemperature(
                self._handle, pynvml.NVML_TEMPERATURE_GPU))
            clock = float(pynvml.nvmlDeviceGetClockInfo(
                self._handle, pynvml.NVML_CLOCK_SM))
            util = float(pynvml.nvmlDeviceGetUtilizationRates(self._handle).gpu)
            return power, temp, clock, util
        except Exception:
            return None

    def _sample_smi(self) -> list[tuple[float, float, float, float]]:
        try:
            out = subprocess.run(
                ["nvidia-smi", f"--query-gpu={self.FIELDS}",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=20, check=False,
            ).stdout
        except Exception:
            return []
        rows = []
        for line in out.strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) != 5:
                continue
            if self.uuid and parts[0] != self.uuid:
                continue
            try:
                rows.append(tuple(float(p) for p in parts[1:]))  # type: ignore[arg-type]
            except ValueError:
                continue
        return rows

    def _poll(self) -> None:
        while not self._stop.is_set():
            if self._handle is not None:
                row = self._sample_nvml()
                rows = [row] if row else []
            else:
                rows = self._sample_smi()
            for power, temp, clock, util in rows:
                self.samples["power"].append(power)
                self.samples["temp"].append(temp)
                self.samples["clock"].append(clock)
                self.samples["util"].append(util)
                self.samples["t"].append(time.monotonic())
            self._stop.wait(self.interval)

    def __enter__(self) -> "PowerSampler":
        self._thread = threading.Thread(target=self._poll, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5)

    def mark(self) -> int:
        """Index of the next sample; use with :meth:`since` for per-phase stats."""
        return len(self.samples["power"])

    def since(self, mark: int, skip_seconds: float = 1.0) -> dict[str, float]:
        """Summary of the samples collected after ``mark``.

        The first ``skip_seconds`` of a phase are dropped: a GB300 needs about
        a second to ramp from idle to its sustained clock, and averaging that
        ramp in understates power by 40 % or more (it made a card that sits at
        its 1300 W limit look like it drew 550 W).
        """
        start = mark
        if skip_seconds > 0 and len(self.samples["t"]) > mark:
            # Skip the ramp, and never less than a quarter of the window: a
            # phase that overran its budget would otherwise average in the
            # idle tail and report a fraction of the real power.
            window = self.samples["t"][-1] - self.samples["t"][mark]
            cutoff = self.samples["t"][mark] + max(skip_seconds, 0.25 * window)
            start = next((i for i in range(mark, len(self.samples["t"]))
                          if self.samples["t"][i] >= cutoff), mark)
            if len(self.samples["power"]) - start < 2:  # phase too short to trim
                start = mark
        power = self.samples["power"][start:]
        if not power:
            return {}
        return {
            "power_avg_w": round(statistics.fmean(power), 1),
            "power_max_w": round(max(power), 1),
            "temp_max_c": round(max(self.samples["temp"][start:]), 1),
            "clock_avg_mhz": round(statistics.fmean(self.samples["clock"][start:]), 0),
            "util_avg_pct": round(statistics.fmean(self.samples["util"][start:]), 1),
            "samples": len(power),
        }

    def summary(self) -> dict[str, float]:
        return self.since(0, skip_seconds=0.0)


# --------------------------------------------------------------------------- #
# Device information
# --------------------------------------------------------------------------- #
def _nvml_gpus() -> list[dict[str, Any]]:
    """All GPUs the driver knows about, in nvidia-smi index order."""
    fields = "index,uuid,name,memory.total,power.limit,compute_mode"
    try:
        out = subprocess.run(
            ["nvidia-smi", f"--query-gpu={fields}", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10, check=False,
        ).stdout
    except Exception:
        return []
    gpus: list[dict[str, Any]] = []
    for line in out.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != 6:
            continue
        try:
            gpus.append({
                "index": int(parts[0]),
                "uuid": parts[1],
                "name": parts[2],
                "memory_total_mib": float(parts[3]),
                "power_limit_w": float(parts[4]),
                "compute_mode": parts[5],
            })
        except ValueError:
            continue
    return gpus


def _module_power(target: str) -> dict[str, float]:
    """Module-level power budget (GPU + CPU + memory on Grace-Blackwell).

    A module limit below its default is the usual reason a GB300 stops at
    ~1125 W although its own limit says 1300 W: the platform caps the whole
    module, and the CPU/memory share comes out of the same budget.
    """
    try:
        out = subprocess.run(
            ["nvidia-smi", "-i", target, "-q", "-d", "POWER"],
            capture_output=True, text=True, timeout=10, check=False,
        ).stdout
    except Exception:
        return {}
    parts = out.split("Module Power Readings")
    if len(parts) < 2:
        return {}
    values: dict[str, float] = {}
    for line in parts[1].splitlines()[1:]:
        if ":" not in line:
            break
        key, _, raw = line.partition(":")
        key = key.strip().lower()
        token = raw.strip().split()[0] if raw.strip() else ""
        try:
            value = float(token)
        except ValueError:
            continue
        if key.startswith("current power limit"):
            values["module_power_limit_w"] = value
        elif key.startswith("default power limit"):
            values["module_power_default_w"] = value
        elif key.startswith("instantaneous power draw"):
            values["module_power_draw_w"] = value
    return values


def device_info(index: int, uuid: str | None = None,
                label: int | None = None) -> dict[str, Any]:
    props = torch.cuda.get_device_properties(index)
    info: dict[str, Any] = {
        "index": label if label is not None else index,
        "uuid": uuid,
        "name": props.name,
        "compute_capability": f"{props.major}.{props.minor}",
        "sm_count": props.multi_processor_count,
        "memory_total_gib": round(props.total_memory / 1024**3, 1),
    }
    target = uuid or str(index)
    try:
        info["power_limit_w"] = float(subprocess.run(
            ["nvidia-smi", "-i", target, "--query-gpu=power.limit",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5, check=False,
        ).stdout.strip())
    except Exception:
        pass
    try:
        info["driver"] = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=5, check=False,
        ).stdout.strip().splitlines()[0]
    except Exception:
        pass
    try:
        info["clock_max_mhz"] = float(subprocess.run(
            ["nvidia-smi", "-i", target, "--query-gpu=clocks.max.sm",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5, check=False,
        ).stdout.strip())
    except Exception:
        pass
    info.update(_module_power(target))
    info["datasheet"] = datasheet_for(props.name)
    return info


def topology() -> str:
    try:
        return subprocess.run(
            ["nvidia-smi", "topo", "-m"], capture_output=True, text=True,
            timeout=10, check=False,
        ).stdout.strip()
    except Exception:
        return ""


def cpu_info() -> dict[str, Any]:
    info: dict[str, Any] = {
        "machine": platform.machine(),
        "processor": platform.processor(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "threads": torch.get_num_threads(),
        "kernel": platform.release(),
        "os": f"{platform.system()} {platform.release()} ({platform.machine()})",
    }
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as fh:
            text = fh.read()
        models = re.findall(r"^model name\s*:\s*(.+)$", text, re.MULTILINE)
        if models:
            info["cpu_model"] = models[0].strip()
        info["cpu_cores"] = len(re.findall(r"^processor\s*:", text, re.MULTILINE))
    except OSError:
        pass
    if "cpu_model" not in info:
        # aarch64 /proc/cpuinfo has no "model name"; ask lscpu or the device tree.
        for cmd in (["lscpu"], ["sh", "-c", "cat /proc/device-tree/model"]):
            try:
                out = subprocess.run(cmd, capture_output=True, text=True,
                                     timeout=5, check=False).stdout
            except Exception:
                continue
            match = re.search(r"^Model name:\s*(.+)$", out, re.MULTILINE)
            if match:
                info["cpu_model"] = match.group(1).strip()
                break
            if out.strip() and cmd[0] == "sh":
                info["cpu_model"] = out.strip().replace("\x00", "")
                break
    try:
        with open("/proc/meminfo", encoding="utf-8") as fh:
            for line in fh:
                if line.startswith("MemTotal:"):
                    info["ram_gib"] = round(int(line.split()[1]) / 1024**2, 1)
                    break
    except OSError:
        pass
    return info


# --------------------------------------------------------------------------- #
# HBM / GDDR bandwidth
# --------------------------------------------------------------------------- #
def bench_hbm(index: int, size_gib: float = 8.0, seconds: float = 3.0,
              repeats: int = 3, power: "PowerSampler | None" = None) -> dict[str, Any]:
    """Measure read, write, copy and triad bandwidth on one device."""
    torch.cuda.set_device(index)
    free, _total = torch.cuda.mem_get_info(index)
    # Keep three buffers of `size` plus headroom inside free VRAM.
    size = int(min(size_gib * 1024**3, free * 0.20))
    n = size // 4  # float32 elements per buffer

    src = torch.ones(n, dtype=torch.float32, device=f"cuda:{index}")
    src2 = torch.full((n,), 2.0, dtype=torch.float32, device=f"cuda:{index}")
    dst = torch.empty_like(src)

    cases: list[tuple[str, Callable[[], Any], int]] = [
        ("read", lambda: torch.sum(src), size),
        ("write", lambda: dst.fill_(1.0), size),
        ("copy", lambda: dst.copy_(src), 2 * size),
        ("triad", lambda: torch.add(src, src2, out=dst), 3 * size),
    ]

    results: dict[str, Any] = {"buffer_gib": round(size / 1024**3, 2), "phases": {}}
    for name, fn, moved in cases:
        mark = power.mark() if power else None
        best, median, _ = _timed_loop(fn, seconds, repeats)
        results["phases"][name] = {
            "gbs_best": round(moved / best / 1e9, 1),
            "gbs_median": round(moved / median / 1e9, 1),
        }
        if power is not None and mark is not None:
            results["phases"][name]["power"] = power.since(mark)
    del src, src2, dst
    torch.cuda.empty_cache()
    return results


# --------------------------------------------------------------------------- #
# Compute peak (GEMM per precision)
# --------------------------------------------------------------------------- #
def _gemm_case(index: int, dtype: torch.dtype, size: int) -> Callable[[], Any]:
    a = torch.randn(size, size, dtype=dtype, device=f"cuda:{index}")
    b = torch.randn(size, size, dtype=dtype, device=f"cuda:{index}")
    c = torch.empty(size, size, dtype=dtype, device=f"cuda:{index}")
    return lambda: torch.mm(a, b, out=c)


def _fp8_case(index: int, size: int) -> Callable[[], Any]:
    a = torch.randn(size, size, device=f"cuda:{index}").to(torch.float8_e4m3fn)
    b = torch.randn(size, size, device=f"cuda:{index}").to(torch.float8_e4m3fn)
    scale = torch.tensor(1.0, device=f"cuda:{index}")
    out = torch.empty(size, size, dtype=torch.bfloat16, device=f"cuda:{index}")
    return lambda: torch._scaled_mm(
        a, b.t(), scale, scale, out_dtype=torch.bfloat16, out=out
    )


def _fp4_case(index: int, size: int) -> Callable[[], Any]:
    """nvfp4 GEMM via ``_scaled_mm`` with 1x16 block scales (Blackwell+).

    torch requires e4m3 scales for the 1x16 block recipe, one per 16 elements
    along K, contiguous.
    """
    dev = f"cuda:{index}"
    packed = size // 2
    a = torch.empty(size, packed, dtype=torch.float4_e2m1fn_x2, device=dev)
    b = torch.empty(size, packed, dtype=torch.float4_e2m1fn_x2, device=dev)
    scale_a = torch.ones(size, size // 16, dtype=torch.float8_e4m3fn, device=dev)
    scale_b = torch.ones(size, size // 16, dtype=torch.float8_e4m3fn, device=dev)
    out = torch.empty(size, size, dtype=torch.bfloat16, device=dev)
    return lambda: torch._scaled_mm(
        a, b.t(), scale_a, scale_b, out_dtype=torch.bfloat16, out=out
    )


def _set_fp32_precision(mode: str) -> None:
    """Select ieee (true fp32) or tf32 matmul on both old and new torch APIs."""
    try:
        torch.backends.cuda.matmul.fp32_precision = mode  # torch >= 2.9
    except Exception:
        pass
    try:
        torch.backends.cuda.matmul.allow_tf32 = mode == "tf32"
    except Exception:
        pass


def bench_compute(
    index: int,
    dtypes: list[str] | None = None,
    size: int = 16384,
    seconds: float = 3.0,
    repeats: int = 3,
    power: "PowerSampler | None" = None,
) -> dict[str, Any]:
    """Measure GEMM TFLOP/s for every requested precision on one device."""
    torch.cuda.set_device(index)
    name = torch.cuda.get_device_name(index)
    sheet = datasheet_for(name)
    cc = torch.cuda.get_device_capability(index)
    free, _ = torch.cuda.mem_get_info(index)

    # Shrink the matrix so a single case (a, b, c) fits comfortably.
    while 3 * size * size * 4 > free * 0.5 and size > 2048:
        size //= 2

    wanted = dtypes or ["fp32", "tf32", "bf16", "fp16", "fp8", "fp4"]
    results: dict[str, Any] = {"matrix": size, "phases": {}}
    flops = 2.0 * size**3

    for dtype in wanted:
        mark = power.mark() if power else None
        try:
            if dtype == "fp32":
                _set_fp32_precision("ieee")
                fn = _gemm_case(index, torch.float32, size)
            elif dtype == "tf32":
                _set_fp32_precision("tf32")
                fn = _gemm_case(index, torch.float32, size)
            elif dtype == "bf16":
                fn = _gemm_case(index, torch.bfloat16, size)
            elif dtype == "fp16":
                fn = _gemm_case(index, torch.float16, size)
            elif dtype == "fp8":
                if cc[0] < 9:
                    raise RuntimeError("fp8 tensor cores need sm_89+")
                fn = _fp8_case(index, size)
            elif dtype == "fp4":
                if cc[0] < 10:
                    raise RuntimeError("fp4 tensor cores need sm_100+")
                fn = _fp4_case(index, size)
            else:
                raise RuntimeError(f"unknown dtype {dtype!r}")

            best, median, _ = _timed_loop(fn, seconds, repeats)
            tflops = flops / best / 1e12
            peak = sheet.get(f"{dtype}_tflops")
            results["phases"][dtype] = {
                "tflops_best": round(tflops, 1),
                "tflops_median": round(flops / median / 1e12, 1),
                "datasheet_tflops": peak,
                "pct_of_datasheet": round(100.0 * tflops / peak, 1) if peak else None,
            }
        except Exception as exc:  # unsupported precision / API on this build
            results["phases"][dtype] = {"skipped": f"{type(exc).__name__}: {exc}"}
        finally:
            if power is not None and mark is not None:
                results["phases"][dtype]["power"] = power.since(mark)
            _set_fp32_precision("ieee")
            torch.cuda.empty_cache()

    return results


# --------------------------------------------------------------------------- #
# Interconnect: host <-> device and device <-> device
# --------------------------------------------------------------------------- #
def bench_host_device(index: int, size_mib: int = 1024, seconds: float = 3.0,
                      repeats: int = 3) -> dict[str, Any]:
    """Pinned-memory H2D / D2H bandwidth (NVLink-C2C or PCIe)."""
    torch.cuda.set_device(index)
    n = size_mib * 1024**2 // 4
    host = torch.empty(n, dtype=torch.float32, pin_memory=True)
    host.fill_(1.0)
    dev = torch.empty(n, dtype=torch.float32, device=f"cuda:{index}")

    def h2d() -> None:
        dev.copy_(host, non_blocking=True)

    def d2h() -> None:
        host.copy_(dev, non_blocking=True)

    out: dict[str, Any] = {"buffer_mib": size_mib}
    for name, fn in (("h2d", h2d), ("d2h", d2h)):
        best, median, _ = _timed_loop(fn, seconds, repeats)
        out[name] = {
            "gbs_best": round(n * 4 / best / 1e9, 1),
            "gbs_median": round(n * 4 / median / 1e9, 1),
        }
    del host, dev
    torch.cuda.empty_cache()
    return out


def bench_p2p(src: int, dst: int, size_mib: int = 512, seconds: float = 3.0,
              repeats: int = 3) -> dict[str, Any]:
    """Copy bandwidth between two devices (NVLink or PCIe/host bridge)."""
    n = size_mib * 1024**2 // 4
    a = torch.ones(n, dtype=torch.float32, device=f"cuda:{src}")
    b = torch.empty(n, dtype=torch.float32, device=f"cuda:{dst}")
    best, median, _ = _timed_loop(lambda: b.copy_(a), seconds, repeats)
    del a, b
    torch.cuda.empty_cache()
    return {
        "src": src,
        "dst": dst,
        "buffer_mib": size_mib,
        "gbs_best": round(n * 4 / best / 1e9, 1),
        "gbs_median": round(n * 4 / median / 1e9, 1),
    }


# --------------------------------------------------------------------------- #
# CPU
# --------------------------------------------------------------------------- #
def bench_cpu(size_mib: int = 4096, iters: int = 5) -> dict[str, Any]:
    """STREAM triad bandwidth and BLAS GEMM GFLOPS on the host CPU."""
    n = size_mib * 1024**2 // 4
    a = torch.ones(n, dtype=torch.float32)
    b = torch.full((n,), 2.0, dtype=torch.float32)
    c = torch.empty_like(a)

    def triad() -> None:
        torch.add(a, b, out=c)

    start = time.perf_counter()
    for _ in range(iters):
        triad()
    triad_s = (time.perf_counter() - start) / iters

    size = 4096
    x = torch.randn(size, size, dtype=torch.float32)
    y = torch.randn(size, size, dtype=torch.float32)
    z = torch.empty(size, size, dtype=torch.float32)
    start = time.perf_counter()
    torch.mm(x, y, out=z)
    gemm_s = time.perf_counter() - start

    return {
        "threads": torch.get_num_threads(),
        "triad_gbs": round(3 * n * 4 / triad_s / 1e9, 1),
        "gemm_gflops": round(2.0 * size**3 / gemm_s / 1e9, 1),
    }


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #
def _fmt(value: Any, unit: str = "", width: int = 8) -> str:
    if value is None:
        return "n/a".rjust(width)
    if isinstance(value, float):
        return f"{value:,.1f}{unit}".rjust(width)
    return f"{value}{unit}".rjust(width)


def _power_note(power: dict[str, float] | None) -> str:
    """Compact 'at 300 W / 1.0 GHz' suffix for a phase."""
    if not power:
        return ""
    return (f"   [{power['power_avg_w']:.0f} W, "
            f"{power['clock_avg_mhz'] / 1000:.2f} GHz, "
            f"{power['temp_max_c']:.0f} C]")


def print_report(report: dict[str, Any]) -> None:
    cpu = report.get("cpu", {})
    if cpu:
        print("\n== Host ==")
        print(f"  CPU        : {cpu.get('cpu_model', '?')} "
              f"({cpu.get('cpu_cores', '?')} cores, {cpu.get('machine', '?')})")
        print(f"  RAM        : {cpu.get('ram_gib', '?')} GiB")
        print(f"  torch/CUDA : {cpu.get('torch')} / {cpu.get('cuda')}")
        if "triad_gbs" in cpu:
            print(f"  CPU triad  : {cpu['triad_gbs']:,.1f} GB/s "
                  f"({cpu['threads']} threads)")
            print(f"  CPU GEMM   : {cpu['gemm_gflops']:,.1f} GFLOP/s (fp32 BLAS)")

    for gpu in report.get("gpus", []):
        info = gpu["info"]
        sheet = info.get("datasheet", {})
        print(f"\n== GPU {info['index']}: {info['name']} ==")
        print(f"  {info['memory_total_gib']} GiB, CC {info['compute_capability']}, "
              f"{info['sm_count']} SMs, power limit {info.get('power_limit_w', '?')} W")

        hbm = gpu.get("hbm")
        if hbm:
            print(f"  HBM ({hbm['buffer_gib']} GiB buffers)")
            for phase, values in hbm["phases"].items():
                print(f"    {phase:<6} {_fmt(values['gbs_best'], ' GB/s', 12)}"
                      f"   median {_fmt(values['gbs_median'], ' GB/s', 12)}"
                      f"{_power_note(values.get('power'))}")

        comp = gpu.get("compute")
        if comp:
            print(f"  Compute (GEMM {comp['matrix']}^3)")
            for dtype, values in comp["phases"].items():
                if "skipped" in values:
                    print(f"    {dtype:<6} skipped: {values['skipped']}")
                    continue
                peak = values.get("datasheet_tflops")
                share = values.get("pct_of_datasheet")
                extra = f"  ({share:5.1f}% of {peak:,.0f})" if peak and share else ""
                print(f"    {dtype:<6} {_fmt(values['tflops_best'], ' TFLOP/s', 14)}"
                      f"   median {_fmt(values['tflops_median'], ' TFLOP/s', 14)}{extra}"
                      f"{_power_note(values.get('power'))}")

        link = gpu.get("host_device")
        if link:
            print(f"  Host<->device ({link['buffer_mib']} MiB pinned)")
            for direction in ("h2d", "d2h"):
                if direction in link:
                    print(f"    {direction:<6} {_fmt(link[direction]['gbs_best'], ' GB/s', 12)}")

        power = gpu.get("power")
        if power:
            print(f"  Power      : avg {power['power_avg_w']:.0f} W, "
                  f"max {power['power_max_w']:.0f} W, "
                  f"temp max {power['temp_max_c']:.0f} C, "
                  f"SM clock avg {power['clock_avg_mhz']:.0f} MHz, "
                  f"util avg {power['util_avg_pct']:.0f}%")

    for link in report.get("p2p", []):
        if "skipped" in link:
            print(f"\n== P2P ==\n  skipped: {link['skipped']}")
            continue
        print(f"\n== P2P GPU{link['src']} -> GPU{link['dst']} ==")
        print(f"  {_fmt(link['gbs_best'], ' GB/s', 12)} ({link['buffer_mib']} MiB)")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def _run_gpu_suite(args: argparse.Namespace, index: int, parts: set[str],
                   uuid: str | None = None, label: int | None = None) -> dict[str, Any]:
    entry: dict[str, Any] = {"info": device_info(index, uuid, label)}
    tag = f"[gpu {label if label is not None else index}]"
    with PowerSampler(uuid) as sampler:
        if "hbm" in parts:
            entry["hbm"] = bench_hbm(index, size_gib=args.hbm_gib,
                                      seconds=args.seconds, repeats=args.repeats,
                                      power=sampler)
            best = max(v["gbs_best"] for v in entry["hbm"]["phases"].values())
            print(f"{tag} hbm      {best:,.1f} GB/s", file=sys.stderr, flush=True)
        if "compute" in parts:
            entry["compute"] = bench_compute(
                index, dtypes=args.dtype, size=args.matrix,
                seconds=args.seconds, repeats=args.repeats, power=sampler,
            )
            for dtype, values in entry["compute"]["phases"].items():
                if "skipped" in values:
                    print(f"{tag} {dtype:<8} skipped", file=sys.stderr, flush=True)
                else:
                    print(f"{tag} {dtype:<8} {values['tflops_best']:,.1f} TFLOP/s",
                          file=sys.stderr, flush=True)
        if "interconnect" in parts:
            mark = sampler.mark()
            entry["host_device"] = bench_host_device(
                index, seconds=args.seconds, repeats=args.repeats)
            entry["host_device"]["power"] = sampler.since(mark)
            link = entry["host_device"]
            print(f"{tag} h2d      {link['h2d']['gbs_best']:,.1f} GB/s, "
                  f"d2h {link['d2h']['gbs_best']:,.1f} GB/s",
                  file=sys.stderr, flush=True)
        time.sleep(0.5)
    entry["power"] = sampler.summary()
    return entry


def _spawn(command: str, gpus: list[dict[str, Any]], extra: list[str]) -> dict[str, Any] | None:
    """Run one benchmark command in a child process pinned to ``gpus``.

    A child process per GPU is not a style choice: on this class of system a
    GPU in HMM addressing mode and one in ATS mode cannot share a CUDA context
    (cudaErrorSystemNotReady), so mixing them in one process fails outright.
    """
    import tempfile

    fd, tmp = tempfile.mkstemp(suffix=".json")
    os.close(fd)
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = ",".join(g["uuid"] for g in gpus)
    env["SYS_STRESS_GPU_UUID"] = gpus[0]["uuid"]
    env["SYS_STRESS_GPU_INDEX"] = str(gpus[0]["index"])
    cmd = [sys.executable, os.path.abspath(__file__), command,
           "--in-process", "--quiet", "--json", tmp, *extra]
    proc = subprocess.run(cmd, env=env, text=True, capture_output=True)
    if proc.returncode != 0:
        tail = (proc.stderr or proc.stdout).strip().splitlines()[-3:]
        print(f"!! {command} on {[g['name'] for g in gpus]} failed: "
              f"{' | '.join(tail)}", file=sys.stderr)
        return None
    try:
        with open(tmp, encoding="utf-8") as fh:
            return json.load(fh)
    finally:
        os.unlink(tmp)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="FLOPS, memory bandwidth and interconnect benchmarks."
    )
    parser.add_argument("command", choices=["info", "hbm", "compute", "interconnect",
                                            "p2p", "cpu", "all"])
    parser.add_argument("--gpu", type=int, nargs="+",
                        help="nvidia-smi GPU indices (default: all)")
    parser.add_argument("--dtype", nargs="+", help="compute dtypes to run")
    parser.add_argument("--matrix", type=int, default=16384, help="GEMM dimension")
    parser.add_argument("--hbm-gib", type=float, default=8.0, help="HBM buffer size")
    parser.add_argument("--seconds", type=float, default=3.0,
                        help="wall-clock budget per phase")
    parser.add_argument("--repeats", type=int, default=3,
                        help="timed blocks per phase (fastest wins)")
    parser.add_argument("--json", help="write the full report as JSON here")
    parser.add_argument("--quiet", action="store_true", help="suppress the report")
    parser.add_argument("--in-process", action="store_true",
                        help=argparse.SUPPRESS)  # internal: run on visible devices
    args = parser.parse_args(argv)

    if not torch.cuda.is_available():
        sys.exit("CUDA is not available in this PyTorch build.")

    report: dict[str, Any] = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "cpu": cpu_info(),
        "topology": topology(),
        "gpus": [],
        "p2p": [],
    }

    if args.command == "cpu":
        report["cpu"].update(bench_cpu())
        if not args.quiet:
            print_report(report)
        if args.json:
            with open(args.json, "w", encoding="utf-8") as fh:
                json.dump(report, fh, indent=2)
        return 0

    # Child mode: work on whatever CUDA_VISIBLE_DEVICES exposes.
    if args.in_process:
        index = (args.gpu or [0])[0]
        uuid = os.environ.get("SYS_STRESS_GPU_UUID")
        label = int(os.environ.get("SYS_STRESS_GPU_INDEX", index))
        parts = {"hbm", "compute", "interconnect"} if args.command == "all" else {args.command}
        if args.command == "p2p":
            count = torch.cuda.device_count()
            for src in range(count):
                for dst in range(count):
                    if src != dst:
                        report["p2p"].append(bench_p2p(src, dst))
        else:
            report["gpus"].append(_run_gpu_suite(args, index, parts, uuid, label))
        if not args.quiet:
            print_report(report)
        if args.json:
            with open(args.json, "w", encoding="utf-8") as fh:
                json.dump(report, fh, indent=2)
        return 0

    # Parent mode: one child per GPU, pinned by UUID.
    gpus = _nvml_gpus()
    if not gpus:
        sys.exit("nvidia-smi reported no GPUs.")
    selected = [g for g in gpus if not args.gpu or g["index"] in args.gpu]
    if not selected:
        sys.exit(f"No GPU matched --gpu {args.gpu} (available: "
                 f"{[g['index'] for g in gpus]}).")

    parts = {"hbm", "compute", "interconnect"} if args.command == "all" else {args.command}
    extra = ["--matrix", str(args.matrix), "--hbm-gib", str(args.hbm_gib),
             "--seconds", str(args.seconds), "--repeats", str(args.repeats)]
    if args.dtype:
        extra += ["--dtype", *args.dtype]

    for gpu in selected:
        print(f">> GPU {gpu['index']}: {gpu['name']}", file=sys.stderr, flush=True)
        child = _spawn("all" if args.command == "all" else args.command, [gpu], extra)
        if child and child.get("gpus"):
            report["gpus"].append(child["gpus"][0])

    if args.command == "all" and len(selected) > 1:
        print(">> P2P between all GPU pairs", file=sys.stderr, flush=True)
        child = _spawn("p2p", selected, extra)
        if child:
            report["p2p"] = child.get("p2p", [])
        else:
            report["p2p"] = [{
                "skipped": "GPUs cannot share one CUDA context on this host "
                           "(mixed HMM/ATS addressing modes)",
            }]

    if not args.quiet:
        print_report(report)
    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=2)
        print(f"\nJSON report written to {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
