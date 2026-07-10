"""Empirical M5 Max ceiling probes (bandwidth + matmul FLOPS) — issue #8.

Userspace MLX measurements only. Not Apple Silicon performance-counter truth.
Results are lower-bound *achievable* ceilings for roofline denominators when
combined conservatively with published specs.
"""

from __future__ import annotations

from dataclasses import dataclass
import subprocess
import time
from typing import Any


@dataclass(frozen=True)
class BandwidthProbeResult:
    """Result of one empirical bandwidth probe (GB/s)."""

    gbs: float
    method: str
    buffer_bytes: int
    iterations: int
    notes: str


@dataclass(frozen=True)
class FlopsProbeResult:
    """Result of one empirical matmul FLOPS probe (TFLOPS)."""

    tflops: float
    method: str
    m: int
    n: int
    k: int
    iterations: int
    notes: str


@dataclass(frozen=True)
class ChipProbeResult:
    """Best-effort host chip identity from sysctl / system_profiler."""

    chip: str | None
    memsize_bytes: int | None
    cpu_cores: int | None
    gpu_cores: int | None
    hw_model: str | None
    raw: dict[str, Any]


def detect_chip() -> ChipProbeResult:
    """Best-effort host identity (sysctl + system_profiler)."""
    chip = _sysctl("machdep.cpu.brand_string")
    mem = _sysctl("hw.memsize")
    ncpu = _sysctl("hw.ncpu")
    hw_model = _sysctl("hw.model")
    gpu_cores = _gpu_cores_system_profiler()
    return ChipProbeResult(
        chip=chip,
        memsize_bytes=int(mem) if mem and mem.isdigit() else None,
        cpu_cores=int(ncpu) if ncpu and ncpu.isdigit() else None,
        gpu_cores=gpu_cores,
        hw_model=hw_model,
        raw={"sysctl_memsize": mem, "sysctl_ncpu": ncpu},
    )


def _sysctl(key: str) -> str | None:
    try:
        out = subprocess.run(
            ["sysctl", "-n", key],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        if out.returncode == 0:
            return out.stdout.strip()
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return None


def _gpu_cores_system_profiler() -> int | None:
    try:
        out = subprocess.run(
            ["system_profiler", "SPDisplaysDataType"],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        if out.returncode != 0:
            return None
        # Prefer GPU block "Total Number of Cores: N"
        for line in out.stdout.splitlines():
            if "Total Number of Cores" in line:
                tail = line.split(":")[-1].strip()
                if tail.isdigit():
                    return int(tail)
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return None


def measure_memory_bandwidth_gbs(
    *,
    n_elements: int = 256 * 1024 * 1024,  # 256M float32 ≈ 1 GiB working set
    iterations: int = 25,
    warmup: int = 8,
) -> BandwidthProbeResult:
    """STREAM-like bandwidth via compiled MLX triad: ``a = b + s * c``.

    Bytes moved per iteration ≈ 3 * n_elements * 4 (two reads + one write).
    Uses ``mx.compile`` so the runtime can fuse the kernel (eager triad
    under-reports badly on M5 Max — ~half of compiled). Reports peak GB/s
    over timed iterations after warmup.

    Do **not** use ``mx.array(x)`` as a "copy" probe: it is not a full DRAM
    traffic path and invents absurd GB/s.
    """
    import mlx.core as mx

    n = int(n_elements)
    b = mx.random.normal((n,), dtype=mx.float32)
    c = mx.random.normal((n,), dtype=mx.float32)
    s = mx.array(1.0001, dtype=mx.float32)
    mx.eval(b, c, s)

    @mx.compile
    def triad(bb: Any, cc: Any, ss: Any) -> Any:
        return bb + ss * cc

    for _ in range(warmup):
        mx.eval(triad(b, c, s))
    mx.synchronize()

    best = 0.0
    bytes_per_iter = 3 * n * 4  # triad: 2 reads + 1 write
    for _ in range(iterations):
        mx.synchronize()
        t0 = time.perf_counter()
        a = triad(b, c, s)
        mx.eval(a)
        mx.synchronize()
        dt = time.perf_counter() - t0
        if dt > 0:
            best = max(best, bytes_per_iter / dt / 1e9)

    return BandwidthProbeResult(
        gbs=float(best),
        method="mlx_compiled_stream_triad",
        buffer_bytes=n * 4,
        iterations=iterations,
        notes=(
            "Compiled MLX triad a=b+s*c on ~1GiB f32 buffers; peak of timed iters. "
            "Userspace achievable bandwidth (often ~80-90% of Apple published peak), "
            "not a DRAM counter measurement."
        ),
    )


def measure_matmul_tflops(
    *,
    size: int = 4096,
    iterations: int = 15,
    warmup: int = 3,
) -> FlopsProbeResult:
    """Dense matmul throughput as a practical FP32-ish compute ceiling.

    FLOPs per matmul = 2 * M * N * K. This is *achievable* GEMM throughput
    on the MLX stack, not a guaranteed silicon peak (Apple does not publish
    GPU FP32 TFLOPS for M5 Max as of 2026-07).
    """
    import mlx.core as mx

    m = n = k = int(size)
    x = mx.random.normal((m, k), dtype=mx.float32)
    y = mx.random.normal((k, n), dtype=mx.float32)
    mx.eval(x, y)

    for _ in range(warmup):
        z = x @ y
        mx.eval(z)
    mx.synchronize()

    best = 0.0
    flops_per = 2.0 * m * n * k
    for _ in range(iterations):
        mx.synchronize()
        t0 = time.perf_counter()
        z = x @ y
        mx.eval(z)
        mx.synchronize()
        dt = time.perf_counter() - t0
        if dt > 0:
            best = max(best, flops_per / dt / 1e12)

    return FlopsProbeResult(
        tflops=float(best),
        method="mlx_matmul_fp32",
        m=m,
        n=n,
        k=k,
        iterations=iterations,
        notes=(
            "Peak TFLOPS over timed M×K @ K×N FP32 matmuls via MLX. "
            "Methodology stand-in for peak-FLOPS microkernel; not vendor peak."
        ),
    )
