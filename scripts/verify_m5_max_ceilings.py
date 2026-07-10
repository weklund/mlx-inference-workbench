"""Verify M5 Max published vs empirical ceilings (issue #8).

Runs:
  1) Rust Metal STREAM (kernel-grade denominator) when Cargo/macOS available
  2) MLX compiled triad (Python proxy)
  3) MLX FP32 matmul (compute proxy)

Usage:
    make hardware-ceilings
    make hardware-ceilings-write
    make metal-stream
"""

from __future__ import annotations

from datetime import date
import json
from pathlib import Path
import shutil
import subprocess
import sys

import click
import yaml

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROFILE = ROOT / "configs" / "hardware" / "m5_max_128gb.yaml"

PUBLISHED_BANDWIDTH_BY_GPU_CORES: dict[int, float] = {
    32: 460.0,
    40: 614.0,
}
APPLE_SPECS_URL = "https://support.apple.com/en-us/126319"


def _run_metal_stream() -> dict | None:
    """Build and run ``target/release/metal_stream --json`` (avoids cargo log noise)."""
    cargo = shutil.which("cargo")
    if cargo is None:
        return None
    try:
        build = subprocess.run(
            [cargo, "build", "-p", "metal_stream", "--release", "--quiet"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=600,
            check=False,
        )
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return None
    if build.returncode != 0:
        return {"_error": (build.stderr or build.stdout or "cargo build failed")[-500:]}

    binary = ROOT / "target" / "release" / "metal_stream"
    if not binary.is_file():
        return {"_error": f"missing binary {binary}"}
    try:
        proc = subprocess.run(
            [str(binary), "--json"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=600,
            check=False,
        )
    except (subprocess.TimeoutExpired, FileNotFoundError) as e:
        return {"_error": str(e)}
    if proc.returncode != 0:
        return {"_error": (proc.stderr or proc.stdout or "metal_stream failed")[-500:]}
    try:
        return json.loads(proc.stdout)
    except json.JSONDecodeError as e:
        return {"_error": f"JSON parse: {e}; out={proc.stdout[:200]!r}"}


@click.command()
@click.option(
    "--profile",
    type=click.Path(path_type=Path),
    default=DEFAULT_PROFILE,
    show_default=True,
)
@click.option("--write/--no-write", default=False)
@click.option("--skip-metal-stream/--with-metal-stream", default=False)
@click.option("--bandwidth-iters", default=20, show_default=True)
@click.option("--flops-iters", default=15, show_default=True)
@click.option("--matmul-size", default=4096, show_default=True)
def main(
    profile: Path,
    write: bool,
    skip_metal_stream: bool,
    bandwidth_iters: int,
    flops_iters: int,
    matmul_size: int,
) -> None:
    from workbench.ceilings import (
        detect_chip,
        measure_matmul_tflops,
        measure_memory_bandwidth_gbs,
    )
    from workbench.hardware_profile import (
        bandwidth_utilization_pct,
        load_hardware_profile,
    )

    click.echo("M5 Max ceiling verification (#8)")
    click.echo(f"  Profile: {profile}")
    click.echo()

    chip = detect_chip()
    click.echo("Host probe")
    click.echo(f"  chip:       {chip.chip}")
    click.echo(f"  hw_model:   {chip.hw_model}")
    click.echo(f"  cpu_cores:  {chip.cpu_cores}")
    click.echo(f"  gpu_cores:  {chip.gpu_cores}")
    if chip.memsize_bytes:
        click.echo(f"  memory_gb:  {chip.memsize_bytes / (1024**3):.0f}")
    click.echo()

    gpu_cores = chip.gpu_cores
    published_bw = (
        PUBLISHED_BANDWIDTH_BY_GPU_CORES.get(gpu_cores) if gpu_cores is not None else None
    )

    metal_report: dict | None = None
    metal_gbs: float | None = None
    if not skip_metal_stream:
        click.echo("Running Rust Metal STREAM (kernel-grade ceiling)...")
        metal_report = _run_metal_stream()
        if metal_report and "_error" not in metal_report:
            metal_gbs = float(metal_report["best_gbs"])
            click.echo(f"  method: {metal_report.get('method')}")
            click.echo(f"  device: {metal_report.get('device')}")
            for k in metal_report.get("kernels") or []:
                click.echo(f"  {k['name']:<6} {k['gbs']:.2f} GB/s")
            click.echo(f"  best:   {metal_gbs:.2f} GB/s  ({metal_report.get('best_kernel')})")
        else:
            err = (metal_report or {}).get("_error", "unavailable")
            click.echo(f"  skipped: {err[:200]}")
        click.echo()

    click.echo("Running MLX compiled triad (Python proxy)...")
    bw = measure_memory_bandwidth_gbs(iterations=bandwidth_iters)
    click.echo(f"  method: {bw.method}")
    click.echo(f"  peak:   {bw.gbs:.2f} GB/s")
    click.echo()

    click.echo("Running MLX FP32 matmul (compute proxy)...")
    fl = measure_matmul_tflops(size=matmul_size, iterations=flops_iters)
    click.echo(f"  method: {fl.method}  {fl.m}x{fl.k}@{fl.k}x{fl.n}")
    click.echo(f"  peak:   {fl.tflops:.2f} TFLOPS")
    click.echo()

    # Kernel-grade empirical prefers Metal STREAM
    empirical_primary = metal_gbs if metal_gbs is not None else bw.gbs
    util_vs_pub = bandwidth_utilization_pct(empirical_primary, published_bw)
    if published_bw is not None:
        conservative_bw = min(published_bw, empirical_primary)
    else:
        conservative_bw = empirical_primary

    click.echo("=" * 60)
    click.echo("PUBLISHED vs EMPIRICAL")
    click.echo(f"  GPU cores:              {gpu_cores}")
    click.echo(f"  Bandwidth published:    {published_bw} GB/s")
    if metal_gbs is not None:
        click.echo(f"  Bandwidth Metal STREAM: {metal_gbs:.2f} GB/s  ← kernel denominator")
    click.echo(f"  Bandwidth MLX triad:    {bw.gbs:.2f} GB/s  (proxy)")
    if util_vs_pub is not None:
        click.echo(f"  Primary as % published: {util_vs_pub:.1f}%")
    click.echo(f"  FP32 TFLOPS empirical:  {fl.tflops:.2f} (Apple does not publish)")
    click.echo(f"  Conservative BW:        {conservative_bw}")
    click.echo(f"  Source: {APPLE_SPECS_URL}")
    click.echo("=" * 60)

    if published_bw is not None and empirical_primary > published_bw * 1.05:
        click.echo(
            "WARNING: empirical bandwidth > 105% of published — check method/SKU.",
            err=True,
        )

    if write:
        _write_profile(
            profile,
            gpu_cores=gpu_cores,
            published_bw=published_bw,
            empirical_metal=metal_gbs,
            empirical_mlx=bw.gbs,
            empirical_tflops=fl.tflops,
            mlx_method=bw.method,
            fl_method=fl.method,
            metal_method=(metal_report or {}).get("method") if metal_report else None,
            chip_name=chip.chip or "Apple M5 Max",
            mem_gb=(chip.memsize_bytes or 0) / (1024**3) if chip.memsize_bytes else 128,
        )
        click.echo()
        click.echo(f"Wrote updated profile → {profile}")
        reloaded = load_hardware_profile(profile)
        click.echo(f"  conservative BW:     {reloaded.bandwidth_ceiling_gbs()}")
        click.echo(f"  conservative TFLOPS: {reloaded.flops_ceiling_tflops()}")

    click.echo()
    click.echo("Re-verify: make hardware-ceilings-write | make metal-stream")
    sys.exit(0)


def _write_profile(
    path: Path,
    *,
    gpu_cores: int | None,
    published_bw: float | None,
    empirical_metal: float | None,
    empirical_mlx: float,
    empirical_tflops: float,
    mlx_method: str,
    fl_method: str,
    metal_method: str | None,
    chip_name: str,
    mem_gb: float,
) -> None:
    primary = empirical_metal if empirical_metal is not None else empirical_mlx
    sources = []
    if empirical_metal is not None:
        sources.append(f"metal_stream={metal_method or 'rust_metal_stream'}")
    sources.append(f"mlx={mlx_method}")

    data = {
        "profile": "m5_max_128gb",
        "profile_version": "1.2",
        "chip": chip_name,
        "unified_memory_gb": int(mem_gb) if mem_gb == int(mem_gb) else mem_gb,
        "verified_at": date.today().isoformat(),
        "memory_bandwidth_gbs": {
            "published": published_bw,
            "empirical": round(primary, 2),
            "empirical_metal_stream": (
                round(empirical_metal, 2) if empirical_metal is not None else None
            ),
            "empirical_mlx": round(empirical_mlx, 2),
            "unit": "GB/s",
            "source_published": APPLE_SPECS_URL,
            "source_empirical": "; ".join(sources),
            "notes": (
                "Kernel-grade denominator prefers empirical_metal_stream "
                "(Rust + MSL STREAM). empirical_mlx is a fast Python proxy. "
                "conservative = min(published, empirical)."
            ),
        },
        "gpu_cores": {
            "published": gpu_cores,
            "empirical": gpu_cores,
            "unit": "cores",
            "source_published": APPLE_SPECS_URL,
            "source_empirical": "system_profiler SPDisplaysDataType",
            "notes": "32-core SKU → 460 GB/s; 40-core SKU → 614 GB/s (Apple).",
        },
        "gpu_fp32_tflops": {
            "published": None,
            "empirical": round(empirical_tflops, 2),
            "unit": "TFLOPS",
            "source_published": None,
            "source_empirical": f"{fl_method} via scripts/verify_m5_max_ceilings.py",
            "notes": ("Apple does not publish GPU FP32 TFLOPS. Empirical = MLX FP32 matmul peak."),
        },
        "notes": (
            "Issue #8 / HLD Appendix B / Phase 3 kernel ceiling. "
            "Re-run: make hardware-ceilings-write. Custom kernel % of peak uses "
            "conservative bandwidth (Metal STREAM when available)."
        ),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, default_flow_style=False, sort_keys=False)


if __name__ == "__main__":
    main()
