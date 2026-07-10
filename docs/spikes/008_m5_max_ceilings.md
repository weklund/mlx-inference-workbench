# Spike 008: M5 Max bandwidth & FLOPS ceilings (#8)

**Date:** 2026-07-10  
**Host:** MacBook Pro Mac17,7 · Apple M5 Max · 18 CPU / 40 GPU · 128 GB  
**Command:** `make hardware-ceilings-write`

## Published (Apple)

| Spec | Value | Source |
|------|------:|--------|
| Memory bandwidth (40-core GPU) | **614 GB/s** | [Apple MacBook Pro tech specs](https://support.apple.com/en-us/126319) |
| Memory bandwidth (32-core GPU) | **460 GB/s** | same |
| GPU cores | 32 or 40 | same + `system_profiler` |
| GPU FP32 TFLOPS | **not published** | — |

## Empirical (this host)

| Probe | Method | Result |
|-------|--------|-------:|
| Bandwidth (v1, eager triad) | MLX `a=b+s*c` without compile, ~512 MiB | **~316 GB/s** (~51% of published) — **under-reports** |
| Bandwidth (v2 MLX proxy) | **`mx.compile` triad**, ~1 GiB f32 | **~534–536 GB/s** (~87% of published) |
| Bandwidth (**v3 kernel L1**) | **Rust + MSL STREAM** (`crates/metal_stream`) | **~537 GB/s** triad best (~87% of published) |
| Compute | MLX FP32 matmul 4096³, peak of timed iters | **~40 TFLOPS** |

Lesson: eager MLX under-reported; compiled MLX and Metal STREAM **agree** at ~87% of published. Kernel claims use Metal STREAM (see `008b_rust_metal_stream_ceiling.md`).

Invalid: treating `mx.array(x)` as a STREAM copy (reports absurd multi‑TB/s — not full DRAM traffic).

## Conservative ceilings (roofline denominators)

| Ceiling | Rule | Value |
|---------|------|------:|
| Bandwidth | `min(published, empirical)` | **~533.7 GB/s** |
| FP32 TFLOPS | empirical only (no published) | **~40.3 TFLOPS** |

Stored in `configs/hardware/m5_max_128gb.yaml` (`profile_version: "1.1"`).

## Harness API

```python
from pathlib import Path
from workbench.hardware_profile import load_hardware_profile, bandwidth_utilization_pct

hp = load_hardware_profile(Path("configs/hardware/m5_max_128gb.yaml"))
pct = bandwidth_utilization_pct(achieved_gbs=200.0, ceiling_gbs=hp.bandwidth_ceiling_gbs())
```

## Re-verify

```bash
make hardware-ceilings          # print only
make hardware-ceilings-write    # re-probe + rewrite YAML
```

Re-run after major macOS / MLX upgrades or if the machine is a different M5 Max SKU (32 vs 40 GPU cores).

## Caveats

- Empirical bandwidth is **userspace MLX achievable**, not Metal performance-counter peak.
- Matmul TFLOPS is **achievable GEMM** on the MLX stack, not a silicon marketing peak.
- Do not claim “% of Apple peak” without stating whether the denominator is published or conservative.
- Prefer quiet machine + AC + high performance when re-verifying; load can shave a few %, not explain the old 50% gap.
