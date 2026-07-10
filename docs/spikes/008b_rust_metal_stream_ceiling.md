# Spike 008b: Rust + Metal STREAM kernel bandwidth ceiling

**Date:** 2026-07-10  
**Status:** Implemented (`crates/metal_stream`)  
**Aligns with:** HLD Phase 3 (Rust kernel micro-bench) + issue #8

## Decision

Custom Metal kernels will live in **Rust host + MSL**. The **kernel-grade bandwidth denominator** is therefore **Rust-hosted Metal STREAM**, not only MLX Python probes.

| Layer | Source | Role |
|-------|--------|------|
| L0 | Apple published (614 GB/s @ 40-core) | SKU upper bound |
| **L1** | **`metal_stream` (MSL STREAM)** | **Authoritative for custom kernels** |
| L2 | MLX compiled triad | Fast Python proxy |

## Implementation

```text
crates/metal_stream/
  shaders/stream.metal   # copy, scale, add, triad (float4)
  src/lib.rs             # metal-rs host, wall-clock GB/s
  src/main.rs            # CLI + --json
```

```bash
make metal-stream                 # human-readable
cargo run -p metal_stream --release -- --json
make hardware-ceilings-write      # Metal STREAM + MLX → YAML
```

## Measured (this host, 2026-07-10)

| Kernel | GB/s |
|--------|-----:|
| copy | ~525 |
| scale | ~523 |
| add | ~533 |
| **triad (best)** | **~537** |

≈ **87% of published 614 GB/s** — consistent with HPC Metal STREAM literature (~80–90%).

MLX triad the same day: ~536 GB/s (proxy agrees within noise).

## Profile fields (`profile_version: 1.2`)

- `empirical_metal_stream` — kernel denominator  
- `empirical_mlx` — proxy  
- `empirical` — primary (= metal stream when present)  
- conservative = `min(published, empirical)`

## Why not Swift-only / pure Python?

HLD rejected pure-Python long-term for kernel micro-bench; Phase 3 uses Rust + Metal. Ceiling and product kernels share the same host stack so “% of peak bandwidth” is apples-to-apples.

## Engineering DoD (Rust / Metal)

| Gate | Command |
|------|---------|
| Format | `cargo fmt --all -- --check` |
| Lint | `cargo clippy --workspace --all-targets -- -D warnings` |
| Unit tests | `cargo test --workspace` |
| Host oracle | `verify_stream_kernels` — GPU vs CPU reference for copy/scale/add/triad |
| Performance | `run_stream` / `make metal-stream` — **report only**, no absolute GB/s asserts |

Pre-commit runs fmt + clippy + test when `.rs` / `Cargo.*` / `.metal` change.

**Correctness before performance:** oracle tests use small buffers; never `assert!(gbs > 500)`.
