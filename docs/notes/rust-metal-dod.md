# Rust / Metal Feature Definition of Done

> Parity with Python quality bar (ruff + pytest behavior tests).  
> Applies to `crates/*` kernels and micro-benches (Phase 3 / #8 STREAM).

## Merge gates

| Gate | Local | Pre-commit | CI |
|------|-------|------------|-----|
| Format | `cargo fmt --all -- --check` | yes (`.rs` / Cargo / `.metal`) | lint-rust |
| Lint | `cargo clippy --workspace --all-targets -- -D warnings` | yes | lint-rust |
| Tests | `cargo test --workspace` | yes | test-rust |

```bash
make ci-rust    # fmt + clippy + test
make metal-stream
```

## Correctness before performance

1. **Host oracle** for every MSL kernel (CPU reference vs GPU readback).  
2. Small buffers in unit tests; edge case non-multiple of threadgroup size.  
3. **Never** assert absolute GB/s or TFLOPS in unit tests (machine/load dependent).  
4. Performance: Criterion or release CLI; record in YAML / reports.

## STREAM example

- Oracle: `metal_stream::oracle` + `verify_stream_kernels`  
- Perf: `run_stream` / `make metal-stream`  
- Soft bound only: `0 < best_gbs < absurd_ceiling` in report contract test  

## Optional (later)

- Metal Shader Validation in Xcode while debugging kernels  
- `cargo llvm-cov` floors  
- Criterion regression baselines / Bencher  
