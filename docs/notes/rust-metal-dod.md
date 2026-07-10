# Rust / Metal Feature Definition of Done

> Parity with (and beyond) the Python quality bar.  
> Applies to `crates/*` kernels and micro-benches (Phase 3 / #8 STREAM).

## Merge gates

| Gate | Local | Pre-commit | CI |
|------|-------|------------|-----|
| Format | `cargo fmt --all -- --check` | yes (`.rs` / Cargo / `.metal`) | lint-rust |
| Lint | `cargo clippy --workspace --all-targets --all-features -- -D warnings` | yes | lint-rust |
| Docs | `RUSTDOCFLAGS=-D warnings cargo doc --workspace --no-deps` | yes | lint-rust |
| Tests | `cargo test --workspace` | yes | test-rust |

```bash
make ci-rust    # fmt + clippy + doc + test
make metal-stream
```

## Lint policy (pedantic)

Configured in root `Cargo.toml` under `[workspace.lints.*]`; crates set:

```toml
[lints]
workspace = true
```

| Layer | Level | Notes |
|-------|-------|--------|
| `clippy::all` | **deny** | Baseline correctness |
| `clippy::pedantic` | **deny** | Style / API polish; selective `allow`s for math/Metal |
| `clippy::nursery` | **warn** | Experimental; review noise before promoting |
| `clippy::cargo` | **warn** | Manifest hygiene |
| `rust.missing_docs` | **warn** (treated as deny via `-D warnings` in CI) | Public items documented |
| `rustdoc::*` | deny broken links | Intra-doc links must resolve |
| `unsafe_op_in_unsafe_fn` | **deny** | Explicit unsafe blocks inside `unsafe fn` |

Value knobs: `clippy.toml` (complexity threshold, short idents for STREAM).

**Intentional pedantic allows** (measurement / Metal host): short math names, cast precision on buffer sizes, float abs-diff tests, wildcard `metal::*` imports, missing_errors/panics doc (we use rustc `missing_docs` instead of clippy’s Result essay style).

## Documentation standard

- Crate-level `//!` module docs on every lib and bin.
- Every `pub` item: purpose, units (GB/s, float4, bytes), and correctness vs performance role.
- Link host oracles to GPU verify entry points with rustdoc links.
- Prefer examples in docs for pure oracles.

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
- `cargo llvm-cov` floors (#34)
- Criterion regression baselines / Bencher
- Promote selected `nursery` lints to deny once stable for this crate
