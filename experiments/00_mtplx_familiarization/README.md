# Experiment 00: MTPLX Familiarization

## Hypothesis

MTPLX's custom Metal kernels (verify-shape QMM, fused MLP, GQA packed SDPA, innovation tape GDN capture) are the primary source of its 1.6–2.24× decode gains over stock MLX. Understanding exactly how and where they're called will inform all future optimization work.

## Background

MTPLX implements native Multi-Token Prediction (MTP) speculative decoding using a model's own MTP heads. It achieves significant speedups via ~20+ custom Metal kernels (registered via `mx.fast.metal_kernel`) that target the "verify shape gap" at M=4..6 rows where stock MLX is suboptimal.

## Setup

- MTPLX 2.0.1 installed via `brew install youssofal/mtplx/mtplx`
- Hardware: Apple M5 Max, 128 GB unified memory, macOS 26.5.2
- MLX 0.31.2 (bundled with MTPLX)

## Method

1. Installed MTPLX, explored package structure at `/opt/homebrew/var/mtplx/venv-2.0.1/lib/python3.13/site-packages/mtplx/`
2. Read all kernel files (`kernels/`, `verify_kernels.py`, `verify_qmv.py`, `nax_verify.py`, `gdn_capture.py`)
3. Read the speculative decoding flow (`speculative.py`, `generation.py`, `native_mlp.py`, `mtp_patch.py`)
4. Read the state management (`cache_state.py`, `gdn_capture.py`, `adaptive.py`)
5. Ran `mtplx hardware` to confirm M5 Max detection and eligibility
6. Documented all findings in `docs/spikes/00_mtplx_familiarization.md`

## Results

See [docs/spikes/00_mtplx_familiarization.md](/docs/spikes/00_mtplx_familiarization.md) for the complete findings.

**Key discoveries:**
- 20+ custom Metal kernels, ALL implemented via `mx.fast.metal_kernel` (inline MSL)
- No C++ native extensions shipped in production (the native_gdn_tail path exists but isn't built)
- The primary win is verify-shape QMM at M=4..6 (kernels fill the dead zone between M=1 decode and M>>16 prefill)
- Innovation tape enables O(T) rollback for hybrid attention+GDN models without full recomputation
- MTPLX is fully programmatic: `mtplx.load()` → `MTPLXRuntime` (resolves OQ#1)
- Runtime kernel self-check at model load time automatically disables any kernel that produces wrong results on untested hardware

## Learnings

1. `mx.fast.metal_kernel` is the production path — not C++ extensions. Prototyping is accessible.
2. The verify shape gap (M=4..6) is THE target. Stock MLX leaves significant performance on the table here.
3. Innovation tape = write-ahead log for recurrent state. Enables cheap speculative rollback.
4. Adaptive EV depth policy is already a "self-tuning controller" at the depth-selection level.
5. Hardware-aware block tuning (architecture suffix 'd'/'s') means kernels already differentiate between Max and Pro chips.

## Next Steps

- Phase 0.5: Use mlx-lm for thermal reproducibility spike (MTPLX models need to be downloaded first)
- Phase 1: Implement MTPLX engine plugin using programmatic API
- Phase 2: Reproduce verify_m4/m6 kernels from scratch to build deep understanding
