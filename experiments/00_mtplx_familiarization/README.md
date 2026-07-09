# Experiment 00: MTPLX Familiarization

## Hypothesis
MTPLX's custom Metal kernels (verify_qmv, GDN, innovation-tape, capture/replay) are the primary source of its 1.6-2.24x decode gains over stock MLX. Understanding exactly how and where they're called will inform all future optimization work.

## Background
MTPLX implements native Multi-Token Prediction (MTP) speculative decoding using a model's own MTP heads. It achieves significant speedups via 4 custom Metal kernels registered as MLX primitives, built into an optimized MLX fork.

## Setup
- Install: `brew install youssofal/mtplx/mtplx` (or pip)
- Models: Qwen3/3.6 series (MTP-capable)
- Hardware: M5 Max, 128 GB unified memory

## Method
1. Install MTPLX and explore repo structure
2. Locate the 4 custom Metal kernels (in native_extensions/ or the MLX fork)
3. Map the MTP speculative decoding flow: draft → verify → accept/reject + residual correction
4. Identify where each kernel is called in the hot path
5. Run `mtplx bench` and `mtplx models` to see the system in action
6. Read the innovation-tape / reversible state logic

## Key Questions to Answer
- Where exactly does verify_qmv replace stock MLX's generic qmv? What shapes trigger it?
- How does the innovation-tape record/replay work? What's stored per draft step?
- How are kernels registered as MLX primitives? What's the C++/Metal interface?
- What does the GDN linear-attention kernel do differently from standard attention?
- Where could a new custom kernel help (identified gaps)?

## Results
(To be filled after completing the spike)

## Learnings
(To be filled)

## Next Steps
Phase 1: Use these findings to design the baseline benchmark suite.
