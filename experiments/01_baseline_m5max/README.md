# Experiment 01: Baseline & Profile on M5 Max

## Hypothesis
Establishing precise, reproducible baselines across multiple backends (MTPLX, stock MLX, llama.cpp, BaseRT) will reveal exactly where headroom exists for custom optimization on the M5 Max 128 GB.

## Background
Before optimizing anything, we need to know what "good" looks like on this specific hardware. The baseline becomes the yardstick for all future spikes.

## Setup
- Hardware: MacBook Pro M5 Max, 128 GB unified memory
- Backends: MTPLX (optimized fork), stock MLX + mlx-lm, llama.cpp (Metal), BaseRT
- Models: Qwen3/3.6 27B, Llama 3.x, others as available
- Quantizations: Q4, Q8, FP16 where feasible

## Workloads
- Short-medium context chat/coding (256-2048 tokens)
- Long-context (32k-128k+)
- Sustained agent-like generation (multi-turn, tool-use patterns)
- Varying temperature (0.0, 0.6, 1.0) and draft depth (for speculative)

## Metrics
- Prefill tokens/sec and TTFT
- Decode tokens/sec (sustained over 30+ seconds)
- Effective tokens/sec with speculative (acceptance rate, mean accepted length)
- Peak and sustained memory usage (weights + KV + speculative state)
- Thermal behavior (temperature, fan speed, throttling events)
- End-to-end latency for representative prompts

## Profiling Tools
- MTPLX built-in benchmarking (`mtplx bench`)
- MLX profiling hooks
- Xcode GPU capture / Instruments (kernel-level)
- Custom scripts in `src/profiling/`

## Results
(To be filled — target: clean comparison table across all backends)

## Analysis
(To be filled — identify: what's fast, what's slow, where does MTPLX win and why?)

## Next Steps
Phase 2: Deep kernel understanding of the bottlenecks identified here.
