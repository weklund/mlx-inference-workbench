# Experiments

Each folder in this directory is a self-contained experiment spike. Spikes are numbered sequentially and should build understanding incrementally.

## How to Structure a New Experiment

1. Create a new folder: `NN_descriptive_name/` (e.g. `04_skew_aware_kernels`)
2. Add a `README.md` inside it using the template below
3. Use the shared `benchmarks/` harness for all measurements
4. Keep experiment-specific scripts, configs, and notes inside the folder
5. Record results and learnings before moving on

## Experiment README Template

```markdown
# Experiment NN: Title

## Hypothesis
What do you expect to find or achieve? Be specific and measurable.

## Background
Why this experiment? What prior spike or profiling result motivated it?

## Setup
- Model(s) used
- Backend(s) compared
- Hardware config (if non-default)
- Any special environment or dependencies

## Method
Step-by-step what you did. Enough detail to reproduce.

## Results
Quantitative results from the benchmark harness. Include:
- Key metrics (tok/s, TTFT, acceptance rate, memory, etc.)
- Comparison to baseline (Phase 1 numbers)
- Profiling data if relevant (screenshots, traces)

## Analysis
What did the results tell you? Why did it work (or not)?

## Learnings
What did you learn that informs future spikes?

## Next Steps
What should be tried next based on these findings?
```

## Naming Convention

- `00_mtplx_familiarization` — Phase 0: Understanding MTPLX architecture and kernels
- `01_baseline_m5max` — Phase 1: Establishing baseline measurements
- `02_stock_mlx_comparison` — Comparing stock MLX performance
- `03_custom_metal_kernels` — First custom kernel experiments

## Rules

1. **Always use the shared harness** — Never roll your own measurement for performance claims
2. **Record everything** — A spike with no written results is wasted work
3. **One variable at a time** — Isolate what you're testing
4. **Compare to baseline** — Every result should reference Phase 1 numbers
5. **Time-box spikes** — If a spike isn't yielding insights after 2-3 days, write up what you learned and move on
