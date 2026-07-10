# Experiment 01: Baseline & Profile on M5 Max

## Status (2026-07-10)

| Item | State |
|------|--------|
| Harness + stub path | Shipped (`main`) |
| mlx-lm engine plugin | Shipped (`MlxLmEngine`) — seed + MLX peak memory |
| Prompt corpus | `datasets/agentic_coding_v1.jsonl` (#6) |
| **Official** baseline numbers | **Blocked** on thermal hard gate [#3](https://github.com/weklund/mlx-inference-workbench/issues/3) |
| **Provisional** baseline config | `configs/experiments/baseline_mlx_lm.yaml` |

Any run logged before #3 closes is **provisional**: useful for harness validation and relative engine bring-up, not for claiming “Phase 1 official baseline.”

## Hypothesis

Establishing precise, reproducible baselines across backends (mlx-lm first; later MTPLX, llama.cpp, …) on the same harness will reveal where headroom exists for custom optimization on the M5 Max 128 GB.

## Design (comparability)

All backends share:

- Prompt dataset + SHA-256
- Orchestrator policy (warmup, timed iterations, timeout, thermal, correctness gate)
- `GenerationResult` → metrics → Parquet/MLflow
- Comparability gate before statistical compare

Only the **engine plugin** changes. mlx-lm is the first system under test, not a one-off measurement path.

### Measurement notes (mlx-lm v1)

- **Prompts:** raw completion text (optional `system_prompt` prepend). No automatic chat template (templates would change prefill and break silent comparability).
- **Sampling:** baseline uses `temperature=0`, `random_seed=42`; engine calls `mx.random.seed` per generate.
- **TTFT / decode tok/s:** from `stream_generate` wall-clock segment timestamps when available; e2e-only path leaves TTFT null (never fabricated).
- **Memory:** `mx.get_peak_memory()` after `reset_peak_memory` at generate start (not process RSS).
- **Timestamps:** stream yields are detokenizer segments (usually ~1 token); decode tok/s is segment-based.

## Configs

| Config | Purpose |
|--------|---------|
| `configs/experiments/smoke_mlx_lm_tiny.yaml` | Tiny real Metal smoke (`Qwen3-0.6B-4bit`, smoke dataset) |
| `configs/experiments/baseline_mlx_lm.yaml` | Provisional baseline (`Qwen3-8B-4bit`, agentic_coding_v1, thermal on, max_prompts=8) |

```bash
make smoke-mlx-tiny          # fast real-weights check
make baseline-mlx-lm         # provisional baseline (AC power recommended)
uv run bench report <run_id>
uv run bench compare <a> <b>
```

## Primary SUT (provisional)

- **Hardware:** MacBook Pro M5 Max, 128 GB (`configs/hardware/m5_max_128gb.yaml`)
- **Backend:** `mlx-lm`
- **Model:** `mlx-community/Qwen3-8B-4bit`
- **Workload:** agentic coding v1, first 8 prompts, max_tokens=128, temp=0, seed=42
- **Policy:** 2 warmup, 5 timed, 30s cooldown, thermal monitor + abort if throttling

## Results — provisional run `9f8c7f967277` (2026-07-10)

Logged on AC power via `make baseline-mlx-lm`. **Not official** until thermal gate #3 closes.

| Field | Value |
|-------|--------|
| run_id | `9f8c7f967277` |
| experiment_name | `provisional-baseline-mlx-lm-qwen3-8b-4bit` |
| quality_tag | `full` (5/5 valid, 0 tainted) |
| unstable (decode CoV flag) | `false` |
| TTFT p50 / p90 (ms) | **176.2** / **185.9** (mean 165.5, CoV **18.6%**) |
| decode tok/s p50 / p90 | **108.35** / **108.59** (mean 107.86, CoV **1.25%**) |
| SITL p50 (ms) | **9.23** |
| e2e_ms p50 / p90 | **1345** / **1377** (mean 1343, CoV 2.9%) |
| memory_peak_bytes mean | **~4.80 GB** (`mx.get_peak_memory`, ≈ 4.797e9) |
| thermal_monitoring | **`degraded`** (not `full` — powermetrics path unavailable or partial) |
| hardware | Apple M5 Max, Mac17,7, 128 GB, macOS 26.5.2 |
| git_sha | `5ab44f26f4df456d8cdaf45e92b265b094e83fe3` (main at run time; local #7 engine changes may be uncommitted) |
| mlx-lm | `0.31.3` |
| prompt hash | `6362fd25…ef22c5` (`agentic_coding_v1`, max_prompts=8) |
| artifacts | `benchmarks/results/9f8c7f967277/` (summary.json, iterations parquet/jsonl) |

```bash
uv run bench report 9f8c7f967277
```

### How to re-record / promote

1. Re-run `make baseline-mlx-lm` after engine PR is on the SHA you want cited.
2. Paste new `run_id` + distributions into this table (keep historical rows if useful).
3. Keep `provisional-` prefix until #3 closes; then re-run under validated thermal methodology and mark **official**.

## Analysis

- **Decode stability looks excellent:** CoV 1.25% ≪ 5% unstable flag — good signal that the harness + cooldown can produce tight decode distributions on this SUT.
- **TTFT is noisier:** CoV ~18.6%; first timed sample was ~112 ms vs ~170–191 ms later. Stream TTFT **is** measured (not e2e-only). Worth a second look after more warmup or after thermal methodology is locked (#3).
- **Thermal class is degraded:** Comparability gate will require matching `thermal_monitoring=degraded` on peer runs. Prefer getting `full` thermal before promoting to official baseline.
- **No thermal taints** on this run (0/5).
- **Memory** ~4.8 GB peak for Qwen3-8B-4bit is plausible; method is MLX allocator peak, not process RSS.
- **MLflow** `mlflow_run_id` is null in summary despite `enable_mlflow: true` — investigate if local MLflow tracking is off; Parquet/JSON remain source of truth.

## Next steps

1. Close or document [#3](https://github.com/weklund/mlx-inference-workbench/issues/3) thermal methodology.
2. Full corpus run (remove `max_prompts` in baseline config).
3. Same config shape for MTPLX / llama.cpp engines (#9, #15) → `bench compare` against this run_id family (same prompt hash + hardware profile).
4. Roofline ceilings [#8](https://github.com/weklund/mlx-inference-workbench/issues/8) for bandwidth % claims.
