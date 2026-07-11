# Experiment 01: Baseline & Profile on M5 Max

## Status (2026-07-10)

| Item | State |
|------|--------|
| Harness + stub path | Shipped (`main`) |
| mlx-lm engine plugin | Shipped (`MlxLmEngine`) |
| Prompt corpus | `datasets/agentic_coding_v1.jsonl` (#6) |
| Thermal gate [#3](https://github.com/weklund/mlx-inference-workbench/issues/3) | **Closed** — [report](../../docs/spikes/005_thermal_reproducibility.md) (protocol CoV 2.57%) |
| **Official** baseline | **`e46a28d62dee`** (see below) — issue [#36](https://github.com/weklund/mlx-inference-workbench/issues/36) |
| Config | `configs/experiments/baseline_mlx_lm.yaml` |

## Hypothesis

Establishing precise, reproducible baselines across backends (mlx-lm first; later MTPLX, llama.cpp, …) on the same harness will reveal where headroom exists for custom optimization on the M5 Max 128 GB.

## Design (comparability)

All backends share:

- Prompt dataset + SHA-256
- Orchestrator policy (warmup, timed iterations, timeout, thermal, correctness gate)
- `GenerationResult` → metrics → Parquet/MLflow
- Comparability gate before statistical compare

Only the **engine plugin** changes.

### Measurement notes (mlx-lm v1)

- **Prompts:** raw completion text (optional `system_prompt` prepend). No automatic chat template.
- **Sampling:** `temperature=0`, `random_seed=42`; engine calls `mx.random.seed` per generate.
- **TTFT / decode tok/s:** from `stream_generate` wall-clock segment timestamps when available.
- **Memory:** `mx.get_peak_memory()` after `reset_peak_memory` at generate start.
- **Official thermal protocol:** exclusive session, AC, high performance, powermetrics (`thermal_monitoring=full` when passwordless sudo is configured), 30 s cooldown — see [005 thermal report](../../docs/spikes/005_thermal_reproducibility.md).

## Configs

| Config | Purpose |
|--------|---------|
| `configs/experiments/smoke_mlx_lm_tiny.yaml` | Tiny real Metal smoke (`Qwen3-0.6B-4bit`) |
| `configs/experiments/baseline_mlx_lm.yaml` | Official-class baseline (`Qwen3-8B-4bit`, agentic_coding_v1, thermal on, max_prompts=8) |

```bash
make smoke-mlx-tiny
make baseline-mlx-lm    # exclusive session recommended
uv run bench report <run_id>
uv run bench compare <a> <b>
```

## Primary SUT

- **Hardware:** MacBook Pro M5 Max, 128 GB (`configs/hardware/m5_max_128gb.yaml`)
- **Backend:** `mlx-lm`
- **Model:** `mlx-community/Qwen3-8B-4bit`
- **Workload:** agentic coding v1, first 8 prompts, max_tokens=128, temp=0, seed=42
- **Policy:** 2 warmup, 5 timed, 30s cooldown, thermal monitor + abort if throttling

---

## Results — **official** run `e46a28d62dee` (2026-07-10)

Logged under Phase 0.5 protocol (exclusive session, AC, high performance, Nominal pressure, GPU idle).  
Methodology: [`docs/spikes/005_thermal_reproducibility.md`](../../docs/spikes/005_thermal_reproducibility.md).  
Harness fix: powermetrics probe uses passwordless `sudo -n` when available so `thermal_monitoring=full`.

| Field | Value |
|-------|--------|
| run_id | **`e46a28d62dee`** |
| experiment_name | `official-baseline-mlx-lm-qwen3-8b-4bit` |
| quality_tag | `full` (5/5 valid, 0 tainted) |
| unstable | `false` |
| **thermal_monitoring** | **`full`** |
| TTFT p50 / p90 (ms) | **159.8** / **~181** (mean 158.6, CoV **4.9%**) |
| decode tok/s p50 / p90 | **107.54** / **~108.9** (mean 108.23, CoV **1.30%**) |
| SITL p50 (ms) | **9.30** |
| e2e_ms p50 | **1330** (mean ~1334, CoV **0.88%**) |
| memory_peak_bytes mean | **~4.80 GB** |
| hardware | Apple M5 Max, Mac17,7, 128 GB, macOS 26.5.2 |
| git_sha | `6c331f7304e0ae82d7cbe4443474118d8e6aec5a` (+ thermal sudo probe on this branch) |
| mlx / mlx-lm | `0.31.2` / `0.31.3` |
| prompt hash | `6362fd25…ef22c5` (`agentic_coding_v1`, max_prompts=8) |
| artifacts | `benchmarks/results/e46a28d62dee/` (local; gitignored raw results) |

```bash
uv run bench report e46a28d62dee
```

### Protocol notes

- Preflight: AC, powermode high performance, thermal Nominal, GPU ~idle before run.
- Decode CoV **1.3%** ≪ 5% unstable flag — consistent with thermal gate findings.
- TTFT still noisier than decode (expected for short first-token path); CoV ~5% on this official run.

---

## Results — provisional run `9f8c7f967277` (historical)

Earlier harness validation run (**not official**). Kept for history.

| Field | Value |
|-------|--------|
| run_id | `9f8c7f967277` |
| thermal_monitoring | `degraded` (pre–sudo-n probe) |
| decode tok/s p50 | ~108.35 (CoV ~1.25%) |
| quality_tag | `full` |

---

## Analysis

- **Official decode stability is excellent** (CoV ~1.3%).
- **Prefer `e46a28d62dee` for compares** so peers match `thermal_monitoring=full` (comparability gate treats full vs degraded as incompatible).
- Full corpus: remove `max_prompts` when ready for a longer official sweep.
- Next: same protocol for MTPLX (#9 compare arm) and other backends.

## Next steps

1. Close [#36](https://github.com/weklund/mlx-inference-workbench/issues/36) when this note + config land on `main`.
2. Optional full 24-prompt official re-run (drop `max_prompts`).
3. MTPLX / llama.cpp under same protocol → `bench compare` vs `e46a28d62dee` (same prompt hash + hardware + thermal class).
