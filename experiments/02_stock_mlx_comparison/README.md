# Experiment 02: Stock mlx-lm vs MTPLX (protocol compare)

## Status (2026-07-11 — issue [#9](https://github.com/weklund/mlx-inference-workbench/issues/9))

| Item | State |
|------|--------|
| MTPLX engine plugin | Shipped (PR #42) |
| Smoke | `make smoke-mtplx-tiny` |
| Paired protocol configs | `protocol_compare_{mlx_lm,mtplx}_qwen35_2b.yaml` |
| Same-session protocol runs | **`81aff0e72f89`** (mlx-lm) / **`f2f5a38b129d`** (MTPLX) |
| `bench compare` + effect size | Done (`--metric e2e_ms`; gate **passed**) |

## Hypothesis

On the same model weights, prompts, hardware profile, and thermal-monitoring class, MTPLX with MTP speculative decoding (`speculative_depth=4`) improves **end-to-end** generation latency vs stock mlx-lm autoregressive decode. Effect size should be reported with CI (not a single tok/s headline). Acceptance rate is secondary (MTPLX arm only).

## Design (fairness)

| Factor | Held constant |
|--------|----------------|
| Model | `Qwen/Qwen3.5-2B` (bf16; `mtplx inspect` → MTP tensors present) |
| Prompts | `agentic_coding_v1` five-prompt pilot (`max_prompts: 8`, `timed_iterations: 5`) |
| Policy | warmup **10**, timed **5**, max_tokens **128**, seed **42**, temp **0**, 30 s cooldown |
| Hardware | `m5_max_128gb` (Apple M5 Max, Mac17,7, 128 GB) |
| Thermal class | **`full`** on both (passwordless `sudo -n /usr/bin/powermetrics`) |

| Factor | Experimental |
|--------|----------------|
| Backend | `mlx-lm` vs `mtplx` |
| Speculative depth | N/A vs **4** |

### Why not reuse `e46a28d62dee` (Qwen3-8B-4bit)?

Comparability Gate allows different models/backends, but a **cross-size / cross-quant** pair is not a meaningful MTPLX effect-size claim. This experiment pairs the **same HF id**. Arms ran back-to-back in one exclusive session (AC, powermode 2, Nominal pressure).

### Metric for `bench compare`

MTPLX v1 is **e2e-only** (`decode_tok_s` / `ttft_ms` null):

```bash
uv run bench compare <mlx_lm_run> <mtplx_run> --metric e2e_ms
```

Lower `e2e_ms` is better. MTPLX `acceptance_rate` from `bench report` only.

**Correctness / acceptance gates:** `require_correctness` stays **false** while `agentic_coding_v1` has no gold `reference` fields (enabling it aborts before timing). Speculative performance claims are gated separately: mean `acceptance_rate == 0` → `quality_tag=speculative_no_accept` → comparability **blocks** compare.

## Configs

| Config | Backend | Make target |
|--------|---------|-------------|
| `configs/experiments/protocol_compare_mlx_lm_qwen35_2b.yaml` | mlx-lm | `make protocol-compare-mlx-lm-qwen35-2b` |
| `configs/experiments/protocol_compare_mtplx_qwen35_2b.yaml` | mtplx | `make protocol-compare-mtplx-qwen35-2b` |

## Results — same-session pair (2026-07-11)

Preflight: AC Power, powermode **2**, thermal pressure **Nominal**, GPU idle-ish, `thermal_monitoring=full` on both.

| Field | mlx-lm `81aff0e72f89` | MTPLX `f2f5a38b129d` |
|-------|------------------------|----------------------|
| experiment_name | `protocol-compare-mlx-lm-qwen35-2b` | `protocol-compare-mtplx-qwen35-2b` |
| thermal_monitoring | **full** | **full** |
| quality_tag | full (5/5) | **would be `speculative_no_accept` under post-#9 gate** (run pre-gated as full) |
| unstable | false | false |
| e2e_ms p50 | **1226** | **4990** |
| e2e_ms mean / CoV | 1034 / **0.46** | 4049 / **0.59** |
| decode_tok_s p50 / CoV | **119.1** / **0.021** | n/a (e2e-only) |
| ttft_ms p50 / CoV | 181.9 / 0.036 | n/a |
| acceptance_rate mean | n/a | **0.0** (n=4 non-null samples; 1/5 timed iters had no draft tokens → `acceptance_rate` null, excluded from n; other four drafted with zero accepts) |
| memory_peak mean | ~3.86 GB | ~4.00 GB |
| git_sha (at run) | `056cf0e5…` | `056cf0e5…` |
| mlx / mlx-lm | 0.31.2 / 0.31.3 | 0.31.2 / 0.31.3 |
| prompt hash | `6362fd25…ef22c5` | same |
| artifacts (local harness cache) | `benchmarks/results/<run_id>/` (gitignored) | same |
| artifacts (durable, immutable) | see [Durable artifacts](#durable-artifacts) below | same |

### Durable artifacts

Raw `summary.json` + `iterations.parquet` for both runs are committed under
`experiments/02_stock_mlx_comparison/artifacts/<run_id>/` (git = durable store;
do not rewrite bytes after publish). Manifest + SHA-256:

| URI (repo-relative) | SHA-256 |
|---------------------|--------|
| `experiments/02_stock_mlx_comparison/artifacts/81aff0e72f89/summary.json` | `2a9213465f739219768bd4325ad475031c9e058845738f3023b5c867a334c98a` |
| `experiments/02_stock_mlx_comparison/artifacts/81aff0e72f89/iterations.parquet` | `de94afee7eb28c07b46c87aaa13d5f39d3a39ef2a87a3bd6a11ca940d89485fb` |
| `experiments/02_stock_mlx_comparison/artifacts/f2f5a38b129d/summary.json` | `14d2a261460eeeb94a8eadd0656957cabe5b47a0f4d2d3094c7fbf248f32ea3d` |
| `experiments/02_stock_mlx_comparison/artifacts/f2f5a38b129d/iterations.parquet` | `3331e6338a83b5db30cee75d23cd072764a6307ce7ea49f4053fdd1e63e1d465` |

Checksum file: [`artifacts/SHA256SUMS`](artifacts/SHA256SUMS) — verify with
`cd experiments/02_stock_mlx_comparison/artifacts && shasum -c SHA256SUMS`.

Recompute stats from Parquet columns `e2e_ms` / `acceptance_rate` (same values
that feed the e2e mean, Mann–Whitney p≈0.151, Cohen's d≈1.76, and
acceptance_rate mean 0.0 over non-null samples). Local gitignored
`benchmarks/results/<run_id>/` is a convenience cache only.

**e2e CoV is high on both arms** because the five-prompt pilot mixes early-EOS (~1 token) and max-length (128 token) generations — absolute e2e is not a stable primary metric for this corpus. Prefer mlx-lm **decode** CoV (~2%) for harness stability stories; for cross-backend claims use e2e with the CI below, or a fixed-length synthetic prompt set later.

### Statistical compare (`--metric e2e_ms`)

```text
comparable: true
violations: []
n_a=5  n_b=5
mean_a (mlx-lm) = 1033.65 ms
mean_b (MTPLX)  = 4048.67 ms
mean_diff (b−a) = +3015.02 ms   # MTPLX slower
test: mannwhitney_u

# Primary inferential result (α=0.05 significance)
p_value ≈ 0.151
significant_at_0_05: false
verdict: no_significant_difference

# Supplementary effect-size statistics (not used for the significance call)
Cohen's d ≈ 1.76
CI95 (mean_diff) ≈ [1005, 4497] ms
```

**How the numbers are produced** (`workbench.statistics_compare`): Shapiro–Wilk
on each arm (when n≥3) selected **Mann–Whitney U** (two-sided) over Welch’s t;
the **p-value** is the sole gate for `significant_at_0_05` / `verdict`.
**Cohen’s d** is pooled-SD standardized mean difference (b−a), reported for
magnitude only. **CI95 (mean_diff)** under Mann–Whitney is a **bootstrap**
interval on the mean difference: 2000 resamples with replacement of each arm
(`numpy` Generator seed 0), then the 2.5th and 97.5th percentiles of
\(\bar{b}^* - \bar{a}^*\) (not a CI on the U statistic).

**Practical reading (not a marketing win):** point estimate says MTPLX e2e is ~**3.9×** mlx-lm mean on this pilot, with **zero** draft acceptance (`acceptance_rate=0`). Large supplementary Cohen's d but **non-significant** Mann–Whitney p at α=0.05 with n=5 and high within-arm e2e variance. Do **not** publish a single tok/s “MTPLX is faster” claim from this pair.

## Analysis

1. **Harness DoD for #9 is met:** MTPLX run succeeds with acceptance fields populated (honest zeros, not silent nulls); report includes effect size + CI + verdict. **Post-fix:** mean `acceptance_rate == 0` now demotes `quality_tag` to `speculative_no_accept` and the Comparability Gate **blocks** performance compare (historical pair stored as `full` before this gate).
2. **Product/performance finding:** On Qwen3.5-2B bf16, agentic pilot prompts, depth=4, temp=0, MTP drafts did not accept. Speculative path paid overhead without speedup → slower e2e than stock mlx-lm AR.
3. **Likely drivers to investigate later (out of #9 scope):** prompt style (raw completion vs chat template), MTP head quality on short agentic JSON tasks, `speculative_depth`, quant/forge profiles, larger MTP models (9B/27B) where custom kernels dominate, stream timing for decode-apples comparison.
4. **Do not compare this pair to `e46a28d62dee`** for effect size (different model/size/quant).

## Learnings

- Fair #9 compare needs **same model id**, not only gate-passing metadata.
- `acceptance_rate=0.0` is a real measurement, not a harness bug (field non-null when drafts run).
- e2e-only MTPLX forces `--metric e2e_ms`; decode-based CoV stories stay on mlx-lm until stream timestamps land.

## Next steps

1. Optional: fixed-token synthetic prompts to cut e2e CoV; re-run pair for tighter CI.
2. Optional: depth / model size sweep; chat-template parity experiment.
3. Close #9 when this note + configs land on `main`.
4. Adaptive depth / EXP work remains separate (#14, epic #4).
