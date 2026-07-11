# Experiments

Each folder in this directory is a self-contained experiment spike. Spikes are
numbered sequentially and should build understanding incrementally.

**Worked examples**

| Folder | Role |
|--------|------|
| [`01_baseline_m5max`](01_baseline_m5max/) | Protocol mlx-lm baseline + thermal class |
| [`02_stock_mlx_comparison`](02_stock_mlx_comparison/) | Same-model backend compare, durable artifacts, stats write-up |

Copy methodology from these folders; do not invent a one-off measurement path
for published numbers.

## How to Structure a New Experiment

1. Create a folder: `NN_descriptive_name/` (e.g. `04_skew_aware_kernels`).
2. Add a `README.md` using the [template](#experiment-readme-template) below.
3. Prefer forking protocol YAML under `configs/experiments/` + a `make` target
   rather than ad-hoc CLI flags alone.
4. Measure **only** through the shared harness (`bench run` / `bench compare`).
5. For official claims, [archive artifacts](#durable-artifacts-checksums) and
   fill the results / stats sections before moving on.

## Rules

1. **Always use the shared harness** — never roll your own timers for performance claims.
2. **One intentional factor at a time** — backends/models under test change; prompts, seed, warmup, and thermal class stay matched for compares.
3. **Comparability Gate is necessary, not sufficient** — it enforces prompt hash, hardware profile, schema, and thermal monitoring class. It does **not** require the same `model_id`; do not treat cross-size/cross-quant pairs as effect-size claims.
4. **Record everything durable** — gitignored `benchmarks/results/` is a cache; published numbers need archived Parquet + summary + SHA-256.
5. **Time-box spikes** — if nothing lands after 2–3 days, write what you learned and stop.

## Official / protocol DoD checklist

Use this before calling a result “protocol”, “official”, or citing effect sizes
in issues/PRs. Smoke and exploratory runs may skip items marked *official only*.

### Design

- [ ] **Hypothesis** is falsifiable (what would count as no effect?).
- [ ] **Paired configs** (when comparing backends): same prompt dataset +
      checksum, warmup, timed iterations, max_tokens, seed, cooldown, thermal
      flags; only the experimental factor(s) differ.
- [ ] **Same model weights** when the claim is “backend A vs backend B”
      (fork `configs/experiments/protocol_compare_*` as a starting point).
- [ ] **Metric choice**: if either arm is e2e-only (e.g. MTPLX v1), compare with
      `--metric e2e_ms` (not `decode_tok_s`).
- [ ] **Correctness**: `require_correctness: true` only when the dataset has
      non-empty `reference` fields (or a CLI override). Otherwise leave false
      and say so — enabling without gold fails closed before timing.

### Run conditions (*official*)

- [ ] Exclusive session: AC power, high performance power mode, GPU/AI idle.
- [ ] Prefer `thermal_monitoring=full` (passwordless `sudo -n` for
      `/usr/bin/powermetrics`); both arms must share the same thermal class.
- [ ] Same calendar session for small effects (residual day-to-day drift).
- [ ] HLD-style warmup for published runs (configs use **warmup ≥ 10**).

### Harness gates (automatic — still check the report)

- [ ] Run `quality_tag` is not `insufficient_data`.
- [ ] Speculative backends: mean `acceptance_rate == 0` → quality
      `speculative_no_accept` → **comparability blocks** performance compare
      (by design; not a valid MTP speedup claim).
- [ ] `bench compare` either passes the gate or documents the block; never
      hand-wave around a blocked compare.

### Artifacts & write-up (*official*)

- [ ] Archive runs with the [script](#durable-artifacts-checksums) into
      `experiments/NN_…/artifacts/`.
- [ ] `verify` the manifest after archive.
- [ ] README results table includes run ids, thermal class, key metrics, and
      **durable artifact URIs + checksums** (or points at `artifacts/SHA256SUMS`).
- [ ] Statistical section labels **Mann–Whitney U / Welch p-value as primary**
      inferential result; **Cohen’s d** and **CI95 (mean_diff)** as
      supplementary. Note CI method (bootstrap under MWU; Welch–Satterthwaite
      under t-test — see `workbench.statistics_compare`).
- [ ] Acceptance distributions: state **n = non-null samples** if some iters
      have no drafts (`acceptance_rate` null).

## Durable artifacts (checksums)

Harness paths (`benchmarks/results/<run_id>/`) are **gitignored**. For any
result that backs a claim:

```bash
# After bench run(s):
uv run python scripts/archive_run_artifacts.py archive \
  --run-id <id_a> --run-id <id_b> \
  --dest experiments/<NN_name>/artifacts

# Or via Make:
make bench-archive RUNS="<id_a> <id_b>" DEST=experiments/<NN_name>/artifacts

# Later / in CI-ish local check:
uv run python scripts/archive_run_artifacts.py verify \
  --dest experiments/<NN_name>/artifacts
# equivalent: cd experiments/<NN_name>/artifacts && shasum -c SHA256SUMS
```

What gets archived (required):

| File | Role |
|------|------|
| `summary.json` | Aggregate metrics + metadata |
| `iterations.parquet` | Per-iteration fields (`e2e_ms`, `acceptance_rate`, …) for recompute |

The script rewrites `artifacts/SHA256SUMS` (paths relative to that directory).
It **refuses to overwrite** differing bytes unless `--force` (keeps published
artifacts immutable by default).

Reference layout: [`02_stock_mlx_comparison/artifacts/`](02_stock_mlx_comparison/artifacts/).

## Experiment README template

```markdown
# Experiment NN: Title

## Status
| Item | State |
|------|--------|
| Configs | |
| Protocol runs | |
| Durable artifacts | |
| Compare / effect size | |

## Hypothesis
What you expect to find; what would falsify it.

## Background
Prior spike, issue links, why this experiment.

## Design (comparability)
| Factor | Held constant |
|--------|----------------|
| Model / prompts / policy / hardware / thermal class | … |

| Factor | Experimental |
|--------|----------------|
| Backend / depth / … | … |

Metric for `bench compare`: `…` (why).

## Setup
- Config paths + make targets
- Hardware profile
- Dependencies (`uv sync --extra …`)

## Method
1. Preflight (power, thermal, exclusive use)
2. `bench run` / make targets
3. `bench compare … --metric …`
4. `scripts/archive_run_artifacts.py archive …`

## Results
Table: run ids, thermal_monitoring, quality_tag, primary metrics, acceptance.
Link durable URIs + SHA-256 (or `artifacts/SHA256SUMS`).

### Statistical compare
- Primary: test name + p-value + verdict
- Supplementary: Cohen's d, CI95 (mean_diff) + how CI was computed

## Analysis
What the numbers mean; what they do *not* claim.

## Learnings
## Next steps
```

## Naming convention

- `00_mtplx_familiarization` — Phase 0: MTPLX architecture
- `01_baseline_m5max` — Phase 1: baselines on M5 Max
- `02_stock_mlx_comparison` — Stock mlx-lm vs MTPLX (protocol pair)
- `03_custom_metal_kernels` — Custom kernel experiments
- `NN_…` — continue the sequence; one primary question per folder

## Configs & make targets

| Purpose | Example |
|---------|---------|
| Protocol mlx-lm baseline | `make baseline-mlx-lm` → `configs/experiments/baseline_mlx_lm.yaml` |
| Protocol compare arms (#9) | `make protocol-compare-mlx-lm-qwen35-2b` / `protocol-compare-mtplx-qwen35-2b` |
| Archive | `make bench-archive RUNS="…" DEST=experiments/…/artifacts` |
| Compare | `make bench-compare A=… B=…` (add `--metric e2e_ms` via `uv run bench compare` when needed) |

Fork YAML from an existing `protocol_*` or `baseline_*` config when starting a
new official arm so warmup, thermal, and pilot prompt policy stay consistent.
