# MTPLX engine plugin (issue #9)

Thin `Engine` adapter over MTPLX. Orchestrator, metrics, and thermal code do
**not** import `mtplx`.

## OQ#1 — programmatic drive

**Resolved (Phase 0 spike + plugin):** yes. Path:

```text
mtplx.load(model_id, mtp=True) → MTPLXRuntime
mtplx.generation.generate_mtpk(rt, prompt_ids, max_tokens=…, sampler=…,
                               speculative_depth=D, seed=…)
```

CLI / `mtplx bench` are not used for harness claims.

## Config (`model.mtplx`)

Namespaced under the adapter (not flat model keys):

```yaml
model:
  backend: mtplx
  model_id: "…"
  mtplx:
    speculative_depth: 4   # default if block present without key; engine default 4 if block omitted
```

`model.mtplx` is rejected when `backend` is not `mtplx`.

## Timing contract (v1)

**E2e-only:** `token_timestamps=[]`, `ttft_ms=None`, wall-clock `e2e_ms`.
No fabricated per-token marks. Stream TTFT is a later enhancement.

## Speculative metrics mapping

Aligned with MTPLX progress counters:

| Workbench field | Formula |
|-----------------|--------|
| `acceptance_rate` | `accepted_drafts / drafted_tokens` |
| `accepted_length_mean` | `accepted_drafts / verify_calls` |

If `drafted_tokens == 0`, both are `None` (N/A, not 0.0).

Implementation: `workbench.engines.mtplx_engine.speculative_metrics_from_stats`.

## Install

```bash
uv sync --extra mtplx
# or system/brew MTPLX visible on PYTHONPATH
```

Missing package → `EngineLoadError` at `load_model` with install hint.

## Model ids and paths

`mtplx.load` needs a **local directory** with `config.json`. The plugin resolves:

1. Local path containing `config.json`, or
2. Hugging Face `org/name` via `huggingface_hub.snapshot_download`

**MTP weights matter.** Many `mlx-community` 4-bit Qwen3.5 quants are architecture-compatible but **missing MTP tensors** (`mtplx inspect` → `missing-mtp-weights`). Prefer official `Qwen/Qwen3.5-*` (or forged) snapshots that include MTP. Smoke config: `configs/experiments/smoke_mtplx_tiny.yaml` (`make smoke-mtplx-tiny`).

Local smoke (2026-07-10, `b0477187636f`): `make smoke-mtplx-tiny` completed with `backend=mtplx`, e2e metrics populated, and **`acceptance_rate` present** (mean 0.0 on smoke prompts / depth 4 — field is non-null; low accept is a measurement finding, not a harness bug). Decode/TTFT null is expected (e2e-only v1).

## Protocol compare vs mlx-lm (#9 remaining DoD)

Paired configs (same model, prompts, policy):

- `configs/experiments/protocol_compare_mlx_lm_qwen35_2b.yaml`
- `configs/experiments/protocol_compare_mtplx_qwen35_2b.yaml`

```bash
make protocol-compare-mlx-lm-qwen35-2b
make protocol-compare-mtplx-qwen35-2b
uv run bench compare <mlx_run> <mtplx_run> --metric e2e_ms
```

Write-up: `experiments/02_stock_mlx_comparison/README.md`. Prefer same exclusive session and matching `thermal_monitoring` (passwordless `sudo -n` → `full`).

**First protocol pair (2026-07-11):** mlx-lm `81aff0e72f89` vs MTPLX `f2f5a38b129d` on `Qwen/Qwen3.5-2B`, `thermal_monitoring=full`, gate passed, `--metric e2e_ms`. Point estimate: MTPLX slower (~3.9× mean e2e) with **acceptance_rate=0**; Mann–Whitney p≈0.15 (n=5, high e2e CoV). See experiment README — not a published speedup claim.

## Out of scope (still)

- Continuous batching / OpenAI server
- Adaptive EV depth as harness config (#14)
- Cross-size compares (e.g. 8B-4bit mlx-lm vs 2B MTPLX) as published effect sizes
