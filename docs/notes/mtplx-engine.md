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

## Out of scope (still)

- Continuous batching / OpenAI server
- Adaptive EV depth as harness config (#14)
- Official cross-backend numbers before thermal gate (#3)
