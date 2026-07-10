# Phase 1 (#5) Architecture Notes — SOLID / DRY

> Living note for implementation. HLD remains normative for *what*; this note is *how* we structure code so it stays maintainable.

## Does the HLD approach make sense?

**Yes**, at the system level:

| HLD choice | Verdict |
|------------|---------|
| Unified harness over per-tool benches | Correct problem statement |
| Python orchestration + later Rust kernels | Right split (backends are Python-native) |
| Parquet primary, MLflow index | Good data integrity story |
| Engine ABC + plugins | Open/closed for new backends |
| Comparability gate before stats | Prevents silent bad science |
| No retries on failed iterations | Avoids selection bias |
| Config as versioned YAML | Reproducibility |

**Risks if implemented naively (not HLD flaws, implementation traps):**

1. **God-object `harness.py`** that owns timing, stats, I/O, CLI, and engines.
2. **Duplicating** thermal/power helpers already in `scripts/thermal_validation.py`.
3. **Split-brain packages** (`benchmarks/` empty stubs + empty `src/`) with no clear ownership.
4. **Stringly-typed** metrics/params in MLflow without a single schema type.
5. **Hard-wiring mlx-lm** into the orchestrator instead of depending on `Engine`.

## Principles we will follow

### SOLID

| Principle | Application |
|-----------|-------------|
| **S**ingle responsibility | One module ≈ one reason to change (metrics ≠ I/O ≠ engines) |
| **O**pen/closed | New backends = new `Engine` plugin + registry entry; no orchestrator edits |
| **L**iskov | Every engine returns the same `GenerationResult` contract (None, not missing fields) |
| **I**nterface segregation | Engines don't need MLflow/Parquet; storage doesn't need mlx |
| **D**ependency inversion | Orchestrator depends on `Engine`, `RunStore`, `ThermalSensor` abstractions |

### DRY

- **One** `GenerationResult` / `RunRecord` / `MetricSummary` type — used by engines, storage, compare, report.
- **One** metrics implementation (`compute_distribution`) — no copy-paste percentiles in CLI/report.
- **One** config loader — validates `schema_version`, no ad-hoc `yaml.safe_load` in three places.
- **Thermal/power helpers** live in library code; `scripts/thermal_validation.py` becomes a thin caller (refactor when convenient, not blocking MVP).
- **Comparability rules** declared once; gate + tests share the same field list.

### Other engineering defaults

- **Pure core, impure edges:** metrics/gate/config parse have no filesystem side effects in unit tests (inject paths/IO).
- **Fail closed:** bad hash, missing hardware profile, correctness fail → abort before timing loops.
- **Explicit None** for optional fields (acceptance_rate, power) — never omit keys.
- **MVP vertical slice:** stub engine for CI; mlx-lm engine for real runs; same orchestrator path.

## Package layout (refined)

```
src/workbench/           # installable library (import workbench.*)
  models.py              # dataclasses: GenerationResult, RunMetadata, MetricSummary, ...
  config.py              # load/validate ExperimentConfig from YAML
  metrics.py             # pure statistics
  comparability.py       # gate + field checklist
  statistics.py          # Welch / Mann-Whitney / Cohen's d
  prompts.py             # JSONL load + SHA-256
  hardware.py            # HardwareProbe
  thermal.py             # ThermalSensor protocol + powermetrics / anomaly impls
  engines/
    base.py              # Engine ABC
    registry.py          # name → factory
    stub.py              # deterministic fake for tests/CI
    mlx_lm.py            # Phase 1 real backend
  storage/
    parquet_store.py     # primary write
    mlflow_index.py      # secondary index (optional flag)
    run_index.py         # list/load runs from disk
  orchestrator.py        # run lifecycle only
benchmarks/
  cli.py                 # thin Click CLI → orchestrator / compare / list / report
configs/
  hardware/m5_max_128gb.yaml
  experiments/smoke_minimal.yaml
datasets/
  smoke_v1.jsonl
  smoke_v1.sha256
tests/
  unit/                  # pure functions, no GPU
  integration/           # stub engine e2e (no large models)
```

**Why not keep logic in `benchmarks/`?**  
`pyproject` already declares packages `benchmarks` + `src`. Putting the library under `src/workbench` and keeping `benchmarks` as the CLI entry matches the HLD “Python layer” without a second dump of business logic.

**Entry point:** `bench = "benchmarks.cli:main"` (already in pyproject; implement `cli.py`).

## Layering

```
CLI (benchmarks.cli)
    → Orchestrator.run(config)
         → HardwareProbe.capture()
         → PromptManager.load(path, expected_hash)
         → ThermalSensor.baseline()
         → Engine.load / validate_correctness / warmup / generate×N
         → MetricsComputer.from_iterations(...)
         → ParquetStore.write(run)   # first
         → MLflowIndex.log(run)      # second, best-effort
    → Comparator.compare(run_a, run_b)
         → ComparabilityGate.check
         → StatisticalComparator.test
```

## MVP slice for #5 (shippable)

Must have:

1. Models + config schema v1  
2. Metrics + unit tests  
3. Prompt manager + unit tests  
4. Comparability gate + statistical compare + unit tests  
5. Stub engine + orchestrator + Parquet store  
6. CLI: `run`, `compare`, `list`, `report`  
7. Smoke config + dataset  

Defer slightly (same issue, second PR if needed):

- Full mlx-lm engine polish (can land immediately after stub path works)  
- Fancy distribution plots  
- `bench validate` / `rebuild-index` (stubs OK)  
- Extracting thermal_validation script to call `workbench.thermal` (follow-up)

## Anti-patterns we will avoid

- Orchestrator importing `mlx_lm` directly  
- Computing percentiles in the CLI and again in storage  
- MLflow as source of truth  
- Silent skip of correctness gate  
- Global mutable “current run” state  

## Success check for “DRY/SOLID enough”

- Adding llama.cpp later = one file in `engines/` + registry line  
- Changing CoV threshold = config + one constant used by metrics tagging  
- Unit tests for metrics/gate run without Metal/GPU  
- `bench run` with stub engine works offline for CI  
