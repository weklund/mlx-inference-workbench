# TDD Process — MLX Inference Workbench

## Honest status

Phase 1 (#5) had an **implementation-first spike**: production code landed, then tests. That is **not** TDD.

**Correct TDD cycles:**

| Property | RED | GREEN |
|----------|-----|-------|
| Deadline exceeded → not a valid sample | timeout test | orchestrator wall-clock timeout |
| n&lt;2 valid samples → not comparable | `test_single_valid_iteration_cannot_claim_statistical_comparison` | `check_runs_comparable` quality/n gate |

**Behavioral suites (properties, not implementation):** compare_runs, store+compare, thermal heuristics, CLI list/compare, orchestrator gates.

What we do have:

- Solid unit tests for **metrics**, **prompts**, **comparability**, **stats compare**
- One integration test for **stub e2e run + compare**
- A working `bench` CLI smoke path

What we do **not** have (TDD gaps):

| Area | Production code | Tests that drove design |
|------|-----------------|-------------------------|
| `config.py` | Yes | No dedicated tests |
| `hardware.py` | Yes | None |
| `thermal.py` | Yes | None |
| `engines/base.py` contract | Yes | None (only via stub e2e) |
| `engines/registry.py` | Yes | None |
| `engines/stub.py` | Yes | Indirect only |
| `engines/mlx_lm_engine.py` | Yes | None |
| `storage/run_store.py` | Yes | Indirect via e2e |
| `orchestrator.py` | Yes | Indirect via e2e |
| CLI (`benchmarks/cli.py`) | Yes | None (manual smoke only) |

**Rule from this point:** no new production behavior without a failing test first. Existing code is treated as a **spike** to be **characterized** (tests that lock current behavior) or **replaced** when a red test forces a better design.

---

## TDD cycle (mandatory)

```
1. RED    Write one failing test for the next behavior (from HLD / issue DoD).
2. GREEN  Write the minimum production code to pass that test.
3. REFACTOR  DRY/SOLID cleanup with tests green.
4. COMMIT     Optional checkpoint after green + refactor.
```

### Constraints

- Prefer **one behavior per test** (or tightly related parametrize).
- Prefer **pure unit tests** (no GPU, no network, no real models) unless the behavior is integration by nature.
- GPU/mlx-lm paths: mark `@pytest.mark.gpu` and keep them out of default CI if needed.
- Do **not** “fix the implementation” (private helpers) — test **observable contracts** (public API, return shapes, fail-closed errors).
- Smoke tests in issue #5 remain the **acceptance** layer; unit tests are the **design** layer.

### Red test quality bar

A good red test states:

1. **Given** fixed inputs (config fragment, timestamps, metadata pair, …)
2. **When** a public function/CLI is invoked
3. **Then** an assertion about an **observable outcome** the HLD/user cares about

If you cannot write the assertion without inventing the API, design the **public signature in the test first**, then implement to match.

### Behavioral tests vs brittle implementation tests

**We test properties and contracts, not the current code shape.**

| Prefer (stable) | Avoid (brittle) |
|-----------------|-----------------|
| Valid samples excluded after deadline | Asserting `ThreadPoolExecutor` / private `_generate_with_timeout` |
| No run published when integrity/correctness fails | `match="Correctness gate"` exact strings |
| `valid_iterations == 0` / `unstable is True` | Exact error text, call-count of private helpers |
| Comparability gate blocks different prompt hashes | Asserting field list order inside gate implementation |
| Round-trip store preserves values needed for compare | Asserting JSON key order or file layout trivia |
| Engine returns `len(timestamps) == total_tokens` | Asserting sleep duration or internal sampler choice |

**Smell:** if a harmless refactor (rename private helper, change timeout mechanism, reword exception message) breaks the test, the test was coupled to implementation.

**Smell:** if the test is a line-by-line replay of the function body, delete it and replace with a property.

**Domain exceptions > message matching.** Prefer `raises(OrchestratorError)` / dedicated error types over `match="substring"`.

**Inject fakes at boundaries** (Engine, ThermalSensor, paths), not mocks of every internal call.

---

## Retrofit plan for existing spike (characterization)

Before adding features, lock the spike with tests **that would have been red first**. Order by risk:

### C1 — Config (high leverage, pure)

- [ ] `load_config` accepts `schema_version: "1.0"` smoke YAML
- [ ] rejects unknown `schema_version`
- [ ] missing required `model.backend` → clear error
- [ ] defaults applied for optional benchmark fields

### C2 — RunStore (data integrity)

- [ ] `write` then `load` round-trips metadata + metric values used by compare
- [ ] `list_runs` sees written run_id
- [ ] missing run_id → `FileNotFoundError`

### C3 — Engine registry / stub contract

- [ ] `create_engine("stub")` returns engine named `stub`
- [ ] unknown engine → `KeyError` with known list
- [ ] stub `generate` SUCCESS, `len(timestamps)==total_tokens`, deterministic output for same seed/prompt
- [ ] stub without `load_model` → `EngineLoadError`

### C4 — Orchestrator (behavior, inject engine)

- [ ] with stub + temp results dir: returns `RunRecord` with `valid_iterations == timed_iterations`
- [ ] bad prompt hash → fails before any generate (inject counting engine)
- [ ] correctness gate failure aborts (engine that returns False)

### C5 — CLI (subprocess or CliRunner)

- [ ] `bench run smoke_minimal.yaml` exit 0
- [ ] `bench compare` same two identical stub runs → not blocked; verdict documented
- [ ] `bench compare` after tampering stored prompt hash on one run → exit non-zero / blocked

### C6 — Thermal / hardware (mock subprocess)

- [ ] `OffThermalSensor.mode() == "off"`
- [ ] degraded sensor flags throttle when last duration > 2× median
- [ ] hardware fingerprint includes `machine` / profile keys

**Do not rewrite production code during C1–C6 unless a characterization test proves a bug against HLD.** Prefer green characterization, then refactor.

---

## Forward backlog (true TDD — red first)

These map to remaining #5 / related work. **Write the test file first; leave production unimplemented until red.**

| ID | Behavior | First red test idea | Production touch |
|----|----------|---------------------|------------------|
| T1 | mlx-lm engine contract (mocked generate) | Mock `mlx_lm.stream_generate` yields 3 tokens → `GenerationResult` shape | `mlx_lm_engine.py` |
| T2 | Comparability allows same prompts, different quant | Two metas differ only quant → still comparable | `comparability.py` (if product decision) |
| T3 | Unstable tag when CoV > threshold | Already partially covered — extend edge cases | `metrics.py` |
| T4 | Zero valid iterations → orchestrator error | All iterations ERROR → raises | `orchestrator.py` |
| T5 | Per-iteration timeout → TIMEOUT status | Fake slow engine + short timeout | engine + orchestrator |
| T6 | MLflow optional: `--no-mlflow` leaves disk artifacts only | Run store without mlflow dir requirement | CLI + store |
| T7 | Real mlx-lm config (gpu) | `@pytest.mark.gpu` loads tiny model or skip | config YAML + engine |

Issue #5 DoD should be driven by T4–T6 + C1–C5 completeness, not by more untested code.

---

## Session template

When starting a work session:

1. Pick **one** row from Characterization or Forward backlog.
2. Write **failing** test(s) only; run pytest → confirm RED.
3. Implement minimum GREEN.
4. Refactor if needed.
5. Stop or pick next row — avoid multi-feature PRs.

### Commands

```bash
# Default (no GPU)
uv run pytest tests/unit -q

# With integration (stub, no model download)
uv run pytest tests/unit tests/integration -q

# Single red test while developing
uv run pytest tests/unit/test_config.py::test_rejects_unknown_schema -q
```

---

## Definition of “TDD-clean” for closing #5

- [ ] Characterization C1–C5 green
- [ ] Forward T4, T6 green (orchestrator edge cases + no-mlflow)
- [ ] No new modules without tests in the same PR
- [ ] `docs/TASKS.md` Phase 1 checkboxes only flipped when backed by tests + smoke

This document is the process authority for coding style; HLD remains the product authority for *what* to build.
