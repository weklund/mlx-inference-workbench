# Coverage gate — workbench scientific core

Issue: [#23](https://github.com/weklund/mlx-inference-workbench/issues/23)  
Parent: [#5](https://github.com/weklund/mlx-inference-workbench/issues/5)

## What is gated

| In gate | Out of gate (for now) |
|---------|------------------------|
| `src/workbench/**` (`--cov=workbench`) | `scripts/` (thermal spikes, analysis) |
| Overall package total **≥ 80%** | `docs/`, spikes JSONL |
| Per-module floors (see below) | `benchmarks/cli.py` pretty-print |
| | Integration-only / GPU / slow markers |

Day-one goal is **measured floors**, not a blunt 90% on the whole repo. Raise overall toward **90%** on core as tests grow.

## Local commands

```bash
# Full unit suite + term report (uses [tool.coverage] fail_under=80)
uv run pytest tests/unit -m "not gpu and not slow" \
  --cov=workbench --cov-report=term-missing -q

# Same + JSON for per-module floors (matches CI)
uv run pytest tests/unit -m "not gpu and not slow" \
  --cov=workbench \
  --cov-report=term-missing \
  --cov-report=json:coverage.json -q
uv run python scripts/check_core_coverage.py
```

CI runs the second form on every PR as job **`Test + Coverage (Python)`**
(`.github/workflows/ci.yml`).

### Merge protection

`main` requires these status checks (strict, include admins):

| Check | What fails the merge |
|-------|----------------------|
| **Test + Coverage (Python)** | Unit test failure, overall coverage &lt; 80%, or core module floor miss |
| **Lint (Python)** | ruff check / format |

If coverage drops, that job exits non-zero → check red → **PR cannot merge**.

## Interpreting results

- **Overall ≥ 80%** — `coverage` / pytest-cov `fail_under` (configured in `pyproject.toml`).
- **Core floors** — `scripts/check_core_coverage.py` fails if a listed module drops below its starting floor.
- **Missing lines** — `term-missing` column; prefer behavioral tests over chasing 100% on dead branches.
- **Low modules today** — `thermal.py` (powermetrics subprocess path) and parts of `mlx_lm_engine.py` are hardware-heavy; floors start lower (~50–60%).

## Starting floors (raise over time)

| Module | Floor |
|--------|------:|
| `orchestrator`, `comparability`, `config` | 80% |
| `metrics`, `statistics_compare`, `prompts`, `run_store`, engine base/timeout/registry/stub, `hardware` | 70% |
| `mlx_lm_engine` | 60% |
| `thermal` | 45% (powermetrics path; branch-aware) |

Edit floors in `scripts/check_core_coverage.py` when intentionally raising the bar.

## What not to do

- Do not add `--cov=benchmarks` to the hard gate until CLI has dedicated behavioral coverage.
- Do not require 90% on thermal spike scripts.
- Prefer failing tests that lock contracts over coverage-only line filling.
