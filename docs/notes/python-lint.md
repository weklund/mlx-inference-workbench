# Python lint & documentation policy

> Quality bar for `src/workbench`, `benchmarks`, and `scripts`.  
> Local + CI: `make lint` / `make ci`. Pre-commit: ruff + ruff-format.

## Tools

| Tool | Role | Config |
|------|------|--------|
| **Ruff check** | Lint (explicit rule set in `pyproject.toml`) | `[tool.ruff.lint]` |
| **Ruff format** | Style / wrapping | `[tool.ruff.format]` |
| **pytest + coverage** | Behavior + core floors | `[tool.coverage.*]`, `scripts/check_core_coverage.py` |

Pin **ruff-pre-commit** to the same major.minor as `uv run ruff --version` (see `.pre-commit-config.yaml`).

## Rule philosophy

1. **Explicit `select`** — do not use `ALL` (new rules on upgrade surprise CI).
2. **Scientific exceptions** — lazy imports for optional backends (`PLC0415`), magic thresholds (`PLR2004`), exception-message style (`TRY003`) are intentionally off or scoped.
3. **Docs are part of the gate** — Google-style pydocstyle (`D`) on product code; tests/scripts relaxed via `per-file-ignores`.
4. **No prints in library code** — `T20` on `src/workbench`; CLI/scripts may print.

## Docstring standard (Google)

| Surface | Required |
|---------|----------|
| Module | 1–4 lines: responsibility + non-goals |
| Public class | Purpose; non-obvious fields called out |
| Public function / method | Summary; `Args` / `Returns` / `Raises` when non-obvious |
| Private `_helper` | Only when behavior is subtle |

Examples of contracts that **must** stay documented: `Engine` ABC, `GenerationResult` field honesty (stream vs e2e), `check_comparable`, `compute_distribution`, config schema loaders.

## Commands

```bash
make lint          # ruff check + format --check
make fmt           # format + ruff --fix
make ci            # lint + coverage gates
uv run ruff check src --select D   # docstring-only
```

## Adding rules

1. Add the code to `select` in `pyproject.toml`.
2. Run `uv run ruff check .` and fix or justify `ignore` / `per-file-ignores`.
3. Update this note if the philosophy changes.
