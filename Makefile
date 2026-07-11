# Developer + CI task entrypoints. Prefer `make <target>` over copying long recipes.
# Thresholds live in pyproject.toml / scripts/check_core_coverage.py — not re-encoded here.

.DEFAULT_GOAL := help
SHELL := /bin/bash
.SHELLFLAGS := -euo pipefail -c

UV ?= uv
PYTEST_MARK_CI := not gpu and not slow
COV_JSON := coverage.json
SMOKE_CONFIG := configs/experiments/smoke_minimal.yaml
AGENTIC_STUB_CONFIG := configs/experiments/agentic_coding_v1_stub.yaml
SMOKE_MLX_TINY_CONFIG := configs/experiments/smoke_mlx_lm_tiny.yaml
SMOKE_MTPLX_TINY_CONFIG := configs/experiments/smoke_mtplx_tiny.yaml
BASELINE_MLX_CONFIG := configs/experiments/baseline_mlx_lm.yaml

# Thermal (override: make thermal-run SESSION=morning DAY=2 RUNS=5)
SESSION ?=
DAY ?=
RUNS ?= 5

.PHONY: help sync sync-update lint fmt test test-unit test-integration \
	coverage ci smoke smoke-agentic smoke-mlx-tiny smoke-mtplx-tiny baseline-mlx-lm \
	bench-list bench-compare bench-report \
	thermal-run thermal-analyze thermal-analyze-protocol \
	hardware-ceilings hardware-ceilings-write metal-stream \
	lint-rust test-rust ci-rust clean

help: ## Show available targets
	@grep -E '^[a-zA-Z0-9_-]+:.*?## ' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  %-26s %s\n", $$1, $$2}'

# --- setup -------------------------------------------------------------------

sync: ## Install deps (frozen + dev extras; CI-like)
	$(UV) sync --frozen --extra dev

sync-update: ## Refresh lockfile and install (local; not for CI)
	$(UV) sync --extra dev

# --- quality (CI parity) -----------------------------------------------------

lint: ## Ruff check + format check (CI)
	$(UV) run ruff check .
	$(UV) run ruff format --check .

fmt: ## Auto-format with ruff (local)
	$(UV) run ruff format .
	$(UV) run ruff check --fix . || true

test: test-unit ## Fast unit suite (alias)

test-unit: ## Unit tests excluding gpu/slow markers
	$(UV) run pytest tests/unit -m "$(PYTEST_MARK_CI)" --tb=short -q

test-integration: ## Integration tests (exit 5 = no tests collected → ok)
	@set +e; \
	$(UV) run pytest -m integration --tb=short -q; \
	ec=$$?; \
	if [ $$ec -eq 0 ] || [ $$ec -eq 5 ]; then exit 0; fi; \
	exit $$ec

coverage: ## Unit + overall cov gate + core module floors (merge gate)
	$(UV) run pytest -m "$(PYTEST_MARK_CI)" --tb=short \
		--cov=workbench \
		--cov-report=term-missing \
		--cov-report=json:$(COV_JSON) \
		--cov-fail-under=80
	$(UV) run python scripts/check_core_coverage.py $(COV_JSON)

ci: lint coverage ## Local mirror of required Python CI gates
	@echo "ci: lint + coverage OK"

# --- bench / harness ---------------------------------------------------------

smoke: ## Run stub smoke experiment end-to-end
	$(UV) run bench run $(SMOKE_CONFIG)

smoke-agentic: ## Stub run pinned to agentic_coding_v1 dataset
	$(UV) run bench run $(AGENTIC_STUB_CONFIG)

smoke-mlx-tiny: ## Real mlx-lm smoke (tiny Qwen3 0.6B-4bit; requires Metal + weights)
	$(UV) run bench run $(SMOKE_MLX_TINY_CONFIG)

smoke-mtplx-tiny: ## Real MTPLX smoke (Qwen3.5-0.8B MTP; requires --extra mtplx + Metal)
	$(UV) run bench run $(SMOKE_MTPLX_TINY_CONFIG)

baseline-mlx-lm: ## Protocol mlx-lm baseline (Qwen3-8B-4bit; exclusive session; warmup 10)
	$(UV) run bench run $(BASELINE_MLX_CONFIG)

bench-list: ## List stored benchmark runs
	$(UV) run bench list

bench-compare: ## Compare two runs: make bench-compare A=<id> B=<id>
	@test -n "$(A)" || (echo "usage: make bench-compare A=<run_a> B=<run_b>" >&2; exit 2)
	@test -n "$(B)" || (echo "usage: make bench-compare A=<run_a> B=<run_b>" >&2; exit 2)
	$(UV) run bench compare $(A) $(B)

bench-report: ## Report one run: make bench-report RUN=<id>
	@test -n "$(RUN)" || (echo "usage: make bench-report RUN=<run_id>" >&2; exit 2)
	$(UV) run bench report $(RUN)

# --- phase 0.5 thermal -------------------------------------------------------

thermal-run: ## Validation session: SESSION=morning|afternoon|evening DAY=1|2 [RUNS=5]
	@test -n "$(SESSION)" || (echo "usage: make thermal-run SESSION=morning DAY=2 [RUNS=5]" >&2; exit 2)
	@test -n "$(DAY)" || (echo "usage: make thermal-run SESSION=morning DAY=2 [RUNS=5]" >&2; exit 2)
	$(UV) run python scripts/thermal_validation.py \
		--runs $(RUNS) --session $(SESSION) --day $(DAY)

thermal-analyze: ## Analyze all thermal cohorts
	$(UV) run python scripts/thermal_analysis.py

thermal-analyze-protocol: ## Analyze protocol gate only (AC + powermetrics)
	$(UV) run python scripts/thermal_analysis.py --valid-only

# --- hardware ceilings / roofline (#8) ---------------------------------------

hardware-ceilings: ## Published vs empirical M5 Max BW/FLOPS (Metal STREAM + MLX)
	$(UV) run python scripts/verify_m5_max_ceilings.py

hardware-ceilings-write: ## Run probes and update configs/hardware/m5_max_128gb.yaml
	$(UV) run python scripts/verify_m5_max_ceilings.py --write

metal-stream: ## Rust + MSL STREAM bandwidth ceiling only (release)
	cargo run -p metal_stream --release

lint-rust: ## rustfmt + clippy -D warnings + rustdoc -D warnings
	cargo fmt --all -- --check
	cargo clippy --workspace --all-targets --all-features -- -D warnings
	@# Public docs are part of the quality bar (missing_docs + rustdoc lints).
	RUSTDOCFLAGS="-D warnings" cargo doc --workspace --no-deps

test-rust: ## cargo test workspace (includes Metal host-oracle tests on macOS)
	cargo test --workspace

ci-rust: lint-rust test-rust ## Local mirror of Rust CI gates
	@echo "ci-rust: fmt + clippy + doc + test OK"

# --- hygiene -----------------------------------------------------------------

clean: ## Remove coverage and pytest caches
	rm -f .coverage $(COV_JSON)
	rm -rf htmlcov .pytest_cache
	find . -type d -name __pycache__ -not -path './.venv/*' -exec rm -rf {} + 2>/dev/null || true
