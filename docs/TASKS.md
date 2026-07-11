# Task Breakdown — MLX Inference Workbench

> Derived from [HLD.md](HLD.md) | Created: 2026-07-08  
> GitHub milestone: [Local inference gaps (HLD-scoped)](https://github.com/weklund/mlx-inference-workbench/milestone/1) · Epic [#4](https://github.com/weklund/mlx-inference-workbench/issues/4)

---

## GitHub roadmap (gap map × HLD)

Issues below are the **source of truth** for purpose, definition of done, and smoke tests. HLD §7 exclusions (continuous batching, multi-user serving, NVIDIA, multimodal images, training/distill, DSPy product work, multi-Mac TP/PP, cascade product routing) are **not** in this milestone.

| Order | Issue | Role |
|------:|-------|------|
| 0 | [#2](https://github.com/weklund/mlx-inference-workbench/issues/2) Phase 0 MTPLX familiarization | Done |
| 1 | [#3](https://github.com/weklund/mlx-inference-workbench/issues/3) Phase 0.5 Thermal gate | HARD GATE |
| 2 | [#5](https://github.com/weklund/mlx-inference-workbench/issues/5) Phase 1 MVP harness | **Done** (MVP; see issue for residuals) |
| 2 | [#6](https://github.com/weklund/mlx-inference-workbench/issues/6) Prompt dataset v1 | Setup |
| 3 | [#7](https://github.com/weklund/mlx-inference-workbench/issues/7) mlx-lm engine + baseline | Setup (engine shipped; provisional baseline config; official numbers wait on #3) |
| 3 | [#8](https://github.com/weklund/mlx-inference-workbench/issues/8) M5 Max ceilings | Roofline gate (probe + profile + util API) |
| 4 | [#9](https://github.com/weklund/mlx-inference-workbench/issues/9) MTPLX engine | Setup |
| 4 | [#15](https://github.com/weklund/mlx-inference-workbench/issues/15) llama.cpp engine | Setup |
| 5 | [#10](https://github.com/weklund/mlx-inference-workbench/issues/10) EXP Prefix cache | Experiment |
| 5 | [#11](https://github.com/weklund/mlx-inference-workbench/issues/11) EXP Free draft | Experiment |
| 5 | [#12](https://github.com/weklund/mlx-inference-workbench/issues/12) EXP KV compression | Experiment |
| 5 | [#13](https://github.com/weklund/mlx-inference-workbench/issues/13) EXP Quant matrix | Experiment |
| 6 | [#14](https://github.com/weklund/mlx-inference-workbench/issues/14) EXP Adaptive draft | Experiment |

---

## Phase 0: Setup & MTPLX Familiarization

### Setup
- [x] Create repo with folder structure
- [x] Write root README.md with phased roadmap
- [x] Write experiments/README.md with spike template
- [x] Write .gitignore (models, results, Metal artifacts, environments)
- [x] Create pyproject.toml with uv-managed dependencies
- [x] Scaffold benchmarks/ package (harness.py, metrics.py, run_benchmark.py)
- [x] Scaffold kernels/ package (metal/, python/, experiments/)
- [x] Scaffold src/ package (utils/, profiling/, hardware/)
- [x] Create experiment folders (00_mtplx_familiarization, 01_baseline_m5max, 02_stock_mlx_comparison, 03_custom_metal_kernels)
- [x] Write HLD for benchmarking harness (full rigor, reviewed, diagrams generated)
- [x] Generate architecture diagrams (context, container, primary-flow sequence, error-scenario sequence)

### MTPLX Familiarization (Experiment 00)
- [ ] Install MTPLX (`brew install youssofal/mtplx/mtplx` or pip)
- [ ] Explore repo structure — locate native_extensions/, vllm_metal/, MLX fork
- [ ] Identify and document the 4 custom Metal kernels (verify_qmv, GDN, innovation-tape, capture/replay)
- [ ] Map the MTP draft → verify → accept/reject flow end-to-end
- [ ] Understand how kernels register as MLX primitives (C++/Metal interface)
- [ ] Run `mtplx models`, `mtplx bench`, `mtplx forge`
- [ ] Write experiment 00 README with findings (key questions answered)
- [ ] Identify 1-2 places where a new custom kernel could help

---

## Phase 0.5: Thermal Reproducibility Validation (HARD GATE)

> **Status (2026-07-10):** **PASS** — protocol CoV **2.57%** (N=25). Report: [`docs/spikes/005_thermal_reproducibility.md`](spikes/005_thermal_reproducibility.md). Issue [#3](https://github.com/weklund/mlx-inference-workbench/issues/3).

- [x] Select test model (mlx-lm, Qwen2.5-Coder-7B-Instruct-4bit, fixed prompt)
- [x] Multi-session measurements over 2 days (morning/afternoon/evening where protocol-clean)
- [x] Compute inter-run CoV for tok/s (protocol gate)
- [x] CoV < 5% under protocol (2.57%); exclusive-use + exclude contended sessions
- [x] Write thermal reproducibility report with validated methodology
- [x] Update HLD §22 with empirical findings
- [ ] Close GitHub issue #3 after this report lands on `main`
- [ ] Official baseline under protocol → [#36](https://github.com/weklund/mlx-inference-workbench/issues/36)

---

## Phase 1: MVP (Benchmarking Harness)

> **Status (2026-07-10):** MVP harness **shipped** on `main` (audit #24).  
> Evidence: PRs #16, #18, #19, #25–#27, #29. Epic issue [#5](https://github.com/weklund/mlx-inference-workbench/issues/5).  
> Not part of harness MVP: official baselines (#3 thermal report, #6 dataset, #7 mlx-lm run, #8 roofline).

### Core Infrastructure
- [x] Implement Orchestrator (Python CLI with Click)
  - [x] `bench run <config.yaml>` command
  - [x] `bench compare <run_a> <run_b>` command
  - [x] `bench list` command
  - [ ] `bench validate <run_id>` command (stub only — residual)
  - [x] `bench report <run_id>` command
  - [ ] `bench rebuild-index` command (rebuild MLflow from Parquet — residual)
- [x] Implement Hardware Probe
  - [x] Capture chip model, memory (sysctl / fingerprint)
  - [x] Capture OS version, library versions, git SHA
  - [ ] Capture ambient temperature (if sensor available) — residual / optional
  - [x] Capture baseline thermal state (via ThermalSensor)
- [x] Implement Prompt Manager
  - [x] Load JSONL datasets
  - [x] SHA-256 verification against manifest
  - [x] Abort on hash mismatch
  - [x] Optional `reference` field for correctness gate (#19)
- [x] Implement Thermal Monitor
  - [x] Parse `powermetrics` output (power / pressure when available)
  - [x] Detect throttling (primary: pressure; fallback: timing anomaly >2× median)
  - [x] Flag tainted iterations with `GenerationStatus.THERMAL_TAINTED`
  - [x] `note_duration` on protocol (#25); degraded history heuristic
  - [x] Handle unavailable powermetrics (degraded / off modes)
- [x] Implement Metrics Computer
  - [x] p50, p90, p95, p99 percentiles
  - [x] Mean, trimmed mean (drop top 1%), std dev
  - [x] Coefficient of variation + flag if > threshold (default 5%)
  - [x] Bandwidth utilization % helper + profile ceilings (#8); wire into per-run metrics later if needed
  - [ ] Power consumption / energy per token as first-class metrics — residual (fields exist, not fully wired)
  - [x] Acceptance rate fields optional/null for non-speculative
  - [x] E2e-only non-stream path does not invent TTFT/decode/SITL (#19)
- [x] Implement Comparability Gate
  - [x] Check: prompt dataset hash match
  - [x] Check: hardware profile match
  - [x] Check: metrics schema version match
  - [x] Check: thermal monitoring mode comparability
  - [ ] Check: ambient temperature within ±5°C (if logged) — residual
  - [x] Check: key metadata equality (schema, engine interface, etc.)
  - [x] Block with specific violation list OR pass to comparator
- [x] Implement Statistical Comparator
  - [x] Normality-aware path (Welch / Mann-Whitney)
  - [x] Effect size (Cohen's d)
  - [x] 95% confidence interval on difference
  - [x] Verdict with fail-closed allowlist (`DISTRIBUTION_METRIC_NAMES`, #27)
- [x] Implement MLflow Reporter (optional via `enable_mlflow`)
  - [x] Log params / metrics / tags when enabled
  - [ ] Full artifact suite (plots, thermal log) — residual / partial
  - [x] Tag runs (stable/unstable, quality, backend, …)
- [x] Implement run store (write-first, atomic summary)
  - [x] iterations JSONL + Parquet + summary.json
  - [x] Per-iteration metadata (thermal, memory, status)
  - [x] Schema versioned (`METRICS_SCHEMA_VERSION`, etc.)
- [x] Hygiene: strict config bools (#26), coverage gate + Makefile (#29)

### Engine Interface + First Backend
- [x] Define Engine ABC + GenParams / timed generate (orchestrator-owned timeout, #18)
- [x] Define GenerationResult dataclass with GenerationStatus enum
- [x] Stub engine for CI / smoke
- [x] Engine load / generation error paths (as used by registry + orchestrator)
- [x] Correctness scoring hook + dataset `reference` / `require_correctness` (#19)
  - [ ] Tiered KL-divergence fallback — residual / future
- [x] mlx-lm engine plugin (load + stream/e2e generate; seed + MLX peak memory)
- [x] Per-iteration timeout (`per_iteration_timeout_sec`) via orchestrator timed_generate
- [x] Provisional baseline config: `configs/experiments/baseline_mlx_lm.yaml` + tiny smoke
- [x] First provisional measured run `9f8c7f967277` + fill `experiments/01_baseline_m5max/README.md` results table
- [ ] Official baseline after thermal gate [#3](https://github.com/weklund/mlx-inference-workbench/issues/3)

### Experiment Config
- [x] Define config YAML schema (schema_version, experiment, hardware, model, benchmark, metrics, reproducibility)
- [x] Validate configs against declared schema version + strict bools (#26)
- [x] Create first hardware profile: `m5_max_128gb.yaml`
- [x] Create default/smoke benchmark config (`configs/experiments/smoke_minimal.yaml`)
- [x] mlx-lm tiny smoke + provisional baseline YAMLs (#7)

### Prompt Dataset
- [x] Smoke dataset + SHA-256 (`datasets/smoke_v1.jsonl`) for harness CI
- [x] Curate ≥20 agentic coding prompts — `datasets/agentic_coding_v1.jsonl` ([#6](https://github.com/weklund/mlx-inference-workbench/issues/6))
- [x] Document versioning + checksum policy in `datasets/README.md`

### M5 Max Hardware Verification (HARD GATE for roofline) — #8
- [x] Find Apple's published M5 Max specs (memory bandwidth, GPU cores; FP32 not published)
- [x] Run STREAM-equivalent (MLX triad) for empirical memory bandwidth
- [x] Run peak-FLOPS stand-in (MLX FP32 matmul) for empirical compute ceiling
- [x] Compare empirical vs. published — conservative = min where both exist
- [x] Update Appendix B in HLD with methodology + sources
- [x] Update hardware profile YAML (`make hardware-ceilings-write`)
- [x] `bandwidth_utilization_pct` + profile loader for harness use
- [x] Re-verify procedure: `make hardware-ceilings` / `hardware-ceilings-write`

### Testing
- [ ] Unit tests: Metrics Computer (known inputs → expected outputs)
- [ ] Unit tests: Comparability Gate (valid pairs pass, invalid pairs block)
- [ ] Unit tests: Statistical Comparator (known distributions → correct p-values)
- [ ] Unit tests: Prompt Manager (SHA-256 catches tampering)
- [x] Integration test: mlx-lm engine contract (implements full interface; mocked weights)
- [ ] Integration test: MLflow round-trip (log + retrieve)
- [ ] E2E test: minimal experiment (small model, 2 warmup, 3 iterations) — use `make smoke-mlx-tiny` (gpu/local)
- [ ] E2E test: comparison of two identical runs → "no significant difference"
- [ ] E2E test: Comparability Gate blocks incompatible runs

### Resolve Open Questions
- [ ] OQ#1: Can MTPLX be driven programmatically?
- [ ] OQ#3: Does `powermetrics` require sudo? What's the fallback?
- [ ] OQ#4: Verify M5 Max specs (see hardware verification above)
- [ ] OQ#6: Rust → MLflow integration approach (resolve before Phase 3)
- [ ] OQ#7: Memory peak comparability methodology

---

## Phase 2: Backend Expansion (~1-2 weeks per backend)

- [ ] Implement MTPLX engine plugin (v1)
  - [ ] Speculative-specific metrics: acceptance rate, mean accepted length
  - [ ] Draft depth configuration
- [ ] Implement llama.cpp engine plugin (v1)
  - [ ] Subprocess management with structured output parsing
  - [ ] Per-iteration timeout with process kill
- [ ] Implement BaseRT engine plugin (v1) — pending API availability
- [ ] Implement mlx-vlm text mode engine plugin (v1)
- [ ] Expand prompt dataset to ~50-100 prompts
- [ ] Run full baseline comparison: all backends × key models × quantizations
- [ ] Write experiment 01 README with baseline results
- [ ] Write experiment 02 README with stock MLX comparison

---

## Phase 3: Rust Kernel Layer (~2-4 weeks)

- [x] Initialize Rust workspace (Cargo.toml at repo root) — seeded by `metal_stream` (#8)
- [x] Metal STREAM bandwidth ceiling crate (`crates/metal_stream`, metal-rs host + MSL)
- [ ] Criterion.rs micro-bench harness for product kernels
- [ ] PyO3 bridge for kernel results → Python/MLflow
- [ ] Implement reference (naive) matmul for correctness baseline
- [ ] Implement first Metal shader (tiled quantized matmul)
- [ ] Criterion micro-benchmarks with kernel sweep configs (M/N/K dimensions)
- [ ] Correctness tests: numerical accuracy vs. reference (tolerance: abs 1e-3, rel 1e-2)
- [ ] Determinism tests: same input → same output
- [ ] Build PyO3 bindings (maturin develop)
- [ ] Implement Rust → MLflow bridge (per OQ#6 resolution)
- [ ] Implement Custom Kernel Engine plugin
- [ ] Roofline analyzer (plot against verified M5 Max ceilings)
- [ ] Run end-to-end: custom kernel plugged into pipeline, compare against baseline
- [ ] Write experiment 03 README

---

## Phase 4: Analysis & Advanced Features (ongoing)

- [ ] Kernel regression tracking across commits
- [ ] Automated roofline plots per kernel version
- [ ] Run-to-run stability tracking over time
- [ ] Comparison reports with effect size + practical significance
- [ ] Power-aware optimization tracking (tokens/watt trends)

---

## Phase 3+ Optimization Spikes (Priority Order)

### Spike A: Skew-Aware / Hot-Path Specialization (est. 15-35%)
- [ ] Profile token entropy/confidence distribution during inference
- [ ] Identify high-confidence vs. high-entropy regions in decode
- [ ] Prototype confidence-aware fast path in custom Metal kernel
- [ ] Benchmark against baseline (measure improvement)

### Spike B: Adaptive Self-Tuning Controller (est. 10-30%)
- [ ] Implement runtime statistics collection (acceptance rate, entropy, thermal)
- [ ] Design feedback loop for dynamic draft depth / strategy selection
- [ ] Prototype controller in Python/MLX
- [ ] Benchmark across workloads (short/long context, varying complexity)

### Spike C: Predictive Thermal Management
- [ ] Extend MTPLX fan control integration into closed-loop controller
- [ ] Implement predictive model (workload → thermal trajectory)
- [ ] Modulate speculative aggressiveness based on thermal headroom
- [ ] Measure sustained throughput over 30-60 min sessions

### Spike D: Value/Outcome Prediction (est. 20-50%)
- [ ] Design lightweight acceptance-length predictor
- [ ] Train on historical speculative decoding traces
- [ ] Integrate into draft/verify loop
- [ ] Benchmark against fixed-depth and heuristic approaches

### Spike E: Precise Agent Rollback Semantics
- [ ] Extend innovation-tape concept to tool/side-effect state
- [ ] Design delta-based rollback for agent trajectories
- [ ] Prototype in custom Metal kernel or MLX extension
- [ ] Validate exact rollback correctness

### Spike F: Durable Recovery (WAL-style)
- [ ] Design lightweight persistent logging layer on top of innovation-tape
- [ ] Implement delta-logging to disk (minimal overhead)
- [ ] Implement crash recovery + resume from last checkpoint
- [ ] Validate with simulated crash scenarios

---

## Future (Out of Scope for Now)

- [ ] NVIDIA / cloud GPU testing
- [ ] DSPy / prompt optimization
- [ ] Self-evolving agent prompt datasets
- [ ] Validation that synthetic prompts match real agent workload characteristics
- [ ] ADR for layered architecture decision (extract from HLD §10.4)
