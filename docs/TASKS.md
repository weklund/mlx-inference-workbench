# Task Breakdown — MLX Inference Workbench

> Derived from [HLD.md](HLD.md) | Created: 2026-07-08

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

- [ ] Select test model (mlx-lm, Qwen3 or similar, fixed prompt)
- [ ] Run 20 measurements spread across 2 days (morning/afternoon/evening)
- [ ] Compute inter-run CoV for decode tok/s
- [ ] If CoV < 5%: document methodology, proceed to Phase 1
- [ ] If CoV > 5%: iterate (increase cooldown, restrict time-of-day, log ambient temp, statistical normalization)
- [ ] Write thermal reproducibility report with validated methodology
- [ ] Update HLD §22 with empirical findings

---

## Phase 1: MVP (Benchmarking Harness)

### Core Infrastructure
- [ ] Implement Orchestrator (Python CLI with Click/Typer)
  - [ ] `bench run <config.yaml>` command
  - [ ] `bench compare <run_a> <run_b>` command
  - [ ] `bench list` command
  - [ ] `bench validate <run_id>` command
  - [ ] `bench report <run_id>` command
  - [ ] `bench rebuild-index` command (rebuild MLflow from Parquet)
- [ ] Implement Hardware Probe
  - [ ] Capture chip model, GPU cores, memory (sysctl, system_profiler)
  - [ ] Capture OS version, library versions, git SHA
  - [ ] Capture ambient temperature (if sensor available)
  - [ ] Capture baseline thermal state
- [ ] Implement Prompt Manager
  - [ ] Load JSONL datasets
  - [ ] SHA-256 verification against manifest
  - [ ] Abort on hash mismatch
- [ ] Implement Thermal Monitor
  - [ ] Parse `powermetrics` output for chip temperature
  - [ ] Detect throttling (primary: powermetrics, fallback: timing anomaly >2× median)
  - [ ] Flag tainted iterations with `GenerationStatus.THERMAL_TAINTED`
  - [ ] Log thermal stability metric (intra-run variance)
  - [ ] Handle unavailable powermetrics (fallback mode with degraded flag)
- [ ] Implement Metrics Computer
  - [ ] p50, p90, p95, p99 percentiles
  - [ ] Mean, trimmed mean (drop top 1%), std dev
  - [ ] Coefficient of variation + flag if >5%
  - [ ] Bandwidth utilization % (after hardware specs verified)
  - [ ] Power consumption (watts) and energy per token (joules/token)
  - [ ] Acceptance rate + mean accepted length (speculative backends)
- [ ] Implement Comparability Gate
  - [ ] Check: prompt dataset hash match
  - [ ] Check: hardware profile match
  - [ ] Check: metrics schema version match
  - [ ] Check: thermal stability similarity
  - [ ] Check: ambient temperature within ±5°C (if logged)
  - [ ] Check: OS/library versions compatible
  - [ ] Check: neither run had degraded thermal monitoring
  - [ ] Block with specific violation list OR pass to comparator
- [ ] Implement Statistical Comparator
  - [ ] Shapiro-Wilk normality test
  - [ ] Welch's t-test (normal distributions)
  - [ ] Mann-Whitney U (non-normal distributions)
  - [ ] Effect size (Cohen's d)
  - [ ] 95% confidence interval on difference
  - [ ] Verdict with practical significance annotation
- [ ] Implement MLflow Reporter
  - [ ] Log params (hardware, config, prompt hash, library versions, schema version)
  - [ ] Log metrics (all computed statistics)
  - [ ] Log artifacts (raw Parquet, distribution plots, thermal log, summary JSON)
  - [ ] Tag runs (stable/unstable, quality level, backend, model, quantization)
- [ ] Implement Parquet data store (write-first, atomic)
  - [ ] Per-token timestamps for all iterations
  - [ ] Per-iteration metadata (thermal state, memory, status)
  - [ ] Schema versioned

### Engine Interface + First Backend
- [ ] Define Engine ABC (load_model, warmup, generate, supports_speculative, get_memory_usage, validate_correctness)
- [ ] Define GenerationResult dataclass with GenerationStatus enum
- [ ] Define EngineLoadError, GenerationError exceptions
- [ ] Implement correctness gate (tiered: bitwise at temp=0, KL-divergence fallback)
- [ ] Implement mlx-lm engine plugin (v1)
  - [ ] load_model via mlx-lm API
  - [ ] warmup (N iterations discarded)
  - [ ] generate with per-token timestamps
  - [ ] validate_correctness at temp=0/fixed seed
  - [ ] get_memory_usage
- [ ] Per-iteration timeout (`per_iteration_timeout_sec`) with kill on timeout

### Experiment Config
- [ ] Define config YAML schema (schema_version, experiment, hardware, model, benchmark, metrics, reproducibility)
- [ ] Validate configs against declared schema version
- [ ] Create first hardware profile: `m5_max_128gb.yaml`
- [ ] Create default benchmark config

### Prompt Dataset
- [ ] Curate ~20 agentic coding prompts (tool calls, multi-turn reasoning, code generation)
- [ ] Save as JSONL with id, category, prompt, expected_tokens_approx
- [ ] Generate SHA-256 checksum file
- [ ] Document dataset v1 in datasets/README.md

### M5 Max Hardware Verification (HARD GATE for roofline)
- [ ] Find Apple's published M5 Max specs (memory bandwidth, GPU cores, TFLOPS)
- [ ] Run STREAM benchmark (or equivalent) for empirical memory bandwidth
- [ ] Run peak-FLOPS micro-kernel for empirical GPU compute ceiling
- [ ] Compare empirical vs. published — use conservative (lower) value
- [ ] Update Appendix B in HLD with verified numbers
- [ ] Update hardware profile YAML

### Testing
- [ ] Unit tests: Metrics Computer (known inputs → expected outputs)
- [ ] Unit tests: Comparability Gate (valid pairs pass, invalid pairs block)
- [ ] Unit tests: Statistical Comparator (known distributions → correct p-values)
- [ ] Unit tests: Prompt Manager (SHA-256 catches tampering)
- [ ] Integration test: mlx-lm engine contract (implements full interface)
- [ ] Integration test: MLflow round-trip (log + retrieve)
- [ ] E2E test: minimal experiment (small model, 2 warmup, 3 iterations)
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

- [ ] Initialize Rust workspace (Cargo.toml at repo root)
- [ ] Set up metal-rs + Criterion.rs + PyO3 dependencies
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
