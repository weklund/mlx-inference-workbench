# HLD: MLX Inference Workbench — Benchmarking Harness

| Field | Value |
|-------|-------|
| Status | Draft |
| Last updated | 2026-07-08 |

## 1. Overview

**Problem statement.** We need a statistically rigorous benchmarking harness that produces trustworthy, comparable measurements of LLM text inference across backends, models, quantizations, and custom compute kernels on Apple Silicon — because the current landscape of benchmark claims is unscientific and nobody trusts anything.

**Objective.** Design a measurement system that makes it impossible to accidentally produce non-comparable results, enforces statistical rigor by default, and scales from kernel micro-benchmarks to end-to-end inference pipelines.

**Goal.** Enable systematic, evidence-based optimization of local LLM inference on M5 Max, working backwards from the goal of running capable coding agents locally on Apple Silicon with acceptable performance.

**Context & motivation.** MTPLX and BaseRT have demonstrated 1.5–2.2× gains via custom Metal kernels on Apple Silicon. The M5 Max (128 GB unified memory) is capable hardware. But without rigorous measurement infrastructure, it's impossible to distinguish real improvements from noise, thermal artifacts, or methodology drift. This harness is the scientific foundation for all subsequent optimization work.

## 2. Glossary

- **TTFT** — Time to first token (ms). Measures prefill + first decode step latency.
- **SITL** — Seconds inter-token latency. Time between first and last token divided by number of inter-token intervals.
- **E2E latency** — Total generation time from prompt submission to final token.
- **Bandwidth utilization** — Actual memory bandwidth consumed divided by theoretical hardware maximum. Primary efficiency metric for kernel work.
- **Roofline model** — Performance model plotting arithmetic intensity (FLOPS/byte) against throughput (GFLOPS), bounded by memory bandwidth and compute ceilings.
- **CoV** — Coefficient of variation (std dev / mean). Used to flag unstable measurements.
- **Trimmed mean** — Mean after dropping top 1% outliers (tm99). Reduces impact of thermal spikes.
- **MTP** — Multi-Token Prediction. Speculative decoding technique using a model's own prediction heads.
- **Acceptance rate** — Fraction of speculative draft tokens accepted by the target model during verification.
- **MLflow** — Experiment tracking framework used as the single source of truth for all benchmark results.
- **Criterion.rs** — Rust micro-benchmarking framework with built-in statistical analysis and regression detection.
- **MSL** — Metal Shading Language. Apple's GPU shader programming language.
- **PyO3** — Rust-Python interop library for building native Python modules from Rust code.

## 4. Requirements

### 4.1 Functional

- **FR1.** Run the same benchmark suite across multiple backends (MTPLX, mlx-lm, mlx-vlm text mode, llama.cpp, BaseRT, custom kernels) with identical prompts and configs.
- **FR2.** Capture per-run metrics: TTFT, decode tok/s, SITL, E2E latency, memory peak, bandwidth utilization %, acceptance rate (for speculative decoding backends), power consumption (watts), energy per token (joules/token).
- **FR3.** Report statistical distributions (p50, p90, p95, p99, trimmed mean, std dev) for every metric — never a single number.
- **FR4.** Track experiments with full reproducibility metadata: git SHA, library versions, hardware profile, OS version, thermal state at start/end, ambient temperature (if sensor available), thermal stability metric (intra-run variance), prompt dataset hash.
- **FR5.** Support two-level benchmarking: kernel micro-benchmarks (Rust/Criterion.rs, correctness validated before performance) and end-to-end macro-benchmarks (Python orchestration).
- **FR6.** Detect thermal throttling during measurement via `powermetrics` or equivalent. Discard affected iterations. Enforce configurable cooldown between runs.
- **FR7.** Compare any two runs with a statistical significance test. Report whether the difference is real (with confidence interval) or within noise.
- **FR8.** Log all runs to MLflow: parameters, metrics, artifacts (raw data, distribution plots). Raw Parquet files are the primary data store (ground truth); MLflow is the queryable index built from them. MLflow can be rebuilt from Parquet if corrupted.
- **FR9.** Validate output correctness before reporting performance. Run a correctness check (output comparison against a reference) as a prerequisite gate — a fast wrong answer is not an optimization.

### 4.2 Non-functional

- **NFR1.** Minimum repetitions per configuration: ≥5 runs (target 10) to establish statistical confidence.
- **NFR2.** Warmup iterations before measurement: ≥10 passes to eliminate JIT, memory allocation, and cold-start effects.
- **NFR3.** Inter-run variance threshold: flag configurations with >5% coefficient of variation as unstable. Require investigation before treating results as valid.
- **NFR4.** Cooldown between heavy runs: ≥30 seconds. Configurable per experiment.
- **NFR5.** Prompt dataset integrity: SHA-256 hashed and versioned. Immutable once used in a published experiment. Dataset changes require a new version with a new hash.

### 4.3 Technical

- **TR1.** Primary hardware target: MacBook Pro M5 Max, 128 GB unified memory.
- **TR2.** Python orchestration layer managed with uv (pyproject.toml, uv.lock).
- **TR3.** Rust workspace for kernel micro-benchmarks: Criterion.rs for statistical rigor, metal-rs for direct Metal GPU access, PyO3 for Python bindings.
- **TR4.** MLflow for experiment tracking, local file-based (`mlruns/` directory).
- **TR5.** Prompt datasets: agentic coding flows (tool calls, multi-turn reasoning, code generation). JSONL format with SHA-256 checksums.

## 5. Use Cases

- **UC1.** As a researcher, I run a baseline benchmark on a new model/backend so that I have a reference point for all future optimization work on that configuration.
- **UC2.** As a researcher, I compare two backends (e.g., MTPLX vs. stock mlx-lm) on the same model and prompts so that I can quantify which is faster and by how much, with statistical confidence.
- **UC3.** As a researcher, I write a custom Metal kernel and run kernel micro-benchmarks so that I can measure its performance (GFLOPS, bandwidth util, correctness) before integrating it into an inference pipeline.
- **UC4.** As a researcher, I plug a custom kernel into an end-to-end pipeline and re-run the macro-benchmark so that I can measure whether the kernel improvement translates to real tok/s gains.
- **UC5.** As a researcher, I review historical results in MLflow so that I can track progress across optimization spikes over weeks/months and detect regressions.
- **UC6.** As a community member, I reproduce a published result on my own Apple Silicon hardware so that I can validate claims and build on them.
- **UC7.** As a researcher, I attempt to compare two runs and the harness blocks me because conditions weren't controlled, so that I never accidentally draw conclusions from incomparable data.

## 6. Assumptions

1. MTPLX, mlx-lm, mlx-vlm, llama.cpp, and BaseRT can all be driven programmatically (via Python API or subprocess with structured output parsing) — not CLI-only with human-readable output.
2. Apple Silicon thermal behavior is reproducible with cooldown discipline and throttle detection. Discarding throttled iterations is sufficient to get clean measurements.
3. Memory bandwidth utilization and GPU occupancy metrics can be obtained from userspace — either via Metal performance counters, Xcode GPU profiling tools, or instrumented kernels.
4. Rust kernel micro-benchmarks (Criterion.rs) can log results to the same MLflow instance as Python macro-benchmarks, either directly or via a thin bridge.
5. Agentic coding prompts (tool calls, multi-turn reasoning, code generation) are a reasonable proxy for local coding agent workloads. To be validated in future phases — real agents have growing context accumulation, tool execution latency, and potentially multi-model routing that synthetic prompts don't capture. Representativeness is an assumption, not a claim.
6. The M5 Max's unified memory architecture means memory bandwidth is the primary bottleneck for most decode workloads — not compute. (To be validated in Phase 1 via roofline analysis.)

## 7. Out of Scope

- **NVIDIA / cloud GPU testing.** This harness targets Apple Silicon only. NVIDIA support may come later but is not designed for here.
- **VLM image/multimodal workloads.** Text-only inference. mlx-vlm is included only for its text generation capabilities.
- **Training or fine-tuning.** This is an inference measurement tool.
- **Production serving.** No continuous batching, multi-user, or load testing. Single-user, single-request benchmarking only.
- **DSPy / prompt optimization.** Future phase. The harness measures inference performance; prompt engineering is a separate concern.
- **Self-evolving agent prompt datasets.** Future phase. Initial datasets are manually curated.
- **Cross-platform portability.** Not designing for Linux/Windows. macOS-only, Apple Silicon-only.

## 8. Current System

*(Greenfield — no existing measurement infrastructure. Current state is ad-hoc CLI runs of `mtplx bench`, `mlx_lm.generate`, or `llama-bench` with no controlled conditions, no statistical rigor, and no comparability guarantees.)*

## 9. Design Tenets

When trade-offs must be made, resolve in this order (earlier wins):

1. **Comparability over speed.** A slower harness that produces trustworthy apples-to-apples results beats a fast one with caveats.
2. **Reproducibility over convenience.** Pin everything, hash everything, record everything, even if it's annoying.
3. **Correctness over performance claims.** Validate outputs before measuring speed. A fast wrong answer is not an optimization.
4. **Simplicity over generality.** Optimize for M5 Max + text LLMs, not a universal benchmarking framework.
5. **Extensibility over completeness.** Ship a working harness for one backend, make adding backends easy.

## 10. Proposed Design

### 10.1 Architecture

**Figure 1: System Context (C4 Level 1)** — [context.csv](HLD.diagrams/context.csv) | Last updated: 2026-07-08
> Shows the Benchmarking Harness as a single system with its actors (Researcher) and external dependencies (MLflow, LLM backends, Apple Silicon hardware). Legend: blue = system under design, white = actor, grey dashed = external system.

**Figure 2: Container Diagram (C4 Level 2)** — [container-proposed.csv](HLD.diagrams/container-proposed.csv) | Last updated: 2026-07-08
> Shows internal containers of the harness: Orchestrator, Hardware Probe, Prompt Manager, Thermal Monitor, Engine Interface (with 6 backend plugins), Metrics Computer, Comparability Gate, Rust Kernel Micro-bench, and MLflow. Legend: blue = orchestration layer, green = measurement/validation, yellow = monitoring/kernel, purple = backend engines, grey dashed = external (MLflow). Solid arrows = sync calls, dashed arrows = async logging.

The system is a **three-layer architecture** unified by MLflow as the tracking spine:

```
┌─────────────────────────────────────────────────────────────────┐
│                        MLflow Tracking                           │
│  (single source of truth: params, metrics, artifacts, lineage)  │
└──────────────────────────────┬──────────────────────────────────┘
                               │
        ┌──────────────────────┼──────────────────────┐
        │                      │                      │
┌───────▼────────┐   ┌────────▼────────┐   ┌────────▼────────┐
│  Python Layer  │   │   Rust Layer    │   │  Analysis Layer │
│  (Macro-bench) │   │  (Micro-bench)  │   │  (Comparison)   │
├────────────────┤   ├─────────────────┤   ├─────────────────┤
│ Runner/Orchestr│   │ Criterion.rs    │   │ Statistical     │
│ Engine plugins │   │ metal-rs shaders│   │ Roofline plots  │
│ Thermal monitor│   │ PyO3 bindings   │   │ Regression track│
│ Hardware probe │   │ Correctness     │   │ Run comparison  │
│ Metrics compute│   │   validation    │   │ Comparability   │
│ Prompt manager │   │                 │   │   gate          │
└───────┬────────┘   └────────┬────────┘   └────────┬────────┘
        │                      │                      │
        ▼                      ▼                      ▼
┌────────────────────────────────────────────────────────────────┐
│                     Backend Engines                              │
│  MTPLX │ mlx-lm │ mlx-vlm │ llama.cpp │ BaseRT │ Custom       │
└────────────────────────────────────────────────────────────────┘
```

### 10.2 Walkthrough

**Primary use case: Run a benchmark experiment and compare against baseline.**

1. Researcher writes an experiment config YAML specifying model, backend, quantization, prompt dataset, repetition count, and thermal settings.
2. `bench run <config.yaml>` invokes the Python orchestrator.
3. Hardware Probe captures system state: chip model, GPU core count, memory bandwidth (theoretical), OS version, library versions, git SHA of the workbench repo, and current thermal baseline.
4. Prompt Manager loads the dataset, verifies SHA-256 checksum against the manifest. Aborts if hash mismatch.
5. Thermal Monitor checks baseline chip temperature. If already in throttle state, aborts with a clear message.
6. Orchestrator opens an MLflow run and logs all parameters (hardware profile, config, prompt hash, library versions).
7. **Correctness gate:** Validates the backend produces correct output before any performance measurement. Uses a **tiered approach** based on temperature:
   - **Temperature 0 (greedy):** Bitwise exact match against stored reference output at fixed seed. Any divergence = abort. This catches model corruption, quantization bugs, and backend implementation errors.
   - **Temperature > 0 (sampling):** Run at temperature 0 with fixed seed for the correctness gate regardless of the experiment's configured temperature. The gate validates deterministic correctness; stochastic quality is a separate concern (not gated).
   - **Fallback if deterministic reproducibility fails** (some backends have non-deterministic parallel kernels even at temp=0): Compare against reference using token-level log-probability divergence. If mean KL divergence exceeds threshold (default: 0.01 nats/token), abort.
   
   The correctness gate always runs at temperature 0 with a fixed seed, even if the benchmark itself uses higher temperature. This separates "is the backend correct?" from "how fast is it at temperature X?".
8. **Warmup phase:** Execute N (≥10) iterations. Discard all results. Purpose: JIT compilation, memory allocation, Metal shader compilation, thermal stabilization.
9. **Measurement phase:** Execute M (≥5, typically 10) timed iterations. **Iteration strategy:** each iteration runs the full prompt dataset in fixed order (same order every iteration, determined by JSONL file order). Fixed ordering ensures that prompt-position effects (MLX prefix caching, shader warmth, thermal drift) are consistent across runs and cancel out in comparisons. Randomized ordering is a future option (configurable) but fixed-order is the default for maximum reproducibility. For each iteration:
   - Record per-token timestamps (for TTFT, SITL, tok/s).
   - Record memory peak.
   - Record thermal state at start and end.
   - If Thermal Monitor detects throttling mid-iteration, flag that iteration as tainted.
   - Wait cooldown period (≥30s) before next iteration.
10. **Metrics computation:** From the M iterations (minus any tainted ones):
    - Compute p50, p90, p95, p99 for each metric.
    - Compute mean, trimmed mean (drop top 1%), std dev.
    - Compute coefficient of variation. If CoV > 5%, flag as unstable.
    - For speculative backends: compute acceptance rate and mean accepted length.
    - For kernel-level runs: compute bandwidth utilization % and GFLOPS.
11. **Artifact generation:** Raw timing data (Parquet), distribution plots, summary JSON.
12. **MLflow logging:** Log all metrics, flag unstable results, attach artifacts.
13. **Comparison (when requested):** User runs `bench compare <run_a> <run_b>`.
    - Comparability Gate checks: same prompt dataset hash? Same hardware profile? Same metrics schema version? Compatible thermal conditions (both runs have similar thermal stability metric and neither had degraded monitoring)? Ambient temperature within ±5°C (if logged)? Same OS/library versions where material?
    - If violations found: block comparison, report specific violations.
    - If comparable: run Welch's t-test (or Mann-Whitney U for non-normal distributions). Report effect size, confidence interval, and verdict ("statistically significant improvement of X% ± Y%").

### 10.3 Components

| Component | Responsibility | Technology |
|-----------|----------------|------------|
| **Orchestrator** | Coordinates the full benchmark lifecycle (warmup → measure → compute → log) | Python, Click/Typer CLI |
| **Engine Interface** | Abstract base class that each backend implements (load model, generate, get metrics) | Python ABC |
| **Engine: mlx-lm** | Drives stock MLX inference via mlx-lm library | Python, mlx-lm API |
| **Engine: MTPLX** | Drives MTPLX with MTP speculative decoding | Python, mtplx API |
| **Engine: mlx-vlm** | Drives mlx-vlm text generation mode | Python, mlx-vlm API |
| **Engine: llama.cpp** | Drives llama.cpp Metal backend | Python subprocess + structured output parsing |
| **Engine: BaseRT** | Drives BaseRT native Metal runtime | Python subprocess or API (TBD) |
| **Engine: Custom Kernel** | Drives custom MLX/Metal kernels via PyO3 bindings | Python + Rust (PyO3) |
| **Hardware Probe** | Captures hardware specs, OS version, library versions, git SHA | Python, `sysctl`, `system_profiler` |
| **Thermal Monitor** | Monitors chip temperature, detects throttling, flags tainted iterations | Python, `powermetrics` (requires sudo or IOKit) |
| **Prompt Manager** | Loads, validates (SHA-256), and serves prompt datasets | Python, JSONL |
| **Metrics Computer** | Computes statistical distributions from raw timing data | Python, NumPy, SciPy |
| **Comparability Gate** | Validates that two runs are comparable before allowing statistical comparison | Python |
| **Statistical Comparator** | Welch's t-test / Mann-Whitney U, confidence intervals, effect size | Python, SciPy |
| **MLflow Reporter** | Logs params, metrics, artifacts to MLflow tracking server | Python, mlflow SDK |
| **Artifact Store** | Raw data (Parquet), plots (PNG), summaries (JSON) | Local filesystem, MLflow artifacts |
| **Rust Kernel Bench** | Micro-benchmarks for custom Metal kernels with correctness validation | Rust, Criterion.rs, metal-rs |
| **PyO3 Bridge** | Exposes Rust kernel results to Python layer; optionally logs to MLflow | Rust (PyO3), Python |
| **Roofline Analyzer** | Plots kernel performance against M5 Max bandwidth/compute ceilings | Python, matplotlib |

### 10.4 Alternatives Considered

**Alternative A: Pure Python (no Rust layer)**

<!-- DIAGRAM: container | alt=a -->

- **Approach:** All benchmarking in Python, including kernel-level work via MLX's `mx.fast.metal_kernel()`. No Rust workspace, no Criterion.rs, no metal-rs.
- **Pros:** Simpler build system (no Cargo + uv coordination). Faster to MVP. Entire team (community) can contribute without Rust knowledge. MLX's Python kernel API is genuinely usable for prototyping.
- **Cons:** No access to Criterion.rs's statistical rigor for micro-benchmarks (automatic outlier detection, CI, regression detection). Can't use metal-rs for direct Metal API access (buffer management, command queue control, pipeline state inspection). Python timing overhead in hot measurement loops. Lower ceiling for custom kernel development long-term.
- **Decision:** Not chosen because the kernel micro-benchmark layer is a core requirement (FR5) and the long-term goal includes custom Metal kernel development where Rust + metal-rs provides the highest ceiling. The Python layer handles orchestration; Rust handles the metal.

**Alternative B: Rust-primary with Python thin layer**

<!-- DIAGRAM: container | alt=b -->

- **Approach:** Most logic in Rust (orchestration, engine drivers, metrics computation, MLflow integration). Python only as a thin wrapper for backends that require it (mlx-lm, mlx-vlm).
- **Pros:** Maximum performance for the harness itself (timing precision, minimal overhead). Single language for kernel + orchestration. No Python GIL concerns.
- **Cons:** MLX, mlx-lm, mlx-vlm, MTPLX are all Python-native — driving them from Rust adds FFI complexity and fragility. MLflow SDK is Python-native. Community contributions become harder (Rust barrier). Significantly higher development cost for the orchestration layer, which isn't performance-critical.
- **Decision:** Not chosen because the orchestration layer is not on the critical measurement path (timing happens inside the engine, not around it), and most backends are Python-native. The overhead of Python orchestration is negligible compared to the inference workloads being measured.

**Alternative C: Framework-specific benchmarks (no unified harness)**

- **Approach:** Use each backend's built-in benchmarking tool (`mtplx bench`, `llama-bench`, MLX's built-in timing). Aggregate results manually or with scripts.
- **Pros:** Zero development effort. Each tool is already optimized for its own backend.
- **Cons:** Different tools measure differently (different warmup, different timing points, different statistical methods). Results are fundamentally not comparable — which is the entire problem we're solving. No thermal discipline, no comparability gates, no unified tracking.
- **Decision:** Not chosen because this is literally the status quo we're trying to fix. This is what "nobody trusts anything" looks like.

#### 10.4.N Comparison

| Criteria | Proposed (Layered) | Alt A (Pure Python) | Alt B (Rust-primary) | Alt C (Per-tool) |
|---|---|---|---|---|
| Meets FR5 (two-level benchmarking)? | Yes — Rust microbench + Python macrobench | Partial — Python-only microbench lacks Criterion rigor | Yes — but over-engineered for macrobench | No |
| Meets FR7 (statistical comparison)? | Yes — SciPy + Criterion.rs | Yes — SciPy only | Yes — would need custom stats in Rust | No |
| Kernel development ceiling | High (metal-rs + direct Metal API) | Medium (MLX Python API) | High | N/A |
| Community accessibility | Good (Python orchestration, Rust for kernels) | Best (Python-only) | Poor (Rust barrier) | Best (existing tools) |
| Build complexity | Medium (uv + Cargo) | Low (uv only) | High (Cargo + FFI to Python backends) | None |
| Time to MVP | Medium (Python first, Rust later) | Fast | Slow | Immediate |
| Comparability guarantees | Strong (unified methodology) | Strong | Strong | None |
| **Winner** | **✓** | | | |

### 10.5 High-Level APIs

**CLI Interface:**

```
bench run <config.yaml>              # Run a benchmark experiment
bench compare <run_a> <run_b>        # Compare two runs with statistical test
bench list                           # List all tracked experiments
bench validate <run_id>              # Re-check correctness for a past run
bench report <run_id>                # Generate summary report with plots
```

**Engine Interface (Python ABC):**

```python
class Engine(ABC):
    def load_model(self, config: ModelConfig) -> None: ...
    def warmup(self, prompts: list[str], n: int) -> None: ...
    def generate(self, prompt: str, params: GenParams) -> GenerationResult: ...
    def supports_speculative(self) -> bool: ...
    def get_memory_usage(self) -> MemoryStats: ...
    def validate_correctness(self, prompt: str, reference: str, tolerance: float) -> bool: ...
```

**Engine contract guarantees:**
- `load_model` raises `EngineLoadError` if model weights are missing, corrupt, or incompatible. Never returns silently on failure.
- `generate` raises `GenerationError` on unrecoverable failure (OOM, model crash). Returns a `GenerationResult` with `status=GenerationStatus.TIMEOUT` if per-iteration timeout is exceeded (iteration is flagged tainted, not retried). Never retries internally — retry policy is the orchestrator's responsibility.
- **Timestamps (stream vs e2e-only):** When the backend measures per-token times (stream path), `len(token_timestamps) == total_tokens` and `ttft_ms` is set from the first mark. When streaming is unavailable, engines use **e2e-only**: `token_timestamps=[]`, `ttft_ms=None`, wall-clock `e2e_ms`, and `total_tokens` from tokenizer/count — **never fabricate placeholder timestamps**. Empty timestamps do not need to equal `total_tokens`. Downstream metrics: decode tok/s, SITL, and TTFT require stream timestamps; e2e-only iterations still contribute to E2E (and memory) distributions only.
- `acceptance_rate` and `accepted_length_mean` are `None` for non-speculative backends (never omitted — explicitly `None`).
- `validate_correctness` returns `False` (never raises) if output diverges beyond tolerance. The orchestrator treats `False` as an abort signal.

**GenerationResult (returned per iteration):**

```python
class GenerationStatus(Enum):
    SUCCESS = "success"
    TIMEOUT = "timeout"           # per-iteration timeout exceeded
    THERMAL_TAINTED = "tainted"   # throttling detected during generation

@dataclass
class GenerationResult:
    status: GenerationStatus
    output_text: str
    # Stream: len == total_tokens (seconds from start). E2e-only: [] — do not invent.
    token_timestamps: list[float]
    ttft_ms: float | None            # None when TTFT not measured (e2e-only)
    total_tokens: int
    memory_peak_bytes: int
    thermal_state: ThermalReading
    acceptance_rate: float | None    # None for non-speculative backends
    accepted_length_mean: float | None  # None for non-speculative backends
    power_watts: float | None        # None if powermetrics unavailable
    energy_per_token_joules: float | None  # power_watts * duration / total_tokens
    e2e_ms: float | None             # wall-clock E2E; required for e2e-only path
```

**Experiment Config (YAML shape):**

```yaml
schema_version: "1.0"  # config schema version — bumped on breaking changes

experiment:
  name: "mtplx-qwen3-27b-q4-baseline"
  description: "Baseline MTPLX performance on Qwen3 27B Q4"

hardware:
  profile: "m5_max_128gb"

model:
  name: "qwen3-27b"
  quantization: "q4_k_m"
  backend: "mtplx"

benchmark:
  warmup_iterations: 10
  timed_iterations: 10
  per_iteration_timeout_sec: 300  # tainted if exceeded, not retried
  prompt_dataset: "datasets/agentic_coding_v1.jsonl"
  cooldown_between_runs_sec: 30
  monitor_thermal: true
  abort_if_throttling: true

metrics:
  report_percentiles: [50, 90, 95, 99]
  report_trimmed_mean: true
  report_std: true
  flag_cov_threshold: 0.05

reproducibility:
  random_seed: 42
  record_git_commit: true
  record_env_versions: true
```

**Versioning Strategy:**

- **Config YAML schema:** versioned via `schema_version` field. The harness validates configs against the declared schema version. Old configs remain parseable; new fields get defaults. Breaking changes (field removal, semantic changes) bump the major version.
- **MLflow metrics schema:** every run is tagged with `metrics_schema_version: "1.0"`. Comparability Gate rejects comparisons across different metrics schema versions. When new metrics are added (additive), minor version bumps. When metric definitions change (e.g., SITL calculation refined), major version bumps — old runs are not comparable to new ones.
- **Engine interface:** breaking changes (new required abstract methods) require a new engine plugin version. Old engine versions remain usable for reproducing historical experiments. Engine plugins carry a `ENGINE_INTERFACE_VERSION` constant checked at registration time.

### 10.6 Data Flow

**Figure 3: Primary Flow Sequence Diagram** — [primary-flow.mmd](HLD.diagrams/primary-flow.mmd) | Last updated: 2026-07-08
> Shows the full benchmark lifecycle: config → hardware probe → prompt verification → thermal check → MLflow run → correctness gate → warmup → measurement (with tainted iteration handling) → metrics computation → MLflow logging. Includes thermal throttle detection as an alt path.

**Figure 4: Error Scenario — Comparability Gate Blocks Invalid Comparison** — [error-comparison-blocked.mmd](HLD.diagrams/error-comparison-blocked.mmd) | Last updated: 2026-07-08
> Shows what happens when a researcher attempts to compare two runs with incompatible conditions. The Comparability Gate checks metadata match (prompt hash, hardware, schema version, thermal quality) and either blocks with specific violations or proceeds to statistical comparison (Welch's t-test / Mann-Whitney U → effect size + CI).

**Data flow summary:**

```
Config YAML → Orchestrator → Hardware Probe (system state)
                           → Prompt Manager (load + verify hash)
                           → Thermal Monitor (baseline check)
                           → MLflow (open run, log params)
                           → Engine (correctness gate)
                           → Engine (warmup × N)
                           → Engine (measure × M) ←→ Thermal Monitor (per-iteration)
                           → Metrics Computer (raw timestamps → distributions)
                           → MLflow (log metrics + artifacts)

Comparison request → Comparability Gate (validate metadata match)
                   → Statistical Comparator (t-test / Mann-Whitney U)
                   → Report (effect size, CI, verdict)
```

**Artifact lineage:**

```
Experiment Config (YAML, hashed)
    └── MLflow Run
         ├── Params: hardware profile, model config, prompt hash, library versions
         ├── Metrics: p50/p90/p95/p99, trimmed mean, std dev, CoV, bandwidth util
         ├── Artifacts:
         │    ├── raw_timings.parquet (per-token timestamps, all iterations)
         │    ├── summary.json (computed statistics)
         │    ├── distribution.png (latency histogram + percentile markers)
         │    └── thermal_log.json (per-iteration thermal readings)
         └── Tags: stable/unstable, backend, model, quantization
```

## 11. Monitoring & Operations

This is a local research tool, not a production service. "Monitoring" here means ensuring the harness itself is producing trustworthy results.

- **Key metrics (harness health):**
  - Inter-run CoV per configuration (should stay <5%)
  - Number of tainted/discarded iterations per run (if consistently high, thermal discipline needs adjustment)
  - Correctness gate pass rate (should be 100% — failures indicate backend bugs or model corruption)
  - MLflow run count and artifact completeness (no orphaned or incomplete runs)

- **Alarms (implemented as CLI warnings, not pagers):**
  - CoV > 5% → "Results unstable, investigate before trusting"
  - >30% of iterations tainted by thermal → "Thermal discipline insufficient, increase cooldown or reduce consecutive runs"
  - Correctness gate failure → "ABORT: Backend producing incorrect output, do not benchmark"

- **Dashboards:** MLflow UI (local) serves as the primary dashboard for comparing runs, viewing metrics over time, and browsing artifacts.

- **Scaling strategy:** N/A — single-user, single-machine tool. If prompt datasets grow large, runs take longer but the architecture doesn't change.

## 12. Security & Privacy

- **Data classification.** Public / non-sensitive. No PII in prompts or outputs.
- **Endpoints.** None — fully local tool. MLflow tracking is file-based (`mlruns/`), not networked.
- **Encryption.** N/A for local file storage. If results are published, they contain only performance metrics and synthetic prompt datasets.
- **Authentication & authorization.** None required.
- **InfoSec considerations.** Only risk: if prompt datasets are later expanded to include real code snippets, ensure no proprietary or credential-bearing code is included. Mitigated by SHA-256 hashing and manual dataset curation. Review dataset contents before any public sharing.

## 13. Resiliency & Availability

This is a local CLI tool — availability targets don't apply. Resiliency concerns are about **data integrity**:

- **Failure scenario: Crash mid-run.** If the harness crashes during measurement (OOM, power loss, kernel panic), partial results must not pollute MLflow. Mitigation: only log to MLflow after all iterations complete and metrics are computed. Raw timing data is buffered in memory or a temp file, written atomically.
- **Failure scenario: Corrupted MLflow state.** MLflow file-based tracking can be corrupted by interrupted writes. Mitigation: Raw Parquet data is the primary store (ground truth) — it is written first, atomically, before MLflow logging. MLflow is the queryable index; it can be rebuilt from Parquet via `bench rebuild-index`. MLflow artifacts directory is git-ignored but backed up periodically.
- **Failure scenario: Model file corruption.** A corrupted model weight file produces silent garbage. Mitigation: the correctness gate (FR9) catches this before any performance measurement.
- **Failure scenario: Backend crash/OOM mid-generation.** Engine raises `GenerationError`. Orchestrator flags that iteration as tainted (status=TIMEOUT or catches exception). If ≥3 consecutive iterations fail, abort the entire run with a clear diagnostic. Do not retry failed iterations — retries introduce confounds.
- **Failure scenario: Backend subprocess hang** (llama.cpp, BaseRT). Per-iteration timeout (`per_iteration_timeout_sec` in config, default 300s). On timeout, kill subprocess, flag iteration as tainted, continue to next iteration if below consecutive-failure threshold.
- **Failure scenario: MLflow write failure** (disk full, permissions). Buffer metrics in memory. If MLflow logging fails after metrics computation, write a fallback JSON file with all data + an error log. The run's data is preserved in Parquet regardless — MLflow is the index, not the store.
- **Failure scenario: Hardware Probe failure** (can't read system specs). Abort with diagnostic. Hardware profile is required metadata — a run without it is non-comparable.
- **Failure scenario: Thermal Monitor unavailable** (`powermetrics` requires sudo and isn't available). Fallback: detect throttling via timing anomaly (iteration >2× slower than median of prior iterations = likely throttled). Log a warning that thermal monitoring is degraded. Comparability Gate flags runs with degraded thermal monitoring.

**Retry policy:** No retries. A failed/timed-out iteration is tainted and excluded from metrics. This is deliberate — retries introduce selection bias (you'd only retry "bad" runs, inflating reported performance).

**Blast radius:** Engine plugins are isolated. A broken engine (e.g., mlx-lm crashes) does not affect other engines or the harness itself. Only the specific experiment using that engine fails.

- **Recovery.** Re-run the experiment. All configs are declarative (YAML), prompts are versioned (SHA-256), and environment is pinned. Any run can be reproduced from its config alone.

## 14. Accessibility

N/A — CLI tool, not user-facing UI.

## 15. Testability

### Unit tests
- Metrics Computer: given known timing arrays, verify p50/p90/p95/p99, trimmed mean, std dev, CoV calculations match expected values.
- Comparability Gate: given two run metadata objects, verify it correctly identifies violations (different prompt hash, different hardware, etc.) and passes valid pairs.
- Statistical Comparator: given known distributions (e.g., two normals with known separation), verify the t-test produces correct p-values and effect sizes.
- Prompt Manager: verify SHA-256 validation catches tampered datasets.

### Integration tests
- Engine interface contract tests: for each backend engine, verify it implements the full interface (load_model, warmup, generate, validate_correctness) and returns well-formed GenerationResult objects.
- MLflow round-trip: run a mock experiment, verify all params/metrics/artifacts are logged and retrievable.
- Thermal Monitor: verify it correctly parses `powermetrics` output and detects throttle conditions (can use recorded sample output).

### End-to-end tests
- Run a minimal experiment (small model, 2 warmup, 3 iterations) against one backend. Verify: MLflow run exists, metrics are within plausible ranges, no tainted iterations on a cold machine, artifacts are complete.
- Run comparison between two identical runs: verify the comparator reports "no significant difference."
- Run comparison between deliberately different conditions: verify the Comparability Gate blocks it.

### What is hard to test
- Thermal behavior reproducibility — depends on ambient temperature, prior workload, and hardware state. Mitigated by the CoV threshold (if results aren't stable, the harness tells you).
- Backend correctness across all models/quantizations — the correctness gate catches failures, but defining "correct" for stochastic generation requires reference outputs at fixed seeds.

## 16. Correctness & Reliability

### Input → output guarantees
- Given a valid experiment config and a passing correctness gate, the harness will produce a complete set of statistical metrics (p50/p90/p95/p99, mean, trimmed mean, std dev, CoV) for every metric type applicable to the backend.
- Given two run IDs, the comparator will either report a statistical comparison (with confidence interval and effect size) or block with specific comparability violations. It will never silently produce an invalid comparison.

### Edge cases
- **Zero valid iterations** (all tainted by thermal or failed): harness reports failure with a clear message ("0 of M iterations valid — cannot compute statistics"), does not log metrics to MLflow. Logs diagnostic info (failure reasons per iteration) as an artifact.
- **Single valid iteration** (M-1 tainted): harness logs a warning ("only 1 valid iteration — no statistical distribution possible"), reports the single value with no percentiles/std/CoV. Tags the MLflow run as `quality: insufficient_data`. Comparability Gate rejects comparisons involving this run.
- **Fewer than 5 valid iterations** (but more than 1): harness computes statistics but tags the run as `quality: low_confidence` and logs a warning. CoV may be unreliable with small N. Comparability Gate allows comparison but includes a caveat in the report.
- **All iterations identical** (CoV = 0): valid result — this can happen with very short prompts or perfectly deterministic backends. Report CoV = 0 without flagging as unstable. Note: if all iterations are identical AND the workload should show variation (e.g., long generation with thermal drift), this may indicate a measurement bug (e.g., cached results being returned). Log a diagnostic note.
- **Backend timeout/hang:** configurable per-iteration timeout (`per_iteration_timeout_sec`). Timed-out iterations are flagged as tainted (status=TIMEOUT), not retried.
- **Identical runs (no difference):** comparator correctly reports "no statistically significant difference" rather than treating zero difference as an error.
- **Very small effect sizes:** report confidence intervals so the researcher can judge whether the improvement is practically meaningful, not just statistically significant. Include Cohen's d for standardized interpretation.

### Data consistency mechanisms
- **Prompt dataset immutability:** SHA-256 hash verified before every run. Any modification to the dataset requires a new versioned file with a new hash.
- **MLflow run atomicity:** metrics and artifacts logged only after successful completion. No partial runs in the tracking store.
- **Reproducibility record:** every run logs sufficient metadata to be fully reproduced — config, git SHA, library versions, hardware profile, random seed. If a result can't be reproduced, the metadata tells you why.

## 17. Operating Cost Estimate

This is a local tool running on hardware already owned. No cloud costs.

- **Cost drivers:**
  - Developer time (primary cost) — building and maintaining the harness
  - Electricity for sustained GPU workloads during benchmarking sessions (negligible)
  - Storage for MLflow artifacts and raw Parquet data (~10-50 MB per experiment run; GB-scale after hundreds of runs)
  - Model weights storage (~10-50 GB per model, already downloaded for inference work)

- **Back-of-envelope:**
  - 1 year of active experimentation: ~50-200 GB of benchmark artifacts (easily fits on local SSD)
  - No recurring subscription costs (MLflow is local/free, all tools are open source)
  - Time investment: ~2-4 weeks to MVP, ~1-2 days per new backend integration

- **Comparison with alternatives:** Using existing per-tool benchmarks (`mtplx bench`, `llama-bench`) costs nothing to build but produces non-comparable results — the "cost" is wasted research time drawing wrong conclusions from uncontrolled measurements.

## 18. Approvals

| Name | Role | Status | Date |
|------|------|--------|------|
| Wes | Author / implementer | Approved | 2026-07-08 |

*(Single-person project. This section exists for process completeness and to mark the methodology as frozen after approval — changes to measurement methodology after this point require a new version of this doc.)*

## 19. Related Documents & Impact Assessment

- **Inference Benchmarking — Research Plan & Methodology.md** — Prior research document establishing principles (Amazon Science rigor). This HLD supersedes it as the authoritative design; the research doc remains as reference for methodology rationale.
- **mlx-inference-workbench README.md** — Repo-level overview and phased roadmap. Update after HLD approval to reference this document.
- **experiments/README.md** — Experiment spike template. No changes needed; it already references the shared benchmarking harness.
- **Future: ADR for layered architecture decision** — Extract §10.4 into a standalone ADR for long-term reference.

## 20. Implementation Strategy

### Phases

**Phase 0.5: Thermal Reproducibility Validation (target: 2-3 days) — HARD GATE**
- Run a single model (mlx-lm, Qwen3 or similar) with a fixed prompt, 20 runs spread across 2 days (morning/afternoon/evening).
- Measure inter-run CoV for decode tok/s.
- If CoV < 5%: methodology is validated, proceed to Phase 1.
- If CoV > 5%: iterate on thermal discipline (increase cooldown, restrict time-of-day, statistical normalization, ambient temp logging) until CoV < 5% is reliably achievable.
- Deliverable: documented thermal reproducibility report with methodology that achieves CoV < 5%.

**Phase 1: MVP (target: 1-2 weeks)**
- Implement: Orchestrator, Hardware Probe, Prompt Manager, Metrics Computer, MLflow Reporter
- One engine: mlx-lm
- One prompt dataset: ~20 agentic coding prompts (tool calls, multi-turn reasoning, code gen), hashed
- Full statistical pipeline: warmup → measure → p50/p90/p95/p99 → MLflow
- Thermal Monitor (basic: detect throttle, flag iterations)
- Correctness gate (compare output against stored reference at fixed seed)
- CLI: `bench run` and `bench compare`
- Deliverable: can run `bench run config.yaml` and get a statistically rigorous result logged to MLflow

**Phase 2: Backend expansion (target: 1-2 weeks per backend)**
- Add MTPLX engine (with speculative-specific metrics: acceptance rate, accepted length)
- Add llama.cpp engine
- Add BaseRT engine (if programmatic API available)
- Add mlx-vlm text mode engine
- Expand prompt dataset to ~50-100 prompts

**Phase 3: Rust kernel layer (target: 2-4 weeks)**
- Initialize Rust workspace (Cargo.toml, metal-rs, Criterion.rs, PyO3)
- Implement reference (naive) matmul for correctness baseline
- Implement first Metal shader (tiled quantized matmul)
- Criterion micro-benchmarks with kernel sweep configs (M/N/K dimensions)
- PyO3 bridge to log kernel results to MLflow
- Roofline analyzer (plot against M5 Max bandwidth/compute ceilings)
- Custom kernel engine plugin in Python layer

**Phase 4: Analysis and advanced features (ongoing)**
- Kernel regression tracking across commits
- Automated roofline plots per kernel version
- Run-to-run stability tracking over time
- Comparison reports with effect size and practical significance annotations

### Blocking dependencies
- M5 Max hardware (available)
- MTPLX installable and drivable programmatically (to be verified in Phase 0 familiarization spike)
- BaseRT availability and API stability (to be verified)
- `powermetrics` access for thermal monitoring (may require sudo or entitlements)

### T-shirt estimates
- Phase 1 (MVP): **M** (medium) — well-scoped, one backend, known libraries
- Phase 2 (backends): **S** per backend (small) — engine plugin is the interface, implementation varies
- Phase 3 (Rust layer): **L** (large) — new toolchain, Metal kernel development, cross-language bridge
- Phase 4 (analysis): **S-M** (ongoing small increments)

### MVP scope
Smallest shippable slice: `bench run` with mlx-lm on one prompt dataset, producing p50/p90/p95/p99 + MLflow tracking + thermal detection + correctness gate + `bench compare` with statistical significance test.

## 21. Open Questions

| # | Question | Owner | Target date |
|---|----------|-------|-------------|
| 1 | Can MTPLX be driven programmatically (Python API), or only via CLI? | Wes | Phase 0 spike |
| 2 | Can BaseRT be driven programmatically, or only binary execution? | Wes | Phase 2 |
| 3 | Does `powermetrics` require root/sudo for thermal readings on macOS? If so, what's the least-privilege alternative? | Wes | Phase 1 |
| 4 | Verify M5 Max specs via (a) Apple published specs AND (b) empirical measurement (STREAM + peak-FLOPS kernel). Hard gate for roofline analysis. See Appendix B for methodology. | Wes | Phase 1 (blocks Phase 3 roofline) |
| 5 | ~~Resolved in §10.2 step 7: correctness gate always runs at temp=0 with fixed seed (bitwise match), with KL-divergence fallback for non-deterministic backends.~~ | Wes | Resolved |
| 6 | Should Criterion.rs results be logged directly to MLflow from Rust, or written to a JSON/Parquet intermediate that Python picks up? Must resolve before any Rust benchmarks are logged — different integration paths may capture different metadata, breaking comparability. | Wes | Phase 1 (resolve before Phase 3 implementation) |
| 7 | Memory peak comparability across backends: backends use different allocation strategies (MLX lazy eval, llama.cpp pre-allocated KV cache, custom kernels with pooled allocators). Is "memory peak" (via `ps`/`vmmap`) a fair comparison, or do we need backend-specific memory accounting? Document limitations in methodology. | Wes | Phase 1 |

## 22. Risks & Ambiguity

- **Risk: Thermal behavior is not reproducible enough.** Even with cooldown discipline, ambient temperature and prior workload may create irreducible variance. Mitigation: CoV threshold flags it; cooldown tuning; benchmark in controlled environment (AC, consistent time of day). **Hard gate:** Phase 0.5 Thermal Reproducibility Spike must validate CoV < 5% is achievable before any Phase 1 baselines are logged as official. If thermal variance is structurally above 5%, methodology must be redesigned (longer cooldown, fewer consecutive runs, statistical normalization by thermal state, or restricting to night-only benchmarking). Likelihood: Medium. Impact: High (undermines core value proposition).

- **Risk: Backend APIs are unstable or undocumented.** MTPLX, BaseRT, and mlx-vlm are actively evolving open-source projects. Their APIs may change between versions. Mitigation: pin library versions per experiment, version the engine plugins (e.g., `engines/mlx_lm_v1.py`, `engines/mlx_lm_v2.py`), abstract behind the Engine interface. When a backend API breaks, ship a new engine plugin version rather than updating the old one — historical experiments remain reproducible with old engines. Each engine plugin declares `ENGINE_INTERFACE_VERSION` and the backend library version it targets. Likelihood: High. Impact: High (unmitigated, this silently breaks historical comparability; mitigated via versioned adapters, impact reduces to Medium — development overhead only).

- **Risk: `powermetrics` thermal monitoring is insufficient.** It may not detect throttling at the granularity needed (per-iteration), or may require sudo which complicates the workflow. Mitigation: investigate IOKit direct access; fallback to detecting throttle via anomalous timing (iteration significantly slower than peers). Likelihood: Medium. Impact: Medium.

- **Risk: Correctness gate is too strict or too loose.** Stochastic generation means outputs vary — defining "correct" is non-trivial at non-zero temperature. Mitigation: use fixed seeds for correctness gate; validate against known-good output; allow configurable tolerance. Likelihood: Medium. Impact: Low (gate can be tuned).

- **Risk: MLflow becomes an overhead rather than a benefit.** If the tracking adds friction to rapid iteration during early spikes, it may slow down the experimentation loop. Mitigation: make MLflow logging optional via config flag for quick iteration; always log for "official" comparison runs. Likelihood: Low. Impact: Low.

## 23. Appendix

### A. Statistical Methods Reference

- **Welch's t-test:** Used for comparing means of two runs when sample sizes may differ and variances may be unequal. Reports t-statistic, p-value, and degrees of freedom.
- **Mann-Whitney U test:** Non-parametric alternative when distributions are non-normal (detected via Shapiro-Wilk test on raw timings). Reports U-statistic and p-value.
- **Effect size (Cohen's d):** Standardized difference between means. <0.2 = negligible, 0.2-0.5 = small, 0.5-0.8 = medium, >0.8 = large.
- **Confidence interval:** 95% CI on the difference in means. If CI excludes zero, the difference is statistically significant at α=0.05.
- **Coefficient of variation (CoV):** std dev / mean. Threshold of 5% for "stable" results based on internal Amazon benchmarking best practices (DemerzelBench).
- **Trimmed mean (tm99):** Mean after dropping top 1% of values. Reduces impact of occasional thermal spikes or GC pauses.

### B. M5 Max Hardware Profile (issue #8)

Profile file: `configs/hardware/m5_max_128gb.yaml` (versioned; re-verify with `make hardware-ceilings-write`).

| Spec | Published | Empirical method | Conservative ceiling | Status |
|------|-----------|------------------|----------------------|--------|
| Unified memory | 128 GB | `sysctl hw.memsize` | 128 GB | Verified on workbench host |
| Memory bandwidth | **460 GB/s** (32-core GPU SKU) or **614 GB/s** (40-core GPU SKU) — [Apple MacBook Pro tech specs](https://support.apple.com/en-us/126319) | **L1 (kernel):** Rust + MSL STREAM (`crates/metal_stream`, copy/scale/add/triad). **L2 (proxy):** MLX compiled triad. | **min(published, empirical)** with `empirical` preferring Metal STREAM | Verified; YAML `empirical_metal_stream` is kernel denominator |
| GPU cores | 32 or 40 | `system_profiler SPDisplaysDataType` | Match detected cores | Verified procedure |
| GPU FP32 | **Not published by Apple** (as of 2026-07) | Peak MLX FP32 matmul TFLOPS (`measure_matmul_tflops`) | Empirical only | Empirical ceiling only — do not invent vendor TFLOPS |
| Neural Engine | Informational | Not directly programmable | N/A | Out of harness scope |
| Chip | M5 Max | `sysctl` / Hardware Probe fingerprint | — | Runtime |

**Verification strategy (both required for bandwidth):**
1. **Published specs:** Apple support / product tech specs URL above. Map GPU core count → 460 vs 614 GB/s.
2. **Empirical measurement:**
   - **Kernel-grade (L1):** `make metal-stream` or `cargo run -p metal_stream --release` — MSL STREAM via Rust host (same stack as Phase 3 custom kernels).
   - **Proxy (L2):** MLX compiled triad via `workbench.ceilings`.
   - Full write: `make hardware-ceilings-write` runs both and updates YAML.
   - Use **min(published, empirical)**; `empirical` prefers `empirical_metal_stream`.

**Hard gate (Phase 3 roofline):** Custom kernel “% of peak bandwidth” uses the **Metal STREAM conservative ceiling**, not brochure peak alone and not MLX-only. If measured ≫ published, treat as measurement artifact / wrong SKU. FLOPS roofline uses empirical matmul TFLOPS until Apple publishes a figure.

**API:** `workbench.hardware_profile.bandwidth_utilization_pct(achieved_gbs, ceiling_gbs)` and `HardwareProfile.bandwidth_ceiling_gbs()`. Rust: `metal_stream::run_stream`.

### C. Prompt Dataset Schema

```jsonl
{"id": "agentic_001", "category": "tool_call", "prompt": "...", "expected_tokens_approx": 150}
{"id": "agentic_002", "category": "multi_turn_reasoning", "prompt": "...", "expected_tokens_approx": 500}
{"id": "agentic_003", "category": "code_generation", "prompt": "...", "expected_tokens_approx": 300}
```

Accompanied by `checksums.sha256`:
```
e3b0c44298fc1c14... datasets/agentic_coding_v1.jsonl
```

### D. Future: ADR for Architecture Decision

Extract §10.4 into a standalone ADR documenting the layered architecture choice. Reference this HLD as the decision context.
