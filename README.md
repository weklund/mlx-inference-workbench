# mlx-inference-workbench

Personal workbench for optimizing LLM text inference on Apple Silicon, with a focus on custom Metal kernels, reproducible benchmarking, and phased experimentation.

## Goals

- Establish reproducible baselines for text LLM inference on M5 Max (128 GB unified memory)
- Maintain a **single, consistent benchmarking harness** so all experiments can be compared fairly
- Build deep understanding of custom Metal kernel engineering (MTPLX, BaseRT, pmetal patterns)
- Iteratively develop and measure improvements via custom kernels and algorithmic optimizations
- Work backwards from the goal of **running capable coding agents locally on Apple Silicon**

## Philosophy

This repo follows a **phased, spike-driven** approach:

1. **Familiarize** — Get deep into MTPLX's custom Metal kernels and MTP speculative decoding
2. **Validate** — Prove thermal reproducibility before any official baselines
3. **Baseline** — Establish strong, reproducible measurements on M5 Max
4. **Understand** — Profile at kernel level to identify real bottlenecks
5. **Optimize** — Build and measure improvements one spike at a time

All experiments must be measurable using the same tools in `benchmarks/`. Every claim requires statistical significance.

## Phased Roadmap

### Phase 0: MTPLX Familiarization
- Understand the MTP draft → verify → accept/reject flow
- Map the 4 custom Metal kernels (verify_qmv, GDN, innovation-tape, capture/replay)
- Understand how they register as MLX primitives

### Phase 0.5: Thermal Reproducibility Validation (HARD GATE)
- 20 runs across 2 days (morning/afternoon/evening)
- Must achieve CoV < 5% on decode tok/s before proceeding
- Iterate on methodology until thermal discipline is validated

### Phase 1: MVP Benchmarking Harness + Baselines
- Build the harness: orchestrator, thermal monitor, metrics computer, comparability gate
- First backend: mlx-lm end-to-end
- Verify M5 Max hardware specs (published + empirical STREAM/peak-FLOPS)
- Expand to: MTPLX, llama.cpp, BaseRT, mlx-vlm (text mode)
- Models: Qwen3/3.6 27B-class, Llama 3.x, others
- Workloads: agentic coding flows (tool calls, multi-turn reasoning, code generation)

### Phase 2: Deep Kernel Understanding
- Replicate simple versions of small-M quantized ops via `mx.fast.metal_kernel()`
- Profile at kernel level (Xcode GPU capture / Instruments)
- Document why kernels win on Apple Silicon's unified memory architecture

### Phase 3: Rust Kernel Layer
- Initialize Rust workspace (Criterion.rs + metal-rs + PyO3)
- Write reference implementations, then custom Metal shaders
- Roofline analysis against verified M5 Max ceilings

### Phase 3+: Optimization Spikes (Priority Order)
1. **Skew-aware / hot-path specialization** — Entropy/confidence-aware fast paths in kernels (est. 15-35%)
2. **Adaptive self-tuning controller** — Dynamic strategy selection based on runtime stats (est. 10-30%)
3. **Value/outcome prediction** — Predict acceptance length for better speculative decisions (est. 20-50%)
4. **Predictive thermal management** — Modulate aggressiveness based on thermal headroom
5. **Precise agent rollback** — Extend innovation-tape to tool/side-effect state
6. **Durable recovery (WAL-style)** — Persistent logging for long-running agent sessions

## Intellectual Foundations

These optimizations map to solved problems in:
- **CPU Architecture** — Branch prediction, speculative execution, ROB, precise exceptions, DVFS
- **Database Engineering** — MVCC, WAL/ARIES, self-tuning query optimization, index specialization
- **Quantitative Finance** — Value prediction, adverse selection, hot-path specialization, hardware co-design
- **Information Theory** — Entropy, mutual information, rate-distortion, delta encoding

## Folder Structure

```
mlx-inference-workbench/
├── benchmarks/          # Measurement harness (orchestrator, engines, metrics)
│   ├── configs/         # Default benchmark configs
│   └── results/         # Output (gitignored raw data)
├── datasets/            # Versioned prompt datasets (JSONL + SHA-256 checksums)
├── experiments/         # Individual experiment spikes (numbered folders)
├── kernels/             # Custom Metal shader work
│   ├── metal/           # .metal shader source files
│   └── python/          # Python bindings / MLX custom kernels
├── src/                 # Shared utilities (profiling, hardware, utils)
├── configs/             # Hardware profiles, experiment configs, kernel sweep configs
│   ├── hardware/        # Hardware spec declarations (m5_max_128gb.yaml)
│   ├── experiments/     # Full experiment configs (model + backend + params)
│   └── kernels/         # Kernel micro-benchmark sweep dimensions
├── analysis/            # Comparison scripts, roofline plots, regression tracking
├── docs/                # HLD, task breakdown, spike reports
│   └── HLD.diagrams/   # Architecture diagrams (draw.io CSV + Mermaid)
└── scripts/             # Utility scripts
```

## Getting Started

### 1. Environment Setup

```bash
make sync                        # frozen lock + dev extras (same as CI)
# or: uv sync --extra profiling / --extra mtplx for optional stacks
make help                        # list all developer/CI targets
```

### Tests & coverage

CI and local use the **same Makefile recipes** (do not re-copy long pytest lines):

```bash
make lint          # ruff check + format --check
make test          # fast unit suite
make coverage      # unit + ≥80% workbench cov + core module floors (merge gate)
make ci            # lint + coverage (local mirror of required Python checks)
make smoke         # stub harness end-to-end
```

Scripts, spikes, and CLI pretty-print are **not** in the hard gate. Details: [`docs/notes/coverage.md`](docs/notes/coverage.md).

### 2. Running Benchmarks

```bash
# From a checkout (project root is discovered via pyproject.toml / configs+datasets)
make smoke
make bench-list
make bench-compare A=<run_a> B=<run_b>
make bench-report RUN=<run_id>

# Outside a checkout, or when installed: set an explicit data root
export MLX_WORKBENCH_ROOT=/path/to/workbench-data   # contains configs/, datasets/, …
# or: uv run bench --project-root /path/to/workbench-data run configs/experiments/smoke_minimal.yaml
```

Relative dataset/results paths in YAML resolve against the **project root** (env/`--project-root`/cwd discovery), never against the package install path (`site-packages`).

Results default to `<project-root>/benchmarks/results` (Parquet/JSON ground truth; optional MLflow index).

### 3. Adding a New Experiment

1. Create a new folder under `experiments/` (e.g. `04_my_spike`)
2. Add a `README.md` describing the goal and hypothesis
3. Use the shared benchmark harness for all measurements
4. Record results and findings

See `experiments/README.md` for the full template.

## Hardware

Primary target: **MacBook Pro M5 Max** (128 GB unified memory)

## Key Metrics

- **TTFT** — Time to first token (ms)
- **Decode tok/s** — Sustained output throughput
- **SITL** — Inter-token latency
- **E2E latency** — Total generation time
- **Memory peak** — Model weights + KV cache + speculative state
- **Bandwidth utilization %** — Actual vs. theoretical maximum (key for kernel work)
- **Acceptance rate** — Speculative decoding efficiency
- **Power (watts)** — Energy consumption during inference
- **Energy/token (J/tok)** — Power efficiency metric

All metrics reported as distributions: p50, p90, p95, p99, trimmed mean, std dev, CoV.

## Design Documents

- [HLD: Benchmarking Harness](docs/HLD.md) — Full architecture, requirements, and design decisions
- [Task Breakdown](docs/TASKS.md) — Implementation plan with phases and checkboxes
- [Milestone: Local inference gaps (HLD-scoped)](https://github.com/weklund/mlx-inference-workbench/milestone/1) — Setup + experiment issues with DoD/smoke tests (epic [#4](https://github.com/weklund/mlx-inference-workbench/issues/4))

## Citing

If you use this benchmarking methodology or reference results from this project, please cite it:

```bibtex
@software{eklund2026mlxinferenceworkbench,
  author       = {Eklund, Westley},
  title        = {mlx-inference-workbench},
  year         = {2026},
  url          = {https://github.com/weklund/mlx-inference-workbench},
  version      = {0.1.0}
}
```

## License

MIT
