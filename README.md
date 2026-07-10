# mlx-inference-workbench

Personal workbench for optimizing LLM text inference on Apple Silicon, with a focus on custom Metal kernels, reproducible benchmarking, and phased experimentation.

## Goals

- Establish reproducible baselines for text LLM inference on M5 Max (128 GB unified memory)
- Maintain a **single, consistent benchmarking harness** so all experiments can be compared fairly
- Build deep understanding of custom Metal kernel engineering (MTPLX, BaseRT, pmetal patterns)
- Iteratively develop and measure improvements via custom kernels and algorithmic optimizations
- Work backwards from the goal of **running capable coding agents locally on Apple Silicon**

## Philosophy

Spike-driven measurement discipline (not a delivery tracker):

1. **Familiarize** → **Validate** (thermal gate before official numbers) → **Baseline**
2. **Understand** bottlenecks → **Optimize** one spike at a time
3. Same harness for every claim; report distributions + statistical significance

## Roadmap (milestones)

Delivery is tracked in **GitHub milestones** (source of truth for issue membership and DoD).  
HLD “Phase 0/1/2…” language is historical design structure; map it to milestones in [docs/TASKS.md](docs/TASKS.md).

| MS | Focus | Status (approx.) |
|----|--------|------------------|
| **[M1: Lab foundation](https://github.com/weklund/mlx-inference-workbench/milestone/1)** | Harness, CI, mlx-lm, ceilings, thermal HARD GATE, official baseline | Harness + ceilings + provisional baseline **done**; open: [#3](https://github.com/weklund/mlx-inference-workbench/issues/3) thermal, [#36](https://github.com/weklund/mlx-inference-workbench/issues/36) official baseline |
| **[M2: Multi-engine comparison](https://github.com/weklund/mlx-inference-workbench/milestone/2)** | MTPLX + llama.cpp plugins; [MLX engine landscape](docs/notes/mlx-text-inference-engines.md) (#38); EXP wave | Next product wave after M1 close-out |
| **[M3: Custom Metal / Rust kernels](https://github.com/weklund/mlx-inference-workbench/milestone/3)** | Kernel maturity, llvm-cov, custom Metal beyond STREAM | Seeded by `metal_stream`; [#34](https://github.com/weklund/mlx-inference-workbench/issues/34) |

Later optimization themes (skew-aware paths, adaptive controllers, thermal closed-loop, agent rollback) stay in HLD / TASKS Phase 3+ — not separate README phases.

Epic: [#4](https://github.com/weklund/mlx-inference-workbench/issues/4).

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
make lint          # ruff check + format --check (explicit rule set; see docs/notes/python-lint.md)
make test          # fast unit suite
make coverage      # unit + ≥80% workbench cov + core module floors (merge gate)
make ci            # lint + coverage (local mirror of required Python checks)
make ci-rust       # rustfmt + clippy (pedantic) + rustdoc -D warnings + tests
make smoke         # stub harness end-to-end
```

Scripts, spikes, and CLI pretty-print are **not** in the hard coverage gate. Details: [`docs/notes/coverage.md`](docs/notes/coverage.md).  
Python lint/docs policy: [`docs/notes/python-lint.md`](docs/notes/python-lint.md).  
Rust/Metal DoD: [`docs/notes/rust-metal-dod.md`](docs/notes/rust-metal-dod.md).

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

- [HLD: Benchmarking Harness](docs/HLD.md) — Architecture, requirements, design decisions (incl. original HLD phase narrative)
- [Task Breakdown](docs/TASKS.md) — Milestone ↔ issue map + residual checkboxes
- [MLX text engine landscape](docs/notes/mlx-text-inference-engines.md) — M2 catalog & plugin prioritization ([#38](https://github.com/weklund/mlx-inference-workbench/issues/38))
- Milestones: [M1](https://github.com/weklund/mlx-inference-workbench/milestone/1) · [M2](https://github.com/weklund/mlx-inference-workbench/milestone/2) · [M3](https://github.com/weklund/mlx-inference-workbench/milestone/3) · Epic [#4](https://github.com/weklund/mlx-inference-workbench/issues/4)

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
