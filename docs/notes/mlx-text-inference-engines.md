# MLX text inference engines — landscape catalog

> **Milestone:** [M2: Multi-engine comparison](https://github.com/weklund/mlx-inference-workbench/milestone/2)  
> **Tracking issue:** [#38](https://github.com/weklund/mlx-inference-workbench/issues/38)  
> **Captured:** 2026-07-10 from local research notes. Ecosystem moves quickly; re-verify before implementing a plugin.  
> **Community index:** [awesome-mlx](https://github.com/raullenchai/awesome-mlx)

## Purpose (this workbench)

We measure **single-run, comparable, scientific** text LLM inference on Apple Silicon via the `Engine` ABC — not multi-tenant serving products.

| In scope for plugins | Out of scope as product goals (HLD §7) |
|----------------------|----------------------------------------|
| Programmatic generate/stream with honest timings | Continuous batching / multi-user serving as a *feature* |
| Same prompts, thermal class, schema gates | OpenAI-compatible server product UX |
| Optional speculative / cache metrics when measurable | Multimodal image workloads (text path only) |
| | **Training / fine-tuning / distillation** (separate from inference engines) |

**Implication:** A project can be an excellent *local server* and still be a **poor** first Engine plugin if it only exposes HTTP, batches many users, or hides per-token timing. Prefer libraries (or servers with a clear single-request programmatic path) that we can drive under our orchestrator.

Related **non-MLX** compare arms (still HLD / M2): **llama.cpp** Metal (#15), **BaseRT** (API TBD), **Custom kernel** (M3). **MTPLX** (#9) is MLX-adjacent speculative stack — keep as first multi-backend target.

---

## Out of scope (adjacent MLX tooling — not Engine plugins)

These matter in the Apple Silicon ecosystem but are **not** text inference engines for our harness. Listed so research notes do not leave gaps.

| Project | Repo | What it is | Why not a plugin |
|---------|------|------------|------------------|
| **mlx-tune** (ARahim3) | https://github.com/ARahim3/mlx-tune | Fine-tune LLMs (and VLM/TTS/STT/… ) on MLX; Unsloth-compatible API (SFT, DPO, GRPO, LoRA, …) | **Training**, not decode. HLD §7 excludes training/distill as product work. May produce models later measured via **mlx-lm** / other engines — no `backend: mlx-tune` arm. |

---

## Foundational library

| Project | Repo | Notes | Plugin priority |
|---------|------|--------|-----------------|
| **mlx-lm** (ml-explore) | https://github.com/ml-explore/mlx-lm | Official text gen/finetune library; HF / mlx-community quants; `generate` / `stream_generate`; chat REPL; prompt cache; rotating KV; sampling. **Shipped** as `MlxLmEngine` (#7). | **Done (M1)** |

Multimodal text+vision is separate (**mlx-vlm**); not primary for this catalog.

---

## High-performance / advanced (often servers)

| Project | Repo | Focus | Fit for our harness | Plugin priority |
|---------|------|--------|---------------------|-----------------|
| **Rapid-MLX** (raullenchai) | https://github.com/raullenchai/Rapid-MLX | Fast OpenAI/Anthropic-compatible; tool calling; prompt caching; strong agent TTFT claims | Likely **HTTP server** — evaluate programmatic/single-req path; good *external* benchmark peer | **Evaluate (M2)** — high interest for agentic TTFT/cache |
| **oMLX** (jundot) | https://github.com/jundot/omlx | Continuous batching + paged/SSD KV; long agent sessions; menu bar app | Serving + disk KV — continuous batching is HLD §7 product-out; **prefix/SSD cache ideas** still inform EXP #10 / #12 | **Research / metrics ideas** — plugin only if single-stream path is clean |
| **vllm-mlx** (waybarrios) | https://github.com/waybarrios/vllm-mlx | vLLM-style continuous batching, paged KV, prefix cache | Throughput/serving oriented; multi-req not our default UC | **Low for plugin** — optional peer; do not adopt multi-req as harness default |
| **mlx-serve** (ddalcu) | https://github.com/ddalcu/mlx-serve | Zig native; MLX (+ GGUF); OpenAI/Anthropic/Ollama APIs; speculative mentions | Native binary — similar to llama.cpp subprocess pattern | **Evaluate (M2)** if single-req timings are exportable |

---

## API-compatible / omni servers

| Project | Repo | Focus | Plugin priority |
|---------|------|--------|-----------------|
| **mlx-omni-server** (madroidmaq) | https://github.com/madroidmaq/mlx-omni-server | Dual OpenAI + Anthropic APIs | Evaluate if thin wrapper over mlx-lm (may duplicate #7) |
| **MLX Engine** (lmstudio-ai) | https://github.com/lmstudio-ai/mlx-engine | Production backend behind LM Studio (text + multimodal pieces) | **Evaluate** for parity claims vs LM Studio; integration path may be app-bound |
| **fastmlx** (arcee-ai) | https://github.com/arcee-ai/fastmlx | Production-ready API server | Evaluate / low unless unique measurement surface |
| **mlx-llm-server** (mzbac) | https://github.com/mzbac/mlx-llm-server | OpenAI-compatible server | Low (thin server) |
| **mlx-openai-server** (cubist38) | https://github.com/cubist38/mlx-openai-server | Simple OpenAI endpoints | Low (thin server) |
| **Toolio** (OoriData) | https://github.com/OoriData/Toolio | JSON-schema / tool-calling with MLX | **Adjacent** — structured-output EXP, not core thruput engine |

---

## Other mentions (watchlist)

| Project / cluster | Notes | Priority |
|-------------------|--------|----------|
| **swama, SwiftLM, PicoMLXServer** | Swift-native OpenAI-compatible apps | Out of scope unless we add a Swift FFI story (not planned) |
| **mlx_parallm, mlx-gui, mlx_sharding** | Parallel / UI / distributed | Partial interest for sharding later; not M2 default |
| **Ollama MLX backend** | Not an “MLX engine” project; common baseline people quote | **External peer** for narrative only; not an Engine plugin unless we add intentional Ollama arm |
| **mlx-chronos** (community) | Often used to compare mlx-lm, oMLX, vllm-mlx, Rapid-MLX, Ollama | Methodology peer — useful when writing compare reports |

---

## Non-MLX engines still in our roadmap

| Backend | Issue / MS | Role |
|---------|------------|------|
| **MTPLX** | #9 / M2 | Speculative MTP + custom Metal vs stock mlx-lm |
| **llama.cpp** Metal | #15 / M2 | Quant matrix, free-draft, non-MLX stack |
| **BaseRT** | HLD / unissued | Native Metal runtime if API usable |
| **mlx-vlm** text mode | HLD / unissued | Text-only path when needed |
| **Custom kernel** | M3 | Own kernels as Engine |

---

## Recommended M2 plugin ladder (this repo)

1. **mlx-lm** — done (control arm).  
2. **MTPLX** (#9) — primary multi-backend scientific arm.  
3. **llama.cpp** (#15) — cross-stack / quant / free-draft.  
4. **Evaluate then maybe issue plugins for:** Rapid-MLX, mlx-serve, LM Studio mlx-engine (if scriptable).  
5. **Do not prioritize as plugins:** pure multi-tenant continuous-batch servers (vllm-mlx, oMLX *as serving products*) — harvest **ideas** (prefix cache, paged KV, SSD tier) into EXPs #10–#12 instead.  
6. **Revisit:** BaseRT, mlx-vlm text, custom kernel engine after M2 core pair works.

---

## Evaluation checklist (before opening a new Engine issue)

For each candidate:

- [ ] Can we invoke **one** generation with fixed seed/temp under our orchestrator?
- [ ] Can we get stream timestamps or honest e2e-only metrics (no fabricated TTFT)?
- [ ] Library versions + model id pinable for `library_versions` / reproducibility?
- [ ] License OK for this personal/research workbench?
- [ ] Does it add a **distinct** scientific question vs another plugin (not just another OpenAI wrapper over mlx-lm)?

Pass → file `Phase 2: <name> engine plugin` under **M2**.  
Fail → leave in this catalog as watchlist / peer only.

---

## Maintenance

- Update this file when adding Engine plugins or killing candidates.  
- Prefer linking GitHub issues over duplicating DoD here.  
- Spot-check [awesome-mlx](https://github.com/raullenchai/awesome-mlx) periodically for new servers.
