# MLX text inference engines — landscape catalog

> **Milestone:** [M2: Multi-engine comparison](https://github.com/weklund/mlx-inference-workbench/milestone/2)  
> **Tracking issue:** [#38](https://github.com/weklund/mlx-inference-workbench/issues/38)  
> **Community index:** [awesome-mlx](https://github.com/raullenchai/awesome-mlx)  
> **Provenance:** Each **named** repo below is pinned to a **default-branch commit** and, where available, a **GitHub release tag**, with a **verification date** and **evidence links**. Do not treat bare live repo URLs as version pins.

## Provenance policy

| Field | Meaning |
|-------|---------|
| **Pin (commit)** | Full SHA of default-branch tip at verification time |
| **Release** | Latest GitHub release tag at verification (or `—` if none) |
| **Verified** | UTC date of API/README snapshot for this catalog row |
| **Evidence** | Commit and/or release URL used as the claim anchor |
| **Status** | `shipped` · `evaluate` · `research` · `low` · `oos` (out of scope) · `watchlist` |
| **Next review** | Date to re-fetch tip/release (or sooner if opening a plugin issue) |

Re-verify before implementing a plugin: ecosystem moves quickly; pins go stale.

**Catalog-wide snapshot:** verified **2026-07-10** (UTC) via GitHub API (`repos/{owner}/{repo}`, default-branch commit, `releases/latest`).  
**Next full pass:** **2026-08-10**.

---

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

## Master pin table (all named candidates)

| Project | Pin (commit) | Release | Verified | Evidence | Status | Next review |
|---------|--------------|---------|----------|----------|--------|-------------|
| **mlx-lm** | `a790972f0f84…` | [v0.31.3](https://github.com/ml-explore/mlx-lm/releases/tag/v0.31.3) (2026-04-22) | 2026-07-10 | [commit](https://github.com/ml-explore/mlx-lm/commit/a790972f0f844d81067ed45c28b524220a10c019) · [release](https://github.com/ml-explore/mlx-lm/releases/tag/v0.31.3) | **shipped** (#7) | 2026-08-10 |
| **Rapid-MLX** | `753ba5b16a3a…` | [v0.10.5](https://github.com/raullenchai/Rapid-MLX/releases/tag/v0.10.5) (2026-07-08) | 2026-07-10 | [commit](https://github.com/raullenchai/Rapid-MLX/commit/753ba5b16a3a47e2cc9389207d2771c073910d75) · [release](https://github.com/raullenchai/Rapid-MLX/releases/tag/v0.10.5) | **evaluate** | 2026-08-10 |
| **oMLX** | `2130f14a1e5a…` | [v0.5.0](https://github.com/jundot/omlx/releases/tag/v0.5.0) (2026-07-10) | 2026-07-10 | [commit](https://github.com/jundot/omlx/commit/2130f14a1e5aa5e47158a5b4ebdf78d48a8aafb6) · [release](https://github.com/jundot/omlx/releases/tag/v0.5.0) | **research** | 2026-08-10 |
| **vllm-mlx** | `0dd115769ef1…` | [v0.4.0](https://github.com/waybarrios/vllm-mlx/releases/tag/v0.4.0) (2026-06-28) | 2026-07-10 | [commit](https://github.com/waybarrios/vllm-mlx/commit/0dd115769ef1196a715b96b181353edacd2a4f69) · [release](https://github.com/waybarrios/vllm-mlx/releases/tag/v0.4.0) | **low** | 2026-08-10 |
| **mlx-serve** | `1f9be78e9857…` | [v26.7.5](https://github.com/ddalcu/mlx-serve/releases/tag/v26.7.5) (2026-07-09) | 2026-07-10 | [commit](https://github.com/ddalcu/mlx-serve/commit/1f9be78e9857b3a91d0bd7eaa12bca3636bfa27e) · [release](https://github.com/ddalcu/mlx-serve/releases/tag/v26.7.5) | **evaluate** | 2026-08-10 |
| **mlx-omni-server** | `4f8e9ef62380…` | [v0.5.3](https://github.com/madroidmaq/mlx-omni-server/releases/tag/v0.5.3) (2026-05-09) | 2026-07-10 | [commit](https://github.com/madroidmaq/mlx-omni-server/commit/4f8e9ef623808b876d2fd08df8fe11745d5a2751) · [release](https://github.com/madroidmaq/mlx-omni-server/releases/tag/v0.5.3) | **evaluate** | 2026-08-10 |
| **mlx-engine** (LM Studio) | `8ae261033bc5…` | — (no GitHub release) | 2026-07-10 | [commit](https://github.com/lmstudio-ai/mlx-engine/commit/8ae261033bc5bc16fdfc19a842bfc1d96db51348) | **evaluate** | 2026-08-10 |
| **fastmlx** | `1fe5d766ac05…` | [v0.2.1](https://github.com/arcee-ai/fastmlx/releases/tag/v0.2.1) (2024-08-10) | 2026-07-10 | [commit](https://github.com/arcee-ai/fastmlx/commit/1fe5d766ac05f4f31daf446f99360aa8fd92938e) · [release](https://github.com/arcee-ai/fastmlx/releases/tag/v0.2.1) | **low** | 2026-08-10 |
| **mlx-llm-server** | `f37dad7d051e…` | — (no GitHub release; last push 2024-03) | 2026-07-10 | [commit](https://github.com/mzbac/mlx-llm-server/commit/f37dad7d051eebb89e5c779bc8669a43bf356a65) | **low** | 2026-08-10 |
| **mlx-openai-server** | `4b7d4b61586d…` | [v1.8.1](https://github.com/cubist38/mlx-openai-server/releases/tag/v1.8.1) (2026-05-03) | 2026-07-10 | [commit](https://github.com/cubist38/mlx-openai-server/commit/4b7d4b61586dd08b5ca1149b73003e85272ff652) · [release](https://github.com/cubist38/mlx-openai-server/releases/tag/v1.8.1) | **low** | 2026-08-10 |
| **Toolio** | `a1e459737694…` | [v0.6.0](https://github.com/OoriData/Toolio/releases/tag/v0.6.0) (2025-05-26) | 2026-07-10 | [commit](https://github.com/OoriData/Toolio/commit/a1e4597376940f626c9aa086f49d7d9b1ad6d7cf) · [release](https://github.com/OoriData/Toolio/releases/tag/v0.6.0) | **oos** (structured-output EXP) | 2026-08-10 |
| **mlx-tune** | `9690fe1fed19…` | [v0.6.0](https://github.com/ARahim3/mlx-tune/releases/tag/v0.6.0) (2026-06-23) | 2026-07-10 | [commit](https://github.com/ARahim3/mlx-tune/commit/9690fe1fed19c72f0810902b9c188d9a2625eb5b) · [release](https://github.com/ARahim3/mlx-tune/releases/tag/v0.6.0) | **oos** (training) | 2026-08-10 |
| **awesome-mlx** (index) | `b87192b2b687…` | — | 2026-07-10 | [commit](https://github.com/raullenchai/awesome-mlx/commit/b87192b2b687d89e6b6cdcf262b3041432902dfa) | **watchlist** (directory only) | 2026-08-10 |
| **mlx-chronos** (peer) | *unpinned* | — | 2026-07-10 | Name collision: e.g. [igurss/mlx-chronos](https://github.com/igurss/mlx-chronos) — not verified as *the* community bench suite | **watchlist** | 2026-08-10 — resolve canonical repo before citing benchmarks |

Full SHAs (for copy/paste):

| Project | Full SHA |
|---------|----------|
| mlx-lm | `a790972f0f844d81067ed45c28b524220a10c019` |
| Rapid-MLX | `753ba5b16a3a47e2cc9389207d2771c073910d75` |
| oMLX | `2130f14a1e5aa5e47158a5b4ebdf78d48a8aafb6` |
| vllm-mlx | `0dd115769ef1196a715b96b181353edacd2a4f69` |
| mlx-serve | `1f9be78e9857b3a91d0bd7eaa12bca3636bfa27e` |
| mlx-omni-server | `4f8e9ef623808b876d2fd08df8fe11745d5a2751` |
| mlx-engine | `8ae261033bc5bc16fdfc19a842bfc1d96db51348` |
| fastmlx | `1fe5d766ac05f4f31daf446f99360aa8fd92938e` |
| mlx-llm-server | `f37dad7d051eebb89e5c779bc8669a43bf356a65` |
| mlx-openai-server | `4b7d4b61586dd08b5ca1149b73003e85272ff652` |
| Toolio | `a1e4597376940f626c9aa086f49d7d9b1ad6d7cf` |
| mlx-tune | `9690fe1fed19c72f0810902b9c188d9a2625eb5b` |
| awesome-mlx | `b87192b2b687d89e6b6cdcf262b3041432902dfa` |

---

## Out of scope (adjacent MLX tooling — not Engine plugins)

These matter in the Apple Silicon ecosystem but are **not** text inference engines for our harness. Listed so research notes do not leave gaps.

| Project | Repo | What it is | Why not a plugin | Pin / status |
|---------|------|------------|------------------|--------------|
| **mlx-tune** (ARahim3) | https://github.com/ARahim3/mlx-tune | Fine-tune LLMs (and VLM/TTS/STT/… ) on MLX; Unsloth-compatible API | **Training**, not decode. HLD §7 excludes training/distill. May produce models later measured via **mlx-lm**. | See master pin table: **oos**, verified 2026-07-10, next 2026-08-10 |

---

## Foundational library

| Project | Notes | Plugin priority | Provenance |
|---------|--------|-----------------|------------|
| **mlx-lm** (ml-explore) | Official text gen/finetune library; HF / mlx-community quants; `generate` / `stream_generate`; chat REPL; prompt cache; rotating KV; sampling. **Shipped** as `MlxLmEngine` (#7). | **Done (M1)** | Pin + release in master table; workbench may use a different installed version — record `library_versions` per run |

Multimodal text+vision is separate (**mlx-vlm**); not primary for this catalog.

---

## High-performance / advanced (often servers)

| Project | Focus | Fit for our harness | Plugin priority | Provenance |
|---------|--------|---------------------|-----------------|------------|
| **Rapid-MLX** (raullenchai) | Fast OpenAI/Anthropic-compatible; tool calling; prompt caching; strong agent TTFT claims | Likely **HTTP server** — evaluate programmatic/single-req path; good *external* benchmark peer | **Evaluate (M2)** | Master pin: tip `753ba5b1…`, release **v0.10.5** |
| **oMLX** (jundot) | Continuous batching + paged/SSD KV; long agent sessions; menu bar app | Serving + disk KV — continuous batching is HLD §7 product-out; **prefix/SSD cache ideas** inform EXP #10 / #12 | **Research / metrics ideas** | Master pin: tip `2130f14a…`, release **v0.5.0** |
| **vllm-mlx** (waybarrios) | vLLM-style continuous batching, paged KV, prefix cache | Throughput/serving oriented; multi-req not our default UC | **Low for plugin** | Master pin: tip `0dd11576…`, release **v0.4.0** |
| **mlx-serve** (ddalcu) | Zig native; MLX (+ GGUF); OpenAI/Anthropic/Ollama APIs; speculative mentions | Native binary — similar to llama.cpp subprocess pattern | **Evaluate (M2)** if single-req timings are exportable | Master pin: tip `1f9be78e…`, release **v26.7.5** |

---

## API-compatible / omni servers

| Project | Focus | Plugin priority | Provenance |
|---------|--------|-----------------|------------|
| **mlx-omni-server** (madroidmaq) | Dual OpenAI + Anthropic APIs | Evaluate if thin wrapper over mlx-lm (may duplicate #7) | Master pin: **v0.5.3** / `4f8e9ef6…` |
| **MLX Engine** (lmstudio-ai) | Production backend behind LM Studio | **Evaluate** for parity claims; path may be app-bound | Master pin: commit only (no GH release) `8ae26103…` |
| **fastmlx** (arcee-ai) | Production-ready API server | Low unless unique measurement surface | Master pin: tip stale relative to release **v0.2.1** (2024); re-check activity |
| **mlx-llm-server** (mzbac) | OpenAI-compatible server | Low (thin server; quiet since 2024) | Master pin: `f37dad7d…`, no release |
| **mlx-openai-server** (cubist38) | Simple OpenAI endpoints | Low (thin server) | Master pin: **v1.8.1** / `4b7d4b61…` |
| **Toolio** (OoriData) | JSON-schema / tool-calling with MLX | **Adjacent** — structured-output EXP | Master pin: **v0.6.0** / `a1e45973…` · status **oos** for thruput plugins |

---

## Other mentions (watchlist — cluster / unpinned)

| Project / cluster | Notes | Priority | Provenance |
|-------------------|--------|----------|------------|
| **swama, SwiftLM, PicoMLXServer** | Swift-native OpenAI-compatible apps | Out of scope unless Swift FFI planned | **Unpinned cluster** — no single commit; next review 2026-08-10: either drop or pin 1 canonical repo each |
| **mlx_parallm, mlx-gui, mlx_sharding** | Parallel / UI / distributed | Not M2 default | **Unpinned cluster** — same |
| **Ollama MLX backend** | Not an “MLX engine” project; common narrative baseline | External peer only | **No pin** (product binary); cite Ollama release notes if used in a report |
| **mlx-chronos** (community) | Often cited for server comparisons | Methodology peer | **Unresolved identity** — multiple GH hits; do not cite numbers until repo is pinned (see master table) |

---

## Non-MLX engines still in our roadmap

| Backend | Issue / MS | Role | Provenance note |
|---------|------------|------|-----------------|
| **MTPLX** | #9 / M2 | Speculative MTP + custom Metal vs stock mlx-lm | Pin install revision when implementing #9 (not part of this MLX-server list) |
| **llama.cpp** Metal | #15 / M2 | Quant matrix, free-draft, non-MLX stack | Pin llama.cpp commit + build flags per experiment |
| **BaseRT** | HLD / unissued | Native Metal runtime if API usable | Unissued — no pin yet |
| **mlx-vlm** text mode | HLD / unissued | Text-only path when needed | Unissued — pin when filed |
| **Custom kernel** | M3 | Own kernels as Engine | In-repo crates (e.g. metal_stream) — pin git SHA of this monorepo |

---

## Recommended M2 plugin ladder (this repo)

1. **mlx-lm** — done (control arm); pin via `library_versions` at run time (catalog pin is landscape only).  
2. **MTPLX** (#9) — primary multi-backend scientific arm.  
3. **llama.cpp** (#15) — cross-stack / quant / free-draft.  
4. **Evaluate then maybe issue plugins for:** Rapid-MLX (**v0.10.5** / `753ba5b1…`), mlx-serve (**v26.7.5** / `1f9be78e…`), LM Studio mlx-engine (`8ae26103…`) if scriptable.  
5. **Do not prioritize as plugins:** pure multi-tenant continuous-batch servers (vllm-mlx, oMLX *as serving products*) — harvest **ideas** (prefix cache, paged KV, SSD tier) into EXPs #10–#12 instead.  
6. **Revisit:** BaseRT, mlx-vlm text, custom kernel engine after M2 core pair works.

---

## Evaluation checklist (before opening a new Engine issue)

For each candidate (re-fetch pin on the day you start work):

- [ ] Can we invoke **one** generation with fixed seed/temp under our orchestrator?
- [ ] Can we get stream timestamps or honest e2e-only metrics (no fabricated TTFT)?
- [ ] Library versions + model id pinable for `library_versions` / reproducibility?
- [ ] License OK for this personal/research workbench?
- [ ] Does it add a **distinct** scientific question vs another plugin (not just another OpenAI wrapper over mlx-lm)?
- [ ] Catalog **pin commit/release** re-verified; evidence links updated?

Pass → file `Phase 2: <name> engine plugin` under **M2**, citing the pin in the issue.  
Fail → leave in this catalog as watchlist / peer only; set status + next review.

---

## Maintenance

- On each full pass: re-run tip + `releases/latest` for every **named** row; update master pin table and **Verified** / **Next review**.  
- Prefer linking GitHub issues over duplicating DoD here.  
- Spot-check [awesome-mlx](https://github.com/raullenchai/awesome-mlx) at pinned commit (see master table).  
- Do **not** rely only on live default-branch URLs without a SHA.
