# Spike 00: MTPLX Familiarization

> **Date:** 2026-07-09
> **Hardware:** Apple M5 Max, 128 GB unified memory, macOS 26.5.2
> **MTPLX version:** 2.0.1
> **MLX version:** 0.31.2

---

## 1. Architecture Overview

MTPLX is a native Apple Silicon inference engine implementing **Multi-Token Prediction (MTP) speculative decoding**. It uses the target model's own MTP heads (trained into Qwen3.5/3.6, DeepSeek, GLM, Nemotron-H, Step3.5) to draft multiple tokens, then verifies them in a single batched forward pass with exact rejection sampling.

The key insight: the verification forward pass creates a "dead zone" at M=4..6 rows where stock MLX kernels are suboptimal (tuned for M=1 decode or M>>16 prefill). MTPLX fills this gap with custom Metal kernels that deliver 1.1–2.4× speedup on these specific shapes.

### Package Structure

```
mtplx/
├── kernels/              # Custom Metal kernels (mx.fast.metal_kernel)
│   ├── verify_mlp_fused.py     # Fused gate+up+SwiGLU QMV4 (M<=6)
│   ├── native_gdn_tail.py      # Fused norm+gate+output (C++ extension, not shipped)
│   ├── fused_norm.py            # Fused add+RMSNorm, GDN norm+gate
│   ├── lm_head_topk.py         # Fused LM head + top-k (avoids full vocab materialization)
│   ├── logits_topk.py          # Dense logits top-k + logsumexp
│   ├── sdpa_2pass.py           # Two-pass attention for tiny verify windows (q<=16)
│   ├── sdpa_2pass_paged.py     # Paged KV variant
│   ├── sdpa_2pass_paged_q8.py  # Int8 KV variant (CLOSED LANE - didn't beat bf16)
│   ├── sdpa_gqa_packed.py      # GQA packed attention (highest perf, q=2..4)
│   └── copy_leaf.py            # Cache detachment helper
├── verify_kernels.py     # Primary verify-shape QMM (M=4..6, MSG + Split-K variants)
├── verify_qmv.py         # Earlier M=3/M=4 QMV probes + fused SwiGLU-into-down
├── nax_verify.py          # NAX (M5 MetalPerformancePrimitives) verify matmul
├── gdn_capture.py         # Innovation tape: GDN state capture/replay (8 Metal kernels)
├── speculative.py         # Acceptance/rejection math (Leviathan & Chen 2023)
├── generation.py          # Full generation loop (generate_ar, generate_mtp1, generate_mtpk)
├── native_mlp.py          # MLP patch for verify-shaped forwards
├── mtp_patch.py           # MTP head injection onto loaded models
├── adaptive.py            # Adaptive draft depth tuning (EV-based)
├── cache_state.py         # KV cache with snapshot/rollback
├── thermal.py             # Thermal monitoring + fan control
├── thermal_sidecar.py     # ThermalForge sidecar
├── runtime.py             # MTPLXRuntime dataclass + load()
├── server/                # OpenAI-compatible server
├── backends/              # Model-specific MTP backends (qwen3, deepseek, glm, etc.)
└── batching/              # Continuous batching + scheduler
```

---

## 2. The Speculative Decoding Flow

### 2.1 High-Level Cycle

```
┌─────────────────────────────────────────────────────────────────┐
│                    One Speculative Decode Cycle                   │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  1. DRAFT (D tokens via MTP heads)                               │
│     ├── MTP forward: concat(norm(embed(token)), norm(hidden))    │
│     ├── → fc → decoder_layer → norm → lm_head → logits          │
│     ├── Sample draft token from q_i                              │
│     └── Repeat for depths 1..D (adaptive: may stop early)        │
│                                                                   │
│  2. VERIFY (target model forward, M=D+1 rows)                   │
│     ├── Forward all D+1 tokens through target model              │
│     ├── GDN capture: record innovation tape per layer            │
│     ├── Custom kernels: verify_m4/m6 QMM, fused MLP, SDPA       │
│     └── Produce target logits for positions [0..D]               │
│                                                                   │
│  3. ACCEPT/REJECT (per-token, left-to-right)                    │
│     ├── For i in 0..D-1:                                         │
│     │   alpha = min(1, p_target(token_i) / q_draft(token_i))     │
│     │   if random() <= alpha: ACCEPT, continue                   │
│     │   else: REJECT, sample from max(0, p-q)/Z, break          │
│     └── If all accepted: bonus token from target logits[D]       │
│                                                                   │
│  4. COMMIT (state restoration)                                   │
│     ├── Trim KV cache to accepted prefix length                  │
│     ├── Replay innovation tape to restore GDN state at K         │
│     └── Update adaptive depth policy                             │
│                                                                   │
└─────────────────────────────────────────────────────────────────┘
```

### 2.2 Draft Phase Details

The MTP head is a separate small decoder block (patched onto the model at load time via `inject_mtp_support`):

```python
# mtp_patch.py: _MTPLXTextModel._mtp_core()
hidden_input = pre_fc_norm_hidden(target_hidden)   # RMSNorm
embed_input  = pre_fc_norm_embedding(embed(token)) # Embed + RMSNorm
x = fc(concat(hidden_input, embed_input))          # Linear projection
x = mtp_decoder_layer(x, attention_mask, cache)    # Full transformer layer
x = final_norm(x)                                   # RMSNorm
logits = lm_head(x)                                 # Quantized linear → vocab
```

Each depth produces logits and updated hidden states that feed into the next depth.

**Draft LM head optimization:** The draft head can be requantized to 2-bit (vs target's 4-bit). Since draft quality only affects acceptance rate, not correctness, this is a pure throughput win.

### 2.3 Verify Phase Details

The verify forward processes D+1 tokens simultaneously. This is where custom kernels dominate:

- **All QuantizedLinear calls** are routed through `nax_verify.py`'s monkey-patched `nn.QuantizedLinear.__call__` for M=4..16
- **MLP blocks** use fused gate+up+SwiGLU via `verify_mlp_fused.py` kernels
- **Attention** uses `sdpa_gqa_packed.py` for q_len=2..4 (highest perf) or `sdpa_2pass.py` for q_len=5..16
- **GDN layers** (Qwen3.5/3.6 hybrid) use capture kernels from `gdn_capture.py`

### 2.4 Acceptance/Rejection Math

Implements exact speculative sampling (Leviathan & Chen 2023):

```
acceptance_probability(p, q, token) = min(1, p[token] / q[token])
residual_distribution(p, q) = normalize(max(0, p - q))
```

**Correctness guarantee:** The marginal output distribution of the speculative sampling process exactly equals the target distribution, regardless of draft quality. Draft quality only affects throughput (acceptance rate).

**Sparse optimization:** Both target and draft distributions are `SparseDistribution` (top-k=20 entries). The residual is computed over the union of supports.

---

## 3. The Innovation Tape (GDN Capture/Replay)

### 3.1 The Problem

Qwen3.5/3.6 uses hybrid attention + Gated Delta Networks (GDN) — linear recurrent layers:

```
S[t] = g[t] * S[t-1] + k[t] * delta[t]
```

After verifying D+1 positions and accepting only K, the GDN state must be restored to position K. Unlike attention (where you just trim the KV cache), recurrent state is not position-addressable without recomputation.

### 3.2 The Solution: Per-Position State Capture

During the verify forward, capture kernels run the GDN recurrence and output the intermediate state at **every** position. On partial acceptance, index into the captured states at position K — instant rollback.

### 3.3 Tape Mode (Memory-Optimized)

Instead of storing full `(B, T, Hv, Dv, Dk)` state tensors (expensive), tape mode records only the scalar "innovation" delta:

```
tape[t] = delta[t]   (the new information at each step)
```

To reconstruct state at position K:
```
S[0] = base_state
for t in 0..K-1:
    S[t+1] = g[t] * S[t] + k[t] * tape[t]
```

This is analogous to a **write-ahead log** in databases: store minimal deltas, replay to reconstruct any intermediate state.

### 3.4 Metal Kernels for GDN Capture

| Kernel | Purpose |
|--------|---------|
| `mtplx_linear_conv1d_capture` | Conv1d with per-position state output |
| `mtplx_linear_gated_delta_capture_v2` | GDN recurrent, full per-position states |
| `mtplx_linear_gated_delta_final_v1` | GDN recurrent, final state only |
| `mtplx_linear_gated_delta_from_conv_v1` | Fused conv+GDN+RMSNorm, full capture |
| `mtplx_linear_gated_delta_from_conv_stream_v1` | Streaming with partial capture window |
| `mtplx_linear_gated_delta_from_conv_tape_v1` | **Tape mode:** output + final_state + tape |
| `mtplx_linear_gated_delta_from_conv_tape_replay_v1` | **Replay:** reconstruct state at K from tape |
| `mtplx_linear_gated_delta_from_conv_inline_g_v1` | Inline gate computation from A_log/dt_bias |

---

## 4. Custom Metal Kernel Catalog

### 4.1 Verify-Shape Quantized Matmul (the primary win)

**Files:** `verify_kernels.py`, `verify_qmv.py`, `nax_verify.py`

**Problem:** Stock MLX's quantized matmul is tuned for M=1 (decode) and M>>16 (prefill). The verify window at M=4..6 is a dead zone with poor occupancy, suboptimal tiling, and high relative overhead.

**Solution:** Shape-specialized kernels:

| Kernel Family | M | Strategy | Key Optimization |
|---------------|---|----------|------------------|
| `verify_m4_kp1` | 4 | Single simdgroup, no barriers | Zero threadgroup memory overhead |
| `verify_m4_ksplit` | 4 | K-split across 2-4 simdgroups | Better for wide K (>4096) |
| `verify_m4_bn6` | 4 | BN=6 wide tile | 1.5× fewer threadgroups |
| `verify_m6_ksplit` | 5-6 | K-split simdgroups | Beats NAX tile at these shapes |
| `verify_m16_nax` | 8-16 | Apple MetalPerformancePrimitives `matmul2d` | M5 NAX units (macOS 26.2+) |

**Routing:** `install_nax_qlinear_patch()` monkey-patches `nn.QuantizedLinear.__call__`. During decode, M=4..16 calls go through custom kernels; M=1 and M>16 stay on stock MLX.

**Implementation:** All via `mx.fast.metal_kernel` with inline MSL. Key patterns:
- `Vec8 = vec<T, 8>` for coalesced 8-element activation loads
- `_Pragma("unroll")` on inner FMA blocks
- Column-major dequant+FMA (better compiler pipelining)
- 24-accumulator ceiling (32 kills GPU occupancy)
- 6-bit hexpack support (16 values from 3 uint32 words)

### 4.2 Fused MLP (Gate + Up + SwiGLU)

**File:** `verify_mlp_fused.py`

Replaces two separate 4-bit QMV calls (gate_proj, up_proj) + SwiGLU activation with one Metal dispatch:

| Variant | Strategy | When Used |
|---------|----------|-----------|
| Default | Both projections in same simdgroup registers | M<=6, general case |
| Rowwise | One threadgroup per row | Independent row parallelism |
| Split | Half simdgroups → gate, half → up, barrier, fuse | Better for larger M |

Also includes `small_m_qmm4_matmul` using Apple's `simdgroup_matrix` hardware MMA instructions (the only kernel using this path).

### 4.3 Attention (Two-Pass + GQA Packed)

**Files:** `sdpa_2pass.py`, `sdpa_2pass_paged.py`, `sdpa_gqa_packed.py`

| Kernel | q_len | Key Innovation |
|--------|-------|----------------|
| `sdpa_gqa_packed` | 2-4 | Loads KV once, carries all q rows in registers as `float4`. Single `simd_shuffle_xor` butterfly reduction. Measured 207 GB/s vs stock 160 GB/s at 128k context. |
| `sdpa_2pass` | 1-16 | Block-strided online softmax with hardware-aware block count (architecture suffix 'd'=desktop/'s'=standard) |
| `sdpa_2pass_paged` | 1-16 | Same, reading from vLLM-style physically-paged KV |

**Hardware awareness:** `_compute_blocks()` inspects `mx.device_info()["architecture"]` last character to tune block count for M5 Max vs M5 Pro.

### 4.4 Fused Norm Kernels

**File:** `fused_norm.py`

| Kernel | Fuses |
|--------|-------|
| `fused_add_rmsnorm` | Residual add + RMSNorm in one pass |
| `fused_gdn_norm_gate` | RMSNorm + SiLU gating for GDN |

Both use two-pass pattern: compute sum-of-squares → barrier → apply normalization. Vectorized N_READS=4 with `simd_sum` reduction.

### 4.5 LM Head + Top-K Fusion

**File:** `lm_head_topk.py`

Avoids full vocabulary materialization during draft sampling. Each threadgroup computes BN output logits and maintains an insertion-sorted top-k list in threadgroup memory. Global merge via Python `argpartition`.

---

## 5. Adaptive Draft Depth

### 5.1 Simple Policy (AdaptiveDepthPolicy)

- Increases depth after 4 consecutive full accepts
- Decreases after 1 early rejection (rejected at position ≤ depth/2)
- Bounded by [min_depth, max_depth]

### 5.2 EV Policy (ExpectedValueDepthPolicy) — Production

Cost-aware gate before each additional draft token:

```
prefix_prob = product(ewma_acceptance[0..D-1])
next_prob   = ewma_acceptance[D]
confidence  = f(top2_margin, top1_prob)  # tanh-scaled

expected_extra_accept = prefix_prob * next_prob * confidence

extra_cost = draft_cost_s + extra_verify_cost_s  # defaults: 4.8ms + 6.0ms
required = max(min_threshold, extra_cost * baseline_tok_s * (1 + safety_margin))

CONTINUE if expected_extra_accept >= required
```

Key features:
- EWMA (α=0.12) of per-position acceptance rates
- Default priors: depth 1=0.92, depth 2=0.64, depth 3=0.32
- Warmup (first 4 cycles) forces full depth
- Exploration every 32 cycles forces full depth to prevent self-locking
- Uses draft logit `top2_margin` as confidence signal

---

## 6. Kernel Registration & Fallback Architecture

MTPLX uses **opt-in kernel selection** rather than op replacement:

1. **Eligibility predicates** (`is_*_eligible()`) check shape, quantization, group_size, dtype
2. **Environment-variable gating** (e.g., `MTPLX_FUSE_POST_NORM_RESIDUAL=1`)
3. **Runtime self-check** (`kernel_selfcheck.py`): on model load, each kernel lane is tested against stock MLX on tiny synthetic tensors. Mismatching lanes are disabled for the process lifetime.
4. **Graceful fallback:** every kernel function has a `else: return stock_mlx_equivalent(...)` path

No monkey-patching of MLX internals except `nn.QuantizedLinear.__call__` (via `install_nax_qlinear_patch`).

---

## 7. Programmatic API (OQ#1 Resolution)

**Yes, MTPLX can be driven programmatically:**

```python
import mtplx

# Load model with MTP support
rt = mtplx.load("/path/to/model")

# Access: rt.model, rt.tokenizer, rt.mtp_enabled, rt.contract
# Forward: rt.forward_ar(input_ids, cache)
# Draft: rt.draft_mtp(hidden, next_token_ids, mtp_cache)
# Generate: generation.generate_mtpk(rt, prompt_ids, ...)
```

The `mtplx.load()` → `MTPLXRuntime` → `generate_mtpk()` path is fully programmatic. The CLI and server are thin wrappers. An engine plugin for our harness can drive this directly.

---

## 8. Hardware Profile (M5 Max)

```
Chip: Apple M5 Max
Apple Silicon generation: m5
Unified memory: 128.0 GB
M5 TensorOps eligible: true
MLX version: 0.31.2
macOS: 26.5.2

MTPLX hardware-aware tuning:
- SDPA block count: uses 'd' (desktop/Max) architecture path → 128-1024 blocks
- GQA packed blocks: capacity >= 65536 → 1024 blocks
- Verify kernel NSG: M4=8 simdgroups, M6=4 simdgroups (tunable via env)
- NAX verify: eligible for MetalPerformancePrimitives matmul2d (macOS >= 26.2)
```

---

## 9. Opportunities for Custom Kernel Work

### 9.1 Closed Lanes / Experimental Paths

1. **Q8 paged SDPA** (`sdpa_2pass_paged_q8.py`) — Measured 0.8× vs dense bf16. The premise (KV-DRAM-bound) was wrong at these shapes. Potential opportunity: try at longer contexts (128k+) where KV bandwidth truly dominates, or try different quantization (asymmetric, per-head adaptive).

2. **Native GDN tail** (C++ extension) — Not shipped in production package. The fused norm+gate+output projection path exists in code but requires a compiled native extension. Could be rebuilt with Rust + metal-rs for our workbench.

### 9.2 Novel Opportunities

1. **Adaptive kernel selection at runtime:** Currently kernel lanes are selected via environment variables or eligibility predicates at load time. A lightweight dispatcher that profiles the first few iterations and selects the best variant per layer per model would be valuable (ties to our "adaptive self-tuning controller" spike).

2. **Wider M support (M=8..16):** The `verify_m8_ksplit` path exists but is documented as a "closed branch" due to register pressure. The M5 Max may have more register capacity than M4 — worth re-exploring.

3. **Innovation tape compression:** Current tape stores float32 per (head, dv, t). Could compress to fp16 or even int8 with minimal quality loss (the delta values have bounded range). Memory savings could enable deeper draft depths.

4. **Fused verify MLP for 8-bit models:** The production fused MLP only handles 4-bit. With Qwen3 models increasingly available at 8-bit, a `verify_mlp_fused_q8` kernel could be high-value.

5. **Speculative verify overlap (SSD-style):** The current flow is sequential: draft → verify → accept. Overlapping the next draft with the current verification (predicting acceptance outcomes) could hide latency — directly from the SSD/Saguaro paper ideas.

---

## 10. Key Learnings

1. **All production kernels use `mx.fast.metal_kernel`** — not C++ extensions or custom Metal libraries. This means prototyping new kernels is accessible from Python.

2. **The verify shape gap (M=4..6) is the primary optimization target.** Stock MLX is poor here because its kernels were designed for M=1 or M>>16.

3. **The innovation tape is the key architectural enabler** for speculative decoding on hybrid attention+recurrent models. Without it, rollback requires expensive re-computation.

4. **Kernel self-check at load time** is a brilliant pattern for hardware portability — automatically disables any kernel that produces wrong results on untested hardware.

5. **The EV depth policy** is production-critical — it prevents wasting cycles on deep drafts when acceptance drops below break-even. This is the "adaptive self-tuning controller" concept already implemented at the depth-selection level.

6. **MTPLX is fully programmatic** (`mtplx.load()` → `MTPLXRuntime`). We can build our harness engine plugin directly against this API.

---

## 11. Next Steps

- **Phase 0.5:** Use MTPLX or mlx-lm for the thermal reproducibility spike
- **Phase 1:** Implement MTPLX engine plugin using `mtplx.load()` + `generate_mtpk()` API
- **Phase 2:** Reproduce the verify_m4/m6 kernels from scratch using `mx.fast.metal_kernel` to build deep understanding
- **Phase 3:** Explore the novel opportunities identified (wider M, tape compression, 8-bit fused MLP)
