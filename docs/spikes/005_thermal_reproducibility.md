# Spike 005: Thermal Reproducibility Validation (Phase 0.5 / #3)

**Status:** PASS (protocol gate)  
**Dates:** 2026-07-09 → 2026-07-10  
**Hardware:** MacBook Pro (Mac17,7), Apple M5 Max, 128 GB unified memory, macOS 26.5  
**Model:** `mlx-community/Qwen2.5-Coder-7B-Instruct-4bit`  
**Workload:** fixed prompt (binary-search function), `max_tokens=256`, temp=0 / seed=42  
**Tooling:** `scripts/thermal_validation.py`, `scripts/thermal_analysis.py`  
**Raw data:** [`005_thermal_data.jsonl`](005_thermal_data.jsonl)

---

## 1. Question

Can we achieve **inter-run CoV &lt; 5%** on decode-related throughput (tok/s) on M5 Max with a documented thermal discipline, so that official baselines (#36) and multi-backend compares are trustworthy?

**Pass criterion (issue #3 / HLD):** protocol CoV on tok/s **&lt; 5%**.

---

## 2. Verdict

| Gate | Result |
|------|--------|
| **Protocol CoV (AC + powermetrics, exclusive-ish sessions)** | **2.57%** → **PASS** |
| Protocol N | **25** runs across 2 days (morning / afternoon / evening on day 2; morning / afternoon on day 1) |
| Mean protocol tok/s | **112.07** (p50 **111.56**, min **105.90**, max **116.46**) |
| All-rows CoV (including battery + GPU-contended evenings) | **26.6%** → FAIL if naively pooled |

**Conclusion:** Thermal reproducibility is **achievable under protocol conditions**. Uncontrolled concurrent GPU/AI load and battery operation **destroy** comparability and must be excluded from official numbers.

---

## 3. Methodology (validated)

### 3.1 Protocol conditions (required for gate rows)

1. **AC power** (not battery).  
2. **High Performance** power mode (`pmset` powermode = 2).  
3. **`powermetrics` available** (passwordless sudo for thermal + CPU/GPU power samples).  
4. **Cooldown exclusivity for the session:** no other Metal/MLX/AI workloads; GPU near idle before runs.  
5. Fixed model, prompt, token budget, seed, 30 s cooldown between timed runs.  
6. Warmup (3) then N timed runs (typically 5) per session.  
7. CPU power gate before each run (≤ 15 W CPU; wait/retry if elevated).

Rows enter the protocol gate only if: `power_source == ac`, `thermal_before.method == powermetrics`, and not `excluded_from_protocol_gate` / exploratory cohort. See `scripts/thermal_analysis.py` (`--valid-only`).

### 3.2 Session plan

| Day | Morning | Afternoon | Evening |
|-----|---------|-----------|---------|
| 1 (2026-07-09) | 5 protocol | 5 protocol | 10 exploratory (battery, no thermal) — **excluded** |
| 2 (2026-07-10) | 5 protocol | 5 protocol (~16:09 EDT) | 5 contaminated (GPU busy) **excluded** + 5 clean protocol (~22:56 EDT) |

Day-1 clean evening was not collected under AC+powermetrics; day-2 clean evening completes the multi-TOD design for day 2.

### 3.3 Commands

```bash
# Per session
uv run python scripts/thermal_validation.py --runs 5 --session evening --day 2 --cooldown 30

# Protocol gate only
uv run python scripts/thermal_analysis.py --valid-only
```

### 3.4 Preflight checklist (official runs)

- [ ] Plugged in (AC)  
- [ ] High Performance mode  
- [ ] Thermal pressure **Nominal**  
- [ ] GPU power low (order of ~0–1 W at idle; not tens of watts)  
- [ ] No concurrent AI / Camera Hub / heavy GPU Chrome if claiming protocol numbers  
- [ ] `powermetrics` works non-interactively  

---

## 4. Results

### 4.1 Protocol gate (N=25)

| Metric | Value |
|--------|------:|
| Mean tok/s | 112.07 |
| Std | 2.88 |
| **CoV** | **2.57%** |
| p50 / p90 / p99 | 111.56 / 115.44 / 116.35 |
| Min / Max | 105.90 / 116.46 |

### 4.2 Per-session (protocol only)

| Session | N | Mean tok/s | CoV% |
|---------|--:|----------:|-----:|
| Day 1 morning | 5 | 108.57 | 1.64 |
| Day 1 afternoon | 5 | 109.79 | 0.48 |
| Day 2 morning | 5 | 115.13 | 0.73 |
| Day 2 afternoon | 5 | 115.07 | 0.43 |
| Day 2 evening (clean) | 5 | 111.78 | 1.03 |

Within-session CoV is typically **&lt; 2%**. Cross-session mean level still varies a few tok/s.

### 4.3 Day-to-day and time-of-day

| Comparison | Observation |
|------------|-------------|
| Day 1 vs Day 2 (protocol) | Means ~**109.2** vs ~**114.0** tok/s (~**4.3%** drift) |
| Morning / afternoon / evening (protocol) | Means ~111.9 / 112.4 / 111.8 — **no large TOD bias** when exclusive |
| Pre-run power vs tok/s | Negative correlation (higher pre-run draw ↔ lower tok/s); combined power stronger than CPU alone |

Day-to-day level shift is **below** the 5% CoV gate when all protocol points are pooled, but large enough that **official compares should prefer same-day or same-session baselines** when effect sizes are small.

---

## 5. Negative evidence (why exclusivity matters)

### 5.1 Day 2 evening — GPU contention (excluded)

~18:03–18:07 EDT, concurrent AI/GPU load:

| Run | tok/s | Pre-run GPU | Pressure |
|----:|------:|------------:|----------|
| 1 | 98.7 | ~9 W | Nominal |
| 2–5 | 18–23 | ~16–31 W | → **Heavy** |

Session CoV **~84%**. Tagged `excluded_from_protocol_gate` with reasons (`session_gpu_contention`, `gpu_busy_pre_run`, `thermal_pressure_heavy`, …).

**Finding:** CPU-only power gate (15 W) did **not** mark these runs `contested` while the GPU was busy. Protocol must treat **GPU / combined power** and **Heavy** pressure as exclusion criteria.

### 5.2 Day 1 evening — battery + no thermal (excluded)

Stable ~114 tok/s on battery without powermetrics — useful as exploratory variance under different power source, **not** protocol.

### 5.3 Pooled all rows

Including excluded evenings → overall CoV **~27%** and FAIL. Demonstrates that **silent mixing of clean and contended runs invalidates the gate**.

---

## 6. Recommended lab policy

| Run class | Machine policy | Counts as official? |
|-----------|----------------|---------------------|
| Protocol thermal / official baseline / publishable compare | **Exclusive** for the session; AC + high perf + Nominal + idle GPU | Yes, if gates pass |
| Smoke / adapter bring-up / debug | Shared load OK | **No** tok/s claims |
| Intentional multi-tenant study | Shared by design; document co-tenants | Separate experiment only |

**Harness follow-ups (not required to close #3):**

1. Contestation / quality tags on **GPU or combined** power, not CPU alone.  
2. Fail or taint when thermal pressure is **Heavy**.  
3. Document exclusive-use rule in experiment README / official baseline (#36).

---

## 7. Implications for #36 and beyond

1. **#3 hard gate is satisfied** under the methodology above.  
2. **#36 (official mlx-lm baseline)** may proceed using this protocol (exclusive machine, AC, high perf, powermetrics, cooldown, quality tags).  
3. Provisional baselines from before this report remain **non-official**.  
4. Multi-backend claims (e.g. mlx-lm vs MTPLX) inherit the same exclusivity and thermal rules.

---

## 8. HLD updates

See **HLD §22** (thermal risk closed with empirical pointer to this spike) and assumption on thermal reproducibility.

---

## 9. Checklist (issue #3 DoD)

- [x] Test model + fixed prompt / seed  
- [x] Multi-session measurements over 2 days  
- [x] Inter-run CoV for tok/s (protocol)  
- [x] PASS CoV &lt; 5%  
- [x] Report with raw data table path + methodology  
- [x] Negative evidence (GPU contention, battery) documented  
- [x] HLD §22 updated  

**Artifacts**

- Data: `docs/spikes/005_thermal_data.jsonl`  
- Scripts: `scripts/thermal_validation.py`, `scripts/thermal_analysis.py`  
- Analysis re-run: `uv run python scripts/thermal_analysis.py --valid-only`
