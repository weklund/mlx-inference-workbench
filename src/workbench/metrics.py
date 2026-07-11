"""Pure metrics — no I/O, no engines. Single place for percentile / CoV math."""

from __future__ import annotations

from collections.abc import Sequence
import math

import numpy as np

from workbench.models import (
    DistributionStats,
    GenerationResult,
    GenerationStatus,
    MetricSummary,
)


def compute_distribution(
    values: Sequence[float],
    percentiles: Sequence[int] = (50, 90, 95, 99),
    trim_top_fraction: float = 0.01,
) -> DistributionStats | None:
    """Compute distribution stats. Returns None if no values."""
    if not values:
        return None

    arr = np.asarray(values, dtype=np.float64)
    n = int(arr.size)
    mean = float(np.mean(arr))
    std = float(np.std(arr, ddof=1)) if n > 1 else 0.0
    cov = (std / mean) if mean != 0.0 else (0.0 if std == 0.0 else math.inf)

    # Trimmed mean: drop top trim_top_fraction (tm99 → drop top 1%)
    if n >= 2 and trim_top_fraction > 0:
        k = max(0, math.floor(n * trim_top_fraction))
        if k > 0 and k < n:
            trimmed = np.sort(arr)[: n - k]
            trimmed_mean = float(np.mean(trimmed))
        else:
            trimmed_mean = mean
    else:
        trimmed_mean = mean

    pct: dict[str, float] = {}
    for p in percentiles:
        pct[f"p{p}"] = float(np.percentile(arr, p))

    return DistributionStats(
        n=n,
        mean=mean,
        std=std,
        cov=float(cov),
        trimmed_mean=trimmed_mean,
        percentiles=pct,
        values=tuple(float(x) for x in arr),
    )


def _decode_tok_s(result: GenerationResult) -> float | None:
    """Tokens after first / time after first token (requires measured per-token times)."""
    if result.total_tokens < 2 or len(result.token_timestamps) < 2:
        return None
    # timestamps are absolute seconds from start of generation
    t0 = result.token_timestamps[0]
    t_last = result.token_timestamps[-1]
    elapsed = t_last - t0
    if elapsed <= 0:
        return None
    return (result.total_tokens - 1) / elapsed


def _sitl_ms(result: GenerationResult) -> float | None:
    """Mean inter-token latency in ms (requires measured per-token times)."""
    if result.total_tokens < 2 or len(result.token_timestamps) < 2:
        return None
    gaps = np.diff(result.token_timestamps)
    if gaps.size == 0:
        return None
    return float(np.mean(gaps) * 1000.0)


def _e2e_ms(result: GenerationResult) -> float:
    if result.e2e_ms is not None:
        return result.e2e_ms
    if result.token_timestamps:
        return float(result.token_timestamps[-1] * 1000.0)
    if result.ttft_ms is not None:
        return result.ttft_ms
    return 0.0


def _measured_ttft_ms(result: GenerationResult) -> float | None:
    """TTFT only when engines recorded it (stream path); e2e-only leaves None."""
    return result.ttft_ms


def summarize_iterations(
    iterations: Sequence[GenerationResult],
    *,
    percentiles: Sequence[int] = (50, 90, 95, 99),
    cov_threshold: float = 0.05,
    min_full_confidence: int = 5,
) -> MetricSummary:
    """Aggregate successful iterations into MetricSummary."""
    total = len(iterations)
    valid = [r for r in iterations if r.status == GenerationStatus.SUCCESS]
    tainted = total - len(valid)

    if len(valid) == 0 or len(valid) == 1:
        quality = "insufficient_data"
    elif len(valid) < min_full_confidence:
        quality = "low_confidence"
    else:
        quality = "full"

    ttft_vals = [v for r in valid if (v := _measured_ttft_ms(r)) is not None]
    ttft = compute_distribution(ttft_vals, percentiles)
    decode_vals = [v for r in valid if (v := _decode_tok_s(r)) is not None]
    decode = compute_distribution(decode_vals, percentiles)
    sitl_vals = [v for r in valid if (v := _sitl_ms(r)) is not None]
    sitl = compute_distribution(sitl_vals, percentiles)
    e2e = compute_distribution([_e2e_ms(r) for r in valid], percentiles)
    mem = compute_distribution([float(r.memory_peak_bytes) for r in valid], percentiles)

    acc_vals = [r.acceptance_rate for r in valid if r.acceptance_rate is not None]
    acc = compute_distribution(acc_vals, percentiles) if acc_vals else None

    # Speculative path drafted but accepted nothing: not a valid performance
    # conclusion for MTP/speculative gains (overhead without accept). Keep
    # metrics for diagnosis; demote quality so the comparability gate blocks.
    if quality != "insufficient_data" and acc is not None and acc.n >= 1 and acc.mean == 0.0:
        quality = "speculative_no_accept"

    # Unstable if primary throughput CoV exceeds threshold (need n>=2)
    unstable = False
    if (decode is not None and decode.n >= 2 and decode.cov > cov_threshold) or (
        ttft is not None and ttft.n >= 2 and decode is None and ttft.cov > cov_threshold
    ):
        unstable = True

    return MetricSummary(
        ttft_ms=ttft,
        decode_tok_s=decode,
        sitl_ms=sitl,
        e2e_ms=e2e,
        memory_peak_bytes=mem,
        acceptance_rate=acc,
        valid_iterations=len(valid),
        total_iterations=total,
        tainted_iterations=tainted,
        unstable=unstable,
        quality_tag=quality,
    )
