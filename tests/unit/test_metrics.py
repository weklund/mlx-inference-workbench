"""Behavioral: metric summaries express stability and sample quality, not internal formulas."""

from workbench.metrics import compute_distribution, summarize_iterations
from workbench.models import GenerationResult, GenerationStatus, ThermalReading


def _ok(ttft_ms: float, tok_s_timestamps: list[float]) -> GenerationResult:
    return GenerationResult(
        status=GenerationStatus.SUCCESS,
        output_text="x",
        token_timestamps=tok_s_timestamps,
        ttft_ms=ttft_ms,
        total_tokens=len(tok_s_timestamps),
        memory_peak_bytes=1000,
        thermal_state=ThermalReading(method="test"),
        e2e_ms=tok_s_timestamps[-1] * 1000,
    )


def test_distribution_reports_central_tendency_and_spread():
    """Known equal-spaced sample: mean and median are the midpoint; cov is finite."""
    d = compute_distribution([1.0, 2.0, 3.0, 4.0, 5.0], percentiles=(50, 90))
    assert d is not None
    assert d.n == 5
    assert d.mean == 3.0
    assert d.percentiles["p50"] == 3.0
    assert d.std > 0
    assert d.cov > 0


def test_empty_sample_has_no_distribution():
    assert compute_distribution([]) is None


def test_high_run_to_run_variance_is_marked_unstable():
    """
    Property: when decode throughput varies wildly across iterations,
    summary.unstable is True at a tight CoV threshold (HLD / Demerzel-style gate).
    """
    iters = [
        _ok(10, [0.01, 0.02]),
        _ok(10, [0.01, 1.0]),
        _ok(10, [0.01, 0.03]),
        _ok(10, [0.01, 0.9]),
        _ok(10, [0.01, 0.04]),
    ]
    summary = summarize_iterations(iters, cov_threshold=0.05)
    assert summary.valid_iterations == 5
    assert summary.decode_tok_s is not None
    assert summary.unstable is True


def test_identical_iterations_are_stable():
    """Property: zero variance is stable (CoV 0), not flagged unstable."""
    iters = [_ok(5.0, [0.01, 0.02, 0.03]) for _ in range(5)]
    summary = summarize_iterations(iters, cov_threshold=0.05)
    assert summary.decode_tok_s is not None
    assert summary.decode_tok_s.cov == 0.0
    assert summary.unstable is False


def test_failed_iterations_do_not_count_as_valid_samples():
    """Only SUCCESS contributes to valid_iterations and decode distribution."""
    good = _ok(5.0, [0.01, 0.02, 0.03])
    bad = GenerationResult(
        status=GenerationStatus.ERROR,
        output_text="",
        token_timestamps=[],
        ttft_ms=0.0,
        total_tokens=0,
        memory_peak_bytes=0,
        thermal_state=ThermalReading(method="test"),
        error_message="x",
    )
    summary = summarize_iterations([good, bad, good], cov_threshold=0.05)
    assert summary.valid_iterations == 2
    assert summary.tainted_iterations == 1
    assert summary.total_iterations == 3


def test_synthesized_timestamps_excluded_from_ttft_and_token_metrics():
    """Non-stream fabricated timestamps must not pollute TTFT / decode / SITL."""
    measured = _ok(12.0, [0.012, 0.04, 0.06])
    synthesized = GenerationResult(
        status=GenerationStatus.SUCCESS,
        output_text="x",
        token_timestamps=[0.02, 0.04, 0.06],
        ttft_ms=None,
        total_tokens=3,
        memory_peak_bytes=1000,
        thermal_state=ThermalReading(method="test"),
        e2e_ms=60.0,
        timestamps_synthesized=True,
    )
    summary = summarize_iterations([measured, synthesized], cov_threshold=0.05)
    assert summary.ttft_ms is not None
    assert summary.ttft_ms.n == 1
    assert summary.ttft_ms.mean == 12.0
    assert summary.decode_tok_s is not None
    assert summary.decode_tok_s.n == 1
    assert summary.sitl_ms is not None
    assert summary.sitl_ms.n == 1
    # e2e still aggregates both (wall-clock is real on the non-stream path)
    assert summary.e2e_ms is not None
    assert summary.e2e_ms.n == 2
