"""Behavioral: statistical comparison requires enough valid samples (HLD edge cases)."""

from workbench.metrics import summarize_iterations
from workbench.models import (
    ENGINE_INTERFACE_VERSION,
    METRICS_SCHEMA_VERSION,
    GenerationResult,
    GenerationStatus,
    RunMetadata,
    RunRecord,
    ThermalReading,
)
from workbench.statistics_compare import compare_runs


def _success(gap: float = 0.05) -> GenerationResult:
    t0, t1 = 0.01, 0.01 + gap
    return GenerationResult(
        status=GenerationStatus.SUCCESS,
        output_text="x",
        token_timestamps=[t0, t1],
        ttft_ms=t0 * 1000,
        total_tokens=2,
        memory_peak_bytes=100,
        thermal_state=ThermalReading(method="off"),
        e2e_ms=t1 * 1000,
    )


def _record(run_id: str, n_success: int) -> RunRecord:
    iters = [_success() for _ in range(n_success)]
    # pad with errors so total_iterations > valid when n_success small
    while len(iters) < max(n_success, 1):
        iters.append(
            GenerationResult(
                status=GenerationStatus.ERROR,
                output_text="",
                token_timestamps=[],
                ttft_ms=0.0,
                total_tokens=0,
                memory_peak_bytes=0,
                thermal_state=ThermalReading(method="off"),
                error_message="x",
            )
        )
    metrics = summarize_iterations(iters)
    meta = RunMetadata(
        run_id=run_id,
        experiment_name="q",
        backend="stub",
        model_name="m",
        quantization="q4",
        prompt_dataset_path="p",
        prompt_dataset_hash="same",
        hardware_profile="m5_max_128gb",
        hardware_fingerprint={"chip": "Apple M5 Max"},
        git_sha=None,
        library_versions={},
        random_seed=42,
        schema_version="1.0",
        metrics_schema_version=METRICS_SCHEMA_VERSION,
        engine_interface_version=ENGINE_INTERFACE_VERSION,
        thermal_monitoring="off",
    )
    return RunRecord(metadata=meta, metrics=metrics, iterations=[i.to_dict() for i in iters])


def test_single_valid_iteration_cannot_claim_statistical_comparison():
    """
    HLD: only 1 valid iteration → no statistical distribution;
    Comparability Gate rejects comparisons involving this run.
    """
    thin = _record("thin", n_success=1)
    solid = _record("solid", n_success=6)
    assert thin.metrics.quality_tag == "insufficient_data"
    assert solid.metrics.quality_tag == "full"

    result = compare_runs(thin, solid, metric_name="decode_tok_s")
    assert result.comparable is False
    assert result.significant_at_0_05 is None
    assert result.verdict == "blocked_by_comparability_gate"


def test_two_full_quality_runs_remain_comparable():
    a = _record("a", n_success=6)
    b = _record("b", n_success=6)
    result = compare_runs(a, b, metric_name="decode_tok_s")
    assert result.comparable is True


def test_zero_acceptance_speculative_run_blocked_from_performance_compare():
    """Mean acceptance_rate 0 ⇒ quality speculative_no_accept ⇒ gate blocks."""
    solid = _record("solid", n_success=6)
    zero_acc_iters = []
    for _ in range(6):
        r = _success()
        r.acceptance_rate = 0.0
        zero_acc_iters.append(r)
    zero_metrics = summarize_iterations(zero_acc_iters)
    assert zero_metrics.quality_tag == "speculative_no_accept"
    zero_run = RunRecord(
        metadata=solid.metadata.__class__(
            **{**solid.metadata.__dict__, "run_id": "zero_acc", "backend": "mtplx"}
        ),
        metrics=zero_metrics,
        iterations=[i.to_dict() for i in zero_acc_iters],
    )
    result = compare_runs(solid, zero_run, metric_name="e2e_ms")
    assert result.comparable is False
    assert result.verdict == "blocked_by_comparability_gate"
    assert any("speculative_no_accept" in v for v in result.violations)
