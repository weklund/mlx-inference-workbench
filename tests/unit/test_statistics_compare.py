import numpy as np
import pytest

from workbench.comparability import GateResult
from workbench.metrics import summarize_iterations
from workbench.models import (
    ENGINE_INTERFACE_VERSION,
    METRICS_SCHEMA_VERSION,
    DistributionStats,
    GenerationResult,
    GenerationStatus,
    RunMetadata,
    RunRecord,
    ThermalReading,
)
from workbench.statistics_compare import compare_distributions, compare_runs


def _dist(values: list[float]) -> DistributionStats:
    arr = np.asarray(values, dtype=float)
    return DistributionStats(
        n=len(values),
        mean=float(arr.mean()),
        std=float(arr.std(ddof=1)),
        cov=float(arr.std(ddof=1) / arr.mean()),
        trimmed_mean=float(arr.mean()),
        percentiles={"p50": float(np.median(arr))},
        values=tuple(values),
    )


def _minimal_run(run_id: str) -> RunRecord:
    iters = [
        GenerationResult(
            status=GenerationStatus.SUCCESS,
            output_text="x",
            token_timestamps=[0.01, 0.02],
            ttft_ms=10.0,
            total_tokens=2,
            memory_peak_bytes=1,
            thermal_state=ThermalReading(method="off"),
            e2e_ms=20.0,
        )
        for _ in range(3)
    ]
    metrics = summarize_iterations(iters)
    meta = RunMetadata(
        run_id=run_id,
        experiment_name="t",
        backend="stub",
        model_name="m",
        quantization="q",
        prompt_dataset_path="p",
        prompt_dataset_hash="h",
        hardware_profile="m5",
        hardware_fingerprint={},
        git_sha=None,
        library_versions={},
        random_seed=42,
        schema_version="1.0",
        metrics_schema_version=METRICS_SCHEMA_VERSION,
        engine_interface_version=ENGINE_INTERFACE_VERSION,
        thermal_monitoring="off",
    )
    return RunRecord(metadata=meta, metrics=metrics, iterations=[i.to_dict() for i in iters])


def test_no_significant_difference_identical():
    vals = [10.0, 10.1, 9.9, 10.0, 10.05]
    gate = GateResult(comparable=True, violations=())
    r = compare_distributions(_dist(vals), _dist(vals), metric_name="decode_tok_s", gate=gate)
    assert r.comparable
    assert r.verdict == "no_significant_difference"
    assert r.significant_at_0_05 is False


def test_blocked_gate():
    gate = GateResult(comparable=False, violations=("prompt_dataset_hash: a != b",))
    r = compare_distributions(_dist([1, 2, 3]), _dist([4, 5, 6]), metric_name="x", gate=gate)
    assert r.verdict == "blocked_by_comparability_gate"


@pytest.mark.parametrize(
    "bad_metric",
    ["quality_tag", "unstable", "valid_iterations", "not_a_metric", "metrics_schema_version"],
)
def test_compare_runs_rejects_non_distribution_metrics(bad_metric: str):
    a, b = _minimal_run("a"), _minimal_run("b")
    with pytest.raises(ValueError, match="Unsupported metric_name"):
        compare_runs(a, b, metric_name=bad_metric)


def test_compare_runs_missing_distribution_is_insufficient_data_not_error():
    """None DistributionStats (e.g. no measured TTFT) keeps existing soft path."""
    a, b = _minimal_run("a"), _minimal_run("b")
    # acceptance_rate is a supported distribution field but often None on non-speculative runs
    r = compare_runs(a, b, metric_name="acceptance_rate")
    assert r.verdict == "insufficient_data_for_test"
