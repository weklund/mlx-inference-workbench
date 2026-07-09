"""Behavioral: comparing two runs must respect the gate and not invent significance."""

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


def _iter(decode_gap: float) -> GenerationResult:
    """Two tokens: first at 0.01s, second at 0.01+gap → decode tok/s ≈ 1/gap."""
    t0, t1 = 0.01, 0.01 + decode_gap
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


def _record(
    run_id: str,
    gaps: list[float],
    *,
    prompt_hash: str = "same-hash",
    hardware_profile: str = "m5_max_128gb",
    quantization: str = "q4",
    backend: str = "stub",
) -> RunRecord:
    iters = [_iter(g) for g in gaps]
    metrics = summarize_iterations(iters)
    meta = RunMetadata(
        run_id=run_id,
        experiment_name="cmp",
        backend=backend,
        model_name="m",
        quantization=quantization,
        prompt_dataset_path="p.jsonl",
        prompt_dataset_hash=prompt_hash,
        hardware_profile=hardware_profile,
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


def test_different_prompt_corpus_blocks_comparison():
    """Property: results on different prompt datasets are not scientifically comparable."""
    a = _record("a", [0.05] * 6, prompt_hash="hash-a")
    b = _record("b", [0.05] * 6, prompt_hash="hash-b")
    result = compare_runs(a, b, metric_name="decode_tok_s")
    assert result.comparable is False
    assert result.significant_at_0_05 is None
    assert result.verdict == "blocked_by_comparability_gate"


def test_different_hardware_profile_blocks_comparison():
    a = _record("a", [0.05] * 6, hardware_profile="m5_max_128gb")
    b = _record("b", [0.05] * 6, hardware_profile="m4_pro_48gb")
    result = compare_runs(a, b, metric_name="decode_tok_s")
    assert result.comparable is False
    assert result.verdict == "blocked_by_comparability_gate"


def test_identical_throughput_samples_are_not_a_significant_win():
    """Property: same sample path → no false 'we improved' claim."""
    gaps = [0.05, 0.051, 0.049, 0.05, 0.0505, 0.0495]
    a = _record("a", gaps)
    b = _record("b", gaps)
    result = compare_runs(a, b, metric_name="decode_tok_s")
    assert result.comparable is True
    assert result.significant_at_0_05 is False
    assert result.verdict == "no_significant_difference"


def test_large_throughput_separation_is_detectable_when_gate_open():
    """
    Property: when comparability holds and means are far apart with low noise,
    the comparator reports a significant difference (direction b vs a).
    """
    # ~20 tok/s vs ~5 tok/s
    a = _record("a", [0.05] * 8)
    b = _record("b", [0.2] * 8)
    result = compare_runs(a, b, metric_name="decode_tok_s")
    assert result.comparable is True
    assert result.significant_at_0_05 is True
    assert result.mean_a is not None and result.mean_b is not None
    # b is slower (lower tok/s) because larger gap
    assert result.mean_b < result.mean_a


def test_quantization_may_differ_as_experimental_factor():
    """
    Property: changing quant is an allowed experimental factor —
    comparability still holds so we can measure the effect of quant.
    """
    a = _record("a", [0.05] * 6, quantization="q4")
    b = _record("b", [0.05] * 6, quantization="q8")
    result = compare_runs(a, b, metric_name="decode_tok_s")
    assert result.comparable is True
