"""Behavioral: persisted runs remain valid inputs to compare_runs (disk is source of truth)."""

from pathlib import Path

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
from workbench.storage.run_store import RunStore


def _record(run_id: str, gaps: list[float], prompt_hash: str = "h1") -> RunRecord:
    iters = []
    for g in gaps:
        t0, t1 = 0.01, 0.01 + g
        iters.append(
            GenerationResult(
                status=GenerationStatus.SUCCESS,
                output_text="x",
                token_timestamps=[t0, t1],
                ttft_ms=t0 * 1000,
                total_tokens=2,
                memory_peak_bytes=100,
                thermal_state=ThermalReading(method="off"),
                e2e_ms=t1 * 1000,
            )
        )
    metrics = summarize_iterations(iters)
    meta = RunMetadata(
        run_id=run_id,
        experiment_name="disk",
        backend="stub",
        model_name="m",
        quantization="q4",
        prompt_dataset_path="p.jsonl",
        prompt_dataset_hash=prompt_hash,
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


def test_compare_after_write_load_matches_in_memory_compare(tmp_path: Path):
    """
    Property: write → load → compare must not lose the ability to detect
    'no significant difference' for identical workloads.
    """
    store = RunStore(tmp_path, enable_mlflow=False)
    gaps = [0.05, 0.05, 0.05, 0.05, 0.05, 0.05]
    store.write(_record("r1", gaps))
    store.write(_record("r2", gaps))

    a, b = store.load("r1"), store.load("r2")
    result = compare_runs(a, b, metric_name="decode_tok_s")
    assert result.comparable is True
    assert result.significant_at_0_05 is False


def test_compare_after_load_still_blocks_on_prompt_hash(tmp_path: Path):
    store = RunStore(tmp_path, enable_mlflow=False)
    store.write(_record("r1", [0.05] * 6, prompt_hash="aaa"))
    store.write(_record("r2", [0.05] * 6, prompt_hash="bbb"))
    result = compare_runs(store.load("r1"), store.load("r2"), metric_name="decode_tok_s")
    assert result.comparable is False
