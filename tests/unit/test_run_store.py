"""TDD: Parquet/JSON run store is primary ground truth."""

from pathlib import Path

import pytest

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
from workbench.storage.run_store import RunStore


def _sample_record(run_id: str = "abc123") -> RunRecord:
    iters = [
        GenerationResult(
            status=GenerationStatus.SUCCESS,
            output_text="hi",
            token_timestamps=[0.01, 0.02, 0.03],
            ttft_ms=10.0,
            total_tokens=3,
            memory_peak_bytes=100,
            thermal_state=ThermalReading(method="off"),
            e2e_ms=30.0,
        )
        for _ in range(5)
    ]
    metrics = summarize_iterations(iters)
    meta = RunMetadata(
        run_id=run_id,
        experiment_name="unit",
        backend="stub",
        model_name="m",
        quantization="n/a",
        prompt_dataset_path="/tmp/p.jsonl",
        prompt_dataset_hash="deadbeef",
        hardware_profile="m5_max_128gb",
        hardware_fingerprint={"chip": "test"},
        git_sha=None,
        library_versions={},
        random_seed=42,
        schema_version="1.0",
        metrics_schema_version=METRICS_SCHEMA_VERSION,
        engine_interface_version=ENGINE_INTERFACE_VERSION,
        thermal_monitoring="off",
    )
    return RunRecord(
        metadata=meta,
        metrics=metrics,
        iterations=[i.to_dict() for i in iters],
    )


def test_write_load_roundtrip_preserves_compare_values(tmp_path: Path):
    store = RunStore(tmp_path, enable_mlflow=False)
    original = _sample_record("round1")
    store.write(original)
    loaded = store.load("round1")
    assert loaded.metadata.prompt_dataset_hash == "deadbeef"
    assert loaded.metrics.decode_tok_s is not None
    assert loaded.metrics.decode_tok_s.values
    assert len(loaded.metrics.decode_tok_s.values) == 5
    assert loaded.metrics.valid_iterations == 5


def test_list_runs_includes_written_id(tmp_path: Path):
    store = RunStore(tmp_path, enable_mlflow=False)
    store.write(_sample_record("listed"))
    ids = [r["run_id"] for r in store.list_runs()]
    assert "listed" in ids


def test_load_missing_run_raises(tmp_path: Path):
    store = RunStore(tmp_path, enable_mlflow=False)
    with pytest.raises(FileNotFoundError):
        store.load("does-not-exist")
