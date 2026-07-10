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


def test_summary_write_leaves_no_tmp_and_is_valid_json(tmp_path: Path):
    store = RunStore(tmp_path, enable_mlflow=False)
    store.write(_sample_record("atomic1"))
    run_dir = tmp_path / "atomic1"
    assert (run_dir / "summary.json").is_file()
    assert not (run_dir / "summary.json.tmp").exists()
    import json

    data = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    assert "metrics" in data and "metrics_values" in data
    assert data["metadata"]["run_id"] == "atomic1"


def test_write_summary_atomic_helper_replaces_via_tmp(tmp_path: Path, monkeypatch):
    """Both write paths must go through tmp + replace (not direct write_text)."""
    replaces: list[tuple[str, str]] = []
    original_replace = Path.replace

    def tracking_replace(self: Path, target: Path):
        replaces.append((str(self), str(target)))
        return original_replace(self, target)

    monkeypatch.setattr(Path, "replace", tracking_replace)

    store = RunStore(tmp_path, enable_mlflow=False)
    store.write(_sample_record("atomic2"))

    # Initial summary + post-parquet rewrite = two atomic replaces of summary.json
    summary_replaces = [p for p in replaces if p[1].endswith("summary.json")]
    assert len(summary_replaces) == 2
    assert all(src.endswith("summary.json.tmp") for src, _ in summary_replaces)
