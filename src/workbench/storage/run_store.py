"""Parquet-primary run storage + optional MLflow index. Atomic write discipline."""

from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Any

from workbench.models import MetricSummary, RunMetadata, RunRecord


class RunStore:
    def __init__(self, root: Path, *, enable_mlflow: bool = True) -> None:
        self.root = root.resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.enable_mlflow = enable_mlflow

    def new_run_id(self) -> str:
        return uuid.uuid4().hex[:12]

    def run_dir(self, run_id: str) -> Path:
        return self.root / run_id

    def write(self, record: RunRecord) -> RunRecord:
        """Write summary.json + iterations.jsonl first; then optional MLflow."""
        d = self.run_dir(record.metadata.run_id)
        d.mkdir(parents=True, exist_ok=True)

        # Raw iterations (JSONL) — ground truth for per-iter data
        iter_path = d / "iterations.jsonl"
        tmp_iter = d / "iterations.jsonl.tmp"
        with tmp_iter.open("w", encoding="utf-8") as f:
            for row in record.iterations:
                f.write(json.dumps(row) + "\n")
        tmp_iter.replace(iter_path)

        summary_path = d / "summary.json"
        _write_summary_atomic(summary_path, record)

        # Optional parquet if pyarrow available
        parquet_path = None
        try:
            parquet_path = str(self._write_parquet(d, record))
        except Exception:
            parquet_path = None

        record.parquet_path = parquet_path

        if self.enable_mlflow:
            try:
                record.mlflow_run_id = self._log_mlflow(record)
            except Exception as e:
                # Fallback note — data already on disk
                (d / "mlflow_error.txt").write_text(str(e), encoding="utf-8")

        # Rewrite summary with paths (parquet / mlflow ids) — also atomic
        _write_summary_atomic(summary_path, record)

        return record

    def load(self, run_id: str) -> RunRecord:
        d = self.run_dir(run_id)
        summary_path = d / "summary.json"
        if not summary_path.is_file():
            raise FileNotFoundError(f"No run found: {run_id} ({summary_path})")
        data = json.loads(summary_path.read_text(encoding="utf-8"))
        meta = _metadata_from_dict(data["metadata"])
        metrics = _metrics_from_dict(data["metrics"], data.get("metrics_values") or {})
        iterations: list[dict[str, Any]] = []
        iter_path = d / "iterations.jsonl"
        if iter_path.is_file():
            for line in iter_path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    iterations.append(json.loads(line))
        return RunRecord(
            metadata=meta,
            metrics=metrics,
            iterations=iterations,
            parquet_path=data.get("parquet_path"),
            mlflow_run_id=data.get("mlflow_run_id"),
        )

    def list_runs(self) -> list[dict[str, Any]]:
        rows = []
        if not self.root.is_dir():
            return rows
        for child in sorted(self.root.iterdir()):
            summary = child / "summary.json"
            if summary.is_file():
                data = json.loads(summary.read_text(encoding="utf-8"))
                rows.append(
                    {
                        "run_id": child.name,
                        "name": data.get("metadata", {}).get("experiment_name"),
                        "backend": data.get("metadata", {}).get("backend"),
                        "unstable": data.get("metrics", {}).get("unstable"),
                        "quality_tag": data.get("metrics", {}).get("quality_tag"),
                    }
                )
        return rows

    def _write_parquet(self, d: Path, record: RunRecord) -> Path:
        import pyarrow as pa
        import pyarrow.parquet as pq

        table = pa.Table.from_pylist(record.iterations)
        path = d / "iterations.parquet"
        pq.write_table(table, path)
        return path

    def _log_mlflow(self, record: RunRecord) -> str:
        import mlflow

        mlflow.set_tracking_uri(f"file:{(self.root.parent / 'mlruns').resolve()}")
        mlflow.set_experiment("mlx-inference-workbench")
        with mlflow.start_run(run_name=record.metadata.experiment_name) as run:
            mlflow.log_params(
                {
                    "backend": record.metadata.backend,
                    "model_name": record.metadata.model_name,
                    "quantization": record.metadata.quantization,
                    "prompt_hash": record.metadata.prompt_dataset_hash[:16],
                    "hardware_profile": record.metadata.hardware_profile,
                    "metrics_schema_version": record.metadata.metrics_schema_version,
                    "run_id": record.metadata.run_id,
                }
            )
            m = record.metrics
            if m.decode_tok_s:
                mlflow.log_metric(
                    "decode_tok_s_p50", m.decode_tok_s.percentiles.get("p50", m.decode_tok_s.mean)
                )
                mlflow.log_metric("decode_tok_s_mean", m.decode_tok_s.mean)
                mlflow.log_metric("decode_tok_s_cov", m.decode_tok_s.cov)
            if m.ttft_ms:
                mlflow.log_metric("ttft_ms_p50", m.ttft_ms.percentiles.get("p50", m.ttft_ms.mean))
            mlflow.set_tags(
                {
                    "stable": str(not m.unstable).lower(),
                    "quality": m.quality_tag,
                    "backend": record.metadata.backend,
                }
            )
            return run.info.run_id


def _metric_values_payload(metrics: MetricSummary) -> dict[str, list[float]]:
    out: dict[str, list[float]] = {}
    for name in (
        "ttft_ms",
        "decode_tok_s",
        "sitl_ms",
        "e2e_ms",
        "memory_peak_bytes",
        "acceptance_rate",
    ):
        dist = getattr(metrics, name)
        if dist is not None and dist.values:
            out[name] = list(dist.values)
    return out


def _summary_payload(record: RunRecord) -> dict[str, Any]:
    """Build summary.json body: metadata + metrics + raw values for compare."""
    payload = record.to_summary_dict()
    payload["metrics"] = record.metrics.to_dict()
    payload["metrics_values"] = _metric_values_payload(record.metrics)
    return payload


def _write_summary_atomic(summary_path: Path, record: RunRecord) -> None:
    """Write summary via .json.tmp then os.replace-equivalent Path.replace."""
    tmp_path = summary_path.with_suffix(summary_path.suffix + ".tmp")
    tmp_path.write_text(
        json.dumps(_summary_payload(record), indent=2),
        encoding="utf-8",
    )
    tmp_path.replace(summary_path)


def _metadata_from_dict(d: dict[str, Any]) -> RunMetadata:
    return RunMetadata(
        run_id=d["run_id"],
        experiment_name=d["experiment_name"],
        backend=d["backend"],
        model_name=d["model_name"],
        quantization=d["quantization"],
        prompt_dataset_path=d["prompt_dataset_path"],
        prompt_dataset_hash=d["prompt_dataset_hash"],
        hardware_profile=d["hardware_profile"],
        hardware_fingerprint=d.get("hardware_fingerprint") or {},
        git_sha=d.get("git_sha"),
        library_versions=d.get("library_versions") or {},
        random_seed=int(d.get("random_seed", 42)),
        schema_version=d["schema_version"],
        metrics_schema_version=d["metrics_schema_version"],
        engine_interface_version=d["engine_interface_version"],
        thermal_monitoring=d.get("thermal_monitoring", "off"),
        config_path=d.get("config_path"),
        notes=d.get("notes"),
    )


def _metrics_from_dict(d: dict[str, Any], values: dict[str, list[float]]) -> MetricSummary:
    from workbench.models import DistributionStats, MetricSummary

    def rebuild(name: str) -> DistributionStats | None:
        block = d.get(name)
        if not block:
            return None
        vals = tuple(values.get(name) or [])
        return DistributionStats(
            n=int(block["n"]),
            mean=float(block["mean"]),
            std=float(block["std"]),
            cov=float(block["cov"]),
            trimmed_mean=float(block["trimmed_mean"]),
            percentiles=dict(block.get("percentiles") or {}),
            values=vals,
        )

    return MetricSummary(
        ttft_ms=rebuild("ttft_ms"),
        decode_tok_s=rebuild("decode_tok_s"),
        sitl_ms=rebuild("sitl_ms"),
        e2e_ms=rebuild("e2e_ms"),
        memory_peak_bytes=rebuild("memory_peak_bytes"),
        acceptance_rate=rebuild("acceptance_rate"),
        valid_iterations=int(d["valid_iterations"]),
        total_iterations=int(d["total_iterations"]),
        tainted_iterations=int(d["tainted_iterations"]),
        unstable=bool(d["unstable"]),
        quality_tag=str(d["quality_tag"]),
        metrics_schema_version=str(d.get("metrics_schema_version", "1.0")),
    )
