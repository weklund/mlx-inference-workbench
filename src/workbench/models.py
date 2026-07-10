"""Shared domain types — single source of truth for engines, metrics, storage, CLI."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any


METRICS_SCHEMA_VERSION = "1.0"
ENGINE_INTERFACE_VERSION = "1.0"


class GenerationStatus(str, Enum):
    SUCCESS = "success"
    TIMEOUT = "timeout"
    THERMAL_TAINTED = "tainted"
    ERROR = "error"


@dataclass(frozen=True)
class ThermalReading:
    """Snapshot of thermal / power state (best-effort)."""

    method: str
    thermal_pressure: str | None = None
    cpu_power_mw: float | None = None
    gpu_power_mw: float | None = None
    combined_power_mw: float | None = None
    notes: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class GenerationResult:
    """Per-iteration result. Engines must always populate required fields."""

    status: GenerationStatus
    output_text: str
    token_timestamps: list[float]
    ttft_ms: float | None  # None when TTFT was not measured (e.g. synthesized timestamps)
    total_tokens: int
    memory_peak_bytes: int
    thermal_state: ThermalReading
    acceptance_rate: float | None = None
    accepted_length_mean: float | None = None
    power_watts: float | None = None
    energy_per_token_joules: float | None = None
    e2e_ms: float | None = None
    error_message: str | None = None
    # True when token_timestamps were fabricated from e2e (non-stream path);
    # TTFT / per-token metrics must not be treated as measured.
    timestamps_synthesized: bool = False

    def __post_init__(self) -> None:
        if self.status == GenerationStatus.SUCCESS:
            if len(self.token_timestamps) != self.total_tokens:
                raise ValueError(
                    f"token_timestamps length {len(self.token_timestamps)} "
                    f"!= total_tokens {self.total_tokens}"
                )

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["status"] = self.status.value
        return d


@dataclass(frozen=True)
class DistributionStats:
    """Statistical summary for one scalar metric across iterations."""

    n: int
    mean: float
    std: float
    cov: float
    trimmed_mean: float
    percentiles: dict[str, float]  # keys "p50", "p90", ...
    values: tuple[float, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        return {
            "n": self.n,
            "mean": self.mean,
            "std": self.std,
            "cov": self.cov,
            "trimmed_mean": self.trimmed_mean,
            "percentiles": dict(self.percentiles),
            # values omitted from default export (large); storage may attach separately
        }


@dataclass
class MetricSummary:
    """All computed metrics for a completed run."""

    ttft_ms: DistributionStats | None
    decode_tok_s: DistributionStats | None
    sitl_ms: DistributionStats | None
    e2e_ms: DistributionStats | None
    memory_peak_bytes: DistributionStats | None
    acceptance_rate: DistributionStats | None
    valid_iterations: int
    total_iterations: int
    tainted_iterations: int
    unstable: bool
    quality_tag: str  # full | low_confidence | insufficient_data
    metrics_schema_version: str = METRICS_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        def maybe(d: DistributionStats | None) -> dict[str, Any] | None:
            return d.to_dict() if d is not None else None

        return {
            "metrics_schema_version": self.metrics_schema_version,
            "ttft_ms": maybe(self.ttft_ms),
            "decode_tok_s": maybe(self.decode_tok_s),
            "sitl_ms": maybe(self.sitl_ms),
            "e2e_ms": maybe(self.e2e_ms),
            "memory_peak_bytes": maybe(self.memory_peak_bytes),
            "acceptance_rate": maybe(self.acceptance_rate),
            "valid_iterations": self.valid_iterations,
            "total_iterations": self.total_iterations,
            "tainted_iterations": self.tainted_iterations,
            "unstable": self.unstable,
            "quality_tag": self.quality_tag,
        }


@dataclass
class RunMetadata:
    """Everything needed for reproducibility and comparability gates."""

    run_id: str
    experiment_name: str
    backend: str
    model_name: str
    quantization: str
    prompt_dataset_path: str
    prompt_dataset_hash: str
    hardware_profile: str
    hardware_fingerprint: dict[str, Any]
    git_sha: str | None
    library_versions: dict[str, str]
    random_seed: int
    schema_version: str
    metrics_schema_version: str
    engine_interface_version: str
    thermal_monitoring: str  # full | degraded | off
    config_path: str | None = None
    notes: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class RunRecord:
    """Complete run: metadata + metrics + raw iteration payloads."""

    metadata: RunMetadata
    metrics: MetricSummary
    iterations: list[dict[str, Any]]
    parquet_path: str | None = None
    mlflow_run_id: str | None = None

    def to_summary_dict(self) -> dict[str, Any]:
        return {
            "metadata": self.metadata.to_dict(),
            "metrics": self.metrics.to_dict(),
            "parquet_path": self.parquet_path,
            "mlflow_run_id": self.mlflow_run_id,
            "iteration_count": len(self.iterations),
        }
