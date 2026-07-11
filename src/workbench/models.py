"""Shared domain types — single source of truth for engines, metrics, storage, CLI."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import StrEnum
from typing import Any

METRICS_SCHEMA_VERSION = "1.0"
ENGINE_INTERFACE_VERSION = "1.0"

# MetricSummary fields that hold DistributionStats (compare + store + CLI allowlist).
# Single source of truth — do not re-list these names elsewhere.
DISTRIBUTION_METRIC_NAMES: tuple[str, ...] = (
    "ttft_ms",
    "decode_tok_s",
    "sitl_ms",
    "e2e_ms",
    "memory_peak_bytes",
    "acceptance_rate",
)


class GenerationStatus(StrEnum):
    """Terminal status for a single generation attempt."""

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
        """Serialize to a JSON-friendly dict."""
        return asdict(self)


@dataclass
class GenerationResult:
    """Per-iteration result. Engines must always populate required fields.

    **Stream vs e2e-only (SUCCESS):**

    - **Stream path:** ``token_timestamps`` has one absolute second mark per
      output token (from generation start). Then
      ``len(token_timestamps) == total_tokens`` is required, and ``ttft_ms``
      is the first mark in ms.
    - **E2e-only path:** when the backend cannot stream (or falls back to
      non-stream generate), engines must **not** fabricate placeholders.
      Use ``token_timestamps=[]``, ``ttft_ms=None``, set ``e2e_ms`` from
      wall clock, and set ``total_tokens`` by tokenizer/count of text.
      Empty timestamps do **not** violate the length guarantee — the
      guarantee applies only when timestamps are non-empty.

    **Metrics implications (see ``workbench.metrics``):** decode tok/s,
    SITL, and measured TTFT require non-empty stream timestamps; e2e-only
    iterations contribute to ``e2e_ms`` (and memory) distributions only.
    """

    status: GenerationStatus
    output_text: str
    # Stream: seconds from start per token. E2e-only SUCCESS: empty list.
    token_timestamps: list[float]
    # Measured TTFT (ms); None when e2e-only / not measured.
    ttft_ms: float | None
    total_tokens: int
    memory_peak_bytes: int
    thermal_state: ThermalReading
    acceptance_rate: float | None = None
    accepted_length_mean: float | None = None
    power_watts: float | None = None
    energy_per_token_joules: float | None = None
    e2e_ms: float | None = None
    error_message: str | None = None

    def __post_init__(self) -> None:
        """Enforce stream-path alignment; empty timestamps allowed for e2e-only.

        When ``token_timestamps`` is non-empty and status is SUCCESS, length
        must equal ``total_tokens``. Empty lists skip this check (honest
        e2e-only results).
        """
        if (
            self.status == GenerationStatus.SUCCESS
            and self.token_timestamps
            and len(self.token_timestamps) != self.total_tokens
        ):
            raise ValueError(
                f"token_timestamps length {len(self.token_timestamps)} "
                f"!= total_tokens {self.total_tokens}"
            )

    def to_dict(self) -> dict[str, Any]:
        """Serialize including enum values (not Enum members)."""
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
        """Serialize stats; omits raw ``values`` (storage may attach separately)."""
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
    quality_tag: str  # full | low_confidence | insufficient_data | speculative_no_accept
    metrics_schema_version: str = METRICS_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        """Serialize all distribution metrics and quality tags."""

        def maybe(d: DistributionStats | None) -> dict[str, Any] | None:
            return d.to_dict() if d is not None else None

        out: dict[str, Any] = {
            "metrics_schema_version": self.metrics_schema_version,
        }
        for name in DISTRIBUTION_METRIC_NAMES:
            out[name] = maybe(getattr(self, name))
        out.update(
            {
                "valid_iterations": self.valid_iterations,
                "total_iterations": self.total_iterations,
                "tainted_iterations": self.tainted_iterations,
                "unstable": self.unstable,
                "quality_tag": self.quality_tag,
            }
        )
        return out


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
        """Serialize metadata for summary.json / comparability."""
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
        """Compact summary without full iteration payloads."""
        return {
            "metadata": self.metadata.to_dict(),
            "metrics": self.metrics.to_dict(),
            "parquet_path": self.parquet_path,
            "mlflow_run_id": self.mlflow_run_id,
            "iteration_count": len(self.iterations),
        }
