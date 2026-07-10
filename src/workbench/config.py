"""Experiment config loading — single schema parser."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


SUPPORTED_SCHEMA_VERSIONS = frozenset({"1.0"})

# Explicit string forms only — never use bool(value) (bool("false") is True).
_BOOL_TRUE = frozenset({"true", "yes", "1", "on"})
_BOOL_FALSE = frozenset({"false", "no", "0", "off"})


def _parse_bool(value: Any, *, field: str) -> bool:
    """Strict boolean: real bools, or a small set of explicit string forms.

    Rejects other types (e.g. int 2) and ambiguous strings so a quoted
    ``"false"`` cannot silently enable a setting via ``bool("false")``.
    """
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        key = value.strip().lower()
        if key in _BOOL_TRUE:
            return True
        if key in _BOOL_FALSE:
            return False
        raise ValueError(
            f"{field} must be a boolean (true/false); got string {value!r} "
            f"(quoted boolean-like strings must be true/false/yes/no/on/off/1/0)"
        )
    raise ValueError(
        f"{field} must be a boolean (true/false); got {value!r} ({type(value).__name__})"
    )


@dataclass
class ModelConfig:
    name: str
    quantization: str
    backend: str
    model_id: str | None = None  # HF / mlx-community id when applicable
    max_tokens: int = 128


@dataclass
class BenchmarkConfig:
    warmup_iterations: int = 2
    timed_iterations: int = 5
    per_iteration_timeout_sec: float = 300.0
    prompt_dataset: str = "datasets/smoke_v1.jsonl"
    prompt_checksum: str | None = "datasets/smoke_v1.sha256"
    cooldown_between_runs_sec: float = 0.0
    monitor_thermal: bool = True
    abort_if_throttling: bool = False
    # If set, only first N prompts are used (smoke tests)
    max_prompts: int | None = None
    # When True, first prompt must supply a non-empty reference (dataset or override).
    require_correctness: bool = False


@dataclass
class MetricsConfig:
    report_percentiles: list[int] = field(default_factory=lambda: [50, 90, 95, 99])
    report_trimmed_mean: bool = True
    report_std: bool = True
    flag_cov_threshold: float = 0.05


@dataclass
class ReproducibilityConfig:
    random_seed: int = 42
    record_git_commit: bool = True
    record_env_versions: bool = True


@dataclass
class ExperimentConfig:
    schema_version: str
    name: str
    description: str
    hardware_profile: str
    model: ModelConfig
    benchmark: BenchmarkConfig
    metrics: MetricsConfig
    reproducibility: ReproducibilityConfig
    results_dir: str = "benchmarks/results"
    enable_mlflow: bool = True
    source_path: Path | None = None

    @staticmethod
    def from_dict(data: dict[str, Any], *, source_path: Path | None = None) -> ExperimentConfig:
        schema = str(data.get("schema_version", ""))
        if schema not in SUPPORTED_SCHEMA_VERSIONS:
            raise ValueError(
                f"Unsupported schema_version {schema!r}; supported={sorted(SUPPORTED_SCHEMA_VERSIONS)}"
            )

        exp = data.get("experiment") or {}
        hw = data.get("hardware") or {}
        model_raw = data.get("model") or {}
        bench_raw = data.get("benchmark") or {}
        metrics_raw = data.get("metrics") or {}
        repro_raw = data.get("reproducibility") or {}

        model = ModelConfig(
            name=str(model_raw["name"]),
            quantization=str(model_raw.get("quantization", "unknown")),
            backend=str(model_raw["backend"]),
            model_id=model_raw.get("model_id"),
            max_tokens=int(model_raw.get("max_tokens", 128)),
        )
        benchmark = BenchmarkConfig(
            warmup_iterations=int(bench_raw.get("warmup_iterations", 2)),
            timed_iterations=int(bench_raw.get("timed_iterations", 5)),
            per_iteration_timeout_sec=float(bench_raw.get("per_iteration_timeout_sec", 300)),
            prompt_dataset=str(bench_raw.get("prompt_dataset", "datasets/smoke_v1.jsonl")),
            prompt_checksum=bench_raw.get("prompt_checksum", "datasets/smoke_v1.sha256"),
            cooldown_between_runs_sec=float(bench_raw.get("cooldown_between_runs_sec", 0)),
            monitor_thermal=_parse_bool(
                bench_raw.get("monitor_thermal", True), field="benchmark.monitor_thermal"
            ),
            abort_if_throttling=_parse_bool(
                bench_raw.get("abort_if_throttling", False),
                field="benchmark.abort_if_throttling",
            ),
            max_prompts=bench_raw.get("max_prompts"),
            require_correctness=_parse_bool(
                bench_raw.get("require_correctness", False),
                field="benchmark.require_correctness",
            ),
        )
        metrics = MetricsConfig(
            report_percentiles=list(metrics_raw.get("report_percentiles", [50, 90, 95, 99])),
            report_trimmed_mean=bool(metrics_raw.get("report_trimmed_mean", True)),
            report_std=bool(metrics_raw.get("report_std", True)),
            flag_cov_threshold=float(metrics_raw.get("flag_cov_threshold", 0.05)),
        )
        repro = ReproducibilityConfig(
            random_seed=int(repro_raw.get("random_seed", 42)),
            record_git_commit=bool(repro_raw.get("record_git_commit", True)),
            record_env_versions=bool(repro_raw.get("record_env_versions", True)),
        )

        return ExperimentConfig(
            schema_version=schema,
            name=str(exp.get("name", "unnamed")),
            description=str(exp.get("description", "")),
            hardware_profile=str(hw.get("profile", "unknown")),
            model=model,
            benchmark=benchmark,
            metrics=metrics,
            reproducibility=repro,
            results_dir=str(data.get("results_dir", "benchmarks/results")),
            enable_mlflow=_parse_bool(data.get("enable_mlflow", True), field="enable_mlflow"),
            source_path=source_path,
        )


def load_config(path: Path) -> ExperimentConfig:
    path = path.resolve()
    with path.open(encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Config root must be a mapping: {path}")
    return ExperimentConfig.from_dict(data, source_path=path)
