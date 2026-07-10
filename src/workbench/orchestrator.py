"""Benchmark lifecycle — coordinates collaborators; contains no metric math or engine details."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FuturesTimeout
from pathlib import Path

from workbench.config import ExperimentConfig
from workbench.engines.base import Engine, GenParams
from workbench.engines.registry import create_engine
from workbench.hardware import capture_fingerprint, capture_git_sha, library_versions
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
from workbench.paths import resolve_path
from workbench.prompts import load_dataset
from workbench.storage.run_store import RunStore
from workbench.thermal import build_thermal_sensor, cooldown


class OrchestratorError(Exception):
    pass


def _generate_with_timeout(
    engine: Engine,
    prompt: str,
    params: GenParams,
) -> GenerationResult:
    """Enforce wall-clock timeout around engine.generate (HLD: TIMEOUT, not hang)."""
    timeout = params.timeout_sec
    if timeout is None or timeout <= 0:
        return engine.generate(prompt, params)

    with ThreadPoolExecutor(max_workers=1) as pool:
        fut = pool.submit(engine.generate, prompt, params)
        try:
            return fut.result(timeout=timeout)
        except FuturesTimeout:
            return GenerationResult(
                status=GenerationStatus.TIMEOUT,
                output_text="",
                token_timestamps=[],
                ttft_ms=0.0,
                total_tokens=0,
                memory_peak_bytes=0,
                thermal_state=ThermalReading(method="timeout"),
                error_message=f"per-iteration timeout exceeded ({timeout}s)",
            )


def run_experiment(
    config: ExperimentConfig,
    *,
    repo_root: Path | None = None,
    engine: Engine | None = None,
) -> RunRecord:
    root = (repo_root or Path.cwd()).resolve()
    store = RunStore(
        resolve_path(config.results_dir, root=root),
        enable_mlflow=config.enable_mlflow,
    )
    run_id = store.new_run_id()

    # Prompts (relative paths resolve against project root, not package install dir)
    ds_path = resolve_path(config.benchmark.prompt_dataset, root=root)
    checksum = None
    if config.benchmark.prompt_checksum:
        checksum = resolve_path(config.benchmark.prompt_checksum, root=root)
    dataset = load_dataset(ds_path, checksum_path=checksum)
    items = list(dataset.items)
    if config.benchmark.max_prompts is not None:
        items = items[: config.benchmark.max_prompts]
    prompts = [i.prompt for i in items]

    # Hardware + thermal
    fingerprint = capture_fingerprint(config.hardware_profile)
    thermal = build_thermal_sensor(config.benchmark.monitor_thermal)
    baseline = thermal.read()
    if config.benchmark.abort_if_throttling and thermal.is_throttling(baseline):
        raise OrchestratorError(f"Aborting: system already throttling ({baseline})")

    # Engine
    eng = engine or create_engine(config.model.backend)
    eng.load_model(config.model)

    params = GenParams(
        max_tokens=config.model.max_tokens,
        temperature=0.0,
        seed=config.reproducibility.random_seed,
        timeout_sec=float(config.benchmark.per_iteration_timeout_sec),
    )

    # Correctness gate (soft for mlx-lm free-form; stub always passes)
    ok = eng.validate_correctness(prompts[0], reference="", tolerance=0.0)
    if not ok:
        raise OrchestratorError("Correctness gate failed — aborting before measurement")

    # Warmup
    eng.warmup(prompts, config.benchmark.warmup_iterations, params)

    # Measure
    iterations: list[GenerationResult] = []
    consecutive_failures = 0
    for i in range(config.benchmark.timed_iterations):
        prompt = prompts[i % len(prompts)]
        pre = thermal.read()
        try:
            result = _generate_with_timeout(eng, prompt, params)
            # Attach orchestrator thermal reading when engine succeeded
            if result.status == GenerationStatus.SUCCESS:
                result.thermal_state = pre
            if result.status == GenerationStatus.SUCCESS and thermal.is_throttling(pre):
                result.status = GenerationStatus.THERMAL_TAINTED
            if hasattr(thermal, "note_duration") and result.e2e_ms is not None:
                thermal.note_duration(result.e2e_ms / 1000.0)
            consecutive_failures = (
                0 if result.status == GenerationStatus.SUCCESS else consecutive_failures + 1
            )
        except Exception as e:
            consecutive_failures += 1
            result = GenerationResult(
                status=GenerationStatus.ERROR,
                output_text="",
                token_timestamps=[],
                ttft_ms=0.0,
                total_tokens=0,
                memory_peak_bytes=0,
                thermal_state=pre,
                error_message=str(e),
            )
        iterations.append(result)
        if consecutive_failures >= 3:
            raise OrchestratorError("Aborting: 3 consecutive iteration failures")
        cooldown(config.benchmark.cooldown_between_runs_sec)

    metrics = summarize_iterations(
        iterations,
        percentiles=config.metrics.report_percentiles,
        cov_threshold=config.metrics.flag_cov_threshold,
    )

    # Always persist iterations for diagnosis; raise if nothing usable for stats.
    # Timeout-only runs still write when we have a store path — caller may catch.
    metadata = RunMetadata(
        run_id=run_id,
        experiment_name=config.name,
        backend=eng.name(),
        model_name=config.model.name,
        quantization=config.model.quantization,
        prompt_dataset_path=str(ds_path),
        prompt_dataset_hash=dataset.sha256,
        hardware_profile=config.hardware_profile,
        hardware_fingerprint=fingerprint,
        git_sha=capture_git_sha() if config.reproducibility.record_git_commit else None,
        library_versions=library_versions() if config.reproducibility.record_env_versions else {},
        random_seed=config.reproducibility.random_seed,
        schema_version=config.schema_version,
        metrics_schema_version=METRICS_SCHEMA_VERSION,
        engine_interface_version=ENGINE_INTERFACE_VERSION,
        thermal_monitoring=thermal.mode(),
        config_path=str(config.source_path) if config.source_path else None,
    )

    record = RunRecord(
        metadata=metadata,
        metrics=metrics,
        iterations=[r.to_dict() for r in iterations],
    )
    written = store.write(record)
    if metrics.valid_iterations == 0:
        raise OrchestratorError(
            "0 valid iterations — cannot compute statistics "
            f"(run_id={run_id}, see results for TIMEOUT/ERROR detail)"
        )
    return written
