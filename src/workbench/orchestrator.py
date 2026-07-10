"""Benchmark lifecycle — coordinates collaborators; contains no metric math or engine details."""

from __future__ import annotations

from pathlib import Path

from workbench.config import ExperimentConfig
from workbench.engines.base import Engine, GenParams, iter_warmup_prompts
from workbench.engines.registry import create_engine
from workbench.engines.timeout import timed_generate
from workbench.hardware import capture_fingerprint, capture_git_sha, library_versions
from workbench.metrics import summarize_iterations
from workbench.models import (
    ENGINE_INTERFACE_VERSION,
    METRICS_SCHEMA_VERSION,
    GenerationResult,
    GenerationStatus,
    RunMetadata,
    RunRecord,
)
from workbench.paths import resolve_path
from workbench.prompts import PromptItem, load_dataset
from workbench.storage.run_store import RunStore
from workbench.thermal import ThermalSensor, build_thermal_sensor, cooldown


class OrchestratorError(Exception):
    pass


def resolve_correctness_reference(
    items: tuple[PromptItem, ...] | list[PromptItem],
    *,
    override: str = "",
) -> str | None:
    """CLI/test override wins; else first item's dataset reference; else None."""
    o = (override or "").strip()
    if o:
        return o
    if not items:
        return None
    ref = (items[0].reference or "").strip()
    return ref or None


def _run_correctness_gate(
    engine: Engine,
    prompt: str,
    params: GenParams,
    *,
    reference: str | None,
    require: bool,
) -> None:
    """Timed generate + base score_correctness when a reference exists.

    - No reference and not require → skip (smoke without gold outputs).
    - No reference and require → fail closed.
    - Reference present → fail closed on score mismatch / TIMEOUT / ERROR.
    """
    if reference is None:
        if require:
            raise OrchestratorError(
                "require_correctness is set but no reference on the first prompt "
                "(add JSONL 'reference' or pass correctness_reference=)"
            )
        return
    result = timed_generate(engine, prompt, params)
    if not engine.score_correctness(result, reference=reference, tolerance=0.0):
        raise OrchestratorError("Correctness gate failed — aborting before measurement")


def run_experiment(
    config: ExperimentConfig,
    *,
    repo_root: Path | None = None,
    engine: Engine | None = None,
    thermal: ThermalSensor | None = None,
    correctness_reference: str = "",
) -> RunRecord:
    """Run a timed benchmark.

    Correctness uses the first prompt's dataset ``reference`` when present, or
    ``correctness_reference`` override. See ``BenchmarkConfig.require_correctness``.
    """
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

    # Hardware + thermal (inject for tests; production builds from config)
    fingerprint = capture_fingerprint(config.hardware_profile)
    sensor = (
        thermal if thermal is not None else build_thermal_sensor(config.benchmark.monitor_thermal)
    )
    baseline = sensor.read()
    if config.benchmark.abort_if_throttling and sensor.is_throttling(baseline):
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

    ref = resolve_correctness_reference(items, override=correctness_reference)
    _run_correctness_gate(
        eng,
        prompts[0],
        params,
        reference=ref,
        require=config.benchmark.require_correctness,
    )

    # Warmup (same timed_generate as measure)
    for p in iter_warmup_prompts(prompts, config.benchmark.warmup_iterations):
        timed_generate(eng, p, params)

    # Measure
    iterations: list[GenerationResult] = []
    consecutive_failures = 0
    for i in range(config.benchmark.timed_iterations):
        prompt = prompts[i % len(prompts)]
        pre = sensor.read()
        try:
            result = timed_generate(eng, prompt, params)
            if result.status == GenerationStatus.SUCCESS:
                result.thermal_state = pre
            if result.status == GenerationStatus.SUCCESS and sensor.is_throttling(pre):
                result.status = GenerationStatus.THERMAL_TAINTED
            if result.e2e_ms is not None:
                sensor.note_duration(result.e2e_ms / 1000.0)
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
        git_sha=(capture_git_sha(root) if config.reproducibility.record_git_commit else None),
        library_versions=library_versions() if config.reproducibility.record_env_versions else {},
        random_seed=config.reproducibility.random_seed,
        schema_version=config.schema_version,
        metrics_schema_version=METRICS_SCHEMA_VERSION,
        engine_interface_version=ENGINE_INTERFACE_VERSION,
        thermal_monitoring=sensor.mode(),
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
