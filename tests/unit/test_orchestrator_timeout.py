"""Behavioral: deadline exceeded → iteration is not a successful measurement."""

from pathlib import Path

from workbench.config import ModelConfig, load_config
from workbench.engines.base import Engine, GenParams
from workbench.models import GenerationResult, GenerationStatus, ThermalReading
from workbench.orchestrator import OrchestratorError, run_experiment
from workbench.storage.run_store import RunStore


class SlowEngine(Engine):
    """Takes longer than the configured deadline; does not self-timeout."""

    def name(self) -> str:
        return "slow"

    def load_model(self, config: ModelConfig) -> None:
        return None

    def generate(self, prompt: str, params: GenParams) -> GenerationResult:
        import time

        time.sleep(1.0)
        return GenerationResult(
            status=GenerationStatus.SUCCESS,
            output_text="late",
            token_timestamps=[0.5, 1.0],
            ttft_ms=500.0,
            total_tokens=2,
            memory_peak_bytes=1,
            thermal_state=ThermalReading(method="off"),
            e2e_ms=1000.0,
        )


def test_iteration_past_deadline_is_excluded_from_success_metrics(tmp_path: Path):
    """
    Property (HLD): if generation exceeds per-iteration deadline, that iteration
    must not count as a successful measurement (not used as a valid sample).
    """
    repo = Path(__file__).resolve().parents[2]
    cfg = load_config(repo / "configs/experiments/smoke_minimal.yaml")
    results = tmp_path / "results"
    cfg.results_dir = str(results)
    cfg.enable_mlflow = False
    cfg.benchmark.warmup_iterations = 0
    cfg.benchmark.timed_iterations = 1
    cfg.benchmark.per_iteration_timeout_sec = 0.2
    cfg.benchmark.cooldown_between_runs_sec = 0
    cfg.benchmark.monitor_thermal = False

    record = None
    try:
        record = run_experiment(cfg, repo_root=repo, engine=SlowEngine())
    except OrchestratorError:
        store = RunStore(results, enable_mlflow=False)
        runs = store.list_runs()
        assert runs, "expected a persisted diagnostic run when all iterations fail"
        record = store.load(runs[0]["run_id"])

    assert record is not None
    assert record.metrics.valid_iterations == 0
    assert all(it["status"] != GenerationStatus.SUCCESS.value for it in record.iterations)
