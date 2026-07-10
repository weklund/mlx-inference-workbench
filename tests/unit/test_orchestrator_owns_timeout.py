"""Orchestrator owns wall-clock timeout; engines only generate."""

from pathlib import Path

from workbench.config import ModelConfig, load_config
from workbench.engines.base import Engine, GenParams, iter_warmup_prompts
from workbench.models import GenerationResult, GenerationStatus, ThermalReading
from workbench.orchestrator import run_experiment


def _ok_result(text: str = "ok") -> GenerationResult:
    return GenerationResult(
        status=GenerationStatus.SUCCESS,
        output_text=text,
        token_timestamps=[0.01, 0.02],
        ttft_ms=10.0,
        total_tokens=2,
        memory_peak_bytes=1,
        thermal_state=ThermalReading(method="off"),
        e2e_ms=20.0,
    )


class CountingEngine(Engine):
    def __init__(self) -> None:
        self.generate_calls: list[tuple[str, float | None]] = []

    def name(self) -> str:
        return "counting"

    def load_model(self, config: ModelConfig) -> None:
        return None

    def generate(self, prompt: str, params: GenParams) -> GenerationResult:
        self.generate_calls.append((prompt, params.timeout_sec))
        return _ok_result(prompt)


def test_iter_warmup_prompts_is_single_warmup_source():
    assert list(iter_warmup_prompts(["a", "b"], n=3)) == ["a", "a", "a"]
    assert list(iter_warmup_prompts([], n=2)) == ["warmup", "warmup"]


def test_run_experiment_uses_timed_generate_for_warmup_and_measure(tmp_path: Path, monkeypatch):
    """Warmup + measure share timed_generate; blank reference skips correctness generate."""
    timed: list[tuple[str, float | None]] = []
    eng = CountingEngine()

    from workbench.engines.timeout import timed_generate as real

    def spy(engine, prompt, params):
        timed.append((prompt, params.timeout_sec))
        return real(engine, prompt, params)

    monkeypatch.setattr("workbench.orchestrator.timed_generate", spy)

    repo = Path(__file__).resolve().parents[2]
    cfg = load_config(repo / "configs/experiments/smoke_minimal.yaml")
    cfg.results_dir = str(tmp_path / "results")
    cfg.enable_mlflow = False
    cfg.benchmark.warmup_iterations = 2
    cfg.benchmark.timed_iterations = 1
    cfg.benchmark.cooldown_between_runs_sec = 0
    cfg.benchmark.monitor_thermal = False
    cfg.benchmark.per_iteration_timeout_sec = 12.0

    run_experiment(cfg, repo_root=repo, engine=eng)

    # no correctness generate when reference blank: warmup 2 + measure 1
    assert len(timed) == 3
    assert all(t == 12.0 for _, t in timed)
    assert len(eng.generate_calls) == 3


def test_run_experiment_correctness_with_reference_adds_timed_call(tmp_path: Path, monkeypatch):
    timed: list[str] = []
    eng = CountingEngine()

    from workbench.engines.timeout import timed_generate as real

    def spy(engine, prompt, params):
        timed.append(prompt)
        return real(engine, prompt, params)

    monkeypatch.setattr("workbench.orchestrator.timed_generate", spy)

    # Match generated output so base score_correctness passes
    eng.generate = lambda prompt, params: (  # type: ignore[method-assign]
        eng.generate_calls.append((prompt, params.timeout_sec)) or _ok_result("ref-text")
    )

    repo = Path(__file__).resolve().parents[2]
    cfg = load_config(repo / "configs/experiments/smoke_minimal.yaml")
    cfg.results_dir = str(tmp_path / "results")
    cfg.enable_mlflow = False
    cfg.benchmark.warmup_iterations = 0
    cfg.benchmark.timed_iterations = 1
    cfg.benchmark.cooldown_between_runs_sec = 0
    cfg.benchmark.monitor_thermal = False

    run_experiment(
        cfg,
        repo_root=repo,
        engine=eng,
        correctness_reference="ref-text",
    )
    # correctness + measure
    assert len(timed) == 2


def test_mlx_engine_source_does_not_depend_on_timeout_module():
    import inspect

    import workbench.engines.mlx_lm_engine as mlx_mod

    source = inspect.getsource(mlx_mod)
    assert "timed_generate" not in source
    assert "generate_with_timeout" not in source
    assert "workbench.engines.timeout" not in source
