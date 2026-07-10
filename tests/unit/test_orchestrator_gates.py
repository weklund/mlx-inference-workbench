"""Behavioral: orchestrator fail-closed gates (integrity, correctness, empty sample)."""

from pathlib import Path

import pytest

from workbench.config import ModelConfig, load_config
from workbench.engines.base import Engine, GenParams
from workbench.models import GenerationResult, GenerationStatus, ThermalReading
from workbench.orchestrator import OrchestratorError, run_experiment
from workbench.storage.run_store import RunStore


class OkEngine(Engine):
    def name(self) -> str:
        return "ok"

    def load_model(self, config: ModelConfig) -> None:
        return None

    def warmup(self, prompts: list[str], n: int, params: GenParams) -> None:
        return None

    def generate(self, prompt: str, params: GenParams) -> GenerationResult:
        return GenerationResult(
            status=GenerationStatus.SUCCESS,
            output_text="ok",
            token_timestamps=[0.01, 0.02],
            ttft_ms=10.0,
            total_tokens=2,
            memory_peak_bytes=1,
            thermal_state=ThermalReading(method="off"),
            e2e_ms=20.0,
        )

    def validate_correctness(self, prompt: str, reference: str, tolerance: float = 0.0) -> bool:
        return True


class BadCorrectnessEngine(OkEngine):
    def validate_correctness(self, prompt: str, reference: str, tolerance: float = 0.0) -> bool:
        return False


class AlwaysErrorEngine(OkEngine):
    def generate(self, prompt: str, params: GenParams) -> GenerationResult:
        return GenerationResult(
            status=GenerationStatus.ERROR,
            output_text="",
            token_timestamps=[],
            ttft_ms=0.0,
            total_tokens=0,
            memory_peak_bytes=0,
            thermal_state=ThermalReading(method="off"),
            error_message="boom",
        )


def _base_cfg(tmp_path: Path):
    repo = Path(__file__).resolve().parents[2]
    cfg = load_config(repo / "configs/experiments/smoke_minimal.yaml")
    cfg.results_dir = str(tmp_path / "results")
    cfg.enable_mlflow = False
    cfg.benchmark.warmup_iterations = 0
    cfg.benchmark.timed_iterations = 2
    cfg.benchmark.cooldown_between_runs_sec = 0
    cfg.benchmark.monitor_thermal = False
    return repo, cfg


def test_corrupt_prompt_dataset_aborts_without_publishing_a_run(tmp_path: Path):
    """Integrity failure must not produce a benchmark run artifact."""
    repo, cfg = _base_cfg(tmp_path)
    bad_cs = tmp_path / "bad.sha256"
    bad_cs.write_text("0" * 64 + "  smoke_v1.jsonl\n", encoding="utf-8")
    cfg.benchmark.prompt_checksum = str(bad_cs)

    with pytest.raises(ValueError):
        run_experiment(cfg, repo_root=repo, engine=OkEngine())

    store = RunStore(tmp_path / "results", enable_mlflow=False)
    assert store.list_runs() == []


def test_failed_correctness_gate_aborts_without_publishing_a_run(tmp_path: Path):
    """Wrong/unsafe backend output must not become a performance result."""
    repo, cfg = _base_cfg(tmp_path)

    with pytest.raises(OrchestratorError):
        run_experiment(cfg, repo_root=repo, engine=BadCorrectnessEngine())

    store = RunStore(tmp_path / "results", enable_mlflow=False)
    assert store.list_runs() == []


def test_no_successful_iterations_is_not_a_usable_baseline(tmp_path: Path):
    """
    A run with zero successful samples must not be treated as a normal baseline:
    either raise, or publish only with valid_iterations == 0 / non-full quality.
    """
    repo, cfg = _base_cfg(tmp_path)
    cfg.benchmark.timed_iterations = 2

    with pytest.raises(OrchestratorError):
        run_experiment(cfg, repo_root=repo, engine=AlwaysErrorEngine())

    # Diagnostic artifact may exist; it must not claim full usable stats
    store = RunStore(tmp_path / "results", enable_mlflow=False)
    runs = store.list_runs()
    if runs:
        rec = store.load(runs[0]["run_id"])
        assert rec.metrics.valid_iterations == 0
        assert rec.metrics.quality_tag != "full"
