"""Dataset references drive the correctness gate (fail closed)."""

from pathlib import Path

import pytest

from workbench.config import ModelConfig, load_config
from workbench.engines.base import Engine, GenParams
from workbench.models import GenerationResult, GenerationStatus, ThermalReading
from workbench.orchestrator import OrchestratorError, resolve_correctness_reference, run_experiment
from workbench.prompts import PromptItem, load_dataset
from workbench.storage.run_store import RunStore


def _item(prompt: str = "p", reference: str | None = None) -> PromptItem:
    return PromptItem(
        id="t1",
        category="test",
        prompt=prompt,
        reference=reference,
        expected_tokens_approx=8,
    )


def test_load_dataset_reads_optional_reference(tmp_path: Path):
    p = tmp_path / "refs.jsonl"
    p.write_text(
        '{"id": "a", "category": "c", "prompt": "hi", "reference": "hello world"}\n'
        '{"id": "b", "category": "c", "prompt": "no-ref"}\n',
        encoding="utf-8",
    )
    ds = load_dataset(p)
    assert ds.items[0].reference == "hello world"
    assert ds.items[1].reference is None


def test_resolve_prefers_override_then_first_item_reference():
    items = (_item(reference="from-dataset"),)
    assert resolve_correctness_reference(items, override="override") == "override"
    assert resolve_correctness_reference(items, override="") == "from-dataset"
    assert resolve_correctness_reference(items, override="   ") == "from-dataset"
    assert resolve_correctness_reference((_item(reference=None),), override="") is None
    assert resolve_correctness_reference((_item(reference="  "),), override="") is None


class _FixedOutputEngine(Engine):
    def __init__(self, text: str) -> None:
        self._text = text

    def name(self) -> str:
        return "fixed"

    def load_model(self, config: ModelConfig) -> None:
        return None

    def generate(self, prompt: str, params: GenParams) -> GenerationResult:
        return GenerationResult(
            status=GenerationStatus.SUCCESS,
            output_text=self._text,
            token_timestamps=[0.01, 0.02],
            ttft_ms=10.0,
            total_tokens=2,
            memory_peak_bytes=1,
            thermal_state=ThermalReading(method="off"),
            e2e_ms=20.0,
        )


def _cfg_with_dataset(tmp_path: Path, jsonl: Path, *, require: bool):
    repo = Path(__file__).resolve().parents[2]
    cfg = load_config(repo / "configs/experiments/smoke_minimal.yaml")
    cfg.results_dir = str(tmp_path / "results")
    cfg.enable_mlflow = False
    cfg.benchmark.prompt_dataset = str(jsonl)
    cfg.benchmark.prompt_checksum = None
    cfg.benchmark.warmup_iterations = 0
    cfg.benchmark.timed_iterations = 1
    cfg.benchmark.cooldown_between_runs_sec = 0
    cfg.benchmark.monitor_thermal = False
    cfg.benchmark.max_prompts = 1
    cfg.benchmark.require_correctness = require
    return repo, cfg


def test_dataset_reference_matching_output_allows_run(tmp_path: Path):
    jsonl = tmp_path / "ok.jsonl"
    jsonl.write_text(
        '{"id": "a", "category": "c", "prompt": "Q", "reference": "exact-answer"}\n',
        encoding="utf-8",
    )
    repo, cfg = _cfg_with_dataset(tmp_path, jsonl, require=False)
    rec = run_experiment(cfg, repo_root=repo, engine=_FixedOutputEngine("exact-answer"))
    assert rec.metrics.valid_iterations == 1


def test_dataset_reference_mismatch_aborts_without_run(tmp_path: Path):
    jsonl = tmp_path / "bad.jsonl"
    jsonl.write_text(
        '{"id": "a", "category": "c", "prompt": "Q", "reference": "expected"}\n',
        encoding="utf-8",
    )
    repo, cfg = _cfg_with_dataset(tmp_path, jsonl, require=False)

    with pytest.raises(OrchestratorError, match="Correctness gate"):
        run_experiment(cfg, repo_root=repo, engine=_FixedOutputEngine("wrong"))

    store = RunStore(tmp_path / "results", enable_mlflow=False)
    assert store.list_runs() == []


def test_require_correctness_without_reference_fails_closed(tmp_path: Path):
    jsonl = tmp_path / "noref.jsonl"
    jsonl.write_text(
        '{"id": "a", "category": "c", "prompt": "Q"}\n',
        encoding="utf-8",
    )
    repo, cfg = _cfg_with_dataset(tmp_path, jsonl, require=True)

    with pytest.raises(OrchestratorError, match="require_correctness"):
        run_experiment(cfg, repo_root=repo, engine=_FixedOutputEngine("anything"))

    store = RunStore(tmp_path / "results", enable_mlflow=False)
    assert store.list_runs() == []


def test_no_reference_without_require_skips_gate(tmp_path: Path):
    jsonl = tmp_path / "skip.jsonl"
    jsonl.write_text(
        '{"id": "a", "category": "c", "prompt": "Q"}\n',
        encoding="utf-8",
    )
    repo, cfg = _cfg_with_dataset(tmp_path, jsonl, require=False)
    rec = run_experiment(cfg, repo_root=repo, engine=_FixedOutputEngine("anything"))
    assert rec.metrics.valid_iterations == 1
