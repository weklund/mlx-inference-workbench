from pathlib import Path

import pytest

from workbench.config import load_config
from workbench.orchestrator import run_experiment
from workbench.statistics_compare import compare_runs
from workbench.storage.run_store import RunStore


REPO = Path(__file__).resolve().parents[2]


@pytest.mark.integration
def test_stub_run_and_compare(tmp_path: Path):
    cfg = load_config(REPO / "configs/experiments/smoke_minimal.yaml")
    cfg.results_dir = str(tmp_path / "results")
    cfg.enable_mlflow = False
    cfg.benchmark.timed_iterations = 5
    cfg.benchmark.warmup_iterations = 1
    cfg.benchmark.cooldown_between_runs_sec = 0

    r1 = run_experiment(cfg, repo_root=REPO)
    r2 = run_experiment(cfg, repo_root=REPO)

    assert r1.metrics.valid_iterations >= 3
    assert r1.metadata.prompt_dataset_hash

    store = RunStore(tmp_path / "results", enable_mlflow=False)
    loaded = store.load(r1.metadata.run_id)
    assert loaded.metadata.run_id == r1.metadata.run_id

    cmp = compare_runs(r1, r2, metric_name="decode_tok_s")
    assert cmp.comparable
    # Stub is near-deterministic; expect no huge false win
    assert cmp.verdict in {
        "no_significant_difference",
        "significant_higher_b_vs_a",
        "significant_lower_b_vs_a",
        "insufficient_data_for_test",
    }
