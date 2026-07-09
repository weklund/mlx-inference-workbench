"""Behavioral: CLI compare/list outcomes (not rich formatting details)."""

from pathlib import Path

from click.testing import CliRunner

from benchmarks.cli import main
from workbench.config import load_config
from workbench.orchestrator import run_experiment
from workbench.storage.run_store import RunStore

REPO = Path(__file__).resolve().parents[2]


def test_list_shows_completed_run(tmp_path: Path):
    cfg = load_config(REPO / "configs/experiments/smoke_minimal.yaml")
    cfg.results_dir = str(tmp_path / "results")
    cfg.enable_mlflow = False
    record = run_experiment(cfg, repo_root=REPO)

    runner = CliRunner()
    result = runner.invoke(main, ["list", "--results-dir", str(tmp_path / "results")])
    assert result.exit_code == 0
    assert record.metadata.run_id in result.output


def test_compare_exits_nonzero_when_prompt_corpus_differs(tmp_path: Path):
    """
    Property: comparing runs that are not on the same prompt corpus must fail closed
    at the CLI boundary (non-zero exit), not print a fake improvement.
    """
    cfg = load_config(REPO / "configs/experiments/smoke_minimal.yaml")
    cfg.results_dir = str(tmp_path / "results")
    cfg.enable_mlflow = False

    r1 = run_experiment(cfg, repo_root=REPO)
    r2 = run_experiment(cfg, repo_root=REPO)

    store = RunStore(tmp_path / "results", enable_mlflow=False)
    rec = store.load(r2.metadata.run_id)
    rec.metadata.prompt_dataset_hash = "tampered-not-same-corpus"
    store.write(rec)

    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "compare",
            r1.metadata.run_id,
            r2.metadata.run_id,
            "--results-dir",
            str(tmp_path / "results"),
        ],
    )
    assert result.exit_code == 3
    # Must not claim a statistical win
    assert "significant_higher" not in result.output
    assert "significant_lower" not in result.output
