"""Thin CLI — no business logic beyond argument parsing and pretty-print."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import click
from rich.console import Console
from rich.table import Table

from workbench.paths import ENV_PROJECT_ROOT, resolve_project_root, resolve_results_dir

console = Console()


def _project_root(ctx: click.Context) -> Path:
    return ctx.obj["project_root"]


def _load_run(store, run_id: str):
    """Load a run; print a friendly error and exit nonzero on failure (no traceback)."""
    try:
        return store.load(run_id)
    except FileNotFoundError as e:
        console.print(f"[red]Run not found:[/red] {e}")
        sys.exit(1)
    except Exception as e:
        console.print(f"[red]Failed to load run {run_id}:[/red] {e}")
        sys.exit(1)


@click.group()
@click.version_option(package_name="mlx-inference-workbench")
@click.option(
    "--project-root",
    type=click.Path(file_okay=False, path_type=Path),
    default=None,
    envvar=ENV_PROJECT_ROOT,
    help=(
        "Project/data root for configs, datasets, and results "
        f"(env: {ENV_PROJECT_ROOT}). Defaults to discovery from cwd, not the install path."
    ),
)
@click.pass_context
def main(ctx: click.Context, project_root: Path | None) -> None:
    """MLX Inference Workbench — comparable, statistically rigorous local benchmarks."""
    ctx.ensure_object(dict)
    ctx.obj["project_root"] = resolve_project_root(explicit=project_root)


@main.command("run")
@click.argument("config_path", type=click.Path(exists=True, path_type=Path))
@click.option(
    "--no-mlflow", is_flag=True, help="Skip MLflow indexing (Parquet/JSON still written)."
)
@click.pass_context
def run_cmd(ctx: click.Context, config_path: Path, no_mlflow: bool) -> None:
    """Run a benchmark from a YAML experiment config."""
    from workbench.config import load_config
    from workbench.orchestrator import OrchestratorError, run_experiment

    root = _project_root(ctx)
    cfg = load_config(config_path.resolve())
    if no_mlflow:
        cfg.enable_mlflow = False
    try:
        record = run_experiment(cfg, repo_root=root)
    except OrchestratorError as e:
        console.print(f"[red]Orchestrator error:[/red] {e}")
        sys.exit(2)
    except Exception as e:
        console.print(f"[red]Failed:[/red] {e}")
        sys.exit(1)

    m = record.metrics
    results_path = resolve_results_dir(cfg.results_dir, project_root=root) / record.metadata.run_id
    console.print(f"[green]Run complete[/green] id={record.metadata.run_id}")
    console.print(f"  backend={record.metadata.backend} model={record.metadata.model_name}")
    console.print(
        f"  quality={m.quality_tag} unstable={m.unstable} "
        f"valid={m.valid_iterations}/{m.total_iterations}"
    )
    if m.decode_tok_s:
        p50 = m.decode_tok_s.percentiles.get("p50", m.decode_tok_s.mean)
        console.print(
            f"  decode_tok_s p50={p50:.2f} mean={m.decode_tok_s.mean:.2f} cov={m.decode_tok_s.cov:.4f}"
        )
    if m.ttft_ms:
        p50 = m.ttft_ms.percentiles.get("p50", m.ttft_ms.mean)
        console.print(f"  ttft_ms p50={p50:.2f}")
    console.print(f"  project_root={root}")
    console.print(f"  results={results_path}")
    if record.mlflow_run_id:
        console.print(f"  mlflow_run_id={record.mlflow_run_id}")


@main.command("list")
@click.option(
    "--results-dir",
    type=click.Path(path_type=Path),
    default=None,
    help="Override results directory (default: <project-root>/benchmarks/results)",
)
@click.pass_context
def list_cmd(ctx: click.Context, results_dir: Path | None) -> None:
    """List local runs (Parquet/JSON store)."""
    from workbench.storage.run_store import RunStore

    root = resolve_results_dir(results_dir, project_root=_project_root(ctx))
    store = RunStore(root, enable_mlflow=False)
    rows = store.list_runs()
    if not rows:
        console.print("No runs found.")
        return
    table = Table(title="Benchmark runs")
    table.add_column("run_id")
    table.add_column("name")
    table.add_column("backend")
    table.add_column("quality")
    table.add_column("unstable")
    for r in rows:
        table.add_row(
            r["run_id"],
            str(r.get("name")),
            str(r.get("backend")),
            str(r.get("quality_tag")),
            str(r.get("unstable")),
        )
    console.print(table)


@main.command("report")
@click.argument("run_id")
@click.option("--results-dir", type=click.Path(path_type=Path), default=None)
@click.pass_context
def report_cmd(ctx: click.Context, run_id: str, results_dir: Path | None) -> None:
    """Print JSON summary for a run."""
    from workbench.storage.run_store import RunStore

    root = resolve_results_dir(results_dir, project_root=_project_root(ctx))
    store = RunStore(root, enable_mlflow=False)
    record = _load_run(store, run_id)
    console.print_json(
        json.dumps(
            record.to_summary_dict()
            | {
                "metrics": record.metrics.to_dict(),
            }
        )
    )


@main.command("compare")
@click.argument("run_a")
@click.argument("run_b")
@click.option("--metric", default="decode_tok_s", show_default=True)
@click.option("--results-dir", type=click.Path(path_type=Path), default=None)
@click.pass_context
def compare_cmd(
    ctx: click.Context, run_a: str, run_b: str, metric: str, results_dir: Path | None
) -> None:
    """Compare two runs (comparability gate + statistical test)."""
    from workbench.statistics_compare import compare_runs
    from workbench.storage.run_store import RunStore

    root = resolve_results_dir(results_dir, project_root=_project_root(ctx))
    store = RunStore(root, enable_mlflow=False)
    a = _load_run(store, run_a)
    b = _load_run(store, run_b)
    result = compare_runs(a, b, metric_name=metric)
    console.print_json(json.dumps(result.to_dict()))
    if not result.comparable:
        console.print("[red]Blocked by comparability gate[/red]")
        sys.exit(3)
    console.print(f"[bold]verdict:[/bold] {result.verdict}")


@main.command("validate")
@click.argument("run_id")
def validate_cmd(run_id: str) -> None:
    """Placeholder — re-check correctness for a past run (Phase 1 stub)."""
    console.print(f"validate not fully implemented yet; run_id={run_id}")
    sys.exit(0)


if __name__ == "__main__":
    main()
