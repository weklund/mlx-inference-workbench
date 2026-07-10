"""Project root and path resolution — independent of package install location."""

from __future__ import annotations

import os
from pathlib import Path

# Env var for explicit data/project root (configs, datasets, results).
ENV_PROJECT_ROOT = "MLX_WORKBENCH_ROOT"
DEFAULT_RESULTS_SUBDIR = Path("benchmarks") / "results"
_PROJECT_NAME_MARKERS = ("mlx-inference-workbench",)


def resolve_project_root(
    *,
    explicit: Path | str | None = None,
    start: Path | None = None,
) -> Path:
    """
    Resolve the workbench project / data root.

    Priority:
      1. explicit path (CLI --project-root)
      2. MLX_WORKBENCH_ROOT environment variable
      3. walk up from start (default: cwd) for pyproject.toml naming this project
         or a directory containing both configs/ and datasets/
      4. cwd

    Never uses the installed package __file__ location (site-packages is wrong
    for user datasets and results).
    """
    if explicit is not None:
        return Path(explicit).expanduser().resolve()

    env = os.environ.get(ENV_PROJECT_ROOT)
    if env:
        return Path(env).expanduser().resolve()

    start_dir = (start or Path.cwd()).resolve()
    if start_dir.is_file():
        start_dir = start_dir.parent

    for directory in [start_dir, *start_dir.parents]:
        if _is_project_root(directory):
            return directory

    return start_dir


def _is_project_root(directory: Path) -> bool:
    pyproject = directory / "pyproject.toml"
    if pyproject.is_file():
        try:
            text = pyproject.read_text(encoding="utf-8")
        except OSError:
            text = ""
        if any(marker in text for marker in _PROJECT_NAME_MARKERS):
            return True
    # Workspace layout without relying on package name alone
    if (directory / "configs").is_dir() and (directory / "datasets").is_dir():
        return True
    return False


def resolve_path(path: Path | str, *, root: Path) -> Path:
    """Resolve path: absolute stays absolute; relative is against project root."""
    p = Path(path).expanduser()
    if p.is_absolute():
        return p.resolve()
    return (root / p).resolve()


def resolve_results_dir(
    results_dir: Path | str | None,
    *,
    project_root: Path,
) -> Path:
    """Default results under project root; honor absolute or relative override."""
    if results_dir is None:
        return (project_root / DEFAULT_RESULTS_SUBDIR).resolve()
    return resolve_path(results_dir, root=project_root)
