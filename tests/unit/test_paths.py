"""Behavioral: project root is configured/discovered, not derived from package __file__."""

from pathlib import Path

import pytest

from workbench.paths import (
    DEFAULT_RESULTS_SUBDIR,
    resolve_path,
    resolve_project_root,
    resolve_results_dir,
)


def test_explicit_root_wins(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("MLX_WORKBENCH_ROOT", raising=False)
    root = resolve_project_root(explicit=tmp_path)
    assert root == tmp_path.resolve()


def test_env_var_used_when_no_explicit(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("MLX_WORKBENCH_ROOT", str(tmp_path))
    root = resolve_project_root(explicit=None, start=tmp_path / "subdir")
    assert root == tmp_path.resolve()


def test_walk_up_finds_marker(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("MLX_WORKBENCH_ROOT", raising=False)
    (tmp_path / "pyproject.toml").write_text(
        '[project]\nname = "mlx-inference-workbench"\n',
        encoding="utf-8",
    )
    nested = tmp_path / "a" / "b"
    nested.mkdir(parents=True)
    root = resolve_project_root(explicit=None, start=nested)
    assert root == tmp_path.resolve()


def test_fallback_is_cwd_not_package_location(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("MLX_WORKBENCH_ROOT", raising=False)
    monkeypatch.chdir(tmp_path)
    root = resolve_project_root(explicit=None, start=tmp_path)
    assert root == tmp_path.resolve()
    # Must not resolve into site-packages / package install path
    assert "site-packages" not in str(root)


def test_resolve_path_absolute_unchanged(tmp_path: Path):
    p = tmp_path / "abs.jsonl"
    p.write_text("x", encoding="utf-8")
    assert resolve_path(p, root=tmp_path / "other") == p.resolve()


def test_resolve_path_relative_to_project_root(tmp_path: Path):
    (tmp_path / "datasets").mkdir()
    f = tmp_path / "datasets" / "x.jsonl"
    f.write_text("{}", encoding="utf-8")
    assert resolve_path("datasets/x.jsonl", root=tmp_path) == f.resolve()


def test_default_results_dir_under_project_root(tmp_path: Path):
    d = resolve_results_dir(None, project_root=tmp_path)
    assert d == (tmp_path / DEFAULT_RESULTS_SUBDIR).resolve()


def test_results_dir_override_absolute(tmp_path: Path):
    custom = tmp_path / "out"
    assert resolve_results_dir(custom, project_root=tmp_path / "proj") == custom.resolve()
