"""Unit tests for scripts/archive_run_artifacts.py (import by path)."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "archive_run_artifacts.py"


def _load():
    spec = importlib.util.spec_from_file_location("archive_run_artifacts", _SCRIPT)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def arch():
    return _load()


def _fake_run(results: Path, run_id: str, *, summary: bytes, parquet: bytes) -> None:
    d = results / run_id
    d.mkdir(parents=True)
    (d / "summary.json").write_bytes(summary)
    (d / "iterations.parquet").write_bytes(parquet)


def test_archive_and_verify_roundtrip(tmp_path: Path, arch) -> None:
    results = tmp_path / "results"
    dest = tmp_path / "artifacts"
    _fake_run(results, "aaa111", summary=b'{"ok":1}', parquet=b"PAR1")
    _fake_run(results, "bbb222", summary=b'{"ok":2}', parquet=b"PAR2")

    arch.archive_run("aaa111", results_dir=results, dest_root=dest)
    arch.archive_run("bbb222", results_dir=results, dest_root=dest)
    manifest = arch.write_manifest(dest)
    assert manifest.name == "SHA256SUMS"
    ok = arch.verify_manifest(dest)
    assert set(ok) == {
        "aaa111/summary.json",
        "aaa111/iterations.parquet",
        "bbb222/summary.json",
        "bbb222/iterations.parquet",
    }


def test_refuse_overwrite_without_force(tmp_path: Path, arch) -> None:
    results = tmp_path / "results"
    dest = tmp_path / "artifacts"
    _fake_run(results, "r1", summary=b"v1", parquet=b"p1")
    arch.archive_run("r1", results_dir=results, dest_root=dest)

    (results / "r1" / "summary.json").write_bytes(b"v2")
    with pytest.raises(arch.ArchiveError, match="refusing to overwrite"):
        arch.archive_run("r1", results_dir=results, dest_root=dest, force=False)

    arch.archive_run("r1", results_dir=results, dest_root=dest, force=True)
    assert (dest / "r1" / "summary.json").read_bytes() == b"v2"


def test_missing_source_fails_closed(tmp_path: Path, arch) -> None:
    with pytest.raises(arch.ArchiveError, match="not found"):
        arch.archive_run("missing", results_dir=tmp_path / "results", dest_root=tmp_path / "a")


def test_verify_detects_tamper(tmp_path: Path, arch) -> None:
    results = tmp_path / "results"
    dest = tmp_path / "artifacts"
    _fake_run(results, "r1", summary=b"good", parquet=b"p")
    arch.archive_run("r1", results_dir=results, dest_root=dest)
    arch.write_manifest(dest)
    (dest / "r1" / "summary.json").write_bytes(b"evil")
    with pytest.raises(arch.ArchiveError, match="checksum mismatch"):
        arch.verify_manifest(dest)
