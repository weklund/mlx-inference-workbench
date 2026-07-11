#!/usr/bin/env python3
"""Copy benchmark run artifacts into a durable experiment tree with SHA-256.

Harness output under ``benchmarks/results/<run_id>/`` is gitignored. Official
experiment claims should pin immutable copies under
``experiments/<NN_name>/artifacts/<run_id>/`` plus a ``SHA256SUMS`` manifest.

Usage:
    uv run python scripts/archive_run_artifacts.py archive \\
        --run-id 81aff0e72f89 --run-id f2f5a38b129d \\
        --dest experiments/02_stock_mlx_comparison/artifacts

    uv run python scripts/archive_run_artifacts.py verify \\
        --dest experiments/02_stock_mlx_comparison/artifacts

    make bench-archive RUNS="81aff0e72f89 f2f5a38b129d" DEST=experiments/02_stock_mlx_comparison/artifacts
"""

from __future__ import annotations

import hashlib
from pathlib import Path
import sys

import click

# Files required for recomputing distributions / effect sizes from a run.
ARCHIVED_NAMES: tuple[str, ...] = ("summary.json", "iterations.parquet")
MANIFEST_NAME = "SHA256SUMS"
CHUNK = 1 << 20


class ArchiveError(Exception):
    """Fail-closed archive / verify error."""


def sha256_file(path: Path) -> str:
    """Return hex SHA-256 of a file (streaming)."""
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        while True:
            chunk = fh.read(CHUNK)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def run_source_dir(results_dir: Path, run_id: str) -> Path:
    """Resolve ``results_dir / run_id`` and require a directory."""
    src = (results_dir / run_id).expanduser().resolve()
    if not src.is_dir():
        raise ArchiveError(f"run directory not found: {src}")
    return src


def archive_run(
    run_id: str,
    *,
    results_dir: Path,
    dest_root: Path,
    force: bool = False,
) -> list[Path]:
    """Copy required artifacts for one run into ``dest_root / run_id``.

    Args:
        run_id: Harness run id (directory name under results_dir).
        results_dir: Local harness results root (usually benchmarks/results).
        dest_root: Experiment artifacts root (e.g. experiments/02_.../artifacts).
        force: When True, overwrite existing files that differ in content.

    Returns:
        Paths of files written or already present and identical.

    Raises:
        ArchiveError: Missing source files or immutable conflict without force.
    """
    src = run_source_dir(results_dir, run_id)
    out_dir = dest_root / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    for name in ARCHIVED_NAMES:
        src_file = src / name
        if not src_file.is_file():
            raise ArchiveError(f"missing required artifact {name!r} under {src}")
        dest_file = out_dir / name
        src_hash = sha256_file(src_file)
        if dest_file.is_file():
            dest_hash = sha256_file(dest_file)
            if dest_hash == src_hash:
                written.append(dest_file)
                continue
            if not force:
                raise ArchiveError(
                    f"refusing to overwrite {dest_file} (content differs; "
                    f"pass --force to replace, or keep the published bytes)"
                )
        dest_file.write_bytes(src_file.read_bytes())
        if sha256_file(dest_file) != src_hash:
            raise ArchiveError(f"copy integrity failed for {dest_file}")
        written.append(dest_file)

    return written


def iter_archived_files(dest_root: Path) -> list[Path]:
    """List ``summary.json`` / ``iterations.parquet`` under dest, sorted."""
    if not dest_root.is_dir():
        return []
    found: list[Path] = []
    for name in ARCHIVED_NAMES:
        found.extend(sorted(dest_root.glob(f"*/{name}")))
    return sorted(found, key=lambda p: p.relative_to(dest_root).as_posix())


def write_manifest(dest_root: Path) -> Path:
    """Rewrite ``SHA256SUMS`` for all archived files under dest_root.

    Paths in the manifest are relative to ``dest_root`` (``shasum -c``-friendly
    when cwd is dest_root).
    """
    dest_root = dest_root.expanduser().resolve()
    dest_root.mkdir(parents=True, exist_ok=True)
    files = iter_archived_files(dest_root)
    if not files:
        raise ArchiveError(f"no archived files under {dest_root}")
    lines = [f"{sha256_file(p)}  {p.relative_to(dest_root).as_posix()}" for p in files]
    manifest = dest_root / MANIFEST_NAME
    manifest.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return manifest


def parse_manifest(manifest: Path) -> list[tuple[str, str]]:
    """Parse ``SHA256SUMS`` lines into ``(hex_digest, relative_path)`` pairs."""
    rows: list[tuple[str, str]] = []
    text = manifest.read_text(encoding="utf-8")
    for lineno, raw in enumerate(text.splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        # BSD/GNU: "<hash><two spaces or space+*><path>"
        parts = line.split(None, 1)
        if len(parts) != 2:
            raise ArchiveError(f"{manifest}:{lineno}: malformed line: {raw!r}")
        digest, rel = parts[0], parts[1].removeprefix("*")
        if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest.lower()):
            raise ArchiveError(f"{manifest}:{lineno}: bad digest: {digest!r}")
        rows.append((digest.lower(), rel))
    if not rows:
        raise ArchiveError(f"{manifest}: empty manifest")
    return rows


def verify_manifest(dest_root: Path) -> list[str]:
    """Verify all entries in ``dest_root/SHA256SUMS``; return OK paths.

    Raises:
        ArchiveError: On missing file or digest mismatch.
    """
    dest_root = dest_root.expanduser().resolve()
    manifest = dest_root / MANIFEST_NAME
    if not manifest.is_file():
        raise ArchiveError(f"manifest not found: {manifest}")
    ok: list[str] = []
    for expected, rel in parse_manifest(manifest):
        path = dest_root / rel
        if not path.is_file():
            raise ArchiveError(f"missing file for manifest entry: {rel}")
        actual = sha256_file(path)
        if actual != expected:
            raise ArchiveError(f"checksum mismatch for {rel}: expected {expected}, got {actual}")
        ok.append(rel)
    return ok


@click.group()
def main() -> None:
    """Archive harness runs into experiment artifact trees with SHA-256."""


@main.command("archive")
@click.option(
    "--run-id",
    "run_ids",
    multiple=True,
    required=True,
    help="Run id under results-dir (repeatable).",
)
@click.option(
    "--dest",
    "dest_root",
    type=click.Path(path_type=Path),
    required=True,
    help="Experiment artifacts directory (e.g. experiments/02_.../artifacts).",
)
@click.option(
    "--results-dir",
    type=click.Path(path_type=Path),
    default=Path("benchmarks/results"),
    show_default=True,
    help="Harness results root.",
)
@click.option(
    "--force",
    is_flag=True,
    help="Overwrite existing artifact files when content differs.",
)
def archive_cmd(
    run_ids: tuple[str, ...],
    dest_root: Path,
    results_dir: Path,
    force: bool,
) -> None:
    """Copy summary.json + iterations.parquet and rewrite SHA256SUMS."""
    try:
        dest = dest_root.expanduser().resolve()
        results = results_dir.expanduser().resolve()
        for run_id in run_ids:
            paths = archive_run(run_id, results_dir=results, dest_root=dest, force=force)
            click.echo(f"archived {run_id}: {len(paths)} file(s) → {dest / run_id}")
        manifest = write_manifest(dest)
        click.echo(f"wrote {manifest}")
        ok = verify_manifest(dest)
        click.echo(f"verify OK ({len(ok)} entries)")
    except ArchiveError as e:
        click.echo(f"error: {e}", err=True)
        sys.exit(1)


@main.command("verify")
@click.option(
    "--dest",
    "dest_root",
    type=click.Path(path_type=Path),
    required=True,
    help="Experiment artifacts directory containing SHA256SUMS.",
)
def verify_cmd(dest_root: Path) -> None:
    """Verify SHA256SUMS against files on disk."""
    try:
        ok = verify_manifest(dest_root)
        for rel in ok:
            click.echo(f"OK  {rel}")
        click.echo(f"verify OK ({len(ok)} entries)")
    except ArchiveError as e:
        click.echo(f"error: {e}", err=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
