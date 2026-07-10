"""Git SHA capture must use the target repo root, not process cwd."""

from pathlib import Path
import subprocess

from workbench.hardware import capture_git_sha, library_versions


def _init_repo(path: Path, message: str) -> str:
    path.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "init"], cwd=path, check=True, capture_output=True)
    subprocess.run(
        ["git", "config", "user.email", "test@example.com"],
        cwd=path,
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "test"],
        cwd=path,
        check=True,
        capture_output=True,
    )
    (path / "README").write_text(message, encoding="utf-8")
    subprocess.run(["git", "add", "README"], cwd=path, check=True, capture_output=True)
    subprocess.run(
        ["git", "commit", "-m", message],
        cwd=path,
        check=True,
        capture_output=True,
    )
    out = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=path,
        check=True,
        capture_output=True,
        text=True,
    )
    return out.stdout.strip()


def test_capture_git_sha_uses_repo_root_not_cwd(tmp_path: Path, monkeypatch):
    repo_a = tmp_path / "repo_a"
    repo_b = tmp_path / "repo_b"
    sha_a = _init_repo(repo_a, "commit-a")
    sha_b = _init_repo(repo_b, "commit-b")
    assert sha_a != sha_b

    # Process cwd is an unrelated directory; SHA must still come from repo_a.
    other = tmp_path / "elsewhere"
    other.mkdir()
    monkeypatch.chdir(other)

    assert capture_git_sha(repo_a) == sha_a
    assert capture_git_sha(repo_b) == sha_b


def test_capture_git_sha_none_without_git_repo(tmp_path: Path, monkeypatch):
    empty = tmp_path / "not-a-repo"
    empty.mkdir()
    monkeypatch.chdir(empty)
    assert capture_git_sha(empty) is None


def test_library_versions_records_mlx_and_mlx_lm_as_report_dimension():
    """Versions are a free metadata dimension for reports (not a pin / gate)."""
    v = library_versions()
    assert "python" in v
    assert "platform" in v
    # Installed in workbench env; must not be unknown/not_installed
    assert v["mlx"] not in {"unknown", "not_installed"}
    assert v["mlx_lm"] not in {"unknown", "not_installed"}
    # Sanity: look like version strings
    assert any(c.isdigit() for c in v["mlx"])
    assert any(c.isdigit() for c in v["mlx_lm"])
