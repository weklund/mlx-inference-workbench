"""Hardware probe — capture fingerprint for reproducibility."""

from __future__ import annotations

from pathlib import Path
import platform
import subprocess
import sys
from typing import Any


def _sysctl(key: str) -> str | None:
    try:
        out = subprocess.run(
            ["sysctl", "-n", key],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        if out.returncode == 0:
            return out.stdout.strip()
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return None


def _git_sha(repo_root: Path | None = None) -> str | None:
    """Return HEAD for the target repository (cwd=repo_root), not the process cwd."""
    try:
        kwargs: dict[str, Any] = {
            "args": ["git", "rev-parse", "HEAD"],
            "capture_output": True,
            "text": True,
            "timeout": 5,
            "check": False,
        }
        if repo_root is not None:
            kwargs["cwd"] = str(Path(repo_root).resolve())
        out = subprocess.run(**kwargs)
        if out.returncode == 0:
            return out.stdout.strip() or None
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return None


def _package_version(import_name: str, *, dist_names: tuple[str, ...] = ()) -> str:
    """Resolve a library version for run metadata (report dimension, not a pin).

    Prefer importlib.metadata (works when ``__version__`` is missing, e.g. top-level
    ``mlx``), then module ``__version__``, then nested ``mlx.core`` for MLX.
    """
    import importlib.metadata as importlib_metadata

    candidates = dist_names or (import_name.replace("_", "-"), import_name)
    for dist in candidates:
        try:
            return importlib_metadata.version(dist)
        except importlib_metadata.PackageNotFoundError:
            continue

    try:
        mod = __import__(import_name)
    except ImportError:
        return "not_installed"

    ver = getattr(mod, "__version__", None)
    if ver:
        return str(ver)

    # mlx: version lives on mlx.core, not the package root
    if import_name == "mlx":
        try:
            import mlx.core as mx

            nested = getattr(mx, "__version__", None)
            if nested:
                return str(nested)
        except ImportError:
            pass

    return "unknown"


def library_versions() -> dict[str, str]:
    """Capture interpreter + key deps as a free report dimension.

    Stored on every run when ``reproducibility.record_env_versions`` is true.
    Not enforced by the comparability gate — same backend across library versions
    is a valid experiment factor (meta-analysis over upgrades).
    """
    versions: dict[str, str] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
    }
    # import name → optional PyPI dist name overrides
    packages: tuple[tuple[str, tuple[str, ...]], ...] = (
        ("numpy", ()),
        ("scipy", ()),
        ("yaml", ("PyYAML", "pyyaml", "yaml")),
        ("mlflow", ()),
        ("mlx", ("mlx",)),
        ("mlx_lm", ("mlx-lm", "mlx_lm")),
    )
    for import_name, dist_names in packages:
        versions[import_name] = _package_version(import_name, dist_names=dist_names)
    return versions


def capture_fingerprint(profile_name: str) -> dict[str, Any]:
    """Capture host identity fields for run metadata / comparability.

    Args:
        profile_name: Hardware profile id from experiment config.

    Returns:
        Dict with chip, memory, OS, and profile name.
    """
    mem = _sysctl("hw.memsize")
    chip = _sysctl("machdep.cpu.brand_string") or _sysctl("hw.model")
    # Apple Silicon machine
    ncpu = _sysctl("hw.ncpu")
    return {
        "profile": profile_name,
        "chip": chip,
        "hw_model": _sysctl("hw.model"),
        "memsize_bytes": int(mem) if mem and mem.isdigit() else mem,
        "ncpu": ncpu,
        "os": platform.platform(),
        "machine": platform.machine(),
    }


def capture_git_sha(repo_root: Path | None = None) -> str | None:
    """Capture git HEAD for *repo_root* (project/data root), not process cwd."""
    return _git_sha(repo_root)
