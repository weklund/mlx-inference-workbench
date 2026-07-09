"""Hardware probe — capture fingerprint for reproducibility."""

from __future__ import annotations

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


def _git_sha() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
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


def library_versions() -> dict[str, str]:
    versions: dict[str, str] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
    }
    for mod_name in ("numpy", "scipy", "yaml", "mlflow", "mlx", "mlx_lm"):
        try:
            mod = __import__(mod_name if mod_name != "yaml" else "yaml")
            versions[mod_name] = getattr(mod, "__version__", "unknown")
        except ImportError:
            versions[mod_name] = "not_installed"
    return versions


def capture_fingerprint(profile_name: str) -> dict[str, Any]:
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


def capture_git_sha() -> str | None:
    return _git_sha()
