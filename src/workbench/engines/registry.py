"""Engine registry — open/closed: add backends without editing orchestrator."""

from __future__ import annotations

from collections.abc import Callable

from workbench.engines.base import Engine

_REGISTRY: dict[str, Callable[[], Engine]] = {}
# Mutable flag container avoids module-level ``global`` (Ruff PLW0603).
_STATE: dict[str, bool] = {"builtins_loaded": False}


def register_engine(name: str, factory: Callable[[], Engine]) -> None:
    """Register a backend factory under ``name`` (overwrites existing)."""
    _REGISTRY[name] = factory


def create_engine(name: str) -> Engine:
    """Instantiate a registered engine; loads built-ins lazily once.

    Custom engines may be registered before any create_engine call without
    blocking built-in discovery. Custom factories already present under a
    built-in name are preserved (not overwritten by lazy load).

    Args:
        name: Backend key from experiment config (e.g. ``mlx-lm``).

    Returns:
        Fresh Engine instance.

    Raises:
        KeyError: Unknown backend name.
    """
    _ensure_builtins()
    if name not in _REGISTRY:
        known = ", ".join(sorted(_REGISTRY)) or "(none)"
        raise KeyError(f"Unknown engine {name!r}. Known: {known}")
    return _REGISTRY[name]()


def _register_builtin(name: str, factory: Callable[[], Engine]) -> None:
    """Register a built-in only if ``name`` is not already taken."""
    if name not in _REGISTRY:
        register_engine(name, factory)


def _ensure_builtins() -> None:
    """Import and register built-in backends once (idempotent)."""
    if _STATE["builtins_loaded"]:
        return
    _STATE["builtins_loaded"] = True

    from workbench.engines.stub import StubEngine

    _register_builtin("stub", StubEngine)

    try:
        from workbench.engines.mlx_lm_engine import MlxLmEngine

        _register_builtin("mlx-lm", MlxLmEngine)
        _register_builtin("mlx_lm", MlxLmEngine)
    except ImportError:
        pass

    # Register even when the optional ``mtplx`` package is missing so
    # create_engine("mtplx") returns a plugin that fails at load_model with
    # a clear EngineLoadError (mirrors optional-dep UX for local smoke).
    try:
        from workbench.engines.mtplx_engine import MtplxEngine

        _register_builtin("mtplx", MtplxEngine)
    except ImportError:
        pass
