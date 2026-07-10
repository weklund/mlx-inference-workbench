"""Engine registry — open/closed: add backends without editing orchestrator."""

from __future__ import annotations

from collections.abc import Callable

from workbench.engines.base import Engine

_REGISTRY: dict[str, Callable[[], Engine]] = {}


def register_engine(name: str, factory: Callable[[], Engine]) -> None:
    _REGISTRY[name] = factory


def create_engine(name: str) -> Engine:
    if name not in _REGISTRY:
        # lazy import built-ins
        _ensure_builtins()
    if name not in _REGISTRY:
        known = ", ".join(sorted(_REGISTRY)) or "(none)"
        raise KeyError(f"Unknown engine {name!r}. Known: {known}")
    return _REGISTRY[name]()


def _ensure_builtins() -> None:
    if _REGISTRY:
        return
    from workbench.engines.stub import StubEngine

    register_engine("stub", StubEngine)

    try:
        from workbench.engines.mlx_lm_engine import MlxLmEngine

        register_engine("mlx-lm", MlxLmEngine)
        register_engine("mlx_lm", MlxLmEngine)
    except ImportError:
        pass
