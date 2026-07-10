"""TDD: engine registry + stub contract (HLD Engine ABC)."""

from collections.abc import Iterator
from contextlib import contextmanager

import pytest

from workbench.config import ModelConfig
from workbench.engines import registry
from workbench.engines.base import Engine, EngineLoadError, GenParams
from workbench.engines.registry import create_engine, register_engine
from workbench.models import GenerationStatus


@contextmanager
def _isolated_registry() -> Iterator[None]:
    """Reset registry globals for tests that mutate registration state."""
    old_reg = registry._REGISTRY.copy()
    old_loaded = registry._STATE["builtins_loaded"]
    registry._REGISTRY.clear()
    registry._STATE["builtins_loaded"] = False
    try:
        yield
    finally:
        registry._REGISTRY.clear()
        registry._REGISTRY.update(old_reg)
        registry._STATE["builtins_loaded"] = old_loaded


def test_create_stub_engine_by_name():
    eng = create_engine("stub")
    assert eng.name() == "stub"


def test_unknown_engine_raises_key_error():
    with pytest.raises(KeyError, match="Unknown engine"):
        create_engine("nope-backend-xyz")


def test_custom_register_before_create_does_not_block_builtins():
    """Registering a plugin first must not prevent lazy built-in load (stub)."""

    class CustomEngine(Engine):
        def name(self) -> str:
            return "custom"

        def load_model(self, config: ModelConfig) -> None:
            return None

        def generate(self, prompt: str, params: GenParams):
            raise NotImplementedError

    with _isolated_registry():
        register_engine("custom", CustomEngine)
        custom = create_engine("custom")
        assert custom.name() == "custom"
        stub = create_engine("stub")
        assert stub.name() == "stub"


def test_custom_registration_of_builtin_name_is_preserved():
    """Explicit register_engine('stub', ...) wins over lazy built-in stub."""

    class HijackStub(Engine):
        def name(self) -> str:
            return "hijacked"

        def load_model(self, config: ModelConfig) -> None:
            return None

        def generate(self, prompt: str, params: GenParams):
            raise NotImplementedError

    with _isolated_registry():
        register_engine("stub", HijackStub)
        eng = create_engine("stub")
        assert eng.name() == "hijacked"
        assert isinstance(eng, HijackStub)


def test_stub_requires_load_before_generate():
    eng = create_engine("stub")
    with pytest.raises(EngineLoadError):
        eng.generate("hello", GenParams(max_tokens=8, seed=1))


def test_stub_generate_contract_and_determinism():
    eng = create_engine("stub")
    eng.load_model(ModelConfig(name="m", quantization="n/a", backend="stub"))
    params = GenParams(max_tokens=8, seed=7)
    a = eng.generate("prompt-a", params)
    b = eng.generate("prompt-a", params)
    assert a.status == GenerationStatus.SUCCESS
    assert len(a.token_timestamps) == a.total_tokens
    assert a.total_tokens == 8
    assert a.acceptance_rate is None  # non-speculative: explicit None
    assert a.output_text == b.output_text  # same seed+prompt → same text
