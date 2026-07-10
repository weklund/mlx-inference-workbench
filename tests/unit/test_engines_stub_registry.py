"""TDD: engine registry + stub contract (HLD Engine ABC)."""

import pytest

from workbench.config import ModelConfig
from workbench.engines.base import EngineLoadError, GenParams
from workbench.engines.registry import create_engine
from workbench.models import GenerationStatus


def test_create_stub_engine_by_name():
    eng = create_engine("stub")
    assert eng.name() == "stub"


def test_unknown_engine_raises_key_error():
    with pytest.raises(KeyError, match="Unknown engine"):
        create_engine("nope-backend-xyz")


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
