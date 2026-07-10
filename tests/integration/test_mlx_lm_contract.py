"""Integration: mlx-lm engine satisfies the Engine contract (CI-safe, no weights)."""

from types import SimpleNamespace

import pytest

from workbench.config import ModelConfig
from workbench.engines.base import Engine, EngineLoadError, GenParams
from workbench.engines.mlx_lm_engine import MlxLmEngine
from workbench.engines.registry import create_engine
from workbench.models import GenerationStatus


@pytest.mark.integration
def test_registry_exposes_mlx_lm_backend():
    eng = create_engine("mlx-lm")
    assert isinstance(eng, Engine)
    assert eng.name() == "mlx-lm"
    # alias
    eng2 = create_engine("mlx_lm")
    assert eng2.name() == "mlx-lm"


@pytest.mark.integration
def test_mlx_lm_requires_load_before_generate():
    eng = MlxLmEngine()
    with pytest.raises(EngineLoadError, match="load_model"):
        eng.generate("hi", GenParams(max_tokens=4, seed=1))


@pytest.mark.integration
def test_mlx_lm_generate_contract_shape(monkeypatch):
    """Mocked mlx-lm: SUCCESS + timestamps alignment + explicit speculative Nones."""
    import sys

    class _FakeRandom:
        @staticmethod
        def seed(_s: int) -> None:
            return None

    class _FakeMx:
        random = _FakeRandom()

        @staticmethod
        def get_peak_memory() -> int:
            return 1000

        @staticmethod
        def reset_peak_memory() -> None:
            return None

    monkeypatch.setitem(sys.modules, "mlx.core", _FakeMx)
    monkeypatch.setitem(sys.modules, "mlx", SimpleNamespace(core=_FakeMx))

    def stream_generate(*_a, **_k):
        for t in ("hel", "lo"):
            yield SimpleNamespace(text=t)

    sample_utils = SimpleNamespace(make_sampler=lambda **_k: object())
    mlx_mod = SimpleNamespace(
        stream_generate=stream_generate,
        generate=lambda *_a, **_k: "unused",
        sample_utils=sample_utils,
    )
    monkeypatch.setitem(sys.modules, "mlx_lm", mlx_mod)
    monkeypatch.setitem(sys.modules, "mlx_lm.sample_utils", sample_utils)

    eng = MlxLmEngine()
    eng._model = object()
    eng._tokenizer = SimpleNamespace(encode=lambda text: list(range(max(1, len(text)))))
    eng._config = ModelConfig(
        name="qwen3-0.6b",
        quantization="4bit",
        backend="mlx-lm",
        model_id="mlx-community/Qwen3-0.6B-4bit",
    )

    result = eng.generate("hello", GenParams(max_tokens=8, seed=42, temperature=0.0))
    assert result.status == GenerationStatus.SUCCESS
    assert result.output_text == "hello"
    assert result.ttft_ms is not None
    assert len(result.token_timestamps) == result.total_tokens
    assert result.total_tokens == 2
    assert result.acceptance_rate is None
    assert result.accepted_length_mean is None
    assert result.e2e_ms is not None and result.e2e_ms >= 0
    assert result.memory_peak_bytes == 1000
    assert eng.supports_speculative() is False
    assert eng.score_correctness(result, reference="hello") is True
    assert eng.score_correctness(result, reference="nope") is False
