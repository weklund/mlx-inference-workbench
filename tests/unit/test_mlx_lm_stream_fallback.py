"""stream_generate fallback must not swallow real generation failures."""

from types import SimpleNamespace

import pytest

from workbench.config import ModelConfig
from workbench.engines.base import GenerationError, GenParams
from workbench.engines.mlx_lm_engine import MlxLmEngine


def _engine() -> MlxLmEngine:
    eng = MlxLmEngine()
    eng._model = object()
    eng._tokenizer = SimpleNamespace(encode=lambda text: [1, 2, 3])
    eng._config = ModelConfig(name="m", quantization="q", backend="mlx-lm")
    return eng


def _install_fake_mlx(monkeypatch, *, stream_generate=None, generate=None):
    """Install a minimal fake mlx_lm module for engine.generate imports."""
    import sys

    def _default_generate(*_a, **_k):
        return "fallback-text"

    gen_fn = generate if generate is not None else _default_generate
    sample_utils = SimpleNamespace(make_sampler=lambda **_k: object())
    mlx_mod = SimpleNamespace(generate=gen_fn, sample_utils=sample_utils)
    if stream_generate is not None:
        mlx_mod.stream_generate = stream_generate

    monkeypatch.setitem(sys.modules, "mlx_lm", mlx_mod)
    monkeypatch.setitem(sys.modules, "mlx_lm.sample_utils", sample_utils)


def test_stream_runtime_error_does_not_fall_back_to_mlx_generate(monkeypatch):
    """Genuine stream failure must become GenerationError, not a second generate."""
    calls: list[str] = []

    def boom_stream(*_a, **_k):
        calls.append("stream")
        raise RuntimeError("OOM during stream")
        yield  # pragma: no cover

    def mlx_generate(*_a, **_k):
        calls.append("generate")
        return "should-not-run"

    _install_fake_mlx(monkeypatch, stream_generate=boom_stream, generate=mlx_generate)

    eng = _engine()
    with pytest.raises(GenerationError, match="OOM"):
        eng.generate("hello", GenParams(max_tokens=4, seed=1))
    assert calls == ["stream"]


def test_missing_stream_api_is_e2e_only_no_fake_timestamps(monkeypatch):
    _install_fake_mlx(monkeypatch, stream_generate=None, generate=lambda *_a, **_k: "fallback-text")

    eng = _engine()
    result = eng.generate("hello", GenParams(max_tokens=4, seed=1))
    assert result.output_text == "fallback-text"
    assert result.total_tokens == 3  # from tokenizer.encode
    assert result.token_timestamps == []
    assert result.ttft_ms is None
    assert result.e2e_ms is not None and result.e2e_ms > 0


def test_type_error_stream_signature_is_e2e_only(monkeypatch):
    def bad_stream(*_a, **_k):
        raise TypeError("unexpected keyword argument 'sampler'")

    _install_fake_mlx(
        monkeypatch,
        stream_generate=bad_stream,
        generate=lambda *_a, **_k: "compat-fallback",
    )

    eng = _engine()
    result = eng.generate("hello", GenParams(max_tokens=4, seed=1))
    assert result.output_text == "compat-fallback"
    assert result.token_timestamps == []
    assert result.ttft_ms is None
    assert result.e2e_ms is not None and result.e2e_ms > 0


def test_stream_path_reports_measured_ttft(monkeypatch):
    def stream_generate(*_a, **_k):
        for text in ("hel", "lo"):
            yield SimpleNamespace(text=text)

    _install_fake_mlx(monkeypatch, stream_generate=stream_generate)

    eng = _engine()
    result = eng.generate("hello", GenParams(max_tokens=4, seed=1))
    assert result.output_text == "hello"
    assert result.ttft_ms is not None and result.ttft_ms >= 0
    assert len(result.token_timestamps) == 2
    assert result.total_tokens == 2
