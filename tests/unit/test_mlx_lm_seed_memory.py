"""TDD: MlxLmEngine applies seed and reports MLX peak memory (not RSS-only)."""

from types import SimpleNamespace

from workbench.config import ModelConfig
from workbench.engines.base import GenParams
from workbench.engines.mlx_lm_engine import MlxLmEngine


def _engine() -> MlxLmEngine:
    eng = MlxLmEngine()
    eng._model = object()
    eng._tokenizer = SimpleNamespace(encode=lambda text: [1, 2, 3])
    eng._config = ModelConfig(
        name="m",
        quantization="4bit",
        backend="mlx-lm",
        model_id="mlx-community/Qwen3-0.6B-4bit",
    )
    return eng


def _install_fake_mlx_lm(monkeypatch, *, stream_generate=None, generate=None):
    import sys

    def _default_generate(*_a, **_k):
        return "ok"

    gen_fn = generate if generate is not None else _default_generate
    sample_utils = SimpleNamespace(make_sampler=lambda **_k: object())
    mlx_mod = SimpleNamespace(generate=gen_fn, sample_utils=sample_utils)
    if stream_generate is not None:
        mlx_mod.stream_generate = stream_generate
    monkeypatch.setitem(sys.modules, "mlx_lm", mlx_mod)
    monkeypatch.setitem(sys.modules, "mlx_lm.sample_utils", sample_utils)


def _install_fake_mlx_core(
    monkeypatch, *, peak_bytes: int = 1_024_000, seed_calls=None, reset_calls=None
):
    import sys

    seeds = seed_calls if seed_calls is not None else []
    resets = reset_calls if reset_calls is not None else []

    class _FakeRandom:
        @staticmethod
        def seed(s: int) -> None:
            seeds.append(int(s))

    class _FakeMx:
        random = _FakeRandom()

        @staticmethod
        def get_peak_memory() -> int:
            return peak_bytes

        @staticmethod
        def reset_peak_memory() -> None:
            resets.append("reset")

    monkeypatch.setitem(sys.modules, "mlx.core", _FakeMx)
    monkeypatch.setitem(sys.modules, "mlx", SimpleNamespace(core=_FakeMx))
    return seeds, resets


def test_generate_seeds_mlx_rng_with_params_seed(monkeypatch):
    """GenParams.seed must reach mx.random.seed before generation (repro)."""
    seeds, _resets = _install_fake_mlx_core(monkeypatch)

    def stream_generate(*_a, **_k):
        yield SimpleNamespace(text="hi")

    _install_fake_mlx_lm(monkeypatch, stream_generate=stream_generate)

    eng = _engine()
    eng.generate("prompt", GenParams(max_tokens=4, seed=42, temperature=0.0))
    assert seeds == [42]


def test_generate_resets_and_reports_mlx_peak_memory_bytes(monkeypatch):
    """Prefer mx.get_peak_memory() over process RSS for memory_peak_bytes."""
    _seeds, resets = _install_fake_mlx_core(monkeypatch, peak_bytes=9_876_543)

    def stream_generate(*_a, **_k):
        yield SimpleNamespace(text="a")
        yield SimpleNamespace(text="b")

    _install_fake_mlx_lm(monkeypatch, stream_generate=stream_generate)

    eng = _engine()
    result = eng.generate("prompt", GenParams(max_tokens=4, seed=7))
    assert resets == ["reset"]
    assert result.memory_peak_bytes == 9_876_543


def test_get_memory_usage_bytes_falls_back_when_mlx_unavailable(monkeypatch):
    """If mlx.core peak APIs fail, still return a non-negative int (RSS or 0)."""
    import sys

    class _BrokenMx:
        @staticmethod
        def get_peak_memory() -> int:
            raise RuntimeError("no metal")

        @staticmethod
        def reset_peak_memory() -> None:
            raise RuntimeError("no metal")

    monkeypatch.setitem(sys.modules, "mlx.core", _BrokenMx)
    eng = _engine()
    n = eng.get_memory_usage_bytes()
    assert isinstance(n, int)
    assert n >= 0
