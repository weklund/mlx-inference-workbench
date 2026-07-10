"""Unit tests for MTPLX engine (mocked package — CI-safe)."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from workbench.config import ModelConfig, MtplxOptions
from workbench.engines.base import EngineLoadError, GenerationError, GenParams
from workbench.engines.mtplx_engine import (
    MtplxEngine,
    resolve_mtplx_model_path,
    speculative_metrics_from_stats,
)
from workbench.models import GenerationStatus


def test_speculative_metrics_match_mtplx_progress_formulas():
    stats = SimpleNamespace(accepted_drafts=12, drafted_tokens=20, verify_calls=5)
    rate, mean_len = speculative_metrics_from_stats(stats)
    assert rate == pytest.approx(12 / 20)
    assert mean_len == pytest.approx(12 / 5)


def test_speculative_metrics_none_when_no_drafts():
    stats = SimpleNamespace(accepted_drafts=0, drafted_tokens=0, verify_calls=0)
    assert speculative_metrics_from_stats(stats) == (None, None)


def test_speculative_metrics_mean_none_without_verify_calls():
    stats = SimpleNamespace(accepted_drafts=4, drafted_tokens=8, verify_calls=0)
    rate, mean_len = speculative_metrics_from_stats(stats)
    assert rate == pytest.approx(0.5)
    assert mean_len is None


def test_supports_speculative_and_name():
    eng = MtplxEngine()
    assert eng.name() == "mtplx"
    assert eng.supports_speculative() is True


def test_requires_load_before_generate():
    eng = MtplxEngine()
    with pytest.raises(EngineLoadError, match="load_model"):
        eng.generate("hi", GenParams(max_tokens=4, seed=1))


def test_generate_contract_e2e_and_acceptance(monkeypatch):
    """Mocked generate_mtpk: e2e-only timestamps + non-null acceptance when drafts."""
    import sys

    class _FakeRandom:
        @staticmethod
        def seed(_s: int) -> None:
            return None

    class _FakeMx:
        random = _FakeRandom()

        @staticmethod
        def get_peak_memory() -> int:
            return 2048

        @staticmethod
        def reset_peak_memory() -> None:
            return None

    monkeypatch.setitem(sys.modules, "mlx.core", _FakeMx)
    monkeypatch.setitem(sys.modules, "mlx", SimpleNamespace(core=_FakeMx))

    stats = SimpleNamespace(accepted_drafts=6, drafted_tokens=10, verify_calls=3)
    fake_out = SimpleNamespace(
        text="hello world",
        tokens=[1, 2, 3],
        stats=stats,
    )

    captured: dict = {}

    def fake_generate_mtpk(rt, prompt_ids, **kwargs):
        captured["rt"] = rt
        captured["prompt_ids"] = prompt_ids
        captured["kwargs"] = kwargs
        return fake_out

    def make_sampler(**kw):
        return SimpleNamespace(**kw)

    gen_mod = SimpleNamespace(
        SamplerConfig=make_sampler,
        generate_mtpk=fake_generate_mtpk,
    )
    monkeypatch.setitem(sys.modules, "mtplx.generation", gen_mod)
    # Ensure import mtplx.generation works if package parent missing
    monkeypatch.setitem(sys.modules, "mtplx", SimpleNamespace(generation=gen_mod))

    eng = MtplxEngine()
    eng._runtime = SimpleNamespace(
        tokenizer=SimpleNamespace(encode=lambda text: [10, 11, 12]),
    )
    eng._config = ModelConfig(
        name="mtp-model",
        quantization="4bit",
        backend="mtplx",
        model_id="fake/model",
        mtplx=MtplxOptions(speculative_depth=5),
    )

    result = eng.generate("prompt", GenParams(max_tokens=16, seed=7, temperature=0.0))

    assert result.status == GenerationStatus.SUCCESS
    assert result.output_text == "hello world"
    assert result.token_timestamps == []
    assert result.ttft_ms is None
    assert result.total_tokens == 3
    assert result.e2e_ms is not None and result.e2e_ms >= 0
    assert result.memory_peak_bytes == 2048
    assert result.acceptance_rate == pytest.approx(0.6)
    assert result.accepted_length_mean == pytest.approx(2.0)
    assert captured["kwargs"]["speculative_depth"] == 5
    assert captured["kwargs"]["seed"] == 7
    assert captured["kwargs"]["max_tokens"] == 16
    assert captured["prompt_ids"] == [10, 11, 12]


def test_generate_token_count_failure_is_generation_error(monkeypatch):
    """Empty tokens list + encode failure must not invent word-split counts."""
    import sys

    class _FakeMx:
        class random:
            @staticmethod
            def seed(_s: int) -> None:
                return None

        @staticmethod
        def get_peak_memory() -> int:
            return 0

        @staticmethod
        def reset_peak_memory() -> None:
            return None

    monkeypatch.setitem(sys.modules, "mlx.core", _FakeMx)
    monkeypatch.setitem(sys.modules, "mlx", SimpleNamespace(core=_FakeMx))

    def selective_encode(text: str):
        # Prompt path succeeds; output recount fails (empty tokens list).
        if text == "p":
            return [1]
        raise RuntimeError("encode exploded")

    def fake_generate_mtpk(*_a, **_k):
        return SimpleNamespace(
            text="hello world from model",
            tokens=[],
            stats=SimpleNamespace(drafted_tokens=0, accepted_drafts=0, verify_calls=0),
        )

    def make_sampler(**kw):
        return SimpleNamespace(**kw)

    gen_mod = SimpleNamespace(SamplerConfig=make_sampler, generate_mtpk=fake_generate_mtpk)
    monkeypatch.setitem(sys.modules, "mtplx.generation", gen_mod)
    monkeypatch.setitem(sys.modules, "mtplx", SimpleNamespace(generation=gen_mod))

    eng = MtplxEngine()
    eng._runtime = SimpleNamespace(tokenizer=SimpleNamespace(encode=selective_encode))
    eng._config = ModelConfig(name="m", quantization="q", backend="mtplx")

    with pytest.raises(GenerationError, match="Failed to count output tokens") as ei:
        eng.generate("p", GenParams(max_tokens=4, seed=1))
    assert isinstance(ei.value.__cause__, RuntimeError)


def test_generate_rejects_output_without_text(monkeypatch):
    """API drift: missing .text must not become str(output) comparison garbage."""
    import sys

    class _FakeMx:
        class random:
            @staticmethod
            def seed(_s: int) -> None:
                return None

        @staticmethod
        def get_peak_memory() -> int:
            return 0

        @staticmethod
        def reset_peak_memory() -> None:
            return None

    monkeypatch.setitem(sys.modules, "mlx.core", _FakeMx)
    monkeypatch.setitem(sys.modules, "mlx", SimpleNamespace(core=_FakeMx))

    def fake_generate_mtpk(*_a, **_k):
        return SimpleNamespace(tokens=[1], stats=SimpleNamespace(drafted_tokens=0))

    def make_sampler(**kw):
        return SimpleNamespace(**kw)

    gen_mod = SimpleNamespace(
        SamplerConfig=make_sampler,
        generate_mtpk=fake_generate_mtpk,
    )
    monkeypatch.setitem(sys.modules, "mtplx.generation", gen_mod)
    monkeypatch.setitem(sys.modules, "mtplx", SimpleNamespace(generation=gen_mod))

    eng = MtplxEngine()
    eng._runtime = SimpleNamespace(tokenizer=SimpleNamespace(encode=lambda _t: [1]))
    eng._config = ModelConfig(name="m", quantization="q", backend="mtplx")

    with pytest.raises(GenerationError, match="lacks required 'text'"):
        eng.generate("p", GenParams(max_tokens=4, seed=1))


def test_load_model_missing_package(monkeypatch):
    import builtins

    real_import = builtins.__import__

    def blocked(name, *args, **kwargs):
        if name == "mtplx" or name.startswith("mtplx."):
            raise ImportError("blocked for test")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", blocked)
    eng = MtplxEngine()
    with pytest.raises(EngineLoadError, match="mtplx is not installed"):
        eng.load_model(ModelConfig(name="m", quantization="4bit", backend="mtplx", model_id="x"))


def test_registry_exposes_mtplx():
    from workbench.engines.registry import create_engine

    eng = create_engine("mtplx")
    assert eng.name() == "mtplx"
    assert eng.supports_speculative() is True


def test_resolve_local_model_dir(tmp_path):
    (tmp_path / "config.json").write_text("{}", encoding="utf-8")
    assert resolve_mtplx_model_path(str(tmp_path)) == str(tmp_path.resolve())


def test_resolve_hf_id_uses_snapshot_download(monkeypatch):
    def fake_snapshot(*, repo_id: str):
        assert repo_id == "Qwen/Qwen3.5-0.8B"
        return "/tmp/fake-qwen-snapshot"

    import sys

    hub = SimpleNamespace(snapshot_download=fake_snapshot)
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub)
    assert resolve_mtplx_model_path("Qwen/Qwen3.5-0.8B") == "/tmp/fake-qwen-snapshot"


def test_resolve_missing_local_raises():
    with pytest.raises(EngineLoadError, match="not found"):
        resolve_mtplx_model_path("/no/such/mtplx/model/dir")
