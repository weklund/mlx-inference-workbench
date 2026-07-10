"""Timeout helper behavior; mlx-lm must not own timeout policy."""

from workbench.config import ModelConfig
from workbench.engines.base import GenParams, iter_warmup_prompts
from workbench.engines.mlx_lm_engine import MlxLmEngine
from workbench.models import GenerationResult, GenerationStatus, ThermalReading


def _ok_result(text: str = "ok") -> GenerationResult:
    return GenerationResult(
        status=GenerationStatus.SUCCESS,
        output_text=text,
        token_timestamps=[0.01, 0.02],
        ttft_ms=10.0,
        total_tokens=2,
        memory_peak_bytes=0,
        thermal_state=ThermalReading(method="test"),
        e2e_ms=20.0,
    )


def test_warmup_iterator_drives_mlx_generate(monkeypatch):
    eng = MlxLmEngine()
    eng._model = object()
    eng._tokenizer = object()
    eng._config = ModelConfig(name="m", quantization="q", backend="mlx-lm")

    calls: list[str] = []

    def fake_generate(self, prompt, params):
        calls.append(prompt)
        return _ok_result()

    monkeypatch.setattr(MlxLmEngine, "generate", fake_generate)
    params = GenParams(max_tokens=8, seed=1, timeout_sec=0.5)
    for p in iter_warmup_prompts(["prompt-a", "prompt-b"], n=2):
        eng.generate(p, params)

    assert calls == ["prompt-a", "prompt-a"]


def test_timed_generate_returns_timeout_status():
    import time

    from workbench.engines.timeout import timed_generate

    class SlowEngine:
        def generate(self, prompt, params):
            time.sleep(1.0)
            return _ok_result()

    result = timed_generate(
        SlowEngine(),  # type: ignore[arg-type]
        "p",
        GenParams(max_tokens=4, timeout_sec=0.15),
    )
    assert result.status == GenerationStatus.TIMEOUT


def test_timeout_path_does_not_block_on_worker_shutdown():
    import time

    from workbench.engines.timeout import timed_generate

    worker_sleep_s = 1.5
    deadline_s = 0.15

    class SlowEngine:
        def generate(self, prompt, params):
            time.sleep(worker_sleep_s)
            return _ok_result()

    t0 = time.perf_counter()
    result = timed_generate(
        SlowEngine(),  # type: ignore[arg-type]
        "p",
        GenParams(max_tokens=4, timeout_sec=deadline_s),
    )
    elapsed = time.perf_counter() - t0

    assert result.status == GenerationStatus.TIMEOUT
    assert elapsed < worker_sleep_s * 0.6
    assert elapsed < deadline_s + 0.5


def test_timeout_worker_is_daemon():
    import threading
    import time

    from workbench.engines.timeout import timed_generate

    started = threading.Event()

    class SlowEngine:
        def generate(self, prompt, params):
            started.set()
            time.sleep(2.0)
            return _ok_result()

    before = {t.ident for t in threading.enumerate()}
    result = timed_generate(
        SlowEngine(),  # type: ignore[arg-type]
        "p",
        GenParams(max_tokens=4, timeout_sec=0.1),
    )
    assert result.status == GenerationStatus.TIMEOUT
    assert started.wait(timeout=1.0)

    workers = [
        t
        for t in threading.enumerate()
        if t.ident not in before and t.name == "workbench-generate-timeout"
    ]
    assert workers, "expected timeout worker thread still alive after TIMEOUT"
    assert all(t.daemon for t in workers)


def test_timed_generate_propagates_worker_errors():
    from workbench.engines.timeout import timed_generate

    class BoomEngine:
        def generate(self, prompt, params):
            raise RuntimeError("engine boom")

    try:
        timed_generate(
            BoomEngine(),  # type: ignore[arg-type]
            "p",
            GenParams(max_tokens=4, timeout_sec=1.0),
        )
        raise AssertionError("expected RuntimeError")
    except RuntimeError as e:
        assert "engine boom" in str(e)
