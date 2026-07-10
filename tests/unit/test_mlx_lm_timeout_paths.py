"""mlx-lm warmup/validate must use the shared generate timeout wrapper."""

from workbench.config import ModelConfig
from workbench.engines.base import GenParams
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


def test_warmup_invokes_generate_with_timeout(monkeypatch):
    eng = MlxLmEngine()
    eng._model = object()
    eng._tokenizer = object()
    eng._config = ModelConfig(name="m", quantization="q", backend="mlx-lm")

    calls: list[tuple[str, float | None]] = []

    def fake_timeout(engine, prompt, params):
        calls.append((prompt, params.timeout_sec))
        return _ok_result()

    monkeypatch.setattr("workbench.engines.mlx_lm_engine.generate_with_timeout", fake_timeout)
    params = GenParams(max_tokens=8, seed=1, timeout_sec=0.5)
    eng.warmup(["prompt-a", "prompt-b"], n=2, params=params)

    assert len(calls) == 2  # n=2, first prompt only each round
    assert all(p == "prompt-a" for p, _ in calls)
    assert all(t == 0.5 for _, t in calls)


def test_validate_correctness_invokes_generate_with_timeout(monkeypatch):
    eng = MlxLmEngine()
    eng._model = object()
    eng._tokenizer = object()

    seen: list[GenParams] = []

    def fake_timeout(engine, prompt, params):
        seen.append(params)
        return _ok_result("hello world")

    monkeypatch.setattr("workbench.engines.mlx_lm_engine.generate_with_timeout", fake_timeout)
    assert eng.validate_correctness("p", reference="", tolerance=0.0) is True
    assert len(seen) == 1
    assert seen[0].max_tokens == 64
    assert seen[0].temperature == 0.0
    assert seen[0].seed == 42
    assert seen[0].timeout_sec is not None and seen[0].timeout_sec > 0


def test_validate_correctness_timeout_fails_closed(monkeypatch):
    eng = MlxLmEngine()
    eng._model = object()
    eng._tokenizer = object()

    def fake_timeout(engine, prompt, params):
        return GenerationResult(
            status=GenerationStatus.TIMEOUT,
            output_text="",
            token_timestamps=[],
            ttft_ms=0.0,
            total_tokens=0,
            memory_peak_bytes=0,
            thermal_state=ThermalReading(method="timeout"),
            error_message="timeout",
        )

    monkeypatch.setattr("workbench.engines.mlx_lm_engine.generate_with_timeout", fake_timeout)
    assert eng.validate_correctness("p", reference="x") is False


def test_generate_with_timeout_returns_timeout_status():
    import time

    from workbench.engines.timeout import generate_with_timeout

    class SlowEngine:
        def generate(self, prompt, params):
            time.sleep(1.0)
            return _ok_result()

    result = generate_with_timeout(
        SlowEngine(),  # type: ignore[arg-type]
        "p",
        GenParams(max_tokens=4, timeout_sec=0.15),
    )
    assert result.status == GenerationStatus.TIMEOUT


def test_timeout_path_does_not_block_on_worker_shutdown():
    """
    Regression: context-managed ThreadPoolExecutor waits for workers on exit,
    so a hung generate() would block long after the deadline. Timeout must
    return near the deadline, not after the full worker sleep.
    """
    import time

    from workbench.engines.timeout import generate_with_timeout

    worker_sleep_s = 1.5
    deadline_s = 0.15

    class SlowEngine:
        def generate(self, prompt, params):
            time.sleep(worker_sleep_s)
            return _ok_result()

    t0 = time.perf_counter()
    result = generate_with_timeout(
        SlowEngine(),  # type: ignore[arg-type]
        "p",
        GenParams(max_tokens=4, timeout_sec=deadline_s),
    )
    elapsed = time.perf_counter() - t0

    assert result.status == GenerationStatus.TIMEOUT
    # Allow some scheduling slack, but must not wait for the full worker sleep.
    assert elapsed < worker_sleep_s * 0.6
    assert elapsed < deadline_s + 0.5
