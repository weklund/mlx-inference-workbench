"""Unit tests for ceiling probes — mock MLX / subprocess (no Metal required)."""

from types import SimpleNamespace

from workbench.ceilings import (
    detect_chip,
    measure_matmul_tflops,
    measure_memory_bandwidth_gbs,
)


class _FakeMx:
    """Minimal stand-in so probes can run offline."""

    float32 = "float32"

    class random:
        @staticmethod
        def normal(shape, dtype=None):
            return SimpleNamespace(shape=shape, dtype=dtype)

    @staticmethod
    def array(val, dtype=None):
        return SimpleNamespace(val=val, dtype=dtype)

    @staticmethod
    def eval(*_a, **_k):
        return None

    @staticmethod
    def synchronize():
        return None

    def __matmul__(self, other):  # pragma: no cover - not used on type
        return self


def test_measure_memory_bandwidth_gbs_mocked(monkeypatch):
    import sys
    import time

    # Make triad ops return dummy arrays
    class Arr:
        def __add__(self, other):
            return self

        def __mul__(self, other):
            return self

        def __rmul__(self, other):
            return self

    class Fake:
        float32 = "f32"

        class random:
            @staticmethod
            def normal(shape, dtype=None):
                return Arr()

        @staticmethod
        def array(val, dtype=None):
            return Arr()

        @staticmethod
        def eval(*_a, **_k):
            return None

        @staticmethod
        def synchronize():
            return None

        @staticmethod
        def compile(fn):
            return fn

    monkeypatch.setitem(sys.modules, "mlx.core", Fake)
    monkeypatch.setitem(sys.modules, "mlx", SimpleNamespace(core=Fake))

    # Force a known dt so GB/s is finite and positive
    times = iter([0.0, 0.01] * 50)

    def fake_perf():
        return next(times)

    monkeypatch.setattr(time, "perf_counter", fake_perf)

    result = measure_memory_bandwidth_gbs(
        n_elements=1_000_000,
        iterations=3,
        warmup=1,
    )
    assert result.method == "mlx_compiled_stream_triad"
    assert result.gbs > 0
    assert result.buffer_bytes == 1_000_000 * 4


def test_measure_matmul_tflops_mocked(monkeypatch):
    import sys
    import time

    class Arr:
        def __matmul__(self, other):
            return self

    class Fake:
        float32 = "f32"

        class random:
            @staticmethod
            def normal(shape, dtype=None):
                return Arr()

        @staticmethod
        def eval(*_a, **_k):
            return None

        @staticmethod
        def synchronize():
            return None

    monkeypatch.setitem(sys.modules, "mlx.core", Fake)
    monkeypatch.setitem(sys.modules, "mlx", SimpleNamespace(core=Fake))

    times = iter([0.0, 0.05] * 40)

    def fake_perf():
        return next(times)

    monkeypatch.setattr(time, "perf_counter", fake_perf)

    result = measure_matmul_tflops(size=64, iterations=2, warmup=1)
    assert result.method == "mlx_matmul_fp32"
    assert result.m == result.n == result.k == 64
    assert result.tflops > 0


def test_detect_chip_uses_sysctl(monkeypatch):
    from workbench import ceilings as c

    def fake_sysctl(key: str):
        return {
            "machdep.cpu.brand_string": "Apple M5 Max",
            "hw.memsize": "137438953472",
            "hw.ncpu": "18",
            "hw.model": "Mac17,7",
        }.get(key)

    monkeypatch.setattr(c, "_sysctl", fake_sysctl)
    monkeypatch.setattr(c, "_gpu_cores_system_profiler", lambda: 40)

    chip = detect_chip()
    assert chip.chip == "Apple M5 Max"
    assert chip.gpu_cores == 40
    assert chip.cpu_cores == 18
    assert chip.memsize_bytes == 137438953472
