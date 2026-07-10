"""Fail-closed correctness validation on Engine base implementation."""

from workbench.config import ModelConfig
from workbench.engines.base import Engine, GenParams
from workbench.models import GenerationResult, GenerationStatus, ThermalReading


class _ScriptedEngine(Engine):
    """Minimal engine: scripted generate results for base.validate_correctness."""

    def __init__(self, result: GenerationResult) -> None:
        self._result = result

    def name(self) -> str:
        return "scripted"

    def load_model(self, config: ModelConfig) -> None:
        return None

    def generate(self, prompt: str, params: GenParams) -> GenerationResult:
        return self._result


def _result(
    *,
    status: GenerationStatus = GenerationStatus.SUCCESS,
    output: str = "hello world",
) -> GenerationResult:
    tokens = max(1, len(output.split())) if status == GenerationStatus.SUCCESS else 0
    timestamps = (
        [0.01 * (i + 1) for i in range(tokens)] if status == GenerationStatus.SUCCESS else []
    )
    return GenerationResult(
        status=status,
        output_text=output,
        token_timestamps=timestamps,
        ttft_ms=10.0,
        total_tokens=tokens,
        memory_peak_bytes=0,
        thermal_state=ThermalReading(method="test"),
        e2e_ms=20.0,
        error_message="boom" if status != GenerationStatus.SUCCESS else None,
    )


def test_blank_reference_fails_closed():
    eng = _ScriptedEngine(_result(output="anything"))
    assert eng.validate_correctness("p", reference="") is False
    assert eng.validate_correctness("p", reference="   ") is False
    assert eng.validate_correctness("p", reference="", tolerance=0.5) is False


def test_error_status_fails_before_output_match():
    # Even if output text would match the reference, ERROR must fail closed.
    eng = _ScriptedEngine(_result(status=GenerationStatus.ERROR, output="exact"))
    assert eng.validate_correctness("p", reference="exact") is False


def test_timeout_status_fails_before_output_match():
    eng = _ScriptedEngine(_result(status=GenerationStatus.TIMEOUT, output="exact"))
    assert eng.validate_correctness("p", reference="exact") is False


def test_exact_match_zero_tolerance():
    eng = _ScriptedEngine(_result(output="  hello  "))
    assert eng.validate_correctness("p", reference="hello", tolerance=0.0) is True
    assert eng.validate_correctness("p", reference="hello!", tolerance=0.0) is False


def test_tolerant_requires_nonempty_ref_contained_or_equal():
    eng = _ScriptedEngine(_result(output="prefix hello world suffix"))
    assert eng.validate_correctness("p", reference="hello world", tolerance=0.1) is True
    assert eng.validate_correctness("p", reference="missing", tolerance=0.1) is False
    # Blank must not pass via empty-substring semantics ("" in text is True in Python).
    assert eng.validate_correctness("p", reference="", tolerance=0.1) is False
