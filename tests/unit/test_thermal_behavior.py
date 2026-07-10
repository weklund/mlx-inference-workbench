"""Behavioral: thermal sensors expose usable modes and throttle heuristics."""

from pathlib import Path

from workbench.config import ModelConfig, load_config
from workbench.engines.base import Engine, GenParams
from workbench.models import GenerationResult, GenerationStatus, ThermalReading
from workbench.orchestrator import run_experiment
from workbench.thermal import (
    DegradedThermalSensor,
    OffThermalSensor,
    PowermetricsThermalSensor,
    build_thermal_sensor,
)


def test_off_sensor_never_reports_throttling():
    s = OffThermalSensor()
    assert s.mode() == "off"
    reading = s.read()
    assert s.is_throttling(reading) is False


def test_degraded_sensor_flags_sudden_slowdown_vs_recent_median():
    """
    Property: after a few stable durations, a >2× slowdown is treated as throttle-like.
    (HLD fallback when powermetrics unavailable.)
    """
    s = DegradedThermalSensor()
    reading = s.read()
    s.note_duration(1.0)
    s.note_duration(1.0)
    s.note_duration(1.0)
    assert s.is_throttling(reading) is False
    s.note_duration(3.1)  # > 2× median of prior
    assert s.is_throttling(reading) is True


def test_degraded_sensor_needs_history_before_flagging():
    s = DegradedThermalSensor()
    reading = ThermalReading(method="timing_anomaly")
    s.note_duration(10.0)
    s.note_duration(10.0)
    # only 2 samples — heuristic must not fire yet
    assert s.is_throttling(reading) is False


def test_build_thermal_sensor_off_when_monitor_disabled():
    s = build_thermal_sensor(monitor=False)
    assert s.mode() == "off"


def test_all_built_in_sensors_implement_note_duration():
    """Protocol surface: every sensor accepts duration notes without duck-typing."""
    sensors = [
        OffThermalSensor(),
        DegradedThermalSensor(),
        PowermetricsThermalSensor(),
    ]
    for s in sensors:
        s.note_duration(0.05)  # must not raise


def test_off_and_full_note_duration_are_noop_for_throttling():
    off = OffThermalSensor()
    full = PowermetricsThermalSensor()
    reading = off.read()
    for _ in range(5):
        off.note_duration(99.0)
        full.note_duration(99.0)
    assert off.is_throttling(reading) is False
    # full sensor throttling is pressure-based, not duration-based
    assert (
        full.is_throttling(ThermalReading(method="powermetrics", thermal_pressure="Nominal"))
        is False
    )


class _OkEngine(Engine):
    def name(self) -> str:
        return "ok"

    def load_model(self, config: ModelConfig) -> None:
        return None

    def generate(self, prompt: str, params: GenParams) -> GenerationResult:
        return GenerationResult(
            status=GenerationStatus.SUCCESS,
            output_text="ok",
            token_timestamps=[0.01, 0.02],
            ttft_ms=10.0,
            total_tokens=2,
            memory_peak_bytes=1,
            thermal_state=ThermalReading(method="off"),
            e2e_ms=40.0,
        )


class _RecordingThermal(OffThermalSensor):
    def __init__(self) -> None:
        self.durations: list[float] = []

    def note_duration(self, seconds: float) -> None:
        self.durations.append(seconds)


def test_orchestrator_always_notes_duration_without_hasattr(tmp_path: Path):
    """Hot path: duration is always pushed to the sensor protocol (no hasattr)."""
    repo = Path(__file__).resolve().parents[2]
    cfg = load_config(repo / "configs/experiments/smoke_minimal.yaml")
    cfg.results_dir = str(tmp_path / "results")
    cfg.enable_mlflow = False
    cfg.benchmark.warmup_iterations = 0
    cfg.benchmark.timed_iterations = 2
    cfg.benchmark.cooldown_between_runs_sec = 0
    cfg.benchmark.monitor_thermal = False

    thermal = _RecordingThermal()
    run_experiment(cfg, repo_root=repo, engine=_OkEngine(), thermal=thermal)

    assert thermal.durations == [0.04, 0.04]


def test_orchestrator_source_has_no_hasattr_for_note_duration():
    import inspect

    from workbench import orchestrator

    source = inspect.getsource(orchestrator)
    assert 'hasattr(thermal, "note_duration")' not in source
    assert "note_duration" in source
