"""Behavioral: thermal sensors expose usable modes and throttle heuristics."""

from workbench.models import ThermalReading
from workbench.thermal import DegradedThermalSensor, OffThermalSensor, build_thermal_sensor


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
