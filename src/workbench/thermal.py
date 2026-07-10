"""Thermal sensing — protocol + implementations (full / degraded / off)."""

from __future__ import annotations

import subprocess
import time
from typing import Protocol

from workbench.models import ThermalReading


class ThermalSensor(Protocol):
    """Thermal / power observability plug-in used by the orchestrator."""

    def mode(self) -> str:
        """Return monitoring quality class: full, degraded, or off."""
        ...

    def read(self) -> ThermalReading:
        """Sample current thermal/power state (best-effort)."""
        ...

    def is_throttling(self, reading: ThermalReading) -> bool:
        """Return True if ``reading`` indicates thermal throttling."""
        ...

    def note_duration(self, seconds: float) -> None:
        """Record iteration wall time for anomaly heuristics. Default: no-op."""
        ...


class OffThermalSensor:
    """No thermal monitoring (explicit opt-out)."""

    def mode(self) -> str:
        """Return ``off``."""
        return "off"

    def read(self) -> ThermalReading:
        """Return an empty reading tagged as disabled."""
        return ThermalReading(method="off", notes="thermal monitoring disabled")

    def is_throttling(self, reading: ThermalReading) -> bool:
        """Never treat runs as throttling when monitoring is off."""
        return False

    def note_duration(self, seconds: float) -> None:
        """No-op duration sink."""


class DegradedThermalSensor:
    """Timing-anomaly based; orchestrator calls note_duration after each iter."""

    def __init__(self) -> None:
        self._durations: list[float] = []

    def mode(self) -> str:
        """Return ``degraded``."""
        return "degraded"

    def read(self) -> ThermalReading:
        """Return a placeholder reading (no powermetrics)."""
        return ThermalReading(
            method="timing_anomaly",
            thermal_pressure=None,
            notes="powermetrics unavailable; using duration heuristics",
        )

    def note_duration(self, seconds: float) -> None:
        """Append iteration duration for median-based anomaly detection."""
        self._durations.append(seconds)

    def is_throttling(self, reading: ThermalReading) -> bool:
        """Flag sudden >2x slowdown vs recent median duration."""
        if len(self._durations) < 3:
            return False
        median = sorted(self._durations)[len(self._durations) // 2]
        last = self._durations[-1]
        return median > 0 and last > 2.0 * median


class PowermetricsThermalSensor:
    """Sample via macOS ``powermetrics`` (may require privileges)."""

    def __init__(self, sample_sec: float = 0.5) -> None:
        self._sample_sec = sample_sec

    def mode(self) -> str:
        """Return ``full``."""
        return "full"

    def read(self) -> ThermalReading:
        """Invoke powermetrics once and parse thermal pressure / power."""
        # Lightweight sample; may require sudo on some macOS versions
        try:
            proc = subprocess.run(
                [
                    "powermetrics",
                    "--samplers",
                    "cpu_power,gpu_power",
                    "-n",
                    "1",
                    "-i",
                    str(int(self._sample_sec * 1000)),
                ],
                capture_output=True,
                text=True,
                timeout=15,
                check=False,
            )
            if proc.returncode != 0:
                raise RuntimeError(proc.stderr[:200] or "powermetrics failed")
            cpu_mw = _parse_power_mw(proc.stdout, "CPU Power")
            gpu_mw = _parse_power_mw(proc.stdout, "GPU Power")
            combined = None
            if cpu_mw is not None or gpu_mw is not None:
                combined = (cpu_mw or 0) + (gpu_mw or 0)
            pressure = "Nominal"
            if "Thermal pressure" in proc.stdout:
                for line in proc.stdout.splitlines():
                    if "Thermal pressure" in line:
                        pressure = line.split(":")[-1].strip()
            return ThermalReading(
                method="powermetrics",
                thermal_pressure=pressure,
                cpu_power_mw=cpu_mw,
                gpu_power_mw=gpu_mw,
                combined_power_mw=combined,
            )
        except (FileNotFoundError, subprocess.TimeoutExpired, RuntimeError) as e:
            return ThermalReading(method="powermetrics_failed", notes=str(e))

    def is_throttling(self, reading: ThermalReading) -> bool:
        """True when thermal pressure is present and not nominal/normal."""
        if not reading.thermal_pressure:
            return False
        return reading.thermal_pressure.lower() not in {"nominal", "normal", "0"}

    def note_duration(self, seconds: float) -> None:
        """No-op; full mode uses powermetrics, not duration heuristics."""


def _parse_power_mw(text: str, label: str) -> float | None:
    for line in text.splitlines():
        if label in line and "mW" in line:
            # e.g. "CPU Power: 12345 mW"
            parts = line.replace(":", " ").split()
            for i, p in enumerate(parts):
                if p.replace(".", "", 1).isdigit() and i + 1 < len(parts) and "mW" in parts[i + 1]:
                    return float(p)
    return None


def build_thermal_sensor(monitor: bool) -> ThermalSensor:
    """Choose off / full / degraded based on config and probe success.

    Args:
        monitor: When False, returns OffThermalSensor. When True, probes
            powermetrics and falls back to DegradedThermalSensor.
    """
    if not monitor:
        return OffThermalSensor()
    # Probe once
    sensor = PowermetricsThermalSensor()
    reading = sensor.read()
    if reading.method == "powermetrics":
        return sensor
    return DegradedThermalSensor()


def cooldown(seconds: float) -> None:
    """Sleep between runs when ``seconds`` is positive."""
    if seconds > 0:
        time.sleep(seconds)
