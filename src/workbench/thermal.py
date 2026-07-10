"""Thermal sensing — protocol + implementations (full / degraded / off)."""

from __future__ import annotations

import subprocess
import time
from typing import Protocol

from workbench.models import ThermalReading


class ThermalSensor(Protocol):
    def mode(self) -> str:
        """full | degraded | off"""
        ...

    def read(self) -> ThermalReading: ...

    def is_throttling(self, reading: ThermalReading) -> bool: ...

    def note_duration(self, seconds: float) -> None:
        """Record iteration wall time for anomaly heuristics. Default: no-op."""
        ...


class OffThermalSensor:
    def mode(self) -> str:
        return "off"

    def read(self) -> ThermalReading:
        return ThermalReading(method="off", notes="thermal monitoring disabled")

    def is_throttling(self, reading: ThermalReading) -> bool:
        return False

    def note_duration(self, seconds: float) -> None:
        return None


class DegradedThermalSensor:
    """Timing-anomaly based; orchestrator calls note_duration after each iter."""

    def __init__(self) -> None:
        self._durations: list[float] = []

    def mode(self) -> str:
        return "degraded"

    def read(self) -> ThermalReading:
        return ThermalReading(
            method="timing_anomaly",
            thermal_pressure=None,
            notes="powermetrics unavailable; using duration heuristics",
        )

    def note_duration(self, seconds: float) -> None:
        self._durations.append(seconds)

    def is_throttling(self, reading: ThermalReading) -> bool:
        if len(self._durations) < 3:
            return False
        median = sorted(self._durations)[len(self._durations) // 2]
        last = self._durations[-1]
        return median > 0 and last > 2.0 * median


class PowermetricsThermalSensor:
    def __init__(self, sample_sec: float = 0.5) -> None:
        self._sample_sec = sample_sec

    def mode(self) -> str:
        return "full"

    def read(self) -> ThermalReading:
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
        if not reading.thermal_pressure:
            return False
        return reading.thermal_pressure.lower() not in {"nominal", "normal", "0"}

    def note_duration(self, seconds: float) -> None:
        return None


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
    if not monitor:
        return OffThermalSensor()
    # Probe once
    sensor = PowermetricsThermalSensor()
    reading = sensor.read()
    if reading.method == "powermetrics":
        return sensor
    return DegradedThermalSensor()


def cooldown(seconds: float) -> None:
    if seconds > 0:
        time.sleep(seconds)
