"""Hardware profile YAML — published vs empirical ceilings for roofline (#8)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class SpecPair:
    """Published (vendor) vs empirical (measured) value for one ceiling."""

    published: float | None
    empirical: float | None
    unit: str
    source_published: str | None = None
    source_empirical: str | None = None
    notes: str | None = None

    def conservative(self) -> float | None:
        """Roofline ceiling: lower of published and empirical when both exist.

        If only one side is known, use that. Prefer not inventing numbers.
        """
        vals = [v for v in (self.published, self.empirical) if v is not None]
        if not vals:
            return None
        return float(min(vals))

    def to_dict(self) -> dict[str, Any]:
        return {
            "published": self.published,
            "empirical": self.empirical,
            "unit": self.unit,
            "conservative": self.conservative(),
            "source_published": self.source_published,
            "source_empirical": self.source_empirical,
            "notes": self.notes,
        }


@dataclass(frozen=True)
class HardwareProfile:
    profile: str
    chip: str
    unified_memory_gb: float
    memory_bandwidth_gbs: SpecPair
    gpu_cores: SpecPair
    gpu_fp32_tflops: SpecPair
    notes: str | None = None
    verified_at: str | None = None  # ISO date when empirical was last run
    profile_version: str = "1.0"

    def bandwidth_ceiling_gbs(self) -> float | None:
        return self.memory_bandwidth_gbs.conservative()

    def flops_ceiling_tflops(self) -> float | None:
        return self.gpu_fp32_tflops.conservative()


def bandwidth_utilization_pct(
    achieved_gbs: float,
    ceiling_gbs: float | None,
) -> float | None:
    """Achieved / ceiling as percent. None if ceiling unknown or non-positive."""
    if ceiling_gbs is None or ceiling_gbs <= 0 or achieved_gbs < 0:
        return None
    return 100.0 * float(achieved_gbs) / float(ceiling_gbs)


def _pair_from_mapping(raw: Any, *, unit: str, default: SpecPair | None = None) -> SpecPair:
    if raw is None:
        return default or SpecPair(published=None, empirical=None, unit=unit)
    if not isinstance(raw, dict):
        raise ValueError(f"spec pair must be a mapping, got {type(raw).__name__}")

    def _num(key: str) -> float | None:
        v = raw.get(key)
        if v is None:
            return None
        return float(v)

    # Prefer Metal STREAM (kernel-grade) as empirical when present; else legacy keys.
    empirical = _num("empirical_metal_stream") or _num("empirical") or _num("empirical_mlx")

    return SpecPair(
        published=_num("published"),
        empirical=empirical,
        unit=str(raw.get("unit", unit)),
        source_published=raw.get("source_published"),
        source_empirical=raw.get("source_empirical"),
        notes=raw.get("notes"),
    )


def load_hardware_profile(path: Path) -> HardwareProfile:
    path = path.resolve()
    with path.open(encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Hardware profile root must be a mapping: {path}")

    return HardwareProfile(
        profile=str(data.get("profile", path.stem)),
        chip=str(data.get("chip", "unknown")),
        unified_memory_gb=float(data.get("unified_memory_gb", 0)),
        memory_bandwidth_gbs=_pair_from_mapping(data.get("memory_bandwidth_gbs"), unit="GB/s"),
        gpu_cores=_pair_from_mapping(data.get("gpu_cores"), unit="cores"),
        gpu_fp32_tflops=_pair_from_mapping(data.get("gpu_fp32_tflops"), unit="TFLOPS"),
        notes=data.get("notes"),
        verified_at=data.get("verified_at"),
        profile_version=str(data.get("profile_version", "1.0")),
    )
