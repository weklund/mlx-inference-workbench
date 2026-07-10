"""Comparability gate — single checklist for whether two runs may be compared."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from workbench.models import RunMetadata

if TYPE_CHECKING:
    from workbench.models import RunRecord


# Fields that must match for a fair comparison (experiment factor held constant).
# Quantization/backend/model are intentional experimental factors — compared runs
# must still share prompts, hardware profile, schema, and thermal quality class.
# thermal_monitoring is full | degraded | off; any mismatch (including full vs
# degraded) is a gate violation — different observability is not comparable.
REQUIRED_EQUAL_FIELDS: tuple[str, ...] = (
    "prompt_dataset_hash",
    "hardware_profile",
    "metrics_schema_version",
    "schema_version",
    "engine_interface_version",
    "thermal_monitoring",
)

# HLD: n<2 valid iterations → no distribution; gate rejects comparison.
MIN_VALID_ITERATIONS_FOR_COMPARE = 2


@dataclass(frozen=True)
class GateResult:
    """Outcome of the comparability checklist."""

    comparable: bool
    violations: tuple[str, ...]

    def raise_if_blocked(self) -> None:
        """Raise ComparabilityError when not comparable."""
        if not self.comparable:
            joined = "; ".join(self.violations)
            raise ComparabilityError(f"Comparability gate blocked: {joined}")


class ComparabilityError(Exception):
    """Raised when two runs fail the comparability gate."""


def check_comparable(a: RunMetadata, b: RunMetadata) -> GateResult:
    """Check metadata fields that must match for a fair comparison.

    Args:
        a: First run metadata.
        b: Second run metadata.

    Returns:
        GateResult with violations listed when not comparable.
    """
    violations: list[str] = []

    for field_name in REQUIRED_EQUAL_FIELDS:
        va = getattr(a, field_name)
        vb = getattr(b, field_name)
        if va != vb:
            violations.append(f"{field_name}: {va!r} != {vb!r}")

    # Chip fingerprint: require same chip model when present
    chip_a = a.hardware_fingerprint.get("chip")
    chip_b = b.hardware_fingerprint.get("chip")
    if chip_a and chip_b and chip_a != chip_b:
        violations.append(f"hardware chip: {chip_a!r} != {chip_b!r}")

    return GateResult(comparable=len(violations) == 0, violations=tuple(violations))


def check_runs_comparable(a: RunRecord, b: RunRecord) -> GateResult:
    """Metadata match + enough valid samples on both sides (HLD edge cases)."""
    base = check_comparable(a.metadata, b.metadata)
    violations = list(base.violations)

    for label, run in (("a", a), ("b", b)):
        n = run.metrics.valid_iterations
        if n < MIN_VALID_ITERATIONS_FOR_COMPARE:
            violations.append(
                f"run_{label}_valid_iterations: {n} < {MIN_VALID_ITERATIONS_FOR_COMPARE}"
            )
        if run.metrics.quality_tag == "insufficient_data":
            violations.append(f"run_{label}_quality: insufficient_data")

    # de-dupe while preserving order
    seen: set[str] = set()
    uniq: list[str] = []
    for v in violations:
        if v not in seen:
            seen.add(v)
            uniq.append(v)

    return GateResult(comparable=len(uniq) == 0, violations=tuple(uniq))
