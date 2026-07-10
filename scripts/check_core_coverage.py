#!/usr/bin/env python3
"""Fail CI if core workbench modules drop below starting coverage floors.

Reads coverage.json from pytest-cov. Floors are intentionally below the overall
80% gate so hardware-heavy modules (powermetrics, full mlx-lm paths) do not
block the harness while contracts stabilize. Raise floors toward 90% over time.

Usage:
    uv run pytest tests/unit -m "not gpu and not slow" \\
      --cov=workbench --cov-report=json:coverage.json -q
    uv run python scripts/check_core_coverage.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Starting per-module floors (%). Keys match suffixes of coverage file paths.
CORE_MODULE_FLOORS: dict[str, float] = {
    "workbench/orchestrator.py": 80.0,
    "workbench/metrics.py": 70.0,
    "workbench/comparability.py": 80.0,
    "workbench/statistics_compare.py": 70.0,
    "workbench/config.py": 80.0,
    "workbench/prompts.py": 70.0,
    "workbench/storage/run_store.py": 70.0,
    "workbench/engines/base.py": 70.0,
    "workbench/engines/timeout.py": 70.0,
    "workbench/engines/registry.py": 70.0,
    "workbench/engines/stub.py": 70.0,
    # Live hardware / heavy MLX paths — lower starting floors (branch-aware %)
    "workbench/thermal.py": 45.0,  # powermetrics subprocess path needs live hardware
    "workbench/engines/mlx_lm_engine.py": 60.0,
    "workbench/hardware.py": 70.0,
}

DEFAULT_JSON = Path("coverage.json")


def _pct(file_data: dict) -> float:
    summary = file_data.get("summary") or {}
    # Prefer branch-aware percent when present
    if "percent_covered" in summary:
        return float(summary["percent_covered"])
    covered = float(summary.get("covered_lines", 0))
    n = float(summary.get("num_statements", 0))
    return (100.0 * covered / n) if n else 100.0


def _match_floor(path: str) -> tuple[str, float] | None:
    normalized = path.replace("\\", "/")
    for suffix, floor in CORE_MODULE_FLOORS.items():
        if normalized.endswith(suffix) or f"/{suffix}" in normalized:
            return suffix, floor
    return None


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    report_path = Path(args[0]) if args else DEFAULT_JSON
    if not report_path.is_file():
        print(f"error: coverage report not found: {report_path}", file=sys.stderr)
        print(
            "run pytest with --cov=workbench --cov-report=json:coverage.json first", file=sys.stderr
        )
        return 2

    data = json.loads(report_path.read_text(encoding="utf-8"))
    files = data.get("files") or {}
    seen: set[str] = set()
    failures: list[str] = []
    lines: list[str] = []

    for path, file_data in sorted(files.items()):
        hit = _match_floor(path)
        if hit is None:
            continue
        suffix, floor = hit
        seen.add(suffix)
        pct = _pct(file_data)
        status = "ok" if pct + 1e-9 >= floor else "FAIL"
        lines.append(f"  {status:4}  {pct:5.1f}%  (floor {floor:4.0f}%)  {suffix}")
        if status == "FAIL":
            failures.append(f"{suffix}: {pct:.1f}% < {floor:.0f}%")

    missing = sorted(set(CORE_MODULE_FLOORS) - seen)
    for suffix in missing:
        failures.append(f"{suffix}: missing from coverage report")
        lines.append(f"  FAIL  missing coverage data for {suffix}")

    print("Core module coverage floors (issue #23)")
    print("\n".join(lines) if lines else "  (no core modules found in report)")
    if failures:
        print("\nCoverage floor failures:", file=sys.stderr)
        for f in failures:
            print(f"  - {f}", file=sys.stderr)
        return 1
    print("\nAll core module floors met.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
