"""TDD: hardware profile load + bandwidth util % (issue #8)."""

from pathlib import Path

import pytest
import yaml

from workbench.hardware_profile import (
    SpecPair,
    bandwidth_utilization_pct,
    load_hardware_profile,
)


def test_spec_pair_conservative_uses_min():
    p = SpecPair(published=614.0, empirical=400.0, unit="GB/s")
    assert p.conservative() == 400.0


def test_spec_pair_conservative_single_side():
    assert SpecPair(published=614.0, empirical=None, unit="GB/s").conservative() == 614.0
    assert SpecPair(published=None, empirical=250.0, unit="GB/s").conservative() == 250.0
    assert SpecPair(published=None, empirical=None, unit="GB/s").conservative() is None


def test_bandwidth_utilization_pct():
    assert bandwidth_utilization_pct(307.0, 614.0) == pytest.approx(50.0)
    assert bandwidth_utilization_pct(100.0, None) is None
    assert bandwidth_utilization_pct(100.0, 0.0) is None
    assert bandwidth_utilization_pct(-1.0, 100.0) is None


def test_load_hardware_profile_round_trip(tmp_path: Path):
    path = tmp_path / "m5.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "profile": "m5_max_128gb",
                "profile_version": "1.1",
                "chip": "Apple M5 Max",
                "unified_memory_gb": 128,
                "verified_at": "2026-07-10",
                "memory_bandwidth_gbs": {
                    "published": 614.0,
                    "empirical": 300.0,
                    "unit": "GB/s",
                    "source_published": "https://example.test",
                },
                "gpu_cores": {"published": 40, "empirical": 40, "unit": "cores"},
                "gpu_fp32_tflops": {
                    "published": None,
                    "empirical": 35.0,
                    "unit": "TFLOPS",
                },
                "notes": "test",
            }
        ),
        encoding="utf-8",
    )
    hp = load_hardware_profile(path)
    assert hp.profile == "m5_max_128gb"
    assert hp.unified_memory_gb == 128
    assert hp.bandwidth_ceiling_gbs() == 300.0  # min published/empirical
    assert hp.flops_ceiling_tflops() == 35.0
    assert hp.gpu_cores.published == 40
    assert bandwidth_utilization_pct(150.0, hp.bandwidth_ceiling_gbs()) == pytest.approx(50.0)
