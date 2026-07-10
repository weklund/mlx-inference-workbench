"""Behavioral: comparability is about fair science, not identical experiment names."""

from workbench.comparability import check_comparable
from workbench.models import RunMetadata


def _meta(**overrides) -> RunMetadata:
    base = dict(
        run_id="x",
        experiment_name="e",
        backend="stub",
        model_name="m",
        quantization="q",
        prompt_dataset_path="p",
        prompt_dataset_hash="abc",
        hardware_profile="m5_max_128gb",
        hardware_fingerprint={"chip": "Apple M5 Max"},
        git_sha=None,
        library_versions={},
        random_seed=42,
        schema_version="1.0",
        metrics_schema_version="1.0",
        engine_interface_version="1.0",
        thermal_monitoring="off",
    )
    base.update(overrides)
    return RunMetadata(**base)


def test_same_measurement_context_is_comparable_despite_different_run_ids():
    a, b = _meta(run_id="1", experiment_name="exp-a"), _meta(run_id="2", experiment_name="exp-b")
    r = check_comparable(a, b)
    assert r.comparable is True
    assert r.violations == ()


def test_prompt_corpus_change_is_not_comparable():
    a, b = _meta(prompt_dataset_hash="aaa"), _meta(prompt_dataset_hash="bbb")
    r = check_comparable(a, b)
    assert r.comparable is False
    assert len(r.violations) >= 1


def test_metrics_schema_change_is_not_comparable():
    """Changing how metrics are defined invalidates historical comparisons."""
    a, b = _meta(metrics_schema_version="1.0"), _meta(metrics_schema_version="2.0")
    r = check_comparable(a, b)
    assert r.comparable is False


def test_chip_mismatch_is_not_comparable():
    a = _meta(hardware_fingerprint={"chip": "Apple M5 Max"})
    b = _meta(hardware_fingerprint={"chip": "Apple M1"})
    r = check_comparable(a, b)
    assert r.comparable is False


def test_thermal_monitoring_off_vs_full_is_not_comparable():
    a, b = _meta(thermal_monitoring="off"), _meta(thermal_monitoring="full")
    r = check_comparable(a, b)
    assert r.comparable is False
    assert any("thermal_monitoring" in v for v in r.violations)


def test_thermal_monitoring_full_vs_degraded_is_not_comparable():
    """Different observability classes must not compare silently."""
    a, b = _meta(thermal_monitoring="full"), _meta(thermal_monitoring="degraded")
    r = check_comparable(a, b)
    assert r.comparable is False
    assert any("thermal_monitoring" in v for v in r.violations)


def test_thermal_monitoring_same_mode_is_comparable():
    a, b = _meta(thermal_monitoring="degraded"), _meta(thermal_monitoring="degraded")
    r = check_comparable(a, b)
    assert r.comparable is True
