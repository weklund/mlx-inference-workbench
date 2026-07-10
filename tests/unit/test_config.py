"""TDD: ExperimentConfig loading contracts (HLD schema v1)."""

from pathlib import Path

import pytest
import yaml

from workbench.config import load_config


def _write_yaml(path: Path, data: dict) -> Path:
    path.write_text(yaml.dump(data), encoding="utf-8")
    return path


def test_load_config_accepts_schema_1_0(tmp_path: Path):
    path = _write_yaml(
        tmp_path / "ok.yaml",
        {
            "schema_version": "1.0",
            "experiment": {"name": "t", "description": "d"},
            "hardware": {"profile": "m5_max_128gb"},
            "model": {"name": "m", "quantization": "q4", "backend": "stub"},
            "benchmark": {
                "prompt_dataset": "datasets/smoke_v1.jsonl",
                "warmup_iterations": 1,
                "timed_iterations": 2,
            },
        },
    )
    cfg = load_config(path)
    assert cfg.schema_version == "1.0"
    assert cfg.name == "t"
    assert cfg.model.backend == "stub"
    assert cfg.benchmark.timed_iterations == 2
    assert cfg.source_path == path.resolve()


def test_load_config_rejects_unknown_schema(tmp_path: Path):
    path = _write_yaml(
        tmp_path / "bad.yaml",
        {
            "schema_version": "99.0",
            "experiment": {"name": "t"},
            "hardware": {"profile": "x"},
            "model": {"name": "m", "backend": "stub"},
        },
    )
    with pytest.raises(ValueError, match="schema_version"):
        load_config(path)


def test_load_config_rejects_missing_model_backend(tmp_path: Path):
    path = _write_yaml(
        tmp_path / "nobackend.yaml",
        {
            "schema_version": "1.0",
            "experiment": {"name": "t"},
            "hardware": {"profile": "x"},
            "model": {"name": "m", "quantization": "q"},
            # backend missing
        },
    )
    with pytest.raises((KeyError, ValueError)):
        load_config(path)


def test_benchmark_defaults_when_section_sparse(tmp_path: Path):
    path = _write_yaml(
        tmp_path / "defaults.yaml",
        {
            "schema_version": "1.0",
            "experiment": {"name": "t"},
            "hardware": {"profile": "x"},
            "model": {"name": "m", "backend": "stub", "quantization": "n/a"},
            "benchmark": {},
        },
    )
    cfg = load_config(path)
    assert cfg.benchmark.warmup_iterations >= 1
    assert cfg.benchmark.timed_iterations >= 1
    assert cfg.metrics.flag_cov_threshold == 0.05


def _minimal_config(**overrides) -> dict:
    data = {
        "schema_version": "1.0",
        "experiment": {"name": "t"},
        "hardware": {"profile": "x"},
        "model": {"name": "m", "backend": "stub", "quantization": "n/a"},
        "benchmark": {},
    }
    data.update(overrides)
    return data


def test_yaml_boolean_false_disables_flags(tmp_path: Path):
    """Unquoted YAML false must disable settings (not be treated as truthy)."""
    path = _write_yaml(
        tmp_path / "flags.yaml",
        _minimal_config(
            enable_mlflow=False,
            benchmark={
                "monitor_thermal": False,
                "abort_if_throttling": False,
            },
        ),
    )
    cfg = load_config(path)
    assert cfg.enable_mlflow is False
    assert cfg.benchmark.monitor_thermal is False
    assert cfg.benchmark.abort_if_throttling is False


def test_quoted_false_string_is_not_silently_enabled(tmp_path: Path):
    """Quoted 'false' must never become True via bool('false')."""
    # Write raw YAML so values stay strings after parse.
    path = tmp_path / "quoted.yaml"
    path.write_text(
        """
schema_version: "1.0"
experiment: {name: t}
hardware: {profile: x}
model: {name: m, backend: stub, quantization: n/a}
enable_mlflow: "false"
benchmark:
  monitor_thermal: "false"
  abort_if_throttling: "false"
metrics:
  report_trimmed_mean: "false"
  report_std: "false"
reproducibility:
  record_git_commit: "false"
  record_env_versions: "false"
""",
        encoding="utf-8",
    )
    cfg = load_config(path)
    assert cfg.enable_mlflow is False
    assert cfg.benchmark.monitor_thermal is False
    assert cfg.benchmark.abort_if_throttling is False
    assert cfg.metrics.report_trimmed_mean is False
    assert cfg.metrics.report_std is False
    assert cfg.reproducibility.record_git_commit is False
    assert cfg.reproducibility.record_env_versions is False


def test_yaml_boolean_false_disables_metrics_and_repro_flags(tmp_path: Path):
    path = _write_yaml(
        tmp_path / "metrics_repro.yaml",
        _minimal_config(
            metrics={
                "report_trimmed_mean": False,
                "report_std": False,
            },
            reproducibility={
                "record_git_commit": False,
                "record_env_versions": False,
            },
        ),
    )
    cfg = load_config(path)
    assert cfg.metrics.report_trimmed_mean is False
    assert cfg.metrics.report_std is False
    assert cfg.reproducibility.record_git_commit is False
    assert cfg.reproducibility.record_env_versions is False


def test_programmatic_non_boolean_rejected():
    from workbench.config import ExperimentConfig

    with pytest.raises(ValueError, match="enable_mlflow"):
        ExperimentConfig.from_dict(
            _minimal_config(enable_mlflow="maybe"),
        )
    with pytest.raises(ValueError, match="monitor_thermal"):
        ExperimentConfig.from_dict(
            _minimal_config(benchmark={"monitor_thermal": 2}),
        )
    with pytest.raises(ValueError, match="abort_if_throttling"):
        ExperimentConfig.from_dict(
            _minimal_config(benchmark={"abort_if_throttling": None}),
        )
    with pytest.raises(ValueError, match="report_trimmed_mean"):
        ExperimentConfig.from_dict(
            _minimal_config(metrics={"report_trimmed_mean": "maybe"}),
        )
    with pytest.raises(ValueError, match="report_std"):
        ExperimentConfig.from_dict(
            _minimal_config(metrics={"report_std": 2}),
        )
    with pytest.raises(ValueError, match="record_git_commit"):
        ExperimentConfig.from_dict(
            _minimal_config(reproducibility={"record_git_commit": None}),
        )
    with pytest.raises(ValueError, match="record_env_versions"):
        ExperimentConfig.from_dict(
            _minimal_config(reproducibility={"record_env_versions": "yesplease"}),
        )


def test_config_source_has_no_raw_bool_for_yaml_flags():
    """Regression: YAML-sourced flags must go through _parse_bool, not bool(...)."""
    import inspect

    from workbench import config

    source = inspect.getsource(config.ExperimentConfig.from_dict)
    # Allow bool only if someone reintroduces a non-flag cast; flag paths must use _parse_bool.
    assert "bool(metrics_raw.get" not in source
    assert "bool(repro_raw.get" not in source
    assert "bool(bench_raw.get" not in source
    assert "_parse_bool" in source
