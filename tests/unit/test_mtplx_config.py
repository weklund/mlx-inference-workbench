"""Config parsing for nested model.mtplx options."""

from __future__ import annotations

import pytest

from workbench.config import ExperimentConfig


def _base(**model_extra):
    model = {
        "name": "qwen",
        "quantization": "4bit",
        "backend": "mtplx",
        "model_id": "mlx-community/example",
        "max_tokens": 64,
    }
    model.update(model_extra)
    return {
        "schema_version": "1.0",
        "experiment": {"name": "t", "description": ""},
        "hardware": {"profile": "m5_max_128gb"},
        "model": model,
        "benchmark": {},
        "metrics": {},
        "reproducibility": {},
    }


def test_mtplx_options_default_depth():
    cfg = ExperimentConfig.from_dict(_base(mtplx={}))
    assert cfg.model.mtplx is not None
    assert cfg.model.mtplx.speculative_depth == 4


def test_mtplx_options_custom_depth():
    cfg = ExperimentConfig.from_dict(_base(mtplx={"speculative_depth": 6}))
    assert cfg.model.mtplx is not None
    assert cfg.model.mtplx.speculative_depth == 6


def test_mtplx_block_omitted_is_none():
    cfg = ExperimentConfig.from_dict(_base())
    assert cfg.model.mtplx is None


def test_mtplx_block_rejected_for_other_backend():
    data = _base(mtplx={"speculative_depth": 3})
    data["model"]["backend"] = "mlx-lm"
    with pytest.raises(ValueError, match=r"only valid when model\.backend is 'mtplx'"):
        ExperimentConfig.from_dict(data)


def test_mtplx_depth_must_be_positive():
    with pytest.raises(ValueError, match="speculative_depth"):
        ExperimentConfig.from_dict(_base(mtplx={"speculative_depth": 0}))


def test_mtplx_unknown_key_rejected():
    with pytest.raises(ValueError, match="unknown keys"):
        ExperimentConfig.from_dict(_base(mtplx={"speculative_depth": 4, "nope": 1}))
