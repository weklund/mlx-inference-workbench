"""Dataset v1 integrity — load, categories, fail-closed hash (issue #6)."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import pytest

from workbench.config import load_config
from workbench.prompts import load_dataset, sha256_file

REPO = Path(__file__).resolve().parents[2]
DATASET = REPO / "datasets" / "agentic_coding_v1.jsonl"
CHECKSUM = REPO / "datasets" / "agentic_coding_v1.sha256"
CONFIG = REPO / "configs" / "experiments" / "agentic_coding_v1_stub.yaml"


def test_agentic_coding_v1_loads_with_checksum():
    ds = load_dataset(DATASET, checksum_path=CHECKSUM)
    assert ds.path == DATASET.resolve()
    assert len(ds.items) >= 20
    assert ds.sha256 == sha256_file(DATASET)


def test_agentic_coding_v1_covers_required_categories():
    ds = load_dataset(DATASET, checksum_path=CHECKSUM)
    counts = Counter(i.category for i in ds.items)
    required = {
        "tool_call",
        "multi_turn",
        "code_generation",
        "refactor",
        "explain_failure",
    }
    missing = required - set(counts)
    assert not missing, f"missing categories: {missing}; have {dict(counts)}"
    assert len(counts) >= 4
    ids = [i.id for i in ds.items]
    assert len(ids) == len(set(ids))
    for item in ds.items:
        assert item.prompt.strip()
        assert item.id.startswith("ac_")


def test_agentic_coding_v1_checksum_file_matches_bytes():
    """Smoke: shasum-style sidecar must match file bytes exactly."""
    actual = sha256_file(DATASET)
    line = CHECKSUM.read_text(encoding="utf-8").strip().split()
    assert line[0] == actual
    assert line[1].lstrip("*") == DATASET.name


def test_agentic_coding_v1_tamper_fails_closed(tmp_path: Path):
    """Tampered copy + original manifest entry → hash mismatch before use."""
    import shutil

    tampered = tmp_path / "agentic_coding_v1.jsonl"
    shutil.copy(DATASET, tampered)
    with tampered.open("a", encoding="utf-8") as f:
        f.write("\n")
    # Checksum still names agentic_coding_v1.jsonl with original digest
    cs = tmp_path / "agentic_coding_v1.sha256"
    cs.write_text(CHECKSUM.read_text(encoding="utf-8"), encoding="utf-8")
    with pytest.raises(ValueError, match="hash mismatch"):
        load_dataset(tampered, checksum_path=cs)


def test_experiment_config_points_at_agentic_v1():
    cfg = load_config(CONFIG)
    assert cfg.benchmark.prompt_dataset.endswith("agentic_coding_v1.jsonl")
    assert cfg.benchmark.prompt_checksum.endswith("agentic_coding_v1.sha256")
