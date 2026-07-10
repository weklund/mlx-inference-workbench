import hashlib
from pathlib import Path

import pytest

from workbench.prompts import load_dataset, sha256_file


def test_load_dataset_and_hash(tmp_path: Path):
    p = tmp_path / "p.jsonl"
    p.write_text(
        '{"id": "a", "category": "c", "prompt": "hello"}\n',
        encoding="utf-8",
    )
    digest = sha256_file(p)
    cs = tmp_path / "p.sha256"
    cs.write_text(f"{digest}  p.jsonl\n", encoding="utf-8")
    ds = load_dataset(p, checksum_path=cs)
    assert ds.sha256 == digest
    assert len(ds.items) == 1
    assert ds.items[0].prompt == "hello"


def test_hash_mismatch_fails_closed(tmp_path: Path):
    p = tmp_path / "p.jsonl"
    p.write_text('{"id": "a", "category": "c", "prompt": "hello"}\n', encoding="utf-8")
    cs = tmp_path / "p.sha256"
    cs.write_text(f"{'0' * 64}  p.jsonl\n", encoding="utf-8")
    with pytest.raises(ValueError, match="hash mismatch"):
        load_dataset(p, checksum_path=cs)


def test_tamper_detected(tmp_path: Path):
    p = tmp_path / "p.jsonl"
    content = '{"id": "a", "category": "c", "prompt": "hello"}\n'
    p.write_text(content, encoding="utf-8")
    digest = hashlib.sha256(content.encode()).hexdigest()
    p.write_text(content + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="hash mismatch"):
        load_dataset(p, expected_hash=digest)
