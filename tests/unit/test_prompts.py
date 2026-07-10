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


def test_system_prompt_preserved_from_jsonl(tmp_path: Path):
    """Optional system_prompt is loaded and retained for generation text."""
    import json

    row = {
        "id": "with_sys",
        "category": "tool_call",
        "system_prompt": "You are a careful coding agent.",
        "prompt": "List three files to inspect first.",
        "expected_tokens_approx": 64,
    }
    p = tmp_path / "sys.jsonl"
    p.write_text(json.dumps(row) + "\n", encoding="utf-8")
    ds = load_dataset(p)
    item = ds.items[0]
    assert item.system_prompt == "You are a careful coding agent."
    assert item.prompt == "List three files to inspect first."
    assert item.generation_text() == (
        "You are a careful coding agent.\n\nList three files to inspect first."
    )
    assert ds.prompts() == [item.generation_text()]


def test_system_alias_and_absent_system_prompt(tmp_path: Path):
    import json

    rows = [
        {
            "id": "alias",
            "category": "c",
            "system": "Legacy system key",
            "prompt": "user text",
        },
        {"id": "none", "category": "c", "prompt": "only user"},
    ]
    p = tmp_path / "alias.jsonl"
    p.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    ds = load_dataset(p)
    assert ds.items[0].system_prompt == "Legacy system key"
    assert ds.items[0].generation_text().startswith("Legacy system key\n\n")
    assert ds.items[1].system_prompt is None
    assert ds.items[1].generation_text() == "only user"
