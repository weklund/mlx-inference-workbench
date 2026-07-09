"""Prompt dataset loading and integrity checks."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class PromptItem:
    id: str
    category: str
    prompt: str
    expected_tokens_approx: int | None = None


@dataclass(frozen=True)
class PromptDataset:
    path: Path
    items: tuple[PromptItem, ...]
    sha256: str

    def prompts(self) -> list[str]:
        return [p.prompt for p in self.items]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def load_checksum_file(checksum_path: Path) -> dict[str, str]:
    """Parse `hash  filename` or `hash *filename` lines (sha256sum style)."""
    mapping: dict[str, str] = {}
    text = checksum_path.read_text(encoding="utf-8")
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 2:
            raise ValueError(f"Invalid checksum line: {line!r}")
        digest, name = parts[0], parts[1]
        name = name.lstrip("*")
        mapping[name] = digest
    return mapping


def load_dataset(
    path: Path,
    *,
    expected_hash: str | None = None,
    checksum_path: Path | None = None,
) -> PromptDataset:
    """Load JSONL prompts; fail closed on hash mismatch."""
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Prompt dataset not found: {path}")

    actual = sha256_file(path)

    if checksum_path is not None:
        checksum_path = checksum_path.resolve()
        mapping = load_checksum_file(checksum_path)
        # match by basename
        key = path.name
        if key not in mapping:
            # try relative as stored
            raise ValueError(f"Checksum file has no entry for {key}")
        expected_hash = mapping[key]

    if expected_hash is not None and actual != expected_hash.lower():
        raise ValueError(
            f"Prompt dataset hash mismatch for {path.name}: expected {expected_hash}, got {actual}"
        )

    items: list[PromptItem] = []
    with path.open(encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSONL at line {i}: {e}") from e
            items.append(
                PromptItem(
                    id=str(obj["id"]),
                    category=str(obj.get("category", "unknown")),
                    prompt=str(obj["prompt"]),
                    expected_tokens_approx=obj.get("expected_tokens_approx"),
                )
            )

    if not items:
        raise ValueError(f"Prompt dataset is empty: {path}")

    return PromptDataset(path=path, items=tuple(items), sha256=actual)
