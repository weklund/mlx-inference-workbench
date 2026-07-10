"""Engine ABC — backends plug in here; orchestrator never imports mlx/mtplx directly."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator
from dataclasses import dataclass

from workbench.config import ModelConfig
from workbench.models import ENGINE_INTERFACE_VERSION, GenerationResult, GenerationStatus


class EngineLoadError(Exception):
    pass


class GenerationError(Exception):
    pass


@dataclass(frozen=True)
class GenParams:
    max_tokens: int
    temperature: float = 0.0
    seed: int = 42
    timeout_sec: float | None = None


def iter_warmup_prompts(prompts: list[str], n: int) -> Iterator[str]:
    """Yield prompts for n warmup rounds (first prompt only, or a placeholder)."""
    seed = prompts[:1] or ["warmup"]
    for _ in range(n):
        yield from seed


class Engine(ABC):
    """Backend plugin. Generation only — wall-clock timeout is orchestrator policy."""

    ENGINE_INTERFACE_VERSION = ENGINE_INTERFACE_VERSION

    @abstractmethod
    def name(self) -> str: ...

    @abstractmethod
    def load_model(self, config: ModelConfig) -> None: ...

    @abstractmethod
    def generate(self, prompt: str, params: GenParams) -> GenerationResult: ...

    def supports_speculative(self) -> bool:
        return False

    def get_memory_usage_bytes(self) -> int:
        return 0

    def score_correctness(
        self,
        result: GenerationResult,
        reference: str = "",
        tolerance: float = 0.0,
    ) -> bool:
        """Score an already-produced result. Never generates.

        Requires a non-empty reference. Blank reference is not a soft pass —
        callers (orchestrator) must skip the gate when no reference exists.
        ERROR/TIMEOUT and empty output fail closed.
        """
        ref = (reference or "").strip()
        if not ref:
            return False
        if result.status in (GenerationStatus.ERROR, GenerationStatus.TIMEOUT):
            return False
        if not result.output_text:
            return False
        if tolerance <= 0:
            return result.output_text.strip() == ref
        return ref in result.output_text or result.output_text.strip() == ref

    def validate_correctness(self, prompt: str, reference: str, tolerance: float = 0.0) -> bool:
        """Standalone untimed generate-then-score (not used by run_experiment)."""
        ref = (reference or "").strip()
        result = self.generate(
            prompt,
            GenParams(
                max_tokens=max(16, len(ref.split()) + 8 if ref else 16),
                temperature=0.0,
                seed=42,
            ),
        )
        return self.score_correctness(result, reference, tolerance)
