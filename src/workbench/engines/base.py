"""Engine ABC — backends plug in here; orchestrator never imports mlx/mtplx directly."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from workbench.config import ModelConfig
from workbench.models import ENGINE_INTERFACE_VERSION, GenerationResult


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


class Engine(ABC):
    ENGINE_INTERFACE_VERSION = ENGINE_INTERFACE_VERSION

    @abstractmethod
    def name(self) -> str: ...

    @abstractmethod
    def load_model(self, config: ModelConfig) -> None: ...

    @abstractmethod
    def warmup(self, prompts: list[str], n: int, params: GenParams) -> None: ...

    @abstractmethod
    def generate(self, prompt: str, params: GenParams) -> GenerationResult: ...

    def supports_speculative(self) -> bool:
        return False

    def get_memory_usage_bytes(self) -> int:
        return 0

    def validate_correctness(self, prompt: str, reference: str, tolerance: float = 0.0) -> bool:
        """Return False on mismatch; never raise for divergence."""
        result = self.generate(
            prompt,
            GenParams(max_tokens=max(16, len(reference.split()) + 8), temperature=0.0, seed=42),
        )
        if not result.output_text:
            return False
        if tolerance <= 0:
            return result.output_text.strip() == reference.strip()
        # soft: reference must be substring or equal
        return (
            reference.strip() in result.output_text
            or result.output_text.strip() == reference.strip()
        )
