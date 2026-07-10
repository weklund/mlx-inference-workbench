"""Deterministic fake engine for CI and smoke tests (no GPU / no model weights)."""

from __future__ import annotations

import hashlib
import time

from workbench.config import ModelConfig
from workbench.engines.base import Engine, EngineLoadError, GenParams
from workbench.models import GenerationResult, GenerationStatus, ThermalReading


class StubEngine(Engine):
    def __init__(self) -> None:
        self._loaded: ModelConfig | None = None
        self._base_latency_s = 0.002

    def name(self) -> str:
        return "stub"

    def load_model(self, config: ModelConfig) -> None:
        if config.backend not in {"stub", "fake"}:
            raise EngineLoadError(f"StubEngine cannot load backend={config.backend!r}")
        self._loaded = config

    def generate(self, prompt: str, params: GenParams) -> GenerationResult:
        self._ensure_loaded()
        n_tokens = min(params.max_tokens, 32)
        # Deterministic content from prompt+seed
        digest = hashlib.sha256(f"{params.seed}:{prompt}".encode()).hexdigest()[:16]
        output = f"stub-{digest}"

        start = time.perf_counter()
        timestamps: list[float] = []
        for i in range(n_tokens):
            # small deterministic per-token delay
            time.sleep(self._base_latency_s)
            timestamps.append(time.perf_counter() - start)

        ttft_ms = timestamps[0] * 1000.0 if timestamps else 0.0
        e2e_ms = timestamps[-1] * 1000.0 if timestamps else 0.0

        return GenerationResult(
            status=GenerationStatus.SUCCESS,
            output_text=output,
            token_timestamps=timestamps,
            ttft_ms=ttft_ms,
            total_tokens=n_tokens,
            memory_peak_bytes=1_000_000,
            thermal_state=ThermalReading(method="stub", thermal_pressure="Nominal"),
            acceptance_rate=None,
            accepted_length_mean=None,
            power_watts=None,
            energy_per_token_joules=None,
            e2e_ms=e2e_ms,
        )

    def _ensure_loaded(self) -> None:
        if self._loaded is None:
            raise EngineLoadError("StubEngine.load_model() not called")
