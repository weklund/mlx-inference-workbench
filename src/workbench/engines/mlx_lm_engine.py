"""mlx-lm backend — isolated plugin; optional until imported via registry."""

from __future__ import annotations

import time
from typing import Any

from workbench.config import ModelConfig
from workbench.engines.base import Engine, EngineLoadError, GenerationError, GenParams
from workbench.models import GenerationResult, GenerationStatus, ThermalReading


class MlxLmEngine(Engine):
    def __init__(self) -> None:
        self._model: Any = None
        self._tokenizer: Any = None
        self._config: ModelConfig | None = None

    def name(self) -> str:
        return "mlx-lm"

    def load_model(self, config: ModelConfig) -> None:
        try:
            from mlx_lm import load
        except ImportError as e:
            raise EngineLoadError("mlx-lm is not installed") from e

        model_id = config.model_id or config.name
        try:
            model, tokenizer = load(model_id)
        except Exception as e:
            raise EngineLoadError(f"Failed to load {model_id}: {e}") from e
        self._model = model
        self._tokenizer = tokenizer
        self._config = config

    def warmup(self, prompts: list[str], n: int, params: GenParams) -> None:
        self._ensure()
        for _ in range(n):
            for p in prompts[:1]:
                self.generate(p, params)

    def generate(self, prompt: str, params: GenParams) -> GenerationResult:
        self._ensure()
        try:
            from mlx_lm import generate as mlx_generate
            from mlx_lm.sample_utils import make_sampler
        except ImportError as e:
            raise GenerationError("mlx-lm generate API unavailable") from e

        sampler = (
            make_sampler(temp=params.temperature)
            if params.temperature > 0
            else make_sampler(temp=0.0)
        )

        # Token-level timestamps: approximate via stream if available, else single e2e split.
        start = time.perf_counter()
        timestamps: list[float] = []
        pieces: list[str] = []

        try:
            # Prefer stream_generate when present
            try:
                from mlx_lm import stream_generate

                for response in stream_generate(
                    self._model,
                    self._tokenizer,
                    prompt=prompt,
                    max_tokens=params.max_tokens,
                    sampler=sampler,
                ):
                    pieces.append(response.text)
                    timestamps.append(time.perf_counter() - start)
                output = "".join(pieces)
            except Exception:
                output = mlx_generate(
                    self._model,
                    self._tokenizer,
                    prompt=prompt,
                    max_tokens=params.max_tokens,
                    sampler=sampler,
                )
                # No per-token stream — synthesize uniform timestamps from e2e
                e2e = time.perf_counter() - start
                # estimate token count
                n_tok = max(1, len(self._tokenizer.encode(output)))
                timestamps = [e2e * (i + 1) / n_tok for i in range(n_tok)]
        except Exception as e:
            raise GenerationError(str(e)) from e

        if not timestamps:
            timestamps = [time.perf_counter() - start]

        total_tokens = len(timestamps)
        ttft_ms = timestamps[0] * 1000.0
        e2e_ms = timestamps[-1] * 1000.0

        return GenerationResult(
            status=GenerationStatus.SUCCESS,
            output_text=output if isinstance(output, str) else str(output),
            token_timestamps=timestamps,
            ttft_ms=ttft_ms,
            total_tokens=total_tokens,
            memory_peak_bytes=self.get_memory_usage_bytes(),
            thermal_state=ThermalReading(
                method="none", thermal_pressure=None, notes="filled_by_orchestrator"
            ),
            e2e_ms=e2e_ms,
        )

    def get_memory_usage_bytes(self) -> int:
        # Best-effort RSS
        try:
            import resource

            return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        except Exception:
            return 0

    def validate_correctness(self, prompt: str, reference: str, tolerance: float = 0.0) -> bool:
        result = self.generate(prompt, GenParams(max_tokens=64, temperature=0.0, seed=42))
        if tolerance <= 0:
            # For real models, exact match is too strict for free-form prompts.
            # Smoke path uses stub. Here: non-empty output is the soft gate unless reference set.
            if not reference:
                return bool(result.output_text.strip())
            return result.output_text.strip() == reference.strip()
        return reference.strip() in result.output_text

    def _ensure(self) -> None:
        if self._model is None or self._tokenizer is None:
            raise EngineLoadError("MlxLmEngine.load_model() not called")
