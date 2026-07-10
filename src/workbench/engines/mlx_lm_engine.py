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

    def generate(self, prompt: str, params: GenParams) -> GenerationResult:
        self._ensure()
        try:
            from mlx_lm import generate as mlx_generate
            from mlx_lm.sample_utils import make_sampler
        except ImportError as e:
            raise GenerationError("mlx-lm generate API unavailable") from e

        self._seed_rng(params.seed)
        self._reset_peak_memory()

        sampler = (
            make_sampler(temp=params.temperature)
            if params.temperature > 0
            else make_sampler(temp=0.0)
        )

        # Stream path: measured per-token timestamps + TTFT.
        # Non-stream path: wall-clock e2e only (no fabricated token times / TTFT).
        start = time.perf_counter()
        timestamps: list[float] = []
        pieces: list[str] = []
        output: str | None = None
        e2e_only = False

        # Only treat missing/incompatible stream API as fallback triggers — not
        # RuntimeError/OOM/etc., which must reach the outer GenerationError path.
        try:
            from mlx_lm import stream_generate as _stream_generate
        except ImportError:
            _stream_generate = None

        try:
            if _stream_generate is not None:
                try:
                    for response in _stream_generate(
                        self._model,
                        self._tokenizer,
                        prompt=prompt,
                        max_tokens=params.max_tokens,
                        sampler=sampler,
                    ):
                        pieces.append(response.text)
                        timestamps.append(time.perf_counter() - start)
                    output = "".join(pieces)
                except TypeError:
                    # Signature / return-shape mismatch with this mlx-lm version.
                    pieces.clear()
                    timestamps.clear()
                    output = None

            if output is None:
                output = mlx_generate(
                    self._model,
                    self._tokenizer,
                    prompt=prompt,
                    max_tokens=params.max_tokens,
                    sampler=sampler,
                )
                e2e_only = True
                timestamps = []
        except Exception as e:
            raise GenerationError(str(e)) from e

        text = output if isinstance(output, str) else str(output)
        e2e_s = time.perf_counter() - start
        mem = self.get_memory_usage_bytes()
        if e2e_only or not timestamps:
            # Non-stream (or empty stream): do not invent per-token times.
            total_tokens = max(1, len(self._tokenizer.encode(text))) if text else 0
            return GenerationResult(
                status=GenerationStatus.SUCCESS,
                output_text=text,
                token_timestamps=[],
                ttft_ms=None,
                total_tokens=total_tokens,
                memory_peak_bytes=mem,
                thermal_state=ThermalReading(
                    method="none", thermal_pressure=None, notes="filled_by_orchestrator"
                ),
                e2e_ms=e2e_s * 1000.0,
            )

        return GenerationResult(
            status=GenerationStatus.SUCCESS,
            output_text=text,
            token_timestamps=timestamps,
            ttft_ms=timestamps[0] * 1000.0,
            total_tokens=len(timestamps),
            memory_peak_bytes=mem,
            thermal_state=ThermalReading(
                method="none", thermal_pressure=None, notes="filled_by_orchestrator"
            ),
            e2e_ms=timestamps[-1] * 1000.0,
        )

    def get_memory_usage_bytes(self) -> int:
        """Prefer MLX allocator peak; fall back to process RSS; else 0."""
        try:
            import mlx.core as mx

            return int(mx.get_peak_memory())
        except Exception:
            pass
        try:
            import resource

            # macOS: ru_maxrss is bytes; Linux: kilobytes. Workbench targets Apple Silicon.
            return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        except Exception:
            return 0

    @staticmethod
    def _seed_rng(seed: int) -> None:
        """Apply GenParams.seed to MLX PRNG so temp>0 (and any stochastic paths) are reproducible."""
        try:
            import mlx.core as mx

            mx.random.seed(int(seed))
        except Exception:
            # Non-fatal: greedy temp=0 still works; sampling repro degrades.
            pass

    @staticmethod
    def _reset_peak_memory() -> None:
        try:
            import mlx.core as mx

            mx.reset_peak_memory()
        except Exception:
            pass

    def _ensure(self) -> None:
        if self._model is None or self._tokenizer is None:
            raise EngineLoadError("MlxLmEngine.load_model() not called")
