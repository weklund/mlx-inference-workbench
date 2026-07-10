"""MTPLX backend — optional speculative Engine plugin (issue #9).

Drives ``mtplx.load`` + ``generate_mtpk`` only. Wall-clock timeout stays
orchestrator policy. Timing is e2e-only until a stream path is wired.
"""

from __future__ import annotations

import time
from typing import Any

from workbench.config import ModelConfig, MtplxOptions
from workbench.engines.base import Engine, EngineLoadError, GenerationError, GenParams
from workbench.models import GenerationResult, GenerationStatus, ThermalReading

# Default when YAML omits model.mtplx (engine still usable with defaults).
_DEFAULT_MTPLX = MtplxOptions()


def speculative_metrics_from_stats(stats: Any) -> tuple[float | None, float | None]:
    """Map MTPLX ``GenerationStats`` to harness speculative fields.

    Aligns with MTPLX's own progress metrics:

    - ``acceptance_rate`` = ``accepted_drafts / drafted_tokens``
      (``draft_acceptance_rate``)
    - ``accepted_length_mean`` = ``accepted_drafts / verify_calls``
      (``accepted_per_verify``)

    When no draft tokens were produced, both fields are ``None`` (honest
    N/A — not 0.0).

    Args:
        stats: Object with integer draft/verify counters (real or mock).

    Returns:
        ``(acceptance_rate, accepted_length_mean)``.
    """
    drafted = int(getattr(stats, "drafted_tokens", 0) or 0)
    accepted = int(getattr(stats, "accepted_drafts", 0) or 0)
    verify_calls = int(getattr(stats, "verify_calls", 0) or 0)

    if drafted <= 0:
        return None, None

    acceptance_rate = accepted / drafted
    accepted_length_mean = (accepted / verify_calls) if verify_calls > 0 else None
    return acceptance_rate, accepted_length_mean


class MtplxEngine(Engine):
    """Apple Silicon MTP speculative backend via the ``mtplx`` package."""

    def __init__(self) -> None:
        self._runtime: Any = None
        self._config: ModelConfig | None = None

    def name(self) -> str:
        """Return the stable backend id ``mtplx``."""
        return "mtplx"

    def supports_speculative(self) -> bool:
        """MTPLX reports draft acceptance metrics when MTP drafts run."""
        return True

    def load_model(self, config: ModelConfig) -> None:
        """Load model with MTP support via ``mtplx.load``.

        Args:
            config: Must include ``model_id`` or ``name`` as a path/HF id
                that MTPLX can load. Optional ``config.mtplx`` sets depth.

        Raises:
            EngineLoadError: Package missing or load failure.
        """
        try:
            import mtplx
        except ImportError as e:
            raise EngineLoadError("mtplx is not installed (try: uv sync --extra mtplx)") from e

        model_id = config.model_id or config.name
        if not model_id:
            raise EngineLoadError("MtplxEngine requires model_id or name")

        try:
            self._runtime = mtplx.load(model_id, mtp=True)
        except Exception as e:
            raise EngineLoadError(f"Failed to load {model_id} via mtplx: {e}") from e
        self._config = config

    def generate(self, prompt: str, params: GenParams) -> GenerationResult:
        """Generate with MTP speculative decoding (e2e-only timestamps).

        Uses ``generate_mtpk`` with ``speculative_depth`` from
        ``config.mtplx`` (default 4). Does not invent per-token timestamps.
        """
        self._ensure()
        assert self._config is not None
        assert self._runtime is not None

        try:
            from mtplx.generation import SamplerConfig, generate_mtpk
        except ImportError as e:
            raise GenerationError("mtplx.generation API unavailable") from e

        self._seed_rng(params.seed)
        self._reset_peak_memory()

        opts = self._config.mtplx or _DEFAULT_MTPLX
        tokenizer = self._runtime.tokenizer
        try:
            prompt_ids = self._encode_prompt(tokenizer, prompt)
        except Exception as e:
            raise GenerationError(f"Failed to encode prompt: {e}") from e

        sampler = SamplerConfig(temperature=float(params.temperature))

        start = time.perf_counter()
        try:
            output = generate_mtpk(
                self._runtime,
                prompt_ids,
                max_tokens=int(params.max_tokens),
                sampler=sampler,
                speculative_depth=int(opts.speculative_depth),
                seed=int(params.seed),
            )
        except Exception as e:
            raise GenerationError(str(e)) from e
        e2e_s = time.perf_counter() - start

        text = getattr(output, "text", None)
        if text is None:
            text = str(output)
        tokens = getattr(output, "tokens", None) or []
        total_tokens = len(tokens) if tokens else self._count_tokens(tokenizer, text)

        stats = getattr(output, "stats", None)
        acc_rate, acc_len = (
            speculative_metrics_from_stats(stats) if stats is not None else (None, None)
        )

        return GenerationResult(
            status=GenerationStatus.SUCCESS,
            output_text=text if isinstance(text, str) else str(text),
            token_timestamps=[],
            ttft_ms=None,
            total_tokens=int(total_tokens),
            memory_peak_bytes=self.get_memory_usage_bytes(),
            thermal_state=ThermalReading(
                method="none", thermal_pressure=None, notes="filled_by_orchestrator"
            ),
            acceptance_rate=acc_rate,
            accepted_length_mean=acc_len,
            e2e_ms=e2e_s * 1000.0,
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

            return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        except Exception:
            return 0

    @staticmethod
    def _encode_prompt(tokenizer: Any, prompt: str) -> list[int]:
        """Encode raw completion text (no chat template) for fair mlx-lm compares."""
        encode = getattr(tokenizer, "encode", None)
        if encode is None:
            raise GenerationError("MTPLX tokenizer has no encode()")
        ids = encode(prompt)
        if hasattr(ids, "tolist"):
            ids = ids.tolist()
        return [int(x) for x in ids]

    @staticmethod
    def _count_tokens(tokenizer: Any, text: str) -> int:
        if not text:
            return 0
        try:
            ids = tokenizer.encode(text)
            if hasattr(ids, "tolist"):
                ids = ids.tolist()
            return max(1, len(ids))
        except Exception:
            return max(1, len(text.split()))

    @staticmethod
    def _seed_rng(seed: int) -> None:
        try:
            import mlx.core as mx

            mx.random.seed(int(seed))
        except Exception:
            pass

    @staticmethod
    def _reset_peak_memory() -> None:
        try:
            import mlx.core as mx

            mx.reset_peak_memory()
        except Exception:
            pass

    def _ensure(self) -> None:
        if self._runtime is None:
            raise EngineLoadError("MtplxEngine.load_model() not called")
