"""Wall-clock timeout around engine.generate (shared by orchestrator and engines)."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FuturesTimeout

from workbench.engines.base import Engine, GenParams
from workbench.models import GenerationResult, GenerationStatus, ThermalReading


def generate_with_timeout(
    engine: Engine,
    prompt: str,
    params: GenParams,
) -> GenerationResult:
    """Enforce wall-clock timeout around engine.generate (HLD: TIMEOUT, not hang).

    On deadline, return TIMEOUT immediately. Do not wait for the worker thread to
    finish (``ThreadPoolExecutor`` context-manager shutdown uses wait=True and
    would re-block for the full hung generation).
    """
    timeout = params.timeout_sec
    if timeout is None or timeout <= 0:
        return engine.generate(prompt, params)

    pool = ThreadPoolExecutor(max_workers=1)
    try:
        fut = pool.submit(engine.generate, prompt, params)
        try:
            return fut.result(timeout=timeout)
        except FuturesTimeout:
            # Best-effort only: cancel works if the task has not started yet.
            fut.cancel()
            return GenerationResult(
                status=GenerationStatus.TIMEOUT,
                output_text="",
                token_timestamps=[],
                ttft_ms=0.0,
                total_tokens=0,
                memory_peak_bytes=0,
                thermal_state=ThermalReading(method="timeout"),
                error_message=f"per-iteration timeout exceeded ({timeout}s)",
            )
    finally:
        # wait=False: return control without joining a stuck generate() worker.
        # cancel_futures drops queued work; a running thread is abandoned.
        pool.shutdown(wait=False, cancel_futures=True)
