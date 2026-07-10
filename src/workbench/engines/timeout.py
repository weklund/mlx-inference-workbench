"""Wall-clock timeout around engine.generate (orchestrator policy)."""

from __future__ import annotations

from queue import Empty, Queue
import threading

from workbench.engines.base import Engine, GenParams
from workbench.models import GenerationResult, GenerationStatus, ThermalReading


def _timeout_result(timeout: float) -> GenerationResult:
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


def timed_generate(
    engine: Engine,
    prompt: str,
    params: GenParams,
) -> GenerationResult:
    """Bounded generate for correctness, warmup, and measure (HLD: TIMEOUT, not hang).

    Runs generate on a **daemon** thread so a hung worker cannot block process /
    CLI exit after we return TIMEOUT. Threads cannot be force-killed; the worker
    may still hold resources until process exit, but it will not keep the
    interpreter alive.
    """
    timeout = params.timeout_sec
    if timeout is None or timeout <= 0:
        return engine.generate(prompt, params)

    # (kind, payload): "ok" + GenerationResult | "err" + BaseException
    out: Queue[tuple[str, object]] = Queue(maxsize=1)

    def worker() -> None:
        try:
            out.put(("ok", engine.generate(prompt, params)))
        except BaseException as e:  # noqa: BLE001 — re-raised on caller thread
            out.put(("err", e))

    thread = threading.Thread(
        target=worker,
        name="workbench-generate-timeout",
        daemon=True,
    )
    thread.start()

    try:
        kind, payload = out.get(timeout=timeout)
    except Empty:
        return _timeout_result(float(timeout))

    if kind == "err":
        raise payload  # type: ignore[misc]
    return payload  # type: ignore[return-value]


# Back-compat alias for older imports / tests
generate_with_timeout = timed_generate
