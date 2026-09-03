"""Per-stage timeouts for the xdeep graph (robustness).

A hung LLM call / searxng fetch / pg scan must never stall a whole research
run. Every stage default here is env-tunable so live ops can tune without
code changes:

    XDEEP_TIMEOUT_DECOMPOSE   (default 90s)   orchestrator decomposition
    XDEEP_TIMEOUT_WORKER      (default 360s)  one research worker (all rounds)
    XDEEP_TIMEOUT_CONFLICT    (default 90s)   contradiction detection
    XDEEP_TIMEOUT_RESOLUTION  (default 180s)  resolution per contradiction
    XDEEP_TIMEOUT_GAP         (default 240s)  gap-resolution pass (whole)
    XDEEP_TIMEOUT_SYNTHESIS   (default 180s)  final synthesis
    XDEEP_TIMEOUT_LLM_CALL    (default 120s)  one agent.ainvoke / tool call
    XDEEP_MAX_WORKERS         (default 4)     concurrent research workers

Usage:
    from src.agents.timeouts import timeout_for, run_with_timeout

    await run_with_timeout(coro, timeout_for("worker"), "worker")
"""

from __future__ import annotations

import asyncio
import logging
import os
from typing import Any, Awaitable

logger = logging.getLogger("x_deepagents.timeouts")

_DEFAULTS = {
    "decompose": 90.0,
    "worker": 360.0,
    "conflict": 90.0,
    "resolution": 180.0,
    "gap": 240.0,
    "synthesis": 180.0,
    "llm_call": 120.0,
}
_ENV = {
    "decompose": "XDEEP_TIMEOUT_DECOMPOSE",
    "worker": "XDEEP_TIMEOUT_WORKER",
    "conflict": "XDEEP_TIMEOUT_CONFLICT",
    "resolution": "XDEEP_TIMEOUT_RESOLUTION",
    "gap": "XDEEP_TIMEOUT_GAP",
    "synthesis": "XDEEP_TIMEOUT_SYNTHESIS",
    "llm_call": "XDEEP_TIMEOUT_LLM_CALL",
}


def timeout_for(stage: str) -> float:
    """Seconds allowed for a stage (env-tunable, sane default)."""
    value = os.environ.get(_ENV.get(stage, ""), "")
    if value and value.strip():
        try:
            return max(1.0, float(value))
        except ValueError:
            pass
    return _DEFAULTS.get(stage, 120.0)


def max_workers() -> int:
    """Concurrent research workers (default 4)."""
    raw = os.environ.get("XDEEP_MAX_WORKERS", "")
    if raw and raw.strip().isdigit():
        return max(1, int(raw))
    return 4


async def run_with_timeout(coro: Awaitable[Any], seconds: float,
                           label: str = "") -> Any:
    """Await coro with a hard timeout. On timeout, logs and raises
    asyncio.TimeoutError so the graph's per-node error handling can
    degrade/pivot instead of hanging the run."""
    if seconds <= 0:
        return await coro
    try:
        return await asyncio.wait_for(coro, timeout=seconds)
    except asyncio.TimeoutError:
        logger.warning("xdeep stage timeout: %s (>%.0fs)", label or "?", seconds)
        raise


__all__ = ["timeout_for", "max_workers", "run_with_timeout"]
