"""Centralized LLM call pool: one queue per endpoint with rate limiting.

Every shared-endpoint call in this process (top-level planner, evidence
planner, evidence verifier — all on the same Mistral endpoint) goes through
the shared :class:`ModelPool` for its endpoint, so parallel deep-research
tasks queue instead of burning the quota with 429s.

Defaults match the documented Mistral La Plateforme free tier
(~60 RPM / 500,000 TPM; limits vary per model and account — check
https://admin.mistral.ai/plateforme/limits), overridable via ``LLM_RPM`` /
``LLM_TPM`` / ``LLM_MAX_INFLIGHT``. Per-day/monthly accounting needs
persistent state and is deliberately out of scope.
"""

from __future__ import annotations

import asyncio
import os
import random
import threading
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import httpx
from openai import APIConnectionError, APIStatusError, APITimeoutError, RateLimitError

from src.lib import narrate
from src.lib.trace import get_trace

DEFAULT_RPM = 60.0
DEFAULT_TPM = 500_000.0
DEFAULT_MAX_INFLIGHT = 8

RETRYABLE_STATUS = frozenset({429, 500, 502, 503, 504})


def _env_float(name: str, default: float) -> float:
    try:
        return max(0.0, float(os.environ.get(name, default)))
    except (TypeError, ValueError):
        return default


def _env_int(name: str, default: int) -> int:
    try:
        return max(1, int(os.environ.get(name, default)))
    except (TypeError, ValueError):
        return default


class _TokenBucket:
    """Async token bucket: ``take(n)`` waits until n tokens are available."""

    def __init__(self, rate_per_sec: float, capacity: float) -> None:
        self._rate = max(1e-9, rate_per_sec)
        self._cap = max(1.0, capacity)
        self._tokens = self._cap
        self._stamp = time.monotonic()
        self._lock = asyncio.Lock()

    async def take(self, n: float) -> None:
        n = max(0.0, float(n))
        while True:
            async with self._lock:
                now = time.monotonic()
                self._tokens = min(self._cap, self._tokens + (now - self._stamp) * self._rate)
                self._stamp = now
                if self._tokens >= n:
                    self._tokens -= n
                    return
                wait = (n - self._tokens) / self._rate
            await asyncio.sleep(wait)


class ModelPool:
    """Rate gate for one (endpoint, key): RPM pacing + TPM bucket + inflight cap."""

    def __init__(self, name: str, *, rpm: float, tpm: float, max_inflight: int) -> None:
        self.name = name
        self.rpm = rpm
        self.tpm = tpm
        self._interval = 60.0 / max(1e-9, rpm)
        self._bucket = _TokenBucket(tpm / 60.0, tpm)
        self._pace_lock = asyncio.Lock()
        self._next_start = 0.0
        self._inflight = asyncio.Semaphore(max(1, max_inflight))

    async def _pace(self) -> None:
        async with self._pace_lock:
            now = time.monotonic()
            delay = max(0.0, self._next_start - now)
            self._next_start = max(now, self._next_start) + self._interval
        if delay > 0:
            await asyncio.sleep(delay)

    @asynccontextmanager
    async def slot(self, estimated_tokens: float, *, label: str = "") -> AsyncIterator[None]:
        """Wait for quota, then yield while holding the inflight semaphore.

        Pacing + TPM budget are consumed at entry; the semaphore stays held
        for the call body so bursts cannot pile up downstream.
        """
        queued_for = time.monotonic()
        async with self._inflight:
            await self._pace()
            await self._bucket.take(estimated_tokens)
            waited = time.monotonic() - queued_for
            if waited > 0.5:
                tag = f" [{label}]" if label else ""
                narrate.say(f"  [pool:{self.name}]{tag} queued {waited:.1f}s"
                              f" (rpm={self.rpm:g}, tpm={self.tpm:g})")
                get_trace().log("pool_wait", pool=self.name, label=label,
                                waited_s=round(waited, 1))
            yield


_pools: dict[tuple[str, str], ModelPool] = {}
_pools_lock = threading.Lock()


def get_pool(base_url: str, api_key: str | None = None, *,
             rpm: float | None = None, tpm: float | None = None,
             max_inflight: int | None = None) -> ModelPool:
    """Process-wide pool per (endpoint, key); created once, shared by all callers."""
    key = (base_url, api_key or "")
    with _pools_lock:
        pool = _pools.get(key)
        if pool is None:
            pool = ModelPool(
                base_url,
                rpm=rpm if rpm is not None else _env_float("LLM_RPM", DEFAULT_RPM),
                tpm=tpm if tpm is not None else _env_float("LLM_TPM", DEFAULT_TPM),
                max_inflight=(max_inflight if max_inflight is not None
                              else _env_int("LLM_MAX_INFLIGHT", DEFAULT_MAX_INFLIGHT)),
            )
            _pools[key] = pool
        return pool


def estimate_tokens(*texts: object, reserve: int = 0) -> int:
    """Rough token budget for a call: ~4 chars/token plus completion reserve."""
    return sum(max(0, len(str(t)) // 4) for t in texts) + max(0, reserve)


def _status_of(exc: BaseException) -> int | None:
    for attr in ("status_code", "status"):
        try:
            value = getattr(exc, attr, None)
            if isinstance(value, bool):
                continue
            if value is not None:
                return int(value)
        except (TypeError, ValueError):
            pass
    response = getattr(exc, "response", None)
    if response is not None:
        for attr in ("status_code", "status"):
            try:
                value = getattr(response, attr, None)
                if value is not None and not isinstance(value, bool):
                    return int(value)
            except (TypeError, ValueError):
                pass
    return None


def is_retryable(exc: BaseException) -> bool:
    """True for transient endpoint failures (429/5xx, timeouts, disconnects).

    Never retryable: validation errors, auth errors (400/401/403/404), our
    own timeouts and cancellations — those fail open (or loud) immediately.
    """
    if isinstance(exc, (asyncio.CancelledError, asyncio.TimeoutError,
                        TypeError, ValueError, AttributeError)):
        return False
    if isinstance(exc, (RateLimitError, APIConnectionError, APITimeoutError,
                        httpx.ConnectError, httpx.ReadError,
                        httpx.RemoteProtocolError, httpx.PoolTimeout,
                        httpx.TimeoutException)):
        return True
    if isinstance(exc, APIStatusError):
        return exc.status_code in RETRYABLE_STATUS
    status = _status_of(exc)
    return status in RETRYABLE_STATUS if status is not None else False


def _retry_after_seconds(exc: BaseException) -> float | None:
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", None)
    if not headers:
        return None
    try:
        get = headers.get
    except AttributeError:
        return None
    for key, scale in (("retry-after-ms", 0.001), ("retry-after", 1.0)):
        try:
            raw = get(key)
            if raw is None:
                continue
            return max(0.0, float(raw) * scale)
        except (TypeError, ValueError):
            continue
    return None


def retry_delay(attempt: int, exc: BaseException, *, base: float = 2.0,
                cap: float = 60.0) -> float:
    """Backoff for attempt N (0-based): honors Retry-After, else exp + jitter."""
    hinted = _retry_after_seconds(exc)
    if hinted is not None:
        return min(cap, hinted)
    return min(cap, base * (2.0 ** max(0, attempt))) + random.uniform(0.0, 1.0)
