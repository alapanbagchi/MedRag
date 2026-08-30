"""Process-wide LLM rate limiting and 429-aware retries.

One shared token bucket guards EVERY agent call (planner, verifier, evidence
extractor, synthesizer, rewriter) because provider quotas (e.g. the Gemini
free tier) are per model+project per minute regardless of which agent fired
the request.

Design notes:
  - The bucket NEVER sleeps while holding its lock: waiters compute how long
    to wait under the lock, release it, sleep, then re-check. This avoids
    serializing all callers behind one slow waiter and removes the
    thundering-herd-at-window-reset failure mode.
  - A 429 with a Retry-After hint can *penalize* the bucket: new admissions
    pause until the penalty expires, on top of the token accounting.
  - ``tokens_per_minute <= 0`` disables accounting entirely (local providers).
"""

from __future__ import annotations

import asyncio
import logging
import random
import time
from typing import Any, Awaitable, Callable, Optional, TypeVar

logger = logging.getLogger("src.llm.ratelimit")

T = TypeVar("T")

_RETRYABLE_MARKERS = (
    "429",
    "resource_exhausted",
    "rate limit",
    "rate_limit",
    "too many requests",
    "quota exceeded",
)


def estimate_tokens(text: str) -> int:
    """Rough token estimate (~4 chars/token); good enough for budgeting."""
    return max(1, len(text or "") // 4)


def is_rate_limit_error(exc: BaseException) -> bool:
    """True when an exception looks like an HTTP 429 / quota exhaustion."""
    text = str(exc).lower()
    return any(marker in text for marker in _RETRYABLE_MARKERS)


def retry_after_seconds(exc: BaseException) -> Optional[float]:
    """Extract a Retry-After hint (seconds) from a 429 error if present.

    Handles both the HTTP header form and Gemini's inline form
    ("Please retry in 22.44836987s.") which surfaces inside error bodies.
    """
    import re

    text = str(exc)
    header = getattr(exc, "headers", None)
    if header is not None:
        try:
            value = header.get("retry-after")
            if value is not None:
                # Cap the header form the same way the inline form is capped:
                # an absurd Retry-After (or a date/units we cannot parse
                # meaningfully) must never wedge the shared bucket for minutes.
                return max(0.0, min(float(value), 120.0))
        except Exception:
            pass
    match = re.search(r"retry[ -_]?in\s+(\d+(?:\.\d+)?)\s*s", text, re.IGNORECASE)
    if match:
        return min(float(match.group(1)) + 1.0, 120.0)
    return None


class TokenBucket:
    """Fixed-window token bucket shared by every agent in the process."""

    def __init__(self, tokens_per_minute: int) -> None:
        self.tokens_per_minute = int(tokens_per_minute)
        self._lock = asyncio.Lock()
        self._window_start = 0.0
        self._tokens_used = 0
        self._blocked_until = 0.0
        self._waiters = 0

    @property
    def enabled(self) -> bool:
        return self.tokens_per_minute > 0

    async def acquire(self, est_tokens: int) -> float:
        """Reserve ``est_tokens``; returns seconds spent waiting.

        Never holds the lock while sleeping. When the window is exhausted the
        caller waits for the window to roll over (or for a penalty to lapse),
        then re-checks — accounting is idempotent under contention because the
        reservation itself happens atomically under the lock.
        """
        if not self.enabled or est_tokens <= 0:
            return 0.0
        # A single request larger than the whole per-minute window can NEVER be
        # admitted by waiting (the window rolls but the request still overflows
        # it). Looping here would spin forever with no forward progress — the
        # run would hang on one oversized call (e.g. a verifier sending a full
        # untruncated passage into a 15k TPM bucket). Admit it immediately and
        # let the provider's own quotas + the 429-aware retry loop handle it.
        if est_tokens > self.tokens_per_minute:
            logger.warning(
                "token request (%d tokens) exceeds the %d TPM window; admitting "
                "without accounting (provider quota applies)",
                est_tokens, self.tokens_per_minute,
            )
            return 0.0
        waited_total = 0.0
        while True:
            async with self._lock:
                now = time.monotonic()
                # Roll the fixed window forward if it expired.
                if now - self._window_start >= 60.0:
                    self._window_start = now
                    self._tokens_used = 0
                blocked_for = max(0.0, self._blocked_until - now)
                # The window rolls within a minute, so the longest sensible wait
                # is bounded; never sleep on a home-grown infinite loop value.
                window_left = 60.0 - (now - self._window_start)
                over_budget = self._tokens_used + est_tokens > self.tokens_per_minute
                if blocked_for <= 0.0 and not over_budget:
                    self._tokens_used += est_tokens
                    return waited_total
                wait = blocked_for if blocked_for > 0.0 else window_left
                wait = min(wait, 120.0) + random.uniform(0.05, 0.5)  # cap + jitter
            self._waiters += 1
            try:
                await asyncio.sleep(wait)
                waited_total += wait
            finally:
                self._waiters -= 1

    async def penalize(self, seconds: float) -> None:
        """Block ALL admissions for ``seconds`` (used after a real 429)."""
        if not self.enabled or seconds <= 0:
            return
        async with self._lock:
            self._blocked_until = max(self._blocked_until, time.monotonic() + seconds)
        logger.warning("token bucket penalized for %.1fs after rate-limit signal", seconds)


_bucket: Optional[TokenBucket] = None


def get_bucket(tokens_per_minute: Optional[int] = None) -> TokenBucket:
    """Return the process-wide bucket, creating it on first use."""
    global _bucket
    if _bucket is None:
        if tokens_per_minute is None:
            from src.config import AppConfig

            tokens_per_minute = AppConfig().tokens_per_minute
        _bucket = TokenBucket(tokens_per_minute or 0)
    return _bucket


def reset_bucket() -> None:
    """Drop the singleton (tests)."""
    global _bucket
    _bucket = None


async def run_with_retry(
    fn: Callable[..., Awaitable[T]],
    *args: Any,
    max_attempts: int = 4,
    base_delay: float = 2.0,
    bucket: Optional[TokenBucket] = None,
    **kwargs: Any,
) -> T:
    """Await ``fn(*args, **kwargs)`` with jittered backoff on rate-limit errors.

    Non-rate-limit exceptions propagate immediately. On a 429 we penalize the
    shared bucket with the server's Retry-After hint so that *concurrent*
    callers also pause, then retry with exponential backoff + jitter.
    """
    bucket = bucket or get_bucket()
    last_exc: BaseException | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            return await fn(*args, **kwargs)
        except Exception as exc:
            if not is_rate_limit_error(exc):
                raise
            last_exc = exc
            if attempt >= max_attempts:
                break
            hint = retry_after_seconds(exc)
            delay = hint if hint is not None else base_delay * (2 ** (attempt - 1))
            delay = min(delay, 90.0) + random.uniform(0.0, 1.5)
            await bucket.penalize(hint if hint is not None else 0.0)
            logger.warning(
                "rate limited (attempt %d/%d); retrying in %.1fs (%s)",
                attempt, max_attempts, delay, str(exc)[:160],
            )
            await asyncio.sleep(delay)
    assert last_exc is not None
    raise last_exc


__all__ = [
    "TokenBucket",
    "estimate_tokens",
    "get_bucket",
    "is_rate_limit_error",
    "reset_bucket",
    "retry_after_seconds",
    "run_with_retry",
]
