"""Groq TPM guard: token-per-minute rate limiting for the thinking model.

The thinking side (Groq) is bounded by its quota in TOKENS per minute, not by
request count - so an interval pacer (like Mistral 1 req/1.5s) is the wrong
shape. This module provides:

  * TPMBucket - a fixed-window token bucket (8,000 TPM default) that
    await acquire(est_tokens) reserves from before a call;
  * TPMLimitedChatOpenAI - a ChatOpenAI subclass that estimates the token
    cost of each request and paces it through the process-wide bucket.

Design mirrors src/llm/ratelimit.py: never sleeps holding the lock; a
request larger than the whole window is admitted + left to provider quota.
"""

from __future__ import annotations

import asyncio
import logging
import os
import random
import time
from typing import Any, Optional

from langchain_openai import ChatOpenAI

logger = logging.getLogger("x_deepagents.ratelimit")


# ---------------------------------------------------------------------------
# Token estimation
# ---------------------------------------------------------------------------

def estimate_tokens(text: Any) -> int:
    """Rough token estimate (~4 chars/token)."""
    s = str(text or "")
    return max(1, len(s) // 4)


def estimate_messages_tokens(messages: list) -> int:
    """Estimate input tokens for a langchain message list."""
    total = 0
    for m in messages:
        content = getattr(m, "content", None)
        if isinstance(content, str):
            total += estimate_tokens(content)
        elif isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and block.get("type") == "text":
                    total += estimate_tokens(block.get("text", ""))
                elif isinstance(block, dict) and block.get("text"):
                    total += estimate_tokens(block.get("text"))
    return max(1, total)


# ---------------------------------------------------------------------------
# TPM bucket (fixed window)
# ---------------------------------------------------------------------------

class TPMBucket:
    """Fixed-window token bucket: up to tokens_per_minute per 60s window.

    acquire(n) blocks (without holding the lock) until the reservation fits
    in the current window. A request larger than the whole window is admitted
    immediately (same policy as the backend bucket) - waiting could never
    admit it; provider quota + 429 retry handle it instead.
    """

    def __init__(self, tokens_per_minute: int, *, clock=None) -> None:
        self.tokens_per_minute = int(tokens_per_minute)
        self._lock = asyncio.Lock()
        self._window_start = 0.0
        self._tokens_used = 0
        self._clock = clock or time.monotonic  # injectable for fast tests

    @property
    def enabled(self) -> bool:
        return self.tokens_per_minute > 0

    async def acquire(self, est_tokens: int) -> float:
        """Reserve est_tokens; returns seconds spent waiting."""
        if not self.enabled or est_tokens <= 0:
            return 0.0
        if est_tokens > self.tokens_per_minute:
            logger.warning(
                "token request (%d) exceeds the %d TPM window; admitting "
                "without accounting (provider quota applies)",
                est_tokens, self.tokens_per_minute,
            )
            return 0.0
        waited = 0.0
        while True:
            async with self._lock:
                now = self._clock()
                if now - self._window_start >= 60.0:
                    self._window_start = now
                    self._tokens_used = 0
                window_left = 60.0 - (now - self._window_start)
                if self._tokens_used + est_tokens <= self.tokens_per_minute:
                    self._tokens_used += est_tokens
                    return waited
                wait = window_left + random.uniform(0.05, 0.5)
                wait = min(wait, 120.0)
            await asyncio.sleep(wait)
            waited += wait


_bucket: Optional[TPMBucket] = None


def get_tpm_bucket(tokens_per_minute: Optional[int] = None) -> TPMBucket:
    """Process-wide bucket (shared across all Groq model instances)."""
    global _bucket
    if _bucket is None:
        tpm = tokens_per_minute
        if tpm is None:
            raw = os.environ.get("XDEEP_GROQ_TPM", "").strip()
            tpm = int(raw) if raw.isdigit() else 8000
        _bucket = TPMBucket(tpm)
    return _bucket


def reset_tpm_bucket() -> None:
    """Drop the singleton (tests)."""
    global _bucket
    _bucket = None


# ---------------------------------------------------------------------------
# TPMLimitedChatOpenAI
# ---------------------------------------------------------------------------

class TPMLimitedChatOpenAI(ChatOpenAI):
    """ChatOpenAI that paces each request through the Groq TPM bucket.

    A real BaseChatModel subclass, so deepagents accepts it unchanged.
    Estimates input tokens from the message list and reserves them before
    every generate path (sync + async).
    """

    output_tokens_allowance: int = 1024  # estimated output per call

    def __init__(self, *, tpm: Optional[int] = None,
                 output_tokens_allowance: Optional[int] = None, **kwargs):
        super().__init__(**kwargs)  # type: ignore[call-arg]
        self._bucket = get_tpm_bucket(tpm)
        if output_tokens_allowance is not None:
            self.output_tokens_allowance = max(0, int(output_tokens_allowance))

    def _estimate_request_tokens(self, messages: list) -> int:
        return estimate_messages_tokens(messages) + self.output_tokens_allowance

    def _generate(self, *args, **kwargs):
        _sync_acquire(self._bucket, self._estimate_request_tokens(args[0]))
        return super()._generate(*args, **kwargs)

    async def _agenerate(self, *args, **kwargs):
        tokens = self._estimate_request_tokens(args[0])
        await self._bucket.acquire(tokens)
        return await super()._agenerate(*args, **kwargs)

    def invoke(self, input, config: Optional[dict] = None, **kwargs):
        msgs = input if isinstance(input, list) else [input]
        _sync_acquire(self._bucket, self._estimate_request_tokens(msgs))
        return super().invoke(input, config=config, **kwargs)

    async def ainvoke(self, input, config: Optional[dict] = None, **kwargs):
        msgs = input if isinstance(input, list) else [input]
        await self._bucket.acquire(self._estimate_request_tokens(msgs))
        return await super().ainvoke(input, config=config, **kwargs)


def _sync_acquire(bucket: TPMBucket, tokens: int) -> None:
    """Pace from a sync context (no running loop, or inside one)."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        asyncio.run(bucket.acquire(tokens))
        return
    if loop.is_running():
        future = asyncio.run_coroutine_threadsafe(bucket.acquire(tokens), loop)
        future.result()


__all__ = [
    "TPMBucket", "TPMLimitedChatOpenAI",
    "estimate_tokens", "estimate_messages_tokens",
    "get_tpm_bucket", "reset_tpm_bucket",
]