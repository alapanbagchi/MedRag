"""Central LLM pool tests (fake clocks where possible, tiny real waits)."""
from __future__ import annotations

import asyncio
import time

import httpx
import pytest
from openai import APIStatusError, RateLimitError

from src.llm import pool as pool_mod
from src.llm.pool import (
    ModelPool,
    estimate_tokens,
    get_pool,
    is_retryable,
    retry_delay,
)


def _resp(status, headers=None):
    return httpx.Response(status, headers=headers or {},
                          request=httpx.Request("POST", "http://x"))


def test_registry_identity_and_separation(monkeypatch):
    monkeypatch.setattr(pool_mod, "_pools", {})
    a = get_pool("http://a", "k1")
    assert get_pool("http://a", "k1") is a
    assert get_pool("http://a", "k2") is not a
    assert get_pool("http://b", "k1") is not a


def test_registry_env_overrides(monkeypatch):
    monkeypatch.setattr(pool_mod, "_pools", {})
    monkeypatch.setenv("LLM_RPM", "12")
    monkeypatch.setenv("LLM_TPM", "6000")
    monkeypatch.setenv("LLM_MAX_INFLIGHT", "3")
    pool = get_pool("http://env-test")
    assert (pool.rpm, pool.tpm) == (12.0, 6000.0)


def test_estimate_tokens_counts_chars_plus_reserve():
    assert estimate_tokens("a" * 400) == 100
    assert estimate_tokens("a" * 400, "b" * 40, reserve=50) == 160


def test_is_retryable_matrix():
    assert is_retryable(RateLimitError("429", response=_resp(429), body=None))
    assert is_retryable(APIStatusError("x", response=_resp(503), body=None))
    assert not is_retryable(APIStatusError("x", response=_resp(400), body=None))
    assert not is_retryable(APIStatusError("x", response=_resp(401), body=None))
    assert is_retryable(httpx.ConnectError("down"))
    assert is_retryable(httpx.ReadTimeout("slow"))
    assert not is_retryable(asyncio.TimeoutError())
    assert not is_retryable(ValueError("bad json"))
    assert not is_retryable(asyncio.CancelledError())


def test_is_retryable_duck_typed_status():
    from types import SimpleNamespace

    assert is_retryable(SimpleNamespace(status_code=429))
    assert is_retryable(SimpleNamespace(status_code=500))
    assert not is_retryable(SimpleNamespace(status_code=403))
    assert not is_retryable(SimpleNamespace(status_code=None))


def test_retry_delay_honors_retry_after():
    exc = RateLimitError("429", response=_resp(429, {"retry-after": "3"}), body=None)
    assert retry_delay(0, exc) == 3.0
    exc_ms = RateLimitError("429", response=_resp(429, {"retry-after-ms": "2500"}),
                            body=None)
    assert retry_delay(0, exc_ms) == 2.5
    capped = RateLimitError("429", response=_resp(429, {"retry-after": "999"}), body=None)
    assert retry_delay(0, capped) == 60.0


def test_retry_delay_grows_exponentially():
    assert 2.0 <= retry_delay(0, ValueError("x")) <= 3.0
    assert 8.0 <= retry_delay(2, ValueError("x")) <= 9.0
    assert retry_delay(99, ValueError("x")) <= 61.0


async def test_slot_paces_starts_by_rpm():
    pool = ModelPool("pace", rpm=600.0, tpm=10_000_000.0, max_inflight=8)
    started = time.perf_counter()
    for _ in range(4):
        async with pool.slot(1):
            pass
    # 3 gaps x 0.1s interval.
    assert time.perf_counter() - started >= 0.2


async def test_slot_enforces_tpm_bucket():
    pool = ModelPool("tpm", rpm=3600.0, tpm=600.0, max_inflight=8)
    async with pool.slot(600):  # drains the full bucket instantly
        pass
    started = time.perf_counter()
    async with pool.slot(5):  # ~0.5s to refill at 600/60 per sec
        pass
    assert time.perf_counter() - started >= 0.3


async def test_slot_serializes_inflight():
    pool = ModelPool("ser", rpm=3600.0, tpm=10_000_000.0, max_inflight=1)
    entered = asyncio.Event()
    release = asyncio.Event()
    order: list = []

    async def holder():
        async with pool.slot(1):
            entered.set()
            order.append("holder-in")
            await release.wait()
            order.append("holder-out")

    async def waiter():
        await entered.wait()
        async with pool.slot(1):
            order.append("waiter-in")

    task = asyncio.create_task(holder())
    await entered.wait()
    await asyncio.sleep(0.05)
    waiter_task = asyncio.create_task(waiter())
    await asyncio.sleep(0.05)
    assert order == ["holder-in"]  # waiter blocked on the semaphore
    release.set()
    await asyncio.gather(task, waiter_task)
    assert order == ["holder-in", "holder-out", "waiter-in"]
