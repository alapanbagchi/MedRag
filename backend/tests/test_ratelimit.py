"Unit tests for the process-wide token bucket and 429-aware retries."

import asyncio

import pytest

from src.llm.ratelimit import (
    TokenBucket,
    estimate_tokens,
    is_rate_limit_error,
    reset_bucket,
    retry_after_seconds,
    run_with_retry,
)


def test_retry_after_header_is_capped():
    # An absurd Retry-After header must never wedge the bucket for minutes.
    class Ex(Exception):
        pass
    e = Ex("rate limit")
    e.headers = {"retry-after": "3600"}
    assert retry_after_seconds(e) == 120.0

    e2 = Ex("rate limit")
    e2.headers = {"retry-after": "7"}
    assert retry_after_seconds(e2) == 7.0


def test_retry_after_inline_is_capped():
    e = Exception("Please retry in 22.44836987s.")
    assert retry_after_seconds(e) == pytest.approx(23.4, abs=0.1)
    e2 = Exception("retry in 500s.")
    assert retry_after_seconds(e2) == 120.0


def test_estimate_tokens():
    assert estimate_tokens("a" * 100) == 25
    assert estimate_tokens("") == 1
    assert estimate_tokens(None) == 1


@pytest.mark.asyncio
async def test_oversized_request_does_not_hang():
    # A single call larger than the TPM window must be admitted immediately.
    # Before the fix this looped forever: window_left computed to 0.0 when
    # _tokens_used was 0, so wait = 0 + jitter forever with no admission.
    bucket = TokenBucket(tokens_per_minute=100)
    # 5000 tokens >> 100/min window; must return quickly instead of spinning
    waited = await asyncio.wait_for(bucket.acquire(5000), timeout=1.0)
    assert waited == 0.0


@pytest.mark.asyncio
async def test_normal_admission_within_window():
    bucket = TokenBucket(tokens_per_minute=1000)
    assert await bucket.acquire(100) == 0.0
    assert await bucket.acquire(100) == 0.0
    assert bucket._tokens_used == 200


@pytest.mark.asyncio
async def test_window_rolls_after_60s():
    bucket = TokenBucket(tokens_per_minute=1000)
    await bucket.acquire(950)
    # Force the window to be considered expired.
    bucket._window_start -= 61.0
    await bucket.acquire(50)
    assert bucket._tokens_used == 50  # rolled: 950 dropped


@pytest.mark.asyncio
async def test_penalize_blocks_until_expiry():
    bucket = TokenBucket(tokens_per_minute=1000)
    await bucket.acquire(100)
    await bucket.penalize(0.3)
    # A concurrent small request must wait for the penalty then succeed.
    t0 = asyncio.get_event_loop().time()
    await bucket.acquire(100)
    elapsed = asyncio.get_event_loop().time() - t0
    assert elapsed >= 0.2
    assert bucket._tokens_used == 200


@pytest.mark.asyncio
async def test_run_with_retry_exhausts_and_raises():
    calls = 0

    async def flaky():
        nonlocal calls
        calls += 1
        raise Exception("429 too many requests")

    with pytest.raises(Exception, match="429"):
        await run_with_retry(flaky, max_attempts=2, base_delay=0.01)
    assert calls == 2


@pytest.mark.asyncio
async def test_run_with_retry_passes_non_ratelimit_immediately():
    async def boom():
        raise ValueError("bad schema")

    with pytest.raises(ValueError):
        await run_with_retry(boom, max_attempts=4, base_delay=0.01)


def test_is_rate_limit_error_markers():
    assert is_rate_limit_error(Exception("resource_exhausted quota exceeded"))
    assert is_rate_limit_error(Exception("rate limit exceeded"))
    assert not is_rate_limit_error(Exception("connection reset"))


def test_reset_bucket_singleton():
    reset_bucket()
    from src.llm.ratelimit import get_bucket
    b1 = get_bucket(tokens_per_minute=123)
    reset_bucket()
    b2 = get_bucket(tokens_per_minute=123)
    assert b1 is not b2
