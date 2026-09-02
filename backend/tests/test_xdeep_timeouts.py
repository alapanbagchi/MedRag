"""Timeout + concurrency tests (robustness)."""

from __future__ import annotations

import asyncio

import pytest

from src.x_deepagents.timeouts import (
    max_workers,
    run_with_timeout,
    timeout_for,
)


async def _slow():
    await asyncio.sleep(5)
    return "done"


async def _fast():
    return "ok"


def test_stage_defaults_are_positive():
    for stage in ("decompose", "worker", "conflict", "resolution",
                   "gap", "synthesis", "llm_call"):
        assert timeout_for(stage) >= 1.0


def test_env_override(monkeypatch):
    monkeypatch.setenv("XDEEP_TIMEOUT_WORKER", "99")
    assert timeout_for("worker") == 99.0
    monkeypatch.setenv("XDEEP_TIMEOUT_WORKER", "garbage")
    assert timeout_for("worker") >= 1.0


def test_env_worker_count(monkeypatch):
    monkeypatch.setenv("XDEEP_MAX_WORKERS", "7")
    assert max_workers() == 7
    monkeypatch.setenv("XDEEP_MAX_WORKERS", "abc")
    assert max_workers() >= 1


def test_run_with_timeout_honors_seconds():
    out = asyncio.run(run_with_timeout(_fast(), 2.0, "fast"))
    assert out == "ok"


def test_run_with_timeout_raises_on_slow():
    with pytest.raises(asyncio.TimeoutError):
        asyncio.run(run_with_timeout(_slow(), 0.1, "slow"))
