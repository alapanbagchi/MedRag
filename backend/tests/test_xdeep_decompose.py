"""Decompose failure tests (Gap F): a broken parse retries once with a strict
nudge; a second failure is a LOUD terminal error, never a silent single-task
collapse. No real LLM calls - decompose_requirements is patched.
"""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import pytest

from src.x_deepagents.graph import DecomposeError


def _agent_with(content):
    return SimpleNamespace(ainvoke=async_lambda(
        lambda input: {"messages": [SimpleNamespace(content=content)]}))


def async_lambda(fn):
    async def _inner(*a, **k):
        return fn(*a, **k)
    return _inner


GOOD_JSON = json.dumps({"requirements": [
    {"id": "T1", "text": "Effect of exercise on BP",
     "entities": ["exercise", "blood pressure"], "target_n": 2},
    {"id": "T2", "text": "Effect of diet on BP",
     "entities": ["diet", "blood pressure"], "target_n": 2},
]})


def test_decompose_parses_valid_plan(monkeypatch):
    from src.x_deepagents.agents.stages import decompose_requirements

    calls = {"n": 0}

    async def fake_ainvoke(input):
        calls["n"] += 1
        return {"messages": [SimpleNamespace(content=GOOD_JSON)]}

    agent = SimpleNamespace(ainvoke=fake_ainvoke)
    out = asyncio.run(decompose_requirements(agent, "Does diet reduce BP?"))
    assert len(out) == 2
    assert calls["n"] == 1   # no retry on success


def test_decompose_retries_once_then_succeeds(monkeypatch):
    from src.x_deepagents.agents.stages import decompose_requirements

    calls = {"n": 0}

    async def fake_ainvoke(input):
        calls["n"] += 1
        if calls["n"] == 1:
            return {"messages": [SimpleNamespace(content="no json here")]}
        return {"messages": [SimpleNamespace(content=GOOD_JSON)]}

    agent = SimpleNamespace(ainvoke=fake_ainvoke)
    out = asyncio.run(decompose_requirements(agent, "q?"))
    assert len(out) == 2
    assert calls["n"] == 2   # strict retry fired


def test_decompose_fails_loud_after_two_failures(monkeypatch):
    from src.x_deepagents.agents.stages import decompose_requirements

    calls = {"n": 0}

    async def fake_ainvoke(input):
        calls["n"] += 1
        return {"messages": [SimpleNamespace(content="still not json")]}

    agent = SimpleNamespace(ainvoke=fake_ainvoke)
    with pytest.raises(ValueError):
        asyncio.run(decompose_requirements(agent, "q?"))
    assert calls["n"] == 2


def test_decompose_node_raises_decompose_error_not_single_task(monkeypatch):
    """The graph node must RAISE after retry failure - never silently build
    a single-task run that hides a broken multi-part decomposition."""
    import asyncio

    from src.x_deepagents.graph import decompose_node

    from src.x_deepagents.state import Phase, RunBudget, XDeepRunState

    run = XDeepRunState(run_id="x", question="A, B and C?",
                        phase=Phase.DECOMPOSE, budget=RunBudget())

    class FakeOrch:
        async def ainvoke(self, input):
            return {"messages": [SimpleNamespace(content="broken")]}

    # patch the stages import used inside decompose_node (auto-restored)
    monkeypatch.setattr("src.x_deepagents.agents.stages.make_orchestrator",
                        lambda model=None: FakeOrch())

    async def run_it():
        return await decompose_node({"state": run})

    with pytest.raises(DecomposeError):
        asyncio.run(run_it())
