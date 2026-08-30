"""Unit tests for the LLM observer bridge (prompt/thought/output capture)."""
import json

import pytest
from pydantic import BaseModel
from pydantic_ai import Agent

from src.llm.run import _extract_thought, ask_structured, reset_llm_observer, set_llm_observer

from tests.conftest import native_test_model


class _Decision(BaseModel):
    action: str


def test_extract_thought_handles_think_and_thought_blocks():
    assert _extract_thought('<think>reasoning</think>{"a":1}') == "reasoning"
    assert _extract_thought('<thought>step one</thought><thought>step two</thought>') == "step one\n\nstep two"
    assert _extract_thought('{"a":1}') == ""
    assert _extract_thought("") == ""


@pytest.mark.asyncio
async def test_ask_structured_notifies_observer_with_thought():
    captured = []
    set_llm_observer(lambda **kw: captured.append(kw))
    try:
        model = native_test_model('<think>deciding to stop</think>' + json.dumps({"action": "STOP"}))
        agent = Agent(model, system_prompt="sys")
        out = await ask_structured(agent, "the question", _Decision, label="orchestrator")
    finally:
        reset_llm_observer()

    assert out.action == "STOP"
    assert captured
    ev = captured[-1]
    assert ev["label"] == "orchestrator"
    assert ev["input"] == "the question"
    assert "deciding to stop" in ev["thought"]
    assert ev["output"] == {"action": "STOP"}


@pytest.mark.asyncio
async def test_agentic_v2_observer_emits_llm_call_event():
    from src.agentic_v2 import llm_observer

    class Emitter:
        def __init__(self):
            self.events = []

        def emit(self, type_, **fields):
            self.events.append({"type": type_, **fields})

    em = Emitter()
    llm_observer.install(em)
    llm_observer.set_iteration(7)
    try:
        model = native_test_model(json.dumps({"action": "STOP"}))
        agent = Agent(model, system_prompt="sys")
        await ask_structured(agent, "q", _Decision, label="objective_verifier")
    finally:
        llm_observer.uninstall()

    calls = [e for e in em.events if e["type"] == "llm_call"]
    assert calls
    call = calls[-1]
    assert call["iteration"] == 7
    assert call["agent"] == "verifier"          # aliased from objective_verifier
    assert call["label"] == "objective_verifier"
    assert call["input"] == "q"
    assert call["output"] == {"action": "STOP"}


@pytest.mark.asyncio
async def test_agentic_v2_observer_emits_llm_call_start_before_end():
    """The UI drives its live spinner/timing from llm_call_start."""
    from src.agentic_v2 import llm_observer

    class Emitter:
        def __init__(self):
            self.events = []

        def emit(self, type_, **fields):
            self.events.append({"type": type_, **fields})

    em = Emitter()
    llm_observer.install(em)
    llm_observer.set_iteration(9)
    try:
        model = native_test_model(" thinkingchecking intent response" + json.dumps({"action": "STOP"}))
        agent = Agent(model, system_prompt="sys")
        await ask_structured(agent, "question text", _Decision, label="objective_verifier")
    finally:
        llm_observer.uninstall()

    starts = [e for e in em.events if e["type"] == "llm_call_start"]
    ends = [e for e in em.events if e["type"] == "llm_call"]
    assert starts and ends
    start = starts[-1]
    assert start["iteration"] == 9
    assert start["agent"] == "verifier"
    assert start["label"] == "objective_verifier"
    # start carries no prompt/thought/output yet
    assert start.get("input") is None
    # and the start precedes the matching end in the stream
    assert em.events.index(start) < em.events.index(ends[-1])


@pytest.mark.asyncio
async def test_observer_aliases_agent_from_label_prefix():
    """Per-passage verifier labels include an index+doc; agent must stay verifier."""
    from src.agentic_v2 import llm_observer

    class Emitter:
        def __init__(self):
            self.events = []

        def emit(self, type_, **fields):
            self.events.append({"type": type_, **fields})

    em = Emitter()
    llm_observer.install(em)
    llm_observer.set_iteration(3)
    try:
        model = native_test_model(json.dumps({"action": "STOP"}))
        agent = Agent(model, system_prompt="sys")
        await ask_structured(agent, "q", _Decision, label="objective_verifier:1:PMC123")
    finally:
        llm_observer.uninstall()

    ends = [e for e in em.events if e["type"] == "llm_call"]
    assert ends
    call = ends[-1]
    assert call["agent"] == "verifier"                     # prefix aliasing
    assert call["label"] == "objective_verifier:1:PMC123"  # full label preserved


@pytest.mark.asyncio
async def test_pipeline_emits_llm_call_events_with_iteration():
    from src.agentic_v2.actions import ActionExecutor
    from src.agentic_v2.orchestrator import OrchestratorAgent
    from src.agentic_v2.pipeline import AgenticV2Pipeline
    from src.config import AppConfig

    class Emitter:
        def __init__(self):
            self.events = []

        def emit(self, type_, **fields):
            self.events.append({"type": type_, **fields})

    class LLMOrch:
        def __init__(self, model):
            self.inner = OrchestratorAgent(config=AppConfig(), model=model)

        async def decide(self, state):
            return await self.inner.decide(state)

    em = Emitter()
    cfg = AppConfig()
    ex = ActionExecutor(config=cfg, corpus=None)
    pipeline = AgenticV2Pipeline(
        config=cfg,
        orchestrator=LLMOrch(native_test_model(json.dumps({"action": "STOP"}))),
        executor=ex, events=em,
    )
    await pipeline.answer("q")
    calls = [e for e in em.events if e["type"] == "llm_call"]
    assert calls
    assert calls[0]["agent"] == "orchestrator"
    assert calls[0]["iteration"] == 1
