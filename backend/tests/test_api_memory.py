"""Memory + Context layer — API wiring tests (backend/api.py, singular flow).

Verifies the frontend contract: the chat stream emits first-class
``memory`` events (prepare before the run, commit after), the events are not
duplicated into the thinking-log trace, and the research flow receives the
frontend's conversation_id for memory persistence. No LLM / no live DB:
run_research and the memory singleton are faked/stubbed.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

from fastapi.testclient import TestClient

import api as api_module
import src.agents.bridge as bridge_module
from src.memory.store import InMemoryMemoryStore


class FakeMemory:
    """Stands in for MemoryAPI with canned prepare/commit behavior."""

    def __init__(self):
        self.prepare_calls: list[tuple] = []
        self.record_calls: list[tuple] = []

    def prepare_run(self, query: str, conversation_id=None):
        self.prepare_calls.append((query, conversation_id))
        mem = SimpleNamespace(claims=[1, 2], contradictions=[1], gaps=[1])
        return SimpleNamespace(
            session_id="rs_api",
            session=SimpleNamespace(title="hypertension"),
            context=SimpleNamespace(memory=mem),
            context_text=lambda: "CTX")

    def record_run(self, result, session_id="", conversation_id=None,
                   record_conclusion=True):
        self.record_calls.append((result, session_id, conversation_id))
        return SimpleNamespace(
            session_id=session_id or "rs_api",
            as_dict=lambda: {"claims_committed": 2, "claims_deduped": 0,
                             "contradictions": 1, "gaps": 1, "questions": 1})


class EmptyRun:
    run_id = "run1"
    question = "q"
    answer = ""
    gaps: list = []
    contradictions: list = []
    gap_resolutions: list = []
    requirements: list = []

    def verified_items(self):
        return []

    def all_items(self):
        return []


def _stub_flow(monkeypatch):
    """Fake run_research (captures its inputs) + fake memory singleton."""
    calls: list[tuple] = []
    mem = FakeMemory()

    async def fake_run_research(question, memory_context=""):
        calls.append((question, memory_context))
        return EmptyRun()

    monkeypatch.setattr(bridge_module, "run_research", fake_run_research)
    monkeypatch.setattr(bridge_module, "_get_memory_api", lambda: mem)
    return calls, mem


def _stream_lines(client: TestClient, body: dict) -> list[dict]:
    with client.stream("POST", "/v1/chat/stream", json=body) as resp:
        assert resp.status_code == 200
        lines = []
        for raw in resp.iter_lines():
            raw = raw.strip()
            if raw:
                lines.append(json.loads(raw))
    return lines


def test_chat_stream_emits_memory_prepare_and_commit(monkeypatch):
    _stub_flow(monkeypatch)
    client = TestClient(api_module.app)
    lines = _stream_lines(
        client, {"question": "does vitamin D lower blood pressure?",
                 "conversation_id": "conv_fe_123"})

    memory_lines = [l for l in lines if l.get("type") == "memory"]
    prepares = [l for l in memory_lines if l.get("kind") == "prepare"]
    commits = [l for l in memory_lines if l.get("kind") == "commit"]

    assert len(prepares) == 1
    assert prepares[0]["session_id"] == "rs_api"
    assert prepares[0]["session_title"] == "hypertension"
    assert prepares[0]["prior_claims"] == 2
    assert prepares[0]["prior_contradictions"] == 1
    assert prepares[0]["prior_gaps"] == 1

    assert len(commits) == 1
    assert commits[0]["session_id"] == "rs_api"
    assert commits[0]["stats"]["claims_committed"] == 2

    # memory events are first-class: NOT duplicated into the thinking-log
    pipeline_events = [l.get("event") for l in lines if l.get("type") == "pipeline"]
    assert "memory_prepare" not in pipeline_events
    assert "memory_commit" not in pipeline_events
    assert "done" in {l["type"] for l in lines}


def test_research_flow_receives_conversation_id(monkeypatch):
    calls, mem = _stub_flow(monkeypatch)
    client = TestClient(api_module.app)
    _stream_lines(client, {"question": "q", "conversation_id": "conv_fe_456"})
    assert calls == [("q", "CTX")]
    assert mem.prepare_calls == [("q", "conv_fe_456")]
    assert mem.record_calls
    assert mem.record_calls[0][1] == "rs_api"
    assert mem.record_calls[0][2] == "conv_fe_456"


def test_memory_api_disabled_by_env(monkeypatch):
    bridge_module._memory_api = None
    bridge_module._memory_failed = False
    monkeypatch.setenv("MEMORY_ENABLED", "0")
    assert bridge_module._get_memory_api() is None
    bridge_module._memory_failed = False


def test_memory_api_builds_in_memory_backend(monkeypatch):
    bridge_module._memory_api = None
    bridge_module._memory_failed = False
    monkeypatch.setenv("MEMORY_BACKEND", "memory")
    api = bridge_module._get_memory_api()
    assert api is not None
    assert isinstance(api.store, InMemoryMemoryStore)
    bridge_module._memory_api = None
    bridge_module._memory_failed = False
