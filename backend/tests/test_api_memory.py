"""Memory + Context layer — API wiring tests (backend/api.py).

Verifies the frontend contract: the chat stream emits first-class
``memory`` events (prepare before the run, commit after), the events are not
duplicated into the thinking-log trace, and the pipeline receives the
frontend's conversation_id for memory persistence. No LLM / no live DB:
the pipeline class and the memory singleton are faked/stubbed.
"""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

import api as api_module
from src.memory.api import MemoryAPI
from src.memory.config import MemoryConfig
from src.memory.store import InMemoryMemoryStore


class FakePipeline:
    """Stands in for AgenticV3Pipeline: emits memory prepare/commit events
    through the same V3Events emitter and returns a canned result."""

    def __init__(self, config=None, events=None, memory=None):
        from src.agentic.events import V3Events
        self.events = V3Events(events) if events is not None else None
        self.memory = memory
        self.answer_calls: list[tuple[str, str]] = []

    async def answer(self, query: str, conversation_id: str = ""):
        self.answer_calls.append((query, conversation_id))
        if self.events is not None:
            if self.memory is not None:  # pipeline-level memory events
                self.events.memory_prepare(
                    session_id="rs_api", session_title="hypertension",
                    prior_claims=2, prior_contradictions=1, prior_gaps=1)
                self.events.memory_commit(
                    session_id="rs_api",
                    stats={"claims_committed": 2, "claims_deduped": 0,
                           "contradictions": 1, "gaps": 1, "questions": 1})
        return {"answer": {"summary": "ok"}}


@pytest.fixture(autouse=True)
def _stub_pipeline_and_memory(monkeypatch):
    """Point chat_stream's pipeline at the fake and give it a real in-memory
    MemoryAPI, so the full memory event path is exercised."""
    from src.agentic import pipeline as pipeline_module

    class _Fake:
        instance = None

        def __init__(self, *a, **k):
            self._inst = FakePipeline(*a, **k)
            _Fake.instance = self._inst

        async def answer(self, *a, **k):
            return await self._inst.answer(*a, **k)

    monkeypatch.setattr(pipeline_module, "AgenticV3Pipeline", _Fake)
    monkeypatch.setattr(api_module, "_memory_api", MemoryAPI(
        store=InMemoryMemoryStore(), config=MemoryConfig(backend="memory")))
    yield
    api_module._memory_api = None


def _stream_lines(client: TestClient, body: dict) -> list[dict]:
    with client.stream("POST", "/v1/chat/stream", json=body) as resp:
        assert resp.status_code == 200
        lines = []
        for raw in resp.iter_lines():
            raw = raw.strip()
            if raw:
                lines.append(json.loads(raw))
    return lines


def test_chat_stream_emits_memory_prepare_and_commit():
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
    assert {"done", "token", "pipeline"} & {l["type"] for l in lines}


def test_pipeline_receives_conversation_id():
    client = TestClient(api_module.app)
    _stream_lines(client, {"question": "q", "conversation_id": "conv_fe_456"})
    from src.agentic import pipeline as pipeline_module
    inst = pipeline_module.AgenticV3Pipeline.instance
    assert ("q", "conv_fe_456") in inst.answer_calls


def test_memory_api_disabled_by_env(monkeypatch):
    api_module._memory_api = None           # clear the singleton first
    monkeypatch.setenv("MEMORY_ENABLED", "0")
    assert api_module.get_memory_api() is None


def test_memory_api_builds_in_memory_backend(monkeypatch):
    api_module._memory_api = None           # clear the singleton first
    monkeypatch.setenv("MEMORY_BACKEND", "memory")
    api = api_module.get_memory_api()
    assert api is not None
    assert isinstance(api.store, InMemoryMemoryStore)
    api_module._memory_api = None