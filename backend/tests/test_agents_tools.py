"""x_deepagents tool adapters + registry tests (offline, no LLM).

Covers:
  * the tool registry wires retrieve/umls_lookup/postgres_search (all live);
  * tool objects are real LangChain tools (name, description, .args);
  * postgres_search degrades cleanly when postgres is unreachable;
  * the tool set attaches to a create_deep_agent.
"""

from __future__ import annotations

import json

import pytest


def test_registry_wires_expected_tools():
    from src.agents.tools import all_tools, get, selectable, tools_for_agent

    tools = all_tools()
    assert {"retrieve", "umls_lookup", "postgres_search"} <= set(tools)

    # retrieve + umls + postgres_search are ALL live and selectable
    for name in ("retrieve", "umls_lookup", "postgres_search"):
        assert tools[name]["enabled"] is True, name
        assert tools[name]["future"] is False, name
        assert name in selectable(), name

    # only enabled, implemented tools are handed to the agent
    names = {getattr(t, "name", None) for t in tools_for_agent()}
    assert "postgres_search" in names
    assert "retrieve" in names
    assert "umls_lookup" in names


def test_langchain_tool_shape():
    from src.agents.tools.postgres_search import postgres_search
    from src.agents.tools.retrieve import retrieve
    from src.agents.tools.umls import umls_lookup

    for t in (retrieve, umls_lookup, postgres_search):
        assert getattr(t, "name", "")       # each tool has a name
        assert getattr(t, "description", "")  # and a description for the LLM
        assert hasattr(t, "args")           # langchain @tool exposes .args


def test_postgres_search_degrades_when_pg_unreachable(monkeypatch):
    """No live postgres -> available:false, clean note (never throws)."""
    import asyncio

    from src.agents.tools.postgres_search import postgres_search, reset_pg_store

    reset_pg_store()
    monkeypatch.setenv("PGHOST", "127.0.0.1")
    monkeypatch.setenv("PGPORT", "1")        # nothing listens here
    monkeypatch.setenv("PGDATABASE", "medrag")
    out = asyncio.run(postgres_search.ainvoke(
        {"requirement_id": "R1", "query": "hypertension", "top_k": 5}))
    data = json.loads(out)
    assert data["available"] is False
    assert data["results"] == []
    reset_pg_store()


def test_postgres_search_reports_empty_corpus(monkeypatch):
    """A reachable but EMPTY medrag schema -> empty_corpus:true so the agent
    knows the gap is the DB, not the query."""
    import asyncio

    from src.agents.tools import postgres_search as ps_mod
    from src.agents.tools.postgres_search import postgres_search, reset_pg_store

    class FakeStore:
        def connect(self):
            return object()

        def count_chunks(self):
            return 0

        def count_embeddings(self):
            return 0

        def count_documents(self):
            return 0

        def close(self):
            pass

    reset_pg_store()
    ps_mod._pg_store = FakeStore()
    try:
        out = asyncio.run(postgres_search.ainvoke(
            {"requirement_id": "R1", "query": "hypertension", "top_k": 5}))
    finally:
        reset_pg_store()
    data = json.loads(out)
    assert data["available"] is True
    assert data["empty_corpus"] is True
    assert data["stats"]["chunks"] == 0


def test_tool_set_attaches_to_deep_agent():
    from deepagents import create_deep_agent
    from langchain_core.messages import HumanMessage

    from src.agents.tools import tools_for_agent
    # construct an agent with the registry tools (no model call happens here)
    try:
        from src.agents.config import build_chat_model
        agent = create_deep_agent(
            model=build_chat_model(),
            system_prompt="test",
            tools=tools_for_agent(),
        )
    except Exception as exc:  # model build may need provider config
        pytest.skip(f"model build unavailable: {exc}")
    nodes = set(agent.get_graph().nodes)
    assert "tools" in nodes
