"""x_deepagents scaffold smoke tests: stack imports, reuse boundary, graph.

Offline (no LLM calls, no network). Live-invocation is a manual run:
PYTHONPATH=. .venv/bin/python -m src.agents "<q>" --run
"""

from __future__ import annotations

import os


def test_stack_imports():
    import deepagents
    import langchain_core
    import langgraph
    import langchain_openai

    assert deepagents is not None
    assert langgraph is not None
    assert langchain_core is not None
    assert langchain_openai is not None


def test_reuse_boundary_resolves_backend():
    from src.agents.reuse import AppConfig, load_prompt

    cfg = AppConfig()
    assert cfg.provider in ("kaggle", "openai", "ollama", "vllm", "gemini", "mistral")
    prompt = load_prompt("agents", "critic.txt")
    assert prompt and len(prompt) > 20


def test_reused_tool_classes_importable():
    from src.agents.reuse import (
        get_hybrid_retriever,
        get_paper_retriever,
        get_terminology_enricher,
        get_umls_enricher,
    )

    assert get_hybrid_retriever() is not None
    assert get_paper_retriever() is not None
    assert get_terminology_enricher() is not None
    assert get_umls_enricher() is not None


def test_tool_registry_has_wired_tools():
    # non-mutating: the registry wires its tools at import. postgres_search
    # (pg-chunk retrieval over medrag.chunks) is LIVE and selectable.
    from src.agents.tools import all_tools, selectable

    tools = all_tools()
    assert {"retrieve", "umls_lookup", "postgres_search"} <= set(tools)
    assert "postgres_search" in selectable()
    assert "retrieve" in selectable()


def test_orchestrator_agent_constructs():
    from src.agents.agents.stages import make_orchestrator

    agent = make_orchestrator()
    # deepagents 0.7 returns a CompiledStateGraph (ainvoke/get_graph), and
    # its subgraph carries the model + tools loop nodes.
    for attr in ("ainvoke", "invoke", "get_graph"):
        assert hasattr(agent, attr), f"missing {attr}"
    nodes = set(agent.get_graph().nodes)
    assert {"model", "tools"} <= nodes, f"deep agent tool loop missing: {nodes}"


def test_research_graph_compiles():
    from src.agents.graph import build_research_graph

    graph = build_research_graph()
    assert graph is not None
    nodes = {n for n in graph.get_graph().nodes}
    # mandatory workflow stages present (orchestrator + parallel workers +
    # join + conflict + resolution + synthesis)
    assert {"decompose", "research_worker", "join", "conflict",
            "resolution", "synthesize"} <= nodes
