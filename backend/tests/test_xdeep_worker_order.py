"""Worker ordering tests (Gap A + Gap B): local-first, pg-primary, then
informed web gap-fill - all enforced by the worker code, no LLM calls.

Covers:
  * _local_retrieve prefers postgres (medrag.chunks) when available;
  * _local_retrieve falls back to the parquet hybrid when postgres is down;
  * postgres empty-corpus is surfaced (health) and falls back to hybrid;
  * _web_fill runs only when the requirement is unsatisfied and stops on
    budget exhaustion;
  * research_worker: web gap-fill only after local rounds, verified items
    aggregated, searches_used returned.
"""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import pytest

PAYLOAD = {
    "task_id": "R1",
    "text": "Does vitamin D lower blood pressure?",
    "entities": ["vitamin D", "blood pressure"],
    "target_n": 3,
    "budget_max_searches": 3,
    "budget_max_rounds": 2,
}


def _pg_json(available=True, empty=False, results=None, stats=None):
    return json.dumps({
        "requirement_id": "R1",
        "query": "q",
        "available": available,
        "empty_corpus": empty,
        "stats": stats or {"chunks": 10, "embeddings": 10, "documents": 2},
        "count": len(results or []),
        "results": results or [],
    })


# ---------------------------------------------------------------------------
# _local_retrieve
# ---------------------------------------------------------------------------

def test_local_retrieve_prefers_postgres(monkeypatch):
    """Postgres results win when available; method stamped pgfts+pgvector."""
    from src.x_deepagents.agents import worker as worker_mod

    async def fake_pg(**kwargs):
        return _pg_json(results=[{
            "rank": 1, "chunk_id": "c1", "document_id": "PMC1",
            "section": "Results", "unit_kind": "paragraph",
            "score": 0.9, "methods": ["pgfts", "pgvector"],
            "text": "Vitamin D supplementation lowered systolic BP by 4 mmHg.",
        }])

    monkeypatch.setattr("src.x_deepagents.tools.postgres_search.postgres_search_impl", fake_pg)
    from src.x_deepagents.agents.worker import _local_retrieve

    results, method, health = asyncio.run(_local_retrieve(PAYLOAD, "q", set()))
    assert method == "pgfts+pgvector"
    assert results[0]["text"].startswith("Vitamin D")
    assert health["available"] is True and health["empty_corpus"] is False


def test_local_retrieve_falls_back_when_pg_down(monkeypatch):
    """Postgres unreachable -> parquet hybrid still returns candidates."""
    from src.x_deepagents.agents import worker as worker_mod

    async def fake_pg(**kwargs):
        return _pg_json(available=False, results=[])

    class FakeRetriever:
        async def search(self, sub, top_k=5, exclude_chunk_ids=None, restore_paragraphs=True):
            return [SimpleNamespace(
                chunk_id="h1", document_id="PMC9", section="Abstract",
                unit_kind="paragraph", rrf_score=0.5, methods=["bm25", "dense"],
                paragraph_text="A meta-analysis found a small BP reduction.",
            )]

    monkeypatch.setattr("src.x_deepagents.tools.postgres_search.postgres_search_impl", fake_pg)
    monkeypatch.setattr("src.x_deepagents.reuse.get_hybrid_retriever",
                        lambda config=None: FakeRetriever())
    from src.x_deepagents.agents.worker import _local_retrieve

    results, method, health = asyncio.run(_local_retrieve(PAYLOAD, "q", set()))
    assert method == "hybrid:bm25+dense"
    assert results[0]["chunk_id"] == "h1"
    assert health["available"] is False


def test_local_retrieve_surfaces_empty_corpus(monkeypatch):
    """Empty medrag.chunks is surfaced in health (not silently empty)."""
    from src.x_deepagents.agents import worker as worker_mod

    async def fake_pg(**kwargs):
        return _pg_json(empty=True, results=[],
                        stats={"chunks": 0, "embeddings": 0, "documents": 0})

    class FakeRetriever:
        async def search(self, sub, top_k=5, exclude_chunk_ids=None, restore_paragraphs=True):
            return []

    monkeypatch.setattr("src.x_deepagents.tools.postgres_search.postgres_search_impl", fake_pg)
    monkeypatch.setattr("src.x_deepagents.reuse.get_hybrid_retriever",
                        lambda config=None: FakeRetriever())
    from src.x_deepagents.agents.worker import _local_retrieve

    results, method, health = asyncio.run(_local_retrieve(PAYLOAD, "q", set()))
    assert health["empty_corpus"] is True
    assert results == []  # hybrid also returned nothing -> clean empty


def test_local_retrieve_excludes_seen_chunks(monkeypatch):
    """exclude_chunk_ids is forwarded to postgres (loop avoidance)."""
    from src.x_deepagents.agents import worker as worker_mod

    captured = {}

    async def fake_pg(**kwargs):
        captured.update(kwargs)
        return _pg_json(results=[])

    monkeypatch.setattr("src.x_deepagents.tools.postgres_search.postgres_search_impl", fake_pg)
    from src.x_deepagents.agents.worker import _local_retrieve

    asyncio.run(_local_retrieve(PAYLOAD, "q", {"c1", "c2"}))
    assert set(captured.get("exclude_chunk_ids") or []) == {"c1", "c2"}


# ---------------------------------------------------------------------------
# _web_fill
# ---------------------------------------------------------------------------

def _web_json(available=True, results=None):
    return json.dumps({
        "query": "q", "available": available, "trusted_only": True,
        "count": len(results or []),
        "dropped_blocked": 2, "dropped_unverified": 1,
        "results": results or [],
    })


async def _fake_reliability(agent, url, text):
    return {"reliability": "high", "authority": "cdc", "evidence": "yes",
            "recency": "recent", "conflicts": "none", "note": "ok"}


async def _accepting_verifier(agent, requirement, item):
    from src.x_deepagents.state import AnswersTask, SupportDirection, VerifierVerdict

    item.submit_to_verifier()
    item.set_verdict(VerifierVerdict(
        evidence_id=item.id,
        requirement_id=item.requirement_id,
        relevance="relevant",
        answers_task=AnswersTask.YES,
        support=SupportDirection.SUPPORTS,
        confidence=0.9,
        note="answers the requirement",
    ))
    return item


def test_web_fill_called_only_when_unsatisfied(monkeypatch):
    """_web_fill invokes searxng and builds verified web candidates."""
    from src.x_deepagents.agents import worker as worker_mod
    from src.x_deepagents.state import ResearchRequirement, RunBudget

    calls = {"n": 0}

    async def fake_searxng(query, top_k=None, trusted_only=True):
        calls["n"] += 1
        return _web_json(results=[{
            "title": "WHO factsheet",
            "url": "https://www.who.int/news-room/fact-sheets/detail/hypertension",
            "snippet": "WHO fact sheet on hypertension management and blood "
                       "pressure targets in adults.",
            "engine": "google", "trust": "trusted",
        }])

    monkeypatch.setattr("src.x_deepagents.tools.searxng.searxng_search_impl", fake_searxng)
    monkeypatch.setattr("src.x_deepagents.agents.stages.verify_item", _accepting_verifier)
    monkeypatch.setattr("src.x_deepagents.agents.stages.judge_site_reliability", _fake_reliability)

    req = ResearchRequirement(id="R1", text=PAYLOAD["text"], target_n=2)
    budget = RunBudget(max_searches=3)
    asyncio.run(worker_mod._web_fill(PAYLOAD, req, ["q"], None, budget))

    assert calls["n"] == 1
    assert budget.searches_used == 1
    web_items = [i for i in req.items if i.retrieval_method.startswith("web:")]
    assert web_items, "web candidate must carry retrieval_method=web:<engine>"
    assert web_items[0].source_url == "https://www.who.int/news-room/fact-sheets/detail/hypertension"
    assert web_items[0].trust == "trusted"
    assert web_items[0].verified is True
    assert web_items[0].reliability == "high"    # reliability critic stamped it


def test_web_fill_stops_on_budget_exhaustion(monkeypatch):
    """Hard search budget stops the web loop (Rule 5)."""
    from src.x_deepagents.agents import worker as worker_mod
    from src.x_deepagents.state import ResearchRequirement, RunBudget

    calls = {"n": 0}

    async def fake_searxng(query, top_k=None, trusted_only=True):
        calls["n"] += 1
        return _web_json(results=[])

    monkeypatch.setattr("src.x_deepagents.tools.searxng.searxng_search_impl", fake_searxng)
    req = ResearchRequirement(id="R1", text=PAYLOAD["text"], target_n=5)
    budget = RunBudget(max_searches=0)     # already exhausted
    asyncio.run(worker_mod._web_fill(PAYLOAD, req, ["q1", "q2", "q3"], None, budget))
    assert calls["n"] == 0


def test_research_worker_web_only_after_local_and_returns_budget(monkeypatch):
    """Full worker: local rounds run first; web only fires when unsatisfied;
    searches_used is returned for the graph to aggregate."""
    from src.x_deepagents.agents import worker as worker_mod

    calls = {"local": 0, "web": 0}

    async def fake_plan(*a, **k):
        return ["new query one"]

    async def fake_replan(*a, **k):
        return ["new query one"] if calls["local"] == 2 else []

    async def fake_retrieve(payload, query, seen_chunks, top_k=5):
        calls["local"] += 1
        # local never satisfies (postgres + hybrid return nothing useful)
        return [], "pgfts+pgvector", {"available": True, "empty_corpus": False, "stats": {}}

    async def fake_web_impl(query, top_k=None, trusted_only=True):
        calls["web"] += 1
        return _web_json(results=[{
            "title": "T", "url": "https://www.cdc.gov/hypertension/tips.html",
            "snippet": "CDC guidance: DASH diet and sodium reduction lower blood pressure.",
            "engine": "google", "trust": "trusted",
        }])

    async def fake_deep_inspect(*a, **k):
        return []

    monkeypatch.setattr("src.x_deepagents.agents.stages.plan_queries", fake_plan)
    monkeypatch.setattr("src.x_deepagents.agents.stages.replan_queries", fake_replan)
    monkeypatch.setattr("src.x_deepagents.agents.worker._local_retrieve", fake_retrieve)
    monkeypatch.setattr("src.x_deepagents.tools.searxng.searxng_search_impl", fake_web_impl)
    monkeypatch.setattr("src.x_deepagents.agents.stages.verify_item", _accepting_verifier)
    monkeypatch.setattr("src.x_deepagents.agents.stages.deep_inspect", fake_deep_inspect)
    monkeypatch.setattr("src.x_deepagents.agents.stages.judge_site_reliability", _fake_reliability)

    out = asyncio.run(worker_mod.research_worker(dict(PAYLOAD)))
    req = out["requirements"][0]
    assert calls["local"] >= 1        # local ran
    assert calls["web"] == 1          # web fired only after local unsatisfied
    assert out["searches_used"] == 1  # budget aggregated for the graph
    web_items = [i for i in req.items if i.retrieval_method.startswith("web:")]
    assert web_items
    assert web_items[0].reliability == "high"
