"""Retriever tool: hybrid_search over BM25+dense via the shared service."""

from __future__ import annotations

import asyncio
import logging
from typing import List, Optional

logger = logging.getLogger("src.retriever_agent")


async def hybrid_search(
    query: str, top_k: int = 5, exclude_chunk_ids: Optional[List[str]] = None
) -> dict:
    """Hybrid retrieval tool: BM25 + dense, returns top documents.

    Uses the process-wide shared RetrievalService (index/corpus load once per
    process) and caches by query text; ``exclude_chunk_ids`` lets re-search
    rounds request fresh candidates.
    """
    from src.trace import get_trace

    trace = get_trace()
    print(f"  🔍 hybrid_search: {query[:70]}")
    trace.tool("hybrid_search", {"query": query, "top_k": top_k,
                                 "exclude": len(exclude_chunk_ids or [])})
    try:
        from src.config import AppConfig
        from src.retrieval.retriever import get_retrieval_service
        from src.agents.planner import SubQuery

        cfg = AppConfig()
        retriever = get_retrieval_service(cfg)
        sub = SubQuery(id="H1", target=query, focus="evidence", query=query)
        docs = await retriever.search_subquery(sub, exclude_chunk_ids=exclude_chunk_ids)

        papers = [
            {
                "id": doc.chunk_id or doc.document_id,
                "title": doc.document_id,
                "text": (doc.text or "")[:600],
                "score": doc.rrf_score,
                "section": doc.section or "",
            }
            for doc in docs[:top_k]
        ]
        for i, p in enumerate(papers, 1):
            trace.bullet(f"hybrid #{i}: {p['id']} score={p['score']:.4f} section={p['section']!r}")
        return {"papers": papers, "query": query, "total_found": len(papers)}
    except Exception as exc:
        logger.warning("hybrid_search failed: %s", exc)
        trace.tool("hybrid_search", {"query": query, "top_k": top_k}, {"error": str(exc)})
        return {"papers": [], "query": query, "total_found": 0}


if __name__ == "__main__":
    async def _run():
        data = await hybrid_search("risk factors and precipitating causes of COPD exacerbation", top_k=5)
        print(f"Found {data['total_found']} docs:")
        for p in data["papers"]:
            print(f"  - {p['title']} (score={p['score']:.3f}) section={p['section']}")

    asyncio.run(_run())
