"""Tool adapters: retrieve (parquet hybrid) as a LangChain tool.

These wrap the REUSED non-LLM capabilities (src.tools.*) so a deepagents
agent can call them. Inputs/outputs are plain JSON so the LLM sees clean
tool contracts.

The point of the adapter layer: the agent freely CHOOSES these tools (autonomy
B), while the workflow order (verify after retrieve, etc.) stays in rules.py.
"""

from __future__ import annotations

import json
from typing import Optional

from langchain_core.tools import tool

from src.agents.reuse import get_hybrid_retriever
from src.retrieval.plans import SubQueryPlan


@tool
async def retrieve(
    requirement_id: str,
    query: str,
    entities: Optional[list[str]] = None,
    top_k: int = 5,
    exclude_chunk_ids: Optional[list[str]] = None,
) -> str:
    """Hybrid literature search (BM25 + dense + pgvector, full-unit restore).

    Retrieves candidate passages for ONE evidence requirement and returns
    each hit's expanded context (the whole containing paragraph/table/figure),
    ranked by a fusion score. Returns JSON with rank, chunk_id, document_id,
    section, unit_kind, score and the full text.

    These are CANDIDATES - they are NOT verified. Every returned chunk must
    still pass the verifier before it can be treated as evidence.
    """
    from src.agents.logging import xdeep_log

    retriever = get_hybrid_retriever()
    sub = SubQueryPlan(
        id=requirement_id,
        target=query,
        intent="evidence",
        query=query,
        focus="evidence",
        evidence_required=[query],
        entities=[__import__("src.retrieval.plans", fromlist=["PlannedEntity"]).PlannedEntity(text=e)
                  for e in (entities or [])],
    )
    xdeep_log("corpus_retrieve_started", requirement_id=requirement_id,
              query=query, top_k=top_k)
    try:
        results = await retriever.search(
            sub,
            top_k=top_k,
            exclude_chunk_ids=list(exclude_chunk_ids or []),
            restore_paragraphs=True,
        )
    except Exception as exc:  # retrieval must never crash the agent loop
        return json.dumps({"error": str(exc)[:300], "results": []})

    out = []
    for r in results:
        out.append({
            "rank": r.rank,
            "chunk_id": r.chunk_id,
            "document_id": r.document_id,
            "section": r.section,
            "unit_kind": r.unit_kind,
            "score": round(float(getattr(r, "rrf_score", 0.0) or 0.0), 4),
            "methods": getattr(r, "methods", None) or [],
            "text": r.paragraph_text,
        })
    return json.dumps({"requirement_id": requirement_id, "results": out}, ensure_ascii=False)


__all__ = ["retrieve"]
