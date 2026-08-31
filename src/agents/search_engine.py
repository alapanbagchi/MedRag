"""Agentic v3 — hybrid retrieval + rerank + paragraph restoration engine.

One call performs the whole retrieval leg of the agentic design:

    subquery plan
        -> shared hybrid retrieval (BM25 + dense, union)   [RetrievalService]
        -> intent rerank + paper diversification            [reranker]
        -> restore each retrieved CHUNK to its full         [StructuralUnitIndex]
           containing PARAGRAPH / TABLE / FIGURE
        -> ranked, deduped list of RetrievalResult objects

The tool exposes an already-seen exclusion set so workers can request FRESH
candidates on later search rounds without re-fetching the same chunks.

It is a plain async callable (no LLM), so it can be used deterministically
during the retrieval step AND (later) as an agent tool.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, List, Optional

from pydantic import BaseModel, Field

from src.retrieval.plans import SubQuery as LegacySubQuery
from src.retrieval.plans import SubQueryPlan

logger = logging.getLogger("src.agents.search_engine")


class RetrievalResult(BaseModel):
    rank: int = 0
    chunk_id: str = ""
    document_id: str = ""
    section: str = ""
    unit_kind: str = "paragraph"      # paragraph | table | figure
    rrf_score: float = 0.0
    methods: List[str] = Field(default_factory=list)
    # FULL containing structural unit (whole paragraph/table/figure), restored
    # from the chunk pointer — not a bare snippet.
    paragraph_text: str = ""
    token_count: int = 0


class HybridRetrieverTool:
    """Wraps the shared RetrievalService + StructuralUnitIndex into one tool."""

    def __init__(self, config: Any = None, service: Any = None):
        from src.config import AppConfig

        self.config = config or AppConfig()
        self.service = service

    # ------------------------------------------------------------------
    def _service(self):
        if self.service is None:
            from src.retrieval.retriever import get_retrieval_service
            self.service = get_retrieval_service(self.config)
        return self.service

    def _unit_index(self, corpus: Any):
        from src.retrieval.fullpaper import get_unit_index
        return get_unit_index(corpus)

    # ------------------------------------------------------------------
    def _to_legacy_subquery(self, sub: SubQueryPlan) -> LegacySubQuery:
        """Adapt a SubQueryPlan to the SubQuery the shared service expects."""
        return LegacySubQuery(
            id=sub.id,
            target=sub.target or sub.query,
            focus=sub.focus or "evidence",
            query=sub.query or sub.target,
            evidence_required=list(sub.evidence_required or []),
            terminology=[e.text for e in sub.entities],
        )

    async def search(
        self,
        sub: SubQueryPlan,
        top_k: Optional[int] = None,
        exclude_chunk_ids: Optional[List[str]] = None,
        restore_paragraphs: bool = True,
    ) -> List[RetrievalResult]:
        """Hybrid-retrieve + rerank for one subquery, restoring full units."""
        from src.lib.trace import get_trace

        trace = get_trace()
        service = self._service()
        legacy = self._to_legacy_subquery(sub)

        query_text = sub.query or sub.target
        trace.tool("hybrid_retrieve", {"query": query_text, "top_k": top_k,
                                       "exclude": len(exclude_chunk_ids or [])})
        try:
            docs = await service.search_subquery(
                legacy, exclude_chunk_ids=list(exclude_chunk_ids or [])
            )
        except Exception as exc:
            logger.warning("hybrid retrieval failed for %r: %s", sub.id, exc)
            return []

        k = top_k or self.config.max_documents
        docs = docs[:k]

        # Restore the full containing unit for every retrieved chunk pointer.
        if restore_paragraphs:
            docs = await self._restore_units(service, docs, query_text)

        results: List[RetrievalResult] = []
        for i, doc in enumerate(docs, start=1):
            text = getattr(doc, "paragraph_text", None) or getattr(doc, "text", "")
            results.append(RetrievalResult(
                rank=i,
                chunk_id=getattr(doc, "chunk_id", "") or "",
                document_id=getattr(doc, "document_id", "") or "",
                section=getattr(doc, "section", "") or "",
                unit_kind=getattr(doc, "unit_kind", None) or getattr(doc, "node_type", "paragraph"),
                rrf_score=float(getattr(doc, "rrf_score", 0.0) or 0.0),
                methods=[m for m in (getattr(doc, "methods", None) or [])],
                paragraph_text=text,
                token_count=len(text.split()),
            ))
        return results

    async def _restore_units(self, service: Any, docs: List[Any], query_text: str):
        """Attach the full containing structural unit to each retrieved doc.

        Runs the (blocking) StructuralUnitIndex lookups in the executor thread
        so we don't stall the event loop; dedupes chunks that restore to the
        same unit (e.g. several rows of one table).
        """
        import asyncio

        corpus = service._components()["corpus"]
        unit_index = self._unit_index(corpus)
        max_tokens = self.config.max_paper_tokens

        out: List[Any] = []
        seen_units: set = set()
        for doc in docs:
            cid = getattr(doc, "chunk_id", "")
            unit_text = ""
            kind = getattr(doc, "node_type", "") or "paragraph"
            try:
                unit_text = await asyncio.to_thread(unit_index.get, cid)
            except Exception:
                unit_text = ""
            if unit_text:
                try:
                    kind = await asyncio.to_thread(unit_index.unit_kind, cid)
                except Exception:
                    pass
            else:
                # fall back to the retrieved chunk text itself
                unit_text = getattr(doc, "text", "") or ""

            # dedupe: multiple chunks of the same table/figure restore to the
            # same unit text; keep the first (highest-ranked) occurrence.
            unit_key = (kind, unit_text[:200])
            if unit_key in seen_units:
                continue
            seen_units.add(unit_key)

            if max_tokens and len(unit_text.split()) > max_tokens:
                words = unit_text.split()
                unit_text = " ".join(words[:max_tokens]) + (
                    f"\n[...unit truncated to {max_tokens} tokens for context]\n"
                )
            setattr(doc, "paragraph_text", unit_text)
            setattr(doc, "unit_kind", kind)
            out.append(doc)
        return out


# ---------------------------------------------------------------------------
# Process-wide singleton (index/corpus load once); mirrors the shared service.
# ---------------------------------------------------------------------------

_TOOL_SINGLETON: Optional[HybridRetrieverTool] = None


def get_retriever_tool(config: Any = None) -> HybridRetrieverTool:
    global _TOOL_SINGLETON
    if _TOOL_SINGLETON is None:
        _TOOL_SINGLETON = HybridRetrieverTool(config)
    return _TOOL_SINGLETON


def reset_retriever_tool() -> None:
    global _TOOL_SINGLETON
    _TOOL_SINGLETON = None