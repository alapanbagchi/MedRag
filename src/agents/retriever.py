"""Agentic v3 - Stage 6 + 7: the paper retriever tool + context expansion.

The selected search terms are passed to the retriever, which searches the
medical literature and returns CANDIDATE papers with their relevant excerpts.

CRITICAL distinction (spec section 8):

    Retriever output = candidate evidence, NOT verified evidence.

Context expansion (spec section 9): a search result often yields only a small
excerpt ("Vitamin D supplementation was associated with..."). This stage
expands each hit to the FULL containing paragraph / table / figure before the
CRITIC ever sees it:

If a search result returns only a small excerpt, the system retrieves the
surrounding paragraph and evaluates THE EXPANDED CONTEXT - never a truncated
snippet. That is why every candidate here carries the full unit text.

The implementation reuses src.agentic.retriever_tool.HybridRetrieverTool
(BM25/SPLADE + dense hybrid, full-unit restoration via the StructuralUnitIndex)
and adapts the worker's evidence requirement into the SubQueryPlan
retrieval expects. Loop-avoidance: excluded already-seen chunk ids are passed
in so later rounds return FRESH candidates.
"""

from __future__ import annotations

import logging
from typing import Any, List, Optional

from src.retrieval.plans import SubQueryPlan
from src.agents.state import EvidenceRequirement, ResearchTask, RetrievedPaper

logger = logging.getLogger("src.agents.retriever")


class PaperRetrieverTool:
    """Searches the corpus for one evidence requirement and restores context."""

    def __init__(self, config: Any = None, tool: Any = None):
        from src.config import AppConfig

        self.config = config or AppConfig()
        self._tool = tool

    def _tool_obj(self) -> Any:
        if self._tool is None:
            from src.agents.search_engine import HybridRetrieverTool

            self._tool = HybridRetrieverTool(config=self.config)
        return self._tool

    async def search(
        self,
        task: ResearchTask,
        requirement: EvidenceRequirement,
        query: str,
        *,
        exclude_chunk_ids: Optional[List[str]] = None,
        top_k: Optional[int] = None,
        round_no: int = 0,
    ) -> List[RetrievedPaper]:
        """Run ONE query for ONE requirement; returns expanded-context papers.

        Each hit's FULL containing structural unit is restored (context
        expansion); nothing is judged here - the CRITIC does that.
        """
        from src.lib.trace import get_trace

        trace = get_trace()
        sub = SubQueryPlan(
            id=requirement.id,
            target=requirement.text or query,
            intent=task.intent,
            query=query,
            focus="evidence",
            evidence_required=[requirement.text],
            terminology=[c.all_terms()[0] if c.all_terms() else "" 
                         for c in getattr(task, "terminology", [])],
        )
        k = top_k or getattr(self.config, "agentic_v3_papers_per_search", None)             or self.config.max_documents
        trace.tool("v3_retrieve", {"query": query, "requirement": requirement.id,
                                   "top_k": k, "exclude": len(exclude_chunk_ids or [])})
        try:
            results = await self._tool_obj().search(
                sub, top_k=max(1, k), exclude_chunk_ids=list(exclude_chunk_ids or [])
            )
        except Exception as exc:
            logger.warning("retrieval failed for %s/%s (%s)", task.id,
                           requirement.id, exc)
            return []
        papers: List[RetrievedPaper] = []
        seen: set = set()
        for r in results:
            key = (r.document_id or "", r.chunk_id or "", r.paragraph_text[:80])
            if key in seen:
                continue
            seen.add(key)
            papers.append(RetrievedPaper(
                chunk_id=r.chunk_id,
                document_id=r.document_id,
                section=r.section,
                unit_kind=r.unit_kind,
                score=r.rrf_score,
                text=r.paragraph_text,          # EXPANDED context (full unit)
                source_query=query,
                round_no=round_no,
                rank=int(getattr(r, "rank", 0) or 0),
                retrieval_method="+".join(getattr(r, "methods", None) or ["hybrid"]),
            ))
        return papers
