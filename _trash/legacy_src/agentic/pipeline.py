"""End-to-end agentic pipeline: decompose -> (enrich -> loop) per subquery -> answer.

This ties the five steps together:

  1. DecomposePlanner.plan(query)          -> subqueries (+ intent/entities)
  2. UMLSEnricher.enrich_subquery(...)     -> fold synonyms into each query
  3. AgenticLoop.run(sub, base_query)      -> per-subquery retrieval loop
       (inside: hybrid retrieve -> restore paragraph -> verify -> re-search;
        retriever + umls exposed as tools the LLM drives autonomously)
  4. merge EvidenceReports across subqueries

It returns a plain dict so it can be consumed by a CLI or an API without
depending on the legacy Synthesizer. A lightweight final synthesis over the
collected evidence can be layered on in a later step.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict, List, Optional

from src.agentic.planner import DecomposePlanner, Decomposition, SubQueryPlan
from src.agentic.umls_tool import UMLSEnricher
from src.agentic.loop import AgenticLoop, EvidenceReport

logger = logging.getLogger("src.agentic.pipeline")


class AgenticPipeline:
    """Runs the whole agentic retrieval pipeline for one question."""

    def __init__(self, config: Any = None):
        from src.config import AppConfig
        from src.llm import build_model

        self.config = config or AppConfig()
        model = build_model(self.config)
        self.planner = DecomposePlanner(config=self.config)
        self.enricher = UMLSEnricher(config=self.config)
        self.loop = AgenticLoop(config=self.config, model=model,
                                umls_enricher=self.enricher)

    async def enrich_all(self, decomposition: Decomposition) -> None:
        """Fold UMLS terms into every subquery before the loops run."""
        from src.trace import get_trace
        trace = get_trace()
        for sub in decomposition.subqueries:
            trace.bullet(f"enriching {sub.id}: entities={[e.text for e in sub.entities]}")
            try:
                terms = await self.enricher.enrich_subquery(sub)
                sub.enriched_query = self.enricher.apply_to_query(sub, terms)
                trace.bullet(f"  {sub.id} enriched_query={sub.enriched_query!r}")
            except Exception as exc:
                logger.warning("enrich failed for %s: %s", sub.id, exc)
                trace.bullet(f"  {sub.id} enrich FAILED: {exc}")
                sub.enriched_query = sub.query

    async def answer(self, query: str) -> Dict[str, Any]:
        """Decompose -> enrich -> per-subquery agentic loop -> merged report."""
        from src.trace import get_trace
        trace = get_trace()

        trace.stage("STAGE 1/3 - DECOMPOSE")
        decomposition = await self.planner.plan(query)
        trace.bullet(f"decomposition: question_type={decomposition.question_type} subqueries={[s.id for s in decomposition.subqueries]}")
        for s in decomposition.subqueries:
            trace.bullet(f"  {s.id}: target={s.target!r} intent={s.intent!r} evidence={s.evidence_required} entities={[e.text for e in s.entities]}")

        trace.stage("STAGE 2/3 - UMLS ENRICH")
        await self.enrich_all(decomposition)

        reports: List[EvidenceReport] = []
        trace.stage(f"STAGE 3/3 - AGENTIC LOOP ({len(decomposition.subqueries)} subqueries)")
        # run subquery loops sequentially (each is LLM-expensive; parallel would
        # multiply rate-limit pressure). Callers can parallelize if desired.
        for i, sub in enumerate(decomposition.subqueries, start=1):
            trace.bullet(f"loop [{i}/{len(decomposition.subqueries)}] {sub.id}: {sub.target!r}")
            try:
                report = await self.loop.run(sub, base_query=query)
            except Exception as exc:
                logger.warning("loop failed for %s: %s", sub.id, exc)
                trace.bullet(f"  {sub.id} loop FAILED: {exc}")
                report = EvidenceReport(
                    subquery_id=sub.id, succeeded=False,
                    summary=f"loop error: {str(exc)[:200]}",
                    notes=[str(exc)[:200]],
                )
            reports.append(report)
            trace.bullet(f"  {sub.id} result: succeeded={report.succeeded} searches={len(report.searches_performed)} citations={report.citations}")

        trace.stage("DONE - MERGE")
        merged = self._merge(decomposition, reports)
        trace.bullet(f"subqueries succeeded: {merged['succeeded_subqueries']}/{merged['num_subqueries']}")
        trace.bullet(f"citations: {merged['citations']}")
        return merged

    @staticmethod
    def _merge(decomposition: Decomposition, reports: List[EvidenceReport]) -> Dict[str, Any]:
        by_id = {r.subquery_id: r for r in reports}
        subquery_summaries = []
        all_citations: List[str] = []
        all_excerpts: List[str] = []
        succeeded = 0
        for sub in decomposition.subqueries:
            r = by_id.get(sub.id)
            if r is None:
                continue
            if r.succeeded:
                succeeded += 1
            all_citations.extend(r.citations)
            all_excerpts.extend(r.evidence_excerpts)
            subquery_summaries.append({
                "id": sub.id,
                "target": sub.target,
                "intent": sub.intent,
                "succeeded": r.succeeded,
                "summary": r.summary,
                "searches": r.searches_performed,
                "citations": r.citations,
            })
        return {
            "question_type": decomposition.question_type,
            "num_subqueries": len(decomposition.subqueries),
            "subqueries": subquery_summaries,
            "succeeded_subqueries": succeeded,
            "citations": list(dict.fromkeys(all_citations)),
            "evidence_excerpts": all_excerpts,
        }
