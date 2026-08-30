"""Agentic v3 - Stage 4: UMLS terminology enrichment (per Worker).

The Worker's FIRST retrieval step is medical-terminology enrichment: take the
task's concepts and build a terminology pool of concept -> surface variants
(spec section 6):

    Vitamin D
    |-- Vitamin D
    |-- Cholecalciferol
    |-- Ergocalciferol
    |-- 25-hydroxyvitamin D
    |-- 25(OH)D
    '-- Vitamin D status

    Hypertension
    |-- Hypertension
    |-- High blood pressure
    |-- Elevated blood pressure
    '-- Arterial hypertension

The pool becomes search-planning vocabulary (Stage 5) and the search-term
planner picks the most probable medical formulations from it.

Implementation reuses the maintained UMLS client through
src.agentic.umls_tool.UMLSEnricher (no modifications to agentic v1/v2).
Degrades gracefully: no API key / network failure leaves the task's own
entities as the pool.
"""

from __future__ import annotations

import logging
from typing import Any, List

from src.agentic_v3.state import ResearchTask, TermConcept

logger = logging.getLogger("src.agentic_v3.umls")


class TerminologyEnricher:
    """Builds a UMLS terminology pool for one research task."""

    def __init__(self, config: Any = None, enricher: Any = None):
        from src.config import AppConfig

        self.config = config or AppConfig()
        self._enricher = enricher

    def _umls_enricher(self) -> Any:
        """Lazily build the shared UMLSEnricher (src.agentic.umls_tool)."""
        if self._enricher is None:
            from src.agentic.umls_tool import UMLSEnricher

            self._enricher = UMLSEnricher(config=self.config)
        return self._enricher

    @property
    def enabled(self) -> bool:
        try:
            return bool(getattr(self._umls_enricher(), "enabled", False))
        except Exception:
            return False

    async def enrich(self, task: ResearchTask) -> List[TermConcept]:
        """Resolve the task's entities against UMLS/MeSH.

        Returns the terminology pool (empty list on any failure - entities
        alone are still usable for search). Populates 'task.terminology'.
        """
        enricher = self._umls_enricher()
        if not getattr(enricher, "enabled", False):
            # No UMLS available: the pool is just the planner's entities.
            pool = [TermConcept(surface_form=e) for e in task.entities if e]
            task.terminology = pool
            return pool

        entities = [e for e in task.entities if e]
        if not entities:
            task.terminology = []
            return []

        from src.agentic.planner import PlannedEntity, SubQueryPlan

        sub = SubQueryPlan(
            id=task.id,
            target=task.objective or task.title,
            intent=task.intent,
            query=task.objective or task.title,
            focus="evidence",
            evidence_required=[r.text for r in task.evidence_requirements],
            entities=[PlannedEntity(text=e) for e in entities],
        )
        pool: List[TermConcept] = []
        try:
            terms = await enricher.enrich_subquery(sub)
        except Exception as exc:
            logger.warning("UMLS enrichment failed for %s: %s", task.id, exc)
            terms = []
        for t in terms:
            pool.append(TermConcept(
                surface_form=t.surface_form,
                preferred_name=t.preferred_name,
                cui=t.cui,
                synonyms=list(t.synonyms or []),
            ))
        if pool:
            task.terminology = pool
        else:
            task.terminology = [TermConcept(surface_form=e) for e in entities]
        return task.terminology

    @staticmethod
    def pool_terms(task: ResearchTask) -> List[str]:
        """Flatten the terminology pool into deduped search vocabulary."""
        seen: set = set()
        out: List[str] = []
        for concept in task.terminology:
            for t in concept.all_terms():
                t = " ".join((t or "").split())
                if t and t.casefold() not in seen:
                    seen.add(t.casefold())
                    out.append(t)
        return out

    @staticmethod
    def pool_summary(task: ResearchTask) -> List[dict]:
        """A small JSON-safe view of the pool (for events / UI)."""
        return [
            {"surface_form": c.surface_form, "preferred_name": c.preferred_name,
             "synonyms": list(c.synonyms)}
            for c in task.terminology
        ]
