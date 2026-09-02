"""UMLS terminology tool adapter (reuses src.tools.umls.UMLSEnricher)."""

from __future__ import annotations

import json
from typing import Optional

from langchain_core.tools import tool

from src.x_deepagents.reuse import get_umls_enricher
from src.retrieval.plans import PlannedEntity, SubQueryPlan


@tool
async def umls_lookup(
    requirement_id: str,
    query: str,
    entities: list[str],
) -> str:
    """Resolve medical entities against UMLS/MeSH.

    For each entity, returns its preferred name, CUI, and synonyms plus an
    enriched_query with the useful terms folded in. Use this to broaden the
    vocabulary of a search query (e.g. 'hypertension' -> 'high blood
    pressure', 'arterial hypertension').
    """
    from src.x_deepagents.logging import xdeep_log

    xdeep_log("umls_lookup_started", requirement_id=requirement_id,
              query=query, entities=entities)
    enricher = get_umls_enricher()
    if not enricher.enabled:
        return json.dumps({
            "requirement_id": requirement_id,
            "enriched_query": query,
            "terms": [],
            "note": "UMLS unavailable (no key / network) - query unchanged",
        })

    sub = SubQueryPlan(
        id=requirement_id,
        target=query,
        query=query,
        intent="evidence",
        focus="evidence",
        evidence_required=[query],
        entities=[PlannedEntity(text=e) for e in (entities or [])],
    )
    terms = await enricher.enrich_subquery(sub)
    enriched = enricher.apply_to_query(sub, terms)

    return json.dumps({
        "requirement_id": requirement_id,
        "enriched_query": enriched,
        "terms": [
            {
                "surface_form": t.surface_form,
                "preferred_name": t.preferred_name,
                "cui": t.cui,
                "synonyms": t.synonyms,
            }
            for t in terms
        ],
    }, ensure_ascii=False)


__all__ = ["umls_lookup"]
