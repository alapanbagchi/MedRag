"""Agentic v3 — per-task UMLS entity enrichment.

Given a task's extracted entities, resolve each against UMLS/MeSH to obtain
its preferred name and synonyms, and fold those terms back into the
terminology pool the worker searches with.

This is a plain async function — no LLM — so it can be both:
  * called deterministically during the enrich step, and
  * exposed later as an LLM tool the agent may invoke at will.

Design notes:
  * Reuses src.umls.client.UMLSClient (cached, async, injectable transport).
  * Degrades gracefully: no API key / network failure -> entities unchanged.
    Enrichment must never break retrieval.
  * Dedupes synonyms, drops the echo of the surface form, drops MeSH-inverted
    forms (those containing ',') and over-long terms (> 40 chars) that would
    poison BM25.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from src.retrieval.plans import SubQueryPlan

logger = logging.getLogger("src.agents.umls_enricher")

# generic/common words and single-letter noise that must never be folded into a
# retrieval query (they would poison BM25 with off-domain dictionary matches).
_GENERIC_TERMS = {
    "site", "sites", "active", "reactive", "social", "stratification",
    "risk", "access", "management", "prevention", "use", "used", "using",
    "treatment", "the", "with", "without", "versus", "against",
}

# terms that look like drug/product codes ("AKI 001", "AKI-001", "PENK-123")
_CODE_TERM_RE = re.compile(r"\b[a-z]{1,5}[- ]?\d+\b", re.IGNORECASE)
_INVERTED_RE = re.compile(r"(syndrome|disease|disorder|spasm|occlusion),", re.IGNORECASE)


def _usable_term(name: str, surface_form: str = "") -> bool:
    """True if a UMLS name is worth folding into a retrieval query.

    Rejects: empties, over-long terms, MeSH-inverted forms, code-like terms
    (e.g. "AKI 001"), generic dictionary words, and exact echoes of the
    surface form.
    """
    name = " ".join(name.split()) if name else ""
    if not name:
        return False
    low = name.casefold()
    if len(name) < 3 or len(name) > 40:
        return False
    if "," in name or _INVERTED_RE.search(name):
        return False
    if low in _GENERIC_TERMS:
        return False
    # a phrase made entirely of generic words ("Active Site", "Social
    # Stratification") is off-domain — reject it too.
    words = low.split()
    if words and all(w in _GENERIC_TERMS for w in words):
        return False
    if _CODE_TERM_RE.search(name) and not any(ch.isalpha() and ch.islower() for ch in name):
        # bare "CODE 123" style (all-caps + digits, no real lowercase word)
        return False
    if surface_form and low == surface_form.casefold():
        return False
    # a term that is ONLY the surface acronym followed by digits is a code
    if surface_form and low.startswith(surface_form.casefold()) and re.search(r"\d", low):
        return False
    return True


@dataclass
class EnrichedTerm:
    surface_form: str
    preferred_name: Optional[str] = None
    cui: str = ""
    synonyms: List[str] = field(default_factory=list)


@dataclass
class EnrichmentResult:
    subquery_id: str
    terms: List[EnrichedTerm] = field(default_factory=list)

    @property
    def flat_terms(self) -> List[str]:
        """All candidate terms (preferred + synonyms), deduped and cleaned."""
        seen: set = set()
        out: List[str] = []
        for t in self.terms:
            for name in ([t.preferred_name] if t.preferred_name else []) + t.synonyms:
                if not _usable_term(name, t.surface_form):
                    continue
                name = " ".join(name.split())
                key = name.casefold()
                if key in seen:
                    continue
                seen.add(key)
                out.append(name)
                if len(out) >= 8:
                    return out
        return out


class UMLSEnricher:
    """Resolves task entities against UMLS and folds terms into the pool."""

    def __init__(self, config: Any = None, umls_client: Any = None):
        from src.config import AppConfig

        self.config = config or AppConfig()
        self.umls = umls_client
        if self.umls is None:
            from src.umls.client import UMLSClient

            self.umls = UMLSClient(
                api_key=self.config.umls_api_key,
                base_url=self.config.umls_base_url,
                sabs=self.config.umls_sabs,
                max_synonyms=self.config.umls_max_synonyms,
                timeout=self.config.umls_timeout,
            )

    @property
    def enabled(self) -> bool:
        return bool(getattr(self.umls, "enabled", False))

    async def enrich_subquery(self, sub: SubQueryPlan) -> List[EnrichedTerm]:
        """Resolve a subquery's entities; returns enriched terms (empty on failure)."""
        if not self.enabled:
            return []
        entities = [e for e in sub.entities if e.text]
        if not entities:
            return []
        terms: List[EnrichedTerm] = []
        for e in entities:
            try:
                concept = await self.umls.search_concept(e.text)
            except Exception as exc:  # network hiccup must not break the step
                logger.warning("UMLS search failed for %r: %s", e.text, exc)
                continue
            if not concept.found:
                continue
            terms.append(EnrichedTerm(
                surface_form=e.text,
                preferred_name=concept.preferred_name,
                cui=concept.cui,
                synonyms=list(concept.synonyms or []),
            ))
        return terms

    async def enrich(self, subs: List[SubQueryPlan]) -> Dict[str, List[EnrichedTerm]]:
        """Enrich every subquery (concurrently); keyed by subquery id."""
        results = await asyncio_gather_results(
            [(sub.id, self.enrich_subquery(sub)) for sub in subs]
        )
        return dict(results)

    def apply_to_query(self, sub: SubQueryPlan, terms: List[EnrichedTerm]) -> str:
        """Fold enriched terms into the subquery's retrieval query.

        Returns the expanded query and (side-effect) stores the additions back
        onto the subquery so later rounds can reuse them. Capped so BM25 stays
        meaningful.
        """
        base = (sub.query or sub.target or "").strip()
        additions: List[str] = []
        base_cf = f" {base.casefold()} "
        seen: set = set()
        for t in terms:
            for name in ([t.preferred_name] if t.preferred_name else []) + t.synonyms:
                if not _usable_term(name, t.surface_form):
                    continue
                name = " ".join(name.split())
                low = name.casefold()
                if low in seen:
                    continue
                if f" {low} " in base_cf:
                    continue
                seen.add(low)
                additions.append(name)
                if len(additions) >= 5:
                    break
            if len(additions) >= 5:
                break
        if additions:
            sub.synonyms = additions  # keep for later rounds / agent tooling
            return f"{base} {' '.join(additions)}"
        return base


async def asyncio_gather_results(pairs: List[tuple]):
    """Gather (key, awaitable) pairs, swallowing per-item exceptions."""
    import asyncio

    results = []
    for key, coro in pairs:
        try:
            val = await coro
            results.append((key, val))
        except Exception as exc:
            logger.warning("enrichment failed for %r: %s", key, exc)
            results.append((key, []))
    return results