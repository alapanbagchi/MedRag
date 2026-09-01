"""UMLS terminology client."""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict, List, Optional

import httpx
from pydantic import BaseModel, Field

from src.retrieval.plans import ClinicalEntity

logger = logging.getLogger("src.umls")


class UMLSConcept(BaseModel):
    term: str
    found: bool = False
    cui: str = ""
    preferred_name: Optional[str] = None
    synonyms: List[str] = Field(default_factory=list)
    semantic_types: List[str] = Field(default_factory=list)
    source: str = "MSH"


def serialize_umls_context(entities: List["ClinicalEntity"]) -> str:
    """Render enriched entities as compact text for LLM prompts."""
    lines = []
    for e in entities:
        parts = [f"- {e.surface_form}"]
        if e.cui:
            parts.append(f"CUI={e.cui}")
        if e.preferred_name:
            parts.append(f"preferred={e.preferred_name}")
        if e.synonyms:
            lines.append(" ".join(parts) + " | synonyms: " + "; ".join(e.synonyms))
        else:
            lines.append(" ".join(parts) + " | Match: none")
    return "\n".join(lines)


class UMLSClient:
    def __init__(self, api_key: str, base_url: str = "https://uts-ws.nlm.nih.gov/rest",
                 sabs: str = "MSH", max_synonyms: int = 5, timeout: float = 30.0,
                 transport: Optional[Any] = None):
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.sabs = sabs
        self.max_synonyms = max_synonyms
        self.timeout = timeout
        self._transport = transport  # injectable httpx transport (tests)
        self._client: Optional[httpx.AsyncClient] = None
        self._cache: Dict[str, UMLSConcept] = {}
        self._lock = asyncio.Lock()

    @property
    def enabled(self) -> bool:
        return bool(self.api_key)

    async def enrich_entities(self, entities: List[ClinicalEntity]) -> List[ClinicalEntity]:
        if not entities:
            return entities
        return list(await asyncio.gather(*(self.enrich_entity(e) for e in entities)))

    async def enrich_entity(self, entity: ClinicalEntity) -> ClinicalEntity:
        if not self.enabled:
            return entity

        lookup_terms = [entity.base_concept or entity.surface_form]
        if entity.surface_form.lower() != (entity.base_concept or entity.surface_form).lower():
            lookup_terms.append(entity.surface_form)

        concept = None
        for term in dict.fromkeys(lookup_terms):
            candidate = await self.search_concept(term)
            if candidate.found:
                concept = candidate
                break

        if concept is None or not concept.found:
            return entity

        return entity.model_copy(update={
            "preferred_name": concept.preferred_name or entity.base_concept,
            "ontology": "UMLS" if concept.source == "MSH" else concept.source,
            "cui": concept.cui,
            "synonyms": list(concept.synonyms),
        })

    async def search_concept(self, term: str) -> UMLSConcept:
        term = " ".join((term or "").split()).strip()
        if not term:
            return UMLSConcept(term=term)

        async with self._lock:
            cached = self._cache.get(term.lower())
        if cached is not None:
            return cached

        concept = await self._search(term)
        async with self._lock:
            self._cache[term.lower()] = concept
        return concept

    async def _search(self, term: str) -> UMLSConcept:
        if not self.enabled:
            return UMLSConcept(term=term)
        if self._client is None or self._client.is_closed:
            kwargs = {"timeout": self.timeout}
            if self._transport is not None:
                kwargs["transport"] = self._transport
            self._client = httpx.AsyncClient(**kwargs)
        params = {"string": term, "apiKey": self.api_key, "sabs": self.sabs,
                  "returnIdType": "concept", "searchType": "words", "pageSize": 20}
        try:
            resp = await self._client.get(f"{self.base_url}/search/current", params=params)
            resp.raise_for_status()
            data = resp.json()
        except (httpx.HTTPError, ValueError) as exc:
            logger.warning("UMLS lookup failed for %r: %s", term, exc)
            return UMLSConcept(term=term)

        results = (data.get("result") or {}).get("results") or []
        if not results:
            return UMLSConcept(term=term)

        target = term.lower()
        exact = [r for r in results if " ".join(str(r.get("name", "")).split()).lower() == target]
        selected = exact[0] if exact else results[0]
        cui = " ".join(str(selected.get("ui", "")).split())
        preferred = " ".join(str(selected.get("name", "")).split()) or None
        semantic_types = [str(t) for t in (selected.get("semanticTypes") or [])]

        synonyms = []
        if cui:
            synonyms = await self._fetch_atoms(cui, preferred)

        return UMLSConcept(term=term, found=True, cui=cui, preferred_name=preferred,
                           synonyms=synonyms, semantic_types=semantic_types)

    async def _fetch_atoms(self, cui: str, preferred: Optional[str]) -> List[str]:
        if self._client is None or self._client.is_closed:
            kwargs = {"timeout": self.timeout}
            if self._transport is not None:
                kwargs["transport"] = self._transport
            self._client = httpx.AsyncClient(**kwargs)
        params = {"apiKey": self.api_key, "sabs": self.sabs, "language": "ENG", "pageSize": 100}
        try:
            resp = await self._client.get(f"{self.base_url}/content/current/CUI/{cui}/atoms", params=params)
            if not resp.is_success:
                return []
            atoms = resp.json().get("result") or []
        except (httpx.HTTPError, ValueError):
            return []

        synonyms = []
        seen = set()
        plow = " ".join((preferred or "").split()).lower()
        for atom in atoms:
            name = " ".join(str(atom.get("name", "")).split())
            if not name:
                continue
            key = name.lower()
            if plow and key == plow:
                continue
            if key in seen:
                continue
            seen.add(key)
            synonyms.append(name)
            if len(synonyms) >= self.max_synonyms:
                break
        return synonyms
