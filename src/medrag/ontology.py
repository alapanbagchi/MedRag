"""BioPortal ontology enrichment for biomedical concepts.

Queries the BioPortal REST API to find ontology terms related to extracted
concepts. Returns enriched term metadata (definitions, synonyms, semantic types).

BioPortal API docs: https://bioportal.bioontology.org/wiki/index.php/REST_api

Environment variables:
    BIOPORTAL_API_KEY   API key (optional — BioPortal allows limited free queries)
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

import requests


def _env(key: str, default: str = "") -> str:
    return os.environ.get(key, default)


class BioPortalEnricher:
    """Query BioPortal for ontology terms matching biomedical concepts."""

    BASE_URL = "https://bioportal.bioontology.org/api"

    def __init__(self, api_key: Optional[str] = None) -> None:
        self.api_key = api_key or _env("BIOPORTAL_API_KEY")
        self._session = requests.Session()

    def search(
        self,
        query: str,
        max_results: int = 5,
        require_exact_match: bool = False,
    ) -> List[Dict[str, Any]]:
        """Search BioPortal for ontology terms matching the query.

        Returns a list of dicts with:
            - concept_id    (e.g. "http://purl.obolibrary.org/obo/DOID_1234")
            - pref_label    (preferred term name)
            - semantic_type (e.g. "Disease or Syndrome")
            - ontology      (source ontology, e.g. "DOID", "MESH")
            - definition    (if available)
            - synonyms      (list of alternative names)
        """
        params: Dict[str, Any] = {
            "q": query,
            "require_exact_match": str(require_exact_match).lower(),
            "include_categories": "true",
            "pagesize": max_results,
            "display_context": "false",
            "include": "prefLabel,semanticType,definition,synonym",
        }
        if self.api_key:
            params["apikey"] = self.api_key

        try:
            resp = self._session.get(
                f"{self.BASE_URL}/search",
                params=params,
                timeout=10,
            )
            resp.raise_for_status()
            data = resp.json()
        except (requests.RequestException, ValueError):
            return []

        hits = data.get("collection", [])
        results: List[Dict[str, Any]] = []
        for hit in hits:
            result: Dict[str, Any] = {
                "concept_id": hit.get("@id", ""),
                "pref_label": hit.get("prefLabel", ""),
                "semantic_type": "",
                "ontology": "",
                "definition": hit.get("definition", [""])[0] if isinstance(hit.get("definition"), list) else hit.get("definition", ""),
                "synonyms": hit.get("synonym", []),
            }
            # Extract semantic type
            types = hit.get("semanticType", [])
            if types:
                result["semantic_type"] = types[0] if isinstance(types, list) else str(types)
            # Extract ontology name from concept ID
            concept_id = result["concept_id"]
            for prefix in ["DOID", "MESH", "HP", "MONDO", "NCIT", "CUI", "GO", "UBERON"]:
                if prefix in concept_id.upper():
                    result["ontology"] = prefix
                    break
            if not result["ontology"] and "/" in concept_id:
                result["ontology"] = concept_id.split("/")[-1].split("_")[0]

            results.append(result)

        return results

    def enrich_concepts(
        self,
        concepts: List[Dict[str, Any]],
        max_per_concept: int = 3,
    ) -> List[Dict[str, Any]]:
        """Enrich a list of extracted concepts with BioPortal ontology data.

        Each concept dict should have at least a 'name' key.
        Adds BioPortal metadata in place and returns the enriched list.
        """
        for concept in concepts:
            name = concept.get("name", "")
            if not name:
                continue
            bioportal_hits = self.search(name, max_results=max_per_concept)
            concept["bioportal"] = bioportal_hits

            # Collect synonyms from BioPortal
            all_synonyms: List[str] = []
            for hit in bioportal_hits:
                syns = hit.get("synonyms", [])
                if isinstance(syns, list):
                    all_synonyms.extend(syns)
            if all_synonyms:
                concept["synonyms"] = list(set(all_synonyms))[:10]

            # Collect semantic types
            types = [h.get("semantic_type", "") for h in bioportal_hits if h.get("semantic_type")]
            if types:
                concept["semantic_types"] = list(set(types))

        return concepts
