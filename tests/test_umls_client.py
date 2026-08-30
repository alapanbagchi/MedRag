"""UMLS client tests: local terminology enrichment (ported from the notebook),
with the HTTP layer mocked."""

from __future__ import annotations

import httpx
import pytest

from src.agents.planner import ClinicalEntity
from src.umls.client import UMLSClient, serialize_umls_context


def _handler(counter: dict, *, fail_search: bool = False):
    def handle(request: httpx.Request) -> httpx.Response:
        counter["requests"] = counter.get("requests", 0) + 1
        path = request.url.path
        if fail_search:
            return httpx.Response(500, json={"error": "boom"})
        if path.endswith("/search/current"):
            return httpx.Response(
                200,
                json={
                    "result": {
                        "results": [
                            {
                                "ui": "C2930803",
                                "name": "Coarctation of aorta dominant",
                                "semanticTypes": ["Disease or Syndrome"],
                            },
                            {
                                "ui": "C9999999",
                                "name": "coarctation of the aorta",
                                "semanticTypes": ["Disease or Syndrome"],
                            },
                        ]
                    }
                },
            )
        if "/atoms" in path:
            return httpx.Response(
                200,
                json={
                    "result": [
                        {"name": "Aorta Dominant Coarctation"},
                        {"name": "Coarctation of aorta dominant"},  # != preferred -> kept
                        {"name": "coarctation of the aorta"},        # == preferred -> skipped
                        {"name": "Aorta Dominant Coarctations"},
                        {"name": "recurrent coarctation"},
                        {"name": "aorta coarctation"},
                        {"name": "coarctation"},
                    ]
                },
            )
        return httpx.Response(404, json={"error": "not found"})

    return handle


def _client(counter: dict, **kw) -> UMLSClient:
    return UMLSClient(
        api_key="test-key",
        base_url="https://uts-ws.nlm.nih.gov/rest",
        sabs="MSH",
        max_synonyms=3,
        timeout=5,
        transport=httpx.MockTransport(_handler(counter, **kw)),
    )


@pytest.mark.asyncio
async def test_search_selects_exact_match_and_synonyms():
    counter: dict = {}
    client = _client(counter)
    concept = await client.search_concept("coarctation of the aorta")

    assert concept.found is True
    # exact lexical match on name wins over the first result
    assert concept.cui == "C9999999"
    assert concept.preferred_name == "coarctation of the aorta"
    assert concept.source == "MSH"
    assert "Disease or Syndrome" in concept.semantic_types
    # atom equal to the preferred name is skipped; capped at max_synonyms
    assert not any(s.lower() == "coarctation of the aorta" for s in concept.synonyms)
    assert len(concept.synonyms) <= 3
    assert concept.synonyms[0] == "Aorta Dominant Coarctation"


@pytest.mark.asyncio
async def test_search_is_cached():
    counter: dict = {}
    client = _client(counter)
    await client.search_concept("diabetes")
    await client.search_concept("diabetes")
    await client.search_concept("diabetes")
    assert counter["requests"] == 2  # one search + one atoms fetch, cached after


@pytest.mark.asyncio
async def test_fail_soft_on_error():
    counter: dict = {}
    client = _client(counter, fail_search=True)
    concept = await client.search_concept("anything")
    assert concept.found is False
    assert concept.cui == ""


@pytest.mark.asyncio
async def test_disabled_without_key():
    client = UMLSClient(api_key="")
    concept = await client.search_concept("diabetes")
    assert concept.found is False


@pytest.mark.asyncio
async def test_enrich_entity_preserves_surface_form():
    counter: dict = {}
    client = _client(counter)
    entity = ClinicalEntity(
        surface_form="coarctation of the aorta",
        base_concept="coarctation of the aorta",
        role="condition",
    )
    enriched = await client.enrich_entity(entity)
    assert enriched.surface_form == "coarctation of the aorta"  # user wording kept
    assert enriched.cui == "C9999999"
    assert enriched.preferred_name
    assert enriched.synonyms

    # enrich many entities concurrently
    many = await client.enrich_entities([entity, entity.model_copy(update={"surface_form": "x"})])
    assert len(many) == 2


def test_serialize_context_renders_entities():
    entities = [
        ClinicalEntity(
            surface_form="recurrent coarctation",
            base_concept="coarctation",
            preferred_name="Coarctation of aorta",
            cui="C2930803",
            synonyms=["Aorta dominant coarctation"],
        ),
        ClinicalEntity(surface_form="surgical repair", role="procedure"),
    ]
    text = serialize_umls_context(entities)
    assert "C2930803" in text
    assert "Aorta dominant coarctation" in text
    assert "Match: none" in text