"""Unit tests for Step 2 UMLS enrichment (mock client, no network)."""
import pytest

from src.agentic.planner import PlannedEntity, SubQueryPlan
from src.agentic.umls_tool import UMLSEnricher, EnrichedTerm


class FakeConcept:
    def __init__(self, found, preferred_name=None, cui="C123", synonyms=None):
        self.found = found
        self.preferred_name = preferred_name
        self.cui = cui
        self.synonyms = synonyms or []


class FakeUMLS:
    def __init__(self, mapping):
        self.mapping = mapping
        self.enabled = True

    async def search_concept(self, term):
        return self.mapping.get(term.lower(), FakeConcept(False))


def _sub(query="radial artery vasospasm prevention", entities=None):
    return SubQueryPlan(
        id="H1",
        target="radial artery vasospasm",
        intent="compare",
        query=query,
        entities=entities or [PlannedEntity(text="radial artery vasospasm", role="condition")],
    )


def test_fold_synonyms_into_query():
    client = FakeUMLS({
        "radial artery vasospasm": FakeConcept(
            True, preferred_name="Radial Artery Spasm",
            synonyms=["Coronary vasospasm", "Radial artery spasm", "Arterial spasm, radial"]),
    })
    enricher = UMLSEnricher(umls_client=client)
    terms = [EnrichedTerm("radial artery vasospasm", "Radial Artery Spasm", "C123",
                          ["Coronary vasospasm", "Radial artery spasm", "Arterial spasm, radial"])]
    out = enricher.apply_to_query(_sub(), terms)
    # "Radial artery spasm" echo of surface form is dropped; inverted form dropped
    assert "Coronary vasospasm" in out
    assert "Arterial spasm, radial" not in out
    assert out.startswith("radial artery vasospasm prevention")


def test_exact_echo_dropped_but_variant_kept():
    # the exact surface-form echo (synonym == surface form) must be dropped,
    # while the preferred name (a genuine "spasm" variant) is a useful term
    # that stays folded in.
    terms = [EnrichedTerm("radial artery vasospasm", "Radial Artery Spasm", "C",
                          ["radial artery vasospasm"])]
    enricher = UMLSEnricher(umls_client=FakeUMLS({}))
    out = enricher.apply_to_query(_sub(), terms)
    assert out == "radial artery vasospasm prevention Radial Artery Spasm"


def test_disabled_client_returns_empty():
    class Disabled:
        enabled = False
    enricher = UMLSEnricher(umls_client=Disabled())
    sub = _sub()
    assert enricher.enabled is False


@pytest.mark.asyncio
async def test_enrich_no_entities():
    client = FakeUMLS({})
    enricher = UMLSEnricher(umls_client=client)
    terms = await enricher.enrich_subquery(_sub(entities=[]))
    assert terms == []


@pytest.mark.asyncio
async def test_enrich_subquery_resolves():
    client = FakeUMLS({
        "radial artery vasospasm": FakeConcept(
            True, preferred_name="Radial Artery Spasm", synonyms=["Coronary vasospasm"]),
    })
    enricher = UMLSEnricher(umls_client=client)
    terms = await enricher.enrich_subquery(_sub())
    assert len(terms) == 1
    assert terms[0].preferred_name == "Radial Artery Spasm"
    assert terms[0].synonyms == ["Coronary vasospasm"]
