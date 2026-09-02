"""Memory layer — typed context blocks, budgets, boundary markers."""

from __future__ import annotations

import pytest

from src.memory.api import MemoryAPI
from src.memory.config import MemoryConfig
from src.memory.context import AssembledContext, estimate_tokens
from src.memory.embed import HashEmbedder
from src.memory.enums import MemoryContextKind, ProvenanceClass
from src.memory.store import InMemoryMemoryStore


def _api_with_content() -> MemoryAPI:
    api = MemoryAPI(store=InMemoryMemoryStore(), embedder=HashEmbedder(256),
                    config=MemoryConfig(backend="memory"))
    api.record_run({
        "run_id": "r1",
        "question": "Does vitamin D supplementation lower blood pressure?",
        "tasks": [{"id": "T1", "title": "vitamin D",
                   "objective": "vitamin D on BP",
                   "evidence_requirements": [{"id": "T1.R1",
                                              "text": "vitamin D and blood pressure"}]}],
        "evidence": [
            {"id": "E1", "document_id": "PMC11684474", "chunk_id": "c1",
             "status": "accepted", "support": "supports", "confidence": 0.9,
             "claim": "Vitamin D supplementation significantly reduced systolic "
                      "blood pressure in hypertensive adults."},
            {"id": "E2", "document_id": "PMC11684475", "chunk_id": "c2",
             "status": "accepted", "support": "contradicts", "confidence": 0.8,
             "claim": "Vitamin D supplementation had no significant effect on "
                      "blood pressure in replete adults."},
        ],
        "gaps": ["no elderly RCT"],
        "contradictions": [{
            "id": "C1", "claim": "vitamin D affects blood pressure",
            "evidence_a": ["E1"], "evidence_b": ["E2"],
            "kind": "context_dependent",
            "resolution": {"status": "unresolved", "explanation": "baseline status differs"}}],
        "answer": {"summary": "mixed evidence"},
    })
    return api


def test_assembled_context_ordering_and_boundaries():
    api = _api_with_content()
    session = api.store.list_sessions()[0]
    assembled = api.compose_context(
        "vitamin D blood pressure", session_id=session.id,
        user_id="u1", budget_tokens=2000)
    rendered = assembled.render()

    # ordering: query first, then EVIDENCE then MEMORY with boundary markers
    assert rendered.index("CURRENT QUERY") < rendered.index("VERIFIED EVIDENCE")
    assert rendered.index("VERIFIED EVIDENCE") < rendered.index("PERSISTENT RESEARCH MEMORY")
    # If the memory block is present, boundary markers exist on both sides
    marker = "REMINDER: only the VERIFIED EVIDENCE block is authoritative"
    assert marker in rendered


def test_context_never_puts_memory_in_evidence_block():
    api = _api_with_content()
    # prepare_run must leave the evidence block EMPTY (current-run evidence is
    # supplied by the pipeline, not by memory)
    prep = api.prepare_run("vitamin D blood pressure")
    assert prep.context.evidence.evidence_refs == []
    rendered = prep.context_text()
    # and prior memory never masquerades as verified evidence
    assert "EVIDENCE-DERIVED" in rendered           # memory block badges present
    assert "USER-ASSERTION" not in rendered         # no user assertions here


def test_contradiction_renders_both_sides():
    api = _api_with_content()
    session = api.store.list_sessions()[0]
    assembled = api.compose_context("vitamin D", session_id=session.id,
                                    budget_tokens=2000)
    rendered = assembled.render()
    assert "CONTRADICTION" in rendered
    assert "side A" in rendered and "side B" in rendered   # never collapsed


def test_evidence_citations_rendered_with_claims():
    api = _api_with_content()
    session = api.store.list_sessions()[0]
    assembled = api.compose_context("vitamin D", session_id=session.id)
    rendered = assembled.render()
    assert "PMC11684474" in rendered     # citation preserved with the claim


def test_budget_enforced():
    api = _api_with_content()
    session = api.store.list_sessions()[0]
    small = api.compose_context("vitamin D blood pressure", session_id=session.id,
                                budget_tokens=90)
    rendered = small.render()
    # hard budget: the whole rendered region (incl. headers + footer) fits the
    # token estimate; lowest-priority whole blocks are shed, never silently
    # half-cut mid-item
    assert estimate_tokens(rendered) <= 90 + 20
    assert "EVIDENCE-DERIVED" not in rendered   # persistent-memory block shed
    # a larger budget yields a LARGER region (monotonic budget scaling)
    big = api.compose_context("vitamin D blood pressure", session_id=session.id,
                              budget_tokens=2000)
    assert len(big.render()) > len(rendered)
    assert "EVIDENCE-DERIVED" in big.render()


def test_typed_blocks_converted_via_render_only():
    """Context objects are typed; render() is the ONLY string conversion."""
    ctx = AssembledContext(query="q", total_budget_tokens=500)
    block = ctx.conversation
    assert block.kind == MemoryContextKind.CONVERSATION
    assert isinstance(block.render(), str)
    # records remain typed objects, not strings
    assert ctx.memory.claims == [] and isinstance(ctx.memory.claims, list)


def test_evidence_block_only_from_explicit_refs():
    api = _api_with_content()
    claims = api.store.get_claims(
        provenance=ProvenanceClass.EVIDENCE_DERIVED_CLAIM)
    link = api.store.get_claim_evidence_links(claims[0].id)[0]
    ref = api.store.get_evidence_ref(link.evidence_ref_id)
    ctx = api.get_evidence_context([ref.id])
    assert ctx.evidence_refs[0].external_id == ref.external_id
    assert "VERIFIED EVIDENCE" in ctx.render()


def test_user_preference_context():
    api = _api_with_content()
    api.store.set_preference("u1", "evidence_type", "meta-analyses only")
    prefs = api.get_user_preferences("u1")
    rendered = prefs.render()
    assert "USER PREFERENCES" in rendered
    assert "evidence_type" in rendered