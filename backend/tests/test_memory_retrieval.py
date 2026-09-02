"""Memory layer — hybrid retrieval (recall/precision, staleness, contradictions)."""

from __future__ import annotations

import pytest

from src.memory.api import MemoryAPI
from src.memory.config import MemoryConfig
from src.memory.embed import HashEmbedder
from src.memory.enums import ProvenanceClass
from src.memory.retrieval import retrieve_claims
from src.memory.store import InMemoryMemoryStore


def seeded_api() -> MemoryAPI:
    """Two sessions with distinct research topics + one contradiction."""
    api = MemoryAPI(store=InMemoryMemoryStore(),
                    embedder=HashEmbedder(256),
                    config=MemoryConfig(backend="memory"))

    # session 1: vitamin D / blood pressure — with a contradiction
    run1 = {
        "run_id": "r1", "question": "Does vitamin D supplementation lower blood pressure?",
        "tasks": [{"id": "T1", "title": "vitamin D and blood pressure",
                   "objective": "effect of vitamin D on blood pressure",
                   "evidence_requirements": [
                       {"id": "T1.R1", "text": "vitamin D supplementation and blood pressure"}]}],
        "evidence": [
            {"id": "E1", "document_id": "PMC100001", "chunk_id": "c1", "status": "accepted",
             "claim": "Vitamin D supplementation significantly reduced systolic blood pressure "
                      "in hypertensive adults.", "support": "supports", "confidence": 0.9},
            {"id": "E2", "document_id": "PMC100002", "chunk_id": "c2", "status": "accepted",
             "claim": "Vitamin D supplementation had no significant effect on blood pressure "
                      "in vitamin-D-replete adults.", "support": "contradicts", "confidence": 0.8},
        ],
        "gaps": ["no elderly RCT"],
        "contradictions": [{"id": "C1", "claim": "vitamin D effects on blood pressure",
                            "evidence_a": ["E1"], "evidence_b": ["E2"],
                            "kind": "context_dependent",
                            "resolution": {"status": "unresolved"}}],
        "answer": {"summary": "mixed"},
    }
    # session 2: statins and stroke — unrelated topic
    run2 = {
        "run_id": "r2", "question": "Do statins reduce stroke risk after TIA?",
        "tasks": [{"id": "T1", "title": "statins and stroke",
                   "objective": "statin effect on stroke risk",
                   "evidence_requirements": [
                       {"id": "T1.R1", "text": "statin therapy and stroke recurrence"}]}],
        "evidence": [
            {"id": "S1", "document_id": "PMC200001", "chunk_id": "c1", "status": "accepted",
             "claim": "High-intensity statin therapy reduced stroke recurrence after transient "
                      "ischemic attack.", "support": "supports", "confidence": 0.85},
        ],
        "gaps": [], "contradictions": [], "answer": {"summary": "statins help"},
    }
    api.record_run(run1)
    api.record_run(run2)
    return api


def test_recall_prior_claim_by_reworded_query():
    api = seeded_api()
    memories = api.get_relevant_memories(
        "Does vitamin D lower high blood pressure?", top_k=6)
    texts = {c.text for c in memories.claims}
    assert any("vitamin d" in t.casefold() for t in texts)
    assert not any("statin" in t.casefold() for t in texts)


def test_session_scoped_recall():
    api = seeded_api()
    s1 = [s for s in api.store.list_sessions() if "vitamin" in s.title.casefold()][0]
    memories = api.get_relevant_memories("blood pressure", session_id=s1.id, top_k=6)
    assert memories.claims  # anything is a hit because session filters
    assert all(c.session_id == s1.id for c in memories.claims)


def test_lexical_recall_precision():
    api = seeded_api()
    scored = retrieve_claims(api.store, "statin stroke", api.embedder, top_k=5)
    assert scored, "no candidates retrieved"
    top = scored[0]
    assert "statin" in top.claim.text.casefold() \
        or "stroke" in top.claim.text.casefold()
    # any vitamin-D claim that surfaces must score below the statin claim
    for s in scored:
        if "vitamin d" in s.claim.text.casefold():
            assert s.score < top.score


def test_unresolved_contradiction_surfaced():
    api = seeded_api()
    memories = api.get_relevant_memories("vitamin D and blood pressure", top_k=6)
    assert memories.contradictions
    assert memories.contradictions[0].unresolved


def test_open_gap_surfaced():
    api = seeded_api()
    memories = api.get_relevant_memories("vitamin D blood pressure", top_k=6)
    assert any("elderly" in g.question for g in memories.gaps)


def test_stale_claim_excluded_by_default():
    api = seeded_api()
    claim = [c for c in api.store.get_claims()
             if c.provenance_class == ProvenanceClass.EVIDENCE_DERIVED_CLAIM][0]
    api.mark_stale(claim.id, reason="revalidation needed")
    memories = api.get_relevant_memories("vitamin D blood pressure", top_k=10)
    assert not any(c.id == claim.id for c in memories.claims)
    memories_stale = api.get_relevant_memories(
        "vitamin D blood pressure", top_k=10, include_stale=True)
    assert any(c.id == claim.id for c in memories_stale.claims)


def test_evidence_derived_claims_outrank_inferences_on_tie():
    api = MemoryAPI(store=InMemoryMemoryStore(), embedder=HashEmbedder(256),
                    config=MemoryConfig(backend="memory"))
    from src.memory.models import ClaimRecord
    from src.memory.enums import ClaimStatus
    api.store.add_claim(ClaimRecord(
        id="clm_ev", session_id="rs_x", text="X reduces Y in adults",
        provenance_class=ProvenanceClass.EVIDENCE_DERIVED_CLAIM,
        status=ClaimStatus.SUPPORTED))
    api.store.add_claim(ClaimRecord(
        id="clm_inf", session_id="rs_x", text="X reduces Y in adults",
        provenance_class=ProvenanceClass.MODEL_INFERENCE, status=ClaimStatus.UNRESOLVED))
    scored = retrieve_claims(api.store, "X reduces Y", api.embedder, top_k=2)
    ids = [s.claim.id for s in scored]
    assert ids[0] == "clm_ev"   # evidence-backed claim ranks first (penalty applied to inference)