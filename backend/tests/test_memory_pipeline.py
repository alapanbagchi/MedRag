"""Memory layer — the run record path (record_run) + background consolidation.

record_run is the bridge from an AgenticV3 result dict into persistent
research memory; consolidation is the offline dedup/merge/contradiction/
temporal pass. Both are exercised with the in-memory backend.
"""

from __future__ import annotations

import pytest

from src.memory.api import MemoryAPI
from src.memory.config import MemoryConfig
from src.memory.embed import HashEmbedder
from src.memory.enums import (
    ClaimStatus,
    ContradictionResolution,
    EvidenceRole,
    ProvenanceClass,
)
from src.memory.models import ClaimRecord
from src.memory.store import InMemoryMemoryStore


def build_api() -> MemoryAPI:
    return MemoryAPI(
        store=InMemoryMemoryStore(),
        embedder=HashEmbedder(256),
        config=MemoryConfig(backend="memory", claim_min_sim=0.75),
    )


def canned_run(question: str = "Does vitamin D supplementation lower blood pressure?",
               run_id: str = "run-1") -> dict:
    return {
        "run_id": run_id,
        "question": question,
        "tasks": [{
            "id": "T1",
            "title": "Vitamin D and blood pressure",
            "objective": "determine whether vitamin D supplementation lowers blood pressure",
            "evidence_requirements": [
                {"id": "T1.R1", "text": "vitamin D supplementation and blood pressure"}
            ],
        }],
        "evidence": [
            {"id": "E1", "document_id": "PMC11684474", "chunk_id": "c1",
             "section": "Results", "excerpt": "…",
             "claim": "Vitamin D supplementation significantly reduced systolic "
                      "blood pressure in hypertensive adults.",
             "support": "supports", "confidence": 0.9, "status": "accepted",
             "source": "retrieval", "retrieval_method": "bm25", "rank": 1,
             "critic_verdict": {"verifier": "critic-v3"}},
            {"id": "E2", "document_id": "PMC11684475", "chunk_id": "c2",
             "section": "Results", "excerpt": "…",
             "claim": "Vitamin D supplementation had no significant effect on "
                      "blood pressure in vitamin-D-replete adults.",
             "support": "contradicts", "confidence": 0.8, "status": "accepted",
             "source": "retrieval", "retrieval_method": "bm25", "rank": 2,
             "critic_verdict": {}},
        ],
        "gaps": ["no prospective RCT in older adults found"],
        "contradictions": [{
            "id": "C1",
            "claim": "vitamin D supplementation affects systolic blood pressure",
            "evidence_a": ["E1"], "evidence_b": ["E2"],
            "kind": "context_dependent",
            "resolution": {"status": "unresolved", "explanation": "baseline "
                           "vitamin D status may explain the difference",
                           "characterization": "context-dependent: baseline status"},
        }],
        "answer": {"summary": "Evidence is mixed; effect may depend on baseline "
                   "vitamin D status.", "sections": [], "confidence": 0.5,
                   "citations": []},
    }


# ---------------------------------------------------------------------------
# record_run
# ---------------------------------------------------------------------------

def test_record_run_persists_research_state():
    api = build_api()
    stats = api.record_run(canned_run(), conversation_id="con_1")

    assert stats.claims_committed == 2           # two verified evidence claims
    assert stats.contradictions == 1
    assert stats.gaps == 1
    assert stats.questions >= 1
    assert stats.conclusion                        # labeled conclusion claim

    # session exists; claims are EVIDENCE_DERIVED with full lineage
    session = api.store.get_session(stats.session_id)
    assert session is not None
    claims = [c for c in api.store.get_claims(session_id=session.id)]
    evidence_claims = [c for c in claims
                       if c.provenance_class == ProvenanceClass.EVIDENCE_DERIVED_CLAIM]
    assert len(evidence_claims) == 2
    for c in evidence_claims:
        links = api.store.get_claim_evidence_links(c.id)
        assert len(links) >= 1
        supported = [l for l in links if l.role == EvidenceRole.SUPPORTED_BY]
        assert supported, "evidence-backed claim must carry SUPPORTED_BY link"
        ref = api.store.get_evidence_ref(supported[0].evidence_ref_id)
        assert ref.verified is True
        assert ref.external_id.startswith("PMC")

    # the conclusion must be labeled, never evidence-backed
    conclusion = [c for c in claims
                  if c.provenance_class == ProvenanceClass.MODEL_INFERENCE]
    assert len(conclusion) == 1
    assert api.store.get_claim_evidence_links(conclusion[0].id) == []

    # contradiction preserved with BOTH sides
    ctrs = api.store.get_contradictions(session.id)
    assert len(ctrs) == 1
    assert len(ctrs[0].evidence_a_ids) == 1 and len(ctrs[0].evidence_b_ids) == 1
    assert ctrs[0].resolution == ContradictionResolution.UNRESOLVED


def test_record_run_idempotent_dedup():
    api = build_api()
    first = api.record_run(canned_run())
    second = api.record_run(canned_run(run_id="run-2"))

    assert second.session_id == first.session_id   # resumed same session
    assert second.claims_deduped == 2              # nothing new committed
    assert second.claims_committed == 0
    assert second.contradictions == 0              # duplicate contradiction skipped
    assert second.gaps == 0
    claims = api.store.get_claims(session_id=first.session_id,
                                  provenance=ProvenanceClass.EVIDENCE_DERIVED_CLAIM)
    assert len(claims) == 2                        # not 4


def test_record_run_relates_two_conversations_to_one_session():
    api = build_api()
    api.record_run(canned_run(question="Does vitamin D supplementation lower blood pressure?"))
    hits = api.store.find_sessions("what about vitamin D for older adults?")
    assert hits  # follow-up conversation identifies the same investigation


def test_evidence_lineage_audit():
    api = build_api()
    api.record_run(canned_run())
    claim = [c for c in api.store.get_claims()
             if c.provenance_class == ProvenanceClass.EVIDENCE_DERIVED_CLAIM][0]
    lineage = api.get_evidence_lineage(claim.id)
    assert lineage.claim.id == claim.id
    assert len(lineage.links) >= 1
    assert any(r.verified for r in lineage.evidence_refs)
    rendered = lineage.render()
    assert "PMC11684474" in rendered or "PMC11684475" in rendered
    assert "evidence links" in rendered


# ---------------------------------------------------------------------------
# Session resume via API
# ---------------------------------------------------------------------------

def test_resume_identify_or_create():
    api = build_api()
    api.record_run(canned_run())
    resumed = api.resume_research_session(
        "Does vitamin D supplementation lower blood pressure?")
    assert resumed.id  # resumed (found by overlap)
    fresh = api.resume_research_session("radial artery vasospasm after CABG")
    assert fresh.id != resumed.id


# ---------------------------------------------------------------------------
# Consolidation
# ---------------------------------------------------------------------------

def test_consolidate_derives_statuses_and_keeps_contradictions():
    api = build_api()
    api.record_run(canned_run())
    report = api.consolidate()
    # claim statuses derived from their links (supported AND contradicted, so
    # MIXED for the vitamin D pair — the conflict is never collapsed)
    evidence_claims = [c for c in api.store.get_claims()
                       if c.provenance_class == ProvenanceClass.EVIDENCE_DERIVED_CLAIM]
    assert evidence_claims
    assert all(c.status in (ClaimStatus.SUPPORTED, ClaimStatus.CONTRADICTED,
                            ClaimStatus.MIXED)
               for c in evidence_claims)
    # contradictions preserved with BOTH sides
    ctrs = api.store.get_contradictions()
    assert ctrs
    assert all(c.evidence_a_ids and c.evidence_b_ids for c in ctrs)
    assert report["contradictions_detected"] >= 0


def test_consolidate_merges_near_duplicate_claims():
    """Legacy duplicates (committed before dedup existed) are merged by the
    background pass: evidence links re-pointed, older claim SUPERSEDED."""
    api = build_api()
    api.record_run(canned_run())
    orig = [c for c in api.store.get_claims(
        provenance=ProvenanceClass.EVIDENCE_DERIVED_CLAIM)][0]
    # a near-duplicate committed directly (bypasses the commit-time dedup)
    dup = ClaimRecord(
        id="clm_legacy", session_id=orig.session_id,
        text=orig.text + " (per the trial results as reported).",
        provenance_class=orig.provenance_class)
    api.store.add_claim(dup)
    before = len(api.store.get_claims(
        provenance=ProvenanceClass.EVIDENCE_DERIVED_CLAIM))
    api.consolidate()
    superseded = [c for c in api.store.get_claims(include_superseded=True)
                  if c.status == ClaimStatus.SUPERSEDED]
    assert len(superseded) >= 1
    after = len(api.store.get_claims(
        provenance=ProvenanceClass.EVIDENCE_DERIVED_CLAIM))
    assert after <= before


def test_consolidate_stale_horizon():
    api = build_api()
    api.record_run(canned_run())
    # no-op with disabled horizon (default)
    assert api.consolidate(staleness_days=0)["stale_flagged"] == 0
    # explicit mark_stale works regardless of horizon
    claim = api.store.get_claims()[0]
    api.mark_stale(claim.id, reason="explicit")
    assert api.store.get_claim(claim.id).needs_revalidation is True