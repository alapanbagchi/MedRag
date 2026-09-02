"""Memory layer — store CRUD, session lifecycle, temporal behavior.

Uses the in-memory backend so tests are deterministic and DB-free; the
contract exercised here is shared by the Postgres backend (same MemoryStore
interface).
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from src.memory.enums import (
    ClaimRelationKind,
    ClaimStatus,
    ContradictionResolution,
    EvidenceRole,
    ProvenanceClass,
    SessionStatus,
)
from src.memory.models import (
    ClaimEvidenceLinkRecord,
    ClaimRecord,
    ClaimRelationRecord,
    ContradictionRecord,
    EvidenceReferenceRecord,
    ResearchGapRecord,
)
from src.memory.store import InMemoryMemoryStore


@pytest.fixture
def store() -> InMemoryMemoryStore:
    return InMemoryMemoryStore()


def _claim(text: str, cls: ProvenanceClass = ProvenanceClass.MODEL_INFERENCE,
           session: str = "", cid: str = "") -> ClaimRecord:
    return ClaimRecord(id=cid, session_id=session, text=text,
                       provenance_class=cls)


# ---------------------------------------------------------------------------
# Conversations
# ---------------------------------------------------------------------------

def test_conversation_flow(store):
    con = store.start_conversation(user_id="u1")
    store.add_message(con.id, "user", "does vitamin D lower BP?")
    store.add_message(con.id, "assistant", "mixed evidence")
    msgs = store.get_messages(con.id)
    assert [m.role for m in msgs] == ["user", "assistant"]
    summary = store.set_conversation_summary(
        con.id, "discussed vitamin D", open_questions=["older adults?"])
    assert summary.version == 1
    summary2 = store.set_conversation_summary(
        con.id, "still vitamin D", open_questions=["older adults?"])
    assert summary2.version == 2
    assert store.get_conversation_summary(con.id).open_questions == ["older adults?"]


# ---------------------------------------------------------------------------
# Sessions
# ---------------------------------------------------------------------------

def test_session_lifecycle(store):
    s = store.create_session("Does vitamin D supplementation lower blood pressure?", user_id="u1")
    assert s.status == SessionStatus.ACTIVE
    store.add_question(s.id, "vitamin D supplementation and blood pressure")

    # identification by question overlap
    hits = store.find_sessions("vitamin D and blood pressure in older adults", user_id="u1")
    assert hits and hits[0][0].id == s.id

    # an unrelated query should miss
    assert store.find_sessions("radial artery vasospasm after CABG", user_id="u1") == []

    # close + archive
    store.close_session(s.id)
    assert store.get_session(s.id).status == SessionStatus.CLOSED
    store.archive_session(s.id)
    assert store.get_session(s.id).status == SessionStatus.ARCHIVED
    assert store.list_sessions(status=SessionStatus.ARCHIVED)[0].id == s.id


def test_session_merge_repoints_children(store):
    a = store.create_session("session A", user_id="u")
    b = store.create_session("session B", user_id="u")
    store.add_question(a.id, "question from A")
    claim = _claim("a claim", session=a.id)
    store.add_claim(claim)
    store.merge_sessions(a.id, b.id)
    assert store.get_session(a.id).status == SessionStatus.MERGED
    assert store.get_session(a.id).merged_into == b.id
    assert all(q.session_id == b.id for q in store.get_questions(b.id))
    assert store.get_claim(claim.id).session_id == b.id


# ---------------------------------------------------------------------------
# Evidence refs + claims + lineage
# ---------------------------------------------------------------------------

def test_evidence_ref_dedup_and_verified_upgrade(store):
    r1 = store.add_evidence_ref(EvidenceReferenceRecord(
        external_id="PMC11684474", id_type="PMCID",
        verification_status="unverified"))
    r2 = store.add_evidence_ref(EvidenceReferenceRecord(
        external_id="PMC11684474", id_type="PMCID",
        verification_status="verified", verified_by="critic"))
    assert r2.id == r1.id              # same external id -> same record
    assert store.get_evidence_ref(r1.id).verified is True
    # different chunk -> different record
    r3 = store.add_evidence_ref(EvidenceReferenceRecord(
        external_id="PMC11684474", id_type="PMCID", chunk_id="c9",
        verification_status="verified"))
    assert r3.id != r1.id


def test_claim_supersession_temporal(store):
    now = datetime.now(timezone.utc)
    born = now - timedelta(days=30)
    old = _claim("old finding: X reduces Y", session="rs_1")
    new = _claim("new finding: X increases Y", session="rs_1")
    store.add_claim(old)
    store.add_claim(new)
    # append-only: supersede, never overwrite. valid_from back-dated to the
    # claim's true historical interval so as_of() is meaningful.
    store.update_claim(old.id, status=ClaimStatus.SUPERSEDED.value,
                       superseded_by=new.id, superseded_at=now,
                       valid_from=born, valid_to=now)
    assert store.get_claim(old.id).superseded_by == new.id
    # current view excludes the superseded claim
    current = store.get_claims(session_id="rs_1")
    assert [c.id for c in current] == [new.id]
    assert store.get_claims(session_id="rs_1", include_superseded=True)
    # as_of: valid during its life, invalid after supersession (history kept)
    assert old.as_of(born + timedelta(days=1)) is True
    assert old.as_of(now + timedelta(days=1)) is False


def test_claim_links_and_relations(store):
    c = store.add_claim(_claim("claim about dose", session="rs_1"))
    ref = store.add_evidence_ref(EvidenceReferenceRecord(
        external_id="PMC1", id_type="PMCID", verification_status="verified"))
    store.add_claim_evidence_link(ClaimEvidenceLinkRecord(
        claim_id=c.id, evidence_ref_id=ref.id, role=EvidenceRole.SUPPORTED_BY))
    store.add_claim_relation(ClaimRelationRecord(
        from_claim_id=c.id, to_claim_id="clm_other",
        relation=ClaimRelationKind.REFINED_BY))
    links = store.get_claim_evidence_links(c.id)
    assert links[0].evidence_ref_id == ref.id
    rels = store.get_claim_relations(c.id)
    assert rels[0].relation == ClaimRelationKind.REFINED_BY
    assert store.get_claim_relations(c.id, ClaimRelationKind.RELATED_TO) == []


def test_contradiction_and_gap_flows(store):
    ctr = store.add_contradiction(ContradictionRecord(
        session_id="rs_1", claim="A vs B", evidence_a_ids=["ev1"],
        evidence_b_ids=["ev2"]))
    assert store.get_contradictions("rs_1", unresolved_only=True)[0].id == ctr.id
    store.resolve_contradiction(ctr.id, ContradictionResolution.EXPLAINED,
                                "baseline vitamin D status differs")
    assert store.get_contradictions("rs_1", unresolved_only=True) == []
    assert store.get_contradictions("rs_1")[0].resolution == \
        ContradictionResolution.EXPLAINED

    gap = store.add_gap(ResearchGapRecord(session_id="rs_1",
                                          question="no RCT for elderly"))
    assert store.get_gaps("rs_1", open_only=True)[0].id == gap.id
    store.fill_gap(gap.id)
    assert store.get_gaps("rs_1", open_only=True) == []


def test_preferences_and_audit(store):
    store.set_preference("u1", "evidence_type", "meta-analyses")
    store.set_preference("u1", "evidence_type", "RCTs", source="inferred")
    prefs = store.get_preferences("u1")
    assert len(prefs) == 1 and prefs[0].value == "RCTs"
    from src.memory.enums import MemoryEventType, MemoryObjectType
    store.add_event("test", MemoryEventType.COMMIT,
                    MemoryObjectType.CLAIM, "clm_1", {"k": "v"})
    assert store.recent_events(5)[0].event_type == "commit"


def test_counts(store):
    store.create_session("q", user_id="u")
    counts = store.counts()
    assert counts["sessions"] == 1
    assert counts["claims"] == 0