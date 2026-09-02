"""Memory + Context layer — the write pipeline.

Safe pipeline for creating/updating memory:

    New information
        -> classification        (classify_text: what kind is this?)
        -> candidate extraction  (CandidateMemory)
        -> schema validation     (pydantic records)
        -> provenance validation (validation.evidence_gate)
        -> deduplication         (dedup_key + embedding similarity)
        -> relationship detection(SUPERSEDES / REFINED_BY / contradicts)
        -> temporal update       (valid intervals, supersession)
        -> commit                (transaction + append-only audit event)

Candidate vs Committed: nothing reaches the store except through
``commit()``, and ``commit()`` never writes a claim whose provenance class is
EVIDENCE_DERIVED_CLAIM without a verified evidence link (the gate degrades it
to a labeled MODEL_INFERENCE instead).

Also contains ``extract_run_candidates()`` — the deterministic bridge from an
AgenticV3 pipeline result dict to candidate memory. Only CRITIC-verified
evidence items become evidence references + EVIDENCE_DERIVED_CLAIMs; answers
become labeled MODEL_INFERENCE conclusions (never evidence).
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

from src.memory.enums import (
    CandidateState,
    ClaimStatus,
    ContradictionKind,
    ContradictionResolution,
    EvidenceRole,
    GapKind,
    MemoryEventType,
    MemoryObjectType,
    ProvenanceClass,
)
from src.memory.models import (
    CandidateMemory,
    ClaimEvidenceLinkRecord,
    ClaimRecord,
    ContradictionRecord,
    EvidenceReferenceRecord,
    ResearchGapRecord,
    ValidationResult,
    new_id,
)
from src.memory.store import MemoryStore
from src.memory import validation as v
from src.memory.validation import (
    claim_dedup_key,
    classify_text,
    evidence_gate,
    normalized_claim_text,
)

logger = logging.getLogger("src.memory.pipeline")

# trusted actors that may propose EVIDENCE_DERIVED_CLAIM candidates (they are
# the only paths where critic-verified evidence exists)
_TRUSTED_EXTRACTORS = {"pipeline", "critic", "evidence_extractor"}


class MemoryWritePipeline:
    """Candidate -> validated -> committed writes with full audit."""

    def __init__(self, store: MemoryStore, embedder: Any | None = None,
                 min_sim: float = 0.86):
        self.store = store
        self.embedder = embedder
        self.min_sim = min_sim

    # -- classification + candidate extraction --------------------------

    def classify(self, text: str) -> v.Classification:
        return classify_text(text)

    def make_candidate(
        self,
        *,
        text: str,
        provenance_class: ProvenanceClass,
        session_id: str = "",
        user_id: str = "",
        evidence_ref_ids: list[str] | None = None,
        source_actor: str = "memory_pipeline",
        source_text: str = "",
        confidence: float = 0.0,
        topic_ids: list[str] | None = None,
        entity_ids: list[str] | None = None,
        meta: dict[str, Any] | None = None,
        claim_status: str | None = None,
    ) -> CandidateMemory:
        """Build a claim candidate (schema validation happens on commit)."""
        normal = normalized_claim_text(text)
        claim = ClaimRecord(
            id=new_id("clm"), session_id=session_id, user_id=user_id,
            text=v.sanitize_stored_text(text),
            normalized_text=normal,
            provenance_class=provenance_class,
            topic_ids=topic_ids or [],
            entity_ids=entity_ids or [],
            confidence=confidence,
            meta=meta or {},
        )
        claim.dedup_key = claim_dedup_key(claim)
        if claim_status:
            claim.status = ClaimStatus(claim_status)
        cand = CandidateMemory.from_claim(
            claim,
            evidence_ref_ids=list(evidence_ref_ids or []),
            source_actor=source_actor,
            source_text=source_text or text,
            confidence=confidence,
        )
        return cand

    # -- propose / validate / commit ------------------------------------

    def propose(self, candidate: CandidateMemory) -> CandidateMemory:
        candidate.state = CandidateState.CANDIDATE
        self.store.add_event(
            candidate.source_actor, MemoryEventType.PROPOSE,
            candidate.object_type, candidate.id,
            {"class": candidate.provenance_class.value,
             "text": (candidate.source_text or "")[:200]})
        return candidate

    def validate(self, candidate: CandidateMemory) -> ValidationResult:
        """Provenance gate + dedup pre-check. Never upgrades a class; may
        degrade EVIDENCE_DERIVED_CLAIM to MODEL_INFERENCE when unverified."""
        result = evidence_gate(candidate, self.store.get_evidence_ref)

        # dedup pre-check for claims (only when not already degraded to a
        # non-claim situation)
        if candidate.object_type == MemoryObjectType.CLAIM and result.ok:
            existing = self._find_duplicate(candidate)
            if existing is not None:
                result.dedup_match_id = existing.id
                result.reasons.append(f"duplicate of existing claim {existing.id}")
        candidate.state = CandidateState.VALIDATED if result.ok else CandidateState.REJECTED
        self.store.add_event(
            candidate.source_actor, MemoryEventType.VALIDATE,
            candidate.object_type, candidate.id,
            {"ok": result.ok, "reasons": result.reasons,
             "degraded_to": result.degraded_to.value if result.degraded_to else None})
        return result

    def commit(self, candidate: CandidateMemory,
               validated: ValidationResult | None = None) -> tuple[str, Any]:
        """Commit a validated candidate; returns (object_id, object).

        Atomic: claim + links + audit event are written in one transaction.
        """
        result = validated or self.validate(candidate)
        if not result.ok:
            raise ValueError("refusing to commit invalid candidate: "
                             + "; ".join(result.reasons))

        # dedup suppression: the candidate duplicates an existing object. For
        # evidence-backed claims the candidate's evidence refs are UNIONED
        # into the survivor (a claim supported by 2 papers must carry both
        # links) — provenance is never discarded by dedup.
        if result.dedup_match_id:
            existing = self.store.get_claim(result.dedup_match_id)
            if (existing is not None
                    and candidate.object_type == MemoryObjectType.CLAIM
                    and existing.provenance_class.evidence_backed
                    and candidate.provenance_class.evidence_backed):
                have = {l.evidence_ref_id
                        for l in self.store.get_claim_evidence_links(existing.id)}
                for ref_id in candidate.evidence_ref_ids:
                    if ref_id and ref_id not in have:
                        self.store.add_claim_evidence_link(ClaimEvidenceLinkRecord(
                            claim_id=existing.id, evidence_ref_id=ref_id,
                            role=EvidenceRole.SUPPORTED_BY))
            self.store.add_event(
                candidate.source_actor, MemoryEventType.REJECT,
                candidate.object_type, candidate.id,
                {"reason": "dedup", "match": result.dedup_match_id})
            return result.dedup_match_id, existing

        with self.store.transaction():
            if candidate.object_type == MemoryObjectType.CLAIM:
                claim = ClaimRecord.model_validate(candidate.payload)
                claim.id = claim.id or new_id("clm")
                claim.normalized_text = normalized_claim_text(claim.text)
                claim.dedup_key = claim_dedup_key(claim)
                claim.provenance_class = (
                    result.degraded_to or candidate.provenance_class)
                self.store.add_claim(claim)
                # provenance discipline: only evidence-backed claims may carry
                # evidence links (a degraded claim must not look like evidence)
                if claim.provenance_class.evidence_backed:
                    for ref_id in candidate.evidence_ref_ids:
                        self.store.add_claim_evidence_link(ClaimEvidenceLinkRecord(
                            claim_id=claim.id, evidence_ref_id=ref_id,
                            role=EvidenceRole.SUPPORTED_BY))
                    # an evidence-backed claim inherits the verified-at stamp
                    # from its evidence references (temporal provenance)
                    verified_at = self._latest_verified_at_store(
                        candidate.evidence_ref_ids)
                    if verified_at is not None:
                        self.store.update_claim(claim.id,
                                                last_verified_at=verified_at)
                self.store.add_event(
                    candidate.source_actor, MemoryEventType.COMMIT,
                    MemoryObjectType.CLAIM, claim.id,
                    {"class": claim.provenance_class.value,
                     "evidence_refs": candidate.evidence_ref_ids})
                return claim.id, claim

            if candidate.object_type == MemoryObjectType.USER_PREFERENCE:
                payload = candidate.payload
                pref = self.store.set_preference(
                    candidate.user_id, payload.get("key", "format"),
                    payload.get("value"), source="explicit")
                self.store.add_event(
                    candidate.source_actor, MemoryEventType.COMMIT,
                    MemoryObjectType.USER_PREFERENCE, pref.id)
                return pref.id, pref

            if candidate.object_type == MemoryObjectType.RESEARCH_QUESTION:
                payload = candidate.payload
                q = self.store.add_question(
                    candidate.session_id, payload.get("question", ""),
                    payload.get("evidence_requirement") or None)
                self.store.add_event(
                    candidate.source_actor, MemoryEventType.COMMIT,
                    MemoryObjectType.RESEARCH_QUESTION, q.id)
                return q.id, q

            raise ValueError(
                f"unsupported candidate object_type: {candidate.object_type.value}")

    def process(self, candidate: CandidateMemory) -> tuple[str, Any]:
        """propose -> validate -> commit in one call."""
        self.propose(candidate)
        result = self.validate(candidate)
        return self.commit(candidate, result)

    # -- helpers ----------------------------------------------------------

    def _find_duplicate(self, candidate: CandidateMemory) -> ClaimRecord | None:
        if candidate.object_type != MemoryObjectType.CLAIM:
            return None
        claim = ClaimRecord.model_validate(candidate.payload)
        claim.normalized_text = normalized_claim_text(claim.text)
        claim.dedup_key = claim_dedup_key(claim)

        for existing in self.store.get_claims(
                session_id=candidate.session_id or None,
                provenance=claim.provenance_class):
            if existing.id == claim.id:
                continue
            if v.is_near_duplicate(claim, existing, self.embedder, self.min_sim) \
                    if self.embedder is not None else (
                        claim.dedup_key == existing.dedup_key):
                return existing
        return None

    def _latest_verified_at_store(self, ref_ids: list[str]) -> Any:
        """Earliest verified_at across the candidate's evidence refs."""
        latest = None
        for ref in self.store.get_evidence_refs([rid for rid in ref_ids if rid]):
            if ref.verified and ref.verified_at:
                if latest is None or ref.verified_at > latest:
                    latest = ref.verified_at
        return latest


# ---------------------------------------------------------------------------
# Bridge: AgenticV3 run result -> candidate memory
# ---------------------------------------------------------------------------

def extract_run_candidates(
    result: dict[str, Any],
    session_id: str,
    user_id: str = "",
    evidence_id_map: dict[str, str] | None = None,
) -> tuple[list[EvidenceReferenceRecord], list[CandidateMemory],
           list[ContradictionRecord], list[ResearchGapRecord]]:
    """Deterministically translate one pipeline run result into memory
    candidates. Only CRITIC-verified evidence items participate.

    Returns (evidence_refs, claim_candidates, contradictions, gaps).

    Invariant enforced here: EVIDENCE_DERIVED_CLAIM candidates carry exactly
    the evidence ref ids of the verified evidence they were derived from —
    the provenance gate keeps them honest at commit time.
    """
    evidence_id_map = evidence_id_map or {}
    refs: list[EvidenceReferenceRecord] = []
    ref_by_evidence_id: dict[str, str] = dict(evidence_id_map)
    claims: list[CandidateMemory] = []

    for item in result.get("evidence", []):
        status = str(item.get("status") or "accepted").lower()
        verified = status in ("accepted", "contradictory")
        ref = EvidenceReferenceRecord(
            id=new_id("ev"),
            source="PMC",
            external_id=str(item.get("document_id") or ""),
            id_type="PMCID",
            chunk_id=str(item.get("chunk_id") or ""),
            title="",
            verification_status="verified" if verified else "unverified",
            verified_by=str((item.get("critic_verdict") or {}).get("verifier")
                            or item.get("verifier") or "critic"),
            retrieved_at=None,
            publication_date=None,
            meta={
                "run_id": result.get("run_id", ""),
                "evidence_id": item.get("id", ""),
                "task_id": item.get("task_id", ""),
                "requirement_id": item.get("requirement_id", ""),
                "claim": item.get("claim", ""),
                "excerpt": (item.get("excerpt") or "")[:1200],
                "source": item.get("source", ""),
                "retrieval_method": item.get("retrieval_method", ""),
                "rank": item.get("rank", 0),
            },
        )
        refs.append(ref)
        ref_by_evidence_id[str(item.get("id", ""))] = ref.id

        claim_text = (item.get("claim") or "").strip()
        if not claim_text or not verified:
            continue
        support = str(item.get("support") or "supports").lower()
        cand = CandidateMemory(
            id=new_id("can"),
            object_type=MemoryObjectType.CLAIM,
            payload=ClaimRecord(
                id=new_id("clm"), session_id=session_id, user_id=user_id,
                text=v.sanitize_stored_text(claim_text),
                normalized_text=normalized_claim_text(claim_text),
                provenance_class=ProvenanceClass.EVIDENCE_DERIVED_CLAIM,
                confidence=float(item.get("confidence") or 0.0),
                last_verified_at=datetime.now(timezone.utc),
                meta={"run_id": result.get("run_id", ""),
                      "evidence_id": item.get("id", ""),
                      "support": support},
            ).model_dump(),
            provenance_class=ProvenanceClass.EVIDENCE_DERIVED_CLAIM,
            evidence_ref_ids=[ref.id],
            session_id=session_id, user_id=user_id,
            source_actor="pipeline",
            source_text=claim_text,
            confidence=float(item.get("confidence") or 0.0),
        )
        claims.append(cand)

    # contradiction records (evidence ids remapped to memory ref ids)
    contradictions: list[ContradictionRecord] = []
    for c in result.get("contradictions", []):
        res = c.get("resolution") or {}
        rec = ContradictionRecord(
            id=new_id("ctr"), session_id=session_id,
            claim=(c.get("claim") or "")[:600],
            evidence_a_ids=[ref_by_evidence_id.get(i, i)
                            for i in c.get("evidence_a", [])],
            evidence_b_ids=[ref_by_evidence_id.get(i, i)
                            for i in c.get("evidence_b", [])],
            kind=_contradiction_kind(c.get("kind")),
            dimensions=_dimensions_from_resolution(res),
            resolution=ContradictionResolution(
                str(res.get("status") or "unresolved")),
            explanation=str(res.get("explanation") or "")[:800],
        )
        contradictions.append(rec)

    gaps: list[ResearchGapRecord] = []
    for g in result.get("gaps", []):
        text = str(g).strip()
        if text:
            gaps.append(ResearchGapRecord(
                id=new_id("gap"), session_id=session_id, question=text[:500],
                kind=GapKind.MISSING_EVIDENCE,
                meta={"run_id": result.get("run_id", "")}))

    return refs, claims, contradictions, gaps


def _contradiction_kind(kind: str | None):
    try:
        return ContradictionKind(kind)
    except Exception:
        return ContradictionKind.DIRECT_CONFLICT


def _dimensions_from_resolution(res: dict) -> dict[str, str]:
    dims: dict[str, str] = {}
    char = str(res.get("characterization") or "")
    if char:
        dims["characterization"] = char[:300]
    for key in ("population", "intervention", "comparator", "outcome",
                "study_design", "follow_up", "dosage", "baseline",
                "measurement", "publication_time"):
        val = res.get(key)
        if val:
            dims[key] = str(val)[:200]
    return dims


__all__ = [
    "MemoryWritePipeline",
    "extract_run_candidates",
]