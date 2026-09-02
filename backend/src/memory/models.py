"""Memory + Context layer — typed record models (L0-L2).

Every object the memory layer persists has an explicit Pydantic record.
There are no free-string objects: records carry structured fields,
timestamps (event-time ``*_at``) and, where world-time matters, a
``valid_from``/``valid_to`` interval so historical state is never lost.

Temporal design (bi-temporal, append-only):
    * event time  — when *we* recorded it  (created_at, *_at fields)
    * valid time  — when it was true in the world (valid_from / valid_to;
                    NULL valid_to = still current, as_of(t) filters on it)

The only object that crosses from the L3 evidence side into memory is an
EvidenceReferenceRecord — a pointer (PMCID/PMID/DOI + chunk) with its
verification status, never a copy of full evidence text. Memory may ONLY
reference L3; it can never become an alternative source of medical truth.

This module contains NO LLM / corpus / DB code — pure data, unit-testable
offline (same philosophy as src.agentic.state).
"""

from __future__ import annotations

import hashlib
import uuid
from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, Field

from src.memory.enums import (
    CandidateState,
    ClaimRelationKind,
    ClaimStatus,
    ContradictionKind,
    ContradictionResolution,
    ConversationStatus,
    EvidenceRole,
    GapKind,
    MemoryObjectType,
    ProvenanceClass,
    SessionStatus,
)


def new_id(prefix: str) -> str:
    """Compact, collision-resistant id (e.g. ``clm_a1b2c3d4e5f6``)."""
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


def now_utc() -> datetime:
    return datetime.now(timezone.utc)


def stable_key(text: str) -> str:
    """Deterministic dedup key: normalized text -> sha1 hex.

    Keeps near-duplicate claims out of the store even across processes
    (same normalization => same key), without relying on embeddings.
    """
    return hashlib.sha1((text or "").strip().casefold().encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# L0 — Conversation
# ---------------------------------------------------------------------------

class ConversationRecord(BaseModel):
    id: str = ""
    user_id: str = ""
    status: ConversationStatus = ConversationStatus.ACTIVE
    started_at: datetime = Field(default_factory=now_utc)
    ended_at: datetime | None = None
    meta: dict[str, Any] = Field(default_factory=dict)


class MessageRecord(BaseModel):
    id: str = ""
    conversation_id: str = ""
    role: str = ""                       # user | assistant | system | tool
    content: str = ""
    created_at: datetime = Field(default_factory=now_utc)
    token_count: int = 0
    flags: dict[str, Any] = Field(default_factory=dict)   # citations, refs, ...


class ConversationSummaryRecord(BaseModel):
    id: str = ""
    conversation_id: str = ""
    summary: str = ""
    unresolved_refs: list[dict[str, Any]] = Field(default_factory=list)
    open_questions: list[str] = Field(default_factory=list)
    decisions: list[str] = Field(default_factory=list)
    version: int = 1
    created_at: datetime = Field(default_factory=now_utc)


# ---------------------------------------------------------------------------
# L1 / L2 — Research session machinery
# ---------------------------------------------------------------------------

class ResearchSessionRecord(BaseModel):
    id: str = ""
    user_id: str = ""
    conversation_id: str = ""   # UI chat id this research session is bound to
    title: str = ""
    topic_ids: list[str] = Field(default_factory=list)
    entity_ids: list[str] = Field(default_factory=list)
    status: SessionStatus = SessionStatus.ACTIVE
    merged_into: str = ""
    created_at: datetime = Field(default_factory=now_utc)
    first_activity_at: datetime = Field(default_factory=now_utc)
    last_activity_at: datetime = Field(default_factory=now_utc)
    summary: str = ""
    meta: dict[str, Any] = Field(default_factory=dict)


class ResearchQuestionRecord(BaseModel):
    id: str = ""
    session_id: str = ""
    question: str = ""
    parent_question_id: str = ""
    evidence_requirement: dict[str, Any] = Field(default_factory=dict)
    status: str = "open"                 # open | answered | superseded
    created_at: datetime = Field(default_factory=now_utc)
    resolved_at: datetime | None = None


# ---------------------------------------------------------------------------
# L2 — Evidence reference (the ONLY crossing object from L3)
# ---------------------------------------------------------------------------

class EvidenceReferenceRecord(BaseModel):
    """Pointer + provenance to verified evidence. NOT a copy of evidence.

    ``verification_status`` is stamped by the Critic pipeline, never by the
    memory layer. A claim may only link to references whose status is
    ``verified`` (enforced by validation.evidence_gate).
    """
    id: str = ""
    source: str = "PMC"
    external_id: str = ""                # PMCID / PMID / DOI
    id_type: str = ""                    # PMCID | PMID | DOI
    doi: str = ""
    pmid: str = ""
    chunk_id: str = ""
    title: str = ""
    verification_status: str = "unverified"   # set by Critic
    verified_by: str = ""
    verified_at: datetime | None = None
    retrieved_at: datetime | None = None
    indexed_at: datetime | None = None
    publication_date: datetime | None = None
    meta: dict[str, Any] = Field(default_factory=dict)

    @property
    def verified(self) -> bool:
        return self.verification_status == "verified"

    def citation(self) -> str:
        """Human-readable evidence handle for context rendering."""
        for key in ("external_id", "doi", "pmid"):
            if getattr(self, key):
                return f"{self.id_type or self.source}:{getattr(self, key)}"
        return f"{self.source}:{self.external_id or self.id}"


# ---------------------------------------------------------------------------
# L2 — Claims
# ---------------------------------------------------------------------------

class ClaimRecord(BaseModel):
    """A structured factual assertion with full provenance + temporal state.

    ``provenance_class`` is the single most important field: only
    EVIDENCE_DERIVED_CLAIM may ever be presented as evidence-backed.
    """
    id: str = ""
    session_id: str = ""
    user_id: str = ""
    text: str = ""
    normalized_text: str = ""
    dedup_key: str = ""                  # stable_key(normalized_text)
    provenance_class: ProvenanceClass = ProvenanceClass.MODEL_INFERENCE
    status: ClaimStatus = ClaimStatus.UNRESOLVED
    confidence: float = 0.0
    topic_ids: list[str] = Field(default_factory=list)
    entity_ids: list[str] = Field(default_factory=list)
    first_seen_at: datetime = Field(default_factory=now_utc)
    last_verified_at: datetime | None = None
    valid_from: datetime = Field(default_factory=now_utc)
    valid_to: datetime | None = None     # NULL = currently valid (as_of filter)
    superseded_by: str = ""
    superseded_at: datetime | None = None
    invalidated_at: datetime | None = None
    needs_revalidation: bool = False
    meta: dict[str, Any] = Field(default_factory=dict)

    @property
    def current(self) -> bool:
        """True when the claim is in its current (unsuperseded) version."""
        return self.valid_to is None and not self.superseded_by

    def as_of(self, moment: datetime) -> bool:
        return self.valid_from <= moment and (
            self.valid_to is None or self.valid_to > moment)


class ClaimEvidenceLinkRecord(BaseModel):
    """Provenance-bearing edge: claim ↔ evidence reference with a role.

    Only EVIDENCE_DERIVED_CLAIMs may carry SUPPORTED_BY links; CONTRADICTED_BY
    links may also point to verified evidence that opposes the claim.
    Edges are insert-only (audited via MemoryEvent).
    """
    id: str = ""
    claim_id: str = ""
    evidence_ref_id: str = ""
    role: EvidenceRole = EvidenceRole.SUPPORTED_BY
    weight: float = 1.0
    created_at: datetime = Field(default_factory=now_utc)


class ClaimRelationRecord(BaseModel):
    """Associative/temporal claim→claim edge (never proof)."""
    id: str = ""
    from_claim_id: str = ""
    to_claim_id: str = ""
    relation: ClaimRelationKind = ClaimRelationKind.RELATED_TO
    created_at: datetime = Field(default_factory=now_utc)


# ---------------------------------------------------------------------------
# L2 — Contradictions, gaps, preferences, audit
# ---------------------------------------------------------------------------

class ContradictionRecord(BaseModel):
    """A preserved conflict between two claims (or claim vs evidence).

    Never collapsed: both sides + their evidence ids + the dimensions that
    may explain the difference (population, intervention, comparator,
    outcome, design, follow-up, dosage, baseline, measurement, publication
    time). Differing results are only 'explained/spurious/resolved' via an
    explicit resolution, never by averaging.
    """
    id: str = ""
    session_id: str = ""
    claim: str = ""
    claim_a_id: str = ""
    claim_b_id: str = ""
    evidence_a_ids: list[str] = Field(default_factory=list)
    evidence_b_ids: list[str] = Field(default_factory=list)
    kind: ContradictionKind = ContradictionKind.DIRECT_CONFLICT
    dimensions: dict[str, str] = Field(default_factory=dict)
    resolution: ContradictionResolution = ContradictionResolution.UNRESOLVED
    explanation: str = ""
    first_seen_at: datetime = Field(default_factory=now_utc)
    resolved_at: datetime | None = None

    @property
    def unresolved(self) -> bool:
        return self.resolution == ContradictionResolution.UNRESOLVED


class ResearchGapRecord(BaseModel):
    id: str = ""
    session_id: str = ""
    question: str = ""
    kind: GapKind = GapKind.MISSING_EVIDENCE
    status: str = "open"                 # open | filled | superseded
    created_at: datetime = Field(default_factory=now_utc)
    filled_at: datetime | None = None
    meta: dict[str, Any] = Field(default_factory=dict)


class UserPreferenceRecord(BaseModel):
    id: str = ""
    user_id: str = ""
    key: str = ""                        # evidence_type | format | recency_bias | topic_interest ...
    value: Any = None
    source: str = "explicit"             # explicit | inferred
    confidence: float = 1.0
    created_at: datetime = Field(default_factory=now_utc)
    updated_at: datetime = Field(default_factory=now_utc)


class MemoryEventRecord(BaseModel):
    """Append-only audit record. Every write to persistent memory emits one."""
    id: str = ""
    actor: str = ""                      # memory_pipeline | critic | user | orchestrator
    event_type: str = ""                 # MemoryEventType value
    object_type: str = ""                # MemoryObjectType value
    object_id: str = ""
    payload: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=now_utc)


# ---------------------------------------------------------------------------
# Write pipeline: candidate / validation / lineage
# ---------------------------------------------------------------------------

class CandidateMemory(BaseModel):
    """Something the system thinks might be worth remembering.

    Deliberately distinct from committed records: a candidate carries the
    raw proposal + an intended provenance class + supporting evidence ref
    ids. It only becomes committed memory after validation.evidence_gate
    passes (or degrades to a labeled class).
    """
    id: str = ""
    object_type: MemoryObjectType = MemoryObjectType.CLAIM
    # The record payload (e.g. a ClaimRecord, UserPreferenceRecord ...).
    # Kept as dict so any record type can be proposed uniformly.
    payload: dict[str, Any] = Field(default_factory=dict)
    provenance_class: ProvenanceClass = ProvenanceClass.MODEL_INFERENCE
    evidence_ref_ids: list[str] = Field(default_factory=list)
    session_id: str = ""
    user_id: str = ""
    source_actor: str = "memory_pipeline"
    source_text: str = ""                # where it came from (audit)
    confidence: float = 0.0
    state: CandidateState = CandidateState.CANDIDATE
    proposed_at: datetime = Field(default_factory=now_utc)
    note: str = ""

    @classmethod
    def from_claim(cls, claim: ClaimRecord, **kw: Any) -> "CandidateMemory":
        return cls(
            object_type=MemoryObjectType.CLAIM,
            payload=claim.model_dump(),
            provenance_class=claim.provenance_class,
            session_id=claim.session_id,
            user_id=claim.user_id,
            **kw,
        )


class ValidationResult(BaseModel):
    """Outcome of the provenance gate + dedup pass for one candidate."""
    ok: bool = True
    committed: bool = False
    reasons: list[str] = Field(default_factory=list)
    degraded_to: ProvenanceClass | None = None
    dedup_match_id: str = ""             # existing claim this candidate duplicates
    candidate: CandidateMemory | None = None


class EvidenceLineage(BaseModel):
    """Full audit trail for one claim: claim -> links -> evidence -> paper."""
    claim: ClaimRecord
    links: list[ClaimEvidenceLinkRecord] = Field(default_factory=list)
    evidence_refs: list[EvidenceReferenceRecord] = Field(default_factory=list)

    def render(self) -> str:
        lines = [f"CLAIM: {self.claim.text}",
                 f"  class: {self.claim.provenance_class.value} | "
                 f"status: {self.claim.status.value} | "
                 f"first_seen: {self.claim.first_seen_at.isoformat()}"]

        if self.claim.last_verified_at:
            lines.append(f"  last_verified: {self.claim.last_verified_at.isoformat()}")
        if self.claim.superseded_at:
            lines.append(f"  superseded: {self.claim.superseded_at.isoformat()} "
                         f"(by {self.claim.superseded_by})")
        if not self.links:
            lines.append("  evidence links: (none — NOT evidence-backed)")
            return "\n".join(lines)
        lines.append("  evidence links:")
        for link, ref in zip(self.links, self.evidence_refs):
            lines.append(
                f"    [{link.role.value}] {ref.citation()} "
                f"(verified={ref.verified}, chunk={ref.chunk_id or 'paper'})")
        return "\n".join(lines)


__all__ = [
    "CandidateMemory",
    "ClaimEvidenceLinkRecord",
    "ClaimRecord",
    "ClaimRelationRecord",
    "ConversationRecord",
    "ConversationSummaryRecord",
    "ContradictionRecord",
    "EvidenceLineage",
    "EvidenceReferenceRecord",
    "MemoryEventRecord",
    "MessageRecord",
    "ResearchGapRecord",
    "ResearchQuestionRecord",
    "ResearchSessionRecord",
    "UserPreferenceRecord",
    "ValidationResult",
    "new_id",
    "now_utc",
    "stable_key",
]