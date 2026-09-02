"""Memory + Context layer — enumerations.

Central principle (MedPat invariant):

    Memory is not medical evidence.

The PMC corpus + verified-evidence pipeline remain the authoritative source
for medical factual claims. Memory provides context, continuity, prior
research state and personalization — but it must never silently become an
alternative source of unsupported medical truth.

The enums below encode that invariant structurally: every persisted medical
claim carries a ``ProvenanceClass``, and only ``EVIDENCE_DERIVED_CLAIM`` may
be treated as evidence-backed in downstream context. The write pipeline
(validation.py) enforces the provenance gate so the classes cannot be
forged.

This module contains NO LLM / corpus / DB code — pure data, unit-testable
offline (same philosophy as src.agentic.state).
"""

from __future__ import annotations

import enum


class ProvenanceClass(str, enum.Enum):
    """How a piece of stored information entered the system.

    These five classes are NOT interchangeable. Downstream context rendering
    (context.py) uses them to decide whether an item may be presented as
    evidence-backed or must stay labeled as memory:
      * USER_ASSERTION          — the user's stated belief; never medical truth.
      * MODEL_INFERENCE         — LLM-generated, no evidence backing.
      * UNVERIFIED_INFORMATION  — retrieved but not yet Critic-verified.
      * VERIFIED_EVIDENCE       — Critic-verified evidence (lives in L3,
                                  referenced from memory, never copied).
      * EVIDENCE_DERIVED_CLAIM  — a claim whose text is backed by
                                  VERIFIED_EVIDENCE links (provenance gate).
    """
    USER_ASSERTION = "user_assertion"
    MODEL_INFERENCE = "model_inference"
    UNVERIFIED_INFORMATION = "unverified_information"
    VERIFIED_EVIDENCE = "verified_evidence"
    EVIDENCE_DERIVED_CLAIM = "evidence_derived_claim"

    @property
    def evidence_backed(self) -> bool:
        """True only for classes that may be presented as evidence-backed."""
        return self in (
            ProvenanceClass.VERIFIED_EVIDENCE,
            ProvenanceClass.EVIDENCE_DERIVED_CLAIM,
        )


class ClaimStatus(str, enum.Enum):
    """State machine of a persisted claim (temporal, never overwritten).

    * SUPPORTED     — supported by current verified evidence.
    * CONTRADICTED  — contradicted by current verified evidence.
    * MIXED         — both supporting and contradicting verified evidence.
    * SUPERSEDED    — replaced by a newer claim (old one stays queryable).
    * INVALIDATED   — shown wrong by new evidence.
    * UNRESOLVED    — evidence exists but support is not settled.
    """
    SUPPORTED = "supported"
    CONTRADICTED = "contradicted"
    MIXED = "mixed"
    SUPERSEDED = "superseded"
    INVALIDATED = "invalidated"
    UNRESOLVED = "unresolved"


class EvidenceRole(str, enum.Enum):
    """Direction of a claim→evidence link. These are provenance-bearing edges
    and are stored separately from associative claim→claim edges."""
    SUPPORTED_BY = "supported_by"
    CONTRADICTED_BY = "contradicted_by"


class ClaimRelationKind(str, enum.Enum):
    """Associative / temporal claim→claim edges.

    * REFINED_BY   — a sharper claim replaces the scope of an older one.
    * SUPERSEDES   — temporal replacement (old claim still historically true).
    * DERIVED_FROM — inference lineage (claim ← earlier claim).
    * RELATED_TO   — associative, for retrieval only (never proof).
    """
    REFINED_BY = "refined_by"
    SUPERSEDES = "supersedes"
    DERIVED_FROM = "derived_from"
    RELATED_TO = "related_to"


class ContradictionResolution(str, enum.Enum):
    """State of a contradiction. Contradictions are preserved, never
    silently collapsed; differing results only become 'explained' or
    'spurious' via explicit dimension attribution."""
    UNRESOLVED = "unresolved"
    EXPLAINED = "explained"     # dimension-attributed (population/design/dose...)
    SPURIOUS = "spurious"       # not a real conflict (methodology/measurement)
    RESOLVED = "resolved"       # superseded by new evidence


class ContradictionKind(str, enum.Enum):
    DIRECT_CONFLICT = "direct_conflict"        # same claim, opposite findings
    CONTEXT_DEPENDENT = "context_dependent"    # differs by population/design/dose
    ANOMALY = "anomaly"                        # outlier / irregularity


class GapKind(str, enum.Enum):
    """Why a research question remains open."""
    MISSING_EVIDENCE = "missing_evidence"
    CONFLICTING_EVIDENCE = "conflicting_evidence"
    OUTDATED_EVIDENCE = "outdated_evidence"


class SessionStatus(str, enum.Enum):
    ACTIVE = "active"
    CLOSED = "closed"
    ARCHIVED = "archived"
    MERGED = "merged"


class ConversationStatus(str, enum.Enum):
    ACTIVE = "active"
    ARCHIVED = "archived"


class MemoryObjectType(str, enum.Enum):
    """Every persisted object kind; used by the append-only audit log."""
    CONVERSATION = "conversation"
    MESSAGE = "message"
    CONVERSATION_SUMMARY = "conversation_summary"
    RESEARCH_SESSION = "research_session"
    RESEARCH_QUESTION = "research_question"
    EVIDENCE_REFERENCE = "evidence_reference"
    CLAIM = "claim"
    CLAIM_EVIDENCE_LINK = "claim_evidence_link"
    CLAIM_RELATION = "claim_relation"
    CONTRADICTION = "contradiction"
    RESEARCH_GAP = "research_gap"
    USER_PREFERENCE = "user_preference"
    MEMORY_EVENT = "memory_event"


class MemoryEventType(str, enum.Enum):
    """Audit log event types (append-only, insert-only provenance edges)."""
    PROPOSE = "propose"
    VALIDATE = "validate"
    REJECT = "reject"
    COMMIT = "commit"
    UPDATE = "update"
    SUPERSEDE = "supersede"
    INVALIDATE = "invalidate"
    MARK_STALE = "mark_stale"
    MERGE = "merge"
    SESSION_CREATE = "session_create"
    SESSION_RESUME = "session_resume"
    SESSION_CLOSE = "session_close"
    SESSION_ARCHIVE = "session_archive"
    SESSION_MERGE = "session_merge"


class CandidateState(str, enum.Enum):
    """The candidate → committed lifecycle (write pipeline)."""
    CANDIDATE = "candidate"
    VALIDATED = "validated"
    REJECTED = "rejected"
    COMMITTED = "committed"


class MemoryContextKind(str, enum.Enum):
    """Typed context blocks passed to downstream prompts."""
    CURRENT_QUERY = "current_query"
    CONVERSATION = "conversation"
    RESEARCH_STATE = "research_state"
    PERSISTENT_MEMORY = "persistent_memory"
    EVIDENCE = "evidence"
    USER_PREFERENCE = "user_preference"


# Contradiction dimension vocabulary (PICO + timing/measurement). Differing
# results are attributed along these axes before they may be called a
# contradiction at all.
CONTRADICTION_DIMENSIONS = (
    "population",
    "intervention",
    "comparator",
    "outcome",
    "study_design",
    "follow_up",
    "dosage",
    "baseline",
    "measurement",
    "publication_time",
)

__all__ = [
    "CONTRADICTION_DIMENSIONS",
    "CandidateState",
    "ClaimRelationKind",
    "ClaimStatus",
    "ContradictionKind",
    "ContradictionResolution",
    "ConversationStatus",
    "EvidenceRole",
    "GapKind",
    "MemoryContextKind",
    "MemoryEventType",
    "MemoryObjectType",
    "ProvenanceClass",
    "SessionStatus",
]