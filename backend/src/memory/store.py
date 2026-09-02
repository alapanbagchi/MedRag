"""Memory + Context layer — storage backends.

One store interface, two implementations:

* ``InMemoryMemoryStore``  — dict-backed; zero dependencies; used by tests and
  local runs without a database (``MEMORY_BACKEND=memory``).
* ``PostgresMemoryStore`` — the production store: schema ``medrag_memory`` in
  the same PostgreSQL instance as the corpus (pgvector extension optional).
  Relational columns are the truth; pgvector ``emb`` columns exist for future
  scalable dense recall (the v1 retrieval path scores in Python over SQL-
  filtered candidates so both backends behave identically and are unit-tested
  offline).

Schema (matches DESIGN §12; see ``PostgresMemoryStore.SCHEMA``):

    users / conversations / messages / conversation_summaries
    research_sessions / research_questions
    evidence_references            (the ONLY crossing object from L3)
    claims / claim_evidence_links / claim_relations
    contradictions / research_gaps
    memory_events (append-only audit) / user_preferences

Provenance rules enforced at the SQL level too:
  * claim_evidence_links.role is CHECK-constrained to the two evidence roles;
  * claim_relations.relation is CHECK-constrained to the four claim relations;
  * memory_events is append-only by convention (no UPDATE path exposed).

Timestamps: all ``*_at`` are TIMESTAMPTZ; claims carry bi-temporal
``valid_from``/``valid_to`` (NULL = currently valid) so history is queryable
with ``as_of`` and never overwritten.
"""

from __future__ import annotations

import abc
import json
import logging
import os
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any, Iterator, Sequence

from src.memory.enums import (
    CandidateState,
    ClaimRelationKind,
    ClaimStatus,
    ContradictionKind,
    ContradictionResolution,
    ConversationStatus,
    EvidenceRole,
    GapKind,
    MemoryEventType,
    MemoryObjectType,
    ProvenanceClass,
    SessionStatus,
)
from src.memory.models import (
    ClaimEvidenceLinkRecord,
    ClaimRecord,
    ClaimRelationRecord,
    ConversationRecord,
    ConversationSummaryRecord,
    ContradictionRecord,
    EvidenceReferenceRecord,
    MemoryEventRecord,
    MessageRecord,
    ResearchGapRecord,
    ResearchQuestionRecord,
    ResearchSessionRecord,
    UserPreferenceRecord,
    new_id,
)

logger = logging.getLogger("src.memory.store")

MIN_MATCH_LEN = 4


def _utc() -> datetime:
    return datetime.now(timezone.utc)


def _as_utc(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    return value


def _json_dumps(value: Any) -> str:
    return json.dumps(value, default=str)


def _json_loads(value: str | None, default: Any) -> Any:
    if value is None:
        return default
    try:
        return json.loads(value)
    except Exception:
        return default


class DuplicateEvidenceReference(Exception):
    """An evidence reference with the same (external_id, id_type, chunk_id)
    already exists — the pipeline deduped it."""
    pass


# ---------------------------------------------------------------------------
# Abstract store
# ---------------------------------------------------------------------------

class MemoryStore(abc.ABC):
    """Minimal storage contract used by MemoryAPI / retrieval / validation.

    Every method is synchronous. Multi-row writes that must be atomic
    (claim + links + audit event) are wrapped in ``transaction()``.
    """

    # -- conversations ---------------------------------------------------

    @abc.abstractmethod
    def start_conversation(self, user_id: str = "") -> ConversationRecord: ...

    @abc.abstractmethod
    def ensure_conversation(self, conversation_id: str,
                            user_id: str = "") -> ConversationRecord: ...

    @abc.abstractmethod
    def get_conversation(self, conversation_id: str) -> ConversationRecord | None: ...

    @abc.abstractmethod
    def add_message(self, conversation_id: str, role: str, content: str,
                    **flags: Any) -> MessageRecord: ...

    @abc.abstractmethod
    def get_messages(self, conversation_id: str, limit: int = 20) -> list[MessageRecord]: ...

    @abc.abstractmethod
    def set_conversation_summary(self, conversation_id: str, summary: str,
                                 open_questions: list[str] | None = None,
                                 unresolved_refs: list[dict] | None = None,
                                 decisions: list[str] | None = None) -> ConversationSummaryRecord: ...

    @abc.abstractmethod
    def get_conversation_summary(self, conversation_id: str) -> ConversationSummaryRecord | None: ...

    # -- research sessions ----------------------------------------------

    @abc.abstractmethod
    def create_session(self, seed_question: str, user_id: str = "",
                       conversation_id: str = "", title: str = "") -> ResearchSessionRecord: ...

    @abc.abstractmethod
    def get_session_for_conversation(self, conversation_id: str,
                                     status: SessionStatus | None = None) -> ResearchSessionRecord | None: ...

    @abc.abstractmethod
    def get_session(self, session_id: str) -> ResearchSessionRecord | None: ...

    @abc.abstractmethod
    def touch_session(self, session_id: str) -> None: ...

    @abc.abstractmethod
    def update_session_summary(self, session_id: str, summary: str) -> None: ...

    @abc.abstractmethod
    def close_session(self, session_id: str) -> None: ...

    @abc.abstractmethod
    def archive_session(self, session_id: str) -> None: ...

    @abc.abstractmethod
    def merge_sessions(self, from_id: str, into_id: str) -> None: ...

    @abc.abstractmethod
    def list_sessions(self, user_id: str = "",
                      status: SessionStatus | None = None,
                      conversation_id: str | None = None) -> list[ResearchSessionRecord]: ...

    @abc.abstractmethod
    def find_sessions(self, query: str, user_id: str = "",
                      limit: int = 5) -> list[tuple[ResearchSessionRecord, float]]: ...

    # -- research questions ---------------------------------------------

    @abc.abstractmethod
    def add_question(self, session_id: str, question: str,
                     requirement: dict | None = None) -> ResearchQuestionRecord: ...

    @abc.abstractmethod
    def get_questions(self, session_id: str) -> list[ResearchQuestionRecord]: ...

    @abc.abstractmethod
    def mark_question_answered(self, question_id: str) -> None: ...

    # -- evidence references (L3 crossing objects) -----------------------

    @abc.abstractmethod
    def add_evidence_ref(self, ref: EvidenceReferenceRecord) -> EvidenceReferenceRecord: ...

    @abc.abstractmethod
    def get_evidence_ref(self, ref_id: str) -> EvidenceReferenceRecord | None: ...

    @abc.abstractmethod
    def get_evidence_refs(self, ref_ids: Sequence[str]) -> list[EvidenceReferenceRecord]: ...

    @abc.abstractmethod
    def evidence_ref_by_external(self, external_id: str,
                                 id_type: str = "PMCID") -> EvidenceReferenceRecord | None: ...

    # -- claims ----------------------------------------------------------

    @abc.abstractmethod
    def add_claim(self, claim: ClaimRecord) -> ClaimRecord: ...

    @abc.abstractmethod
    def get_claim(self, claim_id: str) -> ClaimRecord | None: ...

    @abc.abstractmethod
    def update_claim(self, claim_id: str, **patch: Any) -> ClaimRecord | None: ...

    @abc.abstractmethod
    def get_claims(self, session_id: str | None = None,
                   provenance: ProvenanceClass | Sequence[ProvenanceClass] | None = None,
                   status: ClaimStatus | None = None,
                   include_superseded: bool = False) -> list[ClaimRecord]: ...

    @abc.abstractmethod
    def scan_claims(self, session_ids: Sequence[str] | None = None,
                    provenance: ProvenanceClass | Sequence[ProvenanceClass] | None = None,
                    include_stale: bool = False,
                    include_superseded: bool = False) -> list[ClaimRecord]: ...

    @abc.abstractmethod
    def claims_by_dedup_key(self, key: str) -> list[ClaimRecord]: ...

    # -- claim edges -----------------------------------------------------

    @abc.abstractmethod
    def add_claim_evidence_link(self, link: ClaimEvidenceLinkRecord) -> ClaimEvidenceLinkRecord: ...

    @abc.abstractmethod
    def get_claim_evidence_links(self, claim_id: str) -> list[ClaimEvidenceLinkRecord]: ...

    @abc.abstractmethod
    def get_links_for_claim(self, claim_id: str) -> list[ClaimEvidenceLinkRecord]: ...

    @abc.abstractmethod
    def add_claim_relation(self, rel: ClaimRelationRecord) -> ClaimRelationRecord: ...

    @abc.abstractmethod
    def get_claim_relations(self, claim_id: str,
                            relation: ClaimRelationKind | None = None) -> list[ClaimRelationRecord]: ...

    # -- contradictions --------------------------------------------------

    @abc.abstractmethod
    def add_contradiction(self, rec: ContradictionRecord) -> ContradictionRecord: ...

    @abc.abstractmethod
    def get_contradictions(self, session_id: str | None = None,
                           unresolved_only: bool = False) -> list[ContradictionRecord]: ...

    @abc.abstractmethod
    def resolve_contradiction(self, contradiction_id: str,
                              resolution: ContradictionResolution,
                              explanation: str = "") -> None: ...

    # -- research gaps ---------------------------------------------------

    @abc.abstractmethod
    def add_gap(self, rec: ResearchGapRecord) -> ResearchGapRecord: ...

    @abc.abstractmethod
    def get_gaps(self, session_id: str | None = None,
                 open_only: bool = False) -> list[ResearchGapRecord]: ...

    @abc.abstractmethod
    def fill_gap(self, gap_id: str) -> None: ...

    # -- preferences -----------------------------------------------------

    @abc.abstractmethod
    def set_preference(self, user_id: str, key: str, value: Any,
                       source: str = "explicit", confidence: float = 1.0) -> UserPreferenceRecord: ...

    @abc.abstractmethod
    def get_preferences(self, user_id: str) -> list[UserPreferenceRecord]: ...

    # -- audit -----------------------------------------------------------

    @abc.abstractmethod
    def add_event(self, actor: str, event_type: MemoryEventType,
                  object_type: MemoryObjectType, object_id: str,
                  payload: dict | None = None) -> MemoryEventRecord: ...

    @abc.abstractmethod
    def recent_events(self, limit: int = 50) -> list[MemoryEventRecord]: ...

    # -- plumbing --------------------------------------------------------

    @abc.abstractmethod
    def counts(self) -> dict[str, int]: ...

    @contextmanager
    def transaction(self) -> Iterator[None]:
        yield


# ---------------------------------------------------------------------------
# In-memory backend (tests / local mode)
# ---------------------------------------------------------------------------

class InMemoryMemoryStore(MemoryStore):
    """Dict-backed store. Deterministic, dependency-free. Useful for tests,
    local research, and as the ``memory`` backend when no DB is available."""

    def __init__(self) -> None:
        self.conversations: dict[str, ConversationRecord] = {}
        self.messages: dict[str, list[MessageRecord]] = {}
        self.summaries: dict[str, ConversationSummaryRecord] = {}
        self.sessions: dict[str, ResearchSessionRecord] = {}
        self.questions: dict[str, list[ResearchQuestionRecord]] = {}
        self.evidence_refs: dict[str, EvidenceReferenceRecord] = {}
        self._evidence_by_external: dict[tuple[str, str], str] = {}
        self.claims: dict[str, ClaimRecord] = {}
        self.links: dict[str, list[ClaimEvidenceLinkRecord]] = {}
        self.relations: dict[str, list[ClaimRelationRecord]] = {}
        self.contradictions: dict[str, ContradictionRecord] = {}
        self.gaps: dict[str, ResearchGapRecord] = {}
        self.preferences: dict[tuple[str, str], UserPreferenceRecord] = {}
        self.events: list[MemoryEventRecord] = []

    # -- conversations ---------------------------------------------------

    def start_conversation(self, user_id: str = "") -> ConversationRecord:
        rec = ConversationRecord(id=new_id("con"), user_id=user_id)
        self.conversations[rec.id] = rec
        self.messages.setdefault(rec.id, [])
        return rec

    def ensure_conversation(self, conversation_id: str,
                            user_id: str = "") -> ConversationRecord:
        """Return the conversation with this id, creating it if needed
        (external ids, e.g. frontend conversation ids, are preserved)."""
        existing = self.conversations.get(conversation_id)
        if existing is not None:
            return existing
        rec = ConversationRecord(id=conversation_id, user_id=user_id)
        self.conversations[rec.id] = rec
        self.messages.setdefault(rec.id, [])
        return rec

    def get_conversation(self, conversation_id: str) -> ConversationRecord | None:
        return self.conversations.get(conversation_id)

    def add_message(self, conversation_id: str, role: str, content: str,
                    **flags: Any) -> MessageRecord:
        rec = MessageRecord(id=new_id("msg"), conversation_id=conversation_id,
                            role=role, content=content, flags=dict(flags))
        self.messages.setdefault(conversation_id, []).append(rec)
        return rec

    def get_messages(self, conversation_id: str, limit: int = 20) -> list[MessageRecord]:
        return self.messages.get(conversation_id, [])[-limit:]

    def set_conversation_summary(self, conversation_id: str, summary: str,
                                 open_questions: list[str] | None = None,
                                 unresolved_refs: list[dict] | None = None,
                                 decisions: list[str] | None = None) -> ConversationSummaryRecord:
        prev = self.summaries.get(conversation_id)
        rec = ConversationSummaryRecord(
            id=new_id("cs"), conversation_id=conversation_id, summary=summary,
            open_questions=open_questions or [],
            unresolved_refs=unresolved_refs or [], decisions=decisions or [],
            version=(prev.version + 1) if prev else 1,
        )
        self.summaries[conversation_id] = rec
        return rec

    def get_conversation_summary(self, conversation_id: str) -> ConversationSummaryRecord | None:
        return self.summaries.get(conversation_id)

    # -- sessions --------------------------------------------------------

    def create_session(self, seed_question: str, user_id: str = "",
                       conversation_id: str = "", title: str = "") -> ResearchSessionRecord:
        rec = ResearchSessionRecord(
            id=new_id("rs"), user_id=user_id, conversation_id=conversation_id,
            title=title or (seed_question or "")[:90],
            summary=seed_question or "",
        )
        self.sessions[rec.id] = rec
        return rec

    def get_session_for_conversation(self, conversation_id: str,
                                     status: SessionStatus | None = None) -> ResearchSessionRecord | None:
        if not conversation_id:
            return None
        for rec in self.sessions.values():
            if rec.conversation_id != conversation_id:
                continue
            if status and rec.status != status:
                continue
            return rec
        return None

    def get_session(self, session_id: str) -> ResearchSessionRecord | None:
        return self.sessions.get(session_id)

    def touch_session(self, session_id: str) -> None:
        rec = self.sessions.get(session_id)
        if rec:
            rec.last_activity_at = _utc()

    def update_session_summary(self, session_id: str, summary: str) -> None:
        rec = self.sessions.get(session_id)
        if rec and summary:
            rec.summary = summary[:600]

    def close_session(self, session_id: str) -> None:
        rec = self.sessions.get(session_id)
        if rec:
            rec.status = SessionStatus.CLOSED

    def archive_session(self, session_id: str) -> None:
        rec = self.sessions.get(session_id)
        if rec:
            rec.status = SessionStatus.ARCHIVED

    def merge_sessions(self, from_id: str, into_id: str) -> None:
        src = self.sessions.get(from_id)
        dst = self.sessions.get(into_id)
        if src and dst:
            src.status = SessionStatus.MERGED
            src.merged_into = into_id
            for q in self.questions.get(from_id, []):
                q.session_id = into_id
                self.questions.setdefault(into_id, []).append(q)
            self.questions[from_id] = []
            for claim in self.claims.values():
                if claim.session_id == from_id:
                    claim.session_id = into_id
            dst.last_activity_at = _utc()

    def list_sessions(self, user_id: str = "",
                      status: SessionStatus | None = None,
                      conversation_id: str | None = None) -> list[ResearchSessionRecord]:
        out = []
        for rec in self.sessions.values():
            if user_id and rec.user_id and rec.user_id != user_id:
                continue
            if conversation_id and rec.conversation_id != conversation_id:
                continue
            if status and rec.status != status:
                continue
            out.append(rec)
        out.sort(key=lambda r: r.last_activity_at, reverse=True)
        return out

    def find_sessions(self, query: str, user_id: str = "",
                      limit: int = 5,
                      conversation_id: str | None = None) -> list[tuple[ResearchSessionRecord, float]]:
        q = _token_overlap(query)
        scored = []
        for rec in self.list_sessions(user_id=user_id, status=SessionStatus.ACTIVE):
            if conversation_id and rec.conversation_id != conversation_id:
                continue    # never resume a session from a DIFFERENT chat
            texts = [rec.title, rec.summary]
            texts += [q_.question for q_ in self.questions.get(rec.id, [])]
            best = max((_token_overlap(t) & q) for t in texts if t) or set()
            score = len(best) / max(1.0, len(q))
            if score > 0:
                scored.append((rec, float(score)))
        scored.sort(key=lambda t: t[1], reverse=True)
        return scored[:limit]

    # -- questions -------------------------------------------------------

    def add_question(self, session_id: str, question: str,
                     requirement: dict | None = None) -> ResearchQuestionRecord:
        rec = ResearchQuestionRecord(
            id=new_id("rq"), session_id=session_id, question=question,
            evidence_requirement=requirement or {})
        self.questions.setdefault(session_id, []).append(rec)
        return rec

    def get_questions(self, session_id: str) -> list[ResearchQuestionRecord]:
        return self.questions.get(session_id, [])

    def mark_question_answered(self, question_id: str) -> None:
        for lst in self.questions.values():
            for q in lst:
                if q.id == question_id:
                    q.status = "answered"
                    q.resolved_at = _utc()

    # -- evidence refs ---------------------------------------------------

    def add_evidence_ref(self, ref: EvidenceReferenceRecord) -> EvidenceReferenceRecord:
        key = (ref.external_id or "", ref.id_type or "", ref.chunk_id or "")
        existing = self._evidence_by_external.get(key)
        if existing and key[0]:
            # same external id re-verified later: refresh the record in place
            prev = self.evidence_refs[existing]
            ref.id = existing
            if ref.verified and not prev.verified:
                self.evidence_refs[existing] = ref
            return self.evidence_refs[existing]
        if not ref.id:
            ref.id = new_id("ev")
        self.evidence_refs[ref.id] = ref
        if key[0]:
            self._evidence_by_external[key] = ref.id
        return ref

    def get_evidence_ref(self, ref_id: str) -> EvidenceReferenceRecord | None:
        return self.evidence_refs.get(ref_id)

    def get_evidence_refs(self, ref_ids: Sequence[str]) -> list[EvidenceReferenceRecord]:
        return [self.evidence_refs[i] for i in ref_ids if i in self.evidence_refs]

    def evidence_ref_by_external(self, external_id: str,
                                 id_type: str = "PMCID") -> EvidenceReferenceRecord | None:
        rid = self._evidence_by_external.get((external_id, id_type, ""))
        return self.evidence_refs.get(rid) if rid else None

    # -- claims ----------------------------------------------------------

    def add_claim(self, claim: ClaimRecord) -> ClaimRecord:
        if not claim.id:
            claim.id = new_id("clm")
        if not claim.normalized_text:
            from src.memory.validation import normalized_claim_text
            claim.normalized_text = normalized_claim_text(claim.text)
        if not claim.dedup_key:
            from src.memory.validation import claim_dedup_key
            claim.dedup_key = claim_dedup_key(claim)
        self.claims[claim.id] = claim
        return claim

    def get_claim(self, claim_id: str) -> ClaimRecord | None:
        return self.claims.get(claim_id)

    def update_claim(self, claim_id: str, **patch: Any) -> ClaimRecord | None:
        rec = self.claims.get(claim_id)
        if rec is None:
            return None
        for k, v in patch.items():
            if k == "status" and isinstance(v, str):
                v = ClaimStatus(v)
            if k == "provenance_class" and isinstance(v, str):
                v = ProvenanceClass(v)
            setattr(rec, k, v)
        return rec

    def get_claims(self, session_id: str | None = None,
                   provenance: ProvenanceClass | Sequence[ProvenanceClass] | None = None,
                   status: ClaimStatus | None = None,
                   include_superseded: bool = False) -> list[ClaimRecord]:
        out = []
        for rec in self.claims.values():
            if session_id and rec.session_id != session_id:
                continue
            if provenance is not None:
                wanted = provenance if isinstance(provenance, (list, tuple, set)) else (provenance,)
                if rec.provenance_class not in wanted:
                    continue
            if status and rec.status != status:
                continue
            if not include_superseded and not rec.current:
                continue
            out.append(rec)
        return out

    def scan_claims(self, session_ids: Sequence[str] | None = None,
                    provenance: ProvenanceClass | Sequence[ProvenanceClass] | None = None,
                    include_stale: bool = False,
                    include_superseded: bool = False) -> list[ClaimRecord]:
        out = []
        for rec in self.claims.values():
            if session_ids and rec.session_id not in session_ids:
                continue
            if provenance is not None:
                wanted = provenance if isinstance(provenance, (list, tuple, set)) else (provenance,)
                if rec.provenance_class not in wanted:
                    continue
            if not include_stale and rec.needs_revalidation:
                continue
            if not include_superseded and not rec.current:
                continue
            out.append(rec)
        return out

    def claims_by_dedup_key(self, key: str) -> list[ClaimRecord]:
        return [c for c in self.claims.values() if c.dedup_key == key]

    # -- edges -----------------------------------------------------------

    def add_claim_evidence_link(self, link: ClaimEvidenceLinkRecord) -> ClaimEvidenceLinkRecord:
        if not link.id:
            link.id = new_id("cel")
        self.links.setdefault(link.claim_id, []).append(link)
        return link

    def get_claim_evidence_links(self, claim_id: str) -> list[ClaimEvidenceLinkRecord]:
        return self.links.get(claim_id, [])

    def get_links_for_claim(self, claim_id: str) -> list[ClaimEvidenceLinkRecord]:
        return self.get_claim_evidence_links(claim_id)

    def add_claim_relation(self, rel: ClaimRelationRecord) -> ClaimRelationRecord:
        if not rel.id:
            rel.id = new_id("cr")
        self.relations.setdefault(rel.from_claim_id, []).append(rel)
        return rel

    def get_claim_relations(self, claim_id: str,
                            relation: ClaimRelationKind | None = None) -> list[ClaimRelationRecord]:
        out = list(self.relations.get(claim_id, []))
        if relation:
            out = [r for r in out if r.relation == relation]
        return out

    # -- contradictions --------------------------------------------------

    def add_contradiction(self, rec: ContradictionRecord) -> ContradictionRecord:
        if not rec.id:
            rec.id = new_id("ctr")
        self.contradictions[rec.id] = rec
        return rec

    def get_contradictions(self, session_id: str | None = None,
                           unresolved_only: bool = False) -> list[ContradictionRecord]:
        out = []
        for rec in self.contradictions.values():
            if session_id and rec.session_id != session_id:
                continue
            if unresolved_only and not rec.unresolved:
                continue
            out.append(rec)
        return out

    def resolve_contradiction(self, contradiction_id: str,
                              resolution: ContradictionResolution,
                              explanation: str = "") -> None:
        rec = self.contradictions.get(contradiction_id)
        if rec:
            rec.resolution = resolution
            rec.explanation = explanation
            rec.resolved_at = _utc() if resolution != ContradictionResolution.UNRESOLVED else None

    # -- gaps ------------------------------------------------------------

    def add_gap(self, rec: ResearchGapRecord) -> ResearchGapRecord:
        if not rec.id:
            rec.id = new_id("gap")
        self.gaps[rec.id] = rec
        return rec

    def get_gaps(self, session_id: str | None = None,
                 open_only: bool = False) -> list[ResearchGapRecord]:
        out = []
        for rec in self.gaps.values():
            if session_id and rec.session_id != session_id:
                continue
            if open_only and rec.status != "open":
                continue
            out.append(rec)
        return out

    def fill_gap(self, gap_id: str) -> None:
        rec = self.gaps.get(gap_id)
        if rec and rec.status == "open":
            rec.status = "filled"
            rec.filled_at = _utc()

    # -- preferences -----------------------------------------------------

    def set_preference(self, user_id: str, key: str, value: Any,
                       source: str = "explicit", confidence: float = 1.0) -> UserPreferenceRecord:
        pk = (user_id, key)
        existing = self.preferences.get(pk)
        rec = UserPreferenceRecord(
            id=existing.id if existing else new_id("pref"),
            user_id=user_id, key=key, value=value, source=source,
            confidence=confidence,
            updated_at=_utc(),
            created_at=existing.created_at if existing else _utc(),
        )
        self.preferences[pk] = rec
        return rec

    def get_preferences(self, user_id: str) -> list[UserPreferenceRecord]:
        return [rec for (uid, _), rec in self.preferences.items() if uid == user_id]

    # -- audit -----------------------------------------------------------

    def add_event(self, actor: str, event_type: MemoryEventType,
                  object_type: MemoryObjectType, object_id: str,
                  payload: dict | None = None) -> MemoryEventRecord:
        rec = MemoryEventRecord(id=new_id("evt"), actor=actor,
                                event_type=event_type.value,
                                object_type=object_type.value,
                                object_id=object_id, payload=payload or {})
        self.events.append(rec)
        return rec

    def recent_events(self, limit: int = 50) -> list[MemoryEventRecord]:
        return self.events[-limit:][::-1]

    def counts(self) -> dict[str, int]:
        return {
            "conversations": len(self.conversations),
            "messages": sum(len(v) for v in self.messages.values()),
            "sessions": len(self.sessions),
            "questions": sum(len(v) for v in self.questions.values()),
            "evidence_refs": len(self.evidence_refs),
            "claims": len(self.claims),
            "claim_links": sum(len(v) for v in self.links.values()),
            "claim_relations": sum(len(v) for v in self.relations.values()),
            "contradictions": len(self.contradictions),
            "gaps": len(self.gaps),
            "preferences": len(self.preferences),
            "events": len(self.events),
        }


def _token_overlap(text: str) -> set[str]:
    import re
    return set(re.findall(r"[a-z0-9]{3,}", (text or "").casefold()))


# ---------------------------------------------------------------------------
# PostgreSQL backend
# ---------------------------------------------------------------------------

class PostgresMemoryStore(MemoryStore):
    """PostgreSQL + optional pgvector backend for the memory layer.

    Connection uses the same environment variables as the corpus store
    (PGHOST/PGPORT/PGUSER/PGPASSWORD/PGDATABASE). Schema is created
    idempotently by ``ensure_schema()`` — run it once via
    ``scripts/memory_init.py`` or lazily on first connect.
    """

    SCHEMA_OWNER = "medrag_memory"

    def __init__(self, dsn: str | None = None, dim: int = 256):
        if dsn is None:
            from src.retrieval.pgvector_store import PgConfig
            dsn = PgConfig.from_env().dsn()
        self.dsn = dsn
        self.dim = int(dim)
        self._conn = None
        self.has_dense = False
        self.schema = self.SCHEMA_OWNER

    # ------------------------------------------------------------------
    # Connection
    # ------------------------------------------------------------------

    def connect(self):
        if self._conn is not None and not self._conn.closed:
            return self._conn
        import psycopg2
        self._conn = psycopg2.connect(self.dsn)
        self._conn.autocommit = True        # per-statement; transactions via transaction()
        return self._conn

    def close(self) -> None:
        if self._conn is not None and not self._conn.closed:
            self._conn.close()
        self._conn = None

    def __enter__(self) -> "PostgresMemoryStore":
        self.connect()
        self.ensure_schema()
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    @contextmanager
    def transaction(self) -> Iterator[None]:
        conn = self.connect()
        conn.autocommit = False
        try:
            yield
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.autocommit = True

    def _execute(self, sql: str, params: tuple | list = ()) -> Any:
        conn = self.connect()
        cur = conn.cursor()
        try:
            cur.execute(sql, params)
            return cur
        except Exception:
            conn.rollback()
            raise

    def _fetchone(self, sql: str, params: tuple = ()) -> tuple | None:
        cur = self._execute(sql, params)
        try:
            return cur.fetchone()
        finally:
            cur.close()

    def _fetchall(self, sql: str, params: tuple = ()) -> list[tuple]:
        cur = self._execute(sql, params)
        try:
            return cur.fetchall()
        finally:
            cur.close()

    # ------------------------------------------------------------------
    # Schema (DDL — the DESIGN §12 schema, idempotent)
    # ------------------------------------------------------------------

    def ensure_schema(self) -> None:
        conn = self.connect()
        cur = conn.cursor()
        s = self.schema
        cur.execute(f"CREATE SCHEMA IF NOT EXISTS {s}")

        # pgvector optional: vector columns are created only when the type exists.
        try:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
            cur.execute("SELECT 1 FROM pg_type WHERE typname = 'vector'")
            self.has_dense = cur.fetchone() is not None
        except Exception:
            conn.rollback()
            try:
                cur.execute("SELECT 1 FROM pg_type WHERE typname = 'vector'")
                self.has_dense = cur.fetchone() is not None
            except Exception:
                self.has_dense = False

        def _emb_type() -> str:
            return f"vector({self.dim})" if self.has_dense else "REAL[]"

        cur.execute(f"""
            CREATE TABLE IF NOT EXISTS {s}.users (
                id         TEXT PRIMARY KEY,
                created_at TIMESTAMPTZ NOT NULL DEFAULT now()
            )
        """)
        cur.execute(f"""
            CREATE TABLE IF NOT EXISTS {s}.conversations (
                id         TEXT PRIMARY KEY,
                user_id    TEXT REFERENCES {s}.users(id),
                status     TEXT NOT NULL DEFAULT 'active',
                started_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                ended_at   TIMESTAMPTZ,
                meta       JSONB NOT NULL DEFAULT '{{}}'
            )
        """)
        cur.execute(f"""
            CREATE TABLE IF NOT EXISTS {s}.messages (
                id              TEXT PRIMARY KEY,
                conversation_id TEXT NOT NULL REFERENCES {s}.conversations(id) ON DELETE CASCADE,
                role            TEXT NOT NULL,
                content         TEXT NOT NULL,
                created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
                token_count     INT NOT NULL DEFAULT 0,
                flags           JSONB NOT NULL DEFAULT '{{}}'
            )
        """)
        cur.execute(f"""
            CREATE TABLE IF NOT EXISTS {s}.conversation_summaries (
                id              TEXT PRIMARY KEY,
                conversation_id TEXT NOT NULL REFERENCES {s}.conversations(id) ON DELETE CASCADE,
                summary         TEXT NOT NULL,
                unresolved_refs JSONB NOT NULL DEFAULT '[]',
                open_questions  JSONB NOT NULL DEFAULT '[]',
                decisions       JSONB NOT NULL DEFAULT '[]',
                version         INT NOT NULL DEFAULT 1,
                created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
            )
        """)
        cur.execute(f"""
            CREATE TABLE IF NOT EXISTS {s}.research_sessions (
                id                TEXT PRIMARY KEY,
                user_id           TEXT REFERENCES {s}.users(id),
                conversation_id   TEXT NOT NULL DEFAULT '',
                title             TEXT NOT NULL DEFAULT '',
                topic_ids         TEXT[] NOT NULL DEFAULT '{{}}',
                entity_ids        TEXT[] NOT NULL DEFAULT '{{}}',
                status            TEXT NOT NULL DEFAULT 'active',
                merged_into       TEXT,
                created_at        TIMESTAMPTZ NOT NULL DEFAULT now(),
                first_activity_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                last_activity_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
                summary           TEXT NOT NULL DEFAULT '',
                meta              JSONB NOT NULL DEFAULT '{{}}'
            )
        """)
        cur.execute(f"""
            CREATE TABLE IF NOT EXISTS {s}.research_questions (
                id                   TEXT PRIMARY KEY,
                session_id           TEXT NOT NULL REFERENCES {s}.research_sessions(id) ON DELETE CASCADE,
                question             TEXT NOT NULL,
                parent_question_id   TEXT,
                evidence_requirement JSONB NOT NULL DEFAULT '{{}}',
                status               TEXT NOT NULL DEFAULT 'open',
                created_at           TIMESTAMPTZ NOT NULL DEFAULT now(),
                resolved_at          TIMESTAMPTZ
            )
        """)
        cur.execute(f"""
            CREATE TABLE IF NOT EXISTS {s}.evidence_references (
                id                  TEXT PRIMARY KEY,
                source              TEXT NOT NULL DEFAULT 'PMC',
                external_id         TEXT NOT NULL DEFAULT '',
                id_type             TEXT NOT NULL DEFAULT 'PMCID',
                doi                 TEXT NOT NULL DEFAULT '',
                pmid                TEXT NOT NULL DEFAULT '',
                chunk_id            TEXT NOT NULL DEFAULT '',
                title               TEXT NOT NULL DEFAULT '',
                verification_status TEXT NOT NULL DEFAULT 'unverified',
                verified_by         TEXT NOT NULL DEFAULT '',
                verified_at         TIMESTAMPTZ,
                retrieved_at        TIMESTAMPTZ,
                indexed_at          TIMESTAMPTZ,
                publication_date    DATE,
                emb                 {_emb_type()},
                meta                JSONB NOT NULL DEFAULT '{{}}'
            )
        """)
        cur.execute(f"""
            CREATE TABLE IF NOT EXISTS {s}.claims (
                id                 TEXT PRIMARY KEY,
                session_id         TEXT REFERENCES {s}.research_sessions(id) ON DELETE SET NULL,
                user_id            TEXT REFERENCES {s}.users(id),
                text               TEXT NOT NULL,
                normalized_text    TEXT NOT NULL DEFAULT '',
                dedup_key          TEXT NOT NULL DEFAULT '',
                provenance_class   TEXT NOT NULL DEFAULT 'model_inference'
                    CHECK (provenance_class IN
                        ('user_assertion','model_inference','unverified_information',
                         'verified_evidence','evidence_derived_claim')),
                status             TEXT NOT NULL DEFAULT 'unresolved'
                    CHECK (status IN
                        ('supported','contradicted','mixed','superseded',
                         'invalidated','unresolved')),
                confidence         DOUBLE PRECISION NOT NULL DEFAULT 0,
                topic_ids          TEXT[] NOT NULL DEFAULT '{{}}',
                entity_ids         TEXT[] NOT NULL DEFAULT '{{}}',
                first_seen_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
                last_verified_at   TIMESTAMPTZ,
                valid_from         TIMESTAMPTZ NOT NULL DEFAULT now(),
                valid_to           TIMESTAMPTZ,
                superseded_by      TEXT,
                superseded_at      TIMESTAMPTZ,
                invalidated_at     TIMESTAMPTZ,
                needs_revalidation BOOLEAN NOT NULL DEFAULT FALSE,
                emb                {_emb_type()},
                meta               JSONB NOT NULL DEFAULT '{{}}'
            )
        """)
        cur.execute(f"""
            CREATE TABLE IF NOT EXISTS {s}.claim_evidence_links (
                id              TEXT PRIMARY KEY,
                claim_id        TEXT NOT NULL REFERENCES {s}.claims(id) ON DELETE CASCADE,
                evidence_ref_id TEXT NOT NULL REFERENCES {s}.evidence_references(id),
                role            TEXT NOT NULL
                    CHECK (role IN ('supported_by', 'contradicted_by')),
                weight          DOUBLE PRECISION NOT NULL DEFAULT 1.0,
                created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
            )
        """)
        cur.execute(f"""
            CREATE TABLE IF NOT EXISTS {s}.claim_relations (
                id             TEXT PRIMARY KEY,
                from_claim_id  TEXT NOT NULL REFERENCES {s}.claims(id) ON DELETE CASCADE,
                to_claim_id    TEXT NOT NULL REFERENCES {s}.claims(id) ON DELETE CASCADE,
                relation       TEXT NOT NULL
                    CHECK (relation IN ('refined_by','supersedes','derived_from','related_to')),
                created_at     TIMESTAMPTZ NOT NULL DEFAULT now()
            )
        """)
        cur.execute(f"""
            CREATE TABLE IF NOT EXISTS {s}.contradictions (
                id             TEXT PRIMARY KEY,
                session_id     TEXT REFERENCES {s}.research_sessions(id) ON DELETE SET NULL,
                claim          TEXT NOT NULL,
                claim_a_id     TEXT,
                claim_b_id     TEXT,
                evidence_a_ids TEXT[] NOT NULL DEFAULT '{{}}',
                evidence_b_ids TEXT[] NOT NULL DEFAULT '{{}}',
                kind           TEXT NOT NULL DEFAULT 'direct_conflict',
                dimensions     JSONB NOT NULL DEFAULT '{{}}',
                resolution     TEXT NOT NULL DEFAULT 'unresolved'
                    CHECK (resolution IN ('unresolved','explained','spurious','resolved')),
                explanation    TEXT NOT NULL DEFAULT '',
                first_seen_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
                resolved_at    TIMESTAMPTZ
            )
        """)
        cur.execute(f"""
            CREATE TABLE IF NOT EXISTS {s}.research_gaps (
                id         TEXT PRIMARY KEY,
                session_id TEXT NOT NULL REFERENCES {s}.research_sessions(id) ON DELETE CASCADE,
                question   TEXT NOT NULL,
                kind       TEXT NOT NULL DEFAULT 'missing_evidence',
                status     TEXT NOT NULL DEFAULT 'open',
                created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                filled_at  TIMESTAMPTZ,
                meta       JSONB NOT NULL DEFAULT '{{}}'
            )
        """)
        cur.execute(f"""
            CREATE TABLE IF NOT EXISTS {s}.memory_events (
                id          TEXT PRIMARY KEY,
                actor       TEXT NOT NULL,
                event_type  TEXT NOT NULL,
                object_type TEXT NOT NULL,
                object_id   TEXT NOT NULL,
                payload     JSONB NOT NULL DEFAULT '{{}}',
                created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
            )
        """)
        cur.execute(f"""
            CREATE TABLE IF NOT EXISTS {s}.user_preferences (
                id         TEXT PRIMARY KEY,
                user_id    TEXT NOT NULL REFERENCES {s}.users(id) ON DELETE CASCADE,
                key        TEXT NOT NULL,
                value      JSONB NOT NULL,
                source     TEXT NOT NULL DEFAULT 'explicit',
                confidence DOUBLE PRECISION NOT NULL DEFAULT 1.0,
                created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                UNIQUE (user_id, key)
            )
        """)

        # ---- indexes ---------------------------------------------------
        cur.execute(f"CREATE INDEX IF NOT EXISTS idx_msg_conv ON {s}.messages (conversation_id, created_at)")
        cur.execute(f"CREATE INDEX IF NOT EXISTS idx_cs_conv ON {s}.conversation_summaries (conversation_id, version)")
        # conversation_id added later: idempotent migration for existing DBs
        cur.execute(f"ALTER TABLE {s}.research_sessions ADD COLUMN IF NOT EXISTS conversation_id TEXT NOT NULL DEFAULT ''")
        cur.execute(f"CREATE INDEX IF NOT EXISTS idx_rs_conv ON {s}.research_sessions (conversation_id)")
        cur.execute(f"CREATE INDEX IF NOT EXISTS idx_rs_user ON {s}.research_sessions (user_id, last_activity_at DESC)")
        cur.execute(f"CREATE INDEX IF NOT EXISTS idx_rq_session ON {s}.research_questions (session_id)")
        cur.execute(f"""
            CREATE UNIQUE INDEX IF NOT EXISTS idx_er_external
            ON {s}.evidence_references (external_id, id_type, chunk_id)
        """)
        cur.execute(f"CREATE INDEX IF NOT EXISTS idx_claims_session ON {s}.claims (session_id)")
        cur.execute(f"CREATE INDEX IF NOT EXISTS idx_claims_prov ON {s}.claims (provenance_class, valid_to)")
        cur.execute(f"CREATE INDEX IF NOT EXISTS idx_claims_dedup ON {s}.claims (dedup_key)")
        cur.execute(f"CREATE INDEX IF NOT EXISTS idx_cel_claim ON {s}.claim_evidence_links (claim_id)")
        cur.execute(f"CREATE INDEX IF NOT EXISTS idx_cel_evidence ON {s}.claim_evidence_links (evidence_ref_id)")
        cur.execute(f"CREATE INDEX IF NOT EXISTS idx_cr_from ON {s}.claim_relations (from_claim_id)")
        cur.execute(f"CREATE INDEX IF NOT EXISTS idx_cr_to ON {s}.claim_relations (to_claim_id)")
        cur.execute(f"CREATE INDEX IF NOT EXISTS idx_ctr_session ON {s}.contradictions (session_id)")
        cur.execute(f"CREATE INDEX IF NOT EXISTS idx_gap_session ON {s}.research_gaps (session_id, status)")
        cur.execute(f"CREATE INDEX IF NOT EXISTS idx_mev_obj ON {s}.memory_events (object_type, object_id, created_at DESC)")
        cur.execute(f"CREATE INDEX IF NOT EXISTS idx_pref_user ON {s}.user_preferences (user_id)")

        if self.has_dense:
            try:
                cur.execute(f"""
                    CREATE INDEX IF NOT EXISTS idx_claims_emb
                    ON {s}.claims USING hnsw (emb vector_cosine_ops)
                """)
            except Exception:
                self._conn.rollback()
        conn.commit()
        cur.close()

    # ------------------------------------------------------------------
    # Row <-> record mapping helpers
    # ------------------------------------------------------------------

    _CLAIM_COLS = ("id", "session_id", "user_id", "text", "normalized_text",
                   "dedup_key", "provenance_class", "status", "confidence",
                   "topic_ids", "entity_ids", "first_seen_at", "last_verified_at",
                   "valid_from", "valid_to", "superseded_by", "superseded_at",
                   "invalidated_at", "needs_revalidation", "meta")

    def _claim_from_row(self, row: tuple) -> ClaimRecord:
        return ClaimRecord(
            id=row[0], session_id=row[1] or "", user_id=row[2] or "",
            text=row[3], normalized_text=row[4] or "", dedup_key=row[5] or "",
            provenance_class=ProvenanceClass(row[6]),
            status=ClaimStatus(row[7]),
            confidence=float(row[8] or 0.0),
            topic_ids=list(row[9] or []), entity_ids=list(row[10] or []),
            first_seen_at=_as_utc(row[11]),
            last_verified_at=_as_utc(row[12]),
            valid_from=_as_utc(row[13]), valid_to=_as_utc(row[14]),
            superseded_by=row[15] or "", superseded_at=_as_utc(row[16]),
            invalidated_at=_as_utc(row[17]),
            needs_revalidation=bool(row[18]),
            meta=_json_loads(row[19], {}) if not isinstance(row[19], dict) else row[19],
        )

    def _claim_to_params(self, claim: ClaimRecord, emb: list[float] | None = None) -> tuple:
        return (
            claim.id, claim.session_id or None, claim.user_id or None,
            claim.text, claim.normalized_text, claim.dedup_key,
            claim.provenance_class.value, claim.status.value,
            float(claim.confidence),
            list(claim.topic_ids), list(claim.entity_ids),
            claim.first_seen_at, claim.last_verified_at,
            claim.valid_from, claim.valid_to,
            claim.superseded_by or None, claim.superseded_at,
            claim.invalidated_at, claim.needs_revalidation,
            _json_dumps(claim.meta),
            emb or None,
        )

    # ------------------------------------------------------------------
    # Conversations
    # ------------------------------------------------------------------

    def _ensure_user(self, user_id: str) -> None:
        if not user_id:
            return
        self._execute(
            f"INSERT INTO {self.schema}.users (id) VALUES (%s) "
            f"ON CONFLICT (id) DO NOTHING", (user_id,))

    def start_conversation(self, user_id: str = "") -> ConversationRecord:
        self._ensure_user(user_id)
        rec = ConversationRecord(id=new_id("con"), user_id=user_id)
        self._execute(
            f"INSERT INTO {self.schema}.conversations (id, user_id, status, meta) "
            f"VALUES (%s,%s,%s,%s)",
            (rec.id, user_id or None, rec.status.value, _json_dumps(rec.meta)))
        return rec

    def ensure_conversation(self, conversation_id: str,
                            user_id: str = "") -> ConversationRecord:
        """Return the conversation with this id, creating it if needed."""
        self._ensure_user(user_id)
        self._execute(
            f"INSERT INTO {self.schema}.conversations (id, user_id, status, meta) "
            f"VALUES (%s,%s,%s,%s) ON CONFLICT (id) DO NOTHING",
            (conversation_id, user_id or None,
             ConversationStatus.ACTIVE.value, _json_dumps({})))
        return self.get_conversation(conversation_id) or ConversationRecord(
            id=conversation_id, user_id=user_id)

    def get_conversation(self, conversation_id: str) -> ConversationRecord | None:
        row = self._fetchone(
            f"SELECT id, user_id, status, started_at, ended_at, meta "
            f"FROM {self.schema}.conversations WHERE id = %s", (conversation_id,))
        if not row:
            return None
        return ConversationRecord(id=row[0], user_id=row[1] or "",
                                  status=ConversationStatus(row[2]),
                                  started_at=_as_utc(row[3]), ended_at=_as_utc(row[4]),
                                  meta=_json_loads(row[5], {}))

    def add_message(self, conversation_id: str, role: str, content: str,
                    **flags: Any) -> MessageRecord:
        rec = MessageRecord(id=new_id("msg"), conversation_id=conversation_id,
                            role=role, content=content, flags=dict(flags))
        self._execute(
            f"INSERT INTO {self.schema}.messages (id, conversation_id, role, content, flags) "
            f"VALUES (%s,%s,%s,%s,%s)",
            (rec.id, conversation_id, role, content, _json_dumps(rec.flags)))
        return rec

    def get_messages(self, conversation_id: str, limit: int = 20) -> list[MessageRecord]:
        rows = self._fetchall(
            f"SELECT id, conversation_id, role, content, created_at, token_count, flags "
            f"FROM {self.schema}.messages WHERE conversation_id = %s "
            f"ORDER BY created_at DESC LIMIT %s", (conversation_id, limit))
        return [MessageRecord(id=r[0], conversation_id=r[1], role=r[2], content=r[3],
                              created_at=_as_utc(r[4]), token_count=int(r[5] or 0),
                              flags=_json_loads(r[6], {})) for r in rows][::-1]

    def set_conversation_summary(self, conversation_id: str, summary: str,
                                 open_questions: list[str] | None = None,
                                 unresolved_refs: list[dict] | None = None,
                                 decisions: list[str] | None = None) -> ConversationSummaryRecord:
        prev = self._fetchone(
            f"SELECT version FROM {self.schema}.conversation_summaries "
            f"WHERE conversation_id = %s ORDER BY version DESC LIMIT 1",
            (conversation_id,))
        version = int(prev[0]) + 1 if prev else 1
        rec = ConversationSummaryRecord(id=new_id("cs"), conversation_id=conversation_id,
                                        summary=summary, open_questions=open_questions or [],
                                        unresolved_refs=unresolved_refs or [],
                                        decisions=decisions or [], version=version)
        self._execute(
            f"INSERT INTO {self.schema}.conversation_summaries "
            f"(id, conversation_id, summary, unresolved_refs, open_questions, decisions, version) "
            f"VALUES (%s,%s,%s,%s,%s,%s,%s)",
            (rec.id, conversation_id, summary, _json_dumps(rec.unresolved_refs),
             _json_dumps(rec.open_questions), _json_dumps(rec.decisions), version))
        return rec

    def get_conversation_summary(self, conversation_id: str) -> ConversationSummaryRecord | None:
        row = self._fetchone(
            f"SELECT id, conversation_id, summary, unresolved_refs, open_questions, "
            f"decisions, version, created_at FROM {self.schema}.conversation_summaries "
            f"WHERE conversation_id = %s ORDER BY version DESC LIMIT 1", (conversation_id,))
        if not row:
            return None
        return ConversationSummaryRecord(id=row[0], conversation_id=row[1], summary=row[2],
                                         unresolved_refs=_json_loads(row[3], []),
                                         open_questions=_json_loads(row[4], []),
                                         decisions=_json_loads(row[5], []),
                                         version=int(row[6]), created_at=_as_utc(row[7]))

    # ------------------------------------------------------------------
    # Sessions
    # ------------------------------------------------------------------

    def create_session(self, seed_question: str, user_id: str = "",
                       conversation_id: str = "", title: str = "") -> ResearchSessionRecord:
        self._ensure_user(user_id)
        rec = ResearchSessionRecord(id=new_id("rs"), user_id=user_id,
                                    conversation_id=conversation_id,
                                    title=title or (seed_question or "")[:90],
                                    summary=seed_question or "")
        self._execute(
            f"INSERT INTO {self.schema}.research_sessions "
            f"(id, user_id, conversation_id, title, topic_ids, entity_ids, status, summary) "
            f"VALUES (%s,%s,%s,%s,%s,%s,%s,%s)",
            (rec.id, user_id or None, conversation_id or "", rec.title,
             list(rec.topic_ids),
             list(rec.entity_ids), rec.status.value, rec.summary))
        return rec

    def get_session(self, session_id: str) -> ResearchSessionRecord | None:
        row = self._fetchone(
            f"SELECT id, user_id, conversation_id, title, topic_ids, entity_ids, status, merged_into, "
            f"created_at, first_activity_at, last_activity_at, summary, meta "
            f"FROM {self.schema}.research_sessions WHERE id = %s", (session_id,))
        if not row:
            return None
        return ResearchSessionRecord(id=row[0], user_id=row[1] or "",
                                     conversation_id=row[2] or "",
                                     title=row[3],
                                     topic_ids=list(row[4] or []), entity_ids=list(row[5] or []),
                                     status=SessionStatus(row[6]), merged_into=row[7] or "",
                                     created_at=_as_utc(row[8]),
                                     first_activity_at=_as_utc(row[9]),
                                     last_activity_at=_as_utc(row[10]),
                                     summary=row[10] or "",
                                     meta=_json_loads(row[11], {}))

    def touch_session(self, session_id: str) -> None:
        self._execute(
            f"UPDATE {self.schema}.research_sessions SET last_activity_at = now() "
            f"WHERE id = %s", (session_id,))

    def update_session_summary(self, session_id: str, summary: str) -> None:
        self._execute(
            f"UPDATE {self.schema}.research_sessions "
            f"SET summary = %s, last_activity_at = now() WHERE id = %s",
            (summary[:600], session_id))

    def close_session(self, session_id: str) -> None:
        self._execute(
            f"UPDATE {self.schema}.research_sessions SET status = 'closed' "
            f"WHERE id = %s", (session_id,))

    def archive_session(self, session_id: str) -> None:
        self._execute(
            f"UPDATE {self.schema}.research_sessions SET status = 'archived' "
            f"WHERE id = %s", (session_id,))

    def merge_sessions(self, from_id: str, into_id: str) -> None:
        with self.transaction():
            self._execute(
                f"UPDATE {self.schema}.research_sessions SET status='merged', "
                f"merged_into=%s WHERE id = %s", (into_id, from_id))
            self._execute(
                f"UPDATE {self.schema}.research_questions SET session_id = %s "
                f"WHERE session_id = %s", (into_id, from_id))
            self._execute(
                f"UPDATE {self.schema}.claims SET session_id = %s WHERE session_id = %s",
                (into_id, from_id))
            self.touch_session(into_id)

    def list_sessions(self, user_id: str = "",
                      status: SessionStatus | None = None,
                      conversation_id: str | None = None) -> list[ResearchSessionRecord]:
        sql = (f"SELECT id, user_id, conversation_id, title, topic_ids, entity_ids, status, merged_into, "
               f"created_at, first_activity_at, last_activity_at, summary, meta "
               f"FROM {self.schema}.research_sessions")
        where, params = [], []
        if user_id:
            where.append("user_id = %s"); params.append(user_id)
        if status:
            where.append("status = %s"); params.append(status.value)
        if conversation_id:
            where.append("conversation_id = %s"); params.append(conversation_id)
        if where:
            sql += " WHERE " + " AND ".join(where)
        sql += " ORDER BY last_activity_at DESC"
        rows = self._fetchall(sql, tuple(params))
        return [self._session_from_row(r) for r in rows]

    def get_session_for_conversation(self, conversation_id: str,
                                     status: SessionStatus | None = None) -> ResearchSessionRecord | None:
        if not conversation_id:
            return None
        recs = self.list_sessions(status=status, conversation_id=conversation_id)
        return recs[0] if recs else None

    @staticmethod
    def _session_from_row(row: tuple) -> ResearchSessionRecord:
        return ResearchSessionRecord(
            id=row[0], user_id=row[1] or "", conversation_id=row[2] or "",
            title=row[3], topic_ids=list(row[4] or []),
            entity_ids=list(row[5] or []),
            status=SessionStatus(row[6]), merged_into=row[7] or "",
            created_at=_as_utc(row[8]), first_activity_at=_as_utc(row[9]),
            last_activity_at=_as_utc(row[10]),
            summary=row[11] or "",
            meta=_json_loads(row[12], {}),
        )

    def find_sessions(self, query: str, user_id: str = "",
                      limit: int = 5,
                      conversation_id: str | None = None) -> list[tuple[ResearchSessionRecord, float]]:
        """Session identification: score ACTIVE sessions by lexical overlap
        with their title/summary/questions.

        conversation_id scope: when set, ONLY sessions bound to that chat are
        considered - a session from another conversation is never resumed even
        when the query is nearly identical."""
        q = _token_overlap(query)
        if not q:
            return []
        scored: list[tuple[ResearchSessionRecord, float]] = []
        for rec in self.list_sessions(user_id=user_id, status=SessionStatus.ACTIVE,
                                      conversation_id=conversation_id):
            texts = [rec.title, rec.summary]
            texts += [qq[0] for qq in self._fetchall(
                f"SELECT question FROM {self.schema}.research_questions "
                f"WHERE session_id = %s", (rec.id,))]
            best = 0.0
            for t in texts:
                if not t:
                    continue
                overlap = _token_overlap(t) & q
                score = len(overlap) / max(1.0, len(q))
                best = max(best, score)
            if best > 0:
                scored.append((rec, float(best)))
        scored.sort(key=lambda t: t[1], reverse=True)
        return scored[:limit]

    # ------------------------------------------------------------------
    # Questions
    # ------------------------------------------------------------------

    def add_question(self, session_id: str, question: str,
                     requirement: dict | None = None) -> ResearchQuestionRecord:
        rec = ResearchQuestionRecord(id=new_id("rq"), session_id=session_id,
                                     question=question,
                                     evidence_requirement=requirement or {})
        self._execute(
            f"INSERT INTO {self.schema}.research_questions "
            f"(id, session_id, question, evidence_requirement) VALUES (%s,%s,%s,%s)",
            (rec.id, session_id, question, _json_dumps(rec.evidence_requirement)))
        return rec

    def get_questions(self, session_id: str) -> list[ResearchQuestionRecord]:
        rows = self._fetchall(
            f"SELECT id, session_id, question, parent_question_id, evidence_requirement, "
            f"status, created_at, resolved_at FROM {self.schema}.research_questions "
            f"WHERE session_id = %s", (session_id,))
        return [ResearchQuestionRecord(id=r[0], session_id=r[1], question=r[2],
                                       parent_question_id=r[3] or "",
                                       evidence_requirement=_json_loads(r[4], {}),
                                       status=r[5], created_at=_as_utc(r[6]),
                                       resolved_at=_as_utc(r[7])) for r in rows]

    def mark_question_answered(self, question_id: str) -> None:
        self._execute(
            f"UPDATE {self.schema}.research_questions SET status='answered', "
            f"resolved_at=now() WHERE id = %s", (question_id,))

    # ------------------------------------------------------------------
    # Evidence references
    # ------------------------------------------------------------------

    def add_evidence_ref(self, ref: EvidenceReferenceRecord) -> EvidenceReferenceRecord:
        key = (ref.external_id or "", ref.id_type or "", ref.chunk_id or "")
        existing = self.evidence_ref_by_external(key[0], key[1]) if key[0] else None
        if existing and existing.chunk_id == (ref.chunk_id or ""):
            if ref.verified and not existing.verified:
                self._execute(
                    f"UPDATE {self.schema}.evidence_references SET "
                    f"verification_status=%s, verified_by=%s, verified_at=%s, title=%s, "
                    f"doi=%s, pmid=%s, publication_date=%s, meta=%s WHERE id=%s",
                    (ref.verification_status, ref.verified_by, _as_utc(ref.verified_at),
                     ref.title, ref.doi, ref.pmid, ref.publication_date,
                     _json_dumps(ref.meta), existing.id))
                return self.get_evidence_ref(existing.id) or existing
            return existing
        if not ref.id:
            ref.id = new_id("ev")
        self._execute(
            f"INSERT INTO {self.schema}.evidence_references "
            f"(id, source, external_id, id_type, doi, pmid, chunk_id, title, "
            f"verification_status, verified_by, verified_at, retrieved_at, "
            f"indexed_at, publication_date, meta) "
            f"VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)",
            (ref.id, ref.source, ref.external_id, ref.id_type, ref.doi, ref.pmid,
             ref.chunk_id, ref.title, ref.verification_status, ref.verified_by,
             _as_utc(ref.verified_at), _as_utc(ref.retrieved_at),
             _as_utc(ref.indexed_at), ref.publication_date, _json_dumps(ref.meta)))
        return ref

    def get_evidence_ref(self, ref_id: str) -> EvidenceReferenceRecord | None:
        row = self._fetchone(
            f"SELECT id, source, external_id, id_type, doi, pmid, chunk_id, title, "
            f"verification_status, verified_by, verified_at, retrieved_at, indexed_at, "
            f"publication_date, meta FROM {self.schema}.evidence_references WHERE id = %s",
            (ref_id,))
        if not row:
            return None
        return EvidenceReferenceRecord(
            id=row[0], source=row[1], external_id=row[2] or "", id_type=row[3] or "",
            doi=row[4] or "", pmid=row[5] or "", chunk_id=row[6] or "",
            title=row[7] or "", verification_status=row[8] or "unverified",
            verified_by=row[9] or "", verified_at=_as_utc(row[10]),
            retrieved_at=_as_utc(row[11]), indexed_at=_as_utc(row[12]),
            publication_date=_as_utc(row[13]), meta=_json_loads(row[14], {}))

    def get_evidence_refs(self, ref_ids: Sequence[str]) -> list[EvidenceReferenceRecord]:
        if not ref_ids:
            return []
        rows = self._fetchall(
            f"SELECT id, source, external_id, id_type, doi, pmid, chunk_id, title, "
            f"verification_status, verified_by, verified_at, retrieved_at, indexed_at, "
            f"publication_date, meta FROM {self.schema}.evidence_references "
            f"WHERE id = ANY(%s)", (list(ref_ids),))
        return [EvidenceReferenceRecord(
            id=r[0], source=r[1], external_id=r[2] or "", id_type=r[3] or "",
            doi=r[4] or "", pmid=r[5] or "", chunk_id=r[6] or "",
            title=r[7] or "", verification_status=r[8] or "unverified",
            verified_by=r[9] or "", verified_at=_as_utc(r[10]),
            retrieved_at=_as_utc(r[11]), indexed_at=_as_utc(r[12]),
            publication_date=_as_utc(r[13]), meta=_json_loads(r[14], {})) for r in rows]

    def evidence_ref_by_external(self, external_id: str,
                                 id_type: str = "PMCID") -> EvidenceReferenceRecord | None:
        row = self._fetchone(
            f"SELECT id, source, external_id, id_type, doi, pmid, chunk_id, title, "
            f"verification_status, verified_by, verified_at, retrieved_at, indexed_at, "
            f"publication_date, meta FROM {self.schema}.evidence_references "
            f"WHERE external_id = %s AND id_type = %s "
            f"ORDER BY verified_at DESC NULLS LAST LIMIT 1",
            (external_id, id_type))
        if not row:
            return None
        return self.get_evidence_ref(row[0])

    # ------------------------------------------------------------------
    # Claims
    # ------------------------------------------------------------------

    def add_claim(self, claim: ClaimRecord) -> ClaimRecord:
        if not claim.id:
            claim.id = new_id("clm")
        if not claim.normalized_text:
            from src.memory.validation import normalized_claim_text
            claim.normalized_text = normalized_claim_text(claim.text)
        if not claim.dedup_key:
            from src.memory.validation import claim_dedup_key
            claim.dedup_key = claim_dedup_key(claim)
        emb: list[float] | None = claim.meta.pop("_emb", None)
        params = self._claim_to_params(claim, emb)
        if self.has_dense and emb:
            try:
                from pgvector.psycopg2 import register_vector
                register_vector(self.connect())
            except Exception:
                pass
        self._execute(
            f"INSERT INTO {self.schema}.claims (id, session_id, user_id, text, "
            f"normalized_text, dedup_key, provenance_class, status, confidence, "
            f"topic_ids, entity_ids, first_seen_at, last_verified_at, valid_from, "
            f"valid_to, superseded_by, superseded_at, invalidated_at, "
            f"needs_revalidation, meta, emb) "
            f"VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)",
            params)
        return claim

    def get_claim(self, claim_id: str) -> ClaimRecord | None:
        row = self._fetchone(
            f"SELECT {', '.join(self._CLAIM_COLS)} FROM {self.schema}.claims "
            f"WHERE id = %s", (claim_id,))
        if not row:
            return None
        return self._claim_from_row(row)

    def update_claim(self, claim_id: str, **patch: Any) -> ClaimRecord | None:
        if not patch:
            return self.get_claim(claim_id)
        allowed = set(self._CLAIM_COLS) - {"id"}
        sets, params = [], []
        for k, v in patch.items():
            if k not in allowed:
                continue
            sets.append(f"{k} = %s")
            if isinstance(v, (dict, list)):
                v = _json_dumps(v)
            params.append(v)
        params.append(claim_id)
        if sets:
            self._execute(
                f"UPDATE {self.schema}.claims SET {', '.join(sets)} WHERE id = %s",
                tuple(params))
        return self.get_claim(claim_id)

    def get_claims(self, session_id: str | None = None,
                   provenance: ProvenanceClass | Sequence[ProvenanceClass] | None = None,
                   status: ClaimStatus | None = None,
                   include_superseded: bool = False) -> list[ClaimRecord]:
        where, params = [], []
        if session_id:
            where.append("session_id = %s"); params.append(session_id)
        if provenance is not None:
            wanted = provenance if isinstance(provenance, (list, tuple, set)) else (provenance,)
            where.append("provenance_class = ANY(%s)")
            params.append([p.value for p in wanted])
        if status:
            where.append("status = %s"); params.append(status.value)
        if not include_superseded:
            where.append("(superseded_by IS NULL OR superseded_by = '')")
        sql = (f"SELECT {', '.join(self._CLAIM_COLS)} FROM {self.schema}.claims")
        if where:
            sql += " WHERE " + " AND ".join(where)
        sql += " ORDER BY first_seen_at DESC"
        return [self._claim_from_row(r) for r in self._fetchall(sql, tuple(params))]

    def scan_claims(self, session_ids: Sequence[str] | None = None,
                    provenance: ProvenanceClass | Sequence[ProvenanceClass] | None = None,
                    include_stale: bool = False,
                    include_superseded: bool = False) -> list[ClaimRecord]:
        where, params = [], []
        if session_ids:
            where.append("session_id = ANY(%s)"); params.append(list(session_ids))
        if provenance is not None:
            wanted = provenance if isinstance(provenance, (list, tuple, set)) else (provenance,)
            where.append("provenance_class = ANY(%s)")
            params.append([p.value for p in wanted])
        if not include_stale:
            where.append("needs_revalidation = FALSE")
        if not include_superseded:
            where.append("(superseded_by IS NULL OR superseded_by = '')")
        sql = (f"SELECT {', '.join(self._CLAIM_COLS)} FROM {self.schema}.claims")
        if where:
            sql += " WHERE " + " AND ".join(where)
        return [self._claim_from_row(r)
                for r in self._fetchall(sql, tuple(params))]

    def claims_by_dedup_key(self, key: str) -> list[ClaimRecord]:
        rows = self._fetchall(
            f"SELECT {', '.join(self._CLAIM_COLS)} FROM {self.schema}.claims "
            f"WHERE dedup_key = %s", (key,))
        return [self._claim_from_row(r) for r in rows]

    # ------------------------------------------------------------------
    # Edges
    # ------------------------------------------------------------------

    def add_claim_evidence_link(self, link: ClaimEvidenceLinkRecord) -> ClaimEvidenceLinkRecord:
        if not link.id:
            link.id = new_id("cel")
        self._execute(
            f"INSERT INTO {self.schema}.claim_evidence_links "
            f"(id, claim_id, evidence_ref_id, role, weight) VALUES (%s,%s,%s,%s,%s)",
            (link.id, link.claim_id, link.evidence_ref_id, link.role.value,
             float(link.weight)))
        return link

    def get_claim_evidence_links(self, claim_id: str) -> list[ClaimEvidenceLinkRecord]:
        return self.get_links_for_claim(claim_id)

    def get_links_for_claim(self, claim_id: str) -> list[ClaimEvidenceLinkRecord]:
        rows = self._fetchall(
            f"SELECT id, claim_id, evidence_ref_id, role, weight, created_at "
            f"FROM {self.schema}.claim_evidence_links WHERE claim_id = %s",
            (claim_id,))
        return [ClaimEvidenceLinkRecord(id=r[0], claim_id=r[1], evidence_ref_id=r[2],
                                        role=EvidenceRole(r[3]), weight=float(r[4]),
                                        created_at=_as_utc(r[5])) for r in rows]

    def add_claim_relation(self, rel: ClaimRelationRecord) -> ClaimRelationRecord:
        if not rel.id:
            rel.id = new_id("cr")
        self._execute(
            f"INSERT INTO {self.schema}.claim_relations "
            f"(id, from_claim_id, to_claim_id, relation) VALUES (%s,%s,%s,%s)",
            (rel.id, rel.from_claim_id, rel.to_claim_id, rel.relation.value))
        return rel

    def get_claim_relations(self, claim_id: str,
                            relation: ClaimRelationKind | None = None) -> list[ClaimRelationRecord]:
        sql = (f"SELECT id, from_claim_id, to_claim_id, relation, created_at "
               f"FROM {self.schema}.claim_relations WHERE from_claim_id = %s")
        params: tuple = (claim_id,)
        if relation:
            sql += " AND relation = %s"; params = (claim_id, relation.value)
        rows = self._fetchall(sql, params)
        return [ClaimRelationRecord(id=r[0], from_claim_id=r[1], to_claim_id=r[2],
                                    relation=ClaimRelationKind(r[3]),
                                    created_at=_as_utc(r[4])) for r in rows]

    # ------------------------------------------------------------------
    # Contradictions
    # ------------------------------------------------------------------

    def add_contradiction(self, rec: ContradictionRecord) -> ContradictionRecord:
        if not rec.id:
            rec.id = new_id("ctr")
        self._execute(
            f"INSERT INTO {self.schema}.contradictions "
            f"(id, session_id, claim, claim_a_id, claim_b_id, evidence_a_ids, "
            f"evidence_b_ids, kind, dimensions, resolution, explanation) "
            f"VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)",
            (rec.id, rec.session_id or None, rec.claim, rec.claim_a_id or None,
             rec.claim_b_id or None, list(rec.evidence_a_ids), list(rec.evidence_b_ids),
             rec.kind.value, _json_dumps(rec.dimensions), rec.resolution.value,
             rec.explanation))
        return rec

    def get_contradictions(self, session_id: str | None = None,
                           unresolved_only: bool = False) -> list[ContradictionRecord]:
        where, params = [], []
        if session_id:
            where.append("session_id = %s"); params.append(session_id)
        if unresolved_only:
            where.append("resolution = 'unresolved'")
        sql = f"SELECT * FROM {self.schema}.contradictions"
        if where:
            sql += " WHERE " + " AND ".join(where)
        cols = ("id", "session_id", "claim", "claim_a_id", "claim_b_id",
                "evidence_a_ids", "evidence_b_ids", "kind", "dimensions",
                "resolution", "explanation", "first_seen_at", "resolved_at")
        rows = self._fetchall(sql, tuple(params))
        out = []
        for r in rows:
            d = dict(zip(cols, r))
            out.append(ContradictionRecord(
                id=d["id"], session_id=d["session_id"] or "",
                claim=d["claim"], claim_a_id=d["claim_a_id"] or "",
                claim_b_id=d["claim_b_id"] or "",
                evidence_a_ids=list(d["evidence_a_ids"] or []),
                evidence_b_ids=list(d["evidence_b_ids"] or []),
                kind=ContradictionKind(d["kind"]),
                dimensions=_json_loads(d["dimensions"], {}),
                resolution=ContradictionResolution(d["resolution"]),
                explanation=d["explanation"] or "",
                first_seen_at=_as_utc(d["first_seen_at"]),
                resolved_at=_as_utc(d["resolved_at"])))
        return out

    def resolve_contradiction(self, contradiction_id: str,
                              resolution: ContradictionResolution,
                              explanation: str = "") -> None:
        self._execute(
            f"UPDATE {self.schema}.contradictions SET resolution=%s, explanation=%s, "
            f"resolved_at = CASE WHEN %s = 'unresolved' THEN NULL ELSE now() END "
            f"WHERE id = %s",
            (resolution.value, explanation, resolution.value, contradiction_id))

    # ------------------------------------------------------------------
    # Gaps
    # ------------------------------------------------------------------

    def add_gap(self, rec: ResearchGapRecord) -> ResearchGapRecord:
        if not rec.id:
            rec.id = new_id("gap")
        self._execute(
            f"INSERT INTO {self.schema}.research_gaps (id, session_id, question, kind, status, meta) "
            f"VALUES (%s,%s,%s,%s,%s,%s)",
            (rec.id, rec.session_id, rec.question, rec.kind.value, rec.status,
             _json_dumps(rec.meta)))
        return rec

    def get_gaps(self, session_id: str | None = None,
                 open_only: bool = False) -> list[ResearchGapRecord]:
        where, params = [], []
        if session_id:
            where.append("session_id = %s"); params.append(session_id)
        if open_only:
            where.append("status = 'open'")
        sql = f"SELECT * FROM {self.schema}.research_gaps"
        if where:
            sql += " WHERE " + " AND ".join(where)
        rows = self._fetchall(sql, tuple(params))
        cols = ("id", "session_id", "question", "kind", "status", "created_at",
                "filled_at", "meta")
        return [ResearchGapRecord(
            id=d["id"], session_id=d["session_id"], question=d["question"],
            kind=GapKind(d["kind"]), status=d["status"],
            created_at=_as_utc(d["created_at"]), filled_at=_as_utc(d["filled_at"]),
            meta=_json_loads(d["meta"], {}))
            for r in rows for d in [dict(zip(cols, r))]]

    def fill_gap(self, gap_id: str) -> None:
        self._execute(
            f"UPDATE {self.schema}.research_gaps SET status='filled', filled_at=now() "
            f"WHERE id = %s AND status = 'open'", (gap_id,))

    # ------------------------------------------------------------------
    # Preferences
    # ------------------------------------------------------------------

    def set_preference(self, user_id: str, key: str, value: Any,
                       source: str = "explicit", confidence: float = 1.0) -> UserPreferenceRecord:
        self._ensure_user(user_id)
        self._execute(
            f"INSERT INTO {self.schema}.user_preferences "
            f"(id, user_id, key, value, source, confidence) VALUES (%s,%s,%s,%s,%s,%s) "
            f"ON CONFLICT (user_id, key) DO UPDATE SET value=EXCLUDED.value, "
            f"source=EXCLUDED.source, confidence=EXCLUDED.confidence, updated_at=now()",
            (new_id("pref"), user_id, key, _json_dumps(value), source, float(confidence)))
        rows = self._fetchall(
            f"SELECT id, user_id, key, value, source, confidence, created_at, updated_at "
            f"FROM {self.schema}.user_preferences WHERE user_id = %s AND key = %s",
            (user_id, key))
        r = rows[-1]
        return UserPreferenceRecord(id=r[0], user_id=r[1], key=r[2],
                                    value=_json_loads(r[3], None), source=r[4],
                                    confidence=float(r[5]),
                                    created_at=_as_utc(r[6]), updated_at=_as_utc(r[7]))

    def get_preferences(self, user_id: str) -> list[UserPreferenceRecord]:
        rows = self._fetchall(
            f"SELECT id, user_id, key, value, source, confidence, created_at, updated_at "
            f"FROM {self.schema}.user_preferences WHERE user_id = %s", (user_id,))
        return [UserPreferenceRecord(id=r[0], user_id=r[1], key=r[2],
                                     value=_json_loads(r[3], None), source=r[4],
                                     confidence=float(r[5]),
                                     created_at=_as_utc(r[6]), updated_at=_as_utc(r[7]))
                for r in rows]

    # ------------------------------------------------------------------
    # Audit
    # ------------------------------------------------------------------

    def add_event(self, actor: str, event_type: MemoryEventType,
                  object_type: MemoryObjectType, object_id: str,
                  payload: dict | None = None) -> MemoryEventRecord:
        rec = MemoryEventRecord(id=new_id("evt"), actor=actor,
                                event_type=event_type.value,
                                object_type=object_type.value,
                                object_id=object_id, payload=payload or {})
        self._execute(
            f"INSERT INTO {self.schema}.memory_events "
            f"(id, actor, event_type, object_type, object_id, payload) "
            f"VALUES (%s,%s,%s,%s,%s,%s)",
            (rec.id, actor, event_type.value, object_type.value, object_id,
             _json_dumps(rec.payload)))
        return rec

    def recent_events(self, limit: int = 50) -> list[MemoryEventRecord]:
        rows = self._fetchall(
            f"SELECT id, actor, event_type, object_type, object_id, payload, created_at "
            f"FROM {self.schema}.memory_events ORDER BY created_at DESC LIMIT %s",
            (limit,))
        return [MemoryEventRecord(id=r[0], actor=r[1], event_type=r[2],
                                  object_type=r[3], object_id=r[4],
                                  payload=_json_loads(r[5], {}), created_at=_as_utc(r[6]))
                for r in rows]

    # ------------------------------------------------------------------
    # Counts
    # ------------------------------------------------------------------

    def counts(self) -> dict[str, int]:
        out = {}
        for table, label in (
            ("conversations", "conversations"), ("messages", "messages"),
            ("research_sessions", "sessions"), ("research_questions", "questions"),
            ("evidence_references", "evidence_refs"), ("claims", "claims"),
            ("claim_evidence_links", "claim_links"),
            ("claim_relations", "claim_relations"),
            ("contradictions", "contradictions"), ("research_gaps", "gaps"),
            ("memory_events", "events"), ("user_preferences", "preferences"),
        ):
            try:
                row = self._fetchone(f"SELECT count(*) FROM {self.schema}.{table}")
                out[label] = int(row[0])
            except Exception:
                out[label] = 0
        return out


def build_store(backend: str = "auto", dsn: str | None = None,
                dim: int = 256) -> MemoryStore:
    """Build the store named by config.

    * ``auto``     — Postgres when reachable, else in-memory.
    * ``postgres`` — Postgres; raises if unreachable.
    * ``memory``   — in-memory (tests / local).
    """
    name = (backend or "auto").strip().lower()
    if name == "memory":
        return InMemoryMemoryStore()
    if name in ("auto", "postgres"):
        try:
            store = PostgresMemoryStore(dsn=dsn, dim=dim)
            store.connect()
            store.ensure_schema()
            if name == "postgres":
                return store
            # auto: keep the PG store only if the schema is actually usable
            store.counts()
            return store
        except Exception as exc:
            if name == "postgres":
                raise
            logger.info("memory backend auto -> in-memory (postgres unavailable: %s)", exc)
            return InMemoryMemoryStore()
    raise ValueError(f"unknown memory backend: {backend!r} (auto | postgres | memory)")


__all__ = [
    "InMemoryMemoryStore",
    "MemoryStore",
    "PostgresMemoryStore",
    "build_store",
]