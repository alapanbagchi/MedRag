"""Memory + Context layer — the public API facade.

This is the ONLY surface the rest of MedPat touches. Orchestration, agents,
and future consumers interact with memory through these typed methods; they
never touch the store or SQL directly.

Read path (context construction):
    prepare_run / compose_context — session resume + typed, budgeted context
    get_conversation_context / get_working_research_state /
    get_relevant_memories / get_evidence_context / get_user_preferences

Session lifecycle:
    create / resume / close / archive / merge / identify

Targeted reads:
    get_relevant_claims / get_unresolved_contradictions /
    get_research_gaps / get_evidence_lineage

Write path (candidate -> validated -> committed, audited):
    propose_memory / validate_memory / commit_memory / process_candidate
    update_claim / mark_stale / record_user_text / record_run

Background:
    consolidate  (dedup / merge / relation / contradiction / temporal)

Evidence boundary: ``record_run`` only creates EVIDENCE_DERIVED_CLAIMs from
CRITIC-verified evidence items; ``prepare_run`` never injects stored memory
into the evidence block of a composed context.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from src.memory import context as ctx
from src.memory import retrieval as mem_retrieval
from src.memory.config import MemoryConfig
from src.memory.consolidate import Consolidator
from src.memory.context import (
    AssembledContext,
    ConversationContext,
    EvidenceContext,
    PersistentMemoryContext,
    ResearchContext,
    UserPreferenceContext,
)
from src.memory.enums import (
    EvidenceRole,
    MemoryEventType,
    MemoryObjectType,
    ProvenanceClass,
    SessionStatus,
)
from src.memory.models import (
    CandidateMemory,
    ClaimEvidenceLinkRecord,
    ClaimRecord,
    EvidenceLineage,
    EvidenceReferenceRecord,
    ResearchSessionRecord,
    ValidationResult,
)
from src.memory.pipeline import (
    MemoryWritePipeline,
    extract_run_candidates,
)
from src.memory.store import MemoryStore, build_store
from src.memory.validation import classify_text

logger = logging.getLogger("src.memory.api")

_USER_ASSERTION_MAX_CHARS = 600


@dataclass
class RunPrep:
    """Everything the pipeline needs before a run: the resumed session and
    the assembled memory/context region."""
    session_id: str = ""
    session: ResearchSessionRecord | None = None
    context: AssembledContext | None = None
    rendered_context: str = ""

    def context_text(self) -> str:
        """The bounded, serialized context region ('' when memory empty)."""
        if self.context is None:
            return ""
        if not self.rendered_context:
            self.rendered_context = self.context.render()
        return self.rendered_context

    def as_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "session_title": (self.session.title if self.session else "") or "",
            "context": self.context_text(),
        }


@dataclass
class RunRecordStats:
    session_id: str = ""
    conversation_id: str = ""
    evidence_refs: int = 0
    claims_committed: int = 0
    claims_deduped: int = 0
    contradictions: int = 0
    gaps: int = 0
    questions: int = 0
    conclusion: str = ""
    events: int = 0

    def as_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


class MemoryAPI:
    """Facade over store + write pipeline + retrieval + context assembly."""

    def __init__(
        self,
        store: MemoryStore,
        embedder: Any | None = None,
        config: MemoryConfig | None = None,
        *,
        write_pipeline: MemoryWritePipeline | None = None,
        consolidator: Consolidator | None = None,
    ):
        self.store = store
        self.embedder = embedder
        self.config = config or MemoryConfig()
        self.pipeline = write_pipeline or MemoryWritePipeline(
            store, embedder, min_sim=self.config.claim_min_sim)
        self.consolidator = consolidator or Consolidator(
            store, embedder, min_sim=self.config.claim_min_sim)

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def build(cls, config: MemoryConfig | None = None) -> "MemoryAPI":
        config = config or MemoryConfig()
        store = build_store(config.backend, dim=config.embed_dim)
        from src.memory.embed import embedder_from_config
        embedder = embedder_from_config(config.embedder, dim=config.embed_dim)
        api = cls(store=store, embedder=embedder, config=config)
        logger.info("MemoryAPI built: backend=%s embedder=%s dim=%d",
                    config.backend, config.embedder, config.embed_dim)
        return api

    # ------------------------------------------------------------------
    # Read path — context construction
    # ------------------------------------------------------------------

    def get_conversation_context(
        self, conversation_id: str,
        budget_tokens: int | None = None,
    ) -> ConversationContext:
        conversation = self.store.get_conversation(conversation_id)
        summary = self.store.get_conversation_summary(conversation_id)
        messages = self.store.get_messages(conversation_id, limit=8)
        return ConversationContext(
            conversation_id=conversation_id,
            recent_messages=messages,
            summary=summary,
            open_questions=list(summary.open_questions) if summary else [],
            unresolved_refs=[r.get("ref", "") for r in
                             (summary.unresolved_refs if summary else [])],
            budget_tokens=budget_tokens or min(400, self.config.context_tokens // 4),
        )

    def get_working_research_state(self, session_id: str) -> ResearchContext:
        session = self.store.get_session(session_id)
        questions = self.store.get_questions(session_id) if session_id else []
        gaps = self.store.get_gaps(session_id, open_only=True)
        return ResearchContext(
            session=session,
            questions=questions,
            gaps=gaps,
            budget_tokens=min(500, self.config.context_tokens // 3),
        )

    def get_relevant_memories(
        self, query: str, session_id: str | None = None,
        budget_tokens: int | None = None,
        include_stale: bool = False,
        top_k: int = 8,
    ) -> PersistentMemoryContext:
        """Hybrid recall of prior claims + unresolved contradictions + open
        gaps, with provenance citations attached for rendering."""
        session_ids = [session_id] if session_id else None
        scored = mem_retrieval.retrieve_claims(
            self.store, query, self.embedder,
            session_ids=session_ids,
            include_stale=include_stale,
            top_k=top_k,
            min_sim=self.config.retrieve_min_sim,
        )
        claims = [sm.claim for sm in scored if sm.claim]
        refs = mem_retrieval.evidence_refs_for_claims(self.store, claims)
        contradictions = mem_retrieval.unresolved_contradictions(
            self.store, session_ids)
        gaps = mem_retrieval.open_gaps(self.store, session_ids, limit=5)
        return PersistentMemoryContext(
            claims=claims[: top_k],
            claim_refs=refs,
            contradictions=contradictions[:5],
            gaps=gaps,
            budget_tokens=budget_tokens or min(700, self.config.context_tokens // 2),
        )

    def get_evidence_context(
        self, evidence_ref_ids: list[str],
        budget_tokens: int | None = None,
    ) -> EvidenceContext:
        """Verified evidence for the CURRENT task (from the pipeline, not
        from memory). Empty refs => empty block — the evidence boundary stays
        explicit."""
        refs = self.store.get_evidence_refs(evidence_ref_ids)
        return EvidenceContext(
            evidence_refs=refs,
            budget_tokens=budget_tokens or 800,
        )

    def get_user_preferences(self, user_id: str) -> UserPreferenceContext:
        return UserPreferenceContext(
            preferences=self.store.get_preferences(user_id),
            budget_tokens=120,
        )

    # ------------------------------------------------------------------
    # Assembled context
    # ------------------------------------------------------------------

    def compose_context(
        self,
        query: str,
        session_id: str | None = None,
        user_id: str = "",
        conversation_id: str | None = None,
        budget_tokens: int | None = None,
        evidence_ref_ids: list[str] | None = None,
        top_memories: int = 8,
    ) -> AssembledContext:
        """The single call that builds the bounded, ordered context region.

        Ordering (see AssembledContext.render): query -> VERIFIED EVIDENCE
        (current task only) -> working research state -> persistent memory ->
        conversation -> preferences, with explicit boundary markers.
        """
        budget = budget_tokens or self.config.context_tokens
        assembled = AssembledContext(query=query, total_budget_tokens=budget)

        if evidence_ref_ids:
            assembled.evidence = self.get_evidence_context(evidence_ref_ids)
        if session_id:
            assembled.research = self.get_working_research_state(session_id)
        assembled.memory = self.get_relevant_memories(
            query, session_id=session_id,
            budget_tokens=min(budget // 2, 700),
            top_k=top_memories,
        )
        if conversation_id:
            assembled.conversation = self.get_conversation_context(
                conversation_id, budget_tokens=min(budget // 4, 400))
        if user_id:
            assembled.preferences = self.get_user_preferences(user_id)
        return assembled

    # ------------------------------------------------------------------
    # Session lifecycle
    # ------------------------------------------------------------------

    def create_research_session(self, seed_question: str,
                                user_id: str = "",
                                conversation_id: str = "") -> ResearchSessionRecord:
        rec = self.store.create_session(seed_question, user_id=user_id,
                                        conversation_id=conversation_id)
        self.store.add_event("orchestrator", MemoryEventType.SESSION_CREATE,
                             MemoryObjectType.RESEARCH_SESSION, rec.id,
                             {"seed": seed_question[:200],
                              "conversation_id": conversation_id or ""})
        if seed_question.strip():
            self.store.add_question(rec.id, seed_question.strip())
        return rec

    def resume_research_session(self, query: str,
                                user_id: str = "",
                                conversation_id: str | None = None) -> ResearchSessionRecord:
        """Identify-or-create a research session for THIS conversation.

        PER-CHAT MEMORY BOUNDARY (the user contract):
          * when conversation_id is given, the session is bound to the chat:
            the chat's EXISTING active session is resumed (memory persists
            WITHIN a chat), and a chat without one gets a FRESH session
            (clean slate when you open a new chat). Similarity scoring only
            ever runs WITHIN the same conversation - a session from a
            different chat is never resumed, even for a near-identical query.
          * when conversation_id is absent (legacy calls), fall back to the
            old global similarity resume.
        """
        if conversation_id:
            existing = self.store.get_session_for_conversation(
                conversation_id, status=SessionStatus.ACTIVE)
            if existing is not None:
                self.store.touch_session(existing.id)
                self.store.add_event(
                    "orchestrator", MemoryEventType.SESSION_RESUME,
                    MemoryObjectType.RESEARCH_SESSION, existing.id,
                    {"query": query[:200], "conversation_id": conversation_id})
                return existing
            return self.create_research_session(
                query, user_id=user_id, conversation_id=conversation_id)

        # legacy path: no conversation context (CLI, tests)
        hits = self.store.find_sessions(query, user_id=user_id, limit=3)
        if hits and hits[0][1] >= 0.3:
            session = hits[0][0]
            self.store.touch_session(session.id)
            self.store.add_event("orchestrator", MemoryEventType.SESSION_RESUME,
                                 MemoryObjectType.RESEARCH_SESSION, session.id,
                                 {"query": query[:200], "score": round(hits[0][1], 3)})
            return session
        return self.create_research_session(query, user_id=user_id)

    def close_research_session(self, session_id: str) -> None:
        self.store.close_session(session_id)

    def archive_research_session(self, session_id: str) -> None:
        self.store.archive_session(session_id)

    def merge_research_sessions(self, from_id: str, into_id: str) -> None:
        self.store.add_event("orchestrator", MemoryEventType.SESSION_MERGE,
                             MemoryObjectType.RESEARCH_SESSION, into_id,
                             {"from": from_id})
        self.store.merge_sessions(from_id, into_id)

    # ------------------------------------------------------------------
    # Targeted reads
    # ------------------------------------------------------------------

    def get_relevant_claims(
        self, query: str | None = None, session_id: str | None = None,
        as_of: Any = None, limit: int = 8,
    ) -> list[ClaimRecord]:
        if query:
            scored = mem_retrieval.retrieve_claims(
                self.store, query, self.embedder,
                session_ids=[session_id] if session_id else None,
                top_k=limit, min_sim=self.config.retrieve_min_sim)
            return [sm.claim for sm in scored if sm.claim]
        return self.store.get_claims(session_id=session_id)

    def get_unresolved_contradictions(self, session_id: str | None = None) -> list[Any]:
        return self.store.get_contradictions(session_id, unresolved_only=True)

    def get_research_gaps(self, session_id: str | None = None,
                          open_only: bool = True) -> list[Any]:
        return self.store.get_gaps(session_id, open_only=open_only)

    def get_evidence_lineage(self, claim_id: str) -> EvidenceLineage:
        claim = self.store.get_claim(claim_id)
        if claim is None:
            raise KeyError(f"unknown claim: {claim_id}")
        links = self.store.get_claim_evidence_links(claim_id)
        refs = self.store.get_evidence_refs([l.evidence_ref_id for l in links])
        return EvidenceLineage(claim=claim, links=links, evidence_refs=refs)

    # ------------------------------------------------------------------
    # Write path
    # ------------------------------------------------------------------

    def propose_memory(self, candidate: CandidateMemory) -> CandidateMemory:
        return self.pipeline.propose(candidate)

    def validate_memory(self, candidate: CandidateMemory) -> ValidationResult:
        return self.pipeline.validate(candidate)

    def commit_memory(self, candidate: CandidateMemory) -> tuple[str, Any]:
        return self.pipeline.commit(candidate)

    def process_candidate(self, candidate: CandidateMemory) -> tuple[str, Any]:
        return self.pipeline.process(candidate)

    def update_claim(self, claim_id: str, **patch: Any) -> ClaimRecord | None:
        """Append-only update: forbidden to rewrite text/provenance (those
        create supersession instead), allowed for status / revalidation /
        supersession bookkeeping. Audited."""
        forbidden = {"text", "normalized_text", "provenance_class", "id"}
        illegal = forbidden & set(patch)
        if illegal:
            raise ValueError(
                f"cannot rewrite {sorted(illegal)} on claim {claim_id}; "
                f"create a superseding claim instead (append-only memory)")
        allowed = {"status", "needs_revalidation", "last_verified_at",
                   "valid_to", "superseded_by", "superseded_at",
                   "invalidated_at", "confidence"}
        clean = {k: v for k, v in patch.items() if k in allowed}
        updated = self.store.update_claim(claim_id, **clean)
        if updated is not None:
            self.store.add_event("memory_pipeline", MemoryEventType.UPDATE,
                                 MemoryObjectType.CLAIM, claim_id,
                                 {"patch": {k: str(v) for k, v in clean.items()}})
        return updated

    def mark_stale(self, claim_id: str, reason: str = "") -> None:
        self.store.update_claim(claim_id, needs_revalidation=True)
        self.store.add_event("memory_pipeline", MemoryEventType.MARK_STALE,
                             MemoryObjectType.CLAIM, claim_id, {"reason": reason})

    def record_user_text(self, text: str, user_id: str = "",
                         session_id: str = "") -> dict[str, Any]:
        """What user statements become memory? (Deliberately conservative.)

        * explicit durable preferences  -> user_preferences (persist)
        * questions                     -> research questions (persist)
        * medical-ish assertions        -> USER_ASSERTION claim, labeled,
                                           only when reasonably sized
        * transient / injected / tiny   -> dropped
        """
        text = (text or "").strip()
        classification = classify_text(text)
        out: dict[str, Any] = {"classification": classification.reason,
                               "persisted": False, "object_id": ""}
        if not classification.persist:
            return out

        if classification.object_type == MemoryObjectType.USER_PREFERENCE:
            cand = CandidateMemory(
                object_type=classification.object_type,
                payload={"key": classification.key, "value": text},
                provenance_class=ProvenanceClass.USER_ASSERTION,
                user_id=user_id, source_actor="user", source_text=text)
            oid, _ = self.pipeline.process(cand)
            out.update(persisted=True, object_id=oid, kind="preference")
            return out

        if classification.object_type == MemoryObjectType.RESEARCH_QUESTION:
            if session_id:
                q = self.store.add_question(session_id, text)
                out.update(persisted=True, object_id=q.id, kind="question")
            return out

        # user assertion claim — labeled, never evidence
        if len(text) > _USER_ASSERTION_MAX_CHARS:
            return out
        cand = self.pipeline.make_candidate(
            text=text, provenance_class=ProvenanceClass.USER_ASSERTION,
            session_id=session_id, user_id=user_id,
            source_actor="user", source_text=text)
        oid, _ = self.pipeline.process(cand)
        out.update(persisted=True, object_id=oid, kind="user_assertion")
        return out

    # ------------------------------------------------------------------
    # Conversation helper
    # ------------------------------------------------------------------

    def start_conversation(self, user_id: str = "") -> Any:
        return self.store.start_conversation(user_id=user_id)

    def record_conversation_turn(self, conversation_id: str,
                                 user_text: str, assistant_text: str) -> None:
        """Record one user/assistant exchange under a conversation, creating
        the conversation if the id is new (e.g. a frontend conversation)."""
        self.store.ensure_conversation(conversation_id)
        self.store.add_message(conversation_id, "user", user_text)
        self.store.add_message(conversation_id, "assistant", assistant_text)

    # ------------------------------------------------------------------
    # Pipeline integration
    # ------------------------------------------------------------------

    def prepare_run(self, query: str, user_id: str = "",
                    conversation_id: str | None = None) -> RunPrep:
        """Before a research run: resume/identify the session and assemble the
        memory context region. The evidence block is intentionally EMPTY — the
        current run's verified evidence is supplied by the pipeline itself."""
        session = self.resume_research_session(
            query, user_id=user_id, conversation_id=conversation_id)
        assembled = self.compose_context(
            query, session_id=session.id, user_id=user_id,
            conversation_id=conversation_id)
        return RunPrep(session_id=session.id, session=session,
                       context=assembled)

    def record_run(self, result: dict[str, Any], session_id: str = "",
                   user_id: str = "", conversation_id: str | None = None,
                   record_conclusion: bool = True) -> RunRecordStats:
        """After a research run: persist questions, verified evidence refs,
        evidence-derived claims, contradictions, gaps and (labeled) the
        answer conclusion. Idempotent across re-runs (dedup by key/text)."""
        question = (result.get("question") or "").strip() or "research run"
        session = self.store.get_session(session_id) if session_id else None
        if session is None:
            # no explicit session: identify-or-create bound to the same chat.
            # A follow-up run in the SAME conversation resumes the same
            # research session; a new chat gets its own fresh session.
            session = self.resume_research_session(
                question, user_id=user_id, conversation_id=conversation_id)
        session_id = session.id
        self.store.touch_session(session_id)

        stats = RunRecordStats(session_id=session_id,
                               conversation_id=conversation_id or "")

        # 1) research questions from the plan/tasks
        existing_q = {q.question.strip().casefold()
                      for q in self.store.get_questions(session_id)}
        plan_tasks = result.get("tasks") or []
        for task in plan_tasks:
            texts = [task.get("objective") or "", task.get("title") or ""]
            for r in (task.get("evidence_requirements") or []):
                texts.append(r.get("text") or "")
            for t in texts:
                t = t.strip()
                if not t:
                    continue
                if t.casefold() in existing_q:
                    continue
                self.store.add_question(session_id, t)
                existing_q.add(t.casefold())
                stats.questions += 1

        # 2) evidence refs + claims (only verified items participate)
        refs, claim_cands, contradictions, gaps = extract_run_candidates(
            result, session_id, user_id=user_id)

        ref_ids_by_evidence: dict[str, str] = {}
        for ref in refs:
            stored = self.store.add_evidence_ref(ref)
            ref_ids_by_evidence[ref.meta.get("evidence_id", "")] = stored.id

        # re-point candidates at the stored (deduped) ref ids
        claim_by_evidence: dict[str, str] = {}
        for cand in claim_cands:
            cand.evidence_ref_ids = [
                ref_ids_by_evidence.get(cand.payload.get("meta", {}).get("evidence_id", ""),
                                        rid)
                for rid in cand.evidence_ref_ids
            ]
            cand.evidence_ref_ids = [rid for rid in cand.evidence_ref_ids if rid]
            if not cand.evidence_ref_ids:
                continue
            evidence_id = cand.payload.get("meta", {}).get("evidence_id", "")
            self.pipeline.propose(cand)
            vresult = self.pipeline.validate(cand)
            cid, _obj = self.pipeline.commit(cand, vresult)
            claim_by_evidence.setdefault(evidence_id, cid)
            if vresult.dedup_match_id:
                stats.claims_deduped += 1
            else:
                stats.claims_committed += 1

        # contradiction-aware linking: evidence on side B is linked as
        # CONTRADICTED_BY to claims derived from side A (and vice versa), so
        # claim.status can reflect the conflict instead of silently assuming
        # support.
        for raw in result.get("contradictions", []):
            sides = [(raw.get("evidence_a", []), raw.get("evidence_b", [])),
                     (raw.get("evidence_b", []), raw.get("evidence_a", []))]
            for opp_evidence, side_evidence in sides:
                for opp_id in opp_evidence:
                    claim_id = claim_by_evidence.get(opp_id)
                    if not claim_id:
                        continue
                    for side_id in side_evidence:
                        ref_id = ref_ids_by_evidence.get(side_id)
                        if not ref_id:
                            continue
                        self.store.add_claim_evidence_link(
                            ClaimEvidenceLinkRecord(
                                claim_id=claim_id, evidence_ref_id=ref_id,
                                role=EvidenceRole.CONTRADICTED_BY))

        # 3) contradictions (preserved, never collapsed). Claim ids are resolved
        # from the evidence mappings so the dedup pair is meaningful.
        existing_ctr = self.store.get_contradictions(session_id)
        existing_pairs = {
            (c.claim_a_id, c.claim_b_id)
            for c in existing_ctr
        } | {(c.claim_b_id, c.claim_a_id) for c in existing_ctr}
        raw_contradictions = result.get("contradictions", []) or []
        for rec, raw in zip(contradictions, raw_contradictions):
            a_ids = [claim_by_evidence.get(i) for i in (raw.get("evidence_a") or [])]
            b_ids = [claim_by_evidence.get(i) for i in (raw.get("evidence_b") or [])]
            rec.claim_a_id = next((x for x in a_ids if x), "")
            rec.claim_b_id = next((x for x in b_ids if x), "")
            pair = (rec.claim_a_id, rec.claim_b_id)
            if pair in existing_pairs:
                continue
            self.store.add_contradiction(rec)
            existing_pairs.add(pair)
            stats.contradictions += 1

        # 4) gaps (dedup by question text)
        existing_gaps = {g.question.strip().casefold()
                         for g in self.store.get_gaps(session_id)}
        for gap in gaps:
            if gap.question.strip().casefold() not in existing_gaps:
                self.store.add_gap(gap)
                existing_gaps.add(gap.question.strip().casefold())
                stats.gaps += 1

        # 5) answer -> labeled MODEL_INFERENCE conclusion (never evidence)
        answer = result.get("answer") or {}
        summary = (answer.get("summary") or "").strip()
        if record_conclusion and summary and len(summary) <= 2400:
            cand = self.pipeline.make_candidate(
                text=summary,
                provenance_class=ProvenanceClass.MODEL_INFERENCE,
                session_id=session_id, user_id=user_id,
                source_actor="orchestrator",
                source_text=summary,
                meta={
                    "kind": "conclusion",
                    "run_id": result.get("run_id", ""),
                    "evidence_ids": [e.get("id", "") for e in result.get("evidence", [])],
                },
                claim_status="unresolved",
            )
            oid, obj = self.pipeline.process(cand)
            stats.conclusion = oid

        # 5b) the run reached a terminal answer: mark the session's questions
        # answered so future planners see them as ALREADY INVESTIGATED instead
        # of re-deriving the same work, and record the run's conclusion as the
        # session's rolling summary (used by session identification + context).
        if result.get("answer"):
            for q in self.store.get_questions(session_id):
                if q.status == "open":
                    self.store.mark_question_answered(q.id)
            if summary:
                self.store.update_session_summary(session_id, summary)

        # 6) conversation messages
        if conversation_id:
            self.record_conversation_turn(conversation_id, question, summary)

        self.store.add_event("orchestrator", MemoryEventType.COMMIT,
                             MemoryObjectType.RESEARCH_SESSION, session_id,
                             {"run_id": result.get("run_id", ""),
                              "stats": stats.as_dict()})
        stats.events = len(self.store.recent_events(1))
        return stats

    # ------------------------------------------------------------------
    # Background consolidation
    # ------------------------------------------------------------------

    def consolidate(self, staleness_days: int | None = None) -> dict[str, int]:
        """Online-triggerable background pass: dedup / merge / relations /
        contradictions / temporal / staleness. Idempotent."""
        days = self.config.staleness_days if staleness_days is None else staleness_days
        return self.consolidator.run(staleness_days=days)

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def stats(self) -> dict[str, Any]:
        counts = self.store.counts()
        counts["backend"] = type(self.store).__name__
        counts["embedder"] = type(self.embedder).__name__ if self.embedder else "none"
        return counts

    def recent_events(self, limit: int = 50) -> list[Any]:
        return self.store.recent_events(limit=limit)


__all__ = ["MemoryAPI", "RunPrep", "RunRecordStats"]