"""Agentic v2 — persistent research state (the system's external working memory).

The orchestrator does NOT keep the research notebook in conversational context;
it keeps it here. ``ResearchState`` is a plain Pydantic model that records:

  * the original question,
  * research objectives and their status,
  * retrieved documents / candidate passages,
  * verified evidence (with evidence-quality labels),
  * unresolved gaps and contradictions,
  * every action taken (the audit trail used for loop avoidance),
  * the current iteration / confidence / budget.

It is deliberately dependency-free (only Pydantic + stdlib) so it can be
serialized, unit-tested, and inspected without touching the LLM or the corpus.
"""

from __future__ import annotations

import enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class ObjectiveStatus(str, enum.Enum):
    """How well a research objective is currently supported by evidence."""
    OPEN = "open"                              # not yet investigated
    PARTIALLY_SUPPORTED = "partially_supported"
    SUPPORTED = "supported"
    SUPPORTED_WITH_CAVEAT = "supported_with_caveat"
    CONTRADICTED = "contradicted"
    UNRESOLVED = "unresolved"                  # investigated but no clear answer


class EvidenceQuality(str, enum.Enum):
    """How directly a piece of evidence addresses the objective.

    Mirrors the orchestrator spec: DIRECT / INDIRECT / BACKGROUND /
    CONTRADICTORY / ABSENT (plus UNKNOWN when the verifier could not decide).
    """
    DIRECT = "direct"
    INDIRECT = "indirect"
    BACKGROUND = "background"
    CONTRADICTORY = "contradictory"
    ABSENT = "absent"
    UNKNOWN = "unknown"


class ActionType(str, enum.Enum):
    """The orchestrator actions."""
    DECOMPOSE = "DECOMPOSE"
    GLOBAL_RETRIEVE = "GLOBAL_RETRIEVE"
    ENRICH = "ENRICH"                              # UMLS/MeSH query enrichment
    READ_DOCUMENT = "READ_DOCUMENT"
    FIND_SECTIONS = "FIND_SECTIONS"
    VERIFY = "VERIFY"
    SYNTHESIZE = "SYNTHESIZE"
    STOP = "STOP"


# ---------------------------------------------------------------------------
# Research objects
# ---------------------------------------------------------------------------

class ResearchObjective(BaseModel):
    """One coherent evidence question the system must answer."""
    id: str
    statement: str = ""
    intent: str = ""
    evidence_required: List[str] = Field(default_factory=list)
    status: ObjectiveStatus = ObjectiveStatus.OPEN
    gap: str = ""                             # what is still missing
    caveats: List[str] = Field(default_factory=list)
    notes: List[str] = Field(default_factory=list)
    entities: List[str] = Field(default_factory=list)      # for UMLS enrichment
    synonyms: List[str] = Field(default_factory=list)      # UMLS/MeSH terms
    enriched_query: str = ""                  # query after synonym folding


class RetrievedDocument(BaseModel):
    """A document/structural-unit pulled into the working set by retrieval."""
    chunk_id: str = ""
    document_id: str = ""
    section: str = ""
    unit_kind: str = "paragraph"              # paragraph | table | figure
    score: float = 0.0
    text: str = ""
    objective_id: str = ""                    # objective it was retrieved for
    source_action: str = ""                   # which action produced it
    source_query: str = ""                    # the retrieval query used


class CandidatePassage(BaseModel):
    """A passage located *inside* already-retrieved material (local excavation)."""
    chunk_id: str = ""
    document_id: str = ""
    section: str = ""
    text: str = ""
    objective_id: str = ""
    score: float = 0.0
    origin: str = "local_search"              # local_search | read_document


class VerifiedEvidence(BaseModel):
    """One evidence item after verification, with an evidence-quality label."""
    id: str = ""
    objective_id: str = ""
    document_id: str = ""
    chunk_id: str = ""
    section: str = ""
    excerpt: str = ""                         # the supporting text
    quality: EvidenceQuality = EvidenceQuality.UNKNOWN
    support: str = "neutral"                  # supports | contradicts | neutral
    confidence: float = 0.0
    note: str = ""


class ActionRecord(BaseModel):
    """One orchestrator decision + its outcome (the audit trail)."""
    iteration: int
    action: ActionType
    objective_id: str = ""
    rationale: str = ""
    instructions: str = ""
    query: str = ""
    status: str = "running"                   # running | done | failed | repeated
    outcome: str = ""


class StrategyAttempt(BaseModel):
    """One strategy attempt, recorded for loop-avoidance / progress tracking."""
    iteration: int = 0
    objective_id: str = ""
    action: str = ""
    query: str = ""
    document_id: str = ""
    strategy_key: str = ""
    progress_made: bool = False
    progress_summary: str = ""


# ---------------------------------------------------------------------------
# The persistent state
# ---------------------------------------------------------------------------

class ResearchState(BaseModel):
    question: str = ""
    objectives: List[ResearchObjective] = Field(default_factory=list)
    documents: List[RetrievedDocument] = Field(default_factory=list)
    candidates: List[CandidatePassage] = Field(default_factory=list)
    evidence: List[VerifiedEvidence] = Field(default_factory=list)
    gaps: List[str] = Field(default_factory=list)
    contradictions: List[str] = Field(default_factory=list)
    prior_actions: List[ActionRecord] = Field(default_factory=list)
    searched_queries: List[str] = Field(default_factory=list)
    iteration: int = 0
    confidence: float = 0.0
    max_rounds: int = 12
    max_global_retrieves: int = 6
    terminal: bool = False
    stop_reason: str = ""
    final_answer: Optional[Dict[str, Any]] = None

    # State-machine additions (progress guarantees).
    phase: str = ""                              # last computed ResearchPhase
    strategy_history: List[StrategyAttempt] = Field(default_factory=list)
    exhausted_strategies: List[str] = Field(default_factory=list)
    sections_seen: List[str] = Field(default_factory=list)  # "doc:sec:sub" keys

    # -- objectives -----------------------------------------------------

    def objective(self, objective_id: str) -> Optional[ResearchObjective]:
        for obj in self.objectives:
            if obj.id == objective_id:
                return obj
        return None

    def upsert_objective(self, obj: ResearchObjective) -> ResearchObjective:
        """Add or replace an objective by id; returns the stored instance."""
        for i, existing in enumerate(self.objectives):
            if existing.id == obj.id:
                self.objectives[i] = obj
                return obj
        self.objectives.append(obj)
        return obj

    def non_terminal_objectives(self) -> List[ResearchObjective]:
        """Objectives that still need work (not yet supported/contradicted)."""
        terminal = {ObjectiveStatus.SUPPORTED, ObjectiveStatus.CONTRADICTED}
        return [o for o in self.objectives if o.status not in terminal]

    # -- documents / passages / evidence --------------------------------

    def seen_chunk_ids(self) -> set:
        ids = {d.chunk_id for d in self.documents if d.chunk_id}
        ids |= {c.chunk_id for c in self.candidates if c.chunk_id}
        ids |= {e.chunk_id for e in self.evidence if e.chunk_id}
        return ids

    def add_documents(self, docs: List[RetrievedDocument]) -> int:
        """Add documents, deduping by chunk_id (or doc+text when id-less);
        returns how many were NEW."""
        seen = {(d.chunk_id if d.chunk_id else (d.document_id, d.text[:120]))
                for d in self.documents}
        added = 0
        for doc in docs:
            key = doc.chunk_id or (doc.document_id, doc.text[:120])
            if key in seen:
                continue
            seen.add(key)
            self.documents.append(doc)
            added += 1
        return added

    def add_candidates(self, passages: List[CandidatePassage]) -> int:
        """Add candidate passages, deduping by (chunk_id) or (doc, excerpt)."""
        seen = {(c.chunk_id) if c.chunk_id else (c.document_id, c.text[:120])
                for c in self.candidates}
        added = 0
        for p in passages:
            key = p.chunk_id or (p.document_id, p.text[:120])
            if key in seen:
                continue
            seen.add(key)
            self.candidates.append(p)
            added += 1
        return added

    def add_evidence(self, items: List[VerifiedEvidence]) -> int:
        seen = {(e.objective_id, e.chunk_id, e.excerpt[:120]) for e in self.evidence}
        added = 0
        for e in items:
            key = (e.objective_id, e.chunk_id, e.excerpt[:120])
            if key in seen:
                continue
            seen.add(key)
            self.evidence.append(e)
            added += 1
        return added

    # -- action log -----------------------------------------------------

    def record_action(self, decision: Any, status: str = "running",
                      outcome: str = "") -> ActionRecord:
        """Append an action to the audit trail (the loop then finishes it)."""
        rec = ActionRecord(
            iteration=self.iteration,
            action=getattr(decision, "action", ActionType.STOP),
            objective_id=getattr(decision, "objective_id", "") or "",
            rationale=getattr(decision, "rationale", "") or "",
            instructions=getattr(decision, "instructions", "") or "",
            query=getattr(decision, "query", "") or "",
            status=status,
            outcome=outcome,
        )
        self.prior_actions.append(rec)
        return rec

    def finish_last_action(self, status: str = "done", outcome: str = "") -> None:
        if self.prior_actions:
            self.prior_actions[-1].status = status
            self.prior_actions[-1].outcome = outcome

    def last_actions(self, n: int = 8) -> List[ActionRecord]:
        return self.prior_actions[-n:]

    # -- strategy history / progress ------------------------------------

    def record_strategy(self, attempt: StrategyAttempt) -> None:
        """Append a strategy attempt to the history."""
        self.strategy_history.append(attempt)

    def exhaust_strategy(self, key: str) -> bool:
        """Mark a strategy as exhausted; returns True if it was newly added."""
        if key and key not in self.exhausted_strategies:
            self.exhausted_strategies.append(key)
            return True
        return False

    def recent_strategies(self, n: int = 6) -> List[StrategyAttempt]:
        return self.strategy_history[-n:]

    # -- budget ---------------------------------------------------------

    def global_retrieves_used(self) -> int:
        return sum(1 for a in self.prior_actions
                   if a.action == ActionType.GLOBAL_RETRIEVE and a.status == "done")

    # -- summaries ------------------------------------------------------

    def summarize(self, max_text: int = 320, max_evidence: int = 900) -> str:
        """A compact, bounded view of the state for the orchestrator LLM.

        Evidence excerpts get a larger cap than document text because they are
        the answer material: they are term-anchored upstream (actions._verify)
        and must stay visible to the orchestrator (a head-slice would make the
        orchestrator misjudge verified evidence as false positives).
        """
        lines: List[str] = []
        lines.append(f"QUESTION: {self.question}")

        lines.append("OBJECTIVES:")
        if self.objectives:
            for o in self.objectives:
                caveat = f" | caveats={len(o.caveats)}" if o.caveats else ""
                lines.append(
                    f"- {o.id} [{o.status.value}] {o.statement[:140]}"
                    f"{(' | GAP: ' + o.gap[:120]) if o.gap else ''}{caveat}"
                )
        else:
            lines.append("- (none yet)")

        lines.append(f"DOCUMENTS RETRIEVED: {len(self.documents)}")
        for d in self.documents[-20:]:
            lines.append(f"- {d.document_id or d.chunk_id} [{d.section}] "
                         f"{' '.join(d.text.split())[:max_text]}")

        lines.append(f"CANDIDATE PASSAGES: {len(self.candidates)}")

        lines.append(f"VERIFIED EVIDENCE: {len(self.evidence)}")
        for e in self.evidence[-20:]:
            excerpt = " ".join(e.excerpt.split())
            if len(excerpt) > max_evidence:
                excerpt = excerpt[:max_evidence] + "... [truncated]"
            lines.append(f"- [{e.objective_id}] {e.quality.value}/{e.support} "
                         f"conf={e.confidence:.2f} {e.document_id or e.chunk_id} "
                         f"{excerpt}")

        if self.gaps:
            lines.append("UNRESOLVED GAPS: " + " | ".join(self.gaps[-8:]))
        if self.contradictions:
            lines.append("CONTRADICTIONS: " + " | ".join(self.contradictions[-8:]))

        lines.append(f"SEARCHES PERFORMED: {len(self.searched_queries)}")
        for q in self.searched_queries[-6:]:
            lines.append(f"- {q[:120]}")

        lines.append("PRIOR ACTIONS:")
        if self.prior_actions:
            for a in self.last_actions(8):
                lines.append(
                    f"- #{a.iteration} {a.action.value}"
                    f"{('[' + a.objective_id + ']') if a.objective_id else ''}"
                    f" [{a.status}] {a.outcome[:120]}"
                )
        else:
            lines.append("- (none)")

        if self.exhausted_strategies:
            lines.append("EXHAUSTED STRATEGIES: " + " | ".join(self.exhausted_strategies[-8:]))
        if self.strategy_history:
            lines.append("RECENT STRATEGIES:")
            for s in self.recent_strategies(6):
                flag = "progress" if s.progress_made else "NO-PROGRESS"
                lines.append(
                    f"- #{s.iteration} {s.action}"
                    f"{('[' + s.objective_id + ']') if s.objective_id else ''}"
                    f" query={s.query[:60]!r} -> {flag}: {s.progress_summary[:80]}"
                )

        lines.append(
            f"BUDGET: iteration={self.iteration}/{self.max_rounds} "
            f"global_retrieves={self.global_retrieves_used()}/{self.max_global_retrieves}"
        )
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        """JSON-safe serialization (enums -> values)."""
        import json

        return json.loads(self.model_dump_json())

    def __str__(self) -> str:
        return self.summarize()
