"""Agentic v3 - evidence-vetted multi-agent retrieval (state models).

This module is the data backbone of the V1 pipeline:

    QUERY -> MASTER PLAN -> PARALLEL WORKERS -> VERIFIED EVIDENCE
            -> CONTRADICTION AGENT -> RESOLUTION AGENT
            -> FINAL EVIDENCE SET -> FINAL ANSWER

It deliberately contains NO LLM / corpus code (like src/agentic_v2.state),
so the whole model layer can be unit-tested offline.

Central principle encoded here (spec section 1):

    Retrieved text is not evidence until it has been verified as relevant
    and actually supportive of the task.

Hence the distinction between:
  * RetrievedPaper      - candidate evidence coming out of retrieval;
  * CriticVerdict       - the verification-gate judgement on one passage;
  * VerifiedEvidence    - only critic-ACCEPTED material may become this,
                          and only it counts toward the N-paper threshold.

Independence + deduplication (spec section 13): the same paper must never
count as multiple independent articles. Every requirement counts DISTINCT
document_ids, not accepted items.
"""

from __future__ import annotations

import enum
from typing import Any

from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class CriticRelevance(str, enum.Enum):
    """How related a passage is to the worker's evidence requirement."""
    RELEVANT = "relevant"
    PARTIALLY_RELEVANT = "partially_relevant"
    NOT_RELEVANT = "not_relevant"


class AnswersTask(str, enum.Enum):
    """Does the passage actually answer the task / evidence requirement?

    Per the spec's CRITIC gate (section 10): a passage can be relevant to
    the topic yet still NOT answer the required relationship (e.g. "vitamin
    D deficiency is common among hypertensives" does not answer "does
    supplementation lower blood pressure?"). Only ANSWERS_TASK = YES passes.
    """
    YES = "yes"
    NO = "no"
    PARTIAL = "partial"


class SupportDirection(str, enum.Enum):
    """Direction of the verified claim relative to the requirement."""
    SUPPORTS = "supports"
    CONTRADICTS = "contradicts"
    NEUTRAL = "neutral"


class EvidenceStatus(str, enum.Enum):
    """Explicit per-evidence-item state machine.

    Every candidate retraces: RETRIEVED -> UNDER_REVIEW -> one of
    ACCEPTED / REJECTED / CONTRADICTORY. The orchestrator drives on these
    states, never on implicit behavior (requirement 2).
    """
    RETRIEVED = "retrieved"            # returned by the retriever (candidate)
    UNDER_REVIEW = "under_review"      # currently being judged by the CRITIC
    ACCEPTED = "accepted"              # CRITIC: answers_task=yes, supports
    REJECTED = "rejected"              # CRITIC: does not answer / off-topic
    CONTRADICTORY = "contradictory"    # CRITIC: answers_task=yes, contradicts


class RequirementStatus(str, enum.Enum):
    """Explicit per-requirement state (UNSATISFIED -> PARTIALLY_SUPPORTED ->
    SATISFIED; EXHAUSTED when the budget was spent below N)."""
    UNSATISFIED = "unsatisfied"            # no independently supported paper yet
    PARTIALLY_SUPPORTED = "partially_supported"  # 1..N-1 supported papers
    SATISFIED = "satisfied"                # N independent papers verified
    EXHAUSTED = "exhausted"                # budget spent before N was reached


class TaskStatus(str, enum.Enum):
    PENDING = "pending"
    RUNNING = "running"
    SATISFIED = "satisfied"            # every evidence requirement satisfied
    INSUFFICIENT = "insufficient"      # investigated but evidence below N
    EXHAUSTED = "exhausted"            # retrieval budget exhausted


class EvidenceSource(str, enum.Enum):
    """Where the accepted evidence came from."""
    RETRIEVAL = "retrieval"            # ordinary search + context expansion
    DEEP_INSPECTION = "deep_inspection"


class ContradictionKind(str, enum.Enum):
    DIRECT_CONFLICT = "direct_conflict"      # same claim, opposite findings
    CONTEXT_DEPENDENT = "context_dependent"  # findings differ by population/design/dose
    ANOMALY = "anomaly"                      # important irregularity / outlier


class ResolutionStatus(str, enum.Enum):
    RESOLVED = "resolved"
    PARTIALLY_RESOLVED = "partially_resolved"
    UNRESOLVED = "unresolved"


# ---------------------------------------------------------------------------
# Master-plan objects (Stage 2)
# ---------------------------------------------------------------------------

class EvidenceRequirement(BaseModel):
    """One evidence obligation of a task, with its minimum-evidence target.

    spec: each evidence requirement has a minimum evidence target, e.g.
    N = 3 independent supporting articles. The worker tracks coverage per
    requirement and does NOT stop after one relevant paper.
    """
    id: str = ""
    text: str = ""                    # e.g. "Association between vitamin D status and hypertension"
    target_n: int = 3                 # N independent supporting articles
    accepted: list[VerifiedEvidence] = Field(default_factory=list)
    reviewed: list[dict[str, Any]] = Field(default_factory=list)
    # scoped review log: one entry per judged candidate (accepted OR rejected)
    # carrying {evidence_id, document_id, chunk_id, attempt_id, relevance,
    # answers_task, support, confidence, note} - the CRITIC reasoning that
    # the replanner consumes. Keeps provenance without any shared globals.
    rejected: int = 0                 # count of REJECTED passages
    status: RequirementStatus = RequirementStatus.UNSATISFIED
    gap: str = ""                     # what is still unsupported (set at exhaustion)
    caveats: list[str] = Field(default_factory=list)

    # -- verification-before-counting ----------------------------------

    def add_evidence(self, item: VerifiedEvidence, seen: set | None = None) -> bool:
        """Add ONE critic-accepted evidence item (deduped). Returns True if new.

        Dedup key = (document_id, chunk_id, excerpt head). The same paper
        retrieved by many searches may only enter once (spec section 13).
        """
        if seen is not None and item.document_id in seen and not item.chunk_id:
            return False
        for existing in self.accepted:
            if (existing.document_id == item.document_id
                    and existing.chunk_id == item.chunk_id
                    and existing.excerpt[:120] == item.excerpt[:120]):
                return False
        self.accepted.append(item)
        self.status = self.derive_status()
        return True

    def derive_status(self) -> RequirementStatus:
        """Explicit transition function used by the orchestrator.

        UNSATISFIED (0 supporting papers) -> PARTIALLY_SUPPORTED
        (1..N-1) -> SATISFIED (>= N). EXHAUSTED is set by mark_exhausted()
        only when the budget is spent while still below N.
        """
        if self.satisfied():
            return RequirementStatus.SATISFIED
        if self.coverage() >= 1:
            return RequirementStatus.PARTIALLY_SUPPORTED
        return RequirementStatus.UNSATISFIED

    def record_review(self, entry: dict[str, Any]) -> None:
        """Append ONE scoped critic review of a candidate (accepted or not).

        This is the only place review records are written - the requirement
        is the single owner of its review log, so critic output can never be
        attached to a different requirement (requirement 3).
        """
        self.reviewed.append(entry)

    def verified(self) -> list[VerifiedEvidence]:
        """Evidence gated by the CRITIC: ACCEPTED + CONTRADICTORY only."""
        return [e for e in self.accepted
                if e.status in (EvidenceStatus.ACCEPTED,
                                EvidenceStatus.CONTRADICTORY)]

    def supporting_papers(self) -> list[str]:
        """DISTINCT document ids with at least one SUPPORTS item (independence)."""
        return sorted({e.document_id for e in self.accepted
                       if e.document_id and e.support == SupportDirection.SUPPORTS})

    def contradicting_papers(self) -> list[str]:
        """DISTINCT document ids carrying at least one CONTRADICTS item."""
        return sorted({e.document_id for e in self.accepted
                       if e.document_id and e.support == SupportDirection.CONTRADICTS})

    def coverage(self) -> int:
        """Number of independent supporting papers (the N the spec counts)."""
        return len(self.supporting_papers())

    def satisfied(self) -> bool:
        return self.coverage() >= max(1, self.target_n)

    def mark_exhausted(self, gap: str = "") -> None:
        if self.satisfied():
            self.status = RequirementStatus.SATISFIED
            return
        self.status = (RequirementStatus.EXHAUSTED if self.accepted
                      else RequirementStatus.UNSATISFIED)
        self.gap = gap or self.gap

    def summary(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "text": self.text,
            "target_n": self.target_n,
            "coverage": self.coverage(),
            "supporting_papers": self.supporting_papers(),
            "contradicting_papers": self.contradicting_papers(),
            "accepted": len(self.accepted),
            "rejected": self.rejected,
            "status": self.status.value,
            "gap": self.gap,
        }


class ResearchTask(BaseModel):
    """One Master task dispatched to one Worker sub-orchestrator.

    spec section 4: a task carries an objective, the evidence required,
    a minimum-evidence threshold (N) and stop criteria. Workers are
    independent and retrieve in parallel (spec section 5).
    """
    id: str = ""
    title: str = ""                    # short label, e.g. "Diet -> hypertension"
    objective: str = ""                # what the task must establish
    intent: str = ""                   # the analysis the evidence must support
    evidence_requirements: list[EvidenceRequirement] = Field(default_factory=list)
    stop_criteria: list[str] = Field(default_factory=list)
    entities: list[str] = Field(default_factory=list)
    terminology: list[TermConcept] = Field(default_factory=list)  # UMLS pool
    status: TaskStatus = TaskStatus.PENDING
    searches_used: int = 0
    deep_inspections_used: int = 0
    notes: list[str] = Field(default_factory=list)

    def requirement(self, req_id: str) -> EvidenceRequirement | None:
        for r in self.evidence_requirements:
            if r.id == req_id:
                return r
        return None

    def all_evidence(self) -> list[VerifiedEvidence]:
        return [e for r in self.evidence_requirements for e in r.accepted]

    def satisfied(self) -> bool:
        return bool(self.evidence_requirements) and all(
            r.satisfied() for r in self.evidence_requirements
        )

    def any_evidence(self) -> bool:
        return any(r.accepted for r in self.evidence_requirements)

    def uncovered(self) -> list[EvidenceRequirement]:
        return [r for r in self.evidence_requirements if not r.satisfied()]

    def finalize(self) -> None:
        if self.satisfied():
            self.status = TaskStatus.SATISFIED
        elif self.any_evidence():
            self.status = TaskStatus.EXHAUSTED
        else:
            self.status = TaskStatus.INSUFFICIENT

    def summary(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "objective": self.objective,
            "status": self.status.value,
            "searches_used": self.searches_used,
            "deep_inspections_used": self.deep_inspections_used,
            "evidence_requirements": [r.summary() for r in self.evidence_requirements],
        }


class MasterPlan(BaseModel):
    """The structured retrieval plan produced by the Master Orchestrator."""
    question: str = ""
    tasks: list[ResearchTask] = Field(default_factory=list)
    global_stop_criteria: list[str] = Field(default_factory=list)
    rationale: str = ""


class TermConcept(BaseModel):
    """One UMLS-enriched concept (Stage 4: terminology pool / variants)."""
    surface_form: str = ""
    preferred_name: str = ""
    cui: str = ""
    synonyms: list[str] = Field(default_factory=list)

    def all_terms(self) -> list[str]:
        out = [self.surface_form]
        if self.preferred_name:
            out.append(self.preferred_name)
        out.extend(self.synonyms)
        seen: set = set()
        clean: list[str] = []
        for t in out:
            t = " ".join((t or "").split())
            if t and t.casefold() not in seen:
                seen.add(t.casefold())
                clean.append(t)
        return clean


# ---------------------------------------------------------------------------
# Retrieval + verification objects (Stages 6-11)
# ---------------------------------------------------------------------------

class RetrievedPaper(BaseModel):
    """CANDIDATE evidence from the retriever - NOT verified (spec section 8).

    The 'text' field is the EXPANDED context (the whole containing paragraph /
    table / figure), because a bare search excerpt is never enough to judge
    whether a passage actually supports the requirement.

    Every candidate carries its own evidence_id plus the FULL scope
    (task_id, requirement_id, attempt_id) so a critic verdict can never be
    attached to the wrong task/requirement/attempt (requirement 3). The
    initial state is always RETRIEVED.
    """
    evidence_id: str = ""
    task_id: str = ""
    requirement_id: str = ""
    attempt_id: str = ""
    status: EvidenceStatus = EvidenceStatus.RETRIEVED
    chunk_id: str = ""
    document_id: str = ""
    section: str = ""
    unit_kind: str = "paragraph"      # paragraph | table | figure
    retrieval_method: str = ""        # e.g. "hybrid:bm25+dense" provenance
    rank: int = 0
    score: float = 0.0
    text: str = ""                    # expanded context (full unit)
    source_query: str = ""
    round_no: int = 0


class CriticVerdict(BaseModel):
    """The verification-gate judgement on ONE passage vs ONE requirement.

    The scope fields (task_id, requirement_id, attempt_id, evidence_id) are
    stamped by the critic BEFORE judgement so a verdict is structurally tied
    to exactly one (task, requirement, attempt, evidence) - a critic result
    is impossible to attach elsewhere (requirement 3).
    """
    run_id: str = ""
    task_id: str = ""
    requirement_id: str = ""
    attempt_id: str = ""
    evidence_id: str = ""
    document_id: str = ""
    chunk_id: str = ""
    section: str = ""
    relevance: CriticRelevance = CriticRelevance.NOT_RELEVANT
    answers_task: AnswersTask = AnswersTask.NO
    support: SupportDirection = SupportDirection.NEUTRAL
    confidence: float = 0.0
    note: str = ""                    # why accepted / rejected (spec section 10)

    @property
    def accepted(self) -> bool:
        """Only passages that actually answer the requirement pass the gate.

        A passage judged ANSWERS_TASK=YES passes EVEN when it contradicts -
        contradictory evidence is real evidence and must reach the
        Contradiction Agent rather than being dropped here (spec section 19).
        """
        return self.answers_task == AnswersTask.YES


class VerifiedEvidence(BaseModel):
    """One critic-gated evidence item (ACCEPTED or CONTRADICTORY terminal
    state) - the ONLY material the synthesizer may cite.

    Carries the FULL auditable provenance chain (requirement 8):
      id (evidence_id) -> critic verdict -> retrieval result (query, method,
      rank) -> task/requirement/attempt scope.
    """
    id: str = ""                      # evidence_id (unique within the run)
    run_id: str = ""
    task_id: str = ""
    requirement_id: str = ""
    attempt_id: str = ""
    document_id: str = ""
    chunk_id: str = ""
    section: str = ""
    excerpt: str = ""                 # the supporting passage (expanded context)
    claim: str = ""                   # the specific claim the passage supports
    support: SupportDirection = SupportDirection.SUPPORTS
    confidence: float = 0.0
    source: EvidenceSource = EvidenceSource.RETRIEVAL
    status: EvidenceStatus = EvidenceStatus.ACCEPTED
    retrieval_method: str = ""        # provenance: how it was retrieved
    rank: int = 0                     # provenance: rank in the retrieval result
    critic_verdict: dict[str, Any] | None = None   # the CRITIC snapshot
    note: str = ""
    search_query: str = ""

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json")


class RequirementReport(BaseModel):
    """Per-requirement evidence package inside a WorkerReport."""
    requirement_id: str = ""
    text: str = ""
    target_n: int = 3
    coverage: int = 0
    status: str = "open"
    gap: str = ""
    papers: list[dict[str, Any]] = Field(default_factory=list)


class WorkerReport(BaseModel):
    """The structured evidence package a Worker returns on completion.

    spec section 18: each worker returns its task plus the evidence it
    gathered, grouped by evidence requirement, with the papers supporting
    each requirement. The orchestrator merges workers ONLY through these
    structured objects (requirement 3) - workers never touch shared state.
    """
    run_id: str = ""
    task_id: str = ""
    task_title: str = ""
    status: str = "pending"           # satisfied | partially_supported | exhausted
    stop_reason: str = ""             # satisfied | search_budget | rounds_exhausted
    requirements: list[RequirementReport] = Field(default_factory=list)
    evidence: list[VerifiedEvidence] = Field(default_factory=list)
    searches_used: int = 0
    deep_inspections_used: int = 0
    budget_exhausted: bool = False
    gaps: list[str] = Field(default_factory=list)
    summary: str = ""

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json")


# ---------------------------------------------------------------------------
# Contradiction + resolution objects (Stages 14-22)
# ---------------------------------------------------------------------------

class Contradiction(BaseModel):
    """A global contradiction / anomaly detected across all workers' evidence."""
    id: str = ""
    claim: str = ""                   # the disputed claim
    task_id: str = ""
    requirement_id: str = ""
    evidence_a: list[str] = Field(default_factory=list)   # evidence ids, side A
    evidence_b: list[str] = Field(default_factory=list)   # evidence ids, side B
    kind: ContradictionKind = ContradictionKind.DIRECT_CONFLICT
    description: str = ""
    resolution: ResolutionOutcome | None = None


class ResolutionOutcome(BaseModel):
    """Outcome of the dedicated Contradiction Resolution Agent."""
    status: ResolutionStatus = ResolutionStatus.UNRESOLVED
    explanation: str = ""             # how it was resolved (or why not)
    additional_queries: list[str] = Field(default_factory=list)
    additional_papers: list[str] = Field(default_factory=list)  # doc ids consulted
    characterization: str = ""        # e.g. "context-dependent: baseline vitamin D status"


class FinalEvidenceSet(BaseModel):
    """Stage 23: the full verified evidence package feeding the answer stage."""
    question: str = ""
    tasks: list[dict[str, Any]] = Field(default_factory=list)
    evidence: list[VerifiedEvidence] = Field(default_factory=list)
    gaps: list[str] = Field(default_factory=list)
    contradictions: list[Contradiction] = Field(default_factory=list)
    resolved: list[Contradiction] = Field(default_factory=list)
    unresolved: list[Contradiction] = Field(default_factory=list)
    confidence: float = 0.0


# ---------------------------------------------------------------------------
# Budget + run state
# ---------------------------------------------------------------------------

class RunBudget(BaseModel):
    """Explicit worker stop-condition budgets (spec section 17 B / req 9).

    A worker stops when the evidence requirement is satisfied OR any of the
    hard budgets is exhausted - never beyond (no infinite agent loops).
    """
    max_searches: int = 5             # max query attempts per requirement
    max_retrieval_rounds: int = 5     # max adaptive retrieval rounds
    max_papers_per_round: int = 5     # max candidate papers per search
    max_deep_inspections: int = 3     # max deep paper inspections per task
    evidence_target: int = 3          # default N independent articles
    max_workers: int = 4              # parallel workers
    searches_used: int = 0            # per-worker attempt counter
    retrieval_rounds_used: int = 0    # per-worker round counter
    deep_inspections_used: int = 0

    def search_budget_exhausted(self) -> bool:
        return self.searches_used >= self.max_searches

    def rounds_exhausted(self) -> bool:
        return self.retrieval_rounds_used >= self.max_retrieval_rounds

    def exhausted(self) -> bool:
        if self.search_budget_exhausted() or self.rounds_exhausted():
            return True
        # max_deep_inspections == 0 means deep inspection is DISABLED - it
        # must never look like an exhausted budget from the start.
        if self.max_deep_inspections > 0 and \
                self.deep_inspections_used >= self.max_deep_inspections:
            return True
        return False


class V3RunState(BaseModel):
    """External working memory for one v3 run (no LLM; plain data).

    run_id scopes every task/requirement/evidence id in the run; worker
    results are merged here ONLY through explicit objects (WorkerReport /
    ResearchTask / VerifiedEvidence), never via mutable shared globals.
    """
    run_id: str = ""
    question: str = ""
    plan: MasterPlan | None = None
    tasks: list[ResearchTask] = Field(default_factory=list)
    contradictions: list[Contradiction] = Field(default_factory=list)
    budget: RunBudget = Field(default_factory=RunBudget)
    gaps: list[str] = Field(default_factory=list)
    terminal: bool = False
    stop_reason: str = ""
    final_answer: dict[str, Any] | None = None
    iterations: int = 0

    def add_task(self, task: ResearchTask) -> None:
        for i, existing in enumerate(self.tasks):
            if existing.id == task.id:
                self.tasks[i] = task
                return
        self.tasks.append(task)

    def all_evidence(self) -> list[VerifiedEvidence]:
        return [e for t in self.tasks for e in t.all_evidence()]

    def verified_evidence(self) -> list[VerifiedEvidence]:
        """Evidence that passed the CRITIC (ACCEPTED + CONTRADICTORY).

        This is the ONLY input to contradiction analysis and synthesis -
        REJECTED / UNDER_REVIEW items are structurally excluded (reqs 5-7).
        """
        out: list[VerifiedEvidence] = []
        for t in self.tasks:
            for r in t.evidence_requirements:
                out.extend(r.verified())
        return out

    def evidence_by_id(self) -> dict[str, VerifiedEvidence]:
        return {e.id: e for e in self.all_evidence()}


# Pydantic forward references
EvidenceRequirement.model_rebuild()
ResearchTask.model_rebuild()
Contradiction.model_rebuild()
V3RunState.model_rebuild()
