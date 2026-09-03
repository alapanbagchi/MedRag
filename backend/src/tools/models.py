"""Tool I/O contracts: the pure-data shapes the retrieval/terminology tools
speak (tasks, requirements, candidate papers, verdicts, terminology).

These models moved here verbatim from the retired v3 pipeline state
(``src.agentic.state``): the tools outlived the pipeline, so their input
contracts live with the tools now. Pure pydantic data - no LLM, no corpus
access. The singular agentic runtime is ``src.agents`` (deepagents +
LangGraph) with its own state; these shapes exist only so the shared
non-LLM capabilities keep stable, pipeline-independent I/O.
"""

from __future__ import annotations

import enum
from typing import Any

from pydantic import BaseModel, Field


class SupportDirection(str, enum.Enum):
    """Direction of the verified claim relative to the requirement."""
    SUPPORTS = "supports"
    CONTRADICTS = "contradicts"
    NEUTRAL = "neutral"


class EvidenceStatus(str, enum.Enum):
    """Explicit per-evidence-item state machine."""
    RETRIEVED = "retrieved"            # returned by the retriever (candidate)
    UNDER_REVIEW = "under_review"      # currently being judged by the CRITIC
    ACCEPTED = "accepted"              # CRITIC: answers_task=yes, supports
    REJECTED = "rejected"              # CRITIC: does not answer / off-topic
    CONTRADICTORY = "contradictory"    # CRITIC: answers_task=yes, contradicts


class RequirementStatus(str, enum.Enum):
    """Explicit per-requirement state."""
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


class TermConcept(BaseModel):
    """One UMLS-enriched concept (terminology pool / variants)."""
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


class VerifiedEvidence(BaseModel):
    """One critic-gated evidence item (ACCEPTED or CONTRADICTORY)."""
    id: str = ""
    run_id: str = ""
    task_id: str = ""
    requirement_id: str = ""
    attempt_id: str = ""
    document_id: str = ""
    chunk_id: str = ""
    section: str = ""
    excerpt: str = ""
    claim: str = ""
    support: SupportDirection = SupportDirection.SUPPORTS
    confidence: float = 0.0
    source: EvidenceSource = EvidenceSource.RETRIEVAL
    status: EvidenceStatus = EvidenceStatus.ACCEPTED
    retrieval_method: str = ""
    rank: int = 0
    critic_verdict: dict[str, Any] | None = None
    note: str = ""
    search_query: str = ""

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json")


class RetrievedPaper(BaseModel):
    """CANDIDATE evidence from the retriever - NOT verified.

    The 'text' field is the EXPANDED context (the whole containing paragraph /
    table / figure), because a bare search excerpt is never enough to judge
    whether a passage actually supports the requirement.
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
    retrieval_method: str = ""
    rank: int = 0
    score: float = 0.0
    text: str = ""                    # expanded context (full unit)
    source_query: str = ""
    round_no: int = 0


class EvidenceRequirement(BaseModel):
    """One evidence obligation of a task, with its minimum-evidence target."""
    id: str = ""
    text: str = ""
    target_n: int = 3
    accepted: list[VerifiedEvidence] = Field(default_factory=list)
    reviewed: list[dict[str, Any]] = Field(default_factory=list)
    rejected: int = 0
    status: RequirementStatus = RequirementStatus.UNSATISFIED
    gap: str = ""
    caveats: list[str] = Field(default_factory=list)

    def add_evidence(self, item: VerifiedEvidence, seen: set | None = None) -> bool:
        """Add ONE critic-accepted evidence item (deduped). Returns True if new."""
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
        if self.satisfied():
            return RequirementStatus.SATISFIED
        if self.coverage() >= 1:
            return RequirementStatus.PARTIALLY_SUPPORTED
        return RequirementStatus.UNSATISFIED

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


class ResearchTask(BaseModel):
    """One research task: objective, evidence obligations, terminology pool."""
    id: str = ""
    title: str = ""
    objective: str = ""
    intent: str = ""
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

    def satisfied(self) -> bool:
        return bool(self.evidence_requirements) and all(
            r.satisfied() for r in self.evidence_requirements
        )


__all__ = [
    "EvidenceRequirement", "EvidenceSource", "EvidenceStatus",
    "RequirementStatus", "ResearchTask", "RetrievedPaper",
    "SupportDirection", "TaskStatus", "TermConcept", "VerifiedEvidence",
]
