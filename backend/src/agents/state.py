"""x_deepagents state models.

Pure data - no LLM, no corpus access - so the whole model layer is
unit-testable offline. These are NEW models for the deepagents + LangGraph
runtime (the agreed plan: only prompts + tools are reused; state is new).

Central invariant (the same one the v3 pipeline proved):

    Retrieved text is not evidence until verified.

So every candidate walks RETRIEVED -> UNDER_REVIEW -> one of
ACCEPTED / REJECTED / CONTRADICTORY, and only ACCEPTED / CONTRADICTORY
material can ever feed contradiction analysis or synthesis. This module
holds the data + explicit transition helpers; the guardrails that make the
transitions MANDATORY live in rules.py.
"""

from __future__ import annotations

import enum
from typing import Any

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Evidence hierarchy (deterministic, unit-testable)
# ---------------------------------------------------------------------------

_STUDY_TYPE_PATTERNS: list[tuple[str, tuple[str, ...]]] = [
    ("meta_analysis", ("meta-analysis", "meta analysis", "systematic review and meta-analysis", "pooled analysis")),
    ("systematic_review", ("systematic review", "systematic literature review")),
    ("rct", ("randomized controlled trial", "randomised controlled trial", "randomized trial", "randomised trial", "double-blind", "placebo-controlled trial")),
    ("cohort", ("cohort study", "prospective cohort", "retrospective cohort", "longitudinal study")),
    ("case_control", ("case-control", "case control")),
    ("guideline", ("clinical guideline", "practice guideline", "guideline recommends", "the guideline")),
]

import re as _re


def web_document_id(source_url: str, fallback: str = "") -> str:
    """Stable document id for a verified web page.

    A verified web page is an independent source: it needs a document id so
    it counts toward requirement coverage (independence rule), like a corpus
    document would. The slug keeps it human-readable; the url-hash suffix
    keeps distinct URLs from colliding after truncation.
    """
    import hashlib

    url = (source_url or "").strip()
    slug = _re.sub(r"[^a-z0-9]+", "-", url.lower()).strip("-")[:60]
    digest = hashlib.sha1(url.encode("utf-8")).hexdigest()[:12]
    if slug:
        return f"web:{slug}:{digest}"
    return f"web:{(fallback or digest)}"


def classify_study_type(text: str) -> str:
    """Deterministic study-type hint from passage text (best effort).

    This is a cheap signal used for evidence hierarchy; it never replaces
    the verifier. Unknown -> "unknown".
    """
    t = (text or "").lower()
    for kind, pats in _STUDY_TYPE_PATTERNS:
        for p in pats:
            if p in t:
                return kind
    return "unknown"



# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class VerdictRelevance(str, enum.Enum):
    """How related a passage is to the evidence requirement."""
    RELEVANT = "relevant"
    PARTIALLY_RELEVANT = "partially_relevant"
    NOT_RELEVANT = "not_relevant"


class AnswersTask(str, enum.Enum):
    """Does the passage actually answer the required relationship?

    Only ANSWERS_TASK = YES may pass the verification gate - and it passes
    EVEN when it contradicts, because contradictory evidence is real evidence
    that must reach contradiction analysis rather than being dropped here.
    """
    YES = "yes"
    NO = "no"
    PARTIAL = "partial"


class SupportDirection(str, enum.Enum):
    SUPPORTS = "supports"
    CONTRADICTS = "contradicts"
    NEUTRAL = "neutral"


class EvidenceStatus(str, enum.Enum):
    """Per-evidence-item state machine (mirrors the v3 invariant):
    RETRIEVED -> UNDER_REVIEW -> ACCEPTED / REJECTED / CONTRADICTORY."""
    RETRIEVED = "retrieved"
    UNDER_REVIEW = "under_review"
    ACCEPTED = "accepted"
    REJECTED = "rejected"
    CONTRADICTORY = "contradictory"


class RequirementStatus(str, enum.Enum):
    UNSATISFIED = "unsatisfied"              # no supporting paper yet
    PARTIALLY_SUPPORTED = "partially_supported"  # 1..N-1 supporting papers
    SATISFIED = "satisfied"                  # >= N supporting papers
    EXHAUSTED = "exhausted"                  # budget spent below N


class ContradictionKind(str, enum.Enum):
    DIRECT_CONFLICT = "direct_conflict"      # same claim, opposite findings
    CONTEXT_DEPENDENT = "context_dependent"  # differs by population/design/dose
    ANOMALY = "anomaly"                      # important outlier


class ResolutionStatus(str, enum.Enum):
    RESOLVED = "resolved"
    PARTIALLY_RESOLVED = "partially_resolved"
    UNRESOLVED = "unresolved"


class GapResolutionStatus(str, enum.Enum):
    """How an evidence gap was (or was not) closed by the gap-resolution pass."""
    RESOLVED_LOCAL = "resolved_local"   # closed with additional local retrievals
    RESOLVED_WEB = "resolved_web"       # closed with trust-gated web search
    PARTIALLY_RESOLVED = "partially_resolved"  # some evidence, but important
                                               # facets remain -> still listed
    UNRESOLVED = "unresolved"           # still missing -> listed in the answer


class Phase(str, enum.Enum):
    """The workflow phases. ORDER matters and is enforced by rules.py -
    the LLM chooses actions INSIDE a phase, never the phase order."""
    DECOMPOSE = "decompose"                  # split the question into obligations
    RETRIEVE = "retrieve"                    # per-requirement tool loops
    VERIFY = "verify"                        # mandatory verification gate
    CONTRADICTION = "contradiction"          # cross-evidence check (mandatory)
    RESOLUTION = "resolution"                # resolve discovered contradictions
    GAP_RESOLUTION = "gap_resolution"        # close remaining evidence gaps:
                                             #   more local retrievals -> web ->
                                             #   list what is still missing
    SYNTHESIS = "synthesis"                  # evidence-gated final answer
    DONE = "done"

    @classmethod
    def ordered(cls) -> list["Phase"]:
        return [p for p in cls]


# ---------------------------------------------------------------------------
# Verification gate objects
# ---------------------------------------------------------------------------

class VerifierVerdict(BaseModel):
    """The verification-gate judgement on ONE passage vs ONE requirement.

    Scope-stamped BEFORE judgement (evidence_id / requirement_id) so a
    verdict can never be attached to a different item.
    """
    evidence_id: str = ""
    requirement_id: str = ""
    relevance: VerdictRelevance = VerdictRelevance.NOT_RELEVANT
    answers_task: AnswersTask = AnswersTask.NO
    support: SupportDirection = SupportDirection.NEUTRAL
    confidence: float = 0.0
    note: str = ""

    @property
    def accepted(self) -> bool:
        """Only passages that actually answer the requirement pass the gate."""
        return self.answers_task == AnswersTask.YES


class EvidenceItem(BaseModel):
    """One candidate / verified evidence item (deepagents shape).

    Full provenance from chunk to claim: chunk_id/document_id/section ->
    source_query/retrieval_method/rank -> verdict -> terminal status.
    """
    id: str = ""                      # evidence_id, unique within the run
    run_id: str = ""
    requirement_id: str = ""
    chunk_id: str = ""
    document_id: str = ""
    section: str = ""
    unit_kind: str = "paragraph"
    text: str = ""                    # expanded context (full unit)
    claim: str = ""                   # the specific claim the passage supports
    source_query: str = ""
    retrieval_method: str = ""        # e.g. "hybrid:bm25+dense", "pgfts+pgvector",
                                      # "web:<engine>", "deep_inspection"
    rank: int = 0
    round_no: int = 0
    # --- source provenance (web + evidence hierarchy) ---------------------
    source_url: str = ""              # web evidence: the fetched page URL
    trust: str = ""                   # web site gate tier: "trusted" | "unverified"
    publication_date: str = ""        # "YYYY-MM-DD" or "" when unknown
    study_type: str = ""              # meta_analysis | rct | cohort | observational
                                      # | guideline | case_series | unknown
    journal: str = ""                 # journal / publisher when known
    reliability: str = ""             # reliability critic: high | medium | low | None
    evidence_level: str = ""          # computed hierarchy label (1a..4)

    status: EvidenceStatus = EvidenceStatus.RETRIEVED
    verdict: VerifierVerdict | None = None

    def derive_evidence_level(self) -> str:
        """Deterministic evidence-hierarchy label from study_type + reliability."""
        out = classify_study_type(self.text or self.claim or "")
        if out == "meta_analysis":
            level = "1a"
        elif out == "systematic_review":
            level = "1a"
        elif out == "rct":
            level = "1b"
        elif out == "cohort":
            level = "2b"
        elif out == "case_control":
            level = "3b"
        elif out == "guideline":
            level = "1a"
        else:
            level = "4"
        # a web/consumer source can never be primary evidence-level
        if self.trust == "consumer" or self.reliability == "low":
            level = "4"
        self.study_type = self.study_type or out
        self.evidence_level = level
        return level
    # -- explicit transitions ------------------------------------------

    def submit_to_verifier(self) -> None:
        """RETRIEVED -> UNDER_REVIEW. Nothing else may happen to a candidate
        until the verifier has judged it (rule: chunk -> verifier)."""
        if self.status is not EvidenceStatus.RETRIEVED:
            raise ValueError(
                f"item {self.id} is {self.status.value}; only RETRIEVED may be "
                "submitted to the verifier"
            )
        self.status = EvidenceStatus.UNDER_REVIEW

    def set_verdict(self, verdict: VerifierVerdict) -> None:
        """UNDER_REVIEW -> ACCEPTED / REJECTED / CONTRADICTORY.

        The state transition is DERIVED from the verdict - an ACCEPTED item
        must carry an ANSWERS_TASK=YES verdict structurally.
        """
        if verdict.evidence_id and verdict.evidence_id != self.id:
            raise ValueError(
                f"verdict scope {verdict.evidence_id!r} != evidence {self.id!r}"
            )
        if self.status is EvidenceStatus.RETRIEVED:
            self.submit_to_verifier()
        if self.status is not EvidenceStatus.UNDER_REVIEW:
            raise ValueError(
                f"item {self.id} is {self.status.value}; only UNDER_REVIEW may "
                "receive a verdict"
            )
        self.verdict = verdict
        if not verdict.accepted:
            self.status = EvidenceStatus.REJECTED
        elif verdict.support is SupportDirection.CONTRADICTS:
            self.status = EvidenceStatus.CONTRADICTORY
        else:
            self.status = EvidenceStatus.ACCEPTED

    @property
    def verified(self) -> bool:
        """The ONLY thing that may enter contradiction analysis / synthesis."""
        return self.status in (EvidenceStatus.ACCEPTED,
                               EvidenceStatus.CONTRADICTORY)

    @property
    def confidence_float(self) -> float:
        """Verifier confidence for UI scoring (0.0 when not yet judged)."""
        try:
            return float(self.verdict.confidence) if self.verdict else 0.0
        except (TypeError, ValueError):
            return 0.0

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json")


# ---------------------------------------------------------------------------
# Requirements + run state
# ---------------------------------------------------------------------------

class ResearchRequirement(BaseModel):
    """One evidence obligation with its minimum-evidence target."""
    id: str = ""
    text: str = ""
    target_n: int = 3
    items: list[EvidenceItem] = Field(default_factory=list)
    status: RequirementStatus = RequirementStatus.UNSATISFIED
    gap: str = ""
    notes: list[str] = Field(default_factory=list)

    def add_item(self, item: EvidenceItem) -> None:
        self.items.append(item)

    def verified_items(self) -> list[EvidenceItem]:
        return [i for i in self.items if i.verified]

    def supporting_papers(self) -> list[str]:
        """DISTINCT documents with at least one SUPPORTS item (independence)."""
        return sorted({i.document_id for i in self.items
                       if i.document_id
                       and i.verified
                       and i.verdict is not None
                       and i.verdict.support is SupportDirection.SUPPORTS})

    def coverage(self) -> int:
        return len(self.supporting_papers())

    def satisfied(self) -> bool:
        return self.coverage() >= max(1, self.target_n)

    def derive_status(self) -> RequirementStatus:
        if self.satisfied():
            self.status = RequirementStatus.SATISFIED
        elif self.coverage() >= 1:
            self.status = RequirementStatus.PARTIALLY_SUPPORTED
        else:
            self.status = RequirementStatus.UNSATISFIED
        return self.status

    def mark_exhausted(self, gap: str = "") -> None:
        if not self.satisfied():
            self.status = RequirementStatus.EXHAUSTED
            self.gap = gap or self.gap

    def summary(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "text": self.text,
            "target_n": self.target_n,
            "coverage": self.coverage(),
            "supporting_papers": self.supporting_papers(),
            "verified": len(self.verified_items()),
            "status": self.status.value,
            "gap": self.gap,
        }


class Contradiction(BaseModel):
    """A cross-evidence contradiction / anomaly detected after verification."""
    id: str = ""
    claim: str = ""
    requirement_id: str = ""      # the requirement this contradiction belongs to
    evidence_a: list[str] = Field(default_factory=list)   # evidence ids, side A
    evidence_b: list[str] = Field(default_factory=list)   # evidence ids, side B
    evidence_a_text: str = ""    # full passage text, side A (for the resolver)
    evidence_b_text: str = ""    # full passage text, side B
    kind: ContradictionKind = ContradictionKind.DIRECT_CONFLICT
    description: str = ""
    resolution: ResolutionOutcome | None = None


class ResolutionOutcome(BaseModel):
    status: ResolutionStatus = ResolutionStatus.UNRESOLVED
    explanation: str = ""
    additional_queries: list[str] = Field(default_factory=list)
    additional_papers: list[str] = Field(default_factory=list)
    new_evidence_ids: list[str] = Field(default_factory=list)  # VERIFIED items
    characterization: str = ""
    search_rounds: int = 0       # how many tool searches were actually run


class GapResolution(BaseModel):
    """One evidence-gap attempt: what was missing, what the gap pass did, and
    whether it was closed locally, closed via web, or is still listed."""
    requirement_id: str = ""          # the unsatisfied requirement
    gap: str = ""                     # human-readable description of the gap
    status: GapResolutionStatus = GapResolutionStatus.UNRESOLVED
    evidence_ids: list[str] = Field(default_factory=list)   # NEW verified items
    queries_used: list[str] = Field(default_factory=list)
    note: str = ""
    searches_used: int = 0            # web searches consumed by this gap

    @property
    def resolved(self) -> bool:
        return self.status in (GapResolutionStatus.RESOLVED_LOCAL,
                               GapResolutionStatus.RESOLVED_WEB)


class RunBudget(BaseModel):
    """Hard stop-condition budgets (enforced by the graph, not the LLM)."""
    max_searches: int = 5
    max_retrieval_rounds: int = 5
    max_papers_per_round: int = 5
    evidence_target: int = 3
    searches_used: int = 0
    retrieval_rounds_used: int = 0
    # Gap-resolution pass (after contradiction resolution)
    max_gap_local_rounds: int = 3      # extra LOCAL retrieval rounds per gap
    max_gap_web_searches: int = 3      # web searches per gap when local fails
    gap_local_rounds_used: int = 0
    gap_web_searches_used: int = 0

    def search_budget_exhausted(self) -> bool:
        return self.searches_used >= self.max_searches

    def rounds_exhausted(self) -> bool:
        return self.retrieval_rounds_used >= self.max_retrieval_rounds

    def gap_local_exhausted(self) -> bool:
        return self.gap_local_rounds_used >= self.max_gap_local_rounds

    def gap_web_exhausted(self) -> bool:
        return self.gap_web_searches_used >= self.max_gap_web_searches

    def gap_exhausted(self) -> bool:
        return self.gap_local_exhausted() and self.gap_web_exhausted()

    def exhausted(self) -> bool:
        return (self.search_budget_exhausted() or self.rounds_exhausted()
                or self.gap_exhausted())


class XDeepRunState(BaseModel):
    """Working memory for one x_deepagents run (plain data, no LLM).

    run_id scopes every evidence id; the graph merges worker output ONLY
    through this object and the Review/Requirement models - never through
    shared mutable globals.
    """
    run_id: str = ""
    question: str = ""
    phase: Phase = Phase.DECOMPOSE
    requirements: list[ResearchRequirement] = Field(default_factory=list)
    contradictions: list[Contradiction] = Field(default_factory=list)
    gap_resolutions: list[GapResolution] = Field(default_factory=list)
    budget: RunBudget = Field(default_factory=RunBudget)
    gaps: list[str] = Field(default_factory=list)
    answer: str = ""
    terminal: bool = False
    stop_reason: str = ""

    def set_phase(self, phase: Phase) -> None:
        self.phase = phase

    def requirement(self, req_id: str) -> ResearchRequirement | None:
        for r in self.requirements:
            if r.id == req_id:
                return r
        return None

    def all_items(self) -> list[EvidenceItem]:
        return [i for r in self.requirements for i in r.items]

    def verified_items(self) -> list[EvidenceItem]:
        """ACCEPTED + CONTRADICTORY only - the synthesis input boundary."""
        return [i for r in self.requirements for i in r.verified_items()]

    def evidence_by_id(self) -> dict[str, EvidenceItem]:
        return {i.id: i for i in self.all_items()}

    def verified_ids(self) -> set[str]:
        return {i.id for i in self.verified_items()}

    def unresolved_contradictions(self) -> list[Contradiction]:
        return [c for c in self.contradictions
                if c.resolution is None
                or c.resolution.status is not ResolutionStatus.RESOLVED]


# Pydantic forward references
Contradiction.model_rebuild()
GapResolution.model_rebuild()
XDeepRunState.model_rebuild()
