"""MedRAG retrieval V2 data model.

Every object carries full provenance (spec section 11: never throw retrieval
provenance away). The paper is the discovery unit, the chunk the evidence
unit, the requirement the intent unit.

The model also carries the V2.1 concepts introduced by the architecture
repair: clinical entities with modifier-preserving terminology, retrieval
variants (V0..Vn), PageIndex node references, table-aware evidence context
and the detected requested fields of a candidate.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Terminology (spec section 8: preserve medical terminology)
# ---------------------------------------------------------------------------


@dataclass
class Terminology:
    """Original / canonical term alignment with synonyms.

    'original_term' is the exact term from the user question and is ALWAYS
    preserved; 'canonical_term' is the accepted concept name; 'synonyms'
    may be used for search but never silently replace the original.
    """

    original_term: str
    canonical_term: str = ""
    synonyms: List[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.canonical_term:
            self.canonical_term = self.original_term

    def to_dict(self) -> Dict[str, Any]:
        return {
            "original_term": self.original_term,
            "canonical_term": self.canonical_term,
            "synonyms": list(self.synonyms),
        }

    def as_text(self) -> str:
        """All searchable surface forms, original first."""
        return " ".join([self.original_term, self.canonical_term] + list(self.synonyms))


@dataclass
class ClinicalEntity:
    """One biomedical entity extracted from the user question (V2.1 part 4).

    The ontology layer NEVER replaces the user's original wording: the
    'surface_form' is always preserved. 'base_concept' is the entity with
    generic modifiers stripped ('recurrent X' -> 'X') and is what gets
    looked up in UMLS/MeSH/BioPortal. 'modifiers' are preserved and
    re-attached for search. 'role' is a generic slot (condition /
    population / target / treatment / outcome / other) - no disease-
    specific branches.
    """

    surface_form: str
    base_concept: str = ""
    modifiers: List[str] = field(default_factory=list)
    role: str = "condition"
    preferred_name: str = ""
    ontology: str = ""          # UMLS | MESH | BIOPORTAL | "" (no lookup)
    cui: str = ""
    synonyms: List[str] = field(default_factory=list)
    abbreviations: List[str] = field(default_factory=list)  # e.g. re-CoA (generic)

    def __post_init__(self) -> None:
        if not self.base_concept:
            self.base_concept = self.surface_form

    def searchable_form(self) -> str:
        """Surface form first, then canonical/synonyms (never the reverse)."""
        parts = [self.surface_form]
        canonical = (
            " ".join(list(self.modifiers) + [self.base_concept])
            if self.modifiers and self.base_concept
            else self.base_concept
        )
        for p in (canonical, *self.synonyms, *self.abbreviations):
            if p and p != self.surface_form and p.lower() not in " ".join(parts).lower():
                parts.append(p)
        return " ".join(parts)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "surface_form": self.surface_form,
            "base_concept": self.base_concept,
            "modifiers": list(self.modifiers),
            "role": self.role,
            "preferred_name": self.preferred_name,
            "ontology": self.ontology,
            "cui": self.cui,
            "synonyms": list(self.synonyms),
            "abbreviations": list(self.abbreviations),
        }


@dataclass
class RetrievalVariant:
    """One compact retrieval query variant for a requirement (V2.1 part 6).

    V0 original query, V1 target + original concept, V2 target +
    modifier-preserved canonical, V3 target + strongest synonym,
    V4 alternate terminology / abbreviation. Generated in the planner,
    executed independently in global AND paper-local retrieval.
    """

    id: str                       # V0, V1, ...
    text: str
    source: str = ""              # original | target+concept | canonical | synonym | alt
    requirement_ids: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "text": self.text,
            "source": self.source,
            "requirement_ids": list(self.requirement_ids),
        }


@dataclass
class PageIndexHit:
    """A node reference returned by PageIndex navigation (V2.1 part 13).

    PageIndex returns NODE REFERENCES, not text blobs; the XML corpus stays the
    source of truth. 'chunk_ids' maps the pageindex node onto existing corpus
    chunk identifiers through the adapter.
    """

    paper_id: str
    pageindex_node_id: str = ""
    title: str = ""
    section: str = ""             # breadcrumb path, e.g. "Results > Recurrent coarctation (re-CoA)"
    chunk_ids: List[str] = field(default_factory=list)
    reason: str = ""
    relevance: float = 0.0
    page_start: Optional[int] = None
    page_end: Optional[int] = None
    level: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "paper_id": self.paper_id,
            "pageindex_node_id": self.pageindex_node_id,
            "title": self.title,
            "section": self.section,
            "chunk_ids": list(self.chunk_ids),
            "reason": self.reason,
            "relevance": round(float(self.relevance), 4),
            "page_start": self.page_start,
            "page_end": self.page_end,
            "level": self.level,
        }


# ---------------------------------------------------------------------------
# Requirements / hops (sections 6-9)
# ---------------------------------------------------------------------------

QUESTION_TYPES = (
    "definition", "factual", "numerical", "comparison", "causal", "mechanism",
    "diagnosis", "treatment", "prognosis", "evidence_synthesis", "multi_hop",
    "table_lookup", "figure_interpretation", "comparative_numerical",
)


@dataclass
class Requirement:
    """One independent evidence requirement (H1, H2, ...).

    A requirement is one genuine evidence obligation: target + clinical
    condition + outcome + requested fields, e.g.

        H1: target=surgical repair techniques, condition=recurrent coarctation,
            focus=comparative_numerical, requested_fields=[percentage, p-value]

    Multiple requirements are created ONLY for genuinely separate evidence
    obligations (multi-hop mechanism+population-difference splits), never
    merely because the /expand endpoint returned several sentences.
    """

    id: str                       # e.g. "H1"
    topic: str                    # e.g. "metabolic syndrome"
    population: str = ""          # e.g. "INOCA"
    target: str = ""              # e.g. "cardiovascular vulnerability"
    focus: str = "mechanism"      # mechanism | outcome | definition | numerical | ...
    outcome: str = ""
    condition: str = ""           # clinical condition incl. modifiers ("recurrent coarctation")
    required_concepts: List[str] = field(default_factory=list)
    preferred_evidence_types: List[str] = field(default_factory=list)
    requested_fields: List[str] = field(default_factory=list)
    terms: List[Terminology] = field(default_factory=list)
    entities: List[ClinicalEntity] = field(default_factory=list)
    retrieval_variants: List[RetrievalVariant] = field(default_factory=list)
    navigation_objective: str = ""   # paper-local PageIndex navigation objective (part 12)
    hop_label: str = ""           # multi-hop description, e.g. "A -> B"
    hop_dependencies: List[str] = field(default_factory=list)
    comparison: bool = False
    n_hops: int = 1

    # ------------------------------------------------------------------
    # Query text builders
    # ------------------------------------------------------------------
    def search_query_text(self) -> str:
        """Compact per-branch lead search query.

        Example: "metabolic syndrome INOCA cardiovascular vulnerability
        mechanisms ... concepts". The original medical terms always stay.
        """
        parts: List[str] = []
        parts.append(self.topic)
        if self.population:
            parts.append(self.population)
        if self.target:
            parts.append(self.target)
        if self.focus:
            parts.append(self.focus)
        if self.outcome and self.outcome.lower() not in str(self.target).lower():
            parts.append(self.outcome)
        joined = " ".join(parts).lower()
        for t in self.terms:
            for extra in (t.canonical_term, *t.synonyms):
                if extra and extra.lower() not in joined and len(parts) < 14:
                    parts.append(extra)
                    joined = " ".join(parts).lower()
        return " ".join(p for p in parts if p)

    def rerank_query_text(self) -> str:
        """Richer requirement representation scored by the cross-encoder.

        The cross-encoder scores requirement <-> evidence, never the giant
        original query <-> evidence (spec section 25). Preserves the joint
        relationship between target, clinical condition, outcome and the
        requested fields (V2.1 part 17).
        """
        lines: List[str] = []
        focus = self.focus or "evidence"
        lines.append("Requirement " + self.id + ": " + focus + ": " + self.topic)
        if self.condition and self.condition.lower() != self.topic.lower():
            lines.append("Clinical condition: " + self.condition)
        if self.population:
            lines.append("Population: " + self.population)
        if self.target:
            lines.append("Target: " + self.target)
        if self.outcome:
            lines.append("Outcome: " + self.outcome)
        if self.required_concepts:
            lines.append("Required concepts: " + ", ".join(self.required_concepts))
        if self.requested_fields:
            lines.append("Requested fields: " + ", ".join(self.requested_fields))
        for t in self.terms:
            extra = [t.original_term, t.canonical_term, *t.synonyms]
            if extra:
                lines.append("Terminology: " + ", ".join(dict.fromkeys(extra)))
        return " | ".join(lines)

    def coverage_key(self) -> str:
        return self.id

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "topic": self.topic,
            "population": self.population,
            "target": self.target,
            "focus": self.focus,
            "outcome": self.outcome,
            "condition": self.condition,
            "required_concepts": list(self.required_concepts),
            "preferred_evidence_types": list(self.preferred_evidence_types),
            "requested_fields": list(self.requested_fields),
            "terms": [t.to_dict() for t in self.terms],
            "entities": [e.to_dict() for e in self.entities],
            "retrieval_variants": [v.to_dict() for v in self.retrieval_variants],
            "navigation_objective": self.navigation_objective,
            "hop_label": self.hop_label,
            "hop_dependencies": list(self.hop_dependencies),
            "comparison": self.comparison,
            "n_hops": self.n_hops,
        }


@dataclass
class SearchQuery:
    """A generated per-branch search query bound to one or more requirements."""

    id: str                       # q1, q2, ... (lead) or q1_v1, q1_v2 (variants)
    requirement_ids: List[str] = field(default_factory=list)
    text: str = ""
    variant_id: str = ""          # V0..V4 when this query is a retrieval variant
    variant_source: str = ""
    intended_population: str = ""
    intended_outcome: str = ""
    branch: str = "hybrid"        # hybrid | bm25 | dense

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "requirement_ids": list(self.requirement_ids),
            "text": self.text,
            "variant_id": self.variant_id,
            "variant_source": self.variant_source,
            "intended_population": self.intended_population,
            "intended_outcome": self.intended_outcome,
            "branch": self.branch,
        }


# ---------------------------------------------------------------------------
# Retrieval provenance (section 11)
# ---------------------------------------------------------------------------


@dataclass
class RetrievalEvent:
    """One raw hit from one query x one retrieval method, fully attributed."""

    query_id: str
    requirement_ids: List[str] = field(default_factory=list)
    paper_id: str = ""
    chunk_id: str = ""
    method: str = "dense"         # bm25 | dense
    variant_id: str = ""          # the retrieval variant this hit belongs to
    rank: int = 0
    score: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query_id": self.query_id,
            "requirement_ids": list(self.requirement_ids),
            "paper_id": self.paper_id,
            "chunk_id": self.chunk_id,
            "method": self.method,
            "variant_id": self.variant_id,
            "rank": self.rank,
            "score": round(float(self.score), 6),
        }


@dataclass
class QueryLocalResult:
    """Per-query retrieval output: raw events + query-local RRF fusion.

    RRF happens ONLY within this query (section 12); branches of different
    requirements never compete at chunk level.
    """

    query: SearchQuery
    events: List[RetrievalEvent] = field(default_factory=list)
    fused: List[Dict[str, Any]] = field(default_factory=list)
    # fused: list of {chunk_id, paper_id, rrf_score, rank, methods, requirement_ids, query_id, variant_id}

    def paper_ids(self) -> List[str]:
        seen: List[str] = []
        for f in self.fused:
            pid = f.get("paper_id")
            if pid and pid not in seen:
                seen.append(pid)
        return seen

    def summary(self) -> Dict[str, Any]:
        ranked = [
            {
                "rank": f.get("rank"),
                "chunk_id": f.get("chunk_id"),
                "paper_id": f.get("paper_id"),
                "rrf_score": round(float(f.get("rrf_score", 0.0)), 6),
                "methods": f.get("methods"),
                "variant_id": f.get("variant_id", ""),
            }
            for f in self.fused[:10]
        ]
        return {
            "query_id": self.query.id,
            "requirement_ids": self.query.requirement_ids,
            "query_text": self.query.text,
            "variant_id": self.query.variant_id,
            "n_events": len(self.events),
            "n_fused": len(self.fused),
            "events_by_method": {
                "bm25": sum(1 for e in self.events if e.method == "bm25"),
                "dense": sum(1 for e in self.events if e.method == "dense"),
            },
            "top10": ranked,
        }


# ---------------------------------------------------------------------------
# Paper-level objects (sections 13-17)
# ---------------------------------------------------------------------------


@dataclass
class PaperQueryScore:
    """paper_query_score for one (query, paper) pair (section 13)."""

    paper_id: str
    query_id: str
    requirement_ids: List[str] = field(default_factory=list)
    max_score: float = 0.0
    mean_topk: float = 0.0
    n_support: int = 0
    section_diversity: float = 0.0
    method_agreement: float = 0.0
    evidence_diversity: float = 0.0
    score: float = 0.0
    best_chunks: List[str] = field(default_factory=list)
    sections: List[str] = field(default_factory=list)
    evidence_types: List[str] = field(default_factory=list)
    supporting_chunks: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "paper_id": self.paper_id,
            "query_id": self.query_id,
            "requirement_ids": list(self.requirement_ids),
            "score": round(float(self.score), 6),
            "max_score": round(float(self.max_score), 6),
            "mean_topk": round(float(self.mean_topk), 6),
            "n_support": self.n_support,
            "section_diversity": round(float(self.section_diversity), 6),
            "method_agreement": round(float(self.method_agreement), 6),
            "evidence_diversity": round(float(self.evidence_diversity), 6),
            "best_chunks": list(self.best_chunks),
            "sections": list(self.sections),
            "evidence_types": list(self.evidence_types),
            "supporting_chunks": list(self.supporting_chunks),
        }


@dataclass
class PaperSelection:
    """Handoff object from the paper-selection layer to V2 local retrieval."""

    paper_id: str
    paper_score: float = 0.0
    supported_requirements: List[str] = field(default_factory=list)
    query_branches: List[str] = field(default_factory=list)
    best_chunks: List[str] = field(default_factory=list)
    sections: List[str] = field(default_factory=list)
    evidence_types: List[str] = field(default_factory=list)
    per_requirement_scores: Dict[str, float] = field(default_factory=dict)
    reason: str = ""
    bridge: bool = False
    guarantee: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "paper_id": self.paper_id,
            "paper_score": round(float(self.paper_score), 6),
            "supported_requirements": list(self.supported_requirements),
            "query_branches": list(self.query_branches),
            "best_chunks": list(self.best_chunks),
            "sections": list(self.sections),
            "evidence_types": list(self.evidence_types),
            "per_requirement_scores": {
                k: round(float(v), 6) for k, v in self.per_requirement_scores.items()
            },
            "reason": self.reason,
            "bridge": self.bridge,
            "guarantee": self.guarantee,
        }


# ---------------------------------------------------------------------------
# V2 local candidates and evidence (sections 18-33)
# ---------------------------------------------------------------------------


@dataclass
class EvidenceCandidate:
    """One evidence candidate produced by V2 local search + expansion.

    Carries per-requirement MedCPT scores, intent slots, penalties and the
    final composited score so every selection decision is explainable.
    V2.1 adds: context_text (table-aware passage scored by MedCPT and intent),
    table_context, detected_fields and PageIndex provenance.
    """

    chunk_id: str
    paper_id: str
    requirement_ids: List[str] = field(default_factory=list)
    source_queries: List[str] = field(default_factory=list)

    # metadata (from the logical document index)
    node_type: str = "paragraph"
    section: str = ""
    subsection: str = ""
    breadcrumb: List[str] = field(default_factory=list)
    position: int = 0
    table_id: Optional[str] = None
    figure_id: Optional[str] = None
    parent_id: Optional[str] = None
    text: str = ""
    token_count: int = 0

    # V2.1 contextual evidence (parts 15-16, 31-32)
    context_text: str = ""                       # the object scored by MedCPT + intent
    context_node_ids: List[str] = field(default_factory=list)  # chunk ids in the context
    table_context: Optional[Dict[str, Any]] = None            # {table_id, caption, headers, rows, footnotes}
    detected_fields: Dict[str, bool] = field(default_factory=dict)
    pageindex_node_ids: List[str] = field(default_factory=list)
    pageindex_reasons: List[str] = field(default_factory=list)

    # local retrieval scores
    local_scores: Dict[str, float] = field(default_factory=dict)
    local_rank: int = 0
    retrieval_methods: List[str] = field(default_factory=list)

    # per-requirement MedCPT scores + normalized
    medcpt_scores: Dict[str, float] = field(default_factory=dict)
    medcpt_norm: Dict[str, float] = field(default_factory=dict)

    # intent layer
    slot_matches: Dict[str, float] = field(default_factory=dict)
    requirement_match: float = 0.0
    evidence_type_bonus: float = 0.0
    requested_field_bonus: float = 0.0
    paper_relevance: float = 0.0
    diversity: float = 1.0
    penalties: Dict[str, float] = field(default_factory=dict)
    penalty_total: float = 0.0

    # composited
    final_score: float = 0.0
    covered_requirements: List[str] = field(default_factory=list)
    selection_rank: int = 0
    selection_reason: Dict[str, Any] = field(default_factory=dict)
    expanded_from: List[str] = field(default_factory=list)

    # helpers
    def base_key(self) -> str:
        return self.chunk_id

    def primary_requirement(self) -> str:
        return self.requirement_ids[0] if self.requirement_ids else ""

    def is_table(self) -> bool:
        return self.node_type in ("table_summary", "table_row", "table_footnotes")

    def is_figure(self) -> bool:
        return self.node_type == "figure"

    def evidence_display_text(self) -> str:
        """Text used by readability / final output."""
        return self.context_text or self.text

    def to_dict(self, include_text: bool = False, max_text: int = 600) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "chunk_id": self.chunk_id,
            "paper_id": self.paper_id,
            "requirement_ids": list(self.requirement_ids),
            "source_queries": list(self.source_queries),
            "node_type": self.node_type,
            "section": self.section,
            "subsection": self.subsection or None,
            "breadcrumb": list(self.breadcrumb),
            "position": self.position,
            "table_id": self.table_id,
            "figure_id": self.figure_id,
            "parent_id": self.parent_id,
            "token_count": self.token_count,
            "context_node_ids": list(self.context_node_ids),
            "table_context": self.table_context,
            "detected_fields": dict(self.detected_fields),
            "pageindex_node_ids": list(self.pageindex_node_ids),
            "pageindex_reasons": list(self.pageindex_reasons),
            "local_scores": {k: round(float(v), 6) for k, v in self.local_scores.items()},
            "local_rank": self.local_rank,
            "retrieval_methods": list(self.retrieval_methods),
            "medcpt_scores": {k: round(float(v), 6) for k, v in self.medcpt_scores.items()},
            "medcpt_norm": {k: round(float(v), 6) for k, v in self.medcpt_norm.items()},
            "slot_matches": {k: round(float(v), 6) for k, v in self.slot_matches.items()},
            "requirement_match": round(float(self.requirement_match), 6),
            "evidence_type_bonus": round(float(self.evidence_type_bonus), 6),
            "requested_field_bonus": round(float(self.requested_field_bonus), 6),
            "paper_relevance": round(float(self.paper_relevance), 6),
            "penalties": {k: round(float(v), 6) for k, v in self.penalties.items()},
            "penalty_total": round(float(self.penalty_total), 6),
            "final_score": round(float(self.final_score), 6),
            "covered_requirements": list(self.covered_requirements),
            "selection_rank": self.selection_rank,
            "selection_reason": self.selection_reason,
            "expanded_from": list(self.expanded_from),
        }
        if include_text:
            d["text"] = (self.text or "")[:max_text]
            if self.context_text:
                d["context_text"] = self.context_text[: max_text * 2]
        return d


# ---------------------------------------------------------------------------
# Plan + coverage
# ---------------------------------------------------------------------------


@dataclass
class V2Plan:
    """Output of the query planner: requirement/hop graph + search queries."""

    query: str = ""
    question_type: str = "factual"
    entities: List[str] = field(default_factory=list)
    clinical_entities: List[ClinicalEntity] = field(default_factory=list)
    conditions: List[str] = field(default_factory=list)
    populations: List[str] = field(default_factory=list)
    query_targets: List[str] = field(default_factory=list)
    requested_fields: List[str] = field(default_factory=list)
    relationships: List[str] = field(default_factory=list)
    target: str = ""
    outcome: str = ""
    comparisons: List[str] = field(default_factory=list)
    n_hops: int = 1
    requirements: List[Requirement] = field(default_factory=list)
    queries: List[SearchQuery] = field(default_factory=list)
    retrieval_variants: List[RetrievalVariant] = field(default_factory=list)
    reranker_intent: Dict[str, Any] = field(default_factory=dict)
    terminology_guard: Dict[str, Any] = field(default_factory=dict)
    planner_method: str = "deterministic"
    warnings: List[str] = field(default_factory=list)

    def requirement_by_id(self, rid: str) -> Optional[Requirement]:
        for r in self.requirements:
            if r.id == rid:
                return r
        return None

    def query_by_id(self, qid: str) -> Optional[SearchQuery]:
        for q in self.queries:
            if q.id == qid:
                return q
        return None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "question_type": self.question_type,
            "entities": list(self.entities),
            "clinical_entities": [e.to_dict() for e in self.clinical_entities],
            "conditions": list(self.conditions),
            "populations": list(self.populations),
            "query_targets": list(self.query_targets),
            "requested_fields": list(self.requested_fields),
            "relationships": list(self.relationships),
            "target": self.target,
            "outcome": self.outcome,
            "comparisons": list(self.comparisons),
            "n_hops": self.n_hops,
            "requirements": [r.to_dict() for r in self.requirements],
            "queries": [q.to_dict() for q in self.queries],
            "retrieval_variants": [v.to_dict() for v in self.retrieval_variants],
            "reranker_intent": self.reranker_intent,
            "terminology_guard": self.terminology_guard,
            "planner_method": self.planner_method,
            "warnings": list(self.warnings),
        }


@dataclass
class CoverageReport:
    """Final coverage accounting (section 32): honest, never fabricated."""

    all_requirements_covered: bool = False
    covered: List[str] = field(default_factory=list)
    uncovered: List[str] = field(default_factory=list)
    coverage_fraction: float = 0.0
    per_requirement: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "all_requirements_covered": self.all_requirements_covered,
            "covered": list(self.covered),
            "uncovered": list(self.uncovered),
            "coverage_fraction": round(float(self.coverage_fraction), 4),
            "per_requirement": self.per_requirement,
        }
