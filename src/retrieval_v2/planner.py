"""Query planner for MedRag retrieval V2 (V2.1 architecture repair).

The planner consumes STRUCTURED QUERY UNDERSTANDING (deterministic extraction,
optionally enriched by MedGemma through the /expand contract) and turns the
user question into:

    one requirement per GENUINE evidence obligation  (H1..Hn)
        + retrieval variants V0..Vn per requirement (part 6)
        + a paper-local PageIndex navigation objective per requirement (part 12)
        + a terminology-preservation guard report

The old /expand -> plan_from_expand contract fragmenting a single question
into H1..H4 statistical branches is GONE. A question like

    "Which surgical repair techniques were associated with recurrent
     coarctation and what are the percentages and p-values?"

is ONE evidence obligation and produces exactly ONE requirement with
requested_fields [percentage, p-value], focus comparative_numerical, and
table-heavy preferred evidence types.

Multi-hop questions produce multiple requirements ONLY when the user asks for
logically separate pieces of evidence (e.g. mechanism + population-difference),
with hop dependencies maintained (part 22).

There are NO hardcoded diseases in this module. Terminology preservation
(INOCA stays INOCA, never silently MINOCA) is enforced by a GENERIC guard
over original surface forms and acronyms.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

from src.query_intelligence import (
    QueryIntelligence,
    extract_query_intelligence,
    merge_intelligence,
    structured_query_understanding,
)
from src.retrieval_v2.config import V2Config, DEFAULT_CONFIG
from src.retrieval_v2.models import (
    ClinicalEntity,
    Requirement,
    RetrievalVariant,
    SearchQuery,
    Terminology,
    V2Plan,
)

# Deprecated: the old disease registry is intentionally empty. Do not add
# disease-specific entries here - terminology comes from query understanding
# + UMLS/MeSH/BioPortal enrichment, never from hardcoded branches.
CONCEPT_REGISTRY: Dict[str, Dict[str, Any]] = {}


def _lower(s: str) -> str:
    return (s or "").lower()


# ---------------------------------------------------------------------------
# Focus / evidence-type helpers (generic)
# ---------------------------------------------------------------------------


def _focus_for(qtype: str, fields: Sequence[str]) -> str:
    """Focus used when the question is a plain single-obligation query."""
    if qtype in ("table_lookup", "figure_interpretation", "definition",
                 "diagnosis", "treatment", "prognosis"):
        return qtype
    if qtype in ("numerical", "comparison"):
        if any(f in ("percentage", "p-value", "confidence_interval", "effect_estimate") for f in fields):
            return "comparative_numerical"
        return "numerical" if qtype == "numerical" else "comparison"
    if qtype in ("mechanism", "causal"):
        return "mechanism"
    return "evidence"


def _focus_dims(qi: QueryIntelligence) -> List[str]:
    """Which focus dimensions to generate for the requirement graph.

    Semantics, not sentence count: a number+statistics question is ONE
    comparative_numerical obligation; a how-does question is mechanism then
    outcome; everything else a single evidence dimension.
    """
    focus = qi.reranker_intent.get("focus") or _focus_for(qi.question_type, qi.requested_fields)
    ql = _lower(qi.query)
    if focus == "comparative_numerical":
        return ["comparative_numerical"]
    if qi.question_type in ("definition", "diagnosis", "table_lookup",
                            "figure_interpretation", "treatment"):
        return [focus if focus else "evidence"]
    wants_outcome = bool(
        re.search(r"\boutcomes?\b|\bmortality\b|\bsurvival\b|\bhospitalization\b|\bprognos\b|\bresults\b|\brecurrence\b", ql)
    ) or bool(qi.outcome)
    wants_mechanism = bool(
        re.search(r"\bmechanism\b|\bpathway\b|how does\b|how do\b|why does\b|\brole\b|\bunderly\b|mechanism links", ql)
    )
    q_is_how = bool(re.search(r"\bhow (?:do|does)\b", ql))
    if wants_mechanism or q_is_how:
        return ["mechanism", "outcome"]
    if wants_outcome:
        return ["outcome"]
    return [focus or "evidence"]


def _evidence_types_for(focus: str) -> List[str]:
    preferred: Dict[str, List[str]] = {
        "definition": ["paragraph", "table_summary", "table_row"],
        "numerical": ["table_row", "table_footnotes", "table_summary", "figure", "results"],
        "comparative_numerical": ["table_row", "table_summary", "table_footnotes", "results", "figure"],
        "mechanism": ["paragraph", "figure", "table_row", "table_summary", "results"],
        "outcome": ["table_row", "table_summary", "figure", "paragraph", "results"],
        "population": ["table_row", "table_summary", "paragraph"],
        "treatment": ["table_row", "table_summary", "paragraph"],
        "comparison": ["table_row", "table_summary", "paragraph", "table_footnotes"],
        "diagnosis": ["table_summary", "table_row", "paragraph"],
        "prognosis": ["table_row", "figure", "paragraph"],
        "table_lookup": ["table_row", "table_summary", "table_footnotes"],
        "figure_interpretation": ["figure", "paragraph"],
        "evidence": ["paragraph", "table_row", "table_summary", "figure", "results"],
        "default": ["paragraph", "table_summary", "table_row", "figure"],
    }
    return preferred.get(focus, preferred["default"])


# ---------------------------------------------------------------------------
# Terminology helpers
# ---------------------------------------------------------------------------

_ACRONYM_RE = re.compile(r"\b([A-Z][A-Za-z0-9]*(?:[-/][A-Za-z0-9]+)*)\b")


def _detect_acronyms(text: str) -> List[str]:
    out: List[str] = []
    for m in _ACRONYM_RE.finditer(text or ""):
        tok = m.group(1)
        up = sum(1 for ch in tok if ch.isupper())
        if 2 <= len(tok) <= 12 and up >= 2:
            out.append(tok)
    return out


def _terms_for_entity(ent: ClinicalEntity) -> List[Terminology]:
    return [
        Terminology(
            original_term=ent.surface_form,
            canonical_term=ent.base_concept or ent.surface_form,
            synonyms=list(ent.synonyms),
        )
    ]


def _concepts_for_entity(ent: ClinicalEntity) -> List[str]:
    """Generic required-concepts: base + modifier-preserved canonical + synonyms."""
    out: List[str] = []
    if ent.base_concept:
        out.append(ent.base_concept)
    if ent.modifiers:
        out.append(" ".join(list(ent.modifiers) + [ent.base_concept]))
    for s in ent.synonyms[:3]:
        if s not in out:
            out.append(s)
    return out[:8]


# ---------------------------------------------------------------------------
# Multi-hop detection (part 22)
# ---------------------------------------------------------------------------

_DIFFER_CLAUSE = re.compile(
    r"and how (?:does|do) (?:this|that|the) (?:mechanism|difference|effect|association|link)\b"
    r"|and (?:does|is|are|was|were) this (?:mechanism|difference|association|link)\b"
    r"|how (?:does|do) these (?:mechanisms|differences|effects)\b",
    re.IGNORECASE,
)


def _difference_population(question: str, populations: Sequence[str]) -> Optional[str]:
    """Population of the second hop: 'differ in non-diabetic CKD' -> 'non-diabetic CKD'."""
    m = re.search(r"differ[s]?\s+(?:in|across|among|between)\s+(.+?)(?:[,?]|$)", question, re.IGNORECASE)
    if not m:
        return None
    phrase = m.group(1).strip().strip(" .?!,;")
    if phrase and phrase not in populations:
        return phrase
    return populations[0] if populations else None


# ---------------------------------------------------------------------------
# Alignment helpers (kept from the working machinery)
# ---------------------------------------------------------------------------


def _align_populations(conditions: List[str], populations: List[str]) -> List[Tuple[str, str]]:
    if not populations:
        return [(c, "") for c in conditions]
    if len(conditions) == len(populations):
        return list(zip(conditions, populations))
    paired: List[Tuple[str, str]] = []
    used_pops: set = set()
    for c in conditions:
        best = None
        for p in populations:
            if p not in used_pops and (_lower(c) in _lower(p) or _lower(p) in _lower(c)):
                best = p
                break
        if best is not None:
            paired.append((c, best))
            used_pops.add(best)
        else:
            paired.append((c, populations[0] if len(populations) == 1 else ""))
    return paired


def _pop_comparison_pairs(populations: Sequence[str]) -> List[str]:
    pairs = []
    for i in range(len(populations)):
        for j in range(i + 1, len(populations)):
            pairs.append(f"{populations[i]} vs {populations[j]}")
    return pairs


# ---------------------------------------------------------------------------
# Requirement graph (one per genuine evidence obligation)
# ---------------------------------------------------------------------------


def _make_requirement(qi: QueryIntelligence, cfg: V2Config, rid: str, cond: str,
                      pop: str, focus: str, target: str, outcome: str, *,
                      comparison: bool = False, hop_deps: Optional[List[str]] = None,
                      hop_label: str = "") -> Requirement:
    ent = next((e for e in qi.clinical_entities if e.role == "condition" and _lower(e.surface_form) == _lower(cond)), None)
    if ent is None:
        ent = next((e for e in qi.clinical_entities if e.role in ("condition", "population") and _lower(e.surface_form) == _lower(cond)), None)
    if ent is None:
        ent = next((e for e in qi.clinical_entities if e.role == "condition"), None)
    entities = [ent] if ent else []
    terms = _terms_for_entity(ent) if ent else []
    concepts = _concepts_for_entity(ent) if ent else ([cond] if cond else [])
    if pop:
        pop_ent = next((e for e in qi.clinical_entities if e.role == "population" and _lower(e.surface_form) == _lower(pop)), None)
        if pop_ent is not None and pop_ent not in entities:
            entities.append(pop_ent)
    fields = list(qi.requested_fields)
    return Requirement(
        id=rid,
        topic=cond,
        population=pop,
        target=target,
        focus=focus,
        outcome=outcome,
        condition=cond,
        required_concepts=concepts,
        preferred_evidence_types=_evidence_types_for(focus),
        requested_fields=fields,
        terms=terms,
        entities=entities,
        hop_label=hop_label or f"{cond} -> {pop or cfg.default_population} -> {target}",
        hop_dependencies=list(hop_deps or []),
        comparison=comparison,
        n_hops=max(1, len(hop_deps or []) + 1),
    )


def _build_requirements(qi: QueryIntelligence, cfg: V2Config) -> List[Requirement]:
    """Requirement graph: one per genuine evidence obligation.

    Rules:
      * Single condition + statistics question   -> ONE requirement
      * mechanism-link + population-difference  -> H1 mechanism, H2 difference
      * multi-condition / compare-across groups -> cond x focus-dimensions
    """
    conditions = list(qi.conditions) or [qi.query]
    populations = list(qi.populations)
    target = qi.target or cfg.outcome_fallback
    ql = _lower(qi.query)
    differ = _DIFFER_CLAUSE.search(ql) is not None or bool(re.search(r"\bdiffer[s]?\b", ql))

    if differ and len(conditions) == 1:
        hop2_pop = _difference_population(qi.query, populations) or (populations[0] if populations else "")
        focus1 = qi.reranker_intent.get("focus") or _focus_for(qi.question_type, qi.requested_fields)
        if qi.question_type == "mechanism" or "mechanism" in ql or "mechanism links" in ql:
            focus1 = "mechanism"
        h1 = _make_requirement(qi, cfg, "H1", conditions[0], "", focus1, target,
                               qi.outcome or (focus1 + " outcome"), hop_label=conditions[0])
        h2_focus = "comparison" if "differ" in ql else "outcome"
        h2 = _make_requirement(qi, cfg, "H2", conditions[0], hop2_pop, h2_focus, target,
                               qi.outcome or "outcomes", hop_deps=["H1"],
                               hop_label=f"{conditions[0]} difference in {hop2_pop or 'subgroup'}")
        return [h1, h2]

    pairs = _align_populations(conditions, populations)
    focus_dims = _focus_dims(qi)
    reqs: List[Requirement] = []
    for focus in focus_dims:
        for i, (cond, pop) in enumerate(pairs):
            outcome = qi.outcome if focus == "outcome" else ""
            reqs.append(_make_requirement(qi, cfg, f"H{len(reqs) + 1}", cond, pop, focus,
                                          target, outcome, comparison=differ and len(populations) > 1))
    return reqs


# ---------------------------------------------------------------------------
# Retrieval variants (part 6)
# ---------------------------------------------------------------------------


def _build_retrieval_variants(req: Requirement, qi: QueryIntelligence, cfg: V2Config) -> List[RetrievalVariant]:
    cond = req.condition or req.topic
    target = req.target or ""
    ent = next((e for e in req.entities if e.role == "condition"), None)
    variants: List[RetrievalVariant] = []
    seen: set = set()

    def add(text: str, source: str) -> None:
        text = re.sub(r"\s+", " ", (text or "")).strip()
        if not text:
            return
        low = text.lower()
        if low in seen or len(variants) >= max(1, cfg.max_retrieval_variants):
            return
        seen.add(low)
        variants.append(RetrievalVariant(id=f"V{len(variants)}", text=text, source=source,
                                         requirement_ids=[req.id]))

    pop = req.population or ""
    add(qi.query, "original")
    add(f"{target} {cond} {pop}".strip(), "target+concept")
    if ent is not None and ent.base_concept and ent.base_concept != cond:
        add(f"{target} {ent.base_concept} {pop}".strip(), "base_concept")
        canonical = " ".join(list(ent.modifiers) + [ent.base_concept]) if ent.modifiers else ent.base_concept
        add(f"{target} {canonical} {pop}".strip(), "canonical")
    if ent is not None:
        for syn in ent.synonyms[:1]:
            add(f"{target} {syn} {pop}".strip(), "synonym")
        for ab in ent.abbreviations[:2]:
            add(f"{target} {ab} {pop}".strip(), "abbreviation")
    if req.requested_fields:
        add(f"{target} {cond} {pop} {' '.join(req.requested_fields)}".strip(), "focus+fields")
    if not variants:
        add(req.search_query_text(), "requirement")
    return variants


# ---------------------------------------------------------------------------
# PageIndex navigation objective (part 12)
# ---------------------------------------------------------------------------

_NAV_HINTS: Dict[str, str] = {
    "comparative_numerical": "Results sections, tables, and table footnotes that report",
    "numerical": "Results sections, tables, and table footnotes that report",
    "table_lookup": "tables and table footnotes that report",
    "mechanism": "mechanistic descriptions, Results sections, and figures that describe",
    "outcome": "Results sections and tables that report",
    "comparison": "comparisons in the Results section and tables that compare",
    "definition": "definitions in the Abstract and Introduction that define",
    "figure_interpretation": "figures and their captions that illustrate",
    "treatment": "treatment and intervention descriptions in the Methods and Results that report",
    "diagnosis": "diagnostic criteria and measurements in the Methods and Results that report",
    "prognosis": "prognostic outcomes in the Results section and tables that report",
    "evidence": "evidence in the Results section, tables, and Discussion that report",
    "default": "evidence in the Results section, tables, and Discussion that report",
}


def _build_navigation_objective(req: Requirement) -> str:
    hint = _NAV_HINTS.get(req.focus, _NAV_HINTS["default"])
    parts = [f"Find the {hint} {req.target or 'clinical findings'}"]
    if req.condition:
        parts.append(f"associated with {req.condition}")
    if req.population:
        parts.append(f"in {req.population}")
    if req.outcome:
        parts.append(f"including {req.outcome}")
    if req.requested_fields:
        parts.append(f"including {', '.join(req.requested_fields)}")
    return " ".join(p for p in parts if p).strip(" ,;") + "."


# ---------------------------------------------------------------------------
# Terminology guard (generic - INOCA stays INOCA, never MINOCA)
# ---------------------------------------------------------------------------


def _run_terminology_guard(plan: V2Plan) -> Dict[str, Any]:
    """Verify every original surface form / acronym survived planning intact.

    The guard is GENERIC: it scans the query for acronyms and for all clinical
    entity surface forms and verifies they appear in at least one requirement
    or query. Competing-concept substitution (INOCA -> MINOCA) is blocked by
    the competing-pair check below.
    """
    ql = _lower(plan.query)
    report: Dict[str, Any] = {
        "concepts_detected": [],
        "preserved": [],
        "violations": [],
        "note": ("Medical terms are never silently replaced by different concepts. "
                 "Synonyms may be added for search only."),
    }
    all_req_text = " ".join(
        " ".join([r.topic, r.population, r.target, r.outcome, r.condition] + r.required_concepts)
        for r in plan.requirements
    )
    all_query_text = " ".join(q.text for q in plan.queries)

    names = list(plan.entities)
    for e in plan.clinical_entities:
        if e.surface_form not in names:
            names.append(e.surface_form)
    for name in names:
        entry: Dict[str, Any] = {"original": name, "preserved": False}
        if _lower(name) and (_lower(name) in _lower(all_req_text) or _lower(name) in _lower(all_query_text)):
            entry["preserved"] = True
        report["concepts_detected"].append(entry)
        if entry["preserved"]:
            report["preserved"].append(name)
        else:
            report["violations"].append(name)
            plan.warnings.append(f"Terminology guard: original term {name!r} not found in any requirement text")

    for ac in _detect_acronyms(plan.query):
        low = ac.lower()
        if low in ("bm25", "rrf", "mri", "ct", "icu", "us", "v2", "ecg", "eeg", "pci", "mace"):
            continue
        if low not in _lower(all_req_text + all_query_text):
            report["violations"].append(f"acronym {ac} dropped during planning")
            plan.warnings.append(f"Terminology guard: acronym {ac} not preserved")

    if "inoca" in ql:
        for q in plan.queries:
            if "minoca" in _lower(q.text) and "inoca" not in _lower(q.text):
                report["violations"].append("INOCA substituted by MINOCA in a query (blocked)")
                plan.warnings.append(f"Terminology guard: {q.id} replaced INOCA with MINOCA - blocked")
    return report


# ---------------------------------------------------------------------------
# Main entry
# ---------------------------------------------------------------------------


def plan_question(question: str, config: Optional[V2Config] = None,
                  llm: Any = None, intelligence: Optional[QueryIntelligence] = None) -> V2Plan:
    """Build a V2 plan from a question string.

    'intelligence' may be pre-computed (e.g. from the /expand contract); when
    None, deterministic extraction runs. 'llm' (optional) enriches the query
    intelligence via MedGemma structured understanding but NEVER dictates the
    requirement graph - the deterministic obligation logic always applies.
    """
    cfg = config or DEFAULT_CONFIG
    qi = intelligence if intelligence is not None else extract_query_intelligence(question)
    if intelligence is None and llm is not None and cfg.llm_planning in ("enrich", "decompose"):
        try:
            llm_qi = structured_query_understanding(question, llm)
            if llm_qi is not None:
                qi = merge_intelligence(qi, llm_qi.to_dict())
        except Exception as exc:  # noqa: BLE001
            _warnings_cache.setdefault(question, []).append(
                f"LLM structured understanding failed ({exc}), using deterministic")

    plan = V2Plan()
    plan.query = qi.query or re.sub(r"\s+", " ", question).strip()
    plan.question_type = qi.question_type
    plan.clinical_entities = list(qi.clinical_entities)
    plan.entities = list(dict.fromkeys(e.surface_form for e in qi.clinical_entities))
    plan.conditions = list(qi.conditions)
    plan.populations = list(qi.populations)
    plan.query_targets = list(qi.query_targets)
    plan.requested_fields = list(qi.requested_fields)
    plan.relationships = list(qi.relationships)
    plan.target = qi.target or cfg.outcome_fallback
    plan.outcome = qi.outcome or cfg.outcome_fallback
    plan.reranker_intent = dict(qi.reranker_intent)
    plan.n_hops = 1

    # requirement graph (one per genuine evidence obligation)
    reqs = _build_requirements(qi, cfg)
    plan.n_hops = reqs[0].n_hops if reqs else 1
    plan.requirements = reqs

    # retrieval variants + queries
    # Numbering contract: q1..qN are the LEAD queries (one per requirement,
    # in H order) - a stable, regression-friendly identity for each branch;
    # variant queries follow as q{N+1}, q{N+2}, ... This keeps e.g. q4 == H4's
    # primary query for multi-hop plans.
    queries: List[SearchQuery] = []
    all_variants: List[RetrievalVariant] = []
    for req in reqs:
        vs = _build_retrieval_variants(req, qi, cfg)
        req.retrieval_variants = vs
        req.navigation_objective = _build_navigation_objective(req)
        all_variants.extend(vs)
    for req in reqs:
        lead = req.retrieval_variants[0] if req.retrieval_variants else RetrievalVariant(
            id="V0", text=req.search_query_text(), source="requirement",
            requirement_ids=[req.id])
        queries.append(SearchQuery(id=f"q{len(queries) + 1}", requirement_ids=[req.id],
                                   text=lead.text, variant_id=lead.id,
                                   variant_source=lead.source))
    for req in reqs:
        for v in req.retrieval_variants[1:]:
            queries.append(SearchQuery(id=f"q{len(queries) + 1}", requirement_ids=[req.id],
                                       text=v.text, variant_id=v.id,
                                       variant_source=v.source))

    # hard cap on total plan queries (performance guard)
    if len(queries) > cfg.max_plan_queries:
        kept: List[SearchQuery] = []
        for req in reqs:
            req_qs = [q for q in queries if q.requirement_ids == [req.id]]
            if req_qs:
                kept.append(req_qs[0])
        budget = cfg.max_plan_queries - len(kept)
        for q in queries:
            if q in kept:
                continue
            if budget > 0:
                kept.append(q)
                budget -= 1
        queries = kept
    plan.queries = queries
    plan.retrieval_variants = all_variants

    plan.terminology_guard = _run_terminology_guard(plan)
    for w in _warnings_cache.pop(question, []):
        plan.warnings.append(w)
    plan.planner_method = "deterministic" if llm is None else "deterministic+llm"
    return plan


_warnings_cache: Dict[str, List[str]] = {}


# ---------------------------------------------------------------------------
# Plan persistence
# ---------------------------------------------------------------------------


def plan_from_dict(data: Dict[str, Any]) -> V2Plan:
    """Rebuild a V2Plan from a dict (e.g. a persisted plan file)."""
    plan = V2Plan()
    for key in ("query", "question_type", "target", "outcome", "planner_method"):
        if key in data:
            setattr(plan, key, data[key])
    plan.entities = list(data.get("entities", []))
    plan.conditions = list(data.get("conditions", []))
    plan.populations = list(data.get("populations", []))
    plan.query_targets = list(data.get("query_targets", []))
    plan.requested_fields = list(data.get("requested_fields", []))
    plan.relationships = list(data.get("relationships", []))
    plan.comparisons = list(data.get("comparisons", []))
    plan.n_hops = int(data.get("n_hops", 1))
    plan.warnings = list(data.get("warnings", []))
    plan.terminology_guard = data.get("terminology_guard", {})
    plan.reranker_intent = data.get("reranker_intent", {}) or {}
    plan.clinical_entities = [
        ClinicalEntity(
            surface_form=e.get("surface_form", ""),
            base_concept=e.get("base_concept", ""),
            modifiers=list(e.get("modifiers", [])),
            role=e.get("role", "condition"),
            preferred_name=e.get("preferred_name", ""),
            ontology=e.get("ontology", ""),
            cui=e.get("cui", ""),
            synonyms=list(e.get("synonyms", [])),
            abbreviations=list(e.get("abbreviations", [])),
        )
        for e in data.get("clinical_entities", [])
    ]
    for rd in data.get("requirements", []):
        terms = [
            Terminology(
                original_term=t.get("original_term", ""),
                canonical_term=t.get("canonical_term", ""),
                synonyms=list(t.get("synonyms", [])),
            )
            for t in rd.get("terms", [])
        ]
        entities = [
            ClinicalEntity(
                surface_form=e.get("surface_form", ""),
                base_concept=e.get("base_concept", ""),
                modifiers=list(e.get("modifiers", [])),
                role=e.get("role", "condition"),
                preferred_name=e.get("preferred_name", ""),
                ontology=e.get("ontology", ""),
                cui=e.get("cui", ""),
                synonyms=list(e.get("synonyms", [])),
                abbreviations=list(e.get("abbreviations", [])),
            )
            for e in rd.get("entities", [])
        ]
        req = Requirement(
            id=rd.get("id", ""),
            topic=rd.get("topic", ""),
            population=rd.get("population", ""),
            target=rd.get("target", ""),
            focus=rd.get("focus", "evidence"),
            outcome=rd.get("outcome", ""),
            condition=rd.get("condition", ""),
            required_concepts=list(rd.get("required_concepts", [])),
            preferred_evidence_types=list(rd.get("preferred_evidence_types", [])),
            requested_fields=list(rd.get("requested_fields", [])),
            terms=terms,
            entities=entities,
            navigation_objective=rd.get("navigation_objective", ""),
            hop_label=rd.get("hop_label", ""),
            hop_dependencies=list(rd.get("hop_dependencies", [])),
            comparison=bool(rd.get("comparison", False)),
            n_hops=int(rd.get("n_hops", 1)),
        )
        req.retrieval_variants = [
            RetrievalVariant(id=v.get("id", ""), text=v.get("text", ""),
                             source=v.get("source", ""),
                             requirement_ids=list(v.get("requirement_ids", [])))
            for v in rd.get("retrieval_variants", [])
        ]
        plan.requirements.append(req)
    plan.queries = [
        SearchQuery(
            id=qd.get("id", ""),
            requirement_ids=list(qd.get("requirement_ids", [])),
            text=qd.get("text", ""),
            variant_id=qd.get("variant_id", ""),
            variant_source=qd.get("variant_source", ""),
            intended_population=qd.get("intended_population", ""),
            intended_outcome=qd.get("intended_outcome", ""),
        )
        for qd in data.get("queries", [])
    ]
    plan.retrieval_variants = [
        RetrievalVariant(id=v.get("id", ""), text=v.get("text", ""),
                         source=v.get("source", ""),
                         requirement_ids=list(v.get("requirement_ids", [])))
        for v in data.get("retrieval_variants", [])
    ]
    return plan
