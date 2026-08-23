"""Intent-aware scoring with explicit bonuses and penalties (V2.1 parts 17-19, 33-34).

The intent layer scores REQUIREMENT-TEXT-AND-STRUCTURE vs the CANDIDATE's
CONTEXTUAL evidence text (table context for rows, passage for paragraphs). The
requirement_match is the weighted sum of contextual components:

    concept_match            (condition surface / base / synonyms / acronyms)
    clinical_condition_match (base concept + modifiers)
    outcome_match
    requested_field_match    (table-aware: header semantics decide whether
                              "0.04" is a p-value - part 19)
    evidence_type_match      (evidence-type priors; "results" = Results section)
    population_match

The old literal-target-phrase requirement is GONE: a table row
"End-to-end | 2 (8) | 2 (50) | 0.04" is evidence for "surgical repair
techniques" because the table header establishes the variable as a repair
technique - the scorer must never demand the phrase appear in the row text.

Penalties are contextual (part 34): generic topic prose (condition match yes,
fields no, target no) gets generic_topic + requested_field + target penalties,
while a structurally-correct table row gets close to none.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Sequence

from medrag.retrieval_v2.config import V2Config, DEFAULT_CONFIG
from medrag.retrieval_v2.models import EvidenceCandidate, Requirement
from medrag.retrieval_v2.table_context import requested_field_fraction


STOPWORDS = {
    "the", "a", "an", "and", "or", "of", "in", "on", "with", "without", "to",
    "for", "was", "were", "is", "are", "be", "been", "as", "by", "at", "from",
    "that", "this", "these", "those", "between", "among", "across", "vs", "versus",
    "patients", "patient", "group", "groups", "cohort", "cohorts", "we", "our",
}


# Competing (mutually exclusive) population / concept pairs - never both.
COMPETING_TERMS: List[tuple[str, str]] = [
    ("inoca", "minoca"),
    ("minoca", "inoca"),
    ("non-diabetic", "diabetic"),
    ("diabetic", "non-diabetic"),
]


def _tokens(text: str) -> set:
    return set(re.findall(r"[a-z0-9]+", (text or "").lower())) - STOPWORDS


def _phrase_in(text: str, phrase: str) -> bool:
    if not phrase or not text:
        return False
    return phrase.lower() in text.lower()


def _fraction_overlap(terms: Sequence[str], text: str) -> float:
    """Fraction of multi-word terms present in the text (word-level)."""
    if not terms:
        return 0.0
    tl = text.lower()
    hits = 0
    for t in terms:
        t = (t or "").strip()
        if not t:
            continue
        if t.lower() in tl:
            hits += 1.0
        else:
            words = _tokens(t)
            if words and words.issubset(_tokens(text)):
                hits += 0.5
    total = max(1, sum(1 for t in terms if t and t.strip()))
    return min(1.0, hits / total)


def _condition_text(requirement: Requirement) -> List[str]:
    """All surface/canonical/synonym/acronym forms of the requirement's condition
    (generic terminology alignment)."""
    out: List[str] = []
    for term in requirement.terms:
        for x in (term.original_term, term.canonical_term, *term.synonyms):
            if x and x not in out:
                out.append(x)
    for ent in requirement.entities:
        for x in (ent.surface_form, ent.base_concept, *ent.synonyms, *ent.abbreviations):
            if x and x not in out:
                out.append(x)
    if requirement.condition and requirement.condition not in out:
        out.append(requirement.condition)
    if requirement.topic and requirement.topic not in out:
        out.append(requirement.topic)
    return out


def _competing_population_hit(req_population: str, text: str) -> bool:
    """True when the text states a competing population for this requirement."""
    if not req_population or not text:
        return False
    rp = req_population.lower()
    tl = text.lower()
    for a, b in COMPETING_TERMS:
        if a in rp and b in tl and a not in tl.replace(b, "") + b:
            return True
    return False


def _confusable_terms(req: Requirement, text: str) -> bool:
    """Explicitly distinguishable concepts (INOCA vs MINOCA etc.)."""
    tl = text.lower()
    for term in req.terms:
        canon = (term.canonical_term or term.original_term or "").lower()
        for a, b in COMPETING_TERMS:
            if canon == a and b in tl and a not in tl:
                return True
    for ent in req.entities:
        name = (ent.surface_form or ent.base_concept or "").lower()
        for a, b in COMPETING_TERMS:
            if name == a and b in tl and a not in tl:
                return True
    if req.population:
        for a, b in COMPETING_TERMS:
            if a in req.population.lower() and b in tl and a not in tl:
                return True
    return False


def _generic_topic_hit(req: Requirement, text: str) -> bool:
    """Paragraph-level: mentions the broad topic but carries no target / fields
    / population / outcome. Tables are never 'generic' (they carry structure)."""
    tl = text.lower()
    condition_forms = _condition_text(req)
    topic_hit = any(f and f.lower() in tl for f in condition_forms) or bool(
        req.topic and req.topic.lower() in tl)
    if not topic_hit:
        return False
    specific = False
    if req.population and req.population.lower() in tl:
        specific = True
    if req.target and req.target.lower() in tl:
        specific = True
    if req.outcome and req.outcome.lower() in tl:
        specific = True
    if req.requested_fields:
        specific = True
    return not specific


def _evidence_type_match(req: Requirement, c: EvidenceCandidate, boosts: Dict[str, float]) -> float:
    """Evidence-type prior + preferred-evidence bonus; 'results' maps to the
    Results section (there is no 'results' node type in the corpus)."""
    bonus = boosts.get(c.node_type, 0.0)
    pref = req.preferred_evidence_types or []
    if c.node_type in pref:
        bonus += 0.10
    elif "results" in pref and c.node_type == "paragraph" and (c.section or "").lower() == "results":
        bonus += 0.10
    return min(1.0, bonus)


def score_intent(
    candidates_by_requirement: Dict[str, List[EvidenceCandidate]],
    requirements: Sequence[Requirement],
    config: Optional[V2Config] = None,
    paper_relevance_map: Optional[Dict[str, float]] = None,
    trace: Any = None,
) -> None:
    """Compute intent scores, penalties and final_score for all candidates.

    Mutates candidates in place.
    """
    cfg = config or DEFAULT_CONFIG
    w = cfg.final_weights
    iw = cfg.intent_weights
    for req in requirements:
        for c in candidates_by_requirement.get(req.id, []):
            c.final_score = -1e9
    for req in requirements:
        pool = candidates_by_requirement.get(req.id, [])
        if not pool:
            continue
        boosts = cfg.evidence_type_boosts.get(req.focus, {})
        cond_forms = _condition_text(req)
        for c in pool:
            text = c.context_text or c.text or ""
            tl = text.lower()

            # ── contextual components ─────────────────────────────
            # concept_match: condition surface/base/synonym/acronym appears
            concept_match = _fraction_overlap(cond_forms, text)
            # clinical_condition_match: base concept + modifiers present
            base_forms: List[str] = []
            for ent in req.entities:
                if ent.role == "condition":
                    for x in (ent.base_concept, ent.surface_form, *ent.abbreviations):
                        if x and x not in base_forms:
                            base_forms.append(x)
            for t in req.terms:
                if t.canonical_term and t.canonical_term not in base_forms:
                    base_forms.append(t.canonical_term)
            if req.condition and req.condition not in base_forms:
                base_forms.append(req.condition)
            clinical_condition_match = _fraction_overlap(base_forms, text)
            if not base_forms:
                clinical_condition_match = concept_match

            population_match = 1.0
            if req.population:
                population_match = _fraction_overlap([req.population], text)
                population_match = max(population_match, 1.0 if _phrase_in(text, req.population) else 0.0)

            outcome_match = 1.0
            if req.outcome:
                outcome_match = 1.0 if (_phrase_in(text, req.outcome) or "outcome" in tl) else 0.0

            # requested fields: table-aware detection (part 19) else regex
            if c.detected_fields:
                field_frac = requested_field_fraction(c.detected_fields, req.requested_fields)
            else:
                from medrag.retrieval_v2.table_context import detect_paragraph_fields
                field_frac = requested_field_fraction(detect_paragraph_fields(text), req.requested_fields)
            if c.is_table():
                field_frac = min(1.0, field_frac * cfg.table_requested_field_multiplier)
            requested_field_match = field_frac

            evidence_type_match = _evidence_type_match(req, c, boosts)

            # target: for TABLES the header/variable establishes the target -
            # the phrase never needs to appear in the row text (part 18), but a
            # table whose variable/headers match the target terms (e.g. "Type of
            # repair (%)") is rewarded over unrelated baseline tables. For
            # paragraphs use target overlap.
            if c.is_table():
                tc = c.table_context or {}
                tbl_parts = " ".join([
                    str(tc.get("variable", "") or ""),
                    " ".join(str(h) for h in (tc.get("headers") or [])),
                    str(tc.get("caption", "") or "")[:200],
                ])
                target_match = _fraction_overlap([req.target], tbl_parts) if req.target else 1.0
            else:
                target_match = _fraction_overlap([req.target], text) if req.target else 1.0

            components = {
                "concept_match": concept_match,
                "clinical_condition_match": clinical_condition_match,
                "target_match": target_match,
                "outcome_match": outcome_match,
                "requested_field_match": requested_field_match,
                "evidence_type_match": evidence_type_match,
                "population_match": population_match,
            }
            weight_sum = sum(iw.get(k, 0.0) for k in components) or 1.0
            requirement_match = sum(
                iw.get(k, 0.0) * v for k, v in components.items()
            ) / weight_sum
            requirement_match = min(1.0, max(0.0, requirement_match))

            # ── penalties (contextual, part 34) ───────────────────
            penalties: Dict[str, float] = {}
            if not c.is_table() and req.target and target_match < 0.10:
                target_words = _tokens(req.target)
                if target_words and not (target_words & _tokens(text)):
                    penalties["target_mismatch"] = round(cfg.penalty_target_mismatch * 0.20, 4)
            if req.population and _competing_population_hit(req.population, text):
                penalties["population_mismatch"] = round(cfg.penalty_population_mismatch * 0.35, 4)
            elif req.population and population_match < 0.25 and len(_tokens(req.population)) > 1:
                penalties["population_mismatch"] = round(cfg.penalty_population_mismatch * 0.10, 4)
            if req.outcome and outcome_match == 0:
                penalties["outcome_mismatch"] = round(cfg.penalty_outcome_mismatch * 0.15, 4)
            if req.requested_fields and requested_field_match < 0.34:
                penalties["requested_field_mismatch"] = round(cfg.penalty_requested_field * 0.20, 4)
            if _confusable_terms(req, text):
                penalties["wrong_concept"] = round(cfg.penalty_wrong_concept * 0.35, 4)
            if not c.is_table() and _generic_topic_hit(req, text):
                penalties["generic_topic"] = round(cfg.penalty_generic_topic * 0.25, 4)
            penalty_total = min(1.0, sum(penalties.values()) * cfg.penalty_scale)

            paper_rel = (paper_relevance_map or {}).get(c.paper_id, 0.0)
            paper_rel = max(0.0, min(1.0, paper_rel))

            final_score = (
                w["medcpt"] * c.medcpt_norm.get(req.id, 0.0)
                + w["requirement_match"] * requirement_match
                + w["evidence_type"] * evidence_type_match
                + w["requested_field"] * requested_field_match
                + w["paper_relevance"] * paper_rel
                + w["diversity"] * 1.0
                - penalty_total
            )
            # normalize to a positive scale so greedy selection behaves
            final_score = max(0.0, final_score)

            c.slot_matches = {
                "concept": round(concept_match, 4),
                "clinical_condition": round(clinical_condition_match, 4),
                "target": round(target_match, 4),
                "outcome": round(outcome_match, 4),
                "requested_fields": round(requested_field_match, 4),
                "evidence_type": round(evidence_type_match, 4),
                "population": round(population_match, 4),
            }
            c.requirement_match = requirement_match
            c.evidence_type_bonus = evidence_type_match
            c.requested_field_bonus = requested_field_match
            c.paper_relevance = paper_rel
            c.penalties = {k: round(v, 4) for k, v in penalties.items()}
            c.penalty_total = penalty_total
            if final_score > c.final_score:
                c.final_score = final_score
            # coverage (V2.1 part 20): "does this evidence satisfy the WHOLE
            # requirement?" - requiring both the contextual match AND the
            # concept (condition) being present. This blocks the anti-pattern
            # where an unrelated statistical paragraph (HR + survival) covers a
            # numerical question whose concept is absent from the evidence.
            med_score = c.medcpt_norm.get(req.id, 0.0)
            if med_score == 0.0 and c.medcpt_scores.get(req.id, 0.0) == 0.0:
                med_score = c.local_scores.get("rrf", 0.0)  # rerank-disabled fallback
            concept_present = concept_match >= 0.10
            if (
                (med_score >= cfg.medcpt_coverage_threshold or med_score == 0.0)
                and requirement_match >= cfg.requirement_match_threshold
                and concept_present
            ) and req.id not in c.covered_requirements:
                c.covered_requirements.append(req.id)

    if trace is not None:
        covered_report: Dict[str, int] = {}
        for req in requirements:
            pool = candidates_by_requirement.get(req.id, [])
            covered_report[req.id] = sum(1 for c in pool if c.covered_requirements)
        trace.log(
            "intent_scoring",
            params={"n_requirements": len(requirements)},
            result={"covered_per_requirement": covered_report},
        )
