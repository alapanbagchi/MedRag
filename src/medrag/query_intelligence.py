"""Structured query understanding for MedRAG retrieval V2.

Shared by the deterministic local planner and the Kaggle /expand server
(MedGemma path). Produces a QueryIntelligence object - the NEW /expand
contract (V2.1 part 3):

    query, clinical_entities, query_targets, requested_fields, relationships,
    populations, retrieval_variants, reranker_intent, umls

MedGemma's job (part 4) is STRUCTURED QUERY UNDERSTANDING - extracting
clinical entities (surface_form / base_concept / modifiers / role), targets,
requested fields, relationships, populations, question type and search
concepts. It must NOT freely decompose the question into H1/H2/... - the
requirement graph is built afterwards by the planner, one requirement per
genuine evidence obligation.

There are NO hardcoded diseases here: everything is generic (modifier
lexicons, verb frames, acronym handling). INOCA, MINOCA, CKD, coarctation
and friends all flow through the same machinery and are preserved verbatim.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from medrag.retrieval_v2.models import ClinicalEntity


def _lower(s: str) -> str:
    return (s or "").lower()


def _clean(text: str) -> str:
    text = re.sub(r"\s+", " ", text or "").strip()
    return text.strip(" ?.!,;")


# ---------------------------------------------------------------------------
# Generic modifier lexicon (data, not diseases)
# ---------------------------------------------------------------------------
# Leading modifiers are stripped from a surface form to obtain the base
# concept that gets looked up in UMLS/MeSH/BioPortal; the modifiers are then
# preserved and re-attached for search ('recurrent X' -> X + [recurrent]).
# "chronic" and "metabolic" are NOT stripped: they are usually part of the
# canonical disease name (chronic kidney disease, metabolic syndrome). The
# generic rule is: strip a known modifier ONLY while at least two content
# words remain as the base concept (so "recurrent coarctation" -> coarctation,
# "advanced chronic kidney disease" -> chronic kidney disease).
LEADING_MODIFIERS = (
    "recurrent", "advanced", "severe", "mild", "moderate", "acute",
    "early", "late", "primary", "secondary", "new-onset", "new onset",
    "non-diabetic", "diabetic", "non-obstructive", "nonobstructive",
    "pre-existing", "preexisting", "preoperative", "postoperative",
    "congenital", "acquired", "isolated", "combined", "refractory", "stable",
    "progressive", "high-risk", "high risk", "low-risk", "low risk",
    "long-standing", "left-sided", "right-sided", "symptomatic",
    "asymptomatic", "elderly", "infantile", "neonatal", "pediatric",
    "gestational", "familial", "hereditary", "occult", "concomitant",
    "suspected", "confirmed", "residual", "resistant", "transcatheter",
    "percutaneous", "conservative",
)

TRAILING_NOISE = ("patients", "patient")


def strip_modifiers(phrase: str) -> Tuple[str, List[str]]:
    """Split 'advanced chronic kidney disease' -> ('chronic kidney disease',
    ['advanced']). Also 'recurrent X' -> ('X', ['recurrent']).

    Modifiers are stripped only while at least two words remain as the base
    concept (this keeps 'chronic kidney disease' intact while allowing
    'recurrent coarctation' -> 'coarctation').
    """
    words = (phrase or "").strip().split()
    mods: List[str] = []
    idx = 0
    while idx < len(words):
        if len(words) - idx < 2:
            break   # never leave a single-word base that is only the modifier itself
        candidate = " ".join(words[idx : idx + 2])
        if candidate in LEADING_MODIFIERS:
            mods.append(candidate)
            idx += 2
            continue
        candidate1 = words[idx]
        if candidate1 in LEADING_MODIFIERS:
            mods.append(candidate1)
            idx += 1
            continue
        break
    base = " ".join(words[idx:]) if idx < len(words) else (phrase or "").strip()
    # strip trailing population-ish noise ("patients", "patient")
    base_words = base.split()
    while base_words and _lower(base_words[-1]) in TRAILING_NOISE and len(base_words) > 1:
        base_words = base_words[:-1]
    base = " ".join(base_words)
    return (base or phrase or "").strip(), mods


_ACRONYM_RE = re.compile(r"\b([A-Z][A-Za-z0-9]*(?:[-/][A-Za-z0-9]+)*)\b")


def detect_acronyms(text: str) -> List[str]:
    """Upper-case tokens like INOCA, MINOCA, CKD, re-CoA (generic)."""
    out: List[str] = []
    for m in _ACRONYM_RE.finditer(text or ""):
        tok = m.group(1)
        up = sum(1 for ch in tok if ch.isupper())
        if 2 <= len(tok) <= 12 and up >= 2 and tok not in ("BM25", "RRF", "MRI", "CT", "ICU", "US", "V2"):
            if tok not in out:
                out.append(tok)
    return out


_PERCENT_FIELD = re.compile(r"\bpercent|percentage|proportion|rates?\b")
_PVALUE_FIELD = re.compile(r"\bp-?values?\b|statistically significant|\bconfidence interval\b")
_CI_FIELD = re.compile(r"con?fidence interval|95\s*%")
_EFFECT_FIELD = re.compile(r"\bmean\b|\bmedian\b|\baverage\b|odds ratio|hazard ratio|relative risk|\beffect size\b")
_SAMPLE_FIELD = re.compile(r"sample size|\bn\s*(?:=)?\s*\d+|number of patients|how many")


def detect_requested_fields(question: str) -> List[str]:
    """Generic requested-field detection (percentages, p-values, CIs, ...)."""
    ql = _lower(question)
    fields: List[str] = []
    if _PERCENT_FIELD.search(ql):
        fields.append("percentage")
    if _PVALUE_FIELD.search(ql):
        fields.append("p-value")
    if _CI_FIELD.search(ql):
        fields.append("confidence_interval")
    if _EFFECT_FIELD.search(ql):
        fields.append("effect_estimate")
    if _SAMPLE_FIELD.search(ql):
        fields.append("sample_size")
    return fields


def classify_question(question: str) -> str:
    """Question-type classification (generic regexes, no diseases)."""
    ql = _lower(question)
    if re.search(r"\btable\s*\d+\b|\btab\s*\d+\b", ql) and re.search(r"\b(what|which|how many|what were|values|percent|row)\b", ql):
        return "table_lookup"
    if re.search(r"\bfigure\b|\btrend\b|how did .*(change|evolve|vary|shift)\b", ql):
        return "figure_interpretation"
    has_stats = bool(re.search(r"percentages?|p-values?|proportions?|confidence interval|how many", ql))
    if (re.search(r"\bdefine\b|is defined as\b|\bdefinition\b", ql) or (
        re.search(r"\bwhat is\b|\bwhat are\b", ql) and not has_stats
        and not re.search(r"\bwhich\b|associated with|\bhow\b", ql))) and not has_stats:
        return "definition"
    if re.search(r"\btreat|\btherapy\b|\btreatment\b|\bpharmaco|\bdrug\b|\bdosing\b|\bdosage\b", ql):
        return "treatment"
    if re.search(r"\bdiagnos|\bcriteria\b|\bscreening\b", ql):
        return "diagnosis"
    if re.search(r"\bprognos\b|\bsurvival\b|\bmortality\b|\bprogression\b", ql):
        return "prognosis"
    if re.search(r"how many|what percentage|what proportion|what percent|p-values?|percentages? and p-values?", ql):
        return "numerical"
    if re.search(r"\bcompare\b|differ(s|ed)?\b|\bdifference\b|\bversus\b|\bvs\.?\b", ql):
        return "comparison"
    if re.search(r"\bmechanism\b|\bpathway\b|how does .*(affect|impact|influence|cause|drive|lead to)\b|\bwhy does\b|\brole of\b", ql):
        return "mechanism"
    if re.search(r"\bsynthes|\bevidence\b|what do studies show|\bsummarize\b|\breview\b", ql):
        return "evidence_synthesis"
    if len(re.findall(r"\b(?:and|as well as)\b", ql)) >= 2:
        return "multi_hop"
    return "factual"


@dataclass
class QueryIntelligence:
    """Structured query understanding output (V2.1 /expand contract)."""

    query: str = ""
    clinical_entities: List[ClinicalEntity] = field(default_factory=list)
    query_targets: List[str] = field(default_factory=list)
    requested_fields: List[str] = field(default_factory=list)
    relationships: List[str] = field(default_factory=list)
    populations: List[str] = field(default_factory=list)
    question_type: str = "factual"
    search_concepts: List[str] = field(default_factory=list)
    conditions: List[str] = field(default_factory=list)
    target: str = ""                      # primary target (reranker intent)
    outcome: str = ""
    reranker_intent: Dict[str, Any] = field(default_factory=dict)

    def entity_for(self, role: str = "") -> Optional[ClinicalEntity]:
        if role:
            for e in self.clinical_entities:
                if e.role == role:
                    return e
        return self.clinical_entities[0] if self.clinical_entities else None

    def condition_entity(self) -> Optional[ClinicalEntity]:
        for e in self.clinical_entities:
            if e.role == "condition":
                return e
        return None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "clinical_entities": [e.to_dict() for e in self.clinical_entities],
            "query_targets": list(self.query_targets),
            "requested_fields": list(self.requested_fields),
            "relationships": list(self.relationships),
            "populations": list(self.populations),
            "question_type": self.question_type,
            "search_concepts": list(self.search_concepts),
            "conditions": list(self.conditions),
            "target": self.target,
            "outcome": self.outcome,
            "reranker_intent": dict(self.reranker_intent),
        }


# ---------------------------------------------------------------------------
# Deterministic extraction
# ---------------------------------------------------------------------------

_AFFECT_VERBS = (
    "affect", "impact", "influence", "contribute to", "relate to", "cause",
    "drive", "modulate", "alter", "increase", "decrease", "predispose to",
    "lead to", "are associated with", "is associated with", "associated with",
    "confer", "explain",
)


def _split_ordered_list(phrase: str) -> List[str]:
    """Split a phrase into list items on commas / and at depth 0."""
    out: List[str] = []
    depth = 0
    buf = ""
    for ch in phrase:
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        if ch == "," and depth == 0:
            out.append(buf.strip())
            buf = ""
            continue
        buf += ch
    out.append(buf.strip())
    final: List[str] = []
    for item in out:
        pieces = re.split(r"\s+(?:and|as well as)\s+", item)
        flat: List[str] = []
        for piece in pieces:
            # handle a leading "and" from "X, Y, and Z" comma-splitting
            piece = re.sub(r"^and\s+", "", piece.strip(), flags=re.IGNORECASE)
            flat.append(piece)
        for piece in flat:
            piece = piece.strip().strip(" .?!,;")
            if piece and _lower(piece) not in ("and", "as well as"):
                final.append(piece)
    return final


def _extract_by_verb(question: str) -> Tuple[Optional[str], Optional[str]]:
    """(subject_text, object_text) for "how do X affect Y", "effect of X on Y",
    "which X are associated with Y" (generic verb frames)."""
    # "What mechanism links A to B ..."
    m = re.search(r"what mechanism links\s+(.+?)\s+to\s+(.+?)(?:[,?]|$)", question, re.IGNORECASE)
    if m:
        return m.group(1).strip(), m.group(2).strip()
    # "Which <exposure> predicted/affected <outcome> ..." - the exposure is the
    # condition, the outcome the target (generic frame, no diseases).
    m = re.search(
        r"which\s+(.+?)\s+(?:predicted|predicted by|affected|influenced|modified|explained|was linked to|were linked to)\s+(.+?)(?:[,?]|$)",
        question, re.IGNORECASE)
    if m:
        return m.group(1).strip(), m.group(2).strip()
    for verb in _AFFECT_VERBS:
        m = re.search(rf"how (?:do|does)\s+(.+?)\s+{re.escape(verb)}\s+(.+?)(?:[,?]|$)", question, re.IGNORECASE)
        if m:
            return m.group(1).strip(), m.group(2).strip()
        m = re.search(rf"(?:effect|impact|influence) of\s+(.+?)\s+on\s+(.+?)(?:[,?]|$)", question, re.IGNORECASE)
        if m:
            return m.group(1).strip(), m.group(2).strip()
        m = re.search(rf"role of\s+(.+?)\s+in\s+(.+?)(?:[,?]|$)", question, re.IGNORECASE)
        if m:
            return m.group(1).strip(), m.group(2).strip()
    # "Which <target> (are|were) associated with <condition>..."
    m = re.search(r"which\s+(.+?)\s+(?:are|were|is|has been)\s+associated with\s+(.+?)(?:[,?]|$)", question, re.IGNORECASE)
    if m:
        cond = m.group(2).strip()
        # cut trailing "and what / and how / and whether ..." clauses
        cond = re.split(r"\s+and\s+(?:what|how|whether|why|which|when)\b", cond, flags=re.IGNORECASE)[0]
        return cond.strip(), m.group(1).strip()   # (subject=condition, object=target)
    return None, None


_POP_PATTERN = r"(?:across|among|in|for|between)\s+(.+?)(?:\s*[?]|$)"


def _extract_populations(question: str, conditions: Sequence[str]) -> List[str]:
    ql = _lower(question)
    candidates: List[str] = []
    for m in re.finditer(_POP_PATTERN, question, re.IGNORECASE):
        phrase = m.group(1).strip()
        phrase = re.split(r"\band how\b|\bhow (?:does|do) this\b|\bsuch as\b", phrase)[0]
        for item in _split_ordered_list(phrase):
            item = item.strip(" .?!,;")
            if item and item not in candidates and _lower(item) not in [_lower(c) for c in conditions]:
                candidates.append(item)
    return candidates


def _make_entity(surface: str, role: str, all_text: str) -> ClinicalEntity:
    surface = _clean(surface)
    base, mods = strip_modifiers(surface)
    abbrevs: List[str] = []
    for ac in detect_acronyms(all_text):
        # attach an acronym to an entity whose text expansion contains a word
        # overlapping the entity surface tokens (generic abbreviation binding)
        if any(_lower(w) in _lower(ac) or _lower(ac) in _lower(w) for w in surface.split()):
            if len(surface.split()) >= 2 or _lower(ac) != _lower(surface):
                abbrevs.append(ac)
    return ClinicalEntity(
        surface_form=surface,
        base_concept=base,
        modifiers=mods,
        role=role,
        abbreviations=abbrevs,
    )


def extract_query_intelligence(question: str) -> QueryIntelligence:
    """Deterministic structured query understanding (no diseases hardcoded)."""
    q = _clean(question)
    ql = _lower(q)
    qi = QueryIntelligence(query=q)
    qi.question_type = classify_question(q)

    subject, target = _extract_by_verb(q)
    conditions: List[str] = []
    targets: List[str] = []
    if subject is not None:
        conditions = _split_ordered_list(subject)
        if conditions:
            conditions[-1] = re.split(r"\s+(?:in|among|across|for)\s+", conditions[-1])[0].strip()
            conditions = [c for c in conditions if c]
        if target:
            targets = _split_ordered_list(target)

    # fall back: "risk of Y in X", "outcomes of Y" -> the clinical focus
    # becomes the CONDITION so coverage can anchor on a real concept.
    if target is None:
        m = re.search(r"\b(?:risk of|outcomes? of|prognosis of)\s+(.+?)(?:[,?]|$)", q)
        if m:
            focus_phrase = m.group(1)
            conditions = [focus_phrase]
            targets = []
            target = "clinical outcomes"
            subject = focus_phrase
    if not conditions and not targets:
        # "What is the definition of heart failure?" -> condition = "heart failure"
        m = re.search(r"what (?:is|are)\s+(?:the\s+)?(?:definition of|role of|prevalence of|incidence of|etiology of|pathophysiology of)\s+(.+?)(?:[,?]|$)", q, re.IGNORECASE)
        if m:
            conditions = [m.group(1).strip()]
    if not conditions:
        conditions = [q] if not targets else []

    qi.conditions = conditions
    qi.query_targets = targets
    qi.target = targets[0] if targets else ("cardiovascular outcomes" if "cardiovascular" in ql else "clinical outcomes")

    # populations
    pops = _extract_populations(q, conditions)
    qi.populations = pops

    # outcome detection (generic outcome lexicon)
    outcome = ""
    m = re.search(r"\b(outcomes?|mortality|survival|hospitalization|prognosis|cardiovascular events?|recurrence)\b", ql)
    if m:
        outcome = m.group(1)
    qi.outcome = outcome or ("cardiovascular outcomes" if "cardiovascular" in qi.target.lower() else "")

    # requested fields
    qi.requested_fields = detect_requested_fields(q)

    # relationships
    rel: List[str] = []
    if target:
        for c in conditions:
            rel.append(f"{c} associated with {target}")
    m = re.search(r"\bmechanism links\s+(.+?)\s+to\s+(.+?)(?:[,?]|$)", q, re.IGNORECASE)
    if m:
        rel.append(f"{m.group(1).strip()} mechanism -> {m.group(2).strip()}")
    qi.relationships = rel or ([f"{conditions[0]} -> {qi.target}"] if conditions else [])

    # clinical entities: conditions, populations, targets, plus acronyms
    entities: List[ClinicalEntity] = []
    seen: set = set()
    for c in conditions:
        if _lower(c) in seen:
            continue
        seen.add(_lower(c))
        entities.append(_make_entity(c, "condition", q))
    for p in pops:
        if _lower(p) in seen:
            continue
        seen.add(_lower(p))
        entities.append(_make_entity(p, "population", q))
    for t in targets:
        if _lower(t) in seen:
            continue
        seen.add(_lower(t))
        entities.append(_make_entity(t, "target", q))
    # standalone acronyms (e.g. INOCA appears only inside a phrase) - keep
    # them attached as abbreviations of their containing entity instead.
    qi.clinical_entities = entities
    qi.search_concepts = [e.searchable_form() for e in entities if e.searchable_form()]

    # reranker intent (part 3)
    qi.reranker_intent = {
        "question_type": qi.question_type,
        "focus": _focus_from(qi.question_type, qi.requested_fields),
        "target": qi.target,
        "outcome": qi.outcome,
        "requested_fields": list(qi.requested_fields),
    }
    return qi


def _focus_from(qtype: str, fields: Sequence[str]) -> str:
    if "numerical" in qtype or "comparison" in qtype or "table_lookup" in qtype:
        if any(f in ("percentage", "p-value", "confidence_interval", "effect_estimate") for f in fields) and qtype in ("numerical", "comparison"):
            return "comparative_numerical"
        if qtype == "table_lookup":
            return "table_lookup"
        return qtype
    return qtype or "evidence"


def merge_intelligence(base: QueryIntelligence, llm_data: Optional[Dict[str, Any]]) -> QueryIntelligence:
    """Merge deterministic extraction with LLM (MedGemma) structured output.

    The LLM output only ENRICHES: clinical entities get ontology-preferred
    names and synonyms; extra targets/fields/populations are appended. The
    deterministic surface forms always win (original wording preserved).
    """
    if not llm_data:
        return base
    qi = base
    # entities
    raw_entities = llm_data.get("clinical_entities") or llm_data.get("entities") or []
    if raw_entities:
        seen = {_lower(e.surface_form) for e in qi.clinical_entities}
        for item in raw_entities:
            if isinstance(item, dict) and item.get("surface_form"):
                sf = _clean(str(item["surface_form"]))
                if not sf or _lower(sf) in seen:
                    continue
                role = str(item.get("role", "condition")).lower()
                if role == "condition" and sf.lower() in [_lower(c) for c in qi.conditions]:
                    # replace the deterministic entity with the LLM-enriched one
                    continue
                seen.add(_lower(sf))
                ent = _make_entity(sf, role, qi.query)
                qi.clinical_entities.append(ent)
                if role == "condition" and sf not in qi.conditions:
                    qi.conditions.append(ent.surface_form)
    # targets / fields / populations / concepts
    for key, dest in (("query_targets", "query_targets"), ("requested_fields", "requested_fields"),
                      ("populations", "populations"), ("search_concepts", "search_concepts")):
        vals = llm_data.get(key) or []
        if isinstance(vals, list):
            existing = {_lower(v) for v in getattr(qi, dest)}
            for v in vals:
                v = _clean(str(v))
                if v and _lower(v) not in existing:
                    getattr(qi, dest).append(v)
                    existing.add(_lower(v))
    # reranker intent
    ri = llm_data.get("reranker_intent") or {}
    if isinstance(ri, dict):
        for key in ("question_type", "focus", "target", "outcome"):
            if ri.get(key):
                setattr(qi, key if key != "target" else "target", str(ri[key])) if key != "target" else setattr(qi, "target", str(ri["target"]))
        qi.reranker_intent = dict(qi.reranker_intent)
        qi.reranker_intent.update({k: v for k, v in ri.items() if v})
    if llm_data.get("question_type"):
        qi.question_type = str(llm_data["question_type"])
    return qi


# ---------------------------------------------------------------------------
# Ontology normalization (part 5): base concept first, modifiers preserved
# ---------------------------------------------------------------------------

def normalize_entities(entities: Sequence[ClinicalEntity], enricher: Any = None,
                       max_synonyms: int = 3) -> List[ClinicalEntity]:
    """Look up the BASE CONCEPT of every entity (never the whole phrase).

    'recurrent X' -> lookup 'X' -> preferredName/CUI/synonyms, then re-attach
    modifier 'recurrent' as terminology variants:
        recurrent X, recurrent canonical X, recurrent synonym X
    The ontology NEVER replaces the user's original wording.
    """
    out: List[ClinicalEntity] = []
    for ent in entities:
        e = ClinicalEntity(
            surface_form=ent.surface_form,
            base_concept=ent.base_concept,
            modifiers=list(ent.modifiers),
            role=ent.role,
            abbreviations=list(ent.abbreviations),
        )
        if enricher is not None and e.base_concept:
            hits = enricher.search(e.base_concept, max_results=4)
            if hits:
                hit = hits[0]
                e.preferred_name = str(hit.get("pref_label") or e.base_concept)
                e.ontology = str(hit.get("ontology") or "")
                e.cui = str(hit.get("concept_id") or "")
                syns: List[str] = []
                for h in hits:
                    for s in (h.get("synonyms") or []):
                        s = str(s)
                        if s and _lower(s) != _lower(e.base_concept) and len(syns) < max_synonyms:
                            syns.append(s)
                # keep the strongest (shortest, case-preserving) synonyms
                syns.sort(key=lambda s: (len(s), s.lower()))
                e.synonyms = syns[:max_synonyms]
        out.append(e)
    return out


# ---------------------------------------------------------------------------
# MedGemma structured query understanding (server side + optional local)
# ---------------------------------------------------------------------------

MEDGEMMA_UNDERSTANDING_PROMPT = """You are a biomedical retrieval planner. Perform STRUCTURED QUERY UNDERSTANDING
on the user's question. Do NOT decompose the question into multiple evidence
requirements and do NOT generate search queries.

Return ONLY valid JSON with EXACTLY these keys:
{
  "clinical_entities": [
    {"surface_form": "...", "base_concept": "...", "modifiers": [...], "role": "condition|population|target|treatment|outcome|other"}
  ],
  "query_targets": [...],
  "requested_fields": ["percentage", "p-value", "confidence_interval", "effect_estimate", "sample_size"],
  "relationships": [...],
  "populations": [...],
  "reranker_intent": {"question_type": "...", "focus": "...", "target": "...", "outcome": "...", "evidence_types": [...]},
  "search_concepts": [...]
}

RULES:
- Preserve every original medical term VERBATIM (INOCA stays INOCA, never MINOCA).
- surface_form is the exact phrase as written by the user.
- base_concept is the surface form with generic clinical modifiers stripped
  ("recurrent coarctation" -> "coarctation"; "advanced chronic kidney disease"
  -> "chronic kidney disease"). modifiers lists the stripped modifiers.
- requested_fields may be empty when the question does not ask for statistics.
- The question may need only ONE evidence obligation; do not invent extra hops.
"""


def structured_query_understanding(question: str, llm: Any) -> Optional[QueryIntelligence]:
    """MedGemma structured query understanding with deterministic fallback.

    Returns None when the LLM is unavailable or its output is unusable.
    """
    base = extract_query_intelligence(question)
    if llm is None or not hasattr(llm, "is_available") or not llm.is_available():
        return None
    try:
        data = llm.generate_json(
            MEDGEMMA_UNDERSTANDING_PROMPT + "\n\nQUESTION:\n" + question,
            max_tokens=2500,
        )
    except Exception:  # noqa: BLE001
        return None
    if not isinstance(data, dict):
        return None
    return merge_intelligence(base, data)


def intelligence_from_expand_payload(payload: Dict[str, Any]) -> QueryIntelligence:
    """Build QueryIntelligence from a /expand response (NEW contract).

    Accepts the new structured object and remains tolerant of the legacy
    'concepts'/'intent' fields so the server can roll out gradually. It NEVER
    looks at the legacy 'evidence_requirements' list - that contract is dead.
    """
    base = extract_query_intelligence(str(payload.get("query", "")))
    qi = base

    # clinical entities (new contract) -> merge; else legacy concepts
    entities = payload.get("clinical_entities") or []
    if not entities and payload.get("concepts"):
        entities = [
            {
                "surface_form": c.get("name", c) if isinstance(c, dict) else str(c),
                "role": "condition",
            }
            for c in payload["concepts"]
        ]
    extra: Dict[str, Any] = {"clinical_entities": entities}
    for key in ("query_targets", "requested_fields", "relationships", "populations", "search_concepts"):
        if payload.get(key) is not None:
            extra[key] = payload[key]
    # legacy intent fields map onto reranker_intent
    legacy_intent = payload.get("intent") or {}
    ri: Dict[str, Any] = dict(payload.get("reranker_intent") or {})
    if not ri and isinstance(legacy_intent, dict):
        for k in ("question_type", "focus", "target", "outcome"):
            if legacy_intent.get(k):
                ri[k] = legacy_intent[k]
        if legacy_intent.get("conditions"):
            # keep the deterministic conditions when the server only provides intent
            pass
    if ri:
        extra["reranker_intent"] = ri
    qi = merge_intelligence(base, extra)
    qi.query = _clean(str(payload.get("query") or qi.query))
    return qi
