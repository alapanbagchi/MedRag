"""Step 1 — Robust query decomposition.

The LLM turns a compound clinical question into MULTIPLE subqueries, each
carrying its own intent and evidence obligations, plus the medical entities
relevant to that subquery. A deterministic post-processing layer filters
junk (template echoes, bare acronyms) and guarantees at least one subquery.

This is intentionally self-contained (no dependency on the legacy
src.agents.planner) so it can be tested in isolation before the rest of the
agentic pipeline is wired up.
"""

from __future__ import annotations
from src.prompts.load import load_prompt

import logging
import re
from typing import Any, List, Optional

from pydantic import BaseModel, Field

logger = logging.getLogger("src.agentic.planner")

# Junk small/medium models echo into free-text fields (observed verbatim:
# an entity whose text was literally "terminology:true,text:").
_JUNK_RE = re.compile(r"[,:{}[\]()]")
_JUNK_WORDS = {
    "true", "false", "null", "none", "text", "role", "terminology", "id",
    "focus", "query", "target", "string", "name", "value", "label",
    "condition", "procedure", "disease", "drug", "anatomy", "symptom",
    "population", "finding", "other", "intent", "evidence", "required",
}


class PlannedEntity(BaseModel):
    """A medical entity extracted for a specific subquery."""
    text: str
    role: str = "condition"
    synonyms: List[str] = Field(default_factory=list)


class SubQueryPlan(BaseModel):
    """One evidence obligation derived from the question."""
    id: str
    target: str = ""
    intent: str = ""
    query: str = ""
    focus: str = "evidence"
    evidence_required: List[str] = Field(default_factory=list)
    entities: List[PlannedEntity] = Field(default_factory=list)
    synonyms: List[str] = Field(default_factory=list)  # UMLS-enriched terms
    enriched_query: str = ""  # query after synonym folding (Step 2)


class Decomposition(BaseModel):
    question_type: str = "comprehensive"
    subqueries: List[SubQueryPlan] = Field(default_factory=list)


PLANNER_SYSTEM_PROMPT = load_prompt('agentic_v1', 'planner.txt')


def _clean_text(value: Any) -> str:
    """Normalize and strip junk from a free-text field."""
    if value is None:
        return ""
    text = " ".join(str(value).split()).strip()
    if not text:
        return ""
    lowered = text.lower()
    if lowered in _JUNK_WORDS:
        return ""
    # strip template "field:value" echoes (e.g. "terminology:true,text:")
    if re.search(r"\b(terminology|text|role|focus|query|target)\s*:", lowered):
        return ""
    # strip trailing JSON fragments the model appends (e.g. "...}, {id:", "},",
    # "],") — treat the first dangling ", {" / "}," as the end of real text.
    for marker in (", {", ",{", "},"):
        pos = text.find(marker)
        if pos != -1:
            text = text[:pos]
    text = text.replace("{", "").replace("}", "")
    words = [w for w in re.split(r"\s+", text) if w]
    if words and all((w.lower() in _JUNK_WORDS or not re.search(r"[a-z0-9]", w)) for w in words):
        return ""
    return " ".join(str(text).split()).strip()


def _is_junk_entity(text: str) -> bool:
    """True when an entity looks like template/junk rather than a real concept."""
    t = (text or "").strip()
    if not t:
        return True
    low = t.lower()
    if low in _JUNK_WORDS:
        return True
    # field:value join like "terminology:true" or "text:foo"
    if re.search(r"\b(terminology|text|role|focus|query|target)\s*:", low):
        return True
    if _JUNK_RE.sub("", low).strip() == "":
        return True
    letters = re.sub(r"[^a-z0-9]", "", low)
    if len(letters) < 3:
        return True
    return False


# stopwords / boilerplate tokens to ignore when building a fallback entity
# list from a subquery's text.
_FALLBACK_STOP = {
    "a", "an", "the", "and", "or", "of", "in", "for", "on", "with", "to", "vs",
    "versus", "how", "what", "why", "does", "during", "after", "before", "use",
    "used", "using", "prevent", "prevention", "manage", "management", "strategy",
    "strategies", "compare", "comparison", "explain", "between", "access",
    "high", "risk", "low", "early", "late", "improve", "improving",
    "site", "sites", "stratification", "stratify", "score", "scoring",
    "score", "utilizes", "utilize", "compared", "traditional", "monitoring",
    "monitor",
    "prediction", "predict", "predicting", "predictive", "predicts",
    "serum", "plasma", "concentration", "levels", "level", "measurement",
    "kidney", "injury", "acute", "chronic", "renal",
}


def _fallback_entities(text: str) -> List[PlannedEntity]:
    """Deterministic entity extraction when the model emitted none.

    Split the target/query into candidate concept phrases (scoring the longest
    multi-word runs first) and keep the ones that look biomedical. This is a
    safety net only — normal operation uses the model's own entities.
    """
    import re as _re

    clean = _clean_text(text)
    if not clean:
        return []
    # tokenize on punctuation but keep hyphenated/abbrev tokens intact
    tokens = [t for t in _re.split(r"[^A-Za-z0-9\-]+", clean) if t]
    candidates: List[str] = []
    i = 0
    while i < len(tokens):
        tok = tokens[i].strip("-").lower()
        # an acronym (all-caps short token) is a strong concept candidate
        if len(tokens[i]) <= 5 and tokens[i].isupper() and tokens[i].isalpha():
            candidates.append(tokens[i])
            i += 1
            continue
        if tok in _FALLBACK_STOP or len(tok) < 4:
            i += 1
            continue
        # a lone abstract noun (ends in -tion/-ing/-ment/-ness/-ity) is almost
        # never a real biomedical concept; skip it unless it extends to a phrase.
        if _re.match(r".+(tion|ting|ment|ness|ity)$", tok):
            i += 1
            continue
        # greedily extend into a multi-word concept while words look contentful
        phrase = [tokens[i]]
        j = i + 1
        while j < len(tokens) and len(phrase) < 3:
            nxt = tokens[j].lower()
            if nxt in _FALLBACK_STOP:
                break
            phrase.append(tokens[j])
            j += 1
        candidates.append(" ".join(phrase))
        i = j
    out: List[PlannedEntity] = []
    seen: set = set()
    for c in candidates:
        if _is_junk_entity(c):
            continue
        key = c.casefold()
        if key in seen:
            continue
        seen.add(key)
        out.append(PlannedEntity(text=c, role=_infer_role(c, "")))
        if len(out) >= 6:
            break
    return out


def _infer_role(text: str, given: str) -> str:
    """Use the model-provided role when sane; else a light lexical default.

    Forcing every entity to "condition" is wrong for anatomy/procedures;
    a cheap keyword heuristic is better than a blanket default.
    """
    role = _clean_text(given)
    if role and role.lower() not in _JUNK_WORDS:
        return role
    low = text.lower()
    if any(w in low for w in ("artery", "vein", "vessel", "graft", "kidney", "arterial")):
        return "anatomy"
    if any(w in low for w in ("embolization", "bypass", "harvest", "surgery", "angioplasty", "catheter", "access", "radiology", "grafting")):
        return "procedure"
    if any(w in low for w in ("score", "creatinine", "proenkephalin", "penk", "biomarker", "troponin", "ejection fraction", "glucose")):
        return "biomarker"
    if any(w in low for w in ("verapamil", "nitroglycerin", "inhibitor", "blocker", "agonist", "antagonist")):
        return "drug"
    return "condition"


def _fallback_intent(focus_or_target: str) -> str:
    """A conservative intent phrase when the model omitted the field."""
    low = (focus_or_target or "").lower()
    if "comparison" in low or "compare" in low or " vs " in low or "versus" in low:
        return "compare the alternatives named in the target"
    if "mechanism" in low or "how" in low or "utiliz" in low:
        return "explain the mechanism / how"
    if "risk" in low or "stratif" in low or "predict" in low:
        return "assess risk stratification / prediction"
    if "prevent" in low or "manage" in low or "treatment" in low or "therap" in low:
        return "identify prevention / management strategies"
    if "diagnos" in low:
        return "identify diagnostic approach"
    return "find evidence addressing the target"


def _fallback_evidence(target: str) -> List[str]:
    """A conservative evidence-requirement list when the model omitted it."""
    low = (target or "").lower()
    out: List[str] = []
    if "comparison" in low or "compare" in low or " vs " in low or "versus" in low:
        out.append("direct comparison of the named alternatives")
    if any(w in low for w in ("pharmacolog", "drug", "agent", "medication")):
        out.append("pharmacological strategies / agents")
    if any(w in low for w in ("imaging", "ultrasound", "doppler", "modality")):
        out.append("imaging strategies / modalities")
    if any(w in low for w in ("risk", "stratif", "predict", "score")):
        out.append("risk stratification / scoring evidence")
    if any(w in low for w in ("prevent", "manage", "treatment", "therap")):
        out.append("prevention / management approach")
    if not out:
        out.append("evidence directly addressing the target")
    return out[:3]


def _clean_subquery(sub: SubQueryPlan, idx: int) -> Optional[SubQueryPlan]:
    """In-place cleanup of one subquery; returns None if it is unusable."""
    target = _clean_text(sub.target)
    query = _clean_text(sub.query)
    intent = _clean_text(sub.intent)
    if not target and not query:
        return None
    if not query and target:
        query = target
    entities: List[PlannedEntity] = []
    seen: set = set()
    for e in sub.entities:
        text = _clean_text(e.text)
        if _is_junk_entity(text):
            continue
        key = text.casefold()
        if key in seen:
            continue
        seen.add(key)
        entities.append(PlannedEntity(text=text, role=_infer_role(text, e.role), synonyms=[]))
    if not entities:
        # model omitted entities entirely -> deterministic fallback from text
        entities = _fallback_entities(query or target)
        logger.debug("subquery %s: no model entities; fallback extracted %d", sub.id, len(entities))
    evidence: List[str] = []
    for ev in sub.evidence_required:
        ev = _clean_text(ev)
        if ev and ev.casefold() not in {x.casefold() for x in evidence}:
            evidence.append(ev)
    # --- deterministic fallbacks: Gemma sometimes truncates its JSON after a
    # huge <thought> block, dropping intent/evidence_required. Derive them from
    # the target so downstream verification still has anchors. ---
    if not intent:
        intent = _fallback_intent(target or query)
    if not evidence:
        evidence = _fallback_evidence(target or query)
    return SubQueryPlan(
        id=sub.id or f"H{idx}",
        target=target,
        intent=intent,
        query=query,
        focus=_clean_text(sub.focus) or "evidence",
        evidence_required=evidence,
        entities=entities,
    )


def _canonicalize_ids(subs: List[SubQueryPlan]) -> List[SubQueryPlan]:
    out: List[SubQueryPlan] = []
    for idx, sub in enumerate(subs, start=1):
        out.append(sub.model_copy(update={"id": f"H{idx}"}))
    return out


def _norm_tokens(text: str) -> set:
    return set(re.sub(r"[^a-z0-9]+", " ", (text or "").lower()).split())


def _dedupe_subqueries(subs: List[SubQueryPlan], cap: int = 4) -> List[SubQueryPlan]:
    """Drop near-duplicate subqueries and enforce an upper bound.

    Gemma-class models sometimes echo near-identical subqueries (the same
    target several times). We keep the first occurrence whose target shares a
    large token overlap with an already-kept target, and cap the total count.
    A subquery is "kept" if its target tokens are not a near-subset of a
    previously kept target's tokens.
    """
    kept: List[SubQueryPlan] = []
    for sub in subs:
        if len(kept) >= cap:
            break
        t = _norm_tokens(sub.target)
        if not t:
            if len(kept) < cap:
                kept.append(sub)
            continue
        dup = False
        for prior in kept:
            p = _norm_tokens(prior.target)
            if not p:
                continue
            inter = len(t & p)
            # ~80% overlap of the smaller set => near-duplicate
            if inter >= 0.8 * min(len(t), len(p)):
                dup = True
                break
        if not dup:
            kept.append(sub)
    return kept


class DecomposePlanner:
    """Plans query decomposition via the LLM, with deterministic fallbacks."""

    def __init__(self, config: Any = None, model: Any = None):
        from src.config import AppConfig
        from src.llm import build_model
        from pydantic_ai import Agent

        self.config = config or AppConfig()
        self.model = model or build_model(self.config)
        self.agent = Agent(self.model, system_prompt=PLANNER_SYSTEM_PROMPT, name="decompose_planner")

    async def plan(self, query: str) -> Decomposition:
        from src.llm.run import ask_structured

        decomposition = await ask_structured(
            self.agent, query, Decomposition,
            label="decompose_planner",
            max_tokens=self.config.agent_max_tokens,
            max_attempts=getattr(self.config, "max_llm_retries", 4),
        )

        cleaned: List[SubQueryPlan] = []
        for idx, sub in enumerate(decomposition.subqueries, start=1):
            result = _clean_subquery(sub, idx)
            if result is not None:
                cleaned.append(result)

        if not cleaned:
            logger.warning("decomposition produced no usable subqueries; single-hop fallback")
            cleaned = [self._single_hop(query)]

        cleaned = _dedupe_subqueries(cleaned, cap=self.config.max_subqueries)
        cleaned = _canonicalize_ids(cleaned)
        return Decomposition(
            question_type=(decomposition.question_type or "comprehensive"),
            subqueries=cleaned,
        )

    @staticmethod
    def _single_hop(query: str) -> SubQueryPlan:
        return SubQueryPlan(
            id="H1",
            target=query.strip()[:140],
            intent="answer the question as posed",
            query=query.strip(),
            focus="evidence",
            evidence_required=[],
            entities=[],
        )


if __name__ == "__main__":
    import asyncio
    import json
    import sys

    q = " ".join(sys.argv[1:]) or "How do genetic risk variants for essential hypertension identified in African‑derived admixed populations influence the pathogenesis of medial arterial calcification, and what role could predictive modeling of heterogeneous treatment effects play in personalising mTOR‑inhibitor therapy (e.g., rapamycin) for these high‑risk groups?"
    planner = DecomposePlanner()

    async def _run():
        d = await planner.plan(q)
        print(json.dumps(d.model_dump(), indent=2))

    asyncio.run(_run())
