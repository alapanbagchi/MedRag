"""Agentic v3 - Stage 5: search-term selection (per Worker, per round).

The terminology pool (Stage 4) is the vocabulary; this stage selects the
ACTUAL search formulations for each evidence requirement (spec section 7):

    instead of                      the worker may generate
    "vitamin D hypertension"        "vitamin D" AND hypertension
                                    25-hydroxyvitamin D AND blood pressure
                                    cholecalciferol AND hypertension
                                    "vitamin D supplementation" AND blood pressure

The planner is progressive: on later rounds it receives the queries already
tried, how many papers each returned, and the CRITIC's rejection reasons, and
it REFORMULATES (different term, different pairing, different outcome phrase)
until evidence is sufficient or the budget is exhausted (spec stages 12/17).

Deterministic fallback keeps the worker moving even when the LLM fails:
quoted AND-pairings of the terminology pool against the requirement's own
outcome terms, rotated across rounds so successive rounds stay distinct.
"""

from __future__ import annotations
from src.prompts.load import load_prompt

import logging
import re
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from src.agentic_v3.state import ResearchTask
from src.agentic_v3.umls import TerminologyEnricher

logger = logging.getLogger("src.agentic_v3.search")


class RequirementSearch(BaseModel):
    """One evidence requirement's search formulations for this round."""
    requirement_id: str = ""
    rationale: str = ""
    queries: List[str] = Field(default_factory=list)


class TaskSearchPlan(BaseModel):
    """The worker's search plan for one round (all requirements covered)."""
    round_no: int = 1
    rationale: str = ""
    requirements: List[RequirementSearch] = Field(default_factory=list)

    def queries_for(self, requirement_id: str) -> List[str]:
        for r in self.requirements:
            if r.requirement_id == requirement_id:
                return list(r.queries)
        return []


SEARCH_PLANNER_SYSTEM_PROMPT = load_prompt('agentic_v3', 'search_planner.txt')


class RequirementPlan(BaseModel):
    """Raw LLM output for ONE requirement."""
    rationale: str = ""
    queries: List[str] = Field(default_factory=list)


def _clean_query(q: Any) -> str:
    q = " ".join(str(q or "").split())
    if not q or len(q) < 7:
        return ""
    # drop trailing junk the model sometimes appends (stray commas/braces)
    for marker in (", {", "}, "):
        pos = q.find(marker)
        if pos != -1:
            q = q[:pos]
    q = q.replace("{", "").replace("}", "").strip()
    return q[:240]


def _requirement_outcome_terms(requirement_text: str) -> List[str]:
    """Outcome-ish phrases to pair with concept terms in fallback queries.

    Uses the requirement's own words (the specific relationship sought)
    rather than generic terms like 'effect'.
    """
    low = str(requirement_text or "").lower()
    phrases = []
    # strip leading gerunds/shorteners ("association between X and Y" ->
    # "X and Y" is still fine to keep)
    for tok in re.findall(r"[a-zA-Z][a-zA-Z0-9+/ ()-]{2,}", requirement_text or ""):
        t = " ".join(tok.split())
        if len(t.split()) >= 2 and t.casefold() not in {p.casefold() for p in phrases}:
            phrases.append(t)
    return phrases[:3]


def fallback_queries(
    task: ResearchTask,
    requirement_text: str,
    round_no: int,
    tried_queries: List[str],
) -> List[str]:
    """Deterministic progressive formulations: quoted AND-pairs from the pool.

    Round rotation picks a different pairing offset each round so successive
    rounds never repeat an already-tried query.
    """
    terms = TerminologyEnricher.pool_terms(task)
    if not terms:
        terms = [t for t in (task.entities or []) if t]
    outcome_terms = _requirement_outcome_terms(requirement_text)
    tried = {q.casefold() for q in tried_queries}
    candidates: List[str] = []
    if terms:
        for i in range(len(terms)):
            for j in range(len(outcome_terms)):
                term = terms[i]
                out = outcome_terms[j]
                quoted = f'"{term}"' if len(term.split()) > 1 else term
                out_quoted = f'"{out}"' if len(out.split()) > 1 else out
                q = f"{quoted} AND {out_quoted}"
                candidates.append(q)
    if not candidates:
        candidates = [requirement_text.strip()[:200]]
    # rotate by round so each round prefers unseen combinations
    start = (round_no - 1) % max(1, len(candidates))
    rotated = candidates[start:] + candidates[:start]
    out: List[str] = []
    for q in rotated:
        q = _clean_query(q)
        if q and q.casefold() not in tried:
            tried.add(q.casefold())
            out.append(q)
        if len(out) >= 3:
            break
    if not out:
        out = [requirement_text.strip()[:200]]
    return out[:3]


def _attempts_block(attempts: List[Dict[str, Any]]) -> str:
    if not attempts:
        return "(no previous searches for this requirement)"
    lines = []
    for a in attempts:
        lines.append(
            f"- query: {a.get('query', '')} | papers: {a.get('papers', 0)} | "
            f"rejected-notes: {(a.get('notes') or '')[:300]}"
        )
    return "\n".join(lines)


class SearchTermPlanner:
    """Selects the most probable medical search terms per requirement/round."""

    def __init__(self, config: Any = None, model: Any = None):
        from src.config import AppConfig
        from src.llm import build_model_for
        from pydantic_ai import Agent

        self.config = config or AppConfig()
        self.model = model or build_model_for(self.config, role="planner")
        self.agent = Agent(
            self.model,
            system_prompt=SEARCH_PLANNER_SYSTEM_PROMPT,
            name="search_term_planner",
        )

    async def plan(
        self,
        task: ResearchTask,
        round_no: int,
        attempts: Optional[Dict[str, List[Dict[str, Any]]]] = None,
    ) -> TaskSearchPlan:
        """One search plan covering every unsatisfied requirement."""
        from src.llm.run import ask_structured
        from src.trace import get_trace

        trace = get_trace()
        attempts = attempts or {}
        trace.agent("search_term_planner", output_type="TaskSearchPlan",
                    meta={"task": task.id, "round": round_no,
                          "requirements": len(task.evidence_requirements)})

        plan = TaskSearchPlan(round_no=round_no, rationale="")
        for req in task.evidence_requirements:
            if req.satisfied():
                plan.requirements.append(RequirementSearch(
                    requirement_id=req.id, rationale="already satisfied",
                    queries=[]))
                continue
            prior = attempts.get(req.id, [])
            plan.requirements.append(RequirementSearch(
                requirement_id=req.id,
                rationale="fallback",
                queries=self._deterministic(task, req.text, round_no, prior),
            ))

        # LLM refinement (offline-safe): only replace the fallback when the
        # model returns clean, non-repeated queries.
        prompt = self._prompt(task, plan, round_no, attempts)
        try:
            raw = await ask_structured(
                self.agent,
                prompt,
                RequirementPlan,
                label=f"search_term_planner:{task.id}:r{round_no}",
                max_tokens=min(900, getattr(self.config, "agent_max_tokens", 2048)),
            )
            queries = [q for q in (_clean_query(q) for q in (raw.queries or [])) if q]
            if queries:
                merged = plan.requirements[0].model_copy(
                    update={"queries": queries, "rationale": raw.rationale or "llm"})
                plan.requirements[0] = merged
        except Exception as exc:
            logger.warning("search planning failed for %s r%d (%s); fallback kept",
                           task.id, round_no, exc)
        plan.requirements = [r for r in plan.requirements if r.queries or r.requirement_id]
        return plan

    def _deterministic(self, task: ResearchTask, req_text: str, round_no: int,
                       prior: List[Dict[str, Any]]) -> List[str]:
        tried = [a.get("query", "") for a in prior]
        return fallback_queries(task, req_text, round_no, tried)

    def _prompt(self, task: ResearchTask, plan: TaskSearchPlan, round_no: int,
                attempts: Dict[str, List[Dict[str, Any]]]) -> str:
        req = next((r for r in plan.requirements if r.queries), None)
        if req is None:
            return "(no unsatisfied requirements)"
        prior = attempts.get(req.requirement_id, [])
        pool = TerminologyEnricher.pool_terms(task)
        return (
            f"TASK {task.id}: {task.title}\n"
            f"OBJECTIVE: {task.objective}\n"
            f"INTENT: {task.intent}\n"
            f"EVIDENCE REQUIREMENT to search for: {req.requirement_id} - "
            f"{task.requirement(req.requirement_id).text if task.requirement(req.requirement_id) else ''}\n"
            f"UMLS TERMINOLOGY POOL: {', '.join(pool) if pool else '(none)'}\n"
            f"PREVIOUS ATTEMPTS for this requirement:\n{_attempts_block(prior)}\n"
            "Generate 2-3 NEW, specific search query strings for this "
            "requirement that differ from the previous attempts."
        )
