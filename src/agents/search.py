"""Agentic v3 - Stage 5: search-term selection per worker round (LLM + fallback)."""

from __future__ import annotations

import logging
import re
from typing import Any

from pydantic import BaseModel, Field

from src.agentic.state import ResearchTask
from src.tools.terminology import TerminologyEnricher
from src.prompts.load import load_prompt

logger = logging.getLogger("src.agents.search")


class RequirementSearch(BaseModel):
    """One evidence requirement's search formulations for this round."""
    requirement_id: str = ""
    rationale: str = ""
    queries: list[str] = Field(default_factory=list)


class TaskSearchPlan(BaseModel):
    """The worker's search plan for one round (all requirements covered)."""
    round_no: int = 1
    rationale: str = ""
    requirements: list[RequirementSearch] = Field(default_factory=list)

    def queries_for(self, requirement_id: str) -> list[str]:
        for r in self.requirements:
            if r.requirement_id == requirement_id:
                return list(r.queries)
        return []


class RequirementPlan(BaseModel):
    """Raw LLM output for ONE requirement."""
    rationale: str = ""
    queries: list[str] = Field(default_factory=list)


SEARCH_PLANNER_SYSTEM_PROMPT = load_prompt('agents', 'search_planner.txt')


def _clean_query(q: Any) -> str:
    q = " ".join(str(q or "").split())
    if not q or len(q) < 7:
        return ""
    for marker in (", {", "}, "):
        pos = q.find(marker)
        if pos != -1:
            q = q[:pos]
    return q.replace("{", "").replace("}", "").strip()[:240]


def _requirement_outcome_terms(requirement_text: str) -> list[str]:
    phrases = []
    for tok in re.findall(r"[a-zA-Z][a-zA-Z0-9+/ ()-]{2,}", requirement_text or ""):
        t = " ".join(tok.split())
        if len(t.split()) >= 2 and t.casefold() not in {p.casefold() for p in phrases}:
            phrases.append(t)
    return phrases[:3]


def _quote(s: str) -> str:
    """Quote multi-word terms for AND-pair formulations."""
    return f'"{s}"' if len(s.split()) > 1 else s


def fallback_queries(task: ResearchTask, requirement_text: str,
                     round_no: int, tried_queries: list[str]) -> list[str]:
    """Deterministic progressive formulations: quoted AND-pairs, rotated per round."""
    terms = TerminologyEnricher.pool_terms(task) or [t for t in (task.entities or []) if t]
    outcome_terms = _requirement_outcome_terms(requirement_text)
    tried = {q.casefold() for q in tried_queries}
    candidates = [f"{_quote(term)} AND {_quote(out)}"
                  for term in terms for out in outcome_terms]
    if not candidates:
        candidates = [requirement_text.strip()[:200]]
    start = (round_no - 1) % max(1, len(candidates))
    out = []
    for q in candidates[start:] + candidates[:start]:
        q = _clean_query(q)
        if q and q.casefold() not in tried:
            tried.add(q.casefold())
            out.append(q)
        if len(out) >= 3:
            break
    return (out or [requirement_text.strip()[:200]])[:3]


def _attempts_block(attempts: list[dict[str, Any]]) -> str:
    if not attempts:
        return "(no previous searches for this requirement)"
    return "\n".join(
        f"- query: {a.get('query', '')} | papers: {a.get('papers', 0)} | "
        f"rejected-notes: {(a.get('notes') or '')[:300]}" for a in attempts)


class SearchTermPlanner:
    """Selects the most probable medical search terms per requirement/round."""

    def __init__(self, config: Any = None, model: Any = None):
        from pydantic_ai import Agent

        from src.config import AppConfig
        from src.llm import build_model_for

        self.config = config or AppConfig()
        self.model = model or build_model_for(self.config, role="planner")
        self.agent = Agent(self.model, system_prompt=SEARCH_PLANNER_SYSTEM_PROMPT,
                           output_type=RequirementPlan, retries=2,
                           name="search_term_planner")

    async def plan(self, task: ResearchTask, round_no: int,
                   attempts: dict[str, list[dict[str, Any]]] | None = None) -> TaskSearchPlan:
        """Search plan covering every unsatisfied requirement (LLM refines the first)."""
        attempts = attempts or {}
        plan = TaskSearchPlan(round_no=round_no, rationale="")
        for req in task.evidence_requirements:
            if req.satisfied():
                plan.requirements.append(RequirementSearch(
                    requirement_id=req.id, rationale="already satisfied", queries=[]))
                continue
            prior = attempts.get(req.id, [])
            tried = [a.get("query", "") for a in prior]
            plan.requirements.append(RequirementSearch(
                requirement_id=req.id, rationale="fallback",
                queries=fallback_queries(task, req.text, round_no, tried)))

        req = next((r for r in plan.requirements if r.queries), None)
        if req and req.requirement_id and task.requirement(req.requirement_id):
            prior = attempts.get(req.requirement_id, [])
            pool = TerminologyEnricher.pool_terms(task)
            prompt = (
                f"TASK {task.id}: {task.title}\nOBJECTIVE: {task.objective}\n"
                f"EVIDENCE REQUIREMENT: {req.requirement_id} - "
                f"{task.requirement(req.requirement_id).text}\n"
                f"UMLS POOL: {', '.join(pool) if pool else '(none)'}\n"
                f"PREVIOUS ATTEMPTS:\n{_attempts_block(prior)}\n"
                "Generate 2-3 NEW, specific search query strings that differ from previous attempts.")
            try:
                out = await self.agent.run(prompt)
                queries = [q for q in (_clean_query(q) for q in (out.output.queries or [])) if q]
                if queries:
                    plan.requirements[0] = plan.requirements[0].model_copy(
                        update={"queries": queries, "rationale": out.output.rationale or "llm"})
            except Exception as exc:  # noqa: BLE001
                logger.warning("search planning failed for %s r%d (%s); fallback kept",
                               task.id, round_no, exc)
        return plan