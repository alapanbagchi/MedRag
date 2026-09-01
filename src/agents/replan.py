"""Agentic v3 - failure analysis + query replanning (LLM + deterministic filter)."""

from __future__ import annotations

import logging
import re
from typing import Any

from pydantic import BaseModel, Field

from src.agents.search import fallback_queries
from src.agentic.state import EvidenceRequirement, ResearchTask
from src.prompts.load import load_prompt

logger = logging.getLogger("src.agents.replan")


class ReplanContext(BaseModel):
    """Everything the replanner may use - fully scoped, no globals."""
    run_id: str = ""
    task_id: str = ""
    task_title: str = ""
    objective: str = ""
    requirement_id: str = ""
    requirement_text: str = ""
    target_n: int = 3
    coverage: int = 0
    previous_queries: list[str] = Field(default_factory=list)
    retrieved_documents: list[dict[str, Any]] = Field(default_factory=list)
    rejected_candidates: list[dict[str, Any]] = Field(default_factory=list)
    accepted_papers: list[str] = Field(default_factory=list)
    remaining_rounds: int = 0
    remaining_searches: int = 0


class FailureAnalysis(BaseModel):
    """Output of one failure-analysis + replanning step for ONE requirement."""
    diagnosis: str = ""
    missing_evidence: str = ""
    strategy: str = ""
    queries: list[str] = Field(default_factory=list)


REPLANNER_SYSTEM_PROMPT = load_prompt('agents', 'replanner.txt')

_STOPWORDS = {
    "a", "an", "the", "and", "or", "of", "in", "on", "for", "with", "to",
    "vs", "versus", "between", "among", "after", "before", "during", "by",
    "at", "from", "no", "not", "effect", "effects", "association", "risk",
    "study", "studies", "patients", "patient", "use", "used",
}


def _content_tokens(query: str) -> set:
    return {t for t in re.findall(r"[a-zA-Z][a-zA-Z0-9-]{1,}", (query or "").lower())
        if t not in _STOPWORDS and len(t) > 2}


def meaningfully_different(queries: list[str], tried: list[str]) -> list[str]:
    """Keep only queries with >=1 content token absent from EVERY tried query."""
    tried_tokens = [_content_tokens(q) for q in tried if q]
    tried_exact = {q.casefold() for q in tried if q}
    out, seen = [], set()
    for raw in queries:
        q = " ".join((raw or "").split()).strip()
        if not q or len(q) < 7 or q.casefold() in tried_exact:
            continue
        toks = _content_tokens(q)
        if not toks or any(toks <= covered for covered in tried_tokens):
            continue
        key = tuple(sorted(toks))
        if key in seen:
            continue
        seen.add(key)
        out.append(q[:240])
        if len(out) >= 3:
            break
    return out


def build_replan_context(run_id: str, task: ResearchTask,
                         requirement: EvidenceRequirement, *,
                         previous_queries: list[str],
                         retrieved_documents: list[dict[str, Any]],
                         budget: Any) -> ReplanContext:
    """Deterministic context assembly for one requirement (no globals)."""
    rejected = [
        {"evidence_id": r.get("evidence_id", ""), "document_id": r.get("document_id", ""),
         "attempt_id": r.get("attempt_id", ""),
         "reason": (r.get("note") or r.get("relevance", ""))[:300],
         "support": r.get("support", "")}
        for r in requirement.reviewed if r.get("accepted") is False
    ]
    seen, docs = set(), []
    for d in retrieved_documents:
        doc_key = d.get("document_id", "")
        if doc_key in seen:
            continue
        seen.add(doc_key)
        docs.append({"document_id": doc_key, "section": d.get("section", ""),
                     "excerpt": " ".join((d.get("excerpt") or "").split())[:240]})
    return ReplanContext(
        run_id=run_id, task_id=task.id, task_title=task.title, objective=task.objective,
        requirement_id=requirement.id, requirement_text=requirement.text,
        target_n=max(1, requirement.target_n), coverage=requirement.coverage(),
        previous_queries=list(previous_queries), retrieved_documents=docs[:20],
        rejected_candidates=rejected[-20:], accepted_papers=requirement.supporting_papers(),
        remaining_rounds=max(0, budget.max_retrieval_rounds - budget.retrieval_rounds_used),
        remaining_searches=max(0, budget.max_searches - budget.searches_used))


def _context_block(ctx: ReplanContext) -> str:
    lines = [
        f"TASK {ctx.task_id}: {ctx.task_title}", f"OBJECTIVE: {ctx.objective}",
        f"EVIDENCE REQUIREMENT {ctx.requirement_id}: {ctx.requirement_text}",
        f"TARGET: {ctx.target_n} independent papers | coverage now: {ctx.coverage}",
        "PREVIOUS QUERIES: " + (" | ".join(ctx.previous_queries) if ctx.previous_queries else "(none)"),
        "ACCEPTED PAPERS SO FAR: " + (", ".join(ctx.accepted_papers) if ctx.accepted_papers else "(none)"),
    ]
    lines.append("RETRIEVED DOCUMENTS:")
    lines += ([f"- {d['document_id']} [{d['section']}]: {d['excerpt']}" for d in ctx.retrieved_documents]
              or ["- (none returned)"])
    lines.append("REJECTED CANDIDATES (critic reasoning):")
    lines += ([f"- {r.get('evidence_id', '?')} doc={r.get('document_id', '?')} "
               f"attempt={r.get('attempt_id', '?')} reason={r.get('reason', '')}"
               for r in ctx.rejected_candidates] or ["- (none)"])
    lines.append(f"REMAINING: rounds={ctx.remaining_rounds} searches={ctx.remaining_searches}")
    return "\n".join(lines)


class ReplannerAgent:
    """Diagnoses insufficiency and produces meaningfully-different queries."""

    def __init__(self, config: Any = None, model: Any = None):
        from pydantic_ai import Agent

        from src.config import AppConfig
        from src.llm import build_model_for

        self.config = config or AppConfig()
        self.model = model or build_model_for(self.config, role="planner")
        self.agent = Agent(self.model, system_prompt=REPLANNER_SYSTEM_PROMPT,
                           output_type=FailureAnalysis, retries=2, name="replanner")

    async def plan(self, ctx: ReplanContext, task: ResearchTask,
                   requirement: EvidenceRequirement) -> FailureAnalysis:
        """One diagnosis + replanning step; deterministic filter guarantees new queries."""
        fallback = self._fallback(ctx, task, requirement)
        try:
            out = (await self.agent.run(
                _context_block(ctx) + "\n\nDiagnose the failure and propose NEW queries.")).output
            queries = meaningfully_different(out.queries or [], ctx.previous_queries)
            if not queries:
                return fallback
            return FailureAnalysis(diagnosis=(out.diagnosis or "").strip(),
                                   missing_evidence=(out.missing_evidence or "").strip(),
                                   strategy=(out.strategy or "").strip(), queries=queries)
        except Exception as exc:  # noqa: BLE001
            logger.warning("replanning failed for %s/%s (%s); fallback used",
                           ctx.task_id, ctx.requirement_id, exc)
            return fallback

    @staticmethod
    def _fallback(ctx: ReplanContext, task: ResearchTask,
                  requirement: EvidenceRequirement) -> FailureAnalysis:
        """Fallback: rotated terminology pairings, skipping all tried."""
        queries = meaningfully_different(
            fallback_queries(task, requirement.text,
                             round_no=len(ctx.previous_queries) + 1,
                             tried_queries=ctx.previous_queries),
            ctx.previous_queries)
        return FailureAnalysis(
            diagnosis="previous attempts did not surface passages that answer the requirement",
            missing_evidence=(f"at least {max(0, ctx.target_n - ctx.coverage)} more "
                              "independent supporting paper(s)"),
            strategy="rotated terminology pairings (deterministic fallback)",
            queries=queries)