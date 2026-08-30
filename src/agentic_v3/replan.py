"""Agentic v3 - the failure-analysis / query-replanning agent (requirement 1).

Adaptive retrieval loop:

    CRITIC says "insufficient" for a requirement
        -> FAILURE ANALYSIS (why did it fail? what evidence is missing?)
        -> QUERY REPLANNING (new queries, meaningfully different from all
           previous attempts)
        -> RETRIEVE AGAIN
        -> CRITIC the new candidates
        -> continue until satisfied or the retry/search budget is exhausted

The replanner receives an EXPLICIT, fully-scoped context (never any shared
mutable state):

    original task
    evidence requirement
    previous queries
    retrieved documents
    rejected candidates + critic reasoning/verdicts
    accepted papers
    remaining budget (rounds and searches)

"Meaningfully different" is enforced DETERMINISTICALLY after the LLM call
(meaningfully_different()): a proposed query must contain at least one
content token absent from EVERY previously tried query. Merely re-wording or
re-ordering the same terms is rejected, so the system never simply runs the
same query again.
"""

from __future__ import annotations
from src.prompts.load import load_prompt

import logging
import re
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from src.agentic_v3.search import fallback_queries
from src.agentic_v3.state import EvidenceRequirement, ResearchTask

logger = logging.getLogger("src.agentic_v3.replan")


class ReplanContext(BaseModel):
    """Everything the replanner may use - fully scoped, explicit, no globals."""
    run_id: str = ""
    task_id: str = ""
    task_title: str = ""
    objective: str = ""
    requirement_id: str = ""
    requirement_text: str = ""
    target_n: int = 3
    coverage: int = 0
    previous_queries: List[str] = Field(default_factory=list)
    retrieved_documents: List[Dict[str, Any]] = Field(default_factory=list)
    rejected_candidates: List[Dict[str, Any]] = Field(default_factory=list)
    accepted_papers: List[str] = Field(default_factory=list)
    remaining_rounds: int = 0
    remaining_searches: int = 0


class FailureAnalysis(BaseModel):
    """Output of one failure-analysis + replanning step for ONE requirement."""
    diagnosis: str = ""               # why the previous attempts failed
    missing_evidence: str = ""        # what evidence is still missing
    strategy: str = ""                # how the new queries differ
    queries: List[str] = Field(default_factory=list)


REPLANNER_SYSTEM_PROMPT = load_prompt('agentic_v3', 'replanner.txt')

_STOPWORDS = {
    "a", "an", "the", "and", "or", "of", "in", "on", "for", "with", "to",
    "vs", "versus", "between", "among", "after", "before", "during", "by",
    "at", "from", "no", "not", "effect", "effects", "association", "risk",
    "study", "studies", "patients", "patient", "use", "used",
}


def _content_tokens(query: str) -> set:
    return set(
        t for t in re.findall(r"[a-zA-Z][a-zA-Z0-9-]{1,}", (query or "").lower())
        if t not in _STOPWORDS and len(t) > 2
    )


def meaningfully_different(queries: List[str], tried: List[str]) -> List[str]:
    """Keep only queries with >=1 content token absent from EVERY tried query.

    This is the deterministic guarantee behind requirement 1: an adaptive
    retry must never just re-run the same query (or a reworded twin).
    """
    tried_tokens = [_content_tokens(q) for q in tried if q]
    tried_exact = {q.casefold() for q in tried if q}
    out: List[str] = []
    seen: set = set()
    for raw in queries:
        q = " ".join((raw or "").split()).strip()
        if not q or len(q) < 7:
            continue
        if q.casefold() in tried_exact:
            continue
        toks = _content_tokens(q)
        if not toks:
            continue
        # a reworded twin of a tried query (all its tokens already covered)
        if any(toks <= covered for covered in tried_tokens):
            continue
        key = tuple(sorted(toks))
        if key in seen:
            continue
        seen.add(key)
        out.append(q[:240])
        if len(out) >= 3:
            break
    return out


def build_replan_context(
    run_id: str,
    task: ResearchTask,
    requirement: EvidenceRequirement,
    *,
    previous_queries: List[str],
    retrieved_documents: List[Dict[str, Any]],
    budget: Any,
) -> ReplanContext:
    """Deterministic context assembly - the worker's whole memory for one
    requirement is handed over explicitly (requirement 3: no globals)."""
    rejected = [
        {
            "evidence_id": r.get("evidence_id", ""),
            "document_id": r.get("document_id", ""),
            "attempt_id": r.get("attempt_id", ""),
            "reason": (r.get("note") or r.get("relevance", ""))[:300],
            "support": r.get("support", ""),
        }
        for r in requirement.reviewed
        if r.get("accepted") is False
    ]
    seen_docs: set = set()
    docs: List[Dict[str, Any]] = []
    for d in retrieved_documents:
        doc_key = d.get("document_id", "")
        if doc_key in seen_docs:
            continue
        seen_docs.add(doc_key)
        docs.append({
            "document_id": doc_key,
            "section": d.get("section", ""),
            "excerpt": " ".join((d.get("excerpt") or "").split())[:240],
        })
    return ReplanContext(
        run_id=run_id,
        task_id=task.id,
        task_title=task.title,
        objective=task.objective,
        requirement_id=requirement.id,
        requirement_text=requirement.text,
        target_n=max(1, requirement.target_n),
        coverage=requirement.coverage(),
        previous_queries=list(previous_queries),
        retrieved_documents=docs[:20],
        rejected_candidates=rejected[-20:],
        accepted_papers=requirement.supporting_papers(),
        remaining_rounds=max(0, budget.max_retrieval_rounds - budget.retrieval_rounds_used),
        remaining_searches=max(0, budget.max_searches - budget.searches_used),
    )


def _context_block(ctx: ReplanContext) -> str:
    lines = [
        f"TASK {ctx.task_id}: {ctx.task_title}",
        f"OBJECTIVE: {ctx.objective}",
        f"EVIDENCE REQUIREMENT {ctx.requirement_id}: {ctx.requirement_text}",
        f"TARGET: {ctx.target_n} independent papers | coverage now: {ctx.coverage}",
        "PREVIOUS QUERIES: " + (" | ".join(ctx.previous_queries) if ctx.previous_queries else "(none)"),
        "ACCEPTED PAPERS SO FAR: " + (", ".join(ctx.accepted_papers) if ctx.accepted_papers else "(none)"),
    ]
    lines.append("RETRIEVED DOCUMENTS:")
    if ctx.retrieved_documents:
        for d in ctx.retrieved_documents:
            lines.append(f"- {d['document_id']} [{d['section']}]: {d['excerpt']}")
    else:
        lines.append("- (none returned)")
    lines.append("REJECTED CANDIDATES (critic reasoning):")
    if ctx.rejected_candidates:
        for r in ctx.rejected_candidates:
            lines.append(
                f"- {r.get('evidence_id', '?')} doc={r.get('document_id', '?')} "
                f"attempt={r.get('attempt_id', '?')} reason={r.get('reason', '')}"
            )
    else:
        lines.append("- (none)")
    lines.append(f"REMAINING: rounds={ctx.remaining_rounds} searches={ctx.remaining_searches}")
    return "\n".join(lines)


class ReplannerAgent:
    """Diagnoses insufficiency and produces meaningfully-different queries."""

    def __init__(self, config: Any = None, model: Any = None):
        from src.config import AppConfig
        from src.llm import build_model_for
        from pydantic_ai import Agent

        self.config = config or AppConfig()
        self.model = model or build_model_for(self.config, role="planner")
        self.agent = Agent(
            self.model,
            system_prompt=REPLANNER_SYSTEM_PROMPT,
            name="replanner",
        )

    async def plan(
        self,
        ctx: ReplanContext,
        task: ResearchTask,
        requirement: EvidenceRequirement,
    ) -> FailureAnalysis:
        """One diagnosis + replanning step for ONE requirement."""
        from src.llm.run import ask_structured
        from src.trace import get_trace

        trace = get_trace()
        fallback = self._deterministic(ctx, task, requirement)
        trace.agent("replanner", output_type="FailureAnalysis",
                    meta={"task": ctx.task_id, "requirement": ctx.requirement_id,
                          "coverage": ctx.coverage, "target": ctx.target_n,
                          "previous_queries": len(ctx.previous_queries)})
        try:
            out = await ask_structured(
                self.agent,
                _context_block(ctx)
                + "\n\nDiagnose the failure and propose NEW queries. "
                  "Return the analysis JSON.",
                FailureAnalysis,
                label=(f"replanner:{ctx.run_id}:{ctx.task_id}:"
                       f"{ctx.requirement_id}"),
                max_tokens=min(1100, getattr(self.config, "agent_max_tokens", 2048)),
            )
        except Exception as exc:
            logger.warning("replanning failed for %s/%s (%s); fallback used",
                           ctx.task_id, ctx.requirement_id, exc)
            return fallback
        queries = meaningfully_different(out.queries or [], ctx.previous_queries)
        if not queries:
            return fallback
        return FailureAnalysis(
            diagnosis=(out.diagnosis or "").strip(),
            missing_evidence=(out.missing_evidence or "").strip(),
            strategy=(out.strategy or "").strip(),
            queries=queries,
        )

    @staticmethod
    def _deterministic(ctx: ReplanContext, task: ResearchTask,
                       requirement: EvidenceRequirement) -> FailureAnalysis:
        """Fallback: rotate terminology pool pairings, skipping all tried."""
        queries = fallback_queries(
            task, requirement.text,
            round_no=(len(ctx.previous_queries) + 1),
            tried_queries=ctx.previous_queries,
        )
        queries = meaningfully_different(queries, ctx.previous_queries)
        if not queries and ctx.previous_queries:
            queries = []
        return FailureAnalysis(
            diagnosis=("previous attempts did not surface passages that "
                       "answer the requirement"),
            missing_evidence=(
                f"at least {max(0, ctx.target_n - ctx.coverage)} more "
                "independent supporting paper(s)"),
            strategy="rotated terminology pairings (deterministic fallback)",
            queries=queries,
        )
