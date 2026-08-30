"""Step 5 - The agentic tool loop.

An LLM agent owns the retrieval decision loop. It is handed two tools:

  search_subquery(query, evidence_required, exclude_chunk_ids=...)
      -> hybrid retrieval + paragraph restore + LLM verification in ONE call;
         returns the kept/rejected/unknown units and the rejection reasons.
  umls_lookup(entity_text)
      -> UMLS/MeSH preferred name + synonyms for a term, so the agent can
         discover canonical search vocabulary on its own.

The agent decides what query to issue, whether to expand a term via UMLS,
how to revise the query after seeing rejection reasons, and when to stop.
It loops through the model tool-calling turns natively (pydantic_ai
Agent.run) until it emits a final EvidenceReport. A hard round cap
(MAX_AGENT_ROUNDS) prevents runaway looping; shared state (seen chunks, kept
evidence) lives in AgentDeps so the model cannot re-fetch the same chunk.
"""

from __future__ import annotations
from src.prompts.load import load_prompt

import logging
from dataclasses import dataclass, field
from typing import Any, List, Optional

from pydantic import BaseModel, Field

from pydantic_ai import RunContext

from src.agentic.planner import SubQueryPlan
from src.agentic.retriever_tool import HybridRetrieverTool
from src.agentic.verify import VerifyTool, VerifiedUnit

logger = logging.getLogger("src.agentic.loop")


@dataclass
class AgentDeps:
    subquery: SubQueryPlan
    retriever: HybridRetrieverTool
    verifier: VerifyTool
    umls_enricher: Any = None
    max_rounds: int = 6
    min_evidence: int = 3
    base_query: str = ""
    searched_queries: List[str] = field(default_factory=list)
    seen_chunk_ids: set = field(default_factory=set)
    kept: List[VerifiedUnit] = field(default_factory=list)
    rounds: int = 0


async def search_subquery(
    ctx: RunContext[AgentDeps],
    query: str,
    evidence_required: Optional[List[str]] = None,
    exclude_chunk_ids: Optional[List[str]] = None,
) -> dict:
    """Search the corpus for *query*, restore hits to full paragraphs, verify
    relevance, and return kept/rejected/unknown units + rejection reasons."""
    from src.trace import get_trace
    trace = get_trace()
    deps: AgentDeps = ctx.deps
    deps.rounds += 1
    query = " ".join(query.split()).strip()
    if not query:
        return {"error": "empty query", "kept": [], "rejected": [],
                "unknown": [], "rejection_reasons": []}
    deps.searched_queries.append(query)
    trace.bullet(f"round {deps.rounds}: search query={query!r} exclude={len(deps.seen_chunk_ids)} seen")

    req = list(evidence_required or deps.subquery.evidence_required or [])
    sub = deps.subquery.model_copy(update={"query": query, "evidence_required": req})
    exclude = list(dict.fromkeys((exclude_chunk_ids or []) + sorted(deps.seen_chunk_ids)))

    results = await deps.retriever.search(sub, top_k=6, exclude_chunk_ids=exclude)
    trace.bullet(f"  retrieved {len(results)} units")
    for r in results:
        deps.seen_chunk_ids.add(r.chunk_id)

    outcome = await deps.verifier.verify(sub, results, base_query=deps.base_query or deps.subquery.target)
    trace.bullet(
        f"  verdicts: kept={len(outcome.kept)} rejected={len(outcome.rejected)} unknown={len(outcome.unknown)}"
    )
    for u in outcome.kept:
        trace.bullet(f"    [KEEP] {u.document_id}/{u.chunk_id} conf={u.confidence:.2f}")
    for u in outcome.rejected:
        trace.bullet(f"    [REJECT] {u.document_id}/{u.chunk_id} reason={u.reason[:80]!r}")
    known = {k.chunk_id for k in deps.kept}
    for u in outcome.kept:
        if u.chunk_id not in known:
            deps.kept.append(u)
            known.add(u.chunk_id)

    return {
        "query": query,
        "round": deps.rounds,
        "retrieved": len(results),
        "kept": [{"chunk_id": u.chunk_id, "document_id": u.document_id,
                   "confidence": u.confidence, "section": u.section,
                   "excerpt": " ".join(u.paragraph_text.split())[:220]} for u in outcome.kept],
        "rejected": [{"chunk_id": u.chunk_id, "document_id": u.document_id,
                       "reason": u.reason} for u in outcome.rejected],
        "unknown": [{"chunk_id": u.chunk_id, "reason": u.reason} for u in outcome.unknown],
        "rejection_reasons": outcome.rejection_reasons,
        "kept_so_far": len(deps.kept),
        "distinct_papers_so_far": len({u.document_id for u in deps.kept if u.document_id}),
    }


async def umls_lookup(ctx: RunContext[AgentDeps], entity_text: str) -> dict:
    """Look up a term in UMLS/MeSH; return its preferred name and synonyms."""
    from src.trace import get_trace
    trace = get_trace()
    trace.tool("umls_lookup", {"entity_text": entity_text})
    deps: AgentDeps = ctx.deps
    enricher = deps.umls_enricher
    if enricher is None:
        from src.agentic.umls_tool import UMLSEnricher
        enricher = deps.umls_enricher = UMLSEnricher(config=deps.retriever.config)
    try:
        concept = await enricher.umls.search_concept(entity_text)
    except Exception as exc:
        return {"term": entity_text, "found": False, "error": str(exc)[:120]}
    if not concept or not concept.found:
        return {"term": entity_text, "found": False, "preferred_name": None, "synonyms": []}
    return {"term": entity_text, "found": True,
            "preferred_name": concept.preferred_name,
            "synonyms": list(concept.synonyms or [])}


class EvidenceReport(BaseModel):
    subquery_id: str = ""
    succeeded: bool = False
    summary: str = ""
    evidence_excerpts: List[str] = Field(default_factory=list)
    citations: List[str] = Field(default_factory=list)
    searches_performed: List[str] = Field(default_factory=list)
    notes: List[str] = Field(default_factory=list)


LOOP_SYSTEM_PROMPT = load_prompt('agentic_v1', 'loop.txt')


class AgenticLoop:
    """Runs the tool-calling retrieval agent for one subquery."""

    def __init__(self, config: Any = None, retriever: HybridRetrieverTool = None,
                 verifier: VerifyTool = None, umls_enricher: Any = None, model: Any = None):
        from src.config import AppConfig
        from src.llm import build_model

        self.config = config or AppConfig()
        self.model = model or build_model(self.config)
        self.retriever = retriever or HybridRetrieverTool(config=self.config)
        self.verifier = verifier or VerifyTool(config=self.config)
        self.umls_enricher = umls_enricher
        self._agent = None

    def _build_agent(self):
        from pydantic_ai import Agent

        agent = Agent(
            self.model,
            deps_type=AgentDeps,
            output_type=EvidenceReport,
            system_prompt=LOOP_SYSTEM_PROMPT,
            name="agentic_retriever",
            retries=3,
        )
        agent.tool(search_subquery)
        agent.tool(umls_lookup)
        return agent

    async def run(self, sub: SubQueryPlan, base_query: str = "") -> EvidenceReport:
        from src.trace import get_trace
        trace = get_trace()
        trace.stage(f"LOOP {sub.id}: {sub.target}")
        trace.bullet(f"intent={sub.intent!r} | evidence_required={sub.evidence_required} | max_rounds={getattr(self.config, 'max_agent_rounds', 6)}")
        agent = self._build_agent()
        deps = AgentDeps(
            subquery=sub,
            retriever=self.retriever,
            verifier=self.verifier,
            umls_enricher=self.umls_enricher,
            max_rounds=int(getattr(self.config, "max_agent_rounds", 6)),
            min_evidence=int(getattr(self.config, "agent_min_evidence", 3)),
            base_query=base_query,
        )
        max_rounds = deps.max_rounds
        prompt = (
            f"SUBQUERY {sub.id}: {sub.target}\n"
            + f"INTENT: {sub.intent or '(unspecified)'}\n"
            + f"EVIDENCE REQUIRED: {', '.join(sub.evidence_required) if sub.evidence_required else '(unspecified)'}\n"
            + f"INITIAL QUERY: {sub.query or sub.target}\n"
            + f"ENTITIES: {', '.join(e.text for e in sub.entities) or '(none)'}\n"
            + f"You have at most {max_rounds} search rounds. Begin."
        )
        try:
            result = await agent.run(prompt, deps=deps)
        except Exception as exc:
            logger.warning("agent loop failed for %s: %s", sub.id, exc)
            return EvidenceReport(
                subquery_id=sub.id, succeeded=False,
                summary=f"agent loop error: {str(exc)[:200]}",
                searches_performed=deps.searched_queries,
                notes=[str(exc)[:200]],
            )
        report: EvidenceReport = result.output
        report.subquery_id = sub.id
        report.searches_performed = deps.searched_queries
        if report.succeeded and not report.evidence_excerpts and not report.citations:
            report.succeeded = False
        trace.bullet(f"loop {sub.id} finished: succeeded={report.succeeded} searches={deps.searched_queries} citations={report.citations}")
        for e in report.evidence_excerpts:
            trace.bullet(f"  excerpt: {' '.join(e.split())[:160]}")
        return report
