"""
Worker sub-orchestrator: adaptive search + deep inspection rounds.
Runs until all evidence requirements are satisfied or budget is exhausted.
"""

from __future__ import annotations

import logging
from typing import Any

from pydantic import BaseModel, Field

from src.agents.critic import CriticAgent
from src.agents.deepinspect import DeepInspector
from src.agents.replan import ReplannerAgent, build_replan_context
from src.tools.paper_retriever import PaperRetrieverTool
from src.agents.search import RequirementSearch, SearchTermPlanner, TaskSearchPlan
from src.agentic.state import (
    EvidenceRequirement,
    RequirementReport,
    ResearchTask,
    RunBudget,
    TaskStatus,
    WorkerReport,
)
from src.tools.terminology import TerminologyEnricher
from src.agentic.worker_pipelines import DeepInspectionPipeline, SearchPipeline
from src.config import AppConfig

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------
# No-op event sink
# ----------------------------------------------------------------------


class NullEvents:
    def __getattr__(self, name: str) -> object:
        return lambda *args, **kwargs: None


# ----------------------------------------------------------------------
# Run state
# ----------------------------------------------------------------------


class RunState(BaseModel):
    budget: RunBudget = Field(default_factory=RunBudget)
    attempts: dict[str, list[dict[str, object]]] = Field(default_factory=dict)
    promising: dict[str, list[dict[str, object]]] = Field(default_factory=dict)
    seen_chunks: set[str] = Field(default_factory=set)
    inspected_docs: set[str] = Field(default_factory=set)
    new_promising: bool = False

    def search_budget_blocks(self, req: EvidenceRequirement) -> bool:
        return self.budget.search_budget_exhausted() and bool(self.attempts.get(req.id))


def _satisfied_entry(req_id: str) -> RequirementSearch:
    return RequirementSearch(requirement_id=req_id, rationale="satisfied", queries=[])


def _stop_reason(task: ResearchTask, ctx: RunState) -> str:
    if task.satisfied(): return "satisfied"
    if ctx.budget.rounds_exhausted(): return "rounds_exhausted"
    if ctx.budget.search_budget_exhausted(): return "search_budget"
    if ctx.budget.deep_inspections_used >= ctx.budget.max_deep_inspections:
        return "deep_inspection_budget"
    return "unknown"


def _p(label: str, msg: str) -> None:
    """Debug narration - stdout-free; only emitted at DEBUG level."""
    if logger.isEnabledFor(logging.DEBUG):
        logger.debug("[%s] %s", label, msg)


# ----------------------------------------------------------------------
# Worker Agent
# ----------------------------------------------------------------------


class WorkerAgent:
    def __init__(self, config: AppConfig | None = None, events: Any = None, *,
                 enricher: TerminologyEnricher | None = None,
                 retriever: PaperRetrieverTool | None = None,
                 critic: CriticAgent | None = None,
                 planner: SearchTermPlanner | None = None,
                 replanner: ReplannerAgent | None = None,
                 deep_inspector: DeepInspector | None = None) -> None:
        self.config = config or AppConfig()
        self.events = events or NullEvents()
        self.enricher = enricher or TerminologyEnricher(self.config)
        self.planner = planner or SearchTermPlanner(self.config)
        self.replanner = replanner or ReplannerAgent(self.config)
        self.search_pipe = SearchPipeline(
            retriever=retriever or PaperRetrieverTool(self.config),
            critic=critic or CriticAgent(self.config),
            events=self.events,
        )
        self.deep_pipe = DeepInspectionPipeline(
            deep_inspector=deep_inspector or DeepInspector(self.config),
            events=self.events,
        )

    # ------------------------------------------------------------------

    async def run(self, task: ResearchTask, budget: RunBudget | None = None,
                  run_id: str = "") -> WorkerReport:
        _p("CHECKING TASK STRUCTURE", f"{task}")
        ctx = RunState(budget=budget or RunBudget())
        _p("run", f"RunState created: budget={ctx.budget.model_dump()}")
        task.status = TaskStatus.RUNNING
        self.events.task_start(task.id, task.title)

        # -- UMLS enrichment (best-effort) --
        _p("enrich", f"Starting UMLS enrichment for {task.id}...")
        try:
            pool = await self.enricher.enrich(task)
            _p("enrich", f"Pool returned {len(pool)} terms: {[t.surface_form for t in pool[:5]]}")
        except Exception:
            logger.exception("UMLS enrichment failed for %s; continuing", task.id)
            _p("enrich", "Enrichment FAILED, continuing with existing terms")

        # -- main loop --
        round_no = 0
        while not task.satisfied() and not ctx.budget.exhausted():
            round_no += 1
            ctx.budget.retrieval_rounds_used += 1
            task.searches_used += 1
            ctx.new_promising = False
            _p("loop", f"=== ROUND {round_no} === satisfied={task.satisfied()} "
               f"budget_left=searches:{ctx.budget.max_searches - ctx.budget.searches_used} "
               f"rounds:{ctx.budget.max_retrieval_rounds - ctx.budget.retrieval_rounds_used}")

            # -- search phase --
            try:
                await self._search_round(task, ctx, run_id, round_no)
            except Exception:
                logger.exception("Search round %d failed", round_no)

            _p("loop", f"After search: attempts={dict(ctx.attempts)}")
            _p("loop", f"Promising candidates: {dict(ctx.promising)}")

            # -- deep inspection (only if new candidates found) --
            if ctx.new_promising:
                _p("inspect", "New promising docs found — running deep inspection")
                try:
                    await self._inspect_round(task, ctx, run_id, round_no)
                except Exception:
                    logger.exception("Inspection round %d failed", round_no)
            else:
                _p("inspect", "No new promising docs — skipping deep inspection")

        _p("loop", f"LOOP ENDED: satisfied={task.satisfied()} budget_exhausted={ctx.budget.exhausted()}")
        return self._finalize(task, ctx, run_id)

    # ------------------------------------------------------------------

    async def _search_round(self, task: ResearchTask, ctx: RunState,
                            run_id: str, round_no: int) -> None:
        _p("search", f"Building plan for round {round_no}...")
        plan = await self._build_plan(task, ctx, run_id, round_no)
        _p("search", f"Plan has {len(plan.requirements)} requirement(s): "
           f"{[(r.requirement_id, r.queries) for r in plan.requirements]}")
        self.events.search_round(
            task.id, round_no,
            {r.requirement_id: r.queries for r in plan.requirements})

        for entry in plan.requirements:
            req = task.requirement(entry.requirement_id)
            if not req or req.satisfied():
                _p("search", f"  {entry.requirement_id}: SKIP (not found or satisfied)")
                continue
            for query in entry.queries:
                if req.satisfied() or ctx.search_budget_blocks(req):
                    _p("search", f"  {entry.requirement_id}: BREAK (satisfied or budget)")
                    break
                _p("search", f"  {entry.requirement_id}: running query={query!r}")
                await self.search_pipe.run(task, req, query, round_no, f"A{round_no}", ctx, run_id=run_id)
                _p("search", f"  {entry.requirement_id}: after query — coverage={req.coverage()}/{req.target_n} "
                   f"status={req.status.value} attempts={len(ctx.attempts.get(req.id, []))}")
                if ctx.promising.get(req.id):
                    ctx.new_promising = True

    # ------------------------------------------------------------------

    async def _build_plan(self, task: ResearchTask, ctx: RunState,
                          run_id: str, round_no: int) -> TaskSearchPlan:
        if round_no == 1:
            _p("plan", "Round 1 → using SearchTermPlanner (LLM)")
            return await self.planner.plan(task, round_no, ctx.attempts)

        _p("plan", f"Round {round_no} → using ReplannerAgent (failure analysis + new queries)")
        plan = TaskSearchPlan(round_no=round_no, rationale="adaptive")
        for req in task.evidence_requirements:
            if req.satisfied():
                plan.requirements.append(_satisfied_entry(req.id))
                _p("plan", f"  {req.id}: already satisfied → empty query list")
                continue
            attempts = ctx.attempts.get(req.id, [])
            _p("plan", f"  {req.id}: {len(attempts)} previous attempts, replanning...")
            try:
                rctx = build_replan_context(
                    run_id, task, req,
                    previous_queries=[a.get("query") for a in attempts],
                    retrieved_documents=attempts,
                    budget=ctx.budget,
                )
                _p("plan", f"  {req.id}: replan context built — prev_queries={rctx.previous_queries}")
                analysis = await self.replanner.plan(rctx, task, req)
                queries = analysis.queries
                rationale = analysis.strategy
                _p("plan", f"  {req.id}: replanner returned queries={queries} strategy={rationale!r}")
                self.events.replan(
                    task.id, req.id, f"A{round_no}",
                    analysis.diagnosis, analysis.missing_evidence,
                    analysis.queries, run_id=run_id)
            except (RuntimeError, ValueError, KeyError, TypeError):
                logger.warning("Replanner failed for %s; using last query", req.id)
                queries = [attempts[-1].get("query")] if attempts else []
                rationale = "fallback"
                _p("plan", f"  {req.id}: replanner FAILED → fallback query={queries}")
            plan.requirements.append(
                RequirementSearch(requirement_id=req.id, rationale=rationale, queries=queries)
            )
        return plan

    # ------------------------------------------------------------------

    async def _inspect_round(self, task: ResearchTask, ctx: RunState,
                             run_id: str, round_no: int) -> None:
        uncovered = task.uncovered()
        _p("inspect", f"Round {round_no}: {len(uncovered)} uncovered requirement(s)")
        for req in uncovered:
            candidates = ctx.promising.get(req.id, [])
            _p("inspect", f"  {req.id}: {len(candidates)} promising candidate(s)")
            for candidate in candidates:
                if req.satisfied() or ctx.budget.deep_inspections_used >= ctx.budget.max_deep_inspections:
                    _p("inspect", "    BREAK: req satisfied or budget exhausted")
                    break
                doc = candidate.get("doc")
                if not doc or doc in ctx.inspected_docs:
                    _p("inspect", f"    SKIP: doc={doc!r} (already inspected or missing)")
                    continue
                ctx.inspected_docs.add(doc)
                _p("inspect", f"    DEEP INSPECT: doc={doc!r} hint={candidate.get('hint', '')!r}")
                await self.deep_pipe.run(task, req, candidate, ctx, run_id=run_id, attempt_id=f"A{round_no}")
                _p("inspect", f"    AFTER inspect: coverage={req.coverage()}/{req.target_n}")

    # ------------------------------------------------------------------

    def _finalize(self, task: ResearchTask, ctx: RunState, run_id: str) -> WorkerReport:
        stop = _stop_reason(task, ctx)
        _p("finalize", f"Stop reason: {stop}")
        for req in task.evidence_requirements:
            if not req.satisfied():
                req.mark_exhausted(f"target {req.target_n}, {len(ctx.attempts.get(req.id, []))} searches")
        task.finalize()

        report = WorkerReport(
            run_id=run_id,
            task_id=task.id,
            task_title=task.title,
            status=task.status.value,
            stop_reason=stop,
            requirements=[
                RequirementReport(
                    requirement_id=r.id, text=r.text, target_n=r.target_n,
                    coverage=r.coverage(), status=r.status.value, gap=r.gap,
                    papers=[{
                        "evidence_id": e.id, "document_id": e.document_id,
                        "chunk_id": e.chunk_id, "attempt_id": e.attempt_id,
                        "retrieval_method": e.retrieval_method, "rank": e.rank,
                        "state": e.status.value, "claim": e.claim,
                        "excerpt": (e.excerpt or "")[:400], "confidence": e.confidence,
                        "support": e.support.value, "source": e.source.value,
                    } for e in r.accepted])
                for r in task.evidence_requirements],
            searches_used=task.searches_used,
            deep_inspections_used=task.deep_inspections_used,
            budget_exhausted=ctx.budget.exhausted(),
            gaps=[r.gap for r in task.evidence_requirements if r.gap],
            summary=" ".join(f"{r.id}:{r.coverage()}/{r.target_n}" for r in task.evidence_requirements),
        )
        _p("finalize", f"Report: status={report.status} searches={report.searches_used} "
           f"deep={report.deep_inspections_used} gaps={report.gaps}")
        self.events.worker_report(
            task.id, report.status, report.searches_used, report.deep_inspections_used,
            report.model_dump()["requirements"], stop, report.model_dump()["evidence"]
        )
        return report

