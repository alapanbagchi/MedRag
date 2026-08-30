"""Agentic v3 - Stage 3/5/10-13/17-18: the Worker sub-orchestrator.

Each Master task is assigned to its OWN Worker. A Worker is itself an
orchestrator: it completes ONE task and satisfies that task's evidence
requirements, running independently (its state is fully local - it never
reads or writes another worker's context, requirement 3).

Adaptive worker loop (requirements 1 + 4):

    UMLS enrichment (Stage 4)
        -> round 1: search-term selection (Stage 5)
        -> retrieve candidates (Stage 6) + context expansion (Stage 7)
        -> CRITIC verification gate (Stage 8)
        -> explicit evidence states (RETRIEVED -> UNDER_REVIEW ->
           ACCEPTED / REJECTED / CONTRADICTORY)
        -> requirement state: UNSATISFIED -> PARTIALLY_SUPPORTED -> SATISFIED
        -> insufficient?
              -> FAILURE ANALYSIS + QUERY REPLANNING (replan.py) with
                 meaningfully-different queries (never the same query)
              -> RETRIEVE AGAIN (rounds 2..N)
        -> sufficient? -> return the verified evidence package

Stop conditions (requirement 9): the requirement is SATISFIED, or one of
the hard budgets is exhausted (max rounds / max searches / max deep
inspections). No infinite loops: every condition is enforced in code.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Set

from src.lib import plural
from src.agentic_v3.critic import (
    CriticAgent,
    CriticVerdict,
    is_promising_for_deep_inspection,
)
from src.agentic_v3.deepinspect import DeepInspector
from src.agentic_v3.retriever import PaperRetrieverTool
from src.agentic_v3.search import SearchTermPlanner
from src.agentic_v3.state import (
    CriticRelevance,
    EvidenceRequirement,
    EvidenceSource,
    EvidenceStatus,
    RequirementReport,
    RequirementStatus,
    ResearchTask,
    RetrievedPaper,
    RunBudget,
    SupportDirection,
    TaskStatus,
    VerifiedEvidence,
    WorkerReport,
)
from src.agentic_v3.umls import TerminologyEnricher

logger = logging.getLogger("src.agentic_v3.worker")


class WorkerAgent:
    """One Worker sub-orchestrator: satisfies one task to its evidence N."""

    def __init__(
        self,
        config: Any = None,
        *,
        enricher: Any = None,
        search_planner: Any = None,
        replanner: Any = None,
        retriever: Any = None,
        critic: Any = None,
        deep_inspector: Any = None,
        events: Any = None,
    ):
        from src.config import AppConfig

        self.config = config or AppConfig()
        self._enricher = enricher
        self._search_planner = search_planner
        self._replanner = replanner
        self._retriever = retriever
        self._critic = critic
        self._deep_inspector = deep_inspector
        self.events = events

    # -- lazy deps ------------------------------------------------------

    def _enricher_obj(self) -> TerminologyEnricher:
        if self._enricher is None:
            self._enricher = TerminologyEnricher(config=self.config)
        return self._enricher

    def _search_planner_obj(self) -> SearchTermPlanner:
        if self._search_planner is None:
            self._search_planner = SearchTermPlanner(config=self.config)
        return self._search_planner

    def _replanner_obj(self):
        if self._replanner is None:
            from src.agentic_v3.replan import ReplannerAgent
            self._replanner = ReplannerAgent(config=self.config)
        return self._replanner

    def _retriever_obj(self) -> PaperRetrieverTool:
        if self._retriever is None:
            self._retriever = PaperRetrieverTool(config=self.config)
        return self._retriever

    def _critic_obj(self) -> CriticAgent:
        if self._critic is None:
            self._critic = CriticAgent(config=self.config)
        return self._critic

    def _deep_inspector_obj(self) -> DeepInspector:
        if self._deep_inspector is None:
            self._deep_inspector = DeepInspector(config=self.config)
        return self._deep_inspector

    def _emit(self, type_: str, **fields: Any) -> None:
        if self.events is not None:
            self.events.emit(type_, **fields)

    # ------------------------------------------------------------------
    async def run(
        self,
        task: ResearchTask,
        budget: Optional[RunBudget] = None,
        *,
        run_id: str = "",
    ) -> WorkerReport:
        """Run the adaptive worker loop to completion for one task.

        All worker state (attempts, promising papers, seen chunks, review
        logs, budget usage) is LOCAL to this call - isolated from every
        other worker (requirement 3). The only outputs are the mutated
        task object (its own requirements hold their own evidence) and the
        returned WorkerReport.
        """
        from src.trace import get_trace

        trace = get_trace()
        budget = budget or RunBudget()
        task.status = TaskStatus.RUNNING
        if self.events is not None:
            self.events.task_start(task.id, task.title)
        trace.bullet(
            f"Worker {task.id} started: “{task.title}” "
            f"(run {run_id or 'n/a'}).",
            agent="worker")

        # Stage 4: UMLS terminology enrichment (first retrieval step).
        try:
            pool = await self._enricher_obj().enrich(task)
            if self.events is not None and pool:
                self.events.emit("terminology_pool", task_id=task.id,
                                 concepts=TerminologyEnricher.pool_summary(task))
            trace.bullet(
                f"UMLS terminology expanded for {task.id}: "
                f"{plural(len(pool), 'concept')} to search with.",
                agent="worker")
        except Exception as exc:
            trace.bullet(
                f"UMLS terminology enrichment failed for {task.id} ({exc}); "
                f"falling back to the task's own terms.",
                agent="worker")

        # -- isolated per-worker session state (never shared) -----------
        attempts: Dict[str, List[Dict[str, Any]]] = {}       # req -> attempts
        promising: Dict[str, List[Dict[str, str]]] = {}      # req -> [{doc, hint}]
        seen_chunks: Set[str] = set()
        inspected_docs: Set[str] = set()
        stop_reason = "search_budget"

        round_no = 0
        while not task.satisfied() and not budget.exhausted():
            round_no += 1
            budget.retrieval_rounds_used += 1
            task.searches_used += 1
            attempt_id = f"A{round_no}"

            # Plan the round: round 1 uses the search-term planner; later
            # rounds use the FAILURE-ANALYSIS / REPLANNER with the full
            # scoped context (previous queries, retrieved docs, rejected
            # verdicts, missing evidence, remaining budget).
            if round_no == 1:
                plan = await self._search_planner_obj().plan(task, round_no, attempts)
            else:
                plan = await self._replanned(task, attempts, budget, run_id,
                                             round_no, trace)
            if self.events is not None:
                self.events.search_round(
                    task.id, round_no,
                    {r.requirement_id: r.queries for r in plan.requirements})
            trace.bullet(
                "Search round " + str(round_no) + " (" + attempt_id + "): "
                + "; ".join(
                    f"{r.requirement_id} with “{'” and “'.join(r.queries)}”"
                    if len(r.queries) == 1
                    else f"{r.requirement_id} with {plural(len(r.queries), 'formulation')}: "
                         f"{'; '.join(r.queries)}"
                    for r in plan.requirements if r.queries)
                + ".",
                agent="worker")

            for req_plan in plan.requirements:
                req = task.requirement(req_plan.requirement_id)
                if req is None or req.satisfied() or not req_plan.queries:
                    continue
                for query in req_plan.queries:
                    if req.satisfied():
                        break
                    if budget.search_budget_exhausted() and attempts.get(req.id):
                        break
                    await self._search_query(
                        task, req, query, round_no, attempt_id,
                        run_id=run_id,
                        seen_chunks=seen_chunks,
                        attempts=attempts,
                        promising=promising,
                        budget=budget,
                        trace=trace,
                    )

            # Stage 11/14: deep paper inspection for promising papers whose
            # excerpts did not answer (the paper might have something in it).
            # max_deep_inspections == 0 means deep inspection is DISABLED.
            for req in task.uncovered():
                if budget.max_deep_inspections <= 0 or \
                        budget.deep_inspections_used >= budget.max_deep_inspections:
                    break
                for cand in promising.get(req.id, []):
                    if budget.max_deep_inspections <= 0 or \
                            budget.deep_inspections_used >= budget.max_deep_inspections:
                        break
                    if req.satisfied():
                        break
                    if cand.get("doc") in inspected_docs:
                        continue
                    inspected_docs.add(cand.get("doc"))
                    await self._deep_inspect(
                        task, req, cand, budget=budget, run_id=run_id,
                        attempt_id=attempt_id, trace=trace)

        # Stop conditions (requirement 9): satisfied OR budget exhausted.
        if task.satisfied():
            stop_reason = "satisfied"
        elif budget.rounds_exhausted():
            stop_reason = "rounds_exhausted"
        elif budget.search_budget_exhausted():
            stop_reason = "search_budget"
        else:
            stop_reason = "deep_inspection_budget"
        for req in task.evidence_requirements:
            if not req.satisfied():
                gap = self._exhaustion_gap(req, attempts.get(req.id, []))
                req.mark_exhausted(gap)
        task.finalize()
        if self.events is not None:
            self.events.task_done(task.id, task.status.value,
                                  self._report_summary(task), stop_reason=stop_reason)

        report = self._build_report(task, budget, run_id=run_id,
                                    stop_reason=stop_reason)
        if self.events is not None:
            self.events.worker_report(
                task.id, report.status, report.searches_used,
                report.deep_inspections_used,
                [r.model_dump() for r in report.requirements],
                stop_reason=stop_reason,
                evidence=[e.model_dump(mode="json") for e in report.evidence])
        trace.bullet(
            f"Worker {task.id} finished — {task.status.value} after "
            f"{plural(task.searches_used, 'search')}"
            + (f" and {plural(task.deep_inspections_used, 'deep inspection')}"
               if task.deep_inspections_used else "")
            + f" (stop reason: {stop_reason}).",
            agent="worker")
        return report

    # ------------------------------------------------------------------
    async def _replanned(self, task: ResearchTask,
                         attempts: Dict[str, List[Dict[str, Any]]],
                         budget: RunBudget, run_id: str, round_no: int,
                         trace: Any) -> Any:
        """Round >=2: failure analysis + replanning per unsatisfied req.

        Returns a TaskSearchPlan whose queries are MEANINGFULLY DIFFERENT
        from every previous attempt (enforced deterministically).
        """
        from src.agentic_v3.replan import build_replan_context
        from src.agentic_v3.search import RequirementSearch, TaskSearchPlan

        plan = TaskSearchPlan(round_no=round_no,
                              rationale="adaptive replanning")
        for req in task.evidence_requirements:
            if req.satisfied():
                plan.requirements.append(RequirementSearch(
                    requirement_id=req.id, rationale="already satisfied",
                    queries=[]))
                continue
            ctx = build_replan_context(
                run_id, task, req,
                previous_queries=[a.get("query", "") for a in attempts.get(req.id, [])],
                retrieved_documents=attempts.get(req.id, []),
                budget=budget,
            )
            analysis = await self._replanner_obj().plan(ctx, task, req)
            if self.events is not None and analysis.queries:
                self.events.replan(
                    task.id, req.id, f"A{round_no}",
                    analysis.diagnosis, analysis.missing_evidence,
                    analysis.queries, run_id=run_id)
            trace.bullet(
                f"Round {round_no} did not settle {req.id} yet "
                f"({(analysis.diagnosis or 'insufficient evidence')[:120]}). "
                f"Replanning — next: {analysis.queries}.",
                agent="worker")
            plan.requirements.append(RequirementSearch(
                requirement_id=req.id, rationale=analysis.strategy,
                queries=analysis.queries))
        return plan

    # ------------------------------------------------------------------
    async def _search_query(
        self,
        task: ResearchTask,
        req: EvidenceRequirement,
        query: str,
        round_no: int,
        attempt_id: str,
        *,
        run_id: str,
        seen_chunks: Set[str],
        attempts: Dict[str, List[Dict[str, Any]]],
        promising: Dict[str, List[Dict[str, str]]],
        budget: RunBudget,
        trace: Any,
    ) -> None:
        """One query: Stage 6 retrieve -> Stage 7 context -> Stage 8 CRITIC.

        Every candidate paper is stamped with its OWN scoped id
        (evidence_id = task.requirement.attempt.seq) BEFORE retrieval
        output is inspected, so a critic result cannot migrate.
        """
        # one SEARCH attempt = one executed query (budget accounting per
        # query, not per round - so adaptive replanning can always run its
        # first query before the search budget is checked again)
        budget.searches_used += 1
        papers = await self._retriever_obj().search(
            task, req, query,
            exclude_chunk_ids=sorted(seen_chunks),
            top_k=budget.max_papers_per_round,
            round_no=round_no,
        )
        new_papers = [p for p in papers if p.chunk_id not in seen_chunks]
        if papers:
            trace.bullet(
                f"Retrieved {plural(len(papers), 'candidate passage')} for "
                f"{req.id} (query: “{query}”): "
                + ", ".join(
                    f"{p.document_id} ({p.section}"
                    + (f", score {p.score:.3f}" if p.score is not None else "")
                    + ")" for p in papers[:5])
                + ("…" if len(papers) > 5 else "") + ".",
                agent="worker")
        if self.events is not None:
            self.events.retrieved(
                task.id, req.id,
                [{"document_id": p.document_id, "section": p.section,
                  "score": round(float(p.score or 0.0), 4)} for p in papers],
                query=query, attempt_id=attempt_id)

        notes: List[str] = []
        # stamp scope BEFORE any judgement (requirement 3): pool AND batch
        # both judge only papers whose scope was fixed here first.
        stamped: List[RetrievedPaper] = []
        for idx, paper in enumerate(new_papers, start=1):
            if paper.chunk_id:
                seen_chunks.add(paper.chunk_id)
            paper = paper.model_copy(update={
                "evidence_id": f"{req.id}.{attempt_id}.E{idx}",
                "task_id": task.id,
                "requirement_id": req.id,
                "attempt_id": attempt_id,
                "status": EvidenceStatus.RETRIEVED,
                "source_query": query,
                "round_no": round_no,
            })
            self._emit_evidence_state(task, req, paper.evidence_id,
                                      EvidenceStatus.RETRIEVED.value,
                                      EvidenceStatus.UNDER_REVIEW.value)
            stamped.append(paper)

        # Stage 8: CRITIC gate — all passages of this query in ONE call
        # (Mistral batch when enabled, otherwise sequential). A None verdict
        # means critic-failed (cannot verify), NOT a rejection.
        critic = self._critic_obj()
        judge_papers = getattr(critic, "judge_papers", None)
        if judge_papers is not None:
            verdict_pairs = await judge_papers(
                task, req, stamped, run_id=run_id, attempt_id=attempt_id)
        else:
            # duck-typed / test critics with only the per-paper judge()
            verdict_pairs = []
            for paper in stamped:
                try:
                    verdict_pairs.append((paper, await critic.judge(
                        task, req, paper, run_id=run_id, attempt_id=attempt_id)))
                except Exception as exc:
                    logger.warning("critic failed %s/%s (%s)", task.id, req.id, exc)
                    verdict_pairs.append((paper, None))
        for paper, verdict in verdict_pairs:
            if verdict is None:
                notes.append(f"critic-failed: {paper.document_id}")
                continue
            self._record_verdict(task, req, paper, verdict, trace, run_id=run_id)
            if verdict.accepted:
                if verdict.support == SupportDirection.SUPPORTS:
                    notes.append(f"accepted(support): {paper.document_id}")
                elif verdict.support == SupportDirection.CONTRADICTS:
                    notes.append(f"accepted(contradicts): {paper.document_id}")
            if is_promising_for_deep_inspection(verdict):
                existing = {d["doc"] for d in promising.get(req.id, [])}
                if paper.document_id and paper.document_id not in existing:
                    promising.setdefault(req.id, []).append({
                        "doc": paper.document_id,
                        "hint": (verdict.note or "")[:240],
                    })
            if req.satisfied():
                break

        attempts.setdefault(req.id, []).append({
            "query": query,
            "round": round_no,
            "attempt_id": attempt_id,
            "papers": len(papers),
            "notes": " | ".join(notes)[:800],
        })
        for p in papers:
            attempts.setdefault(req.id + ":docs", []).append({
                "document_id": p.document_id,
                "section": p.section,
                "excerpt": (p.text or "")[:300],
            })

    def _record_verdict(self, task: ResearchTask, req: EvidenceRequirement,
                        paper: RetrievedPaper, verdict: CriticVerdict,
                        trace: Any, *, run_id: str = "") -> None:
        """Fold one scoped verdict into the requirement.

        The verdict carries its own scope; the requirement that records it
        MUST be the same requirement it was judged against (a reviewer
        mismatch raises - the log can never be polluted cross-requirement).
        """
        if verdict.requirement_id and verdict.requirement_id != req.id:
            raise ValueError(
                f"verdict scope mismatch: verdict.requirement_id="
                f"{verdict.requirement_id} != req.id={req.id}")
        entry = {
            "evidence_id": paper.evidence_id or verdict.evidence_id or "",
            "document_id": paper.document_id,
            "chunk_id": paper.chunk_id,
            "attempt_id": verdict.attempt_id or paper.attempt_id or "",
            "relevance": verdict.relevance.value,
            "answers_task": verdict.answers_task.value,
            "support": verdict.support.value,
            "confidence": verdict.confidence,
            "note": verdict.note,
            "accepted": verdict.accepted,
        }
        req.record_review(entry)
        from src import logfire_obs as lf
        lf.record_metric("evidence_accepted" if verdict.accepted
                         else "evidence_rejected", 1,
                         task_id=task.id, requirement_id=req.id,
                         document=paper.document_id)
        old = req.status
        if verdict.accepted:
            item = self._evidence_from_verdict(
                task, req, paper, verdict, run_id=run_id)
            terminal = (EvidenceStatus.CONTRADICTORY
                        if item.support == SupportDirection.CONTRADICTS
                        else EvidenceStatus.ACCEPTED)
            item.status = terminal
            if req.add_evidence(item):
                self._emit_evidence_state(task, req, item.id,
                                          EvidenceStatus.UNDER_REVIEW.value,
                                          terminal.value)
                if self.events is not None:
                    self.events.evidence_added(
                        task.id, req.id, item.id, item.document_id,
                        item.support.value, item.source.value,
                        attempt_id=item.attempt_id)
                trace.bullet(
                    f"✓ Verified {item.document_id} ({item.section}) for "
                    f"{req.id} — {item.support.value} the requirement "
                    f"(confidence {item.confidence:.0%}): "
                    f"{(verdict.note or '')[:160]}",
                    agent="critic")
        else:
            req.rejected += 1
            self._emit_evidence_state(
                task, req, entry["evidence_id"],
                EvidenceStatus.UNDER_REVIEW.value, EvidenceStatus.REJECTED.value)
            trace.bullet(
                f"✗ Rejected {paper.document_id} ({paper.section}) for "
                f"{req.id} — {(verdict.note or 'not relevant to the requirement')[:160]}",
                agent="critic")
        if req.status != old:
            self._emit_requirement_state(task, req, old, run_id=run_id)

    @staticmethod
    def _evidence_from_verdict(task: ResearchTask, req: EvidenceRequirement,
                               paper: RetrievedPaper,
                               verdict: CriticVerdict,
                               *, run_id: str = "") -> VerifiedEvidence:
        from src.agentic_v3.critic import evidence_excerpt

        return VerifiedEvidence(
            id=paper.evidence_id or f"E-{task.id}-{req.id}-{len(req.accepted) + 1}",
            run_id=run_id,
            task_id=task.id,
            requirement_id=req.id,
            attempt_id=verdict.attempt_id or paper.attempt_id or "",
            document_id=paper.document_id,
            chunk_id=paper.chunk_id,
            section=paper.section,
            excerpt=evidence_excerpt(paper.text),
            claim=verdict.note or verdict.support.value,
            support=verdict.support,
            confidence=verdict.confidence,
            source=EvidenceSource.RETRIEVAL,
            retrieval_method=paper.retrieval_method,
            rank=paper.rank,
            critic_verdict=verdict.model_dump(mode="json"),
            note=verdict.note,
            search_query=paper.source_query,
        )

    # ------------------------------------------------------------------
    async def _deep_inspect(self, task: ResearchTask, req: EvidenceRequirement,
                            cand: Dict[str, str], *,
                            budget: RunBudget, run_id: str, attempt_id: str,
                            trace: Any) -> None:
        """Stage 11 deep paper inspection (bounded by budget)."""
        doc_id = cand.get("doc", "")
        if not doc_id:
            return
        budget.deep_inspections_used += 1
        task.deep_inspections_used += 1
        if self.events is not None:
            self.events.deep_inspection(task.id, req.id, doc_id, "starting",
                                        attempt_id=attempt_id)
        hint = (cand.get("hint") or "").strip()
        trace.bullet(
            f"Deep-inspecting {doc_id} for {req.id} — the search excerpt was "
            f"not decisive, reading the full paper now"
            + (f" (why it was flagged: {hint[:140]})" if hint else "")
            + ".",
            agent="worker")
        try:
            from src import logfire_obs as lf
            with lf.span("deep_inspection", document=doc_id, requirement=req.id):
                found = await self._deep_inspector_obj().inspect(
                    task, req, doc_id, context_hint=cand.get("hint", ""),
                    run_id=run_id, attempt_id=attempt_id)
            lf.record_metric("deep_inspections", 1,
                             task_id=task.id, requirement_id=req.id, document=doc_id)
        except Exception as exc:
            logger.warning("deep inspection failed %s: %s", doc_id, exc)
            if self.events is not None:
                self.events.deep_inspection(task.id, req.id, doc_id, "failed",
                                            detail=str(exc)[:160])
            return
        added = 0
        for item in found:
            item.id = f"{req.id}.{attempt_id}.DI{len(req.accepted) + 1}"
            if req.add_evidence(item):
                added += 1
        if self.events is not None:
            self.events.deep_inspection(task.id, req.id, doc_id,
                                        "accepted" if added else "no_evidence",
                                        findings=len(found), verified=added > 0)
        trace.bullet(
            f"Deep inspection of {doc_id} for {req.id}: "
            f"{'accepted ' + plural(added, 'verified passage') if added else 'found no verifiable evidence — excerpt-based verdict stands'}.",
            agent="deep_inspector")

    # ------------------------------------------------------------------
    @staticmethod
    def _exhaustion_gap(req: EvidenceRequirement, attempts: List[Dict[str, Any]]) -> str:
        got = req.coverage()
        target = max(1, req.target_n)
        searched = len(attempts)
        base = (f"only {got} independent supporting paper(s) verified; "
                f"target {target}; {searched} search attempt(s) made")
        if req.contradicting_papers():
            base += f"; {len(req.contradicting_papers())} paper(s) contradict"
        else:
            base += "; literature did not yield more supporting evidence"
        return base[:300]

    @staticmethod
    def _report_summary(task: ResearchTask) -> str:
        parts = [f"{r.id}:{r.coverage()}/{r.target_n}"
                 for r in task.evidence_requirements]
        return " ".join(parts)

    def _build_report(self, task: ResearchTask, budget: RunBudget, *,
                      run_id: str = "", stop_reason: str = "") -> WorkerReport:
        req_reports: List[RequirementReport] = []
        for req in task.evidence_requirements:
            req_reports.append(RequirementReport(
                requirement_id=req.id,
                text=req.text,
                target_n=req.target_n,
                coverage=req.coverage(),
                status=req.status.value,
                gap=req.gap,
                papers=[
                    {
                        "evidence_id": e.id,
                        "document_id": e.document_id,
                        "chunk_id": e.chunk_id,
                        "attempt_id": e.attempt_id,
                        "retrieval_method": e.retrieval_method,
                        "rank": e.rank,
                        "state": e.status.value,
                        "claim": e.claim,
                        "excerpt": (e.excerpt or "")[:400],
                        "confidence": e.confidence,
                        "support": e.support.value,
                        "source": e.source.value,
                    }
                    for e in req.accepted
                ],
            ))
        return WorkerReport(
            run_id=run_id,
            task_id=task.id,
            task_title=task.title,
            status=task.status.value,
            stop_reason=stop_reason,
            requirements=req_reports,
            evidence=[e for r in task.evidence_requirements for e in r.accepted],
            searches_used=task.searches_used,
            deep_inspections_used=task.deep_inspections_used,
            budget_exhausted=budget.exhausted(),
            gaps=[r.gap for r in task.evidence_requirements if r.gap],
            summary=self._report_summary(task),
        )

    # ------------------------------------------------------------------
    def _emit_evidence_state(self, task: ResearchTask, req: EvidenceRequirement,
                             evidence_id: str, old: str, new: str) -> None:
        if self.events is not None:
            self.events.evidence_state(task.id, req.id, evidence_id, old, new)

    def _emit_requirement_state(self, task: ResearchTask,
                                req: EvidenceRequirement,
                                old: RequirementStatus,
                                *, run_id: str = "") -> None:
        if self.events is not None:
            self.events.requirement_state(
                task.id, req.id, old.value, req.status.value,
                req.coverage(), req.target_n, run_id=run_id)
