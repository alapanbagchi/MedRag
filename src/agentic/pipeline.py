"""Agentic v3 - the run pipeline (Stages 1 -> 24).

                    USER QUERY
                        |
                        v
               MASTER ORCHESTRATOR          (Stage 2: plan with tasks,
                        |                     evidence requirements, N,
                        |                     stop criteria)
            +-----------+-----------+
            v           v           v
         WORKER 1    WORKER 2    WORKER 3   (Stages 3-18, run IN PARALLEL:
            |           |           |         UMLS -> search terms ->
            +-----------+-----------+         retrieve -> context expansion
                        |                     -> CRITIC -> N-threshold loop
                        v                     -> deep inspection -> package)
                 VERIFIED EVIDENCE
                        |
                        v
              CONTRADICTION AGENT           (Stage 14: cross-evidence)
                        |
                   contradiction?
                    /         \
                  NO           YES
                  |             |
                  |             v
                  |      RESOLUTION AGENT    (Stage 15: paper search tool)
                  |             |
                  +------+------+
                         v
                  FINAL EVIDENCE SET         (Stage 23)
                         |
                         v
                    FINAL ANSWER             (Stage 24, honest synthesis)

Runtime guarantees (mirroring the agentic v2 philosophy):
  * the LLM only PROPOSES structure (tasks, search terms, verdicts,
    contradictions, resolutions); the runtime is authoritative about
    thresholds, budgets, dedup, acceptance and citation repair;
  * every stage is wrapped in asyncio.wait_for so a stalled call cannot
    hang the run;
  * hard budgets (searches / deep inspections / parallel workers) are
    enforced in code;
  * insufficient literature and unresolved contradictions are REPORTED,
    never manufactured away.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from typing import Any

from src.agents.contradiction import ContradictionAgent
from src.agentic.events import V3Events
from src.agents.master import MasterOrchestratorAgent
from src.agents.resolution import ResolutionAgent
from src.agentic.state import (
    Contradiction,
    MasterPlan,
    ResearchTask,
    ResolutionStatus,
    RunBudget,
    TaskStatus,
    V3RunState,
    WorkerReport,
)
from src.agents.synthesize import FinalSynthesizer
from src.agents.worker import WorkerAgent
from src.lib.utils import plural

logger = logging.getLogger("src.agentic.pipeline")


class AgenticV3Pipeline:
    """Runs the full multi-agent evidence pipeline for one question."""

    def __init__(
        self,
        config: Any = None,
        *,
        master: Any = None,
        worker: Any = None,
        contradiction_agent: Any = None,
        resolution_agent: Any = None,
        synthesizer: Any = None,
        events: Any = None,
        run_id: str = "",
    ):
        from src.config import AppConfig

        self.config = config or AppConfig()
        self.run_id = run_id
        self.master = master or MasterOrchestratorAgent(config=self.config)
        self.worker = worker or WorkerAgent(config=self.config)
        self.contradiction_agent = (
            contradiction_agent or ContradictionAgent(config=self.config))
        self.resolution_agent = (
            resolution_agent or ResolutionAgent(config=self.config))
        self.synthesizer = synthesizer or FinalSynthesizer(config=self.config)
        self.events = events
        if self.events is not None and not isinstance(self.events, V3Events):
            self.events = V3Events(self.events)
        if self.events is not None:
            self.worker.events = self.events

    # ------------------------------------------------------------------
    def _budget(self) -> RunBudget:
        return RunBudget(
            max_searches=int(getattr(self.config, "agentic_v3_max_searches", 5)),
            max_retrieval_rounds=int(
                getattr(self.config, "agentic_v3_max_retrieval_rounds", 5)),
            max_papers_per_round=int(
                getattr(self.config, "agentic_v3_papers_per_search", 5)),
            max_deep_inspections=int(
                getattr(self.config, "agentic_v3_max_deep_inspections", 3)),
            evidence_target=int(getattr(self.config, "agentic_v3_evidence_target", 3)),
            max_workers=int(getattr(self.config, "agentic_v3_max_workers", 4)),
        )

    def _timeout(self, key: str, default: float) -> float:
        return float(getattr(self.config, key, default))

    # ------------------------------------------------------------------
    async def answer(self, query: str) -> dict[str, Any]:
        """Run the full pipeline; returns a JSON-safe result dict.

        Wraps the run in a top-level Logfire span (``medrag.run``) so every
        stage, retrieval and LLM GenAI span from the whole run nests inside
        one trace; with Logfire disabled this is a thin passthrough.
        """
        from src.lib import logfire_obs as lf

        lf.ensure_configured(self.config)
        if lf.enabled():
            async with lf.run_span(query=query):
                return await self._answer(query)
        return await self._answer(query)

    async def _answer(self, query: str) -> dict[str, Any]:
        from src.lib import logfire_obs as lf
        from src.lib.trace import get_trace

        trace = get_trace()
        budget = self._budget()
        run_id = self.run_id or f"run-{os.getpid()}-{int(time.time() * 1000)}"
        state = V3RunState(run_id=run_id, question=query, budget=budget)
        trace.stage("ORCHESTRATOR STARTED")
        trace.bullet(
            f"New research run {run_id} for the question: “{query}”. "
            f"Budget: up to {plural(budget.max_workers, 'parallel task')}, "
            f"{plural(budget.max_searches, 'search')} per task, "
            f"{plural(budget.evidence_target, 'supporting paper')} per "
            f"evidence requirement.",
            agent="orchestrator")
        lf.set_run_context(run_id=run_id, question=query)
        if self.events is not None:
            self.events.run_start(query, budget.model_dump(), run_id=run_id)

        try:
            # Stage 2: Master Orchestrator.
            plan = await self._master_plan(state, trace)
            for task in plan.tasks:
                state.add_task(task)

            # Stages 3-18: workers run in parallel; each gets the run id and
            # its OWN budget copy (isolated worker context, requirement 3).
            reports = await self._run_workers(state, trace)

            # Stage 14: Contradiction Agent (across ALL workers' evidence).
            contradictions = await self._detect_contradictions(state, trace)

            # Stage 15: dedicated Resolution Agent per contradiction.
            await self._resolve_contradictions(state, contradictions, trace)

            # Stages 23-24: Final evidence set -> synthesis.
            report = await self._synthesize(state, trace)
            state.final_answer = report.model_dump()
            state.terminal = True
            state.stop_reason = "synthesized"
            lf.final_answer(report, run_id=run_id)
            if self.events is not None:
                # the full answer report as its own event (the UI renders it)
                self.events.answer(report.model_dump(mode="json"))
        finally:
            lf.clear_run_context()

        if self.events is not None:
            self.events.final_evidence(
                len(state.verified_evidence()),
                [r.gap for t in state.tasks for r in t.evidence_requirements if r.gap],
                len(state.contradictions),
                sum(1 for c in state.contradictions
                    if c.resolution and c.resolution.status == ResolutionStatus.RESOLVED),
                sum(1 for c in state.contradictions
                    if not c.resolution
                    or c.resolution.status == ResolutionStatus.UNRESOLVED),
            )
            self.events.run_end(state.stop_reason,
                                (state.final_answer or {}).get("confidence", 0.0),
                                (state.final_answer or {}).get("summary", ""))
        trace.bullet(
            f"Run finished — {plural(len(state.tasks), 'task')}, "
            f"{plural(len(state.verified_evidence()), 'verified evidence item')}, "
            f"{plural(len(state.contradictions), 'contradiction')}.",
            agent="orchestrator")
        return self._result_dict(state, reports)

    # ------------------------------------------------------------------
    async def _master_plan(self, state: V3RunState, trace: Any) -> MasterPlan:
        try:
            plan = await asyncio.wait_for(
                self.master.plan(state.question, state.budget),
                timeout=self._timeout("agentic_v3_master_timeout", 120.0),
            )
        except TimeoutError:
            from src.agents.master import Decomposition
            logger.warning("master timeout; single-hop fallback")
            fallback = Decomposition(
                rationale="master timeout; single-hop fallback",
                tasks=[{"id": "T1", "title": "Fallback",
                         "objective": state.question[:300],
                         "intent": "answer the question",
                         "evidence_required": [state.question[:200] or "user question"]}],
            )
            plan = fallback.to_master_plan(state.question, budget=state.budget)
        except Exception as exc:
            from src.agents.master import Decomposition
            logger.warning("master failed (%s); deterministic fallback", exc)
            fallback = Decomposition(
                rationale=f"master failed: {str(exc)[:120]}",
                tasks=[{"id": "T1", "title": "Fallback",
                         "objective": state.question[:300],
                         "intent": "answer the question",
                         "evidence_required": [state.question[:200] or "user question"]}],
            )
            plan = fallback.to_master_plan(state.question, budget=state.budget)
        state.plan = plan
        if self.events is not None:
            self.events.master_plan(plan.model_dump(mode="json"))
        trace.bullet(
            f"Master plan accepted: {plural(len(plan.tasks), 'task')}, "
            f"{plural(sum(len(t.evidence_requirements) for t in plan.tasks), 'evidence requirement')} "
            f"to satisfy.",
            agent="master")
        return plan

    async def _run_workers(self, state: V3RunState, trace: Any) -> list[WorkerReport]:
        tasks = state.tasks[: max(1, state.budget.max_workers)]
        timeout = self._timeout("agentic_v3_worker_timeout", 360.0)

        async def one(task: ResearchTask) -> WorkerReport:
            from src.lib import logfire_obs as lf
            # EACH worker gets its OWN budget copy (spec section 4: every task
            # receives "a retrieval budget"). A shared budget would let the
            # first worker to spend exhaust the others before they ever run.
            worker_budget = state.budget.model_copy(deep=True)
            with lf.task_context(task_id=task.id, task_title=task.title):
                try:
                    return await asyncio.wait_for(
                        self.worker.run(task, worker_budget, run_id=state.run_id),
                        timeout=timeout)
                except Exception as exc:
                    timed_out = isinstance(exc, TimeoutError)
                    reason = "timeout" if timed_out else "error"
                    gap = (f"worker timed out after {timeout:.0f}s; evidence incomplete"
                           if timed_out else f"worker error: {str(exc)[:160]}")
                    logger.warning("worker %s %s (%s)", task.id, reason, gap)
                    return self._worker_failure(
                        task, worker_budget, state.run_id, reason, gap,
                        mark_searches=timed_out)

        trace.bullet(
            f"Dispatching {plural(len(tasks), 'worker')} in parallel — each "
            f"task runs with its own retrieval budget "
            f"(stage timeout {timeout:.0f}s).",
            agent="orchestrator")
        return await asyncio.gather(*(one(t) for t in tasks))

    @staticmethod
    def _worker_failure(task: ResearchTask, budget: RunBudget, run_id: str,
                        stop_reason: str, gap: str, *,
                        mark_searches: bool = False) -> WorkerReport:
        """Mark a failed worker (timeout/error) and return its honest report."""
        if mark_searches:
            task.searches_used = max(task.searches_used, 1)
        task.status = TaskStatus.EXHAUSTED
        for req in task.evidence_requirements:
            if not req.satisfied():
                req.mark_exhausted(gap)
        return WorkerReport(
            run_id=run_id, task_id=task.id, task_title=task.title,
            status=task.status.value, stop_reason=stop_reason,
            requirements=[],
            searches_used=task.searches_used,
            deep_inspections_used=task.deep_inspections_used,
            budget_exhausted=budget.exhausted())

    async def _detect_contradictions(self, state: V3RunState, trace: Any) -> list[Contradiction]:
        if len(state.verified_evidence()) < 2:
            return []
        try:
            contradictions = await asyncio.wait_for(
                self.contradiction_agent.detect(state),
                timeout=self._timeout("agentic_v3_contradiction_timeout", 120.0),
            )
        except TimeoutError:
            from src.agents.contradiction import detect_contradictions_deterministic
            contradictions = detect_contradictions_deterministic(
                state.verified_evidence())
        except Exception as exc:
            logger.warning("contradiction detection failed (%s)", exc)
            from src.agents.contradiction import detect_contradictions_deterministic
            contradictions = detect_contradictions_deterministic(
                state.verified_evidence())
        state.contradictions = contradictions
        doc_of = {e.id: e.document_id for e in state.all_evidence()}
        for c in contradictions:
            if self.events is not None:
                self.events.contradiction(c.id, c.claim, c.kind.value,
                                          c.evidence_a, c.evidence_b, c.description)
            side_a = sorted({doc_of[i] for i in c.evidence_a if i in doc_of})
            side_b = sorted({doc_of[i] for i in c.evidence_b if i in doc_of})
            trace.bullet(
                f"Contradiction {c.id} found: “{c.claim[:140]}” — evidence "
                f"from {', '.join(side_a) or c.evidence_a} conflicts with "
                f"{', '.join(side_b) or c.evidence_b}.",
                agent="contradiction")
        return contradictions

    async def _resolve_contradictions(self, state: V3RunState,
                                      contradictions: list[Contradiction],
                                      trace: Any) -> None:
        if not contradictions:
            return
        timeout = self._timeout("agentic_v3_resolution_timeout", 180.0)

        async def one(c: Contradiction) -> Contradiction:
            try:
                outcome = await asyncio.wait_for(
                    self.resolution_agent.resolve(c, state), timeout=timeout)
            except TimeoutError:
                from src.agentic.state import ResolutionOutcome
                outcome = ResolutionOutcome(
                    status=ResolutionStatus.UNRESOLVED,
                    explanation=f"resolution timed out after {timeout:.0f}s",
                )
            except Exception as exc:
                from src.agentic.state import ResolutionOutcome
                outcome = ResolutionOutcome(
                    status=ResolutionStatus.UNRESOLVED,
                    explanation=f"resolution failed: {str(exc)[:160]}",
                )
            c.resolution = outcome
            if self.events is not None:
                self.events.resolution(c.id, outcome.status.value,
                                       outcome.explanation, outcome.additional_papers)
            extra = (f" ({plural(len(outcome.additional_papers), 'additional paper')} "
                     f"considered)" if outcome.additional_papers else "")
            trace.bullet(
                f"Resolution for {c.id}: {outcome.status.value} — "
                f"{outcome.explanation[:160]}{extra}",
                agent="resolution")
            return c

        resolved = await asyncio.gather(*(one(c) for c in contradictions))
        state.contradictions = list(resolved)

    async def _synthesize(self, state: V3RunState, trace: Any) -> Any:
        try:
            return await asyncio.wait_for(
                self.synthesizer.synthesize(state),
                timeout=self._timeout("agentic_v3_synthesis_timeout", 180.0),
            )
        except TimeoutError:
            from src.agents.synthesize import SynthesisReport
            return SynthesisReport(
                summary="(analysis timed out; see evidence and gaps below)",
                confidence=0.0,
            )
        except Exception as exc:
            logger.warning("synthesis failed (%s); empty honest answer", exc)
            from src.agents.synthesize import SynthesisReport
            return SynthesisReport(
                summary="(synthesis failed; evidence and gaps are reported "
                        "below for manual review)",
                confidence=0.0,
            )

    # ------------------------------------------------------------------
    def _result_dict(self, state: V3RunState, reports: list[WorkerReport]) -> dict[str, Any]:
        # the final evidence set is the VERIFIED set (requirement 7)
        evidence = state.verified_evidence()
        confidence = (state.final_answer or {}).get("confidence", 0.0)
        # actual resource usage comes from the per-worker budget copies
        usage = {
            "searches_used": sum(int(r.searches_used or 0) for r in reports),
            "deep_inspections_used": sum(
                int(r.deep_inspections_used or 0) for r in reports),
            "workers": len(reports),
        }
        if not confidence and evidence:
            confidence = round(
                sum(float(e.confidence or 0.0) for e in evidence) / len(evidence), 4)
        return {
            "run_id": state.run_id,
            "question": state.question,
            "mode": "agentic-v3",
            "terminal": state.terminal,
            "stop_reason": state.stop_reason,
            "budget": state.budget.model_dump(mode="json"),
            "budget_usage": usage,
            "plan": state.plan.model_dump(mode="json") if state.plan else {},
            "workers": [r.to_dict() for r in reports],
            "tasks": [t.summary() for t in state.tasks],
            "evidence": [e.to_dict() for e in evidence],
            "gaps": [r.gap for t in state.tasks
                     for r in t.evidence_requirements if r.gap],
            "contradictions": [
                {
                    "id": c.id,
                    "claim": c.claim,
                    "kind": c.kind.value,
                    "requirement_id": c.requirement_id,
                    "evidence_a": c.evidence_a,
                    "evidence_b": c.evidence_b,
                    "description": c.description,
                    "resolution": c.resolution.model_dump(mode="json")
                    if c.resolution else None,
                }
                for c in state.contradictions
            ],
            "resolved": [
                c.id for c in state.contradictions
                if c.resolution and c.resolution.status == ResolutionStatus.RESOLVED
            ],
            "unresolved": [
                c.id for c in state.contradictions
                if not c.resolution
                or c.resolution.status == ResolutionStatus.UNRESOLVED
            ],
            "confidence": confidence,
            "answer": state.final_answer,
        }
