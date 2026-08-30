"""Agentic v2 — the research state machine (progress-guaranteed loop).

The orchestrator remains agentic (it proposes ONE action per turn), but the
runtime is now AUTHORITATIVE:

    phase = determine_phase(state)             # EXPLORE/EXCAVATE/ASSESS/ANSWER
    decision = orchestrator.decide(state)      # LLM proposes
    decision = policy.validate_or_repair(...)  # runtime enforces legality/no-repeat
    before  = progress_snapshot(state)
    result  = execute_with_timeout(decision)   # bounded async
    after   = progress_snapshot(state)
    progress = evaluate_progress(decision, before, after, result)
    record_progress(...)                        # strategy history
    if not progress: mark_strategy_exhausted()  # force a pivot next turn

Invariants (enforced here, never trusted to the LLM):
  * every non-terminal turn either makes progress, exhausts a strategy, or
    terminates (budget / no legal action / timeout);
  * illegal or already-exhausted decisions are repaired deterministically;
  * global-retrieval and iteration budgets are hard boundaries;
  * top-level awaits are wrapped in ``asyncio.wait_for`` timeouts.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict, Optional

from src.agentic_v2.actions import ActionExecutor
from src.agentic_v2.events import EventEmitter
from src.agentic_v2.orchestrator import ActionDecision, ActionType, OrchestratorAgent
from src.agentic_v2.policy import (
    determine_phase,
    evaluate_progress,
    fallback_action,
    legal_actions,
    progress_snapshot,
    strategy_key,
    validate_or_repair,
)
from src.agentic_v2.state import ActionType, ObjectiveStatus, ResearchState, StrategyAttempt

logger = logging.getLogger("src.agentic_v2.pipeline")


class AgenticV2Pipeline:
    """Runs the progress-guaranteed research state machine for one question."""

    def __init__(
        self,
        config: Any = None,
        *,
        orchestrator: Optional[Any] = None,
        executor: Optional[ActionExecutor] = None,
        events: Optional[EventEmitter] = None,
    ):
        from src.config import AppConfig

        self.config = config or AppConfig()
        self.orchestrator = orchestrator or OrchestratorAgent(config=self.config)
        self.executor = executor or ActionExecutor(config=self.config)
        self.events = events
        if self.events is None and getattr(self.config, "agentic_v2_events_file", ""):
            self.events = EventEmitter(self.config.agentic_v2_events_file)
        if self.events is not None:
            self.executor.events = self.events

    def _emit(self, type_: str, **fields: Any) -> None:
        if self.events is not None:
            self.events.emit(type_, **fields)

    def _orchestrator_timeout(self) -> float:
        return float(getattr(self.config, "agentic_v2_orchestrator_timeout", 120.0))

    def _action_timeout(self, decision: Any = None,
                        state: Optional[ResearchState] = None) -> float:
        """Base per-action budget; VERIFY scales with its passage count.

        VERIFY judges ONE paragraph per sequential LLM call (full unit text),
        so a flat budget would kill it mid-loop ("timeout after 180s"). Give
        each candidate passage its own base-timeout slot instead.
        """
        base = float(getattr(self.config, "agentic_v2_action_timeout", 180.0))
        if (decision is not None and state is not None
                and decision.action == ActionType.VERIFY):
            n_candidates = getattr(self.executor, "n_verify_candidates", None)
            n = n_candidates(state, decision.objective_id) if n_candidates else 1
            return base * max(1, int(n))
        return base

    def _new_state(self, query: str) -> ResearchState:
        return ResearchState(
            question=query,
            max_rounds=int(getattr(self.config, "agentic_v2_max_rounds", 12)),
            max_global_retrieves=int(getattr(self.config, "agentic_v2_max_global_retrieves", 6)),
        )

    async def answer(self, query: str) -> Dict[str, Any]:
        """Run the state machine to completion; return a JSON-safe result dict.

        Wraps the loop in a top-level Logfire span; with Logfire disabled this
        is a thin passthrough.
        """
        from src import logfire_obs as lf

        lf.ensure_configured(self.config)
        if lf.enabled():
            async with lf.run_span(query=query):
                return await self._answer(query)
        return await self._answer(query)

    async def _answer(self, query: str) -> Dict[str, Any]:
        from src import logfire_obs as lf
        from src.trace import get_trace

        trace = get_trace()
        state = self._new_state(query)
        trace.stage("AGENTIC V2 - STATE MACHINE LOOP")
        trace.bullet(f"question: {query!r} | max_rounds={state.max_rounds} "
                     f"| max_global_retrieves={state.max_global_retrieves}")
        self._emit("run_start", question=query, max_rounds=state.max_rounds,
                   max_global_retrieves=state.max_global_retrieves,
                   code_rev="no-full-document@1")

        # Route LLM calls (prompt/thought/response) into the event stream for
        # the UI, and track the current iteration via a contextvar. With
        # Logfire enabled the LLM spans go to Logfire instead (PydanticAI's
        # own instrumentation), so the UI observer is not installed.
        observer_installed = self.events is not None and not lf.enabled()
        if observer_installed:
            from src.agentic_v2 import llm_observer
            llm_observer.install(self.events)

        try:
            await self._run_loop(state, trace)
        finally:
            if observer_installed:
                from src.agentic_v2 import llm_observer
                llm_observer.uninstall()

        trace.bullet(
            f"loop finished: terminal={state.terminal} reason={state.stop_reason!r} "
            f"iterations={state.iteration} evidence={len(state.evidence)}"
        )
        if lf.enabled() and state.final_answer is not None:
            lf.final_answer(state.final_answer)
        self._emit("run_end", stop_reason=state.stop_reason, iterations=state.iteration,
                   terminal=state.terminal, confidence=state.confidence,
                   phase=state.phase,
                   answer=(state.final_answer or {}).get("summary", ""))
        return self._result_dict(state)

    async def _run_loop(self, state: ResearchState, trace: Any) -> None:
        """The main state-machine loop (extracted for observer try/finally)."""
        last_phase: Optional[str] = None
        while not state.terminal:
            # 1. iteration budget is a hard boundary (never continue past it).
            if state.iteration >= state.max_rounds:
                trace.bullet(f"budget exhausted at iteration {state.iteration}")
                break

            # 2. derive phase + legal actions (the runtime's view).
            phase = determine_phase(state)
            if phase.value != last_phase:
                state.phase = phase.value
                self._emit("phase_changed", iteration=state.iteration, phase=phase.value)
                last_phase = phase.value
            legal = legal_actions(state)

            state.iteration += 1
            if self.events is not None:
                from src.agentic_v2 import llm_observer
                llm_observer.set_iteration(state.iteration)

            # 3. LLM proposes (with timeout); hard failure -> terminal.
            decision = await self._decide(state, phase, legal, trace)
            if decision is None:
                break

            # 4. authoritative policy validation / repair.
            original_action = decision.action.value
            decision, repaired, reason = validate_or_repair(decision, state,
                                                            phase=phase, legal=legal)
            if repaired:
                self._emit("policy_repair", iteration=state.iteration,
                           original_action=original_action,
                           repaired_action=decision.action.value,
                           reason=reason)

            # 5. record + snapshot BEFORE.
            state.record_action(decision)
            trace.bullet(
                f"iter {state.iteration}: {decision.action.value}"
                f"{('[' + decision.objective_id + ']') if decision.objective_id else ''} "
                f"— {decision.rationale[:100]}"
            )
            self._emit("decision", iteration=state.iteration,
                       action=decision.action.value, objective_id=decision.objective_id,
                       rationale=decision.rationale, query=decision.query,
                       instructions=decision.instructions)
            self._emit("action_start", iteration=state.iteration,
                       action=decision.action.value, objective_id=decision.objective_id)
            before = progress_snapshot(state)

            # 6. execute (bounded async).
            result = await self._execute(decision, state, trace)

            after = progress_snapshot(state)

            # 7. evaluate progress against the before/after snapshot.
            progress = evaluate_progress(decision, before, after, result)
            key = strategy_key(decision, state)
            state.record_strategy(StrategyAttempt(
                iteration=state.iteration,
                objective_id=decision.objective_id,
                action=decision.action.value,
                query=decision.query,
                document_id=decision.document_id,
                strategy_key=key,
                progress_made=progress.progress,
                progress_summary=progress.summary,
            ))
            self._emit("progress", iteration=state.iteration, action=decision.action.value,
                       objective_id=decision.objective_id, progress=progress.progress,
                       summary=progress.summary, strategy_key=key)

            # 8. no progress -> exhaust the strategy and force a pivot.
            if not progress.progress:
                state.exhaust_strategy(key)
                self._emit("no_progress", iteration=state.iteration,
                           action=decision.action.value, strategy_key=key,
                           reason=progress.summary)
                self._emit("strategy_exhausted", iteration=state.iteration,
                           action=decision.action.value, strategy_key=key)
                trace.bullet(f"  -> NO PROGRESS ({progress.summary}); strategy exhausted: {key}")

            # 9. outcome + state events.
            self._emit("action_done", iteration=state.iteration,
                       action=decision.action.value, objective_id=decision.objective_id,
                       status=(result.status if result else "failed"),
                       outcome=(result.summary if result else state.prior_actions[-1].outcome))
            self._emit("state", iteration=state.iteration, **self._state_snapshot(state))

        # ALWAYS end with a synthesized answer — after budget exhaustion AND
        # after a normal orchestrator STOP.
        await self._finalize_answer(state, trace)

    # ------------------------------------------------------------------
    async def _decide(self, state: ResearchState, phase: Any, legal: Any, trace: Any):
        """Propose an action with a timeout; timeouts pivot, hard errors terminate."""
        try:
            decision = await asyncio.wait_for(
                self.orchestrator.decide(state),
                timeout=self._orchestrator_timeout(),
            )
            return decision
        except asyncio.TimeoutError:
            timeout = self._orchestrator_timeout()
            self._emit("timeout", iteration=state.iteration, action="DECIDE",
                       error=f"orchestrator timed out after {timeout}s")
            self._emit("failure", iteration=state.iteration, action="DECIDE",
                       error="orchestrator timeout")
            decision = ActionDecision(
                action=fallback_action(state),
                rationale=f"orchestrator timed out after {timeout}s; deterministic fallback",
            )
            self._emit("policy_repair", iteration=state.iteration,
                       original_action="(timeout)", repaired_action=decision.action.value,
                       reason="orchestrator timeout")
            trace.bullet(f"  -> orchestrator timeout; fallback {decision.action.value}")
            return decision
        except Exception as exc:
            logger.warning("orchestrator decision failed: %s", exc)
            state.terminal = True
            state.stop_reason = f"orchestrator failed: {str(exc)[:200]}"
            self._emit("failure", iteration=state.iteration, action="DECIDE",
                       error=str(exc)[:200])
            return None

    async def _execute(self, decision: ActionDecision, state: ResearchState, trace: Any):
        """Execute an action with a timeout; failures become no-progress outcomes."""
        try:
            result = await asyncio.wait_for(
                self.executor.execute(decision, state),
                timeout=self._action_timeout(decision, state),
            )
            state.finish_last_action(status=result.status, outcome=result.summary)
            trace.bullet(f"  -> {result.summary[:180]}")
            return result
        except asyncio.TimeoutError:
            timeout = self._action_timeout()
            state.finish_last_action(status="failed", outcome=f"timeout after {timeout}s")
            trace.bullet(f"  -> TIMEOUT after {timeout}s: {decision.action.value}")
            self._emit("timeout", iteration=state.iteration, action=decision.action.value,
                       error=f"action timed out after {timeout}s")
            self._emit("failure", iteration=state.iteration, action=decision.action.value,
                       error=f"timeout after {timeout}s")
            return None
        except Exception as exc:
            logger.warning("action %s failed: %s", decision.action, exc)
            state.finish_last_action(status="failed", outcome=str(exc)[:200])
            trace.bullet(f"  -> FAILED: {str(exc)[:180]}")
            self._emit("failure", iteration=state.iteration, action=decision.action.value,
                       error=str(exc)[:200])
            return None

    # ------------------------------------------------------------------
    @staticmethod
    def _state_snapshot(state: ResearchState) -> Dict[str, Any]:
        return {
            "phase": state.phase,
            "objectives": [
                {"id": o.id, "statement": o.statement, "status": o.status.value,
                 "gap": o.gap}
                for o in state.objectives
            ],
            "documents": len(state.documents),
            "evidence": len(state.evidence),
            "gaps": list(state.gaps),
            "contradictions": list(state.contradictions),
            "confidence": state.confidence,
            "exhausted_strategies": list(state.exhausted_strategies),
        }

    async def _finalize_answer(self, state: ResearchState, trace: Any) -> None:
        """Guarantee a final synthesized answer when the loop ends without one.

        Covers BOTH exits: budget exhaustion and a normal orchestrator STOP.
        Only relevant / partially relevant evidence may back the answer
        (synthesize.usable_evidence); with none, a budget stop is hard-stopped
        honestly while an already-terminal STOP keeps the orchestrator's own
        reason (e.g. "cannot answer with available tools").
        """
        from src.agentic_v2.synthesize import usable_evidence

        if state.final_answer is not None:
            return  # the loop already synthesized inside the run
        if not usable_evidence(state):
            if not state.terminal:
                self._hard_stop(state, "budget exhausted; insufficient evidence to synthesize")
            return

        why = "orchestrator stopped" if state.terminal else "budget exhausted"
        decision = ActionDecision(
            action=ActionType.SYNTHESIZE,
            rationale=f"{why}; synthesize the final answer from verified evidence",
            instructions="synthesize an honest answer noting unresolved gaps",
        )
        state.record_action(decision)
        self._emit("decision", iteration=state.iteration, action="SYNTHESIZE",
                   objective_id="", rationale=decision.rationale,
                   query="", instructions=decision.instructions)
        self._emit("action_start", iteration=state.iteration, action="SYNTHESIZE",
                   objective_id="")
        try:
            result = await asyncio.wait_for(
                self.executor.execute(decision, state),
                timeout=self._action_timeout(decision, state),
            )
            state.finish_last_action(status=result.status, outcome=result.summary)
            trace.bullet(f"  (finalize) -> {result.summary[:180]}")
            self._emit("action_done", iteration=state.iteration, action="SYNTHESIZE",
                       objective_id="", status=result.status, outcome=result.summary)
            if result.status == "failed":
                self._hard_stop(state, f"{why}; synthesis failed: {result.summary}")
        except asyncio.TimeoutError:
            self._emit("timeout", iteration=state.iteration, action="SYNTHESIZE",
                       error="synthesis timed out")
            self._hard_stop(state, f"{why}; synthesis timed out")
        except Exception as exc:
            self._emit("failure", iteration=state.iteration, action="SYNTHESIZE",
                       error=str(exc)[:200])
            self._hard_stop(state, f"{why}; synthesis failed: {str(exc)[:160]}")

    @staticmethod
    def _hard_stop(state: ResearchState, reason: str) -> None:
        state.terminal = True
        state.stop_reason = reason

    def _result_dict(self, state: ResearchState) -> Dict[str, Any]:
        return {
            "question": state.question,
            "iterations": state.iteration,
            "terminal": state.terminal,
            "stop_reason": state.stop_reason,
            "phase": state.phase,
            "objectives": [
                {
                    "id": o.id,
                    "statement": o.statement,
                    "status": o.status.value,
                    "gap": o.gap,
                    "caveats": o.caveats,
                }
                for o in state.objectives
            ],
            "documents": [
                {"document_id": d.document_id, "chunk_id": d.chunk_id, "section": d.section}
                for d in state.documents
            ],
            "evidence": [e.model_dump(mode="json") for e in state.evidence],
            "gaps": state.gaps,
            "contradictions": state.contradictions,
            "confidence": state.confidence,
            "actions": [
                {
                    "iteration": a.iteration,
                    "action": a.action.value,
                    "objective_id": a.objective_id,
                    "status": a.status,
                    "outcome": a.outcome,
                }
                for a in state.prior_actions
            ],
            "strategy_history": [
                {
                    "iteration": s.iteration,
                    "objective_id": s.objective_id,
                    "action": s.action,
                    "query": s.query,
                    "strategy_key": s.strategy_key,
                    "progress_made": s.progress_made,
                    "progress_summary": s.progress_summary,
                }
                for s in state.strategy_history
            ],
            "exhausted_strategies": list(state.exhausted_strategies),
            "answer": state.final_answer,
        }
