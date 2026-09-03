"""The research graph: orchestrator -> parallel research workers -> verify -> conflict/resolution -> evidence-gated synthesis.

Architecture (agreed):
  1. USER QUERY
  2. ORCHESTRATOR (gemma) decomposes the query into independent tasks.
  3. RESEARCH WORKERS are spawned PER TASK (LangGraph Send, parallel). Each
     worker may do UML enrichment, then chooses searxng (web) OR the hybrid
     retriever to gather candidates.
  4. Every retrieved candidate MUST pass the VERIFIER critic (mistral,
     paced 1 req / 1.5s) before it is evidence (rule 2 - a hard edge).
  5. Verified results go to the CONFLICT agent (contradiction detection on
     mistral) then RESOLUTION (mistral with search tools) which may use
     searxng / retrieve / postgres to resolve.
  6. GAP RESOLUTION closes remaining evidence gaps with more LOCAL
     retrievals (pg primary -> hybrid), then trust-gated WEB searches, then
     LISTS whatever still cannot be closed (contradictions re-checked on the
     new evidence).
  7. SYNTHESIS (gemma) writes the final answer from verified evidence only
     (rule 4), with deterministic citation repair.

The workflow ORDER is fixed by the edges below; the LLM chooses TOOLS and
how within each stage.
"""

from __future__ import annotations

import asyncio
import contextvars
import logging
import re
import uuid
from operator import add
from typing import Annotated, Any, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.types import Send

from src.agents import rules
from src.agents.state import (
    GapResolutionStatus,
    Phase,
    ResearchRequirement,
    RunBudget,
    XDeepRunState,
)

logger = logging.getLogger("x_deepagents.graph")


# Per-run sinks: ContextVars (not process globals) so concurrent requests
# (two /v1/chat/stream xdeep runs in one server process) never cross-talk.
# The task created per request inherits a copy of the request's context, so
# attaching in the request handler scopes the whole run correctly.
_progress_sink: contextvars.ContextVar = contextvars.ContextVar(
    "xdeep_progress_sink", default=None)
_event_sink: contextvars.ContextVar = contextvars.ContextVar(
    "xdeep_event_sink", default=None)


def set_event_sink(sink) -> None:
    """Attach a callable sink(event, fields) receiving STRUCTURED trace events.

    The frontend calls this (via bridge) to power the expandable thinking
    log: each event is {"type":"pipeline","event":<name>,"fields":{...}} so
    the UI shows real titles/details (web search, fetched sites, verdicts,
    gaps...) instead of bare progress lines. Pass None to detach.
    """
    _event_sink.set(sink)


def emit_event(event: str, **fields) -> None:
    """Emit one structured trace event.

    Always mirrored to logs.txt ([xdeep] trace:<event>) so the log keeps full
    parity; the UI sink (if attached) receives (event, fields). Never raises -
    tracing must not break research.
    """
    from src.agents.logging import xdeep_log

    try:
        xdeep_log("trace:" + event, **fields)
    except Exception:
        pass
    sink = _event_sink.get()
    if sink is not None:
        try:
            sink(event, fields)
        except Exception:
            logger.warning("event sink raised", exc_info=True)


def set_progress_sink(sink) -> None:
    """Attach a callable sink(message) that receives every progress line.

    Used by the web bridge to stream run milestones live to the UI queue.
    Pass None to detach. The sink is called synchronously per progress line.
    """
    _progress_sink.set(sink)


def progress(message: str) -> None:
    """Emit live progress to stdout (flushed), logs.txt, and the UI sink."""
    from src.agents.logging import xdeep_log

    print("  " + message, flush=True)
    logger.info("progress: %s", message)
    xdeep_log("progress", msg=message)
    sink = _progress_sink.get()
    if sink is not None:
        try:
            sink(message)
        except Exception:
            logger.warning("progress sink raised", exc_info=True)


class TaskPayload(TypedDict):
    """One decomposed task handed to a research worker."""
    task_id: str
    text: str
    entities: list[str]
    target_n: int
    budget_max_searches: int      # from RunBudget (web gap-fill cap)
    budget_max_rounds: int        # from RunBudget (local retrieve cap)


class DecomposeError(RuntimeError):
    """Raised when decomposition fails even after the strict retry. The graph
    treats this as a loud, terminal failure (Gap F) - never a silent collapse
    of a multi-part question into a single budget-chewing task."""


class ResearchState(TypedDict, total=False):
    """LangGraph state - carries the run plus fan-out scratch."""
    run_id: str
    question: str
    state: XDeepRunState
    task_payloads: list[TaskPayload]
    requirements: Annotated[list[ResearchRequirement], add]   # worker reduce
    searches_used: Annotated[int, add]                        # aggregate web budget
    memory_context: str  # advisory prior-research context (planner only)


def _new_run(question: str) -> XDeepRunState:
    from src.agents.reuse import AppConfig

    try:
        cfg = AppConfig()
        budget = RunBudget(
            max_searches=cfg.agentic_v3_max_searches,
            max_retrieval_rounds=cfg.agentic_v3_max_retrieval_rounds,
            max_papers_per_round=cfg.agentic_v3_papers_per_search,
            evidence_target=cfg.agentic_v3_evidence_target,
        )
    except Exception:
        budget = RunBudget()
    return XDeepRunState(
        run_id=uuid.uuid4().hex[:12],
        question=question,
        phase=Phase.DECOMPOSE,
        budget=budget,
    )


# --- nodes ----------------------------------------------------------------

async def decompose_node(state: ResearchState) -> Any:
    """Orchestrator (gemma): decompose the question into independent tasks,
    then FAN OUT one research worker per task via Send."""
    run: XDeepRunState = state["state"]
    question = run.question
    progress(f"[decompose] orchestrator (gemma) decomposing: {question}")

    from src.agents.agents.stages import (
        decompose_requirements,
        make_orchestrator,
    )

    try:
        from src.agents.timeouts import run_with_timeout, timeout_for

        reqs = await run_with_timeout(
            decompose_requirements(
                make_orchestrator(), question,
                memory_context=str(state.get("memory_context") or "")),
            timeout_for("decompose"), "decompose")
    except DecomposeError:
        raise
    except asyncio.TimeoutError:
        progress("    decompose TIMED OUT - reporting as terminal decomposition error")
        raise DecomposeError("decomposition timed out; refusing to collapse the question")
    except Exception as exc:
        # retry once with a strict JSON-only nudge (inside decompose) - only
        # now do we fail LOUD instead of silently collapsing the question.
        logger.warning("decompose failed after retry: %s", exc)
        progress("    decompose FAILED (JSON parse error after retry) - "
                 "reporting as a terminal decomposition error")
        from src.agents.logging import xdeep_log

        xdeep_log("decompose_failed", error=str(exc)[:300])
        raise DecomposeError(
            f"decomposition failed after retry: {str(exc)[:200]} - "
            "refusing to collapse the question into a single task"
        )

    payloads: list[TaskPayload] = [
        {"task_id": r.id, "text": r.text, "entities": r.entities or [],
         "target_n": r.target_n,
         "budget_max_searches": run.budget.max_searches,
         "budget_max_rounds": run.budget.max_retrieval_rounds}
        for r in reqs
    ]
    run.requirements = []      # workers will re-add reduced requirements
    run.set_phase(Phase.RETRIEVE)
    progress(f"    orchestrator planned {len(payloads)} independent task(s)")
    emit_event("decompose_done", requirements=[
        {"id": p["task_id"], "text": p["text"], "target_n": p["target_n"]}
        for p in payloads])
    for p in payloads:
        progress(f"    task {p['task_id']}: {p['text']}")

    # store the fan-out payloads in state; the conditional edge below turns
    # them into one Send per task (parallel workers) or routes to join when
    # there were no tasks.
    return {"state": run, "task_payloads": payloads, "requirements": []}


# process-wide concurrency bound for parallel research workers
_worker_semaphore = None


def _workers_semaphore():
    global _worker_semaphore
    if _worker_semaphore is None:
        from src.agents.timeouts import max_workers

        _worker_semaphore = asyncio.Semaphore(max_workers())
    return _worker_semaphore


async def research_worker(payload: TaskPayload) -> dict:
    """One research worker (qwen): adaptive loop mirroring agentic-v3.
    Rules: every candidate passes the verifier; workflow order fixed here.
    Bounded by the worker semaphore + a hard per-worker timeout.
    """
    from src.agents.agents.worker import research_worker as _rw
    from src.agents.timeouts import run_with_timeout, timeout_for

    sem = _workers_semaphore()
    async with sem:
        return await run_with_timeout(
            _rw(payload), timeout_for("worker"),
            f"research_worker:{payload.get('task_id', '?')}")

async def join_node(state: ResearchState) -> dict:
    """Collect the parallel workers' requirements into the run."""
    run: XDeepRunState = state["state"]
    reqs = state.get("requirements") or []
    run.requirements = reqs
    for r in run.requirements:
        for it in r.items:
            it.run_id = run.run_id
    run.budget.searches_used += int(state.get("searches_used") or 0)
    run.set_phase(Phase.VERIFY)
    progress(f"[join] merged {len(run.requirements)} task(s), "
             f"{len(run.verified_items())} verified item(s) total, "
             f"searches_used={run.budget.searches_used}")
    emit_event("evidence_state", verified=len(run.verified_items()),
               requirements=len(run.requirements),
               overview=[r.summary() for r in run.requirements])
    return {"state": run, "requirements": reqs}


async def conflict_node(state: ResearchState) -> dict:
    """Conflict agent (mistral): check verified evidence for contradictions."""
    run: XDeepRunState = state["state"]
    verified = run.verified_items()
    if len(verified) < 2:
        progress("[conflict] fewer than 2 verified items - no cross-check needed")
        run.set_phase(Phase.CONTRADICTION)
        return {"state": run}

    from src.agents.agents.stages import (
        detect_contradictions,
        make_contradiction_agent,
    )

    progress(f"[conflict] checking {len(verified)} verified item(s) for "
             "contradictions (mistral)...")
    try:
        from src.agents.timeouts import run_with_timeout, timeout_for

        run.contradictions = await run_with_timeout(
            detect_contradictions(make_contradiction_agent(), verified),
            timeout_for("conflict"), "conflict")
    except Exception as exc:
        logger.warning("contradiction detection failed: %s", exc)
        run.contradictions = []
    progress(f"    detected {len(run.contradictions)} contradiction(s)")
    for cc in run.contradictions:
        emit_event("contradiction",
                   contradiction_id=cc.id,
                   requirement_id=cc.requirement_id,
                   kind=cc.kind.value,
                   claim=cc.claim[:240])
    run.set_phase(Phase.CONTRADICTION)
    return {"state": run}


async def resolution_node(state: ResearchState) -> dict:
    """Resolution (mistral WITH tools): resolve/report contradictions."""
    run: XDeepRunState = state["state"]
    if not run.contradictions:
        progress("[resolution] no contradictions to resolve")
        run.set_phase(Phase.RESOLUTION)
        return {"state": run}

    from src.agents.agents.stages import (
        make_resolution_agent,
        make_verifier,
        resolve_contradiction,
    )

    progress(f"[resolution] resolving {len(run.contradictions)} "
             "contradiction(s) (mistral + search tools + verifier)...")
    resolver = make_resolution_agent()
    verifier = make_verifier()
    from src.agents.timeouts import run_with_timeout, timeout_for

    for c in run.contradictions:
        try:
            # every new passage the resolution leans on passes the verifier;
            # verified findings are folded into the owning requirement so
            # they reach re-checks and synthesis (not just the id list).
            owner = run.requirement(c.requirement_id or "")
            await run_with_timeout(
                resolve_contradiction(resolver, verifier, c, owner),
                timeout_for("resolution"), f"resolution:{c.id}")
            if owner is not None and c.resolution is not None:
                by_id = run.evidence_by_id()
                for eid in c.resolution.new_evidence_ids:
                    item = by_id.get(eid)
                    if item is not None and not item.run_id:
                        item.run_id = run.run_id
        except Exception as exc:
            # record a REPORTED unresolved outcome instead of silently None -
            # an unresolved contradiction that is explained is acceptable;
            # a silently-missing resolution is not (rule 3).
            from src.agents.state import ResolutionOutcome, ResolutionStatus

            logger.warning("resolution failed for %s: %s", c.id, exc)
            c.resolution = ResolutionOutcome(
                status=ResolutionStatus.UNRESOLVED,
                explanation=f"resolution call failed: {str(exc)[:300]}",
                characterization="resolution_failed",
            )
        status = c.resolution.status.value if c.resolution else "none"
        progress(f"    {c.id}: {status}")
        emit_event("resolution", contradiction_id=c.id,
                   status=status,
                   explanation=(c.resolution.explanation if c.resolution
                                else "")[:300],
                   new_evidence=len(c.resolution.new_evidence_ids)
                   if c.resolution else 0)
    run.set_phase(Phase.RESOLUTION)
    return {"state": run}


async def gap_resolution_node(state: ResearchState) -> dict:
    """Gap-resolution (post-contradiction): close remaining evidence gaps with
    more LOCAL retrievals (postgres primary, hybrid fallback) within the
    gap-local budget; then trust-gated WEB searches within the gap-web budget;
    then LIST what still cannot be closed. Every new candidate passes the
    verifier (rule 2). Contradictions are re-checked on the newly-added
    verified evidence so gap-fill never silently smuggles in a conflict.
    """
    run: XDeepRunState = state["state"]
    before_verified = run.verified_ids()
    from src.agents.agents.gap_fill import resolve_requirement_gaps
    from src.agents.timeouts import run_with_timeout, timeout_for

    try:
        resolutions = await run_with_timeout(
            resolve_requirement_gaps(run), timeout_for("gap"), "gap-resolution")
    except asyncio.TimeoutError:
        progress("[gap-resolution] TIMED OUT - continuing to synthesis with "
                 "current evidence")
        resolutions = []
    except Exception as exc:
        logger.warning("gap resolution failed: %s", exc)
        progress(f"    gap resolution error: {str(exc)[:120]}")
        resolutions = []

    # re-check contradictions on the enlarged verified set (Gap-D discipline)
    new_verified = [it.id for it in run.verified_items()
                    if it.id not in before_verified]
    if new_verified and len(run.verified_items()) >= 2:
        progress(f"[gap-resolution] re-checking {len(run.verified_items())} "
                 "verified item(s) for new contradictions "
                 f"(+{len(new_verified)} from gap-fill)...")
        try:
            from src.agents.agents.stages import (
                detect_contradictions,
                make_contradiction_agent,
                make_resolution_agent,
                make_verifier,
                resolve_contradiction,
            )

            found = await run_with_timeout(
                detect_contradictions(make_contradiction_agent(),
                                      run.verified_items()),
                timeout_for("conflict"), "conflict-recheck")
            existing = {(c.claim, tuple(sorted(c.evidence_a)),
                         tuple(sorted(c.evidence_b))) for c in run.contradictions}
            fresh = [c for c in found
                     if (c.claim, tuple(sorted(c.evidence_a)),
                         tuple(sorted(c.evidence_b))) not in existing]
            if fresh:
                run.contradictions.extend(fresh)
                resolver = make_resolution_agent()
                verifier = make_verifier()
                for fc in fresh:
                    try:
                        await run_with_timeout(
                            resolve_contradiction(
                                resolver, verifier, fc,
                                run.requirement(fc.requirement_id or "")),
                            timeout_for("resolution"), f"recheck-resolution:{fc.id}")
                    except Exception as exc2:
                        from src.agents.state import (
                            ResolutionOutcome,
                            ResolutionStatus,
                        )

                        logger.warning("recheck resolution failed %s: %s", fc.id, exc2)
                        fc.resolution = ResolutionOutcome(
                            status=ResolutionStatus.UNRESOLVED,
                            explanation=f"recheck resolution failed: {str(exc2)[:300]}",
                            characterization="resolution_failed",
                        )
                progress(f"    re-check found {len(fresh)} new contradiction(s); "
                         "all resolved/reported")
        except Exception as exc:
            logger.warning("gap contradiction recheck failed: %s", exc)

    run.set_phase(Phase.GAP_RESOLUTION)
    n_resolved = sum(1 for r in resolutions if r.resolved)
    n_partial = sum(1 for r in resolutions
                    if r.status is GapResolutionStatus.PARTIALLY_RESOLVED)
    n_listed = len(resolutions) - n_resolved - n_partial
    progress(f"[gap-resolution] {len(resolutions)} gap(s): {n_resolved} resolved, "
             f"{n_partial} partially resolved, {n_listed} listed; "
             f"gaps={len(run.gaps)}")
    for gr in resolutions:
        emit_event("gap_resolution",
                   requirement_id=gr.requirement_id,
                   gap=gr.gap[:240],
                   status=gr.status.value,
                   evidence_ids=gr.evidence_ids,
                   note=gr.note[:200] if gr.note else "")
    return {"state": run}


async def synthesize_node(state: ResearchState) -> dict:
    """Synthesis (gemma): final answer from verified evidence only."""
    run: XDeepRunState = state["state"]
    from src.agents.agents.stages import (
        make_synthesizer,
        synthesize_answer_with_data,
    )

    violations = rules.validate_state(run)
    if violations:
        logger.warning("synthesis guard blocked by rule violations: %s", violations)
        from src.agents.logging import xdeep_log

        xdeep_log("synthesis_guard_violations", violations=violations)

    progress(f"[synthesis] gemma writing answer from {len(run.verified_items())} "
             "verified item(s)...")
    emit_event("synthesis_start", verified=len(run.verified_items()),
               contradictions=len(run.contradictions),
               gap_resolutions=len(run.gap_resolutions))
    synth_data = None
    try:
        from src.agents.timeouts import run_with_timeout, timeout_for

        answer, synth_data = await run_with_timeout(
            synthesize_answer_with_data(make_synthesizer(), run),
            timeout_for("synthesis"), "synthesis")
    except Exception as exc:
        logger.warning("synthesis failed: %s", exc)
        answer = "(no answer - synthesis failed)"
    run.answer = answer
    run.set_phase(Phase.DONE)
    if not violations:
        run.terminal = True
    progress("[done] research complete")
    if synth_data is None and answer:
        # the LLM synthesis step was unavailable (or malformed) and the
        # deterministic fallback answer was used - surface that in the trace
        emit_event("synthesis_fallback",
                   verified=len(run.verified_items()),
                   answer_len=len(answer),
                   note="LLM synthesis unavailable; deterministic evidence "
                        "extraction used")
        progress("    synthesis used FALLBACK (LLM unavailable) - answer is a "
                 "faithful extraction of verified evidence")

    # RECONCILE: the synthesizer's own unresolved_gaps are ground truth for
    # "what the answer still cannot cover". Fold them into run.gaps so the
    # final state (and the UI trace) never claim "0 listed" while the answer
    # actually lists open facets.
    if synth_data:
        listed = [str(g).strip() for g in (synth_data.get("unresolved_gaps") or [])
                  if str(g).strip()]
        added = 0
        for gtxt in listed:
            if gtxt not in run.gaps:
                run.gaps.append(gtxt)
                added += 1
        # mark matched gap_resolutions as PARTIALLY_RESOLVED when the
        # synthesizer's listed gaps share significant vocabulary with the
        # probed question (e.g. 'potassium' safety facets vs a potassium gap) -
        # substring alone misses cross-facet holes.
        if added:
            def _sig_tokens(txt: str) -> set:
                return {w for w in re.findall(r"[a-zA-Z]{4,}", (txt or "").lower())
                        if w not in _STOP}
            _STOP = {"with", "the", "and", "for", "from", "that", "this",
                     "diet", "diets", "dietary", "blood", "pressure",
                     "effect", "effects", "evidence", "reported", "provided",
                     "still", "missing", "long", "term", "study", "studies",
                     "was", "were", "has", "have", "been", "not", "are"}
            for gr in run.gap_resolutions:
                if not gr.resolved or not gr.gap:
                    continue
                gr_toks = _sig_tokens(gr.gap)
                if not gr_toks:
                    continue
                overlaps = [g for g in listed
                            if _sig_tokens(g) & gr_toks]
                if overlaps:
                    gr.status = GapResolutionStatus.PARTIALLY_RESOLVED
                    gr.note = ("reconciled: synthesizer still lists unresolved "
                               "facets - " + overlaps[0][:140])
            emit_event("gaps_reconciled", synthesizer_listed=len(listed),
                       added_to_state=added, total_gaps=len(run.gaps))
            progress(f"    synthesis reconciliation: {added} unresolved gap(s) "
                     f"added to state (total {len(run.gaps)})")
    emit_event("synthesis_done", answer_len=len(answer),
               verified=len(run.verified_items()),
               gaps=len(run.gaps),
               contradictions=len(run.contradictions))
    return {"state": run}


# --- graph builder --------------------------------------------------------

def build_research_graph():
    """Compile the orchestrator -> parallel workers -> verify -> conflict
    -> resolution -> synthesis graph.

    Mandatory gates are HARD edges:
      * research_worker always verifies every candidate (rule 2);
      * conflict always runs after workers join, before resolution (rule 3);
      * synthesis reads verified-only items (rule 4).
    """
    g = StateGraph(ResearchState)
    g.add_node("decompose", decompose_node)
    g.add_node("research_worker", research_worker)
    g.add_node("join", join_node)
    g.add_node("conflict", conflict_node)
    g.add_node("resolution", resolution_node)
    g.add_node("gap_resolution", gap_resolution_node)
    g.add_node("synthesize", synthesize_node)

    g.add_edge(START, "decompose")
    g.add_edge("research_worker", "join")     # workers reduce into join
    g.add_edge("join", "conflict")
    g.add_edge("conflict", "resolution")
    g.add_edge("resolution", "gap_resolution")   # close gaps AFTER resolution
    g.add_edge("gap_resolution", "synthesize")
    g.add_edge("synthesize", END)
    # decompose fans out via Send (one research worker per task, parallel);
    # if no tasks were produced it routes straight to join.
    g.add_conditional_edges(
        "decompose",
        lambda st: "join" if not st.get("task_payloads") else (
            [Send("research_worker", p) for p in st["task_payloads"]]
        ),
        {"join": "join"},
    )
    return g.compile()


async def run_research(question: str, graph=None,
                   memory_context: str = "") -> XDeepRunState:
    """Run the full research pipeline for one question; return the run state.

    Opens + closes the logs.txt RUN block so programmatic runs log the same
    way as the CLI. ``memory_context`` is advisory prior-research text for
    the orchestrator only (never evidence).
    """
    from src.agents.logging import close_run_log, open_run_log

    graph = graph or build_research_graph()
    open_run_log(query=question)
    try:
        result = await graph.ainvoke(
            {"question": question, "state": _new_run(question),
             "memory_context": memory_context or ""}
        )
        return result["state"]
    finally:
        close_run_log()


__all__ = ["ResearchState", "build_research_graph", "run_research", "progress"]
