"""Master orchestrator deep agent: delegates, decides what's next.

The orchestrator owns the final answer but never plans itself: it
spawns the dedicated planner subagent first (``spawn_subagent`` with
``depth="plan"``), publishes that plan (``submit_plan``), spawns sub
deep/shallow agents per task, reads their findings, and either spawns
follow-ups, uses a direct tool call, or finalizes. Spawning is
budgeted (``MAX_SUBAGENTS``) with graceful degradation: a refused
spawn tells the model to finish with what it has.

The orchestrator carries the full research toolkit for quick single
checks, but the system prompt forbids it from running multi-round
retrieval loops itself — bounded research belongs to sub-agents, whose
fresh contexts keep the mission board lean.
"""

from __future__ import annotations

import functools
from pathlib import Path
from typing import Any, Literal

from pydantic_ai import Agent, RunContext

from src.budget.capability import BudgetCapability, budget_status
from src.budget.prompt import BUDGET_SYSTEM_SECTION
from src.agents.synthesizer import (
    build_agent as build_synthesizer,
    build_synthesis_message as build_synth_message,
)
from src.agents.clarify import ask_user
from src.runstate.store import get_store
from src.runstate.tools import check_gaps, run_progress
from src.llm.models import agent_endpoint, make_model, small_endpoint
from src.middleware.llm_as_a_judge import LLMAsJudge
from src.tools.retrieval import local_search
from src.tools.firecrawl import web_search
from src.tools.umls import DeepDeps, lookup_medical_term

PROMPT_FILE = Path(__file__).parent.parent / "prompts" / "orchestrator.txt"

# Hard bound on delegation per run; the spawn tool degrades gracefully
# past it instead of raising.
MAX_SUBAGENTS = 8

# A shallow sub-agent has no tools: single quick answer from the task plus
# whatever context the orchestrator hands it.
SHALLOW_SYSTEM = (
    "You are a focused research assistant. Answer the task concisely "
    "from the task description, any context given, and your own knowledge. "
    "No tools are available — do not ask for any. If the context already "
    "contains the answer, base it on that."
)


@functools.lru_cache(maxsize=1)
def load_orchestrator_prompt() -> str:
    return PROMPT_FILE.read_text(encoding="utf-8")


def normalize_plan_items(items: Any) -> list[dict]:
    """Coerce a submit_plan payload into plan-card items.

    Lenient on purpose: the model may send strings, extra keys, missing
    ids, or ``deep_research`` instead of ``depth``.
    """
    if isinstance(items, dict):
        items = items.get("items", [])
    if not isinstance(items, list):
        return []
    norm: list[dict] = []
    for i, raw in enumerate(items):
        if isinstance(raw, str):
            raw = {"question": raw}
        if not isinstance(raw, dict):
            continue
        question = str(raw.get("question") or "").strip()
        if not question:
            continue
        depth = raw.get("depth", raw.get("deep_research", "shallow"))
        if isinstance(depth, bool):
            deep = depth
        else:
            deep = str(depth or "").strip().lower() == "deep"
        reqs: list[dict] = []
        raw_reqs = raw.get("evidence_requirements", [])
        if isinstance(raw_reqs, list):
            for r in raw_reqs:
                if not isinstance(r, dict) or not r.get("id"):
                    continue
                reqs.append({
                    "id": str(r["id"]),
                    "description": str(r.get("description", ""))[:500],
                })
        norm.append({
            "id": str(raw.get("id") or f"T{i + 1}"),
            "question": question,
            "deep_research": deep,
            "evidence_requirements": reqs,
        })
    return norm


def build_orchestrator(*, spawn_impl, plan_impl=None,
                      synthesize_impl=None) -> Agent[DeepDeps, str]:
    """Build the master orchestrator on the big thinking lane.

    ``spawn_impl`` runs one delegation
    (``await spawn_impl(task, depth, context, task_id, deps,
    budget=None) -> findings``, where ``budget`` is the orchestrator-chosen
    per-sub-agent allocation); ``plan_impl`` records the plan
    (``await plan_impl(items) -> ack``); ``synthesize_impl`` writes the
    final answer (``await synthesize_impl(question, chat_id) -> str``).
    Injected so tests can fake delegation without LLMs.
    """
    async def submit_plan(ctx: RunContext[DeepDeps],
                          items: list[dict[str, Any]]) -> str:
        """Publish the planner subagent's plan of action: tasks with ids
        (T1, …), questions, depth ("deep" for multi-round research,
        "shallow" otherwise), and evidence requirements. Call this with
        the planner spawn's output, before spawning research. Copy its
        items EXACTLY — same ids, same questions, same requirements. Do
        not paraphrase, expand, or add examples; you may drop whole
        items (right-sizing) but never rewrite them."""
        if plan_impl is not None:
            return await plan_impl(items)
        return f"Plan recorded ({len(items or [])} tasks). Research them now."

    async def synthesize(ctx: RunContext[DeepDeps]) -> str:
        """Write the final answer via the SYNTHESIZER agent. It carries
        the chat id and pulls the ledger itself through its own read
        tools — you pass nothing, it takes no arguments. The
        synthesizer needs no budget. GATE: allowed ONLY when the turn
        is marked gap-checked (you called ``check_gaps`` after the last
        task resolved). Otherwise this refuses and tells you to check
        gaps first. Present the returned answer as your final message."""
        chat_id = getattr(ctx.deps, "chat_id", None)
        chat = get_store().get_chat(chat_id) if chat_id else None
        turn = chat.current if chat is not None else None
        if turn is None:
            return ("[SYNTHESIZE DENIED: no research turn exists. Run the "
                    "plan → delegate → check_gaps sequence first.]")
        if not turn.gap_checked:
            return ("[SYNTHESIZE DENIED: the turn is not gap-checked. Call "
                    "`check_gaps` after all task subagents have resolved, "
                    "handle any uncovered requirements, then call "
                    "`synthesize` again.]")
        question = turn.question
        if synthesize_impl is not None:
            # Streaming path (stream adapter): the answer arrives token-by-
            # token instead of as one tool result, so the UI never flashes
            # the finished text.
            return await synthesize_impl(question, str(chat_id or ""))
        agent = build_synthesizer()
        message = build_synth_message(question, str(chat_id or ""))
        result = await agent.run(message)
        return result.output

    async def spawn_subagent(ctx: RunContext[DeepDeps], task: str,
                             depth: Literal["deep", "shallow",
                                            "plan"] = "deep",
                             context: str = "",
                             task_id: str = "",
                             max_tool_calls: int | None = None) -> str:
        """Delegate one bounded task to a fresh sub-agent that reports
        back. depth "plan" spawns the PLANNER subagent (pinned sampling;
        returns plan JSON with tasks, depth, and evidence requirements)
        — your FIRST step for a medical question, and again with a
        focused question for each follow-up. depth "deep" spawns a
        research sub-agent with the full retrieval/verification toolkit;
        "shallow" answers quickly with no tools. Pass task_id (T1, …)
        when spawning for a planned task.

        You assign this sub-agent's EXPLICIT tool-call budget via
        max_tool_calls. The allocation is carved out of YOUR master
        budget — check budget_status first and never promise more than
        you have left. Omit it to use the task default."""
        budget = {
            "max_tool_calls": max_tool_calls,
        }
        return await spawn_impl(task=task, depth=depth, context=context or "",
                                task_id=task_id or "", deps=ctx.deps,
                                budget=budget)

    model = make_model(agent_endpoint())
    return Agent(
        model,
        deps_type=DeepDeps,
        tools=[
            ask_user,
            lookup_medical_term,
            local_search,
            web_search,
            check_gaps,
            synthesize,
            submit_plan,
            spawn_subagent,
            budget_status,
            run_progress,
        ],
        # Budget first: denied spawns/tools never reach the judge
        # middleware, so denials spend no hidden LLM calls either.
        capabilities=[BudgetCapability(), LLMAsJudge()],
        system_prompt=load_orchestrator_prompt() + "\n\n" + BUDGET_SYSTEM_SECTION,
    )


def build_shallow_agent() -> Agent[None, str]:
    """Build a tool-less quick-answer sub-agent on the small lane."""
    from pydantic_ai.settings import ModelSettings

    model = make_model(small_endpoint(),
                       settings=ModelSettings(temperature=0.0))
    return Agent(model, system_prompt=SHALLOW_SYSTEM)
