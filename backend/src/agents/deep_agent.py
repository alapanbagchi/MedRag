"""Deep-research agent (qwen lane): enrich terms, gather passages.

Used for deep sub-agent legs spawned by the master orchestrator. The
leg never plans: its evidence requirements arrive pre-planned (planner
agent → ledger → task context). Flow: enrich query terms via UMLS,
retrieve passages (the verifier middleware scores them against the
requirements before they reach the agent), then answer from the
verified passages.
"""

from __future__ import annotations

import asyncio
import functools
import logging
from pathlib import Path
from typing import Any

from pydantic_ai import Agent

from src.budget.budget import BudgetManager
from src.budget.capability import BudgetCapability, budget_status
from src.budget.prompt import BUDGET_SYSTEM_SECTION
from src.runstate.tools import run_progress
from src.lib import narrate
from src.lib.pretty import style
from src.lib.streaming import console_stream
from src.lib.trace import get_trace
from src.llm.models import agent_endpoint, make_model
from src.middleware.llm_as_a_judge import LLMAsJudge
from src.tools.retrieval import local_search
from src.tools.firecrawl import web_search
from src.tools.umls import DeepDeps, build_umls_client, lookup_medical_term
from src.agents.clarify import ask_user

logger = logging.getLogger(__name__)

# Fire-and-forget warmup tasks, retained so the loop never drops them early.
_warmup_tasks: set = set()

PROMPT_DIR = Path(__file__).parent.parent / "prompts"
SYSTEM_FILE = PROMPT_DIR / "subagent_system.txt"
TASK_FILE = PROMPT_DIR / "subagent_task.txt"


WEB_SECTION_FIRECRAWL = """### C. `web_search`
**Purpose:** Finds external/current sources — search, MyBib credibility check, page scrape, and evidence judgment in one call. Returns JSON with `results` (listings), `pages` (scraped markdown), `trust` and `dropped`.
**Rules:** Run this in the same tool block as `local_search` for your opening retrieval round on every task. Snippets alone are NOT evidence: only the scraped `pages` are judged and become verified evidence. The credibility gate scores each source 0-5 and keeps only 4+ automatically — reason over the kept pages. Re-run only for a targeted follow-up on a thin requirement, with a reformulated query."""

@functools.lru_cache(maxsize=1)
def load_system_prompt() -> str:
    base = SYSTEM_FILE.read_text(encoding="utf-8")
    return base.replace("{{WEB_TOOLS}}", WEB_SECTION_FIRECRAWL)


@functools.lru_cache(maxsize=1)
def _task_prompt_template() -> str:
    return TASK_FILE.read_text(encoding="utf-8")


def render_task_prompt(question: str) -> str:
    # Plain replace (not str.format): the task file carries literal JSON
    # braces for the findings contract, which format() would misread.
    return _task_prompt_template().replace("{question}", question.strip())


def build_deep_agent() -> Agent[DeepDeps, str]:
    """Build the deep agent on the big thinking lane.

    The BudgetCapability always rides along but stays inert until the
    run's ``DeepDeps.budget`` is set (done by whoever drives the run,
    e.g. the stream adapter or ``run_deep_task``) — so attaching the
    manager to deps is the only wiring enforcement needs.
    """
    model = make_model(agent_endpoint())
    return Agent(
        model,
        deps_type=DeepDeps,
        tools=[
            ask_user,
            lookup_medical_term,
            local_search,
            web_search,
            budget_status,
            run_progress,
        ],
        # Budget first: a denied tool never reaches the judge
        # middleware, so denials spend no hidden LLM calls either.
        capabilities=[BudgetCapability(), LLMAsJudge()],
        system_prompt=load_system_prompt() + "\n\n" + BUDGET_SYSTEM_SECTION,
    )


def kick_retriever_warmup(item_id: str = "deep") -> None:
    """Start background retriever warmup (idempotent, never raises).

    Heavy weights (MedCPT encoder/cross-encoder, BM25, pg connect) load in a
    worker thread while the agent plans, so the first retrieval finds hot
    weights. The builders' lock guarantees a racing retrieval never
    double-loads. Tests pass ``warmup=False`` to run_deep_task to opt out.
    """
    try:
        from src.tools.retrieval import get_retriever

        warm = getattr(get_retriever(), "warmup", None)
        if warm is None:
            return

        def _report(stage: str) -> None:
            narrate.say(f"{style('🔥', '33')} [{item_id}] warmup: {stage}")

        def _run() -> None:
            narrate.say(f"{style('🔥', '33')} [{item_id}] warming retriever weights…")
            ready = warm(_report)
            narrate.say(f"{style('🔥', '33')} [{item_id}] warmup"
                        f" {'ready' if ready else 'partial (see warnings)'}")
            get_trace().log("retriever_warmup", item=item_id, ready=ready)

        task = asyncio.create_task(asyncio.to_thread(_run))
        _warmup_tasks.add(task)
        task.add_done_callback(_warmup_tasks.discard)
    except Exception:  # noqa: BLE001 — warmup must never break startup
        logger.exception("retriever warmup kickoff failed")


async def run_deep_task(
    question: str,
    agent: Agent[DeepDeps, str] | None = None,
    umls=None,
    item_id: str = "deep",
    warmup: bool = True,
    budget: BudgetManager | None = None,
    chat_id: str | None = None,
    task_id: str | None = None,
) -> str:
    """Run one deep-research task, streaming thoughts and output to stdout.

    ``budget`` (a child allocation from the run-global manager) turns on
    hard tool-budget enforcement for this task; ``None`` preserves the
    legacy unbounded behavior. ``chat_id`` attaches the task's verified
    evidence and budget ledger to the chat ledger under ``task_id``
    (defaults to ``item_id``); ``None`` leaves the ledger untouched.
    """
    if not isinstance(question, str) or not question.strip():
        raise ValueError("question must be a non-empty string")
    if warmup:
        kick_retriever_warmup(item_id)
    if budget is not None and not isinstance(budget, BudgetManager):
        raise TypeError("budget must be a BudgetManager or None")
    ledger_id = task_id or item_id
    deps = DeepDeps(umls=umls or build_umls_client(), budget=budget,
                    chat_id=chat_id, task_id=ledger_id if chat_id else None)
    runner = agent or build_deep_agent()
    store = None
    if chat_id:
        from src.runstate.store import get_store
        store = get_store()
    try:
        if store is not None:
            await store.record_task_started(chat_id, ledger_id, question)
        async with runner.run_stream(
            render_task_prompt(question), deps=deps, event_stream_handler=console_stream(item_id)
        ) as result:
            output = await result.get_output()
    except Exception as exc:
        if store is not None:
            from src.runstate.models import TaskState
            await store.record_task_finished(chat_id, ledger_id, TaskState.FAILED)
        raise
    if store is not None:
        from src.runstate.models import TaskState
        ledger = None
        if budget is not None:
            ledger = await budget.ledger_snapshot()
        await store.record_task_finished(
            chat_id, ledger_id, TaskState.DONE,
            budget_allocated=(ledger or {}).get("allocated"),
            budget_expenditure_history=(ledger or {}).get("expenditure"),
            budget_remaining=(ledger or {}).get("remaining"))
    return output
