import asyncio
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

from src.agents.deep_agent import build_deep_agent, run_deep_task
from src.agents.planner import generate_plan
from src.lib.pretty import badge, fold_line, rule, style
from src.lib.trace import get_trace

load_dotenv()

_DEFAULT_LOG = Path(__file__).resolve().parent.parent.parent / "logs.txt"


def _usage() -> str:
    return 'usage: python -m src.agents "<query>" [--json]'


def _make_run_budget():
    """Run-global budget for the CLI path (None when disabled)."""
    from src.budget.budget import BudgetManager
    from src.budget.config import budget_enabled, load_budget_config

    if not budget_enabled():
        return None
    cfg = load_budget_config()
    return BudgetManager(
        cfg.global_limits,
        task_id="run",
        low_threshold=cfg.low_threshold,
        critical_threshold=cfg.critical_threshold,
    )


def _task_limits():
    """Per-task allocation for the CLI path."""
    from src.budget.config import budget_enabled, load_budget_config

    if not budget_enabled():
        return None
    return load_budget_config().task_limits


async def _make_run_ledger(plan) -> str:
    """Create the chat ledger with one turn for this CLI run."""
    from src.runstate.store import get_store
    import uuid

    chat_id = uuid.uuid4().hex[:12]
    turn_id = uuid.uuid4().hex[:12]
    store = get_store()
    await store.create_turn(chat_id, turn_id, "")
    await store.record_plan(chat_id, [
        {"id": item.id, "question": item.question,
         "deep_research": item.deep_research}
        for item in plan.items
    ])
    return chat_id


async def _finish_run_ledger(chat_id: str, run_budget) -> None:
    """Mark the CLI chat ledger's turn complete with its budget summary."""
    from src.runstate.models import RunStatus
    from src.runstate.store import get_store

    store = get_store()
    chat = store.get_chat(chat_id)
    turn_id = chat.turn_order[-1] if chat and chat.turn_order else ""
    budget = run_budget.to_dict() if run_budget is not None else None
    await store.record_run_finished(chat_id, turn_id, RunStatus.COMPLETE,
                                    budget=budget)


async def main() -> None:
    args = [a for a in sys.argv[1:] if a.strip()]
    wants_json = "--json" in args
    query = next((a for a in args if not a.startswith("-")), "")
    if not query:
        raise SystemExit(_usage())

    print(style(rule("medrag research"), "1", "36"), flush=True)
    print(fold_line(style("Query: ", "1"), query), flush=True)
    log_path = os.environ.get("LOG_FILE") or _DEFAULT_LOG
    trace = get_trace()
    trace.open_stream(log_path, query=query)
    print(f"Log: {log_path}", flush=True)
    try:
        await _run(query, wants_json)
    finally:
        trace.close_stream()


async def _run(query: str, wants_json: bool) -> None:
    plan = await generate_plan(query)
    deep_items = [item for item in plan.items if item.deep_research]
    shallow_items = [item for item in plan.items if not item.deep_research]
    print(
        style(
            f"Plan: {len(plan.items)} task(s) "
            f"· {len(deep_items)} deep · {len(shallow_items)} shallow",
            "1",
        ),
        flush=True,
    )
    for item in plan.items:
        print(
            fold_line(f"  {style(item.id, '1')} {badge(item.deep_research)} ", item.question),
            flush=True,
        )
    if wants_json:
        print(plan.model_dump_json(indent=2), flush=True)

    if plan.is_empty:
        print("(No tasks)")
        return
    print(style(rule(), "2"), flush=True)
    for item in shallow_items:
        print(
            fold_line(
                f"{style('○', '33')} [{item.id}] shallow task, no deep agent: ",
                item.question,
            ),
            flush=True,
        )
    if not deep_items:
        return
    agent = build_deep_agent()
    for item in deep_items:
        print(
            fold_line(
                f"{style('◈', '35')} [{item.id}] deep_research=True, spawning deep agent: ",
                item.question,
            ),
            flush=True,
        )
    # One run-global budget; each deep task gets a child allocation carved
    # out of it (None = enforcement off via MEDRAG_BUDGET_ENABLED=0).
    run_budget = _make_run_budget()
    task_budgets = (
        {item.id: run_budget.spawn_child(item.id, _task_limits())
         for item in deep_items}
        if run_budget is not None else {}
    )
    # Shared chat ledger: plan, tasks, verified evidence, and structured
    # findings land in the chat's server-side JSON object.
    ledger_id = await _make_run_ledger(plan)
    sequential = os.environ.get("SEQUENTIAL", "").strip().lower() in ("1", "true", "yes")
    if sequential:
        # Debug mode: one deep task at a time so console output reads top-to-bottom.
        print("Sequential mode: running deep tasks one at a time…", flush=True)
        replies = [await run_deep_task(item.question, agent=agent, item_id=item.id,
                                       budget=task_budgets.get(item.id),
                                       chat_id=ledger_id)
                   for item in deep_items]
    else:
        replies = await asyncio.gather(
            *(run_deep_task(item.question, agent=agent, item_id=item.id,
                            budget=task_budgets.get(item.id),
                            chat_id=ledger_id) for item in deep_items)
        )
    if run_budget is not None:
        for item in deep_items:
            child = task_budgets.get(item.id)
            if child is not None:
                await run_budget.reclaim_child(child)
        get_trace().log("run_budget", **run_budget.to_dict()["consumed"])
    if ledger_id is not None:
        await _finish_run_ledger(ledger_id, run_budget)
    print(style(rule("answers"), "1", "32"), flush=True)
    for item, reply in zip(deep_items, replies):
        print(
            fold_line(
                f"{style('✔', '32')} [{item.id}] deep agent replied: ",
                reply,
            ),
            flush=True,
        )
        print(style(rule(), "2"), flush=True)


if __name__ == "__main__":
    asyncio.run(main())
