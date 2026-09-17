#!/usr/bin/env python3
"""Manually spawn ONE subagent leg and watch it work.

Runs a single deep (or shallow) research task exactly as the
orchestrator would spawn it — same task prompt, same ledger-backed
requirements, same budget allocation — and narrates its thinking, tool
calls, and results live to stdout. Ends with the leg's reply, its
budget ledger, and its evidence/gap counts.

    .venv/bin/python scripts/test_subagent.py "Define hypertension"
    .venv/bin/python scripts/test_subagent.py "q" --depth shallow
    .venv/bin/python scripts/test_subagent.py "q" --task-id T2 \
        --req "E1: definition" --req "E2: criteria"
    .venv/bin/python scripts/test_subagent.py "q" --max-tools 5 --json

--req may repeat, or pass --reqs "E1: a; E2: b". Sampling knobs mirror
the planner script (temperature 0.0, seed 43, top_p 0.1). Needs the LLM
env (see .env.example). The ledger lives in an isolated tmp dir, never
in your real chats.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv  # noqa: E402
from pydantic_ai.settings import ModelSettings  # noqa: E402

from src.agents.stream_adapter import _requirements_block  # noqa: E402
from src.budget.budget import BudgetLimits, BudgetManager  # noqa: E402
from src.budget.config import budget_enabled, load_budget_config  # noqa: E402
from src.lib.streaming import console_stream  # noqa: E402
from src.runstate.store import RunStore, set_store  # noqa: E402


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task", help="research task question")
    parser.add_argument("--depth", default="deep",
                        choices=("deep", "shallow"))
    parser.add_argument("--task-id", default="T1")
    parser.add_argument("--req", action="append", default=[],
                        help='requirement as "E1: description" (repeatable)')
    parser.add_argument("--reqs", default="",
                        help='semicolon-separated "E1: a; E2: b"')
    parser.add_argument("--context", default="",
                        help="extra orchestrator context for the task")
    parser.add_argument("--model", default=None,
                        help="override AGENT_MODEL / SMALL_MODEL")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument("--top-p", type=float, default=0.1)
    parser.add_argument("--max-tools", type=int, default=None)
    parser.add_argument("--max-llm", type=int, default=None)
    parser.add_argument("--max-tokens", type=int, default=None)
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--json", action="store_true",
                        help="dump the ledger task record at the end")
    return parser.parse_args()


def _parse_reqs(args: argparse.Namespace) -> list[dict]:
    raw = list(args.req)
    if args.reqs.strip():
        raw.extend(args.reqs.split(";"))
    reqs = []
    for i, item in enumerate(raw):
        text = item.strip()
        if not text:
            continue
        rid, _, desc = text.partition(":")
        rid, desc = rid.strip(), desc.strip()
        if not desc:
            desc, rid = rid, f"E{i + 1}"
        if not rid:
            rid = f"E{i + 1}"
        reqs.append({"id": rid, "description": desc})
    return reqs


def _patch_sampling(args: argparse.Namespace) -> None:
    """Pin sampling on whatever agent module gets built."""
    settings = ModelSettings(temperature=args.temperature,
                             seed=args.seed, top_p=args.top_p)

    def _wrap(mod_name: str) -> None:
        mod = sys.modules.get(mod_name)
        if mod is None:
            __import__(mod_name)
            mod = sys.modules[mod_name]
        orig = mod.make_model
        mod.make_model = lambda endpoint, **kw: orig(
            endpoint, settings=settings, **kw)

    _wrap("src.agents.deep_agent")
    _wrap("src.agents.orchestrator")


async def _main() -> int:
    load_dotenv()
    args = _parse_args()
    _patch_sampling(args)

    reqs = _parse_reqs(args)
    store = RunStore(tempfile.mkdtemp(prefix="subagent-test-"))
    set_store(store)
    chat_id, run_id, sid = "manual", "manual-1", args.task_id
    await store.create_turn(chat_id, run_id, args.task)
    await store.record_plan(chat_id, [{
        "id": sid, "question": args.task,
        "deep_research": args.depth == "deep",
        "evidence_requirements": reqs,
    }])
    await store.record_task_started(chat_id, sid, args.task, args.depth)

    cfg = load_budget_config() if budget_enabled() else None
    run_budget = None
    if cfg is not None:
        run_budget = BudgetManager(cfg.global_limits, task_id="run",
                                   cost_provider=cfg.cost_provider())
    child = None
    if run_budget is not None and cfg is not None:
        limits = cfg.task_limits
        overrides = {
            "max_tool_calls": args.max_tools,
            "max_llm_calls": args.max_llm,
            "max_total_tokens": args.max_tokens,
        }
        if any(v is not None for v in overrides.values()):
            data = dict(limits.__dict__)
            data.update({k: v for k, v in overrides.items()
                         if v is not None})
            limits = BudgetLimits(**data)
        child = run_budget.spawn_child(sid, limits)

    from src.agents import deep_agent as deep_agent_mod
    from src.agents import orchestrator as orchestrator_mod
    from src.tools.umls import DeepDeps, build_umls_client

    prompt = deep_agent_mod.render_task_prompt(args.task)
    if args.context.strip():
        prompt += f"\n\nContext from orchestrator:\n{args.context.strip()}"
    task = store.get_chat(chat_id).turns[run_id].task(sid)
    block = _requirements_block(task)
    if block:
        prompt += f"\n\n{block}"
    print(f"=== prompt ({len(prompt)} chars) ===")
    print(prompt)
    print("=" * 40, flush=True)

    handler = console_stream(f"subagent:{sid}")
    started = time.perf_counter()
    try:
        if args.depth == "shallow":
            if args.model:
                os.environ["SMALL_MODEL"] = args.model
            runner = orchestrator_mod.build_shallow_agent()
            result = await asyncio.wait_for(
                runner.run(prompt, event_stream_handler=handler),
                timeout=args.timeout)
            reply = result.output
            usage = None
        else:
            if args.model:
                os.environ["AGENT_MODEL"] = args.model
            runner = deep_agent_mod.build_deep_agent()
            deps = DeepDeps(umls=build_umls_client(), budget=child,
                            chat_id=chat_id, task_id=sid)
            result = await asyncio.wait_for(
                runner.run(prompt, deps=deps,
                           event_stream_handler=handler),
                timeout=args.timeout)
            reply = result.output
            from src.budget.capability import record_provider_usage, usage_tokens
            await record_provider_usage(child, result)
            usage = usage_tokens(result)
    except Exception as exc:  # noqa: BLE001 -- test script reports, then exits
        print(f"\nLEG FAILED after {time.perf_counter() - started:.1f}s: "
              f"{type(exc).__name__}: {exc}")
        return 1

    elapsed = time.perf_counter() - started
    print(f"\n=== reply ({elapsed:.1f}s) ===")
    print(reply)
    if usage is not None:
        print(f"\nusage: {usage[0]} in / {usage[1]} out")
    if child is not None:
        ledger = await child.ledger_snapshot()
        print("\n=== budget ledger ===")
        print(f"allocated: {ledger['allocated']}")
        print(f"remaining: {ledger['remaining']}")
        print(f"events: {len(ledger['expenditure'])}")
    task = store.get_chat(chat_id).turns[run_id].task(sid)
    ev = sum(len(r.supporting_evidence) for r in task.evidence_requirements)
    gaps = sum(len(r.gaps) for r in task.evidence_requirements)
    print(f"\nledger: {len(task.evidence_requirements)} requirements, "
          f"{ev} verified passages, {gaps} open gaps")
    if args.json:
        print("\n=== task record ===")
        print(task.model_dump_json(indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(_main()))
