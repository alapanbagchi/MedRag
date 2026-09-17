"""Scratch probe: agent-lane reasoning effort (planner, AGENT_MODEL lane).

Runs the real planner twice at the endpoint default effort and twice at low,
printing wall / output tokens / reasoning tokens / plan shape. The planner is
the cheapest live representative of the agent lane (orchestrator, legs,
synthesizer all share make_model).
"""
from __future__ import annotations

import asyncio
import os
import time
from pathlib import Path

from dotenv import load_dotenv

BACKEND = Path(__file__).resolve().parents[1]
load_dotenv(BACKEND / ".env")

from src.agents import planner  # noqa: E402
from src.tools.verifier import _usage_detail  # noqa: E402

QUERY = ("What are the diagnostic thresholds and first-line treatment for "
         "hypertension?")


async def one(effort: str, rep: int) -> None:
    os.environ["AGENT_REASONING_EFFORT"] = effort
    agent = planner.build_agent()
    settings = getattr(agent.model, "settings", None) or {}
    started = time.perf_counter()
    try:
        result = await agent.run(f"Query: {QUERY}")
    except Exception as exc:  # noqa: BLE001
        print(f"  {effort:8} rep{rep}: ERROR {type(exc).__name__}: {exc}",
              flush=True)
        return
    wall = time.perf_counter() - started
    detail = _usage_detail(result)
    try:
        plan = planner._coerce_plan(result.output)
        tasks = len(plan.items)
        reqs = sum(len(i.evidence_requirements) for i in plan.items)
        deep = sum(1 for i in plan.items if i.deep_research)
    except Exception:  # noqa: BLE001
        tasks = reqs = deep = -1
    print(f"  {effort:8} rep{rep}: wall {wall:6.1f}s | effort_setting "
          f"{settings.get('openai_reasoning_effort')!r} | prompt "
          f"{detail.get('prompt', 0):,} (cached {detail.get('cached', 0):,}) | "
          f"completion {detail.get('completion', 0):,} "
          f"(reasoning {detail.get('reasoning', 0):,}) | tasks {tasks} "
          f"(deep {deep}, reqs {reqs})", flush=True)


async def main() -> int:
    print("agent lane = " + str(os.environ.get("AGENT_MODEL"))
          + " | effort arm = endpoint default vs low", flush=True)
    seen = {"default": 0, "low": 0}
    for effort in ("default", "low", "default", "low"):
        seen[effort] += 1
        await one(effort, seen[effort])
    return 0


raise SystemExit(asyncio.run(main()))
