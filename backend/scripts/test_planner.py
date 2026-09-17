#!/usr/bin/env python3
"""Manually test and tweak the planner agent (task decomposition).

Runs one question through generate_plan and prints the planned tasks
plus their evidence requirements as a table, then raw JSON. Knobs for
iterating:

    .venv/bin/python scripts/test_planner.py "Does vitamin D lower BP?"
    .venv/bin/python scripts/test_planner.py "q" --prompt-file /tmp/plan.txt
    .venv/bin/python scripts/test_planner.py "q" --model other-model --temperature 0.7
    .venv/bin/python scripts/test_planner.py "q" --seed 123
    .venv/bin/python scripts/test_planner.py "q" --json

--prompt-file tries a prompt edit without touching src/prompts/plan.txt.
--model / --temperature / --seed / --top-p override the production
agent lane (defaults mirror it: temperature 0.0, seed 43, top_p 0.1).
Needs the LLM env (LLM_BASE_URL / LLM_API_KEY / AGENT_MODEL, see
.env.example); exits 1 on empty plan.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv  # noqa: E402
from pydantic_ai import Agent  # noqa: E402
from pydantic_ai.settings import ModelSettings  # noqa: E402

from src.agents.planner import generate_plan  # noqa: E402
from src.llm.models import agent_endpoint, make_model  # noqa: E402


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("question", help="question to decompose")
    parser.add_argument("--model", default=None,
                        help="override AGENT_MODEL for this run")
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="model temperature (default: 0.0)")
    parser.add_argument("--seed", type=int, default=43,
                        help="fixed model seed for reproducibility "
                             "(default: 43, mirrors the planner)")
    parser.add_argument("--top-p", type=float, default=0.1,
                        help="nucleus sampling cutoff (default: 0.1)")
    parser.add_argument("--prompt-file", default=None,
                        help="system prompt file to try instead of plan.txt")
    parser.add_argument("--json", action="store_true",
                        help="also print the raw plan JSON")
    return parser.parse_args()


def _build_agent(args: argparse.Namespace) -> Agent:
    """Test agent: production prompt/model unless overridden, always with
    the pinned temperature + fixed seed (build_agent() has neither)."""
    endpoint = agent_endpoint()
    if args.model:
        endpoint = type(endpoint)(base_url=endpoint.base_url,
                                  api_key=endpoint.api_key,
                                  model=args.model)
    if args.prompt_file:
        with open(args.prompt_file, encoding="utf-8") as fh:
            system_prompt = fh.read()
    else:
        from src.agents.planner import load_prompt
        system_prompt = load_prompt()
    model = make_model(endpoint, settings=ModelSettings(
        temperature=args.temperature, seed=args.seed, top_p=args.top_p))
    return Agent(model, system_prompt=system_prompt)


async def _main() -> int:
    load_dotenv()
    args = _parse_args()
    started = time.perf_counter()
    used: list = []
    plan = await generate_plan(args.question,
                               agent=_build_agent(args),
                               usage_sink=used)
    elapsed = time.perf_counter() - started
    deep = [i for i in plan.items if i.deep_research]
    print(f"Plan: {len(plan.items)} task(s) "
          f". {len(deep)} deep . {len(plan.items) - len(deep)} shallow "
          f". {elapsed:.1f}s")
    for item in plan.items:
        badge = "DEEP" if item.deep_research else "shallow"
        print(f"\n  [{item.id}] {badge}: {item.question}")
        for req in item.evidence_requirements:
            print(f"    - {req.id}: {req.description}")
        if item.deep_research and not item.evidence_requirements:
            print("    !! deep task with NO evidence requirements")
    for result in used:
        try:
            usage = result.usage()
            print(f"\nPlanner usage: {usage.request_tokens} in / "
                  f"{usage.response_tokens} out tokens")
        except Exception:  # noqa: BLE001 -- usage is best-effort
            pass
    if args.json:
        print()
        print(plan.model_dump_json(indent=2))
    if plan.is_empty:
        print("(No tasks)")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(_main()))
