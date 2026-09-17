from __future__ import annotations

import asyncio
import functools
from pathlib import Path

from collections.abc import Awaitable, Callable

from pydantic import BaseModel, Field
from pydantic_ai import Agent
from pydantic_ai.messages import PartDeltaEvent, ThinkingPartDelta
from pydantic_ai.settings import ModelSettings

from src.lib import narrate
from src.lib.trace import get_trace
from src.lib.utils import decode_structured
from src.llm.models import agent_endpoint, make_model, pool_for
from src.llm.pool import estimate_tokens, is_retryable, retry_delay
from src.tools.verifier import EvidenceRequirement

PLANNER_MAX_ATTEMPTS = 3

# The planner's pinned sampling: greedy, fixed seed, tight nucleus.
# Every delegation (orchestrator generate_plan, CLI, tests) gets these
# unless the caller builds a custom agent explicitly.
PLANNER_TEMPERATURE = 0.0
PLANNER_SEED = 43
PLANNER_TOP_P = 0.1

PROMPT_FILE = Path(__file__).parent.parent / "prompts" / "plan.txt"


class PlanItem(BaseModel):
    id: str = "P1"
    question: str
    deep_research: bool = False   # True if the task requires a deep agent with tools
    # Always pre-filled by the planner: the executing leg works from
    # these verbatim and never plans for itself.
    evidence_requirements: list[EvidenceRequirement] = Field(default_factory=list)

class Plan(BaseModel):
    items: list[PlanItem] = []

    @property
    def is_empty(self) -> bool:
        return not self.items


@functools.lru_cache(maxsize=1)
def load_prompt() -> str:
    return PROMPT_FILE.read_text(encoding="utf-8")


def _coerce_plan(output: object) -> Plan:
    """Accept a parsed ``Plan`` (injected fakes/tests) or raw JSON text."""
    if isinstance(output, Plan):
        return output
    return decode_structured(str(output), Plan)


def build_agent() -> Agent[None, str]:
    """Build the planner agent on the big thinking lane.

    Plain-text output (no output_type): the prompt already demands raw
    JSON, and this provider's thinking mode rejects the forced
    ``tool_choice`` that structured-output mode sends. Sampling is
    pinned (temperature/seed/top_p): every plan comes from the same
    deck unless the caller builds a custom agent explicitly.
    """
    endpoint = agent_endpoint()
    model = make_model(endpoint, settings=ModelSettings(
        temperature=PLANNER_TEMPERATURE, seed=PLANNER_SEED,
        top_p=PLANNER_TOP_P))
    return Agent(model, system_prompt=load_prompt())


async def _run_streamed(
    runner: Agent[None, str],
    prompt: str,
    pool: object | None,
    budget: int,
    on_thinking: Callable[[str], Awaitable[None]],
) -> Plan:
    """Run the planner streaming thinking deltas to ``on_thinking``.

    Same result as ``runner.run`` (raw JSON text parsed into a ``Plan``);
    models that don't emit thinking parts simply yield no deltas.
    """
    async def _thinking_only(ctx: object, events: object) -> None:
        async for event in events:  # type: ignore[union-attr]
            if (isinstance(event, PartDeltaEvent)
                    and isinstance(event.delta, ThinkingPartDelta)
                    and event.delta.content_delta):
                await on_thinking(event.delta.content_delta)

    async def _go() -> Plan:
        async with runner.run_stream(
                prompt, event_stream_handler=_thinking_only) as result:
            return _coerce_plan(await result.get_output())

    if pool is None:
        return await _go()
    async with pool.slot(budget, label="planner"):  # type: ignore[union-attr]
        return await _go()


async def generate_plan(
    query: str,
    agent: Agent[None, str] | None = None,
    on_thinking: Callable[[str], Awaitable[None]] | None = None,
    usage_sink: list | None = None,
) -> Plan:
    """Generates a plan of action to solve the query in small actionable tasks.

    When ``on_thinking`` is given the planner runs streaming and forwards
    the model's thinking deltas to it as they arrive. When ``usage_sink``
    is given, each completed agent result is appended so the caller can
    fold the planner's LLM usage into its own budget ledger (streamed
    runs do not expose their result object, so only non-streamed runs
    are collected).
    """
    # 1. Validate the query
    if not isinstance(query, str) or not query.strip():
        raise ValueError("query must be a non-empty string")

    # 2. Get the agent (build if not provided)
    runner = agent or build_agent()

    # 3. Run the decomposition (shared LLM pool; retry transient errors,
    #    then raise — a missing top-level plan cannot fail open)
    prompt = f"Query: {query.strip()}"
    pool = None
    budget = 0
    if agent is None:
        pool = pool_for(agent_endpoint())
        budget = estimate_tokens(prompt, reserve=500)
    last_error: Exception | None = None
    for attempt in range(1, PLANNER_MAX_ATTEMPTS + 1):
        try:
            if on_thinking is None:
                if pool is None:
                    result = await runner.run(prompt)
                else:
                    async with pool.slot(budget, label="planner"):
                        result = await runner.run(prompt)
                if usage_sink is not None:
                    usage_sink.append(result)
                # 4. Parse the raw JSON text into a Plan
                return _coerce_plan(result.output)
            return await _run_streamed(runner, prompt, pool, budget,
                                       on_thinking)
        except Exception as exc:
            last_error = exc
            if is_retryable(exc) and attempt < PLANNER_MAX_ATTEMPTS:
                delay = retry_delay(attempt - 1, exc)
                narrate.say(f"  [planner] transient error ({exc});"
                            f" retrying in {delay:.1f}s…")
                get_trace().log("planner_retry", attempt=attempt,
                                wait_s=round(delay, 1), error=str(exc)[:200])
                await asyncio.sleep(delay)
                continue
            raise
    assert last_error is not None  # unreachable; guards the loop contract
    raise last_error
