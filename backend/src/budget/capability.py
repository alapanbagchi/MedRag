"""Hard enforcement at the tool/execution boundary.

``BudgetCapability`` is a pydantic-ai ``AbstractCapability`` — the same
extension point the judge middleware (``LLMAsJudge``) uses. Added to an agent's ``capabilities``, it sees **every**
tool invocation on that agent, so no tool can circumvent it via another
path. The model's prompt is only advisory; this layer is the real gate::

    Agent → tool request → BudgetCapability → allowed? execute : deny

A denial returns a structured ``budget_exhausted`` JSON result to the
model (normal control flow — the run continues and can synthesize), it
never raises and never crashes the run. The capability resolves the
``BudgetManager`` from ``ctx.deps.budget``; with no budget attached (or
the feature disabled) it is fully inert — legacy behavior unchanged.
"""

from __future__ import annotations

import logging
from typing import Any

from pydantic_ai import RunContext
from pydantic_ai.capabilities import AbstractCapability

from src.budget.budget import ActionCost, BudgetDenied, BudgetManager
from src.budget.costs import ToolCostModel
from src.budget.prompt import budget_guidance, render_budget_status

logger = logging.getLogger(__name__)


def _manager_from_ctx(ctx: Any) -> BudgetManager | None:
    deps = getattr(ctx, "deps", None)
    manager = getattr(deps, "budget", None)
    return manager if isinstance(manager, BudgetManager) else None


class BudgetCapability(AbstractCapability):
    """Gate every tool call on the run/task budget."""

    def __init__(self, costs: ToolCostModel | None = None,
                 enabled: bool = True) -> None:
        self._costs = costs or ToolCostModel()
        self._enabled = enabled

    async def on_tool_execute_error(self, ctx, *, call, tool_def, args, error):
        # Budget denials are returned as results, never raised — but if a
        # denied-shaped error ever surfaces here, let it propagate as data.
        raise error

    async def wrap_tool_execute(self, ctx, *, call, tool_def, args, handler):
        manager = _manager_from_ctx(ctx)
        if not self._enabled or manager is None:
            return await handler(args)
        tool_name = getattr(call, "tool_name", "") or ""
        cost = self._costs.cost_for(tool_name)
        reservation = await manager.reserve(cost, action=tool_name)
        if isinstance(reservation, BudgetDenied):
            logger.info("budget denied %s for task %s: %s",
                        tool_name, manager.task_id, reservation.resource)
            return reservation.to_result_json()
        try:
            output = await handler(args)
        except Exception:
            # A failed tool did no useful work: release the hold instead of
            # billing the agent for the framework's failure.
            await manager.release(reservation)
            raise
        denied = await manager.consume(reservation)
        if isinstance(denied, BudgetDenied):
            # Defensive only (double-settle/foreign reservation): the
            # output is still valid, just annotate it.
            suffix = f"\n\n[BUDGET NOTE: {denied.message}]"
            return f"{output}{suffix}" if isinstance(output, str) else output
        # Piggyback visibility: when pressure rises, the agent sees its
        # exact position without spending a call asking for it.
        snapshot = await manager.snapshot()
        if snapshot.state.value in ("low", "critical") and isinstance(output, str):
            notice = (f"\n\n[{render_budget_status(snapshot)}\n"
                      f"{budget_guidance(snapshot.state)}]")
            output = output + notice
        return output


async def budget_status(ctx: RunContext[Any]) -> str:
    """Report exact remaining tool calls (free — costs nothing to call).

    Read-only by construction: it exposes a snapshot, and the snapshot
    type carries no mutation API.
    """
    manager = _manager_from_ctx(ctx)
    if manager is None:
        return "No budget is attached to this run (budget enforcement off)."
    snapshot = await manager.snapshot()
    return (f"{render_budget_status(snapshot)}\n"
            f"{budget_guidance(snapshot.state)}")
