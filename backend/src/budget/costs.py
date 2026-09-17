"""Per-action cost model: every tool call costs tool calls, nothing else.

Each tool declares how many tool calls one invocation spends (usually 1;
free reads cost 0 so agents always inspect state). Tokens, model
invocations, time, and money are not budgeted — they never appear here.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.budget.budget import ActionCost


# Canonical costs for this repo's actual tool surface. Priced tools cost
# one tool call; visibility and finalization tools are free.
_DEFAULT_COSTS: dict[str, ActionCost] = {
    "local_search": ActionCost(tool_calls=1),
    "web_search": ActionCost(tool_calls=1),
    "lookup_medical_term": ActionCost(tool_calls=1),
    "submit_plan": ActionCost(tool_calls=1),
    "spawn_subagent": ActionCost(tool_calls=1),
    # The synthesizer needs no budget: free to call, gated by the
    # gap-checked mark instead.
    "synthesize": ActionCost(),
    # Read-only visibility mechanism — must stay free or agents won't check.
    "budget_status": ActionCost(),
    "run_progress": ActionCost(),
    "check_gaps": ActionCost(),
}

_FALLBACK_COST = ActionCost(tool_calls=1)


@dataclass
class ToolCostModel:
    """Resolves the :class:`ActionCost` for a tool name. Extensible at runtime."""

    _costs: dict[str, ActionCost] | None = None

    def __post_init__(self) -> None:
        if self._costs is None:
            self._costs = dict(_DEFAULT_COSTS)

    def cost_for(self, tool_name: str) -> ActionCost:
        return self._costs.get(tool_name or "", _FALLBACK_COST)

    def register(self, tool_name: str, cost: ActionCost) -> None:
        """Declare/override the cost of a tool (e.g. a new retrieval tool)."""
        self._costs[tool_name] = cost


def default_tool_costs() -> dict[str, ActionCost]:
    return dict(_DEFAULT_COSTS)


__all__ = [
    "ActionCost",
    "ToolCostModel",
    "default_tool_costs",
]
