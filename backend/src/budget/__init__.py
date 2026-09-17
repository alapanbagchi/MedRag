"""Hierarchical tool-call budgets for agentic research.

Design principle: **the system controls the constraints; the agent controls
the strategy.** The model decides what to do; this package decides whether
it is allowed, at the tool/execution boundary (see
``src/budget/capability.py``), never inside the model's reasoning loop.

Hierarchy::

    ResearchRun (global BudgetManager)
    ├── Task A (child BudgetManager)
    └── Task B (child BudgetManager)

A child can never consume more than ``min(child_remaining,
parent_remaining)`` — every reservation walks the whole ancestor chain
atomically, so parallel tasks can never overspend the global ceiling.

One dimension only: tool calls. Tokens, model calls, time, and money are
not budgeted. A ``None`` limit means "unbounded".
"""

from src.budget.budget import (
    ActionCost,
    BudgetConsumption,
    BudgetDenied,
    BudgetLimits,
    BudgetManager,
    BudgetSnapshot,
    BudgetState,
    Reservation,
)
from src.budget.capability import BudgetCapability, budget_status
from src.budget.config import BudgetConfig, budget_enabled
from src.budget.costs import (
    ToolCostModel,
    default_tool_costs,
)
from src.budget.prompt import budget_guidance, render_budget_status

__all__ = [
    "ActionCost",
    "BudgetCapability",
    "BudgetConfig",
    "BudgetConsumption",
    "BudgetDenied",
    "BudgetLimits",
    "BudgetManager",
    "BudgetSnapshot",
    "BudgetState",
    "Reservation",
    "ToolCostModel",
    "budget_enabled",
    "budget_guidance",
    "budget_status",
    "default_tool_costs",
    "render_budget_status",
]
