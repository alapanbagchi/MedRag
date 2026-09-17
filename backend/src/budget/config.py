"""Budget configuration from environment (the repo's config convention).

No YAML: every knob is an env var, mirroring ``src/config.py`` and the
``XDEEP_*`` / ``*_TIMEOUT_S`` precedents. Tool-call limits only — tokens,
model calls, time, and money are not budgeted and have no knobs.

Global run ceiling (``MEDRAG_BUDGET_GLOBAL_*``), the orchestrator's own
master budget (``MEDRAG_BUDGET_ORCHESTRATOR_*``), and per-task allocation
(``MEDRAG_BUDGET_TASK_*``). The tree is Global → Orchestrator → Tasks:
the orchestrator's own tool calls and every task allocation draw from
its master budget, which itself can never exceed the global ceiling.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from src.budget.budget import BudgetLimits


def _int(name: str) -> int | None:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return None
    try:
        value = int(raw)
    except ValueError:
        return None
    return value if value >= 0 else None


def _float(name: str) -> float | None:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return None
    try:
        value = float(raw)
    except ValueError:
        return None
    return value if value >= 0 else None


def budget_enabled() -> bool:
    """Kill-switch. ``MEDRAG_BUDGET_ENABLED=0`` restores legacy behavior."""
    return os.environ.get("MEDRAG_BUDGET_ENABLED", "1").strip().lower() not in (
        "0", "false", "no", "off",
    )


@dataclass(frozen=True)
class BudgetConfig:
    global_limits: BudgetLimits
    orch_limits: BudgetLimits
    task_limits: BudgetLimits
    low_threshold: float
    critical_threshold: float


def _limits(prefix: str) -> BudgetLimits:
    return BudgetLimits(
        max_tool_calls=_int(f"{prefix}_MAX_TOOL_CALLS"),
    )


def load_budget_config() -> BudgetConfig:
    """Defaults are generous ceilings (safety net); tighten per deployment."""
    global_limits = _limits("MEDRAG_BUDGET_GLOBAL")
    orch_limits = _limits("MEDRAG_BUDGET_ORCHESTRATOR")
    task_limits = _limits("MEDRAG_BUDGET_TASK")
    if global_limits.max_tool_calls is None:
        global_limits = BudgetLimits(max_tool_calls=100)
    if orch_limits.max_tool_calls is None:
        orch_limits = BudgetLimits(max_tool_calls=80)
    if task_limits.max_tool_calls is None:
        task_limits = BudgetLimits(max_tool_calls=20)
    low = _float("MEDRAG_BUDGET_LOW_THRESHOLD") or 0.40
    critical = _float("MEDRAG_BUDGET_CRITICAL_THRESHOLD") or 0.20
    return BudgetConfig(
        global_limits=global_limits,
        orch_limits=orch_limits,
        task_limits=task_limits,
        low_threshold=low,
        critical_threshold=min(critical, low),
    )
