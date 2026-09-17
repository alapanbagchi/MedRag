"""Agent visibility: read-only budget snapshot rendering + state guidance.

The agent can never change its budget, but it must always know where it
stands. Snapshots are injected as a compact status block (and the
``budget_status`` tool returns the full version on demand). Guidance is
advisory — it shapes strategy, never dictates it — and the budget is
always framed as a MAXIMUM, never a target.
"""

from __future__ import annotations

from src.budget.budget import BudgetSnapshot, BudgetState


def _fmt(value: int | None) -> str:
    if value is None:
        return "unlimited"
    return f"{value:,}"


def render_budget_status(snapshot: BudgetSnapshot) -> str:
    """Compact, paste-ready status block for prompts and tool results."""
    lines = [
        "BUDGET STATUS",
        f"Task: {snapshot.task_id} · State: {snapshot.state.value.upper()}",
        f"Tool calls: {snapshot.consumed.tool_calls} / "
        f"{_fmt(snapshot.limits.max_tool_calls)} "
        f"(remaining {_fmt(snapshot.remaining_tool_calls)})",
    ]
    if snapshot.state != BudgetState.NORMAL:
        lines.append(f"Limiting resource: {snapshot.limiting_resource}")
    return "\n".join(lines)


def budget_guidance(state: BudgetState) -> str:
    """Advisory strategy guidance per budget state (never a command)."""
    if state == BudgetState.NORMAL:
        return ("Budget is plentiful. Continue exploring while additional "
                "research has value — but stop as soon as the evidence is "
                "sufficient. The budget is a maximum, not a target.")
    if state == BudgetState.LOW:
        return ("Budget is LOW. Prefer high-yield actions, avoid redundant "
                "retrieval, and prioritize unresolved evidence gaps over "
                "exploration.")
    if state == BudgetState.CRITICAL:
        return ("Budget is CRITICAL. Focus only on the most important "
                "unresolved gaps, avoid exploratory or low-value actions, "
                "and prepare to synthesize from what you have.")
    return ("Budget is EXHAUSTED. Stop further research immediately. Produce "
            "the best supported result from the evidence already collected "
            "and explicitly identify unresolved gaps.")


BUDGET_SYSTEM_SECTION = """\
# RESOURCE BUDGET (system-enforced)

You operate under a resource budget. The budget is a MAXIMUM, never a target:
having resources left over is good. Stop researching as soon as the evidence
is sufficient — do NOT keep searching just because budget remains.

You cannot increase, reset, or bypass your budget. When a tool call is denied
for budget reasons you receive a `budget_exhausted` result: treat it as a
signal to synthesize from the evidence already collected, explicitly noting
unresolved gaps. Never retry a denied action in a loop.

Call `budget_status` at any time (it is free) to see your exact remaining
resources. Research state + budget state together determine your next action:
sufficient evidence means stop, even with budget to spare.

YOUR MASTER BUDGET: `budget_status` shows YOUR OWN master budget. Every
`spawn_subagent` call and every tool call you make spends from it. When it
is exhausted you cannot spawn more sub-agents or call tools — you must
synthesize from the evidence already collected.

SPAWN BUDGETS ARE EXPLICIT: every sub-agent is created with an explicit
tool-call budget that YOU assign via `spawn_subagent` (`max_tool_calls`).
The allocation is carved out of your master budget: never promise a
sub-agent more than you have left, and size each allocation to the task —
a narrow lookup needs a small budget, open-ended research a larger one.
Omit it to use the task default. A sub-agent stops when its budget runs
out (its receipt says BUDGET_EXHAUSTED); that is a normal stop, not a
failure — synthesize from its stored evidence.
"""
