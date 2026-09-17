# Hierarchical Budget System (`src/budget`)

Resource-aware, bounded research agents **without taking away their
autonomy**: the system controls the constraints, the agent controls the
strategy.

## How it works

```
ResearchRun (global BudgetManager)
├── Task T1 (child BudgetManager)
└── Task T2 (child BudgetManager)
```

- Every billable tool call passes through `BudgetCapability`, a
  pydantic-ai `AbstractCapability` registered on the orchestrator and
  deep agents — the same extension point the judge (`LLMAsJudge`) and
  trust (`TrustMiddleware`) middleware use. One registration covers all
  current and future tools; no per-tool `if budget_remaining` checks.
- A child can never spend more than `min(child_remaining,
  parent_remaining)`. Reservations walk the whole ancestor chain under a
  fixed lock order (root → leaf), so parallel tasks race safely: exactly
  one winner per remaining unit, totals never go negative.
- Denials return a structured `budget_exhausted` JSON result to the
  model — normal control flow, never an exception. The run continues and
  synthesizes from what it has.
- The agent sees its position via the free `budget_status` tool and via
  status notices piggybacked on tool results when pressure rises
  (LOW/CRITICAL). Snapshots are read-only: there is no API to raise
  limits or reset consumption. The only upward flow is the parent's
  `reclaim_child()`, which narrows a finished child's allocation to what
  it actually used (the seam future dynamic reallocation builds on).

## Dimensions

Tool calls only. `None` = unbounded. Tokens, model calls, time, and
money are not budgeted. Per-tool call costs are declared in
`costs.ToolCostModel` (1 per priced call, 0 for free reads) and
overridable with `register()` for new tools.

## Configuration (env, following repo convention)

| Variable | Meaning |
|---|---|
| `MEDRAG_BUDGET_ENABLED` | `0` disables enforcement (legacy behavior) |
| `MEDRAG_BUDGET_GLOBAL_MAX_TOOL_CALLS` | Run ceiling (default: 100) |
| `MEDRAG_BUDGET_ORCHESTRATOR_MAX_TOOL_CALLS` | Orchestrator master budget (default: 80) |
| `MEDRAG_BUDGET_TASK_MAX_TOOL_CALLS` | Per-task allocation (default: 20) |
| `MEDRAG_BUDGET_LOW_THRESHOLD / _CRITICAL_THRESHOLD` | State thresholds (defaults 0.40 / 0.20) |

States (`NORMAL → LOW → CRITICAL → EXHAUSTED`) derive from the worst
remaining tool-call fraction across ancestor levels — a spent
parent degrades the child even when the child's own allocation looks
healthy.

## Wiring

- `DeepDeps.budget` carries the manager (default `None` = inert).
- `stream_adapter` creates the run-global manager, carves a child per
  `spawn_subagent`, reclaims finished children,
  flags `task done` events with `budget_exhausted`, and attaches a
  compact budget summary to the `done` event.
- `agents/__main__.py` (CLI) does the same for planner-spawned deep
  tasks, including parallel `asyncio.gather` legs.
- Observability: `budget_reserved/consumed/denied`,
  `budget_state_transition`, `budget_allocated/reclaimed` events go
  through the existing `get_trace()` log; `manager.history` /
  `to_dict()` give the full auditable ledger.

## Tests

`tests/test_budget.py`: consumption, exhaustion,
overspending, hierarchy, parent exhaustion, concurrency races, state
transitions, exhaustion-during-research, agent visibility, unauthorized
modification, reservations, persistence shape, capability
allow/deny/failure/inert paths, cost model, config, builder wiring.

## Deliberate V1 limits

- No dynamic reallocation yet — `reclaim_child()` only frees headroom;
  the orchestrator policy that re-spends it is a clean follow-up.
- The legacy `worker.py` / `gap_fill.py` / `stages.py` agents were
  deleted: they referenced `src.agents.graph/state/timeouts`, which no
  longer exist in this tree. The live agents are `deep_agent`,
  `orchestrator`, `planner` and `synthesizer`, and the budget system
  integrates with those.
