"""Hierarchical tool-call budget tests (no LLM env required).

Covers: basic consumption, exhaustion, overspending, hierarchical and
parent exhaustion, concurrent consumption, state transitions,
exhaustion-during-research (partial result, no crash), agent visibility,
unauthorized modification, reservations, persistence shape, and the
enforcement capability (allow/deny/failure/inert paths). Tool calls are
the only budgeted dimension — tokens, model calls, time, and money
never factor in.
"""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import pytest

from src.budget.budget import (
    ActionCost,
    BudgetDenied,
    BudgetLimits,
    BudgetManager,
    BudgetState,
)
from src.budget.capability import BudgetCapability, budget_status
from src.budget.config import budget_enabled, load_budget_config
from src.budget.costs import ToolCostModel
from src.budget.prompt import budget_guidance, render_budget_status


def _manager(**kwargs) -> BudgetManager:
    limits = BudgetLimits(
        max_tool_calls=kwargs.pop("tools", 10),
    )
    return BudgetManager(limits, **kwargs)


# --- basic consumption -------------------------------------------------

async def test_basic_consumption():
    manager = _manager(tools=10)
    denied = await manager.try_consume(ActionCost(tool_calls=3), action="search")
    assert denied is None
    snapshot = await manager.snapshot()
    assert snapshot.consumed.tool_calls == 3
    assert snapshot.remaining_tool_calls == 7


async def test_exhaustion_denies_next_action():
    manager = _manager(tools=10)
    assert await manager.try_consume(ActionCost(tool_calls=10)) is None
    denied = await manager.try_consume(ActionCost(tool_calls=1), action="extra")
    assert isinstance(denied, BudgetDenied)
    assert denied.resource == "tool_calls"
    assert denied.remaining == 0
    # A denial spends nothing.
    snapshot = await manager.snapshot()
    assert snapshot.consumed.tool_calls == 10
    assert snapshot.state == BudgetState.EXHAUSTED


async def test_overspend_denied_and_remaining_untouched():
    manager = _manager(tools=5)
    denied = await manager.try_consume(ActionCost(tool_calls=6), action="big")
    assert isinstance(denied, BudgetDenied)
    snapshot = await manager.snapshot()
    assert snapshot.remaining_tool_calls == 5
    assert snapshot.consumed.tool_calls == 0


async def test_can_afford_is_read_only():
    manager = _manager(tools=5)
    assert await manager.can_afford(ActionCost(tool_calls=5)) is None
    denied = await manager.can_afford(ActionCost(tool_calls=6))
    assert isinstance(denied, BudgetDenied)
    snapshot = await manager.snapshot()
    assert snapshot.consumed.tool_calls == 0
    assert snapshot.remaining_tool_calls == 5


# --- hierarchy ----------------------------------------------------------

async def test_hierarchical_budget_child_spend_hits_parent():
    run = _manager(tools=10, task_id="run")
    child = run.spawn_child("T1", BudgetLimits(max_tool_calls=7))
    assert await child.try_consume(ActionCost(tool_calls=7), action="t1-work") is None
    parent_view = await run.snapshot()
    child_view = await child.snapshot()
    assert parent_view.consumed.tool_calls == 7
    assert parent_view.remaining_tool_calls == 3
    assert child_view.remaining_tool_calls == 0


async def test_parent_exhaustion_denies_despite_child_headroom():
    run = _manager(tools=2, task_id="run")
    child = run.spawn_child("T1", BudgetLimits(max_tool_calls=5))
    denied = await child.try_consume(ActionCost(tool_calls=3), action="t1-big")
    assert isinstance(denied, BudgetDenied)
    # Nothing spent anywhere.
    assert (await child.snapshot()).consumed.tool_calls == 0
    assert (await run.snapshot()).consumed.tool_calls == 0


async def test_child_limit_binds_before_parent():
    run = _manager(tools=100, task_id="run")
    child = run.spawn_child("T1", BudgetLimits(max_tool_calls=2))
    assert await child.try_consume(ActionCost(tool_calls=2)) is None
    denied = await child.try_consume(ActionCost(tool_calls=1))
    assert isinstance(denied, BudgetDenied)
    assert (await run.snapshot()).consumed.tool_calls == 2


async def test_sibling_tasks_share_parent_ceiling():
    run = _manager(tools=5, task_id="run")
    a = run.spawn_child("A", BudgetLimits(max_tool_calls=5))
    b = run.spawn_child("B", BudgetLimits(max_tool_calls=5))
    assert await a.try_consume(ActionCost(tool_calls=3)) is None
    # B still has headroom of its own, but the parent has only 2 left.
    denied = await b.try_consume(ActionCost(tool_calls=3))
    assert isinstance(denied, BudgetDenied)
    assert await b.try_consume(ActionCost(tool_calls=2)) is None
    assert (await run.snapshot()).remaining_tool_calls == 0


async def test_concurrent_consumption_never_exceeds_parent():
    run = _manager(tools=10, task_id="run")
    children = [run.spawn_child(f"T{i}", BudgetLimits(max_tool_calls=10))
                for i in range(10)]
    results = await asyncio.gather(*[
        child.try_consume(ActionCost(tool_calls=2), action=f"work-{i}")
        for i, child in enumerate(children)
    ])
    allowed = sum(1 for r in results if r is None)
    assert allowed == 5  # exactly 10 units / 2 per task
    total = (await run.snapshot()).consumed.tool_calls
    assert total == 10


async def test_concurrent_single_unit_race():
    run = _manager(tools=1, task_id="run")
    children = [run.spawn_child(f"T{i}", BudgetLimits(max_tool_calls=1))
                for i in range(8)]
    results = await asyncio.gather(*[
        child.try_consume(ActionCost(tool_calls=1)) for child in children
    ])
    assert sum(1 for r in results if r is None) == 1
    assert (await run.snapshot()).consumed.tool_calls == 1


# --- reservations -------------------------------------------------------

async def test_reserve_consume_release():
    manager = _manager(tools=3)
    reservation = await manager.reserve(ActionCost(tool_calls=2), action="r")
    assert not isinstance(reservation, BudgetDenied)
    # Held grants block others.
    denied = await manager.try_consume(ActionCost(tool_calls=2))
    assert isinstance(denied, BudgetDenied)
    await manager.release(reservation)
    assert await manager.try_consume(ActionCost(tool_calls=2)) is None
    snapshot = await manager.snapshot()
    assert snapshot.consumed.tool_calls == 2


async def test_failed_tool_releases_reservation():
    manager = _manager(tools=3)
    reservation = await manager.reserve(ActionCost(tool_calls=3), action="r")
    assert not isinstance(reservation, BudgetDenied)
    await manager.release(reservation)
    assert (await manager.snapshot()).remaining_tool_calls == 3


async def test_double_consume_rejected():
    manager = _manager(tools=5)
    reservation = await manager.reserve(ActionCost(tool_calls=2), action="r")
    assert not isinstance(reservation, BudgetDenied)
    assert await manager.consume(reservation) is None
    denied = await manager.consume(reservation)
    assert isinstance(denied, BudgetDenied)


async def test_foreign_reservation_rejected():
    a = _manager(tools=5, task_id="a")
    b = _manager(tools=5, task_id="b")
    reservation = await a.reserve(ActionCost(tool_calls=1), action="r")
    assert not isinstance(reservation, BudgetDenied)
    denied = await b.consume(reservation)
    assert isinstance(denied, BudgetDenied)
    assert (await b.snapshot()).consumed.tool_calls == 0


# --- states --------------------------------------------------------------

async def test_budget_state_transitions():
    manager = BudgetManager(BudgetLimits(max_tool_calls=10),
                            low_threshold=0.40, critical_threshold=0.20)
    assert (await manager.snapshot()).state == BudgetState.NORMAL
    assert await manager.try_consume(ActionCost(tool_calls=5)) is None
    assert (await manager.snapshot()).state == BudgetState.NORMAL  # 50% left
    assert await manager.try_consume(ActionCost(tool_calls=2)) is None
    assert (await manager.snapshot()).state == BudgetState.LOW  # 30% left
    assert await manager.try_consume(ActionCost(tool_calls=2)) is None
    assert (await manager.snapshot()).state == BudgetState.CRITICAL  # 10% left
    assert await manager.try_consume(ActionCost(tool_calls=1)) is None
    assert (await manager.snapshot()).state == BudgetState.EXHAUSTED
    transitions = [e for e in manager.history
                   if e["event"] == "budget_state_transition"]
    assert [e["to"] for e in transitions] == ["low", "critical", "exhausted"]


async def test_limiting_resource_is_tool_calls():
    manager = BudgetManager(BudgetLimits(max_tool_calls=10))
    assert await manager.try_consume(ActionCost(tool_calls=9)) is None
    snapshot = await manager.snapshot()
    assert snapshot.state == BudgetState.CRITICAL
    assert snapshot.limiting_resource == "tool_calls"
    unbounded = BudgetManager(BudgetLimits())
    assert (await unbounded.snapshot()).limiting_resource == "none"


async def test_spent_parent_degrades_child_state():
    run = _manager(tools=10, task_id="run")
    child = run.spawn_child("T1", BudgetLimits(max_tool_calls=100))
    assert await run.try_consume(ActionCost(tool_calls=9)) is None
    snapshot = await child.snapshot()
    # Child has 99/100 of its own allocation, but the parent has 10% left.
    assert snapshot.state == BudgetState.CRITICAL
    assert snapshot.remaining_tool_calls == 1


async def test_ledger_snapshot_for_task_record():
    run = _manager(tools=10, task_id="run")
    child = run.spawn_child("T1", BudgetLimits(max_tool_calls=4))
    assert await child.try_consume(ActionCost(tool_calls=1), action="search") is None
    denied = await child.try_consume(ActionCost(tool_calls=9), action="overspend")
    assert isinstance(denied, BudgetDenied)
    ledger = await child.ledger_snapshot()
    assert ledger["allocated"]["max_tool_calls"] == 4
    assert ledger["remaining"]["tool_calls"] == 3
    kinds = [e["event"] for e in ledger["expenditure"]]
    assert kinds == ["budget_consumed", "budget_denied"]
    assert ledger["expenditure"][0]["action"] == "search"
    json.dumps(ledger)  # serializable into the task record


async def test_state_thresholds_configurable():
    manager = BudgetManager(BudgetLimits(max_tool_calls=10),
                            low_threshold=0.9, critical_threshold=0.8)
    assert await manager.try_consume(ActionCost(tool_calls=2)) is None
    assert (await manager.snapshot()).state == BudgetState.LOW


# --- agent visibility -----------------------------------------------------

async def test_snapshot_is_read_only_and_accurate():
    run = _manager(tools=10, task_id="run")
    child = run.spawn_child("T1", BudgetLimits(max_tool_calls=4))
    await child.try_consume(ActionCost(tool_calls=1))
    snapshot = await child.snapshot()
    assert snapshot.task_id == "T1"
    assert snapshot.remaining_tool_calls == 3
    assert isinstance(snapshot.to_dict(), dict)
    assert snapshot.to_dict()["remaining"]["tool_calls"] == 3
    text = render_budget_status(snapshot)
    assert "T1" in text and "NORMAL" in text


async def test_budget_status_tool_is_free_and_read_only():
    manager = _manager(tools=2)
    ctx = SimpleNamespace(deps=SimpleNamespace(budget=manager))
    before = (await manager.snapshot()).consumed.tool_calls
    text = await budget_status(ctx)
    assert "BUDGET STATUS" in text
    after = (await manager.snapshot()).consumed.tool_calls
    assert before == after == 0  # asking costs nothing


async def test_budget_status_without_budget():
    ctx = SimpleNamespace(deps=SimpleNamespace(budget=None))
    text = await budget_status(ctx)
    assert "off" in text.lower()


async def test_guidance_frames_budget_as_maximum():
    assert "maximum, not a target" in budget_guidance(BudgetState.NORMAL)
    assert "unresolved gaps" in budget_guidance(BudgetState.EXHAUSTED).lower()


# --- unauthorized modification --------------------------------------------

async def test_no_api_to_raise_limits_or_reset_consumption():
    manager = _manager(tools=2)
    assert await manager.try_consume(ActionCost(tool_calls=2)) is None
    assert not hasattr(manager, "reset")
    assert not hasattr(manager, "increase")
    assert not hasattr(manager, "set_limits")
    assert not hasattr(manager, "add_funds")
    # Snapshot carries copies: mutating it cannot move the ledger.
    snapshot = await manager.snapshot()
    snapshot.consumed.tool_calls = 0
    assert (await manager.snapshot()).consumed.tool_calls == 2


async def test_child_cannot_reclaim_for_itself():
    run = _manager(tools=10, task_id="run")
    child = run.spawn_child("T1", BudgetLimits(max_tool_calls=4))
    assert await child.try_consume(ActionCost(tool_calls=1)) is None
    with pytest.raises((ValueError, AttributeError)):
        await child.reclaim_child(run)  # parent is not this child's child
    # Only the parent can narrow the child back to what it actually used.
    freed = await run.reclaim_child(child)
    assert freed["tool_calls"] == 3


# --- persistence / observability -------------------------------------------

async def test_history_is_auditable():
    run = _manager(tools=10, task_id="run")
    child = run.spawn_child("T1", BudgetLimits(max_tool_calls=2))
    assert await child.try_consume(ActionCost(tool_calls=2), action="search") is None
    denied = await child.try_consume(ActionCost(tool_calls=1), action="extra")
    assert isinstance(denied, BudgetDenied)
    kinds = [e["event"] for e in child.history]
    assert "budget_created" in kinds
    assert "budget_consumed" in kinds
    assert "budget_denied" in kinds
    consumed = next(e for e in child.history if e["event"] == "budget_consumed")
    assert consumed["action"] == "search"
    assert consumed["task_id"] == "T1"
    assert consumed["remaining"]["tool_calls"] == 0
    assert "timestamp" in consumed
    denied_event = next(e for e in child.history if e["event"] == "budget_denied")
    assert denied_event["resource"] == "tool_calls"


async def test_to_dict_round_trips_run_tree():
    run = _manager(tools=10, task_id="run")
    child = run.spawn_child("T1", BudgetLimits(max_tool_calls=4))
    await child.try_consume(ActionCost(tool_calls=1))
    data = run.to_dict()
    assert data["task_id"] == "run"
    assert data["consumed"]["tool_calls"] == 1
    assert len(data["children"]) == 1
    assert data["children"][0]["task_id"] == "T1"
    json.dumps(data)  # serializable for persistence


# --- enforcement capability -------------------------------------------------

def _tool_ctx(manager) -> SimpleNamespace:
    return SimpleNamespace(deps=SimpleNamespace(budget=manager))


def _tool_call(name: str) -> SimpleNamespace:
    return SimpleNamespace(tool_name=name, tool_call_id="tc1")


async def test_capability_allows_and_bills():
    manager = _manager(tools=5)
    capability = BudgetCapability()
    seen = []

    async def handler(args):
        seen.append(args)
        return "passages"

    out = await capability.wrap_tool_execute(
        _tool_ctx(manager), call=_tool_call("retrieve_evidence"),
        tool_def=None, args={"query": "q"}, handler=handler)
    assert out == "passages"
    assert seen == [{"query": "q"}]
    snapshot = await manager.snapshot()
    # retrieve_evidence costs 1 tool call; hidden judge work is unbilled.
    assert snapshot.consumed.tool_calls == 1


async def test_capability_denies_with_structured_result():
    manager = _manager(tools=1)
    capability = BudgetCapability()
    calls = []

    async def handler(args):
        calls.append(args)
        return "passages"

    assert await capability.wrap_tool_execute(
        _tool_ctx(manager), call=_tool_call("retrieve_evidence"),
        tool_def=None, args={}, handler=handler) == "passages"
    out = await capability.wrap_tool_execute(
        _tool_ctx(manager), call=_tool_call("retrieve_evidence"),
        tool_def=None, args={}, handler=handler)
    denied = json.loads(out)
    assert denied["status"] == "budget_exhausted"
    assert denied["resource"] == "tool_calls"
    assert denied["remaining"] == 0
    assert len(calls) == 1  # denied tool never executed


async def test_capability_denial_does_not_crash_run():
    """Exhaustion mid-research yields a partial result, not an exception."""
    manager = _manager(tools=2)
    capability = BudgetCapability()

    async def handler(args):
        return f"evidence-{args['n']}"

    results = []
    for n in range(4):
        results.append(await capability.wrap_tool_execute(
            _tool_ctx(manager), call=_tool_call("lookup_medical_term"),
            tool_def=None, args={"n": n}, handler=handler))
    assert results[0] == "evidence-0"
    assert results[1] == "evidence-1"
    assert json.loads(results[2])["status"] == "budget_exhausted"
    assert json.loads(results[3])["status"] == "budget_exhausted"


async def test_capability_releases_on_tool_failure():
    manager = _manager(tools=1)
    capability = BudgetCapability()

    async def handler(args):
        raise RuntimeError("retriever down")

    with pytest.raises(RuntimeError):
        await capability.wrap_tool_execute(
            _tool_ctx(manager), call=_tool_call("retrieve_evidence"),
            tool_def=None, args={}, handler=handler)
    # The framework's failure is not billed to the agent.
    assert (await manager.snapshot()).remaining_tool_calls == 1


async def test_capability_inert_without_budget():
    capability = BudgetCapability()
    ctx = SimpleNamespace(deps=SimpleNamespace(budget=None))

    async def handler(args):
        return "ok"

    assert await capability.wrap_tool_execute(
        ctx, call=_tool_call("retrieve_evidence"),
        tool_def=None, args={}, handler=handler) == "ok"


async def test_capability_disabled_flag_passes_through():
    manager = _manager(tools=1)
    capability = BudgetCapability(enabled=False)

    async def handler(args):
        return "ok"

    for _ in range(3):
        assert await capability.wrap_tool_execute(
            _tool_ctx(manager), call=_tool_call("retrieve_evidence"),
            tool_def=None, args={}, handler=handler) == "ok"
    assert (await manager.snapshot()).consumed.tool_calls == 0


async def test_capability_unknown_tool_gets_default_cost():
    manager = _manager(tools=2)
    capability = BudgetCapability()

    async def handler(args):
        return "ok"

    await capability.wrap_tool_execute(
        _tool_ctx(manager), call=_tool_call("brand_new_tool"),
        tool_def=None, args={}, handler=handler)
    assert (await manager.snapshot()).consumed.tool_calls == 1


async def test_capability_free_visibility_tool_unbilled():
    manager = _manager(tools=1)
    capability = BudgetCapability(costs=ToolCostModel())

    async def handler(args):
        return "status-text"

    out = await capability.wrap_tool_execute(
        _tool_ctx(manager), call=_tool_call("budget_status"),
        tool_def=None, args={}, handler=handler)
    assert out == "status-text"
    assert (await manager.snapshot()).consumed.tool_calls == 0


# --- cost model / config -----------------------------------------------------

def test_default_costs_cover_actual_tool_surface():
    model = ToolCostModel()
    for tool in ("local_search", "web_search",
                 "lookup_medical_term",
                 "check_gaps", "submit_plan", "spawn_subagent",
                 "budget_status"):
        cost = model.cost_for(tool)
        assert cost.tool_calls in (0, 1)
    assert model.cost_for("local_search").tool_calls == 1
    assert model.cost_for("budget_status").tool_calls == 0
    # The orchestrator's coverage check is a free ledger read.
    assert model.cost_for("check_gaps").tool_calls == 0


def test_cost_model_extensible():
    model = ToolCostModel()
    model.register("future_reranker", ActionCost(tool_calls=2))
    assert model.cost_for("future_reranker").tool_calls == 2


def test_config_defaults_are_generous_safety_net(monkeypatch):
    cfg = load_budget_config()
    assert cfg.global_limits.max_tool_calls == 100
    assert cfg.task_limits.max_tool_calls == 20
    assert cfg.low_threshold == pytest.approx(0.40)
    assert cfg.critical_threshold == pytest.approx(0.20)


def test_config_env_override(monkeypatch):
    monkeypatch.setenv("MEDRAG_BUDGET_GLOBAL_MAX_TOOL_CALLS", "7")
    monkeypatch.setenv("MEDRAG_BUDGET_TASK_MAX_TOOL_CALLS", "3")
    cfg = load_budget_config()
    assert cfg.global_limits.max_tool_calls == 7
    assert cfg.task_limits.max_tool_calls == 3


def test_kill_switch(monkeypatch):
    assert budget_enabled()
    monkeypatch.setenv("MEDRAG_BUDGET_ENABLED", "0")
    assert not budget_enabled()


# --- builder wiring ------------------------------------------------------------

def test_builders_wire_budget_capability_and_status_tool(monkeypatch):
    monkeypatch.setenv("LLM_BASE_URL", "http://localhost:4001/v1")
    monkeypatch.setenv("LLM_API_KEY", "test-key")
    monkeypatch.setenv("AGENT_MODEL", "test-model")
    monkeypatch.setenv("SMALL_MODEL", "test-small")
    from pydantic_ai.models.test import TestModel

    from src.agents import deep_agent as deep_agent_mod
    from src.agents import orchestrator as orchestrator_mod

    monkeypatch.setattr(deep_agent_mod, "make_model",
                        lambda *args, **kwargs: TestModel())
    monkeypatch.setattr(orchestrator_mod, "make_model",
                        lambda *args, **kwargs: TestModel())

    deep = deep_agent_mod.build_deep_agent()
    orchestrator = orchestrator_mod.build_orchestrator(
        spawn_impl=None, plan_impl=None)

    for agent, extra in ((deep, set()), (orchestrator, {"submit_plan", "spawn_subagent"})):
        tool_names = set(agent._function_toolset.tools.keys())
        assert "budget_status" in tool_names
        assert {"local_search", "lookup_medical_term"} <= tool_names | extra
        leaves = [type(c).__name__
                  for c in agent._root_capability.capabilities]
        assert "BudgetCapability" in leaves
        # Budget gates before the judge lane so denials spend no
        # hidden LLM calls either.
        assert leaves.index("BudgetCapability") < leaves.index("LLMAsJudge")


# --- orchestrator master budget + explicit spawn allocation --------------


def test_config_orchestrator_master_defaults(monkeypatch):
    for key in ("MEDRAG_BUDGET_ORCHESTRATOR_MAX_TOOL_CALLS",):
        monkeypatch.delenv(key, raising=False)
    cfg = load_budget_config()
    assert cfg.orch_limits.max_tool_calls == 80


def test_config_orchestrator_env_override(monkeypatch):
    monkeypatch.setenv("MEDRAG_BUDGET_ORCHESTRATOR_MAX_TOOL_CALLS", "11")
    cfg = load_budget_config()
    assert cfg.orch_limits.max_tool_calls == 11


def test_resolve_allocation_explicit_wins():
    from src.agents.stream_adapter import _resolve_task_allocation
    defaults = BudgetLimits(max_tool_calls=20)
    remaining = SimpleNamespace(remaining_tool_calls=80)
    limits, meta = _resolve_task_allocation(
        {"max_tool_calls": 5},
        defaults, remaining)
    assert limits.max_tool_calls == 5
    assert meta["explicit"] == ["max_tool_calls"]
    assert meta["clamped"] == []


def test_resolve_allocation_clamped_to_master_remainder():
    from src.agents.stream_adapter import _resolve_task_allocation
    defaults = BudgetLimits(max_tool_calls=20)
    remaining = SimpleNamespace(remaining_tool_calls=3)
    limits, meta = _resolve_task_allocation(
        {"max_tool_calls": 50}, defaults,
        remaining)
    # A promise can never exceed the pool it is drawn from.
    assert limits.max_tool_calls == 3
    assert meta["clamped"] == ["max_tool_calls"]


def test_resolve_allocation_non_dict_and_non_positive_fall_back():
    from src.agents.stream_adapter import _resolve_task_allocation
    defaults = BudgetLimits(max_tool_calls=20)
    for requested in (None, "x", {"max_tool_calls": 0},
                      {"max_tool_calls": -4}):
        limits, meta = _resolve_task_allocation(requested, defaults, None)
        assert limits.max_tool_calls == 20
        assert meta["explicit"] == []
        assert meta["clamped"] == []


async def test_master_budget_tree_spend_flows_up():
    """Global → Orchestrator → Task: task spend depletes the master."""
    run = _manager(tools=100)
    orch = run.spawn_child(
        "orchestrator",
        BudgetLimits(max_tool_calls=80))
    task = orch.spawn_child(
        "T1", BudgetLimits(max_tool_calls=20))
    denied = await task.try_consume(
        ActionCost(tool_calls=5), action="search")
    assert denied is None
    orch_snap = await orch.snapshot()
    run_snap = await run.snapshot()
    assert orch_snap.consumed.tool_calls == 5
    assert run_snap.consumed.tool_calls == 5
    assert orch_snap.remaining_tool_calls == 75
    # The task can never outspend the master: min(child, parent) wins.
    denied = await task.try_consume(
        ActionCost(tool_calls=1_000), action="burst")
    assert denied is not None


def test_adapter_usage_tokens_for_run_stats():
    """Run-stat token reader tolerates both provider usage shapes and
    never feeds the budget ledger (tool calls only)."""
    from src.agents.stream_adapter import _usage_tokens

    real = SimpleNamespace(usage=SimpleNamespace(input_tokens=100,
                                                 output_tokens=50))
    assert _usage_tokens(real) == (100, 50)
    legacy = SimpleNamespace(
        usage=lambda: SimpleNamespace(request_tokens=100,
                                      response_tokens=50))
    assert _usage_tokens(legacy) == (100, 50)
    assert _usage_tokens(SimpleNamespace()) == (0, 0)
    assert _usage_tokens(None) == (0, 0)


async def test_free_tools_pass_on_exhausted_budget():
    """Gap checks, synthesis, and status stay callable at zero: the
    orchestrator can always review coverage and finalize from what it
    has, no matter how often it re-checks."""
    orch = _manager(tools=1)
    capability = BudgetCapability()

    async def handler(args):
        return "ok"

    # Spend the single tool call.
    await capability.wrap_tool_execute(
        _tool_ctx(orch), call=_tool_call("spawn_subagent"),
        tool_def=None, args={}, handler=handler)
    # Tool-exhausted: priced calls deny, free calls pass —
    # repeatedly, since re-checks must never bill.
    for free_tool in ("check_gaps", "synthesize", "budget_status",
                      "run_progress", "check_gaps"):
        out = await capability.wrap_tool_execute(
            _tool_ctx(orch), call=_tool_call(free_tool),
            tool_def=None, args={}, handler=handler)
        assert out == "ok", free_tool
    denied = await capability.wrap_tool_execute(
        _tool_ctx(orch), call=_tool_call("spawn_subagent"),
        tool_def=None, args={}, handler=handler)
    assert json.loads(denied)["status"] == "budget_exhausted"


async def test_exhausted_master_budget_denies_spawn():
    """Past its master budget the orchestrator cannot spawn: the spawn
    tool call itself is denied before any delegation runs."""
    orch = _manager(tools=1)
    capability = BudgetCapability()
    calls = []

    async def handler(args):
        calls.append(args)
        return "findings"

    first = await capability.wrap_tool_execute(
        _tool_ctx(orch), call=_tool_call("spawn_subagent"),
        tool_def=None, args={"task": "t"}, handler=handler)
    assert first == "findings"
    out = await capability.wrap_tool_execute(
        _tool_ctx(orch), call=_tool_call("spawn_subagent"),
        tool_def=None, args={"task": "t2"}, handler=handler)
    denied = json.loads(out)
    assert denied["status"] == "budget_exhausted"
    assert len(calls) == 1  # the denied spawn never delegated
