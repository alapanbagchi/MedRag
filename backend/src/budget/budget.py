"""Core budget abstraction: tool-call limits, consumption, snapshots, manager.

One dimension only: tool calls. Every tool invocation costs exactly what
its cost model says (usually 1, sometimes 0 for free reads) — tokens,
model calls, wall-clock time, and money never factor in.

Concurrency model: every mutating operation acquires the locks of the
whole ancestor chain (root first, leaf last — a fixed global order, so no
deadlock) and performs check-and-deduct atomically. Two tasks racing on
the last unit cannot both win: exactly one reservation succeeds.

The manager exposes no way to raise limits or reset consumption. The only
upward flow is ``release()`` (cancel an unused reservation) and
``reclaim_child()`` (a parent reabsorbs a finished child's spare capacity
for reallocation). Agents holding a child manager therefore cannot grant
themselves more resources.
"""

from __future__ import annotations

import asyncio
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional


class BudgetState(str, Enum):
    NORMAL = "normal"
    LOW = "low"
    CRITICAL = "critical"
    EXHAUSTED = "exhausted"


@dataclass(frozen=True)
class BudgetLimits:
    """Maximum tool calls. ``None`` = unbounded."""

    max_tool_calls: Optional[int] = None


@dataclass(frozen=True)
class ActionCost:
    """What one action (tool call, spawn) costs, in tool calls."""

    tool_calls: int = 0


@dataclass
class BudgetConsumption:
    """Running tool-call total consumed through this manager (own + descendants)."""

    tool_calls: int = 0

    def add(self, cost: ActionCost) -> None:
        self.tool_calls += cost.tool_calls


@dataclass(frozen=True)
class BudgetSnapshot:
    """Read-only view handed to agents — observability without control."""

    manager_id: str
    task_id: str
    consumed: BudgetConsumption
    limits: BudgetLimits
    remaining_tool_calls: Optional[int]
    state: BudgetState
    limiting_resource: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "manager_id": self.manager_id,
            "task_id": self.task_id,
            "state": self.state.value,
            "limiting_resource": self.limiting_resource,
            "consumed": {
                "tool_calls": self.consumed.tool_calls,
            },
            "remaining": {
                "tool_calls": self.remaining_tool_calls,
            },
        }


@dataclass(frozen=True)
class BudgetDenied:
    """Structured denial — normal control flow, not a system failure."""

    resource: str
    remaining: Optional[float]
    message: str
    task_id: str = ""
    action: str = ""

    def to_result_json(self) -> str:
        import json

        return json.dumps({
            "status": "budget_exhausted",
            "resource": self.resource,
            "remaining": self.remaining,
            "task_id": self.task_id,
            "action": self.action,
            "message": self.message,
        })


@dataclass
class Reservation:
    """A held-but-not-yet-consumed grant. Must be consumed or released."""

    reservation_id: str
    manager_id: str
    task_id: str
    action: str
    cost: ActionCost
    settled: bool = False


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class BudgetManager:
    """Hierarchical, concurrency-safe tool-call ledger.

    ``parent`` links a task budget to the run-global budget. Children are
    created with :meth:`spawn_child`, which takes an *allocation* carved
    out of the parent's own limits (the child limit is additionally capped
    by whatever the parent still has left at spend time).
    """

    def __init__(
        self,
        limits: BudgetLimits,
        *,
        task_id: str = "run",
        parent: Optional[BudgetManager] = None,
        low_threshold: float = 0.40,
        critical_threshold: float = 0.20,
    ) -> None:
        self.manager_id = uuid.uuid4().hex[:12]
        self.task_id = task_id
        self.limits = limits
        self.low_threshold = low_threshold
        self.critical_threshold = critical_threshold
        self._parent = parent
        self._children: list[BudgetManager] = []
        self._consumed = BudgetConsumption()
        self._reserved = BudgetConsumption()  # held by open reservations
        self._reservations: dict[str, Reservation] = {}
        self._history: list[dict[str, Any]] = []
        self._lock = asyncio.Lock()
        self._started = time.monotonic()
        self._exhausted_at: Optional[str] = None
        self._last_state = BudgetState.NORMAL
        self._history.append({
            "event": "budget_created",
            "manager_id": self.manager_id,
            "task_id": task_id,
            "limits": _limits_dict(limits),
            "timestamp": _now_iso(),
        })

    # -- hierarchy ------------------------------------------------------
    @property
    def parent(self) -> Optional[BudgetManager]:
        return self._parent

    def spawn_child(self, task_id: str, limits: BudgetLimits) -> BudgetManager:
        """Allocate a task budget under this manager.

        The child is a live view onto the parent chain, not a copy: every
        spend is checked against ``min(child_remaining, parent_remaining)``
        at spend time, so the parent ceiling always wins.
        """
        child = BudgetManager(
            limits,
            task_id=task_id,
            parent=self,
            low_threshold=self.low_threshold,
            critical_threshold=self.critical_threshold,
        )
        self._children.append(child)
        self._history.append({
            "event": "budget_allocated",
            "manager_id": self.manager_id,
            "child_id": child.manager_id,
            "child_task_id": task_id,
            "limits": _limits_dict(limits),
            "timestamp": _now_iso(),
        })
        self._trace("budget_allocated", child_task_id=task_id,
                    limits=_limits_dict(limits))
        return child

    def _chain(self) -> list[BudgetManager]:
        node, chain = self, []
        while node is not None:
            chain.append(node)
            node = node._parent
        chain.reverse()  # root first: fixed lock order, no deadlock
        return chain

    # -- core operations ------------------------------------------------
    def _check_locked(self, cost: ActionCost) -> Optional[BudgetDenied]:
        """Deny reason for ``cost`` at current (locked) state, or None.

        Must be called with the full ancestor chain already locked.
        """
        for manager in self._chain():
            denied = manager._check_own(cost)
            if denied is not None:
                return denied
        return None

    def _check_own(self, cost: ActionCost) -> Optional[BudgetDenied]:
        lim = self.limits
        used = self._consumed
        held = self._reserved
        if lim.max_tool_calls is not None and used.tool_calls + held.tool_calls + cost.tool_calls > lim.max_tool_calls:
            return BudgetDenied("tool_calls", lim.max_tool_calls - used.tool_calls - held.tool_calls,
                                "Tool-call budget exhausted.", self.task_id)
        return None

    async def can_afford(self, cost: ActionCost) -> Optional[BudgetDenied]:
        """Read-only affordability probe (no reservation held)."""
        chain = self._chain()
        for manager in chain:
            await manager._lock.acquire()
        try:
            return self._check_locked(cost)
        finally:
            for manager in reversed(chain):
                manager._lock.release()

    async def reserve(self, cost: ActionCost, *, action: str = "") -> Reservation | BudgetDenied:
        """Atomically check ``min(child, parent)`` and hold the grant."""
        chain = self._chain()
        for manager in chain:
            await manager._lock.acquire()
        try:
            denied = self._check_locked(cost)
            if denied is not None:
                denied = BudgetDenied(denied.resource, denied.remaining,
                                      denied.message, self.task_id, action)
                self._record_denied(cost, action, denied)
                return denied
            reservation = Reservation(
                reservation_id=uuid.uuid4().hex[:12],
                manager_id=self.manager_id,
                task_id=self.task_id,
                action=action,
                cost=cost,
            )
            for manager in chain:
                manager._reserved.add(cost)
                manager._reservations[reservation.reservation_id] = reservation
            self._history.append({
                "event": "budget_reserved",
                "manager_id": self.manager_id,
                "task_id": self.task_id,
                "action": action,
                "cost": _cost_dict(cost),
                "timestamp": _now_iso(),
            })
            return reservation
        finally:
            for manager in reversed(chain):
                manager._lock.release()

    async def consume(self, reservation: Reservation) -> Optional[BudgetDenied]:
        """Finalize a reservation into real consumption."""
        if reservation.settled:
            return BudgetDenied("reservation", 0, "Reservation already settled.",
                                self.task_id, reservation.action)
        if reservation.manager_id != self.manager_id:
            return BudgetDenied("reservation", 0,
                                "Reservation belongs to another budget.",
                                self.task_id, reservation.action)
        chain = self._chain()
        for manager in chain:
            await manager._lock.acquire()
        try:
            for manager in chain:
                manager._reserved.tool_calls -= reservation.cost.tool_calls
                manager._consumed.add(reservation.cost)
                manager._reservations.pop(reservation.reservation_id, None)
            reservation.settled = True
            self._history.append({
                "event": "budget_consumed",
                "manager_id": self.manager_id,
                "task_id": self.task_id,
                "action": reservation.action,
                "cost": _cost_dict(reservation.cost),
                "timestamp": _now_iso(),
                "remaining": self._remaining_locked(),
            })
            self._trace("budget_consumed", task_id=self.task_id,
                        action=reservation.action, cost=_cost_dict(reservation.cost))
            self._refresh_state_locked()
            return None
        finally:
            for manager in reversed(chain):
                manager._lock.release()

    async def try_consume(self, cost: ActionCost, *, action: str = "") -> Optional[BudgetDenied]:
        """Atomic check-and-spend for actions that cannot hold a reservation."""
        reservation = await self.reserve(cost, action=action)
        if isinstance(reservation, BudgetDenied):
            return reservation
        return await self.consume(reservation)

    async def release(self, reservation: Reservation) -> None:
        """Cancel an unsettled reservation, freeing the held grant."""
        chain = self._chain()
        for manager in chain:
            await manager._lock.acquire()
        try:
            self._cancel_reservation_locked(reservation)
        finally:
            for manager in reversed(chain):
                manager._lock.release()

    def _cancel_reservation_locked(self, reservation: Reservation) -> None:
        if reservation.settled:
            return
        for manager in self._chain():
            manager._reserved.tool_calls -= reservation.cost.tool_calls
            manager._reservations.pop(reservation.reservation_id, None)
        reservation.settled = True

    async def reclaim_child(self, child: BudgetManager) -> dict[str, Any]:
        """Reabsorb a finished child's *allocation headroom* for reallocation.

        Consumption already flowed upward at spend time, so there is no
        double counting: this only narrows the child's limits to what it
        actually used, freeing the unused allocation back into the parent's
        headroom for sibling tasks (future dynamic reallocation).
        Returns the freed headroom.
        """
        if child._parent is not self:
            raise ValueError("can only reclaim a direct child budget")
        freed = {
            "tool_calls": _narrow(child.limits.max_tool_calls, child._consumed.tool_calls),
        }
        child.limits = BudgetLimits(
            max_tool_calls=_used_or(child.limits.max_tool_calls, child._consumed.tool_calls),
        )
        self._history.append({
            "event": "budget_reclaimed",
            "manager_id": self.manager_id,
            "child_id": child.manager_id,
            "child_task_id": child.task_id,
            "freed": freed,
            "timestamp": _now_iso(),
        })
        return freed

    # -- read-only views --------------------------------------------------
    async def ledger_snapshot(self) -> dict[str, Any]:
        """Per-task resource ledger: allocation, spend history, remainder.

        ``allocated`` is this manager's own limits; ``expenditure`` is
        its consumption/denial events in order; ``remaining`` is
        the effective remainder (min over the ancestor chain).
        """
        chain = self._chain()
        for manager in chain:
            await manager._lock.acquire()
        try:
            return {
                "allocated": _limits_dict(self.limits),
                "expenditure": [
                    dict(event) for event in self._history
                    if event.get("event") in (
                        "budget_consumed", "budget_denied")
                ],
                "remaining": self._remaining_locked(),
            }
        finally:
            for manager in reversed(chain):
                manager._lock.release()

    async def snapshot(self) -> BudgetSnapshot:
        chain = self._chain()
        for manager in chain:
            await manager._lock.acquire()
        try:
            return self._snapshot_locked()
        finally:
            for manager in reversed(chain):
                manager._lock.release()

    def _snapshot_locked(self) -> BudgetSnapshot:
        remaining = self._remaining_locked()
        state, limiting = self._state_locked()
        lim = self.limits
        return BudgetSnapshot(
            manager_id=self.manager_id,
            task_id=self.task_id,
            consumed=BudgetConsumption(
                tool_calls=self._consumed.tool_calls,
            ),
            limits=lim,
            remaining_tool_calls=remaining["tool_calls"],
            state=state,
            limiting_resource=limiting,
        )

    def _remaining_locked(self) -> dict[str, Any]:
        """Remaining tool calls = min over the ancestor chain (own + parents)."""
        out: dict[str, Any] = {"tool_calls": None}
        for manager in self._chain():
            lim, used, held = manager.limits, manager._consumed, manager._reserved
            out["tool_calls"] = _min_opt(out["tool_calls"], _sub_opt(lim.max_tool_calls, used.tool_calls + held.tool_calls))
        return out

    def _state_locked(self) -> tuple[BudgetState, str]:
        """State from the remaining/max fraction over the ancestor chain.

        Each level contributes its *own* remaining/max fraction, so a
        nearly-spent parent degrades the child's state even when the
        child's allocation looks healthy.
        """
        fractions: list[tuple[str, float]] = []

        def _frac(rem: Any, maximum: Any, name: str) -> None:
            if rem is not None and maximum:
                fractions.append((name, max(0.0, rem / maximum)))

        for manager in self._chain():
            lim, used, held = manager.limits, manager._consumed, manager._reserved
            _frac(_sub_opt(lim.max_tool_calls, used.tool_calls + held.tool_calls),
                  lim.max_tool_calls, "tool_calls")
        if not fractions:
            return BudgetState.NORMAL, "none"
        limiting, worst = min(fractions, key=lambda kv: kv[1])
        if worst <= 0:
            return BudgetState.EXHAUSTED, limiting
        if worst < self.critical_threshold:
            return BudgetState.CRITICAL, limiting
        if worst < self.low_threshold:
            return BudgetState.LOW, limiting
        return BudgetState.NORMAL, limiting

    def _refresh_state_locked(self) -> None:
        state, _ = self._state_locked()
        if state != self._last_state:
            previous = self._last_state
            self._last_state = state
            self._history.append({
                "event": "budget_state_transition",
                "manager_id": self.manager_id,
                "task_id": self.task_id,
                "from": previous.value,
                "to": state.value,
                "timestamp": _now_iso(),
            })
            self._trace("budget_state_transition", task_id=self.task_id,
                        previous=previous.value, current=state.value)
            if state == BudgetState.EXHAUSTED and self._exhausted_at is None:
                self._exhausted_at = _now_iso()

    def _record_denied(self, cost: ActionCost, action: str, denied: BudgetDenied) -> None:
        self._history.append({
            "event": "budget_denied",
            "manager_id": self.manager_id,
            "task_id": self.task_id,
            "action": action,
            "resource": denied.resource,
            "remaining": denied.remaining,
            "timestamp": _now_iso(),
        })
        self._trace("budget_denied", task_id=self.task_id, action=action,
                    resource=denied.resource)
        self._refresh_state_locked()

    # -- persistence ------------------------------------------------------
    @property
    def history(self) -> list[dict[str, Any]]:
        return list(self._history)

    @property
    def exhausted_at(self) -> Optional[str]:
        return self._exhausted_at

    def to_dict(self) -> dict[str, Any]:
        """Serializable run/task ledger: initial limits, consumption, history."""
        return {
            "manager_id": self.manager_id,
            "task_id": self.task_id,
            "limits": _limits_dict(self.limits),
            "consumed": {
                "tool_calls": self._consumed.tool_calls,
            },
            "exhausted_at": self._exhausted_at,
            "children": [child.to_dict() for child in self._children],
            "history": list(self._history),
        }

    @staticmethod
    def _trace(event: str, **fields: Any) -> None:
        try:
            from src.lib.trace import get_trace
            get_trace().log(event, **fields)
        except Exception:
            pass


def _limits_dict(lim: BudgetLimits) -> dict[str, Any]:
    return {
        "max_tool_calls": lim.max_tool_calls,
    }


def _cost_dict(cost: ActionCost) -> dict[str, Any]:
    return {
        "tool_calls": cost.tool_calls,
    }


def _sub_opt(limit: Any, used: Any) -> Any:
    if limit is None:
        return None
    return limit - used


def _min_opt(current: Any, candidate: Any) -> Any:
    if candidate is None:
        return current
    if current is None:
        return candidate
    return min(current, candidate)


def _narrow(limit: Any, used: Any) -> Any:
    if limit is None:
        return None
    return max(0, limit - used)


def _used_or(limit: Any, used: Any) -> Any:
    if limit is None:
        return None
    return max(used, 0)
