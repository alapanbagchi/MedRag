"""Compact progress views: what the orchestrator reads instead of full outputs.

Counts, states, and unresolved gaps for the current turn, plus one-line
summaries of earlier turns in the chat — never evidence text. The agent
steers from "T1 has 4 verified passages covering E1–E3 with E4 still
open", not by re-reading every passage.
"""

from __future__ import annotations

import json
from typing import Any

from src.runstate.models import ChatState, RunState, TaskState


def requirement_view(req: Any) -> dict[str, Any]:
    return {
        "id": req.id,
        "verified": req.verified_count,
        "gaps": [gap.evidence_id for gap in req.gaps],
    }


def turn_view(turn: RunState) -> dict[str, Any]:
    plan = []
    unresolved: list[dict[str, Any]] = []
    for task in turn.plan:
        requirements = [requirement_view(req)
                        for req in task.evidence_requirements]
        plan.append({
            "id": task.id,
            "task": task.task,
            "state": task.state.value,
            "depth": task.depth,
            "verified": task.verified_count,
            "evidence_requirements": requirements,
        })
        for req in task.evidence_requirements:
            for gap in req.gaps:
                unresolved.append({
                    "task_id": task.id,
                    "evidence_id": gap.evidence_id,
                    "missing": gap.missing,
                })
    return {
        "turn_id": turn.run_id,
        "status": turn.status.value,
        "gap_checked": turn.gap_checked,
        "plan": plan,
        "turn_verified": turn.run_verified_count,
        "unresolved_gaps": unresolved,
    }


def task_detail(task: Any) -> dict[str, Any]:
    """Full detail for one task, including supporting evidence texts.

    Pull this only when synthesizing — never to decide whether more
    research is needed; the compact view already answers that.
    """
    return {
        "id": task.id,
        "task": task.task,
        "state": task.state.value,
        "depth": task.depth,
        "verified": task.verified_count,
        "evidence_requirements": [
            {
                "id": req.id,
                "description": req.description,
                "supporting_evidence": [record.model_dump()
                                        for record in req.supporting_evidence],
                "gaps": [gap.model_dump() for gap in req.gaps],
            }
            for req in task.evidence_requirements
        ],
        "budget_allocated": task.budget_allocated,
        "budget_expenditure_history": task.budget_expenditure_history,
        "budget_remaining": task.budget_remaining,
    }


def progress_view(chat: ChatState, turn_id: str | None = None) -> dict[str, Any]:
    """Small JSON-ready summary of chat progress (no evidence text)."""
    turn = chat.turns.get(turn_id) if turn_id else chat.current
    if turn is None:
        return {"chat_id": chat.chat_id, "status": "empty", "plan": [],
                "turn_verified": 0, "unresolved_gaps": [],
                "previous_turns": []}
    view = turn_view(turn)
    view["chat_id"] = chat.chat_id
    view["previous_turns"] = [
        {"turn_id": tid,
         "question": chat.turns[tid].question[:160],
         "status": chat.turns[tid].status.value,
         "verified": chat.turns[tid].run_verified_count}
        for tid in chat.turn_order if tid != turn.run_id and tid in chat.turns
    ]
    return view


def render_receipt(chat: ChatState, turn_id: str, task_id: str) -> str:
    """Compact spawn return: arrays and flags only, never findings text.

    The orchestrator routes on this (coverage massive + gaps empty =
    stop) without polluting its context. Full findings are pulled on
    demand via run_progress(task_id) when synthesizing the final answer.
    """
    turn = chat.turns.get(turn_id)
    task = turn.task(task_id) if turn is not None else None
    if task is None:
        return ""
    coverage = task.coverage_by_requirement()
    covered = ",".join(f"{rid}:{count}" for rid, count in sorted(coverage.items()))
    gaps = ",".join(task.open_gaps()) or "none"
    state_note = ("" if task.state == TaskState.DONE
                  else f" task_state={task.state.value}")
    total = turn.run_verified_count if turn is not None else 0
    return (
        f"[RUN STATE] {task_id}: "
        f"{task.verified_count} verified passage(s)"
        f" ({covered or 'no requirement coverage'})"
        f" · gaps: {gaps} · run total: {total}{state_note}"
    )


def progress_json(chat: ChatState, turn_id: str | None = None) -> str:
    return json.dumps(progress_view(chat, turn_id))
