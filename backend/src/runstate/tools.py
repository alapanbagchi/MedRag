"""Agent-facing progress tool: read the chat ledger, never the raw outputs."""

from __future__ import annotations

import json
from typing import Any

from pydantic_ai import RunContext

from src.runstate.progress import progress_json, task_detail
from src.runstate.store import get_store


async def run_progress(ctx: RunContext[Any], task_id: str | None = None) -> str:
    """Report chat progress from the shared state ledger (free to call).

    Without arguments: compact JSON for the current turn — per-task
    state, per-requirement verified counts and open gaps, unresolved
    gaps, and earlier-turn summaries. Enough to decide what to do next;
    call this, not more research.

    With task_id: full detail for ONE task, including every
    supporting-evidence passage with text. Pull this only for tasks you
    are synthesizing the final answer from — never to decide whether
    more research is needed. Read-only: the ledger is written by the
    execution layer, never by this tool.
    """
    chat_id = getattr(getattr(ctx, "deps", None), "chat_id", None)
    if not chat_id:
        return "No run state is attached to this run."
    chat = get_store().get_chat(chat_id)
    if chat is None:
        return f"No run state found for chat {chat_id}."
    if task_id:
        turn = chat.current
        task = turn.task(task_id) if turn is not None else None
        if task is None:
            return f"No task {task_id} in the current turn."
        return json.dumps(task_detail(task))
    return progress_json(chat)


async def check_gaps(ctx: RunContext[Any]) -> str:
    """List uncovered requirements across the current turn (free to call).

    Mechanical, no LLM: a requirement is covered when at least one
    judge-verified passage supports it, uncovered otherwise. Call this
    after all task subagents have resolved; every entry is a candidate
    follow-up spawn (budget permitting), otherwise a declared gap in
    the final answer. Calling this marks the turn gap-checked, which is
    the gate for the synthesizer — any later plan change or task finish
    clears the mark, so a stale review can never unlock synthesis.
    """
    chat_id = getattr(getattr(ctx, "deps", None), "chat_id", None)
    if not chat_id:
        return "No run state is attached to this run."
    store = get_store()
    chat = store.get_chat(chat_id)
    if chat is None:
        return f"No run state found for chat {chat_id}."
    turn = chat.current
    if turn is None:
        return json.dumps([])
    await store.mark_gap_checked(chat_id)
    uncovered = [
        {"task_id": task.id,
         "task_state": task.state.value,
         "requirement_id": req.id,
         "description": req.description}
        for task in turn.plan
        for req in task.evidence_requirements
        if not req.supporting_evidence
    ]
    return json.dumps(uncovered)


__all__ = ["run_progress", "check_gaps"]
