"""Server-side chat store: in-memory registry + one JSON file per chat.

One state JSON per chat; every request in the conversation appends a
turn. Concurrency: one asyncio lock per chat; every mutation runs under
it, then persists atomically (tmp file + rename) so a crash never leaves
a torn write. Writers are best-effort from the agent path's perspective —
a failed write is logged, never raised into the research loop.
"""

from __future__ import annotations

import asyncio
import logging
import os
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional

from src.runstate.models import (
    MAX_TURNS,
    ChatState,
    EvidenceRecord,
    GapRecord,
    QuestionRecord,
    RequirementRecord,
    RunState,
    RunStatus,
    TaskRecord,
    TaskState,
)

logger = logging.getLogger(__name__)

# Bound stored text per passage so one giant fetch cannot bloat the file.
MAX_STORED_TEXT_CHARS = 12_000


def runstate_dir() -> Path:
    return Path(os.environ.get("MEDRAG_RUNSTATE_DIR", "chats"))


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _ref_num(ref: Any) -> int:
    digits = ref[1:] if isinstance(ref, str) and ref[:1] == 'P' else ''
    return int(digits) if digits.isdigit() else 0


def _clip(text: Any, limit: int) -> str:
    collapsed = " ".join(str(text or "").split())
    if len(collapsed) > limit:
        return collapsed[:limit] + f"… [+{len(collapsed) - limit} chars]"
    return collapsed


class RunStore:
    """Create, mutate, persist, and reload per-chat state."""

    def __init__(self, directory: Path | str | None = None) -> None:
        self._dir = Path(directory) if directory is not None else runstate_dir()
        self._chats: dict[str, ChatState] = {}
        self._locks: dict[str, asyncio.Lock] = {}

    @property
    def directory(self) -> Path:
        return self._dir

    def _path(self, chat_id: str) -> Path:
        safe = "".join(c for c in chat_id if c.isalnum() or c in ("-", "_"))[:64]
        return self._dir / f"{safe or 'chat'}.json"

    def _lock_for(self, chat_id: str) -> asyncio.Lock:
        lock = self._locks.get(chat_id)
        if lock is None:
            lock = asyncio.Lock()
            self._locks[chat_id] = lock
        return lock

    def _write(self, chat: ChatState) -> None:
        """Blocking body of a persist (always runs on a worker thread)."""
        try:
            self._dir.mkdir(parents=True, exist_ok=True)
            tmp = self._path(chat.chat_id).with_suffix(".tmp")
            tmp.write_text(chat.to_json(), encoding="utf-8")
            tmp.replace(self._path(chat.chat_id))
        except Exception as exc:  # noqa: BLE001 — persistence never breaks research
            logger.warning("runstate persist failed for %s: %s", chat.chat_id, exc)

    async def _persist(self, chat: ChatState) -> None:
        """Snapshot the ledger without blocking the event loop.

        Every mutation rewrites the whole chat and the judge records
        evidence on each tool call, so this was O(file) synchronous disk I/O
        on the streaming hot path. The write now runs in a worker thread
        (the per-chat lock still serializes writers).
        """
        await asyncio.to_thread(self._write, chat)

    # -- lifecycle ------------------------------------------------------
    async def create_turn(self, chat_id: str, turn_id: str,
                          question: str = "") -> RunState:
        """Start a new turn (one request) in the chat's ledger."""
        async with self._lock_for(chat_id):
            chat = self._chats.get(chat_id)
            if chat is None:
                chat = ChatState(chat_id=chat_id)
                self._chats[chat_id] = chat
            turn = RunState(run_id=turn_id,
                            question=" ".join(question.split()))
            chat.turns[turn_id] = turn
            chat.turn_order.append(turn_id)
            while len(chat.turn_order) > MAX_TURNS:
                chat.turns.pop(chat.turn_order.pop(0), None)
            chat.touch()
            await self._persist(chat)
            return turn

    def get_chat(self, chat_id: str) -> Optional[ChatState]:
        """Read the chat ledger (memory first, then disk)."""
        chat = self._chats.get(chat_id)
        if chat is not None:
            return chat
        try:
            raw = self._path(chat_id).read_text(encoding="utf-8")
        except OSError:
            return None
        try:
            chat = ChatState.from_json(raw)
        except Exception as exc:  # noqa: BLE001 — corrupt file reads as missing
            logger.warning("runstate load failed for %s: %s", chat_id, exc)
            return None
        self._chats[chat_id] = chat
        return chat

    def get_turn(self, chat_id: str, turn_id: str) -> Optional[RunState]:
        chat = self.get_chat(chat_id)
        return chat.turns.get(turn_id) if chat is not None else None

    async def update(self, chat_id: str,
                     fn: Callable[[ChatState], None]) -> Optional[ChatState]:
        """Apply ``fn`` to the chat ledger atomically and persist."""
        async with self._lock_for(chat_id):
            chat = self.get_chat(chat_id)
            if chat is None:
                return None
            try:
                fn(chat)
            except Exception as exc:  # noqa: BLE001 — writer bugs stay local
                logger.warning("runstate update failed for %s: %s", chat_id, exc)
                return chat
            chat.touch()
            await self._persist(chat)
            return chat

    def _turn_or_none(self, chat: ChatState) -> Optional[RunState]:
        return chat.current

    # -- writers (all no-op when the chat is unknown) ---------------------
    @staticmethod
    def _task_or_create(turn: RunState, task_id: str) -> TaskRecord:
        task = turn.task(task_id)
        if task is None:
            task = TaskRecord(id=task_id)
            turn.plan.append(task)
        return task

    @staticmethod
    def _requirement_records(raw: Any) -> list[RequirementRecord]:
        """Coerce planner-shaped requirements into ledger records."""
        recs: list[RequirementRecord] = []
        if not isinstance(raw, list):
            return recs
        for r in raw:
            if not isinstance(r, dict) or not r.get("id"):
                continue
            recs.append(RequirementRecord(
                id=str(r["id"]),
                description=str(r.get("description", ""))[:500]))
        return recs

    async def record_plan(self, chat_id: str, items: list[dict]) -> None:
        def _apply(chat: ChatState) -> None:
            turn = self._turn_or_none(chat)
            if turn is None:
                return
            # A new/changed plan is new research: prior coverage review
            # no longer holds.
            turn.gap_checked = False
            for it in items:
                if not isinstance(it, dict):
                    continue
                tid = str(it.get("id", ""))
                if not tid:
                    continue
                reqs = RunStore._requirement_records(
                    it.get("evidence_requirements", []))
                task = turn.task(tid)
                if task is None:
                    task = TaskRecord(
                        id=tid,
                        task=str(it.get("question", ""))[:500],
                        depth="deep" if it.get("deep_research", True) else "shallow")
                    task.evidence_requirements = reqs
                    turn.plan.append(task)
                else:
                    task.task = str(it.get("question", ""))[:500] or task.task
                    if reqs:
                        # Re-planning keeps already-collected evidence/gaps.
                        kept = {r.id: r for r in task.evidence_requirements}
                        for rec in reqs:
                            old = kept.get(rec.id)
                            if old is not None:
                                rec.supporting_evidence = old.supporting_evidence
                                rec.gaps = old.gaps
                        task.evidence_requirements = reqs
        await self.update(chat_id, _apply)

    async def record_task_started(self, chat_id: str, task_id: str,
                                  question: str = "", depth: str = "deep") -> None:
        def _apply(chat: ChatState) -> None:
            turn = self._turn_or_none(chat)
            if turn is None:
                return
            task = self._task_or_create(turn, task_id)
            task.task = question or task.task
            task.depth = depth
            task.state = TaskState.IN_PROGRESS
            task.started_at = task.started_at or _now_iso()
        await self.update(chat_id, _apply)

    async def record_requirements(self, chat_id: str,
                                  requirements: list[dict],
                                  task_id: Optional[str] = None) -> None:
        def _apply(chat: ChatState) -> None:
            turn = self._turn_or_none(chat)
            if turn is None:
                return
            recs = []
            for r in requirements:
                if not isinstance(r, dict) or not r.get("id"):
                    continue
                recs.append(RequirementRecord(
                    id=str(r["id"]),
                    description=str(r.get("description", ""))[:500]))
            if task_id and turn.task(task_id) is not None:
                task = turn.task(task_id)
                # Re-planning keeps already-collected evidence and gaps.
                kept = {req.id: req for req in task.evidence_requirements}
                for rec in recs:
                    old = kept.get(rec.id)
                    if old is not None:
                        rec.supporting_evidence = old.supporting_evidence
                        rec.gaps = old.gaps
                task.evidence_requirements = recs
            else:
                kept = {req.id: req for req in turn.run_requirements}
                for rec in recs:
                    old = kept.get(rec.id)
                    if old is not None:
                        rec.supporting_evidence = old.supporting_evidence
                        rec.gaps = old.gaps
                turn.run_requirements = recs
        await self.update(chat_id, _apply)

    def _requirement_or_first(self, task: TaskRecord,
                              rid: str) -> Optional[RequirementRecord]:
        for req in task.evidence_requirements:
            if req.id == rid:
                return req
        # Verdicts always name planned requirements in the real flow;
        # anything else lands on the first one rather than vanishing.
        return task.evidence_requirements[0] if task.evidence_requirements else None

    async def record_evidence(self, chat_id: str, records: list[EvidenceRecord],
                              task_id: Optional[str] = None) -> int:
        """Append kept passages under their requirements. Returns newly added."""
        added = 0

        def _apply(chat: ChatState) -> None:
            nonlocal added
            turn = self._turn_or_none(chat)
            if turn is None or not task_id:
                return
            task = self._task_or_create(turn, task_id)
            # Short citation refs (P1, ...) turn-wide: highest existing ref
            # sets the counter, legacy ref-less records backfill in plan
            # order, then these records continue the sequence. Inside the
            # per-chat lock, so parallel legs can never split a number.
            counter = 0
            for t in turn.plan:
                for req in t.evidence_requirements:
                    for rec in req.supporting_evidence:
                        if rec.ref:
                            counter = max(counter, _ref_num(rec.ref))
                        else:
                            counter += 1
                            rec.ref = f'P{counter}'
            for record in records:
                if not record.ref:
                    counter += 1
                    record.ref = f'P{counter}'
            have = {record.passage_id
                    for req in task.evidence_requirements
                    for record in req.supporting_evidence}
            for record in records:
                if not record.passage_id or record.passage_id in have:
                    continue
                targets = {str(rid) for rid in (record.requirement_ids or [])}
                placed = False
                for req in task.evidence_requirements:
                    if req.id in targets:
                        req.supporting_evidence.append(record)
                        placed = True
                if not placed:
                    # Coverage naming no planned requirement must never be
                    # attributed to an arbitrary one: that would report
                    # coverage the judge never confirmed.
                    continue
                have.add(record.passage_id)
                added += 1
        await self.update(chat_id, _apply)
        return added

    async def record_gaps(self, chat_id: str, gaps: list[dict],
                          task_id: Optional[str] = None) -> None:
        def _apply(chat: ChatState) -> None:
            turn = self._turn_or_none(chat)
            if turn is None:
                return
            task = turn.task(task_id) if task_id else None
            if task is None:
                return
            by_req: dict[str, list[GapRecord]] = {}
            for g in gaps:
                if not isinstance(g, dict):
                    continue
                rid = str(g.get("evidence_id", ""))
                by_req.setdefault(rid, []).append(GapRecord(
                    evidence_id=rid, missing=str(g.get("missing", ""))[:500]))
            for rid, recs in by_req.items():
                target = next((req for req in task.evidence_requirements
                               if req.id == rid), None)
                if target is None:
                    target = self._requirement_or_first(task, rid)
                if target is not None:
                    target.gaps = recs
        await self.update(chat_id, _apply)

    # -- human-in-the-loop clarifications ---------------------------------
    async def record_question(self, chat_id: str,
                              question: QuestionRecord) -> None:
        """Persist one asked clarification question (id-keyed)."""

        def _apply(chat: ChatState) -> None:
            chat.questions[question.id] = question

        await self.update(chat_id, _apply)

    async def record_question_answer(self, chat_id: str,
                                     question_id: str,
                                     answer: dict[str, Any]) -> bool:
        """Mark a question answered and store its answer. False when the
        question is unknown (already pruned / different chat)."""
        ok = False

        def _apply(chat: ChatState) -> None:
            nonlocal ok
            q = chat.questions.get(question_id)
            if q is None:
                return
            q.answered = True
            q.answer = answer
            ok = True

        await self.update(chat_id, _apply)
        return ok

    async def record_task_finished(
        self, chat_id: str, task_id: str, task_state: TaskState,
        budget_allocated: Optional[dict] = None,
        budget_expenditure_history: Optional[list] = None,
        budget_remaining: Optional[dict] = None,
    ) -> None:
        def _apply(chat: ChatState) -> None:
            turn = self._turn_or_none(chat)
            if turn is None:
                return
            task = self._task_or_create(turn, task_id)
            task.state = task_state
            task.budget_allocated = budget_allocated
            task.budget_expenditure_history = budget_expenditure_history or []
            task.budget_remaining = budget_remaining
            task.ended_at = _now_iso()
            # Newly resolved evidence invalidates any earlier coverage
            # review — the orchestrator must check_gaps again.
            turn.gap_checked = False
        await self.update(chat_id, _apply)

    async def mark_gap_checked(self, chat_id: str) -> bool:
        """Mark the current turn coverage-reviewed. Returns False when
        there is no current turn to mark."""
        marked = False

        def _apply(chat: ChatState) -> None:
            nonlocal marked
            turn = self._turn_or_none(chat)
            if turn is None:
                return
            turn.gap_checked = True
            marked = True
        await self.update(chat_id, _apply)
        return marked

    async def record_run_finished(
        self, chat_id: str, turn_id: str, status: RunStatus,
        error: Optional[dict] = None,
        budget: Optional[dict] = None,
    ) -> None:
        def _apply(chat: ChatState) -> None:
            turn = chat.turns.get(turn_id)
            if turn is None:
                return
            turn.status = status
            turn.error = error
            turn.budget = budget
        await self.update(chat_id, _apply)


# Module-global default store (same pattern as get_trace()): tools and
# middleware resolve the ledger through the chat id on deps, never threads.
_DEFAULT_STORE: Optional[RunStore] = None
_STORE_LOCK = threading.Lock()


def get_store() -> RunStore:
    global _DEFAULT_STORE
    if _DEFAULT_STORE is None:
        with _STORE_LOCK:
            if _DEFAULT_STORE is None:
                _DEFAULT_STORE = RunStore()
    return _DEFAULT_STORE


def set_store(store: Optional[RunStore]) -> None:
    global _DEFAULT_STORE
    _DEFAULT_STORE = store


def _ids(ctx: Any) -> tuple[Optional[str], Optional[str]]:
    deps = getattr(ctx, "deps", None)
    return getattr(deps, "chat_id", None), getattr(deps, "task_id", None)


def _item_title(item: Any) -> str:
    if not isinstance(item, dict):
        return ""
    title = str(item.get("title", "") or "")
    if title:
        return title
    meta = item.get("metadata")
    if isinstance(meta, dict):
        return str(meta.get("title", "") or "")
    return ""


async def record_evidence(
    ctx: Any,
    tool_name: str,
    items: list,
    verdicts: list,
) -> int:
    """Store judge-kept passages from an evidence-tool call. Returns new count.

    ``items`` are the raw tool-output dicts (carry text + source fields);
    ``verdicts`` are PassageVerdict-likes (carry passage_id, coverage,
    intent, reason). Only verdict-backed passages are stored — "valid
    evidence" means judged, not merely retrieved.
    """
    chat_id, task_id = _ids(ctx)
    if not chat_id or not items or not verdicts:
        return 0
    by_id: dict[str, dict] = {}
    for item in items:
        if not isinstance(item, dict):
            continue
        for key in ("chunk_id", "id", "passage_id", "citation_id", "url"):
            value = item.get(key)
            if value:
                by_id.setdefault(str(value), item)
                break
    records = []
    for verdict in verdicts:
        pid = str(getattr(verdict, "passage_id", "") or "")
        item = by_id.get(pid, {})
        records.append(EvidenceRecord(
            passage_id=pid or f"{tool_name}-{len(records)}",
            requirement_ids=[str(r) for r in (getattr(verdict, "coverage", None) or [])],
            intent=float(getattr(verdict, "intent_score", 0.0) or 0.0),
            reason=str(getattr(verdict, "reason", "") or "")[:500],
            tool=tool_name,
            document_id=str(item.get("document_id", "") or ""),
            chunk_id=str(item.get("chunk_id", "") or ""),
            url=str(item.get("url", "") or item.get("source_url", "") or ""),
            title=_item_title(item),
            section=str(item.get("section", "") or ""),
            # Same fallback chain the verifier coerces with: scraped pages
            # carry text, but web-search listings that were judged instead
            # only have snippet/content/markdown — without the fallback
            # their recorded text is empty and websites never reach the
            # synthesizer.
            text=_clip(item.get("text") or item.get("snippet")
                       or item.get("content") or item.get("markdown"),
                       MAX_STORED_TEXT_CHARS),
        ))
    try:
        return await get_store().record_evidence(chat_id, records, task_id)
    except Exception as exc:  # noqa: BLE001 — recording never breaks tools
        logger.warning("record_evidence failed for %s: %s", chat_id, exc)
        return 0


async def record_gaps(ctx: Any, gaps: list) -> None:
    chat_id, task_id = _ids(ctx)
    if not chat_id:
        return
    try:
        await get_store().record_gaps(chat_id, [dict(g) if isinstance(g, dict) else {"evidence_id": str(g), "missing": ""} for g in (gaps or [])], task_id)
    except Exception as exc:  # noqa: BLE001
        logger.warning("record_gaps failed for %s: %s", chat_id, exc)


async def record_requirements(ctx: Any, requirements: list) -> None:
    chat_id, task_id = _ids(ctx)
    if not chat_id:
        return
    try:
        await get_store().record_requirements(
            chat_id,
            [r if isinstance(r, dict) else {"id": str(getattr(r, "id", "")), "description": str(getattr(r, "description", ""))} for r in (requirements or [])],
            task_id)
    except Exception as exc:  # noqa: BLE001
        logger.warning("record_requirements failed for %s: %s", chat_id, exc)


def snapshot_chat(chat_id: str) -> Optional[dict]:
    """Current chat ledger as plain JSON-ready dict (None when unknown)."""
    if not chat_id:
        return None
    chat = get_store().get_chat(chat_id)
    return chat.model_dump() if chat is not None else None
