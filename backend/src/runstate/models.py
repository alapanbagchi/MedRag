"""Run-state schema: one JSON object per research run, persisted server-side."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class RunStatus(str, Enum):
    RUNNING = "running"
    COMPLETE = "complete"
    FAILED = "failed"


class TaskState(str, Enum):
    PLANNED = "planned"
    IN_PROGRESS = "in_progress"
    DONE = "done"
    FAILED = "failed"
    BUDGET_EXHAUSTED = "budget_exhausted"


class RequirementRecord(BaseModel):
    id: str
    description: str = ""
    # Verified passages supporting this requirement (judge-kept only).
    supporting_evidence: list[EvidenceRecord] = Field(default_factory=list)
    # What is still missing for this requirement.
    gaps: list[GapRecord] = Field(default_factory=list)

    @property
    def verified_count(self) -> int:
        return len(self.supporting_evidence)


class EvidenceRecord(BaseModel):
    """One judge-verified passage, stored once the tool call returns it.

    ``ref`` is the short citation key (P1, …) the synthesizer cites
    inline and the frontend resolves to a source card. Raw passage ids
    (corpus chunk ids, full URLs) are un-citable — no model reproduces
    them verbatim — so the store assigns refs turn-wide at record time.
    """

    passage_id: str
    ref: str = ""
    requirement_ids: list[str] = Field(default_factory=list)
    intent: float = 0.0
    reason: str = ""
    tool: str = ""
    document_id: str = ""
    chunk_id: str = ""
    url: str = ""
    title: str = ""
    section: str = ""
    text: str = ""


class GapRecord(BaseModel):
    evidence_id: str
    missing: str = ""


class TaskRecord(BaseModel):
    id: str
    task: str = ""
    depth: str = "deep"
    state: TaskState = TaskState.PLANNED
    # The deliverable: one entry per evidence requirement, each with
    # its verified supporting passages and its open gaps.
    evidence_requirements: list[RequirementRecord] = Field(default_factory=list)
    # Task resource ledger, snapshotted from the task's budget manager.
    budget_allocated: Optional[dict[str, Any]] = None
    budget_expenditure_history: list[dict[str, Any]] = Field(default_factory=list)
    budget_remaining: Optional[dict[str, Any]] = None
    started_at: str = ""
    ended_at: str = ""

    @property
    def verified_count(self) -> int:
        return sum(req.verified_count for req in self.evidence_requirements)

    def coverage_by_requirement(self) -> dict[str, int]:
        return {req.id: req.verified_count for req in self.evidence_requirements}

    def open_gaps(self) -> list[str]:
        return [gap.evidence_id
                for req in self.evidence_requirements for gap in req.gaps]


class QuestionRecord(BaseModel):
    """One human-in-the-loop clarification question, asked via ask_user.

    Persisted per chat (not per turn) so the ledger and the UI can track
    every clarification regardless of which agent asked it.
    """

    id: str
    text: str = ""
    options: list[str] = Field(default_factory=list)
    multi_select: bool = True
    allow_other: bool = True
    allow_find_all: bool = True
    context: str = ""
    # Which agent asked ("" = orchestrator, "T1"/"S2" = a sub-agent leg).
    by_task: str = ""
    asked_at: str = Field(default_factory=_now_iso)
    answered: bool = False
    answer: Optional[dict[str, Any]] = None


class RunState(BaseModel):
    run_id: str
    question: str = ""
    status: RunStatus = RunStatus.RUNNING
    created_at: str = Field(default_factory=_now_iso)
    updated_at: str = Field(default_factory=_now_iso)
    # The plan IS the task list: each item carries its own state,
    # requirements, evidence, gaps, findings, and budget.
    plan: list[TaskRecord] = Field(default_factory=list)
    run_requirements: list[RequirementRecord] = Field(default_factory=list)
    # Coverage gate: set when check_gaps runs, cleared by any later plan
    # change or task finish. The synthesizer only runs while this holds —
    # it means "coverage reviewed since the last task resolution".
    gap_checked: bool = False
    budget: Optional[dict[str, Any]] = None
    error: Optional[dict[str, Any]] = None

    @property
    def run_verified_count(self) -> int:
        return sum(task.verified_count for task in self.plan)

    def task(self, task_id: str) -> Optional[TaskRecord]:
        for task in self.plan:
            if task.id == task_id:
                return task
        return None

    def touch(self) -> None:
        self.updated_at = _now_iso()

    def to_json(self) -> str:
        # Compact: the ledger is machine-read only, and every mutation
        # rewrites the whole file, so pretty-printing was pure write volume.
        return self.model_dump_json()

    @classmethod
    def from_json(cls, raw: str) -> RunState:
        return cls.model_validate_json(raw)


class ChatState(BaseModel):
    """One state JSON per chat: every turn (request) in the conversation.

    Turns accumulate oldest-first; beyond MAX_TURNS only the newest are
    kept so the file stays bounded.
    """

    chat_id: str
    created_at: str = Field(default_factory=_now_iso)
    updated_at: str = Field(default_factory=_now_iso)
    turn_order: list[str] = Field(default_factory=list)
    turns: dict[str, RunState] = Field(default_factory=dict)
    # Human-in-the-loop clarifications asked during any turn, keyed by
    # question id. Answered ones stay (audit trail) with their answer.
    questions: dict[str, QuestionRecord] = Field(default_factory=dict)

    @property
    def current(self) -> Optional[RunState]:
        if not self.turn_order:
            return None
        return self.turns.get(self.turn_order[-1])

    def touch(self) -> None:
        self.updated_at = _now_iso()

    def to_json(self) -> str:
        # Compact: the ledger is machine-read only, and every mutation
        # rewrites the whole file, so pretty-printing was pure write volume.
        return self.model_dump_json()

    @classmethod
    def from_json(cls, raw: str) -> ChatState:
        return cls.model_validate_json(raw)


MAX_TURNS = 20
