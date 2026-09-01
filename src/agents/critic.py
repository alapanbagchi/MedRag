"""Agentic v3 - Stage 8: the CRITIC verification gate (one LLM call per passage)."""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

from pydantic import BaseModel

from src.agentic.state import (
    AnswersTask,
    CriticRelevance,
    CriticVerdict,
    EvidenceRequirement,
    ResearchTask,
    RetrievedPaper,
    SupportDirection,
)
from src.prompts.load import load_prompt

logger = logging.getLogger("src.agents.critic")


class CriticOutput(BaseModel):
    """Raw LLM verdict for ONE passage against ONE requirement."""
    relevance: str = "not_relevant"
    answers_task: str = "no"
    support: str = "neutral"
    confidence: float = 0.0
    note: str = ""


CRITIC_SYSTEM_PROMPT = load_prompt('agents', 'critic.txt')

_RELEVANCE = {r.value: r for r in CriticRelevance}
_ANSWERS = {a.value: a for a in AnswersTask}
_SUPPORT = {s.value: s for s in SupportDirection}

_MAX_PASSAGES_PER_ROUND = 6

# 1 req/s pacing for free-tier rate limits. ponytail: flat 1 req/s; lower
# _PACE_SECONDS once the tier allows.
_PACE_SECONDS = 1.0
_pace_lock = asyncio.Lock()
_pace_next = 0.0


async def _pace() -> None:
    global _pace_next
    async with _pace_lock:
        now = time.monotonic()
        target = max(now, _pace_next)
        _pace_next = target + _PACE_SECONDS
    wait = target - time.monotonic()
    if wait > 0:
        await asyncio.sleep(wait)


def evidence_excerpt(text: str, max_chars: int = 1600) -> str:
    """Whitespace-collapsed bounded excerpt with an explicit marker."""
    flat = " ".join((text or "").split())
    return flat if len(flat) <= max_chars else flat[:max_chars] + "... [excerpt truncated]"


def is_promising_for_deep_inspection(verdict: CriticVerdict) -> bool:
    """A paper 'that might have something in it' (partial topic, did not answer)."""
    if verdict.accepted:
        return False
    return (verdict.relevance in (CriticRelevance.PARTIALLY_RELEVANT, CriticRelevance.RELEVANT)
            and verdict.answers_task in (AnswersTask.NO, AnswersTask.PARTIAL))


class CriticAgent:
    """The verification gate: fixed scope first, then one LLM verdict per passage."""

    def __init__(self, config: Any = None, model: Any = None):
        from pydantic_ai import Agent

        from src.config import AppConfig
        from src.llm import build_model_for

        self.config = config or AppConfig()
        self.model = model or build_model_for(self.config, role="verifier")
        self.agent = Agent(self.model, system_prompt=CRITIC_SYSTEM_PROMPT,
                           output_type=CriticOutput, retries=2, name="critic")

    @staticmethod
    def _check_scope(task: ResearchTask, requirement: EvidenceRequirement,
                     paper: RetrievedPaper) -> None:
        """Hard isolation guard: scope must match the caller."""
        if (paper.task_id or "").strip() and paper.task_id != task.id:
            raise ValueError(f"critic scope mismatch: paper.task_id={paper.task_id!r} != task.id={task.id!r}")
        if (paper.requirement_id or "").strip() and paper.requirement_id != requirement.id:
            raise ValueError(f"critic scope mismatch: paper.requirement_id={paper.requirement_id!r} "
                             f"!= requirement.id={requirement.id!r}")

    @staticmethod
    def _scope(task: ResearchTask, requirement: EvidenceRequirement,
               paper: RetrievedPaper, *, run_id: str, attempt_id: str) -> dict:
        return {
            "run_id": run_id or "", "task_id": task.id,
            "requirement_id": requirement.id,
            "attempt_id": attempt_id or paper.attempt_id or "",
            "evidence_id": paper.evidence_id or "",
            "document_id": paper.document_id, "chunk_id": paper.chunk_id,
            "section": paper.section,
        }

    def _prompt(self, task: ResearchTask, requirement: EvidenceRequirement,
                paper: RetrievedPaper) -> str:
        text = (paper.text or "").strip()
        return (
            f"WORKER TASK {task.id}: {task.title}\nTASK OBJECTIVE: {task.objective}\n"
            f"EVIDENCE REQUIREMENT {requirement.id}: {requirement.text}\n"
            f"\nCANDIDATE PASSAGE (evidence={paper.evidence_id or '?'}, "
            f"document={paper.document_id}, chunk={paper.chunk_id}, "
            f"section={paper.section}, type={paper.unit_kind}):\n{text}\n"
            "\nJudge whether this passage ACTUALLY ANSWERS the evidence requirement."
        )

    @staticmethod
    def _apply_output(scope: dict, out: CriticOutput) -> CriticVerdict:
        """Map the raw LLM verdict onto a scope-fixed CriticVerdict."""
        relevance = _RELEVANCE.get((out.relevance or "").strip().lower(), CriticRelevance.NOT_RELEVANT)
        answers = _ANSWERS.get((out.answers_task or "").strip().lower(), AnswersTask.NO)
        support = _SUPPORT.get((out.support or "").strip().lower(), SupportDirection.NEUTRAL)
        try:
            confidence = max(0.0, min(1.0, float(out.confidence or 0.0)))
        except (TypeError, ValueError):
            confidence = 0.0
        return CriticVerdict(**scope, relevance=relevance, answers_task=answers,
                             support=support, confidence=confidence,
                             note=(out.note or "").strip())

    async def judge(self, task: ResearchTask, requirement: EvidenceRequirement,
                    paper: RetrievedPaper, *, run_id: str = "",
                    attempt_id: str = "") -> CriticVerdict:
        """Judge ONE passage. Failure != irrelevance: it raises."""
        self._check_scope(task, requirement, paper)
        scope = self._scope(task, requirement, paper, run_id=run_id, attempt_id=attempt_id)
        try:
            out = await self.agent.run(self._prompt(task, requirement, paper))
            return self._apply_output(scope, out.output)
        except Exception as exc:
            logger.warning("critic call failed for %s (%s)",
                           paper.document_id or paper.chunk_id, exc)
            raise

    async def judge_papers(self, task: ResearchTask, requirement: EvidenceRequirement,
                           papers: list[RetrievedPaper], *, run_id: str = "",
                           attempt_id: str = "") -> list[tuple]:
        """Judge all passages in parallel, 1 req/s; None on failure (never a rejection)."""
        papers = list(papers)[:_MAX_PASSAGES_PER_ROUND]
        if not papers:
            return []

        async def _judge_one(paper: RetrievedPaper) -> tuple:
            await _pace()
            try:
                return (paper, await self.judge(task, requirement, paper,
                                                run_id=run_id, attempt_id=attempt_id))
            except Exception as exc:  # noqa: BLE001
                logger.warning("critic skipped %s: %s", paper.document_id or paper.chunk_id, exc)
                return (paper, None)

        return list(await asyncio.gather(*[_judge_one(p) for p in papers]))