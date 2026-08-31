"""Critic parallel verification: single requests, 1 req/s pacing (no batch)."""

from __future__ import annotations

import asyncio
import time

from src.agents.critic import CriticAgent, _PACE_SECONDS
from src.agents.state import (
    EvidenceRequirement,
    EvidenceStatus,
    ResearchTask,
    RetrievedPaper,
)
from src.config import AppConfig


def _task() -> ResearchTask:
    return ResearchTask(
        id="T1", title="eggs and cardiovascular disease",
        objective="does egg intake affect CVD risk", intent="association evidence",
        evidence_requirements=[
            EvidenceRequirement(id="T1.R1", text="egg consumption and CVD",
                                target_n=2)],
    )


def _paper(doc: str) -> RetrievedPaper:
    return RetrievedPaper(
        evidence_id=f"E-{doc}", task_id="T1", requirement_id="T1.R1",
        attempt_id="A1", status=EvidenceStatus.RETRIEVED,
        document_id=doc, chunk_id=f"c-{doc}", section="Results",
        score=0.9, text="A passage about eggs and cardiovascular disease outcomes.",
    )


async def test_judge_papers_parallel_paced_in_order():
    """All papers are judged (each its own single request), started ~1/s apart,
    results in input order."""
    agent = CriticAgent(config=AppConfig())
    starts: list[float] = []

    async def fake_judge(task, requirement, paper, *, run_id="", attempt_id=""):
        starts.append(time.monotonic())
        await asyncio.sleep(0.001)
        return paper.document_id

    agent.judge = fake_judge  # type: ignore[method-assign]
    task = _task()
    papers = [_paper("PMC1"), _paper("PMC2"), _paper("PMC3")]
    out = await agent.judge_papers(task, task.evidence_requirements[0], papers)

    assert [p.document_id for p, _ in out] == ["PMC1", "PMC2", "PMC3"]
    assert [v for _, v in out] == ["PMC1", "PMC2", "PMC3"]
    assert len(starts) == 3
    for i in range(1, len(starts)):
        assert starts[i] - starts[i - 1] >= _PACE_SECONDS - 0.05, \
            f"requests {i - 1}->{i} not paced (gap {starts[i] - starts[i - 1]:.2f}s)"


async def test_judge_papers_failure_is_none_not_rejection():
    """A failed judge call maps to (paper, None) - never a rejection."""
    agent = CriticAgent(config=AppConfig())

    async def flaky_judge(task, requirement, paper, *, run_id="", attempt_id=""):
        await asyncio.sleep(0.001)
        if paper.document_id == "PMC2":
            raise RuntimeError("quota exceeded")
        return paper.document_id

    agent.judge = flaky_judge  # type: ignore[method-assign]
    task = _task()
    papers = [_paper("PMC1"), _paper("PMC2"), _paper("PMC3")]
    out = await agent.judge_papers(task, task.evidence_requirements[0], papers)

    assert [v for _, v in out] == ["PMC1", None, "PMC3"]
