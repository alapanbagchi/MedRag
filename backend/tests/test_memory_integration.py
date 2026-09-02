"""Memory layer — integration with AgenticV3Pipeline (cross-session continuity).

Run 1 answers the vitamin D question and commits research memory; Run 2 is a
follow-up ("what about older adults?") that MUST resume the same research
session and receive bounded memory context — while the evidence boundary stays
intact (memory items are advisory for the planner only; verified evidence
still comes from the workers).
"""

from __future__ import annotations

import asyncio

import pytest

from src.agentic.pipeline import AgenticV3Pipeline
from src.agentic.state import (
    EvidenceRequirement,
    EvidenceSource,
    MasterPlan,
    ResearchTask,
    SupportDirection,
    VerifiedEvidence,
    WorkerReport,
)
from src.agents.synthesize import SynthesisReport
from src.config import AppConfig
from src.memory.api import MemoryAPI
from src.memory.config import MemoryConfig
from src.memory.embed import HashEmbedder
from src.memory.enums import EvidenceRole, ProvenanceClass
from src.memory.store import InMemoryMemoryStore


class MemoryAwareMaster:
    """Records whether it received advisory memory context."""

    def __init__(self):
        self.seen_memory_context = []

    async def plan(self, query, budget, memory_context=""):
        self.seen_memory_context.append(memory_context or "")
        tasks = []
        for tid, title, reqs in (
            ("T1", "Dietary management of hypertension",
             ["sodium and blood pressure"]),
            ("T2", "Vitamin D and hypertension",
             ["vitamin D supplementation and blood pressure"]),
        ):
            tasks.append(ResearchTask(
                id=tid, title=title,
                objective=f"determine: {title}",
                intent="evidence for the objective",
                evidence_requirements=[
                    EvidenceRequirement(id=f"{tid}.R{j}", text=text,
                                        target_n=budget.evidence_target or 3)
                    for j, text in enumerate(reqs, start=1)],
                stop_criteria=["evidence satisfied", "budget exhausted"],
            ))
        return MasterPlan(question=query, tasks=tasks)


class FakeWorker:
    async def run(self, task, budget, run_id=""):
        for req in task.evidence_requirements:
            for i, (doc, support) in enumerate(
                    [("PMC11684474", SupportDirection.SUPPORTS),
                     ("PMC11684475", SupportDirection.SUPPORTS),
                     ("PMC11684476", SupportDirection.SUPPORTS)], start=1):
                req.add_evidence(VerifiedEvidence(
                    id=f"E-{task.id}-{req.id}-{i}",
                    task_id=task.id, requirement_id=req.id,
                    document_id=doc, chunk_id=f"c-{doc}",
                    section="Results",
                    excerpt=f"{doc}: blood-pressure finding {i}",
                    claim=f"Finding from {doc}: vitamin D supplementation "
                          "affects blood pressure in hypertensive adults.",
                    support=support, confidence=0.9,
                    source=EvidenceSource.RETRIEVAL,
                ))
        task.finalize()
        return WorkerReport(run_id=run_id, task_id=task.id,
                            task_title=task.title, status=task.status.value,
                            stop_reason="satisfied", requirements=[],
                            evidence=[e for r in task.evidence_requirements
                                      for e in r.accepted],
                            searches_used=1, deep_inspections_used=0)


class FakeContradiction:
    async def detect(self, state):
        return []


class FakeResolution:
    async def resolve(self, contradiction, state):
        return None


class FakeSynthesizer:
    async def synthesize(self, state):
        return SynthesisReport(
            summary="Dietary sodium reduction and vitamin D findings are "
                    "summarized from the verified evidence.",
            sections=[], limitations=[], unresolved_gaps=[],
            unresolved_contradictions=[], resolved_contradictions=[],
            confidence=0.8, citations=[])


def _pipeline(memory, master=None):
    return AgenticV3Pipeline(
        config=AppConfig(),
        master=master or MemoryAwareMaster(),
        worker=FakeWorker(),
        contradiction_agent=FakeContradiction(),
        resolution_agent=FakeResolution(),
        synthesizer=FakeSynthesizer(),
        memory=memory,
    )


def test_cross_session_continuity():
    api = MemoryAPI(store=InMemoryMemoryStore(), embedder=HashEmbedder(256),
                    config=MemoryConfig(backend="memory"))
    master = MemoryAwareMaster()

    # ---- run 1: full investigation ----
    result1 = asyncio.run(_pipeline(api, master).answer(
        "Does vitamin D supplementation lower blood pressure?"))
    assert result1["terminal"] is True
    assert result1["memory"]["enabled"] is True
    stats1 = result1["memory"]["committed"]
    assert stats1["claims_committed"] >= 1
    assert stats1["claims_deduped"] >= 2      # near-dup claims merged, links unioned
    assert result1["memory"]["context"]           # planner received context
    assert "memory" in master.seen_memory_context[0]

    session1 = result1["memory"]["session_id"]
    assert session1 and api.store.get_session(session1) is not None

    evidence_claims = [c for c in api.store.get_claims()
                       if c.provenance_class == ProvenanceClass.EVIDENCE_DERIVED_CLAIM]
    assert len(evidence_claims) >= 1
    # every evidence-derived claim is fully traceable to verified evidence;
    # support links from all distinct papers accumulate on the survivor
    all_links = [l for c in evidence_claims
                 for l in api.store.get_claim_evidence_links(c.id)
                 if l.role == EvidenceRole.SUPPORTED_BY]
    assert len(all_links) >= 3                 # 3 distinct papers recorded
    for claim in evidence_claims:
        links = api.store.get_claim_evidence_links(claim.id)
        assert links
        assert all(api.store.get_evidence_ref(l.evidence_ref_id).verified
                   for l in links)

    # ---- run 2: follow-up query resumes the SAME research session ----
    master2 = MemoryAwareMaster()
    result2 = asyncio.run(_pipeline(api, master2).answer(
        "What about vitamin D supplementation in older adults?"))
    assert result2["memory"]["enabled"] is True
    session2 = result2["memory"]["session_id"]
    assert session2 == session1                     # continuation, not a new session
    # the planner really received the prior research state
    assert any("vitamin" in (m or "").casefold() and "BLOOD PRESSURE" in m.upper()
               for m in master2.seen_memory_context)
    # the run merged into the same session (no duplicate claims)
    stats2 = result2["memory"]["committed"]
    assert stats2["session_id"] == session1
    claims_after = api.store.get_claims(
        session_id=session1, provenance=ProvenanceClass.EVIDENCE_DERIVED_CLAIM)
    assert len(claims_after) == len(evidence_claims)   # dedup, no explosion


def test_pipeline_without_memory_is_unchanged():
    """No memory attached -> byte-identical behavior, memory block says
    enabled=False."""
    pipeline = AgenticV3Pipeline(
        config=AppConfig(),
        master=MemoryAwareMaster(),
        worker=FakeWorker(),
        contradiction_agent=FakeContradiction(),
        resolution_agent=FakeResolution(),
        synthesizer=FakeSynthesizer(),
    )
    result = asyncio.run(pipeline.answer("dietary salt and hypertension?"))
    assert result["terminal"] is True
    assert result["memory"] == {"enabled": False}


def test_context_boundary_in_integration():
    """prepare-run context never injects memory into the evidence channel:
    the result's verified evidence still comes only from the pipeline."""
    api = MemoryAPI(store=InMemoryMemoryStore(), embedder=HashEmbedder(256),
                    config=MemoryConfig(backend="memory"))
    result = asyncio.run(_pipeline(api).answer("vitamin D and blood pressure?"))
    evidence = result["evidence"]
    assert evidence                                  # pipeline evidence present
    for e in evidence:
        assert e["status"] in ("accepted", "contradictory")
    # the memory context region is bounded and advisory-only
    ctx = result["memory"]["context"]
    assert len(ctx) < 6000
    assert "not evidence" in ctx.casefold() or "VERIFIED EVIDENCE" in ctx