"""Agentic v3 - pipeline integration (all stages scripted, no live LLM/corpus).

Exercises the run coherence: master plan -> parallel workers -> verified
evidence -> contradiction agent -> resolution agent -> final evidence set ->
final answer, plus the JSON-safe result dict and honest stop conditions.
"""

import asyncio
import json

import pytest

from src.agentic_v3.master import MasterPlan
from src.agentic_v3.pipeline import AgenticV3Pipeline
from src.agentic_v3.state import (
    Contradiction,
    ContradictionKind,
    EvidenceRequirement,
    EvidenceSource,
    ResearchTask,
    ResolutionOutcome,
    ResolutionStatus,
    RunBudget,
    SupportDirection,
    VerifiedEvidence,
    WorkerReport,
)
from src.agentic_v3.synthesize import AnswerSection, Citation, SynthesisReport
from src.config import AppConfig


class FakeMaster:
    async def plan(self, query, budget):
        tasks = []
        for tid, title, reqs in (
            ("T1", "Dietary management of hypertension",
             ["sodium and blood pressure", "potassium and blood pressure"]),
            ("T2", "Vitamin D and hypertension",
             ["vitamin D supplementation and blood pressure"]),
        ):
            tasks.append(ResearchTask(
                id=tid,
                title=title,
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
    """Adds verified evidence to the task and returns a report."""

    def __init__(self):
        self.events = None
        self.runs = []

    async def run(self, task, budget, run_id=""):
        self.runs.append((run_id, task.id))
        # Worker 1: both requirements satisfied (3 papers each).
        # Worker 2: 3 supporting papers (T2.R1) AND 1 contradicting -> the
        # contradiction agent should still flag it.
        plan = {
            "T1": {
                "T1.R1": [("S1", SupportDirection.SUPPORTS),
                          ("S2", SupportDirection.SUPPORTS),
                          ("S3", SupportDirection.SUPPORTS)],
                "T1.R2": [("P1", SupportDirection.SUPPORTS),
                          ("P2", SupportDirection.SUPPORTS),
                          ("P3", SupportDirection.SUPPORTS)],
            },
            "T2": {
                "T2.R1": [("D1", SupportDirection.SUPPORTS),
                          ("D2", SupportDirection.SUPPORTS),
                          ("D3", SupportDirection.SUPPORTS),
                          ("D4", SupportDirection.CONTRADICTS)],
            },
        }
        for req in task.evidence_requirements:
            for i, (doc, support) in enumerate(plan[task.id][req.id], start=1):
                req.add_evidence(VerifiedEvidence(
                    id=f"E-{task.id}-{req.id}-{i}",
                    task_id=task.id, requirement_id=req.id,
                    document_id=doc, chunk_id=f"c-{doc}",
                    section="Results",
                    excerpt=f"{doc}: blood-pressure finding number {i}",
                    claim="blood pressure finding",
                    support=support, confidence=0.9,
                    source=EvidenceSource.RETRIEVAL,
                ))
        task.finalize()
        return WorkerReport(
            run_id=run_id,
            task_id=task.id, task_title=task.title, status=task.status.value,
            stop_reason="satisfied",
            requirements=[],
            evidence=[e for r in task.evidence_requirements for e in r.accepted],
            searches_used=2, deep_inspections_used=0,
        )

    def _build_report(self, task, budget):
        return WorkerReport(task_id=task.id, task_title=task.title,
                            status=task.status.value)


class FakeContradictionAgent:
    async def detect(self, state):
        return [Contradiction(
            id="C1",
            claim="vitamin D supplementation affects systolic blood pressure",
            task_id="T2", requirement_id="T2.R1",
            evidence_a=[e.id for e in state.all_evidence()
                        if e.document_id in ("D1", "D2", "D3")],
            evidence_b=[e.id for e in state.all_evidence()
                        if e.document_id == "D4"],
            kind=ContradictionKind.CONTEXT_DEPENDENT,
            description="positive effect vs no significant effect",
        )]


class FakeResolutionAgent:
    async def resolve(self, contradiction, state):
        return ResolutionOutcome(
            status=ResolutionStatus.RESOLVED,
            explanation="effect depends on baseline vitamin D status; "
                        "conflict is context-dependent, not a flat contradiction",
            additional_queries=["vitamin D deficiency hypertension meta-analysis"],
            additional_papers=["PMC-META-1", "PMC-META-2"],
            characterization="context-dependent: baseline vitamin D status",
        )


class FakeSynthesizer:
    async def synthesize(self, state):
        evidence = state.all_evidence()
        return SynthesisReport(
            summary="The evidence supports specific dietary strategies; "
                    "vitamin D findings are context-dependent.",
            sections=[AnswerSection(
                heading="Dietary management",
                body="Sodium and potassium evidence is consistent.",
                citations=[Citation(requirement_id="T1.R1",
                                    document_id="S1", support="supports")],
            )],
            limitations=["limited corpus"],
            unresolved_gaps=["none"],
            unresolved_contradictions=[],
            resolved_contradictions=["vitamin D is context-dependent"],
            confidence=0.85,
            citations=[Citation(requirement_id="T2.R1", document_id="D1",
                                support="supports")],
        )


def _pipeline():
    return AgenticV3Pipeline(
        config=AppConfig(),
        master=FakeMaster(),
        worker=FakeWorker(),
        contradiction_agent=FakeContradictionAgent(),
        resolution_agent=FakeResolutionAgent(),
        synthesizer=FakeSynthesizer(),
    )


def test_pipeline_full_run():
    result = asyncio.run(
        _pipeline().answer("dietary restrictions for hypertension and vitamin D?"))
    assert result["terminal"] is True
    assert result["stop_reason"] == "synthesized"
    assert len(result["tasks"]) == 2
    assert len(result["workers"]) == 2
    assert len(result["evidence"]) == 10   # 3+3+4
    assert len(result["contradictions"]) == 1
    c = result["contradictions"][0]
    assert c["kind"] == "context_dependent"
    assert c["resolution"]["status"] == "resolved"
    assert result["resolved"] == ["C1"]
    assert result["unresolved"] == []
    assert result["answer"]["summary"]
    assert result["answer"]["confidence"] == 0.85
    # citation repair: D1 exists in the evidence set
    assert any(e["document_id"] == "D1" for e in result["evidence"])


def test_pipeline_with_llm_agents_via_test_model():
    """The real agent classes parse the LLM JSON through TestModel."""
    from src.agentic_v3.contradiction import ContradictionAgent
    from src.agentic_v3.master import MasterOrchestratorAgent
    from src.agentic_v3.resolution import ResolutionAgent
    from src.agentic_v3.synthesize import FinalSynthesizer
    from tests.conftest import native_test_model

    master_json = json.dumps({
        "rationale": "single obligation",
        "tasks": [
            {"id": "T1", "title": "eggs and cardiovascular disease",
             "objective": "does egg intake affect cardiovascular disease risk",
             "intent": "association evidence",
             "evidence_required": ["egg consumption and cardiovascular disease"],
             "entities": ["egg", "cardiovascular disease"]},
        ],
    })
    contradiction_json = json.dumps({"contradictions": []})
    resolution_json = json.dumps({"status": "unresolved",
                                  "explanation": "no explanation found",
                                  "characterization": "unresolved"})
    synth_json = json.dumps({
        "summary": "no verified evidence to answer in offline test",
        "sections": [],
        "limitations": ["offline test"],
        "unresolved_gaps": ["no corpus"],
        "unresolved_contradictions": [],
        "resolved_contradictions": [],
        "confidence": 0.0,
        "citations": [],
    })
    pipeline = AgenticV3Pipeline(
        config=AppConfig(),
        master=MasterOrchestratorAgent(model=native_test_model(master_json),
                                       config=AppConfig()),
        worker=FakeWorker(),
        contradiction_agent=ContradictionAgent(
            model=native_test_model(contradiction_json), config=AppConfig()),
        resolution_agent=ResolutionAgent(model=native_test_model(resolution_json),
                                         config=AppConfig()),
        synthesizer=FinalSynthesizer(model=native_test_model(synth_json),
                                     config=AppConfig()),
    )
    result = asyncio.run(pipeline.answer("should healthy adults avoid eggs?"))
    assert result["terminal"] is True
    assert len(result["tasks"]) == 1
    assert result["answer"]["summary"]
