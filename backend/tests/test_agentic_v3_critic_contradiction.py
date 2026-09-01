"""Agentic v3 - critic gate, contradiction detection, resolution, deep inspect."""
import json

import pytest

from src.agents.critic import (
    CriticAgent,
    evidence_excerpt,
    is_promising_for_deep_inspection,
)
from src.agents.contradiction import (
    ContradictionAgent,
    detect_contradictions_deterministic,
)
from src.agents.deepinspect import (
    expand_passage,
    verify_quote,
)
from src.agentic.state import (
    AnswersTask,
    ContradictionKind,
    CriticRelevance,
    CriticVerdict,
    EvidenceRequirement,
    EvidenceSource,
    ResearchTask,
    ResolutionStatus,
    RetrievedPaper,
    SupportDirection,
    V3RunState,
    VerifiedEvidence,
)
from src.config import AppConfig
from tests.conftest import native_test_model


def _task():
    return ResearchTask(
        id="T1",
        title="vitamin D and hypertension",
        objective="does vitamin D affect hypertension",
        evidence_requirements=[EvidenceRequirement(
            id="T1.R1",
            text="effect of vitamin D supplementation on blood pressure",
            target_n=3)],
    )


def test_critic_accepts_answering_passage():
    raw = json.dumps({"relevance": "relevant", "answers_task": "yes",
                      "support": "supports", "confidence": 0.95,
                      "note": "RCT shows reduction"})
    critic = CriticAgent(model=native_test_model(raw), config=AppConfig())
    paper = RetrievedPaper(
        document_id="PMC1", chunk_id="c1", section="Results",
        text=("In this trial, vitamin D supplementation for 12 months "
              "showed no significant difference in systolic blood pressure "
              "compared with placebo."))

    async def _run():
        verdict = await critic.judge(_task(), _task().requirement("T1.R1"), paper)
        return verdict

    import asyncio
    v = asyncio.run(_run())
    assert v.accepted
    assert v.support == SupportDirection.SUPPORTS


def test_critic_rejects_mention_only_passage():
    # spec section 10 example: mentions vitamin D deficiency but does NOT
    # answer whether supplementation lowers blood pressure.
    raw = json.dumps({"relevance": "partially_relevant", "answers_task": "no",
                      "support": "neutral", "confidence": 0.7,
                      "note": "association only, no supplementation outcome"})
    critic = CriticAgent(model=native_test_model(raw), config=AppConfig())
    paper = RetrievedPaper(document_id="PMC2", chunk_id="c2", section="Intro",
                           text="Vitamin D deficiency is common among patients with hypertension.")

    async def _run():
        return await critic.judge(_task(), _task().requirement("T1.R1"), paper)

    import asyncio
    v = asyncio.run(_run())
    assert not v.accepted
    assert is_promising_for_deep_inspection(v)


def test_evidence_excerpt_marks_truncation():
    text = "word " * 500
    ex = evidence_excerpt(text, max_chars=300)
    assert ex.endswith("[excerpt truncated]")


def test_detect_contradictions_deterministic():
    evidence = [
        VerifiedEvidence(id="E1", task_id="T2", requirement_id="T2.R1",
                         document_id="A", excerpt="reduced SBP",
                         support=SupportDirection.SUPPORTS),
        VerifiedEvidence(id="E2", task_id="T2", requirement_id="T2.R1",
                         document_id="B", excerpt="no significant effect",
                         support=SupportDirection.CONTRADICTS),
    ]
    cs = detect_contradictions_deterministic(evidence)
    assert len(cs) == 1
    assert cs[0].evidence_a == ["E1"]
    assert cs[0].evidence_b == ["E2"]
    assert cs[0].kind == ContradictionKind.DIRECT_CONFLICT


def test_contradiction_agent_uses_llm_and_validates_ids():
    raw = json.dumps({"contradictions": [
        {"claim": "vitamin D affects SBP", "requirement_id": "T2.R1",
         "evidence_a": ["E1"], "evidence_b": ["E2"],
         "kind": "context_dependent",
         "description": "depends on baseline vitamin D status"},
        {"claim": "fabricated", "evidence_a": ["NOT-AN-ID"],
         "evidence_b": ["E1"], "kind": "anomaly", "description": "x"},
    ]})
    agent = ContradictionAgent(model=native_test_model(raw), config=AppConfig())
    state = V3RunState(question="q")
    task = ResearchTask(id="T2", title="t", objective="o",
                        evidence_requirements=[EvidenceRequirement(
                            id="T2.R1", text="r", target_n=1)])
    req = task.requirement("T2.R1")
    req.add_evidence(VerifiedEvidence(
        id="E1", task_id="T2", requirement_id="T2.R1", document_id="A",
        excerpt="reduced", support=SupportDirection.SUPPORTS))
    req.add_evidence(VerifiedEvidence(
        id="E2", task_id="T2", requirement_id="T2.R1", document_id="B",
        excerpt="no effect", support=SupportDirection.CONTRADICTS))
    state.add_task(task)

    async def _run():
        return await agent.detect(state)

    import asyncio
    cs = asyncio.run(_run())
    assert len(cs) == 1   # the fabricated one is dropped (unknown ids)
    assert cs[0].kind == ContradictionKind.CONTEXT_DEPENDENT


def test_verify_quote_grep_verification():
    doc = ("In this randomized controlled trial, participants receiving "
           "vitamin D supplementation for 12 months showed no significant "
           "difference in systolic blood pressure compared with placebo.")
    assert verify_quote(doc, "vitamin d supplementation for 12 months") is not None
    assert verify_quote(doc, "reduced systolic blood pressure significantly") is None


def test_expand_passage_returns_context():
    doc = "Background text. The key result was an effect on blood pressure. More text."
    start = doc.lower().find("the key result")
    p = expand_passage(doc, start, start + 10)
    assert "key result" in p
    assert "blood pressure" in p
