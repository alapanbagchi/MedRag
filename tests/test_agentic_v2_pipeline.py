"""Integration-style tests for the agentic v2 loop (scripted orchestrator, no live LLM)."""
import pytest

from src.agentic.planner import Decomposition, SubQueryPlan
from src.agentic.retriever_tool import RetrievalResult
from src.agentic_v2.actions import ActionExecutor
from src.agentic_v2.orchestrator import ActionDecision
from src.agentic_v2.pipeline import AgenticV2Pipeline
from src.agentic_v2.state import ActionType, EvidenceQuality, ObjectiveStatus
from src.agentic_v2.verify import ObjectiveVerdict, PassageAssessment
from src.agentic_v2.synthesize import SynthesisReport
from src.config import AppConfig


class ScriptedOrchestrator:
    def __init__(self, decisions):
        self.decisions = list(decisions)
        self.calls = 0

    async def decide(self, state):
        self.calls += 1
        if not self.decisions:
            return ActionDecision(action=ActionType.STOP, rationale="no more scripted decisions")
        return self.decisions.pop(0)


class FakePlanner:
    async def plan(self, query):
        return Decomposition(question_type="comprehensive", subqueries=[
            SubQueryPlan(id="H1", target="variants -> calcification", query="v calc",
                         evidence_required=["link"]),
            SubQueryPlan(id="H2", target="HTE -> personalized mTOR", query="hte mtor",
                         evidence_required=["personalized"]),
        ])


class FakeRetriever:
    async def search(self, sub, top_k=6, exclude_chunk_ids=None):
        cid = "c1" if sub.query.startswith("v") else "c2"
        doc = "PMC1" if cid == "c1" else "PMC2"
        return [RetrievalResult(rank=1, chunk_id=cid, document_id=doc, section="Results",
                                paragraph_text="relevant text", rrf_score=0.9,
                                unit_kind="paragraph")]


class FakeVerifier:
    async def verify(self, objective, passages):
        return ObjectiveVerdict(
            objective_id=objective.id,
            status=ObjectiveStatus.SUPPORTED,
            confidence=0.8,
            gap="",
            assessments=[PassageAssessment(
                document_id=p.document_id, chunk_id=p.chunk_id, section=p.section,
                quality=EvidenceQuality.DIRECT, support="supports", confidence=0.8,
            ) for p in passages],
        )


class FakeSynthesizer:
    async def synthesize(self, state):
        return SynthesisReport(summary="answer", confidence=0.8)


def _pipeline(decisions):
    cfg = AppConfig()
    cfg.agentic_v2_max_rounds = 10
    executor = ActionExecutor(
        config=cfg,
        planner=FakePlanner(),
        retriever=FakeRetriever(),
        verifier=FakeVerifier(),
        synthesizer=FakeSynthesizer(),
        corpus=None,
    )
    return AgenticV2Pipeline(config=cfg, orchestrator=ScriptedOrchestrator(decisions),
                             executor=executor)


@pytest.mark.asyncio
async def test_full_loop_reaches_synthesis():
    decisions = [
        ActionDecision(action=ActionType.DECOMPOSE),
        ActionDecision(action=ActionType.GLOBAL_RETRIEVE, objective_id="H1", query="v calc"),
        ActionDecision(action=ActionType.VERIFY, objective_id="H1"),
        ActionDecision(action=ActionType.GLOBAL_RETRIEVE, objective_id="H2", query="hte mtor"),
        ActionDecision(action=ActionType.VERIFY, objective_id="H2"),
        ActionDecision(action=ActionType.SYNTHESIZE),
    ]
    result = await _pipeline(decisions).answer("compound question")
    assert result["terminal"] is True
    assert result["stop_reason"] == "synthesized"
    assert result["iterations"] == 6
    assert {o["id"] for o in result["objectives"]} == {"H1", "H2"}
    assert all(o["status"] == "supported" for o in result["objectives"])
    assert len(result["evidence"]) == 2
    assert result["answer"]["summary"] == "answer"


@pytest.mark.asyncio
async def test_budget_exhaustion_synthesizes_from_partial_evidence():
    decisions = [
        ActionDecision(action=ActionType.DECOMPOSE),
        ActionDecision(action=ActionType.GLOBAL_RETRIEVE, objective_id="H1", query="v calc"),
        ActionDecision(action=ActionType.VERIFY, objective_id="H1"),
    ]
    pipeline = _pipeline(decisions)
    pipeline.config.agentic_v2_max_rounds = 3
    result = await pipeline.answer("q")
    assert result["terminal"] is True
    assert result["iterations"] == 3
    # ran out of decisions but had evidence -> finalized with synthesis
    assert result["stop_reason"] == "synthesized"
    assert result["answer"] is not None


@pytest.mark.asyncio
async def test_budget_exhaustion_without_evidence_stops_honestly():
    decisions = [ActionDecision(action=ActionType.DECOMPOSE)]
    pipeline = _pipeline(decisions)
    pipeline.config.agentic_v2_max_rounds = 1
    result = await pipeline.answer("q")
    assert result["terminal"] is True
    assert result["stop_reason"].startswith("budget exhausted")


@pytest.mark.asyncio
async def test_stop_action_ends_loop():
    decisions = [
        ActionDecision(action=ActionType.DECOMPOSE),
        ActionDecision(action=ActionType.STOP, rationale="cannot answer with available tools"),
    ]
    result = await _pipeline(decisions).answer("q")
    assert result["terminal"] is True
    assert "cannot answer" in result["stop_reason"]
    assert result["answer"] is None


@pytest.mark.asyncio
async def test_stop_after_evidence_still_synthesizes():
    """Orchestrator STOP must NOT skip synthesis when verified evidence exists."""
    decisions = [
        ActionDecision(action=ActionType.DECOMPOSE),
        ActionDecision(action=ActionType.GLOBAL_RETRIEVE, objective_id="H1", query="v calc"),
        ActionDecision(action=ActionType.VERIFY, objective_id="H1"),
        ActionDecision(action=ActionType.STOP, rationale="orchestrator is done"),
    ]
    result = await _pipeline(decisions).answer("q")
    assert result["terminal"] is True
    # finalize synthesized from the relevant/partial evidence despite the STOP
    assert result["stop_reason"] == "synthesized"
    assert result["answer"]["summary"] == "answer"

