"""Unit tests for the agentic v2 structured event stream (no live LLM)."""
import json

import pytest

from src.agentic.planner import Decomposition, SubQueryPlan
from src.agentic.retriever_tool import RetrievalResult
from src.agentic_v2.actions import ActionExecutor
from src.agentic_v2.events import EventEmitter
from src.agentic_v2.orchestrator import ActionDecision
from src.agentic_v2.pipeline import AgenticV2Pipeline
from src.agentic_v2.state import ActionType, EvidenceQuality, ObjectiveStatus
from src.agentic_v2.verify import ObjectiveVerdict, PassageAssessment
from src.agentic_v2.synthesize import SynthesisReport
from src.config import AppConfig


class RecordingEmitter:
    def __init__(self):
        self.events = []

    def emit(self, type_, **fields):
        self.events.append({"type": type_, **fields})

    def truncate(self):
        self.events.clear()


def _types(events):
    return [e["type"] for e in events]


def test_event_emitter_writes_jsonl_and_truncates(tmp_path):
    path = tmp_path / "ev.jsonl"
    em = EventEmitter(str(path))
    em.emit("run_start", question="q")
    em.emit("decision", action="DECOMPOSE")
    lines = path.read_text().splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0])["type"] == "run_start"
    em.truncate()
    assert path.read_text() == ""


class ScriptedOrchestrator:
    def __init__(self, decisions):
        self.decisions = list(decisions)

    async def decide(self, state):
        return self.decisions.pop(0) if self.decisions else \
            ActionDecision(action=ActionType.STOP, rationale="done")


class FakePlanner:
    async def plan(self, query):
        return Decomposition(subqueries=[SubQueryPlan(id="H1", target="a", query="a")])


class FakeRetriever:
    async def search(self, sub, top_k=6, exclude_chunk_ids=None):
        return [RetrievalResult(rank=1, chunk_id="c1", document_id="PMC1",
                                section="Results", paragraph_text="text", rrf_score=0.9)]


class FakeVerifier:
    async def verify(self, objective, passages):
        return ObjectiveVerdict(objective_id=objective.id, status=ObjectiveStatus.SUPPORTED,
                                confidence=0.9, assessments=[PassageAssessment(
                                    document_id="PMC1", chunk_id="c1",
                                    quality=EvidenceQuality.DIRECT, support="supports",
                                    confidence=0.9)])


class FakeSynthesizer:
    async def synthesize(self, state):
        return SynthesisReport(summary="answer", confidence=0.9)


def _pipeline(decisions):
    cfg = AppConfig()
    ex = ActionExecutor(config=cfg, planner=FakePlanner(), retriever=FakeRetriever(),
                        verifier=FakeVerifier(), synthesizer=FakeSynthesizer(), corpus=None)
    em = RecordingEmitter()
    p = AgenticV2Pipeline(config=cfg, orchestrator=ScriptedOrchestrator(decisions),
                          executor=ex, events=em)
    return p, em


@pytest.mark.asyncio
async def test_pipeline_emits_lifecycle_and_agent_events():
    decisions = [
        ActionDecision(action=ActionType.DECOMPOSE),
        ActionDecision(action=ActionType.GLOBAL_RETRIEVE, objective_id="H1", query="a"),
        ActionDecision(action=ActionType.VERIFY, objective_id="H1"),
        ActionDecision(action=ActionType.SYNTHESIZE),
    ]
    pipeline, em = _pipeline(decisions)
    await pipeline.answer("q")

    types = _types(em.events)
    assert types[0] == "run_start"
    assert "decision" in types
    assert "agent_spawn" in types
    assert "agent_output" in types
    assert "state" in types
    assert types[-1] == "run_end"

    # agent spawns map to the right agents
    agents = {e.get("agent") for e in em.events if e["type"] == "agent_spawn"}
    assert {"planner", "retriever", "verifier", "synthesizer"} <= agents

    # the board UI depends on per-chunk document + verdict events
    docs = [e for e in em.events if e["type"] == "document"]
    verdicts = [e for e in em.events if e["type"] == "verdict"]
    assert docs and docs[0]["text"] == "text"
    assert docs[0]["chunk_id"] == "c1"
    assert verdicts and verdicts[0]["chunk_id"] == "c1"

    # a failure-free run has no failure events
    assert "failure" not in types


@pytest.mark.asyncio
async def test_pipeline_emits_failure_on_action_error():
    class BoomRetriever:
        async def search(self, sub, top_k=6, exclude_chunk_ids=None):
            raise RuntimeError("boom")

    cfg = AppConfig()
    ex = ActionExecutor(config=cfg, planner=FakePlanner(), retriever=BoomRetriever(),
                        corpus=None)
    em = RecordingEmitter()
    pipeline = AgenticV2Pipeline(
        config=cfg,
        orchestrator=ScriptedOrchestrator([
            ActionDecision(action=ActionType.DECOMPOSE),
            ActionDecision(action=ActionType.GLOBAL_RETRIEVE, objective_id="H1", query="a"),
            ActionDecision(action=ActionType.STOP, rationale="gave up"),
        ]),
        executor=ex, events=em,
    )
    await pipeline.answer("q")
    failures = [e for e in em.events if e["type"] == "failure"]
    assert failures
    assert any("boom" in f.get("error", "") for f in failures)
