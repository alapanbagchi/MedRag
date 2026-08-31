"""State-machine tests: runtime-enforced progress / legality (fake orchestrator)."""
import asyncio

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


class RecordingEmitter:
    def __init__(self):
        self.events = []

    def emit(self, type_, **fields):
        self.events.append({"type": type_, **fields})

    def truncate(self):
        self.events.clear()


class ScriptedOrchestrator:
    def __init__(self, decisions, default=None):
        self.decisions = list(decisions)
        self.default = default or ActionDecision(action=ActionType.STOP, rationale="no more")
        self.calls = 0

    async def decide(self, state):
        self.calls += 1
        if self.decisions:
            return self.decisions.pop(0)
        return self.default


class FakePlanner:
    async def plan(self, query):
        return Decomposition(subqueries=[SubQueryPlan(id="H1", target="t", query="q1")])


class FakeRetriever:
    def __init__(self, docs=True):
        if docs is True:
            docs = [RetrievalResult(rank=1, chunk_id="c1", document_id="PMC1",
                                    section="Results", paragraph_text="text", rrf_score=0.9)]
        self.docs = docs or []
        self.calls = []

    async def search(self, sub, top_k=6, exclude_chunk_ids=None):
        self.calls.append(sub.query)
        return [r for r in self.docs if r.chunk_id not in (exclude_chunk_ids or [])]


class FakeVerifier:
    async def verify(self, objective, passages):
        return ObjectiveVerdict(
            objective_id=objective.id, status=ObjectiveStatus.SUPPORTED, confidence=0.8,
            assessments=[PassageAssessment(document_id=p.document_id, chunk_id=p.chunk_id,
                                           quality=EvidenceQuality.DIRECT, support="supports",
                                           confidence=0.8) for p in passages],
        )


class FakeSynthesizer:
    async def synthesize(self, state):
        return SynthesisReport(summary="answer", confidence=0.8)


def _make(decisions, default=None, *, retriever=None, max_rounds=8, max_retrieves=6,
          orchestrator_timeout=120.0, action_timeout=120.0):
    cfg = AppConfig()
    cfg.agentic_v2_max_rounds = max_rounds
    cfg.agentic_v2_max_global_retrieves = max_retrieves
    cfg.agentic_v2_orchestrator_timeout = orchestrator_timeout
    cfg.agentic_v2_action_timeout = action_timeout
    ex = ActionExecutor(config=cfg, planner=FakePlanner(), retriever=retriever or FakeRetriever(),
                        verifier=FakeVerifier(), synthesizer=FakeSynthesizer(), corpus=None)
    em = RecordingEmitter()
    p = AgenticV2Pipeline(config=cfg, orchestrator=ScriptedOrchestrator(decisions, default),
                          executor=ex, events=em)
    return p, em


# 12. valid agentic sequence still reaches synthesis (also covers #7 and #8).
@pytest.mark.asyncio
async def test_valid_sequence_reaches_synthesis_with_progress():
    p, em = _make([
        ActionDecision(action=ActionType.DECOMPOSE),
        ActionDecision(action=ActionType.GLOBAL_RETRIEVE, objective_id="H1", query="q1"),
        ActionDecision(action=ActionType.VERIFY, objective_id="H1"),
        ActionDecision(action=ActionType.SYNTHESIZE),
    ])
    result = await p.answer("q")
    assert result["terminal"] is True
    assert result["stop_reason"] == "synthesized"

    history = result["strategy_history"]
    gr = [a for a in history if a["action"] == "GLOBAL_RETRIEVE"]
    vv = [a for a in history if a["action"] == "VERIFY"]
    assert gr and gr[0]["progress_made"] is True       # successful retrieval = progress
    assert vv and vv[0]["progress_made"] is True       # status change = progress
    assert "failure" not in [e["type"] for e in em.events]


# 11. a scripted orchestrator cannot force illegal transitions.
@pytest.mark.asyncio
async def test_scripted_cannot_force_illegal_transitions():
    p, em = _make([ActionDecision(action=ActionType.VERIFY, objective_id="H1")],
                  default=ActionDecision(action=ActionType.STOP), max_rounds=3)
    result = await p.answer("q")
    assert result["actions"][0]["action"] == "DECOMPOSE"   # VERIFY repaired -> DECOMPOSE
    assert any(e["type"] == "policy_repair" and e.get("original_action") == "VERIFY"
               for e in em.events)


# 4. repeated identical retrieval query cannot loop forever.
@pytest.mark.asyncio
async def test_repeated_retrieve_query_cannot_loop():
    ret = FakeRetriever(docs=True)
    p, em = _make([], default=ActionDecision(action=ActionType.GLOBAL_RETRIEVE,
                                             objective_id="H1", query="q1"),
                  retriever=ret, max_rounds=6)
    result = await p.answer("q")
    assert result["terminal"] is True
    assert result["iterations"] <= 6
    assert len(ret.calls) == 1                            # only one real fetch
    assert any("GLOBAL_RETRIEVE" in k for k in result["exhausted_strategies"])


# 5. a retrieval that returns zero new documents is detected as no-progress.
@pytest.mark.asyncio
async def test_retrieve_zero_new_docs_is_no_progress():
    ret = FakeRetriever(docs=False)
    p, em = _make([
        ActionDecision(action=ActionType.DECOMPOSE),
        ActionDecision(action=ActionType.GLOBAL_RETRIEVE, objective_id="H1", query="q1"),
    ], default=ActionDecision(action=ActionType.STOP), retriever=ret, max_rounds=4)
    result = await p.answer("q")
    gr = [a for a in result["strategy_history"] if a["action"] == "GLOBAL_RETRIEVE"]
    assert gr and gr[0]["progress_made"] is False
    assert "no new documents" in gr[0]["progress_summary"]
    assert any(e["type"] == "no_progress" for e in em.events)


# 6. repeated no-progress actions force a strategy pivot.
@pytest.mark.asyncio
async def test_repeated_no_progress_forces_pivot():
    ret = FakeRetriever(docs=False)   # always zero new docs
    p, em = _make([ActionDecision(action=ActionType.DECOMPOSE)],
                  default=ActionDecision(action=ActionType.GLOBAL_RETRIEVE,
                                         objective_id="H1", query="q1"),
                  retriever=ret, max_rounds=5)
    result = await p.answer("q")
    assert result["terminal"] is True
    # the exhausted retrieval strategy was later repaired to a different action
    assert any(e["type"] == "policy_repair" and e.get("original_action") == "GLOBAL_RETRIEVE"
               for e in em.events)
    assert any("GLOBAL_RETRIEVE" in k for k in result["exhausted_strategies"])


# 3. global retrieval budget is enforced by the runtime.
@pytest.mark.asyncio
async def test_global_retrieve_budget_enforced():
    ret = FakeRetriever(docs=True)
    p, em = _make([ActionDecision(action=ActionType.DECOMPOSE)],
                  default=ActionDecision(action=ActionType.GLOBAL_RETRIEVE,
                                         objective_id="H1", query="q1"),
                  retriever=ret, max_rounds=5, max_retrieves=1)
    result = await p.answer("q")
    executed = [a for a in result["actions"] if a["action"] == "GLOBAL_RETRIEVE"]
    assert len(executed) == 1
    assert any(e["type"] == "policy_repair" and e.get("original_action") == "GLOBAL_RETRIEVE"
               for e in em.events)


# 10. max-round budget is a hard boundary.
@pytest.mark.asyncio
async def test_max_rounds_enforced():
    p, em = _make([], default=ActionDecision(action=ActionType.DECOMPOSE), max_rounds=3)
    result = await p.answer("q")
    assert result["iterations"] == 3
    assert result["terminal"] is True


# 9. a timeout terminates / falls back correctly.
@pytest.mark.asyncio
async def test_orchestrator_timeout_falls_back():
    class Hang:
        async def decide(self, state):
            await asyncio.sleep(5)

    cfg = AppConfig()
    cfg.agentic_v2_orchestrator_timeout = 0.05
    cfg.agentic_v2_action_timeout = 5.0
    cfg.agentic_v2_max_rounds = 2
    ex = ActionExecutor(config=cfg, planner=FakePlanner(), retriever=FakeRetriever(),
                        verifier=FakeVerifier(), synthesizer=FakeSynthesizer(), corpus=None)
    em = RecordingEmitter()
    p = AgenticV2Pipeline(config=cfg, orchestrator=Hang(), executor=ex, events=em)
    result = await p.answer("q")
    assert result["terminal"] is True
    assert any(e["type"] == "timeout" for e in em.events)
    assert any(e["type"] == "failure" for e in em.events)


# 1 + 2. prerequisites are enforced (VERIFY/SYNTHESIZE rejected with no objective/evidence).
@pytest.mark.asyncio
async def test_prerequisites_enforced():
    p, em = _make([ActionDecision(action=ActionType.SYNTHESIZE)],
                  default=ActionDecision(action=ActionType.STOP), max_rounds=3)
    result = await p.answer("q")
    # SYNTHESIZE with no objectives/evidence was repaired (not executed as-is)
    assert result["actions"][0]["action"] != "SYNTHESIZE"
    assert any(e["type"] == "policy_repair" and e.get("original_action") == "SYNTHESIZE"
               for e in em.events)
