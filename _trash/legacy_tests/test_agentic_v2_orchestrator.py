"""Unit tests for the agentic v2 orchestrator decision (TestModel, no live LLM)."""
import json

import pytest

from src.agentic_v2.orchestrator import ActionDecision, OrchestratorAgent
from src.agentic_v2.state import ActionType, ResearchState
from src.config import AppConfig

from tests.conftest import native_test_model


@pytest.mark.asyncio
async def test_decide_parses_action_decision():
    decision_json = json.dumps({
        "action": "GLOBAL_RETRIEVE",
        "objective_id": "H1",
        "rationale": "no documents yet",
        "query": "medial arterial calcification hypertension variants",
        "terms": [],
        "document_id": "",
        "chunk_id": "",
        "instructions": "search the corpus",
    })
    orch = OrchestratorAgent(model=native_test_model(decision_json), config=AppConfig())
    from src.agentic_v2.state import ResearchObjective
    state = ResearchState(question="how do hypertension variants relate to calcification?")
    # an objective must exist for GLOBAL_RETRIEVE to be legal (policy-enforced)
    state.upsert_objective(ResearchObjective(id="H1", statement="variants -> calcification"))
    decision = await orch.decide(state)
    assert isinstance(decision, ActionDecision)
    assert decision.action == ActionType.GLOBAL_RETRIEVE
    assert decision.objective_id == "H1"
    assert decision.query


@pytest.mark.asyncio
async def test_decide_defaults_stop_action():
    decision_json = json.dumps({"action": "STOP", "rationale": "insufficient evidence"})
    orch = OrchestratorAgent(model=native_test_model(decision_json), config=AppConfig())
    decision = await orch.decide(ResearchState(question="q"))
    assert decision.action == ActionType.STOP


def test_repair_fills_empty_retrieve_query():
    from src.agentic_v2.orchestrator import OrchestratorAgent
    state = ResearchState(question="some question")
    decision = ActionDecision(action=ActionType.GLOBAL_RETRIEVE, query="")
    repaired = OrchestratorAgent._repair(decision, state)
    assert repaired.query == "some question"


def test_repair_fills_query_from_objective():
    from src.agentic_v2.orchestrator import OrchestratorAgent
    from src.agentic_v2.state import ResearchObjective
    state = ResearchState(question="q")
    state.upsert_objective(ResearchObjective(id="H1", statement="rapamycin personalized hypertension"))
    decision = ActionDecision(action=ActionType.GLOBAL_RETRIEVE,
                              objective_id="H1", query="")
    repaired = OrchestratorAgent._repair(decision, state)
    assert repaired.query == "rapamycin personalized hypertension"
