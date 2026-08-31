"""Planner agent tests: a complex medical question -> a valid QueryPlan."""

from __future__ import annotations

import json

from src.agents.planner import PlannerAgent, QueryPlan
from src.config import AppConfig

from tests.conftest import native_test_model

QUESTION = (
    "In patients undergoing surgical repair of coarctation of the aorta, which "
    "repair techniques were associated with early recurrent coarctation, what "
    "percentages and p-values were reported for those associations, and how did "
    "the imaging findings in the same study characterize the recurrent lesions "
    "or anatomical changes?"
)

PLAN_JSON = json.dumps(
    {
        "original_query": QUESTION,
        "question_type": "association",
        "entities": [
            {"text": "coarctation of the aorta", "role": "condition", "terminology": True},
            {"text": "surgical repair", "role": "procedure", "terminology": True},
            {"text": "recurrent coarctation", "role": "condition", "terminology": True},
        ],
        "targets": ["repair techniques", "associations", "imaging findings"],
        "subqueries": [
            {
                "id": "H1",
                "target": "repair techniques associated with recurrent coarctation percentages p-values imaging findings",
                "focus": "association",
                "evidence_required": [
                    "repair techniques associated with recurrent coarctation",
                    "percentages per repair technique",
                    "p-values per association",
                ],
            }
        ],
    }
)


async def test_planner_produces_valid_query_plan():
    planner = PlannerAgent(model=native_test_model(PLAN_JSON), config=AppConfig())
    enriched = await planner.plan(QUESTION)
    plan = enriched.plan
    assert isinstance(plan, QueryPlan)
    assert plan.original_query == QUESTION
    assert len(plan.subqueries) == 1
    sub = plan.subqueries[0]
    assert sub.id == "H1"
    assert "repair techniques" in sub.target
    assert sub.evidence_required
    assert plan.entities
    # EnrichedPlan proxies the inner plan's attributes for convenience.
    assert enriched.original_query == QUESTION
    assert enriched.clinical_entities


async def test_single_hop_question_stays_one_subquery():
    plan_json = json.dumps(
        {
            "original_query": "What is the CHA2DS2-VASc score threshold for anticoagulation in atrial fibrillation?",
            "question_type": "definition",
            "entities": [
                {"text": "atrial fibrillation", "role": "condition", "terminology": True}
            ],
            "targets": ["CHA2DS2-VASc score threshold"],
            "subqueries": [
                {
                    "id": "H1",
                    "target": "CHA2DS2-VASc anticoagulation threshold for atrial fibrillation",
                    "focus": "definition",
                    "evidence_required": [
                        "CHA2DS2-VASc anticoagulation threshold",
                        "score cutoff for oral anticoagulation",
                        "guideline recommendation",
                    ],
                }
            ],
        }
    )
    planner = PlannerAgent(model=native_test_model(plan_json), config=AppConfig())
    enriched = await planner.plan(
        "What is the CHA2DS2-VASc score threshold for anticoagulation in atrial fibrillation?"
    )
    assert len(enriched.plan.subqueries) == 1  # never split a single obligation


async def test_missing_original_query_is_backfilled():
    plan_json = json.dumps({
        "question_type": "factual",
        "subqueries": [{"id": "H1", "target": "t"}],
    })
    planner = PlannerAgent(model=native_test_model(plan_json), config=AppConfig())
    enriched = await planner.plan(QUESTION)
    assert enriched.plan.original_query == QUESTION


async def test_empty_plan_gets_single_hop_fallback():
    """gemma sometimes returns a plan with ZERO subqueries; that must never
    silently no-op the pipeline."""
    planner = PlannerAgent(model=native_test_model('{"question_type":"prognosis"}'),
                           config=AppConfig())
    enriched = await planner.plan("Does EF predict survival in heart failure?")
    assert len(enriched.plan.subqueries) == 1
    sub = enriched.plan.subqueries[0]
    assert sub.id == "H1"
    assert "EF predict survival" in sub.query


def test_planner_has_no_tools_by_default(monkeypatch):
    """Live UMLS lookups inside the planning loop made small models spiral
    (repeated lookups, junk entities); the tool is opt-in now."""
    monkeypatch.delenv("PLANNER_USE_UMLS_TOOL", raising=False)
    cfg = AppConfig()
    planner = PlannerAgent(model=native_test_model('{"question_type":"factual"}'), config=cfg)
    assert planner.use_tool is False
    assert not cfg.planner_use_umls_tool


def test_planner_can_opt_into_umls_tool(monkeypatch):
    monkeypatch.setenv("PLANNER_USE_UMLS_TOOL", "1")
    planner = PlannerAgent(model=native_test_model('{"question_type":"factual"}'),
                           config=AppConfig())
    assert planner.use_tool is True


async def test_search_umls_is_cached_per_process(monkeypatch):
    import httpx

    from src.agents.planner import search_umls

    monkeypatch.setenv("UMLS_API_KEY", "test-key")
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if request.url.path.endswith("/search/current"):
            return httpx.Response(200, json={"result": {"results": [
                {"ui": "C0018801", "name": "Heart failure"}]}})
        return httpx.Response(200, json={"result": [{"name": "Cardiac Failure"}]})

    first = await search_umls("heart failure", transport=httpx.MockTransport(handler))
    assert first["found"] is True and first["cui"] == "C0018801"
    assert calls["n"] == 2  # search + atoms

    second = await search_umls("heart failure", transport=httpx.MockTransport(handler))
    assert second["found"] is True and second.get("cached") is True
    assert calls["n"] == 2, "second identical lookup must be served from cache"
