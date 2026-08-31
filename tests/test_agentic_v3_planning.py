"""Agentic v3 - master planning + search-term selection (offline, no LLM)."""
import json

import pytest

from src.agents.master import (
    MasterOrchestratorAgent,
    PlanTask,
    build_plan,
    fallback_task,
)
from src.agents.search import (
    SearchTermPlanner,
    TaskSearchPlan,
    _clean_query,
    fallback_queries,
)
from src.agents.state import EvidenceRequirement, ResearchTask, RunBudget, TermConcept
from src.config import AppConfig
from tests.conftest import native_test_model


def _task_with_terms(terms=("vitamin D", "hypertension", "blood pressure")):
    task = ResearchTask(
        id="T1",
        title="vitamin D and hypertension",
        objective="does vitamin D affect hypertension",
        intent="find evidence on vitamin D and blood pressure",
        evidence_requirements=[EvidenceRequirement(
            id="T1.R1", text="effect of vitamin D supplementation on blood pressure",
            target_n=3)],
        entities=list(terms),
        terminology=[TermConcept(surface_form=t) for t in terms],
    )
    return task


def test_build_plan_applies_thresholds_and_stop_criteria():
    blueprints = [PlanTask(
        id="T1",
        title="Dietary management of hypertension",
        objective="determine diet-blood pressure strategies",
        evidence_required=["Foods that affect blood pressure",
                           "Potassium intake and blood pressure"],
        entities=["sodium", "potassium"],
    )]
    plan = build_plan("diet question", blueprints, budget=RunBudget(evidence_target=3))
    assert len(plan.tasks) == 1
    t = plan.tasks[0]
    assert len(t.evidence_requirements) == 2
    assert all(r.target_n == 3 for r in t.evidence_requirements)
    assert "retrieval budget exhausted" in t.stop_criteria
    assert plan.question == "diet question"


def test_build_plan_dedupes_by_objective():
    blueprints = [
        PlanTask(id="T1", title="one", objective="eggs and cardiovascular disease",
                 evidence_requirements=["a"], entities=["egg"]),
        PlanTask(id="T2", title="two", objective="eggs and cardiovascular disease risk",
                 evidence_requirements=["b"], entities=["egg"]),
    ]
    plan = build_plan("eggs question", blueprints)
    assert len(plan.tasks) == 1


def test_build_plan_falls_back_to_single_task():
    plan = build_plan("some question", [], budget=RunBudget(evidence_target=2))
    assert len(plan.tasks) == 1
    assert plan.tasks[0].evidence_requirements[0].target_n == 2


def test_master_agent_parses_decomposition():
    raw = json.dumps({
        "rationale": "two obligations",
        "tasks": [
            {"id": "T1", "title": "Diet and hypertension",
             "objective": "how diet affects hypertension",
             "intent": "dietary strategies",
             "evidence_required": ["sodium and blood pressure", "DASH diet"],
             "entities": ["sodium", "DASH"]},
            {"id": "T2", "title": "Vitamin D and hypertension",
             "objective": "does vitamin D affect hypertension",
             "intent": "supplementation evidence",
             "evidence_required": ["vitamin D supplementation and blood pressure"],
             "entities": ["vitamin D", "blood pressure"]},
        ],
    })
    master = MasterOrchestratorAgent(model=native_test_model(raw), config=AppConfig())

    async def _run():
        plan = await master.plan("hypertension diet vitamin D", budget=RunBudget(evidence_target=3))
        return plan

    import asyncio
    plan = asyncio.run(_run())
    assert len(plan.tasks) == 2
    assert all(r.target_n == 3 for t in plan.tasks for r in t.evidence_requirements)
    assert plan.tasks[0].entities == ["sodium", "DASH"]


def test_master_agent_falls_back_on_garbage():
    master = MasterOrchestratorAgent(
        model=native_test_model("not json at all"), config=AppConfig())

    async def _run():
        return await master.plan("q", budget=RunBudget(evidence_target=1))

    import asyncio
    plan = asyncio.run(_run())
    assert len(plan.tasks) == 1  # deterministic fallback never returns zero tasks


def test_fallback_queries_rotate_and_never_repeat():
    task = _task_with_terms()
    r1 = fallback_queries(task, "effect of vitamin D supplementation on blood pressure", 1, [])
    assert r1, "round 1 must produce queries"
    r2 = fallback_queries(task, "effect of vitamin D supplementation on blood pressure", 2, r1)
    assert r2, "round 2 must produce queries"
    assert not any(q.casefold() in {x.casefold() for x in r1} for q in r2), \
        "round 2 must differ from round 1"


def test_clean_query_rejects_junk():
    assert _clean_query("") == ""
    assert _clean_query("hi") == ""   # too short
    assert _clean_query("  vitamin D AND hypertension  ") == "vitamin D AND hypertension"


def test_search_planner_uses_llm_queries():
    raw = json.dumps({
        "rationale": "try supplementation phrasing",
        "queries": [
            "\"vitamin D supplementation\" AND blood pressure",
            "25-hydroxyvitamin D AND hypertension",
        ],
    })
    planner = SearchTermPlanner(model=native_test_model(raw), config=AppConfig())
    task = _task_with_terms()

    async def _run():
        plan = await planner.plan(task, 1, {})
        return plan

    import asyncio
    plan = asyncio.run(_run())
    assert isinstance(plan, TaskSearchPlan)
    queries = plan.queries_for("T1.R1")
    assert len(queries) == 2
    assert "vitamin D supplementation" in queries[0]

def test_master_parses_thought_plus_json():
    """The strict prompt makes the model think in prose THEN emit the JSON -
    the plan must come out fully populated (raw-text JSON salvage)."""
    raw = json.dumps({
        "rationale": "two obligations",
        "tasks": [
            {"id": "T1", "title": "Dietary restrictions for hypertension",
             "objective": "Determine the dietary restrictions and modifications "
                          "supported for the management of hypertension",
             "intent": "Identify and assess dietary strategies",
             "evidence_required": ["Dietary restrictions or modifications "
                                   "associated with improved blood-pressure control",
                                   "Effectiveness of dietary patterns for hypertension"],
             "entities": ["diet", "dietary restriction", "hypertension", "blood pressure"]},
            {"id": "T2", "title": "Vitamin D supplementation in hypertension",
             "objective": "Determine whether increased vitamin D intake is "
                          "recommended for people with hypertension",
             "intent": "Assess the benefit of vitamin D",
             "evidence_required": ["Effect of vitamin D supplementation on blood pressure",
                                   "Association between vitamin D status and hypertension"],
             "entities": ["vitamin D", "hypertension", "blood pressure"]},
        ],
    })
    composite = "<thought>Here is my outline T1..., T2...</thought>" + raw
    master = MasterOrchestratorAgent(model=native_test_model(composite),
                                     config=AppConfig())

    async def _run():
        return await master.plan("hypertension diet vitamin D")

    import asyncio
    plan = asyncio.run(_run())
    assert len(plan.tasks) == 2
    for t in plan.tasks:
        assert t.evidence_requirements and t.entities and t.intent
        assert t.objective != t.title
    assert plan.tasks[0].entities == ["diet", "dietary restriction",
                                      "hypertension", "blood pressure"]


def test_master_salvages_prose_outline_with_task_fields():
    """When the model emits ONLY a bullet outline (Task:/Intent:/Evidence:/
    Entities:), the plan is salvaged from the outline - never empty."""
    outline = ("<thought>\n* T1: Dietary management\n"
               "    * Task: Identify dietary modifications supported for hypertension.\n"
               "    * Intent: Identify and assess dietary strategies.\n"
               "    * Evidence: Dietary restrictions for blood pressure control.\n"
               "    * Entities: Hypertension, diet, blood pressure.\n"
               "* T2: Vitamin D and hypertension\n"
               "    * Task: Determine if increased vitamin D intake is effective.\n"
               "    * Intent: Assess the benefit of vitamin D.\n"
               "    * Evidence: Effect of vitamin D supplementation on blood pressure.\n"
               "    * Entities: Vitamin D, hypertension, blood pressure.\n"
               "</thought>")
    master = MasterOrchestratorAgent(model=native_test_model(outline),
                                     config=AppConfig())

    async def _run():
        return await master.plan("diet for hypertension and vitamin D?")

    import asyncio
    plan = asyncio.run(_run())
    assert len(plan.tasks) == 2
    assert plan.tasks[0].objective.startswith("Identify dietary modifications")
    assert plan.tasks[1].evidence_requirements[0].text.startswith(
        "Effect of vitamin D")


def test_master_rejects_title_only_output():
    """A label-only response (no objectives/evidence anywhere) must not be
    accepted as a plan - it falls back to the honest single task."""
    labels = "<thought>* T1: Dietary stuff\n* T2: Vitamin D stuff</thought>"
    master = MasterOrchestratorAgent(model=native_test_model(labels),
                                     config=AppConfig())

    async def _run():
        return await master.plan("some question")

    import asyncio
    plan = asyncio.run(_run())
    assert len(plan.tasks) == 1
    assert plan.tasks[0].objective
