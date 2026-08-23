"""Tests for the retrieval V2 query planner: requirement decomposition and
the terminology-preservation rule (spec sections 6-9).
"""

from __future__ import annotations

import pytest

from medrag.retrieval_v2.planner import plan_question

MULTI_HOP = (
    "How do metabolic syndrome, advanced CKD, and anemia affect cardiovascular "
    "vulnerability, and how does this differ across INOCA, non-diabetic CKD, and "
    "elderly hip-fracture patients?"
)


def test_benchmark_decomposition():
    plan = plan_question(MULTI_HOP)
    assert "metabolic syndrome" in plan.conditions
    assert "advanced CKD" in plan.conditions
    assert "anemia" in plan.conditions
    assert "INOCA" in plan.populations
    assert "non-diabetic CKD" in plan.populations
    assert plan.target == "cardiovascular vulnerability"
    foci = [r.focus for r in plan.requirements]
    assert foci[0] == "mechanism"
    assert foci[len(plan.conditions)] == "outcome"
    assert len(plan.queries) >= 6


def test_q4_preserves_inoca_never_minoca():
    """Known failure: the INOCA branch query was previously rewritten to MINOCA."""
    plan = plan_question(MULTI_HOP)
    q4 = plan.query_by_id("q4")
    assert q4 is not None
    assert "INOCA" in q4.text
    assert "MINOCA" not in q4.text
    req4 = plan.requirement_by_id(q4.requirement_ids[0])
    assert "metabolic syndrome" in req4.topic


def test_terminology_guard_clean():
    plan = plan_question(MULTI_HOP)
    assert plan.terminology_guard.get("violations") == []
    assert "INOCA" in plan.entities


def test_required_concepts_attached():
    plan = plan_question(MULTI_HOP)
    h1 = plan.requirement_by_id("H1")
    assert h1 is not None and h1.required_concepts
    assert h1.preferred_evidence_types


def test_simple_factual_question():
    plan = plan_question("What is the definition of heart failure?")
    assert plan.requirements
    assert all(q.requirement_ids for q in plan.queries)


def test_ischemia_phrase_maps_to_inoca():
    """Original phrase must stay attached as the original term (section 8)."""
    plan = plan_question("How does ischemia with nonobstructive coronary arteries affect outcomes?")
    blob = " ".join(r.topic + " " + r.population for r in plan.requirements)
    assert "ischemia with nonobstructive coronary arteries" in blob
    assert not any("minoca" in (r.topic + " " + r.population).lower() for r in plan.requirements)

