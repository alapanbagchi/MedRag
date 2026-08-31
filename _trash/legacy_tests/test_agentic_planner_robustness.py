"""Unit tests for planner robustness (dedupe, JSON-echo strip, fallback entities)."""
from src.agentic.planner import (
    _clean_text,
    _dedupe_subqueries,
    _fallback_entities,
    _clean_subquery,
    SubQueryPlan,
    PlannedEntity,
)


def test_clean_text_strips_json_echo():
    assert _clean_text("radial artery vasospasm IR access}, {id:") == "radial artery vasospasm IR access"
    assert _clean_text("KID-ACS score}, {id: H2") == "KID-ACS score"


def test_dedupe_subqueries_drops_near_duplicates_and_caps():
    subs = [
        SubQueryPlan(id="H1", target="radial artery vasospasm IR access"),
        SubQueryPlan(id="H2", target="radial artery vasospasm prevention in IR access"),
        SubQueryPlan(id="H3", target="radial artery vasospasm CABG harvest"),
        SubQueryPlan(id="H4", target="KID-ACS score AKI risk"),
        SubQueryPlan(id="H5", target="PENK vs serum creatinine"),
    ]
    out = _dedupe_subqueries(subs, cap=4)
    assert len(out) == 4
    # H1 and H2 are near-duplicates -> only H1 survives
    assert any("IR access" in s.target for s in out)
    targets = [s.target for s in out]
    assert targets.count("radial artery vasospasm IR access") == 1


def test_fallback_entities_extracts_concepts():
    ents = _fallback_entities("radial artery vasospasm during interventional radiology access")
    texts = [e.text for e in ents]
    assert any("radial artery vasospasm" in t for t in texts)
    assert any("interventional radiology" in t.lower() for t in texts)


def test_clean_subquery_fills_entities_when_empty():
    sub = SubQueryPlan(id="H1", target="radial artery vasospasm in IR", query="")
    out = _clean_subquery(sub, 1)
    assert out is not None
    assert len(out.entities) > 0  # fallback kicked in
