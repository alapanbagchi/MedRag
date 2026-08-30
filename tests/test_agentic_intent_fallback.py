"""Tests for intent/evidence fallbacks when the model omits them."""
from src.agentic.planner import _fallback_intent, _fallback_evidence, _clean_subquery, SubQueryPlan


def test_fallback_intent_comparison():
    assert "compar" in _fallback_intent("compare vasospasm IR vs CABG")


def test_fallback_intent_mechanism():
    assert "mechanism" in _fallback_intent("how KID-ACS uses PENK")


def test_fallback_intent_risk():
    assert "stratification" in _fallback_intent("AKI risk stratification")


def test_fallback_evidence_comparison_and_pharmacology():
    ev = _fallback_evidence("compare pharmacological strategies vasospasm vs CABG")
    assert any("comparison" in e for e in ev)
    assert any("pharmacolog" in e for e in ev)


def test_fallback_evidence_imaging():
    ev = _fallback_evidence("imaging strategies for vasospasm")
    assert any("imaging" in e for e in ev)


def test_fallback_evidence_empty_target_generic():
    ev = _fallback_evidence("")
    assert ev == ["evidence directly addressing the target"]


def test_clean_subquery_fills_intent_and_evidence():
    sub = SubQueryPlan(id="H1", target="compare vasospasm IR vs CABG", query="",
                       intent="", evidence_required=[])
    out = _clean_subquery(sub, 1)
    assert out is not None
    assert out.intent != ""
    assert out.evidence_required != []
