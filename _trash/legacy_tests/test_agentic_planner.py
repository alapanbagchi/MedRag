"""Unit tests for agentic planner post-processing (no LLM)."""
import pytest

from src.agentic.planner import (
    _clean_text,
    _is_junk_entity,
    _clean_subquery,
    SubQueryPlan,
    PlannedEntity,
)


def test_junk_entity_detection():
    assert _is_junk_entity("terminology:true,text:")
    assert _is_junk_entity("text")
    assert _is_junk_entity("true")
    assert _is_junk_entity(",")
    assert _is_junk_entity("")
    assert _is_junk_entity("role")
    assert _is_junk_entity("ab")  # <3 letters


def test_real_entity_kept():
    assert not _is_junk_entity("radial artery vasospasm")
    assert not _is_junk_entity("proenkephalin")
    assert not _is_junk_entity("CABG")
    assert not _is_junk_entity("prostate artery embolization")


def test_clean_text_strips_whitespace_and_junk():
    assert _clean_text("  radial artery vasospasm  ") == "radial artery vasospasm"
    assert _clean_text("terminology:true,text:") == ""
    assert _clean_text(None) == ""
    assert _clean_text("true") == ""


def test_clean_subquery_filters_junk_and_dedupes():
    sub = SubQueryPlan(
        id="H1",
        target="vasospasm IR vs CABG",
        intent="compare",
        query="",
        evidence_required=["pharmacological strategies", "pharmacological strategies"],
        entities=[
            PlannedEntity(text="radial artery vasospasm", role="condition"),
            PlannedEntity(text="terminology:true,text:"),
            PlannedEntity(text="radial artery vasospasm", role="condition"),
        ],
    )
    out = _clean_subquery(sub, 1)
    assert out is not None
    assert out.query == "vasospasm IR vs CABG"  # query filled from target
    assert [e.text for e in out.entities] == ["radial artery vasospasm"]
    assert out.evidence_required == ["pharmacological strategies"]


def test_clean_subquery_returns_none_when_empty():
    sub = SubQueryPlan(id="H1", target="", query="", intent="")
    assert _clean_subquery(sub, 1) is None
