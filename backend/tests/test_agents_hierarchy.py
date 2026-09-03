"""Evidence hierarchy tests (Phase 4)."""

from __future__ import annotations

from src.agents.state import (
    EvidenceItem,
    classify_study_type,
)


def test_classify_study_type():
    assert classify_study_type("A randomized controlled trial found...") == "rct"
    assert classify_study_type("meta-analysis of 12 trials") == "meta_analysis"
    assert classify_study_type("a prospective cohort study") == "cohort"
    assert classify_study_type("case-control study design") == "case_control"
    assert classify_study_type("the clinical guideline recommends...") == "guideline"
    assert classify_study_type("some unrelated prose here") == "unknown"


def test_derive_evidence_level_rankings():
    cases = [
        ("meta_analysis", "1a"),
        ("systematic_review", "1a"),
        ("rct", "1b"),
        ("cohort", "2b"),
        ("case_control", "3b"),
        ("guideline", "1a"),
        ("unknown", "4"),
    ]
    for marker, expected in cases:
        text = {"meta_analysis": "A meta-analysis found...",
                "systematic_review": "This systematic review...",
                "rct": "A randomized controlled trial...",
                "cohort": "In a prospective cohort...",
                "case_control": "A case-control study...",
                "guideline": "The clinical guideline states...",
                "unknown": "something vague"}[marker]
        item = EvidenceItem(id="x", text=text)
        assert item.derive_evidence_level() == expected, marker


def test_consumer_or_low_reliability_downgrades_to_level_4():
    item = EvidenceItem(id="x", text="A meta-analysis of 10 trials...",
                        trust="consumer")
    assert item.derive_evidence_level() == "4"
    item2 = EvidenceItem(id="y", text="A randomized controlled trial...",
                         reliability="low")
    assert item2.derive_evidence_level() == "4"


def test_derive_fills_study_type_when_blank():
    item = EvidenceItem(id="z", text="randomized controlled trial showed...")
    item.derive_evidence_level()
    assert item.study_type == "rct"
    assert item.evidence_level == "1b"
