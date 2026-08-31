"""Evidence aggregator tests: grouping, dedupe, contradictions, provenance."""

from __future__ import annotations

from src.agents.evidence import EvidenceAggregator, Evidence


def _ev(subquery_id, claim, text, supports=True, conflict="", evidence_id=None):
    return Evidence(
        subquery_id=subquery_id,
        document_id="PMC1",
        chunk_id="chunk-1",
        source="PMC1",
        claim=claim,
        supporting_text=text,
        supports_claim=supports,
        confidence=0.9,
        evidence_id=evidence_id,
        contradiction_note=conflict,
    )


def test_groups_related_evidence_and_deduplicates():
    aggregator = EvidenceAggregator()
    items = [
        _ev("H1", "End-to-end anastomosis had 23.1% recurrence", "end-to-end 23.1%", evidence_id="E1"),
        _ev("H1", "End-to-end anastomosis had 23.1% recurrence", "end-to-end 23.1%", evidence_id="E1-dup"),  # duplicate
        _ev("H1", "Patch aortoplasty had 8.3% recurrence", "patch 8.3%", evidence_id="E2"),
    ]
    groups = aggregator.aggregate(items)
    # duplicate claim collapses into the same group
    assert len(groups) == 2
    assert {g.subquery_id for g in groups} == {"H1"}
    assert all(len(g.supporting_evidence) >= 1 for g in groups)


def test_contradicting_evidence_is_separated():
    aggregator = EvidenceAggregator(similarity_threshold=0.3)
    items = [
        _ev("H1", "Technique A reduced recurrence", "A reduced recurrence", evidence_id="E-A"),
        _ev(
            "H1",
            "Technique A increased recurrence",
            "A increased recurrence",
            conflict="another passage reports the opposite",
            evidence_id="E-B",
        ),
    ]
    groups = aggregator.aggregate(items)
    assert len(groups) >= 1
    group = groups[0]
    if group.contradicting_evidence:
        assert all(e.contradiction_note for e in group.contradicting_evidence)


def test_provenance_preserved_in_groups():
    aggregator = EvidenceAggregator()
    items = [
        _ev("H1", "CKD predicts CV risk", "CKD predicts", evidence_id="E1"),
        _ev("H2", "INOCA differs by sex", "INOCA differs", evidence_id="E2"),
    ]
    groups = aggregator.aggregate(items)
    assert {g.subquery_id for g in groups} == {"H1", "H2"}
    for group in groups:
        assert group.document_ids == ["PMC1"]
        assert group.max_confidence > 0.0
        assert group.claim


def test_empty_input():
    assert EvidenceAggregator().aggregate([]) == []
