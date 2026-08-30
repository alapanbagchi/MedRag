"""Evidence extraction agent tests: provenance retained, quotes grounded."""

from __future__ import annotations

import json

from src.agents.evidence import EvidenceExtractor
from src.agents.planner import SubQuery
from src.config import AppConfig
from src.lib import quote_grounded
from src.retrieval.retriever import RetrievedDocument

from tests.conftest import native_test_model

DOC_TEXT = (
    "End-to-end anastomosis was associated with recurrent coarctation in "
    "23.1% of patients (p = 0.04), whereas patch aortoplasty showed a "
    "recurrence rate of 8.3% (p = 0.31). MRI demonstrated focal narrowing at "
    "the anastomotic site in the recurrent lesions."
)

SUB = SubQuery(
    id="H1",
    target="repair techniques",
    focus="association",
    query="repair techniques associated with recurrent coarctation percentages p-values",
    evidence_required=["percentages", "p-values"],
)

DOC = RetrievedDocument(
    subquery_id="H1",
    document_id="PMC11743609",
    chunk_id="PMC11743609-chunk-42",
    rank=1,
    rrf_score=0.5,
    methods=["bm25", "dense"],
    node_type="table_row",
    section="Results",
    breadcrumb=["Results", "Recurrent coarctation"],
    table_id="T2",
    text=DOC_TEXT,
    token_count=len(DOC_TEXT.split()),
)

EV_JSON = json.dumps(
    {
        "document_id": "PMC11743609",
        "subquery_id": "H1",
        "items": [
            {
                "claim": "End-to-end anastomosis was associated with recurrent coarctation in 23.1% of patients (p = 0.04)",
                "supporting_text": "End-to-end anastomosis was associated with recurrent coarctation in 23.1% of patients (p = 0.04)",
                "supports_claim": True,
                "confidence": 0.95,
                "evidence_type": "table_row",
                "section": "Results",
                "subsection": "",
                "breadcrumb": ["Results", "Recurrent coarctation"],
                "table_id": "T2",
                "figure_id": None,
                "page_start": None,
                "page_end": None,
                "is_inference": False,
                "contradiction_note": "",
            },
            {
                "claim": "Patch aortoplasty showed a recurrence rate of 8.3%",
                # whitespace/case drift vs the source -> must STILL ground
                "supporting_text": "patch aortoplasty showed a recurrence rate of 8.3% (p = 0.31)",
                "supports_claim": True,
                "confidence": 0.9,
                "evidence_type": "table_row",
                "is_inference": False,
                "contradiction_note": "",
            },
            {
                "claim": "The study proved patch aortoplasty is always superior",
                "supporting_text": "This fabricated sentence is nowhere in the document.",
                "supports_claim": True,
                "confidence": 0.9,
                "evidence_type": "paragraph",
                "is_inference": False,
                "contradiction_note": "",
            },
        ],
    }
)


async def test_evidence_retains_provenance_and_drops_fabricated_quotes():
    agent = EvidenceExtractor(model=native_test_model(EV_JSON), config=AppConfig())
    items, stats = await agent.extract_with_stats(SUB, DOC)

    # The fabricated quote is not in the document -> dropped deterministically.
    assert len(items) == 2
    assert stats["quotes_dropped"] == 1
    assert stats["kept_items"] == 2

    for item in items:
        assert item.document_id == DOC.document_id
        assert item.chunk_id == DOC.chunk_id
        assert item.subquery_id == SUB.id
        assert item.source == DOC.document_id
        # evidence ids are assigned by the orchestrator AFTER workers finish
        assert item.evidence_id is None

    assert items[0].section == "Results"
    assert items[0].table_id == "T2"


async def test_whitespace_drift_still_grounds():
    agent = EvidenceExtractor(model=native_test_model(EV_JSON), config=AppConfig())
    items, _stats = await agent.extract_with_stats(SUB, DOC)
    ok, how = quote_grounded(items[1].supporting_text, DOC_TEXT)
    assert ok, f"drifted quote must ground (got {how})"


async def test_empty_evidence_is_valid():
    empty = native_test_model('{"document_id": "PMC11743609", "subquery_id": "H1", "items": []}')
    agent = EvidenceExtractor(model=empty, config=AppConfig())
    items = await agent.extract(SUB, DOC)
    assert items == []


def test_grounding_guard_normalizes_whitespace():
    ok, how = quote_grounded("patched   aortoplasty showed", "patched aortoplasty showed a recurrence")
    assert ok
    ok2, how2 = quote_grounded("totally different sentence", DOC_TEXT)
    assert not ok2 and how2 == "none"
