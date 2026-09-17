"""Deterministic citation-guarantee tests (pure text transforms, no LLM)."""
from __future__ import annotations

from src.lib.answer_cite import (
    build_references,
    cited_refs,
    enforce_citations,
    sort_refs,
)


def _rec(ref, title="", url="", document_id=""):
    return {"ref": ref, "title": title, "url": url, "document_id": document_id}


RECORDS = [
    _rec("P1", "Health Benefits of DASH - NHLBI, NIH",
         "https://www.nhlbi.nih.gov/health/dash/health-benefits"),
    _rec("P2", "DASH diet to lower high blood pressure: MedlinePlus",
         "https://medlineplus.gov/ency/patientinstructions/000770.htm"),
    _rec("P3", "DASH - DASH Eating Plan | NHLBI, NIH",
         "https://www.nhlbi.nih.gov/health/dash-eating-plan"),
    _rec("P4", "", "", "PMC11093532"),
    _rec("P10", "", "", "PMC10990009"),
]


def test_cited_refs_first_appearance_order():
    text = "X [P10]. Then [p3], later [P10] again and [P1]."
    assert cited_refs(text) == ["P10", "P3", "P1"]


def test_sort_refs_numeric():
    assert sort_refs(["P10", "P2", "P1"]) == ["P1", "P2", "P10"]


def test_uncited_block_gets_verified_refs_appended():
    text = "Diet reduces blood pressure."
    out = enforce_citations(text, RECORDS)
    assert "[P1][P2][P3][P4][P5]".replace("P5", "P10") in out or "[P1][P2][P3][P4]" in out
    assert out.startswith("Diet reduces blood pressure.[P")
    # the rebuilt References lists every attached ref, deduped
    refs_section = out.split("## References", 1)[1]
    assert "## References" in out
    assert refs_section.count("[P1]") == 1
    assert refs_section.count("[P1] Health Benefits of DASH") == 1


def test_cited_block_untouched():
    text = "Sodium raises pressure [P2]."
    out = enforce_citations(text, RECORDS)
    assert "Sodium raises pressure [P2]." in out
    assert "Sodium raises pressure [P2].[" not in out


def test_headings_and_markdown_structure_left_alone():
    text = "# How Diet Affects Hypertension\n\n## Mechanisms\n\nBody text here."
    out = enforce_citations(text, RECORDS)
    assert "# How Diet Affects Hypertension" in out
    assert "## Mechanisms\n" in out  # no markers glued onto the heading
    assert "Body text here.[P" in out  # the prose block got its refs


def test_model_references_section_replaced_and_deduped():
    text = (\
        "The DASH diet works [P4].\n\n"\
        "WHO guidance [P10].\n\n"\
        "## References\n"\
        "4 PMC11093532 —\n"\
        "4 PMC11093532 —\n"\
        "10 PMC10990009 —\n"\
        "1 Health Benefits of DASH — url")
    out = enforce_citations(text, RECORDS)
    refs_section = out.split("## References", 1)[1]
    # exactly the cited refs, once each, in citation order
    assert refs_section.count("PMC11093532") == 1
    assert "PMC10990009" in refs_section
    # the body’s uncited "WHO guidance" block gained a marker
    assert "WHO guidance [P" in out
    # model-written duplicate is gone (replaced by the ledger build)
    assert refs_section.count("[P4]") == 1
    assert refs_section.count("[P10]") == 1


def test_enforce_citations_without_records_is_noop():
    text = "Plain answer without any sources."
    out = enforce_citations(text, [])
    assert out == "Plain answer without any sources."


def test_build_references_url_and_docid_labels():
    section = build_references([RECORDS[0], RECORDS[3]])
    assert section.startswith("## References")
    assert "Health Benefits of DASH" in section
    assert "https://www.nhlbi.nih.gov" in section
    assert "PMC11093532" in section
