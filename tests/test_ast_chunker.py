"""Golden tests for the AST chunker.

These build synthetic Document ASTs directly (no XML required) so chunking
semantics are tested deterministically and independently of the parser.
"""

from __future__ import annotations

from medrag.models import Block, Document, Section
from medrag.chunker import ASTChunker


def make_document(pmcid="PMC1", sections=None):
    return Document(
        pmcid=pmcid,
        title="Test Article",
        metadata={"article_meta": {"article_ids": {"pmcid": pmcid}}},
        sections=sections or [],
    )


def section(title, blocks=None, children=None, section_type="content", level=1):
    return Section(
        title=title,
        level=level,
        breadcrumb=[title],
        blocks=blocks or [],
        children=children or [],
        metadata={"section_type": section_type},
        section_type=section_type,
    )


def paragraph(text, block_id=None):
    meta = {"block_id": block_id} if block_id else {}
    return Block("paragraph", text, meta)


def run(document):
    return ASTChunker().chunk(document)


def test_simple_prose_packs_paragraphs():
    doc = make_document(sections=[
        section("A", [paragraph("One."), paragraph("Two.")])
    ])
    chunks = run(doc)
    assert len(chunks) == 1
    assert chunks[0].chunk_type == "paragraph"
    assert "One." in chunks[0].text and "Two." in chunks[0].text
    assert chunks[0].breadcrumb == ["A"]


def test_section_boundary_not_crossed():
    doc = make_document(sections=[
        section("Results", [paragraph("A.")]),
        section("Discussion", [paragraph("B.")]),
    ])
    chunks = run(doc)
    assert len(chunks) == 2
    assert chunks[0].breadcrumb == ["Results"]
    assert chunks[1].breadcrumb == ["Discussion"]


def test_subsection_boundary_not_crossed():
    doc = make_document(sections=[
        section("Results", children=[
            section("Baseline", [paragraph("A.")], level=2),
            section("Outcomes", [paragraph("B.")], level=2),
        ])
    ])
    chunks = run(doc)
    assert len(chunks) == 2
    assert chunks[0].breadcrumb == ["Results", "Baseline"]
    assert chunks[1].breadcrumb == ["Results", "Outcomes"]


def test_labelled_list_becomes_one_list_chunk():
    doc = make_document(sections=[
        section("S", [
            Block("list", "- item 1\n- item 2", {"label": "This study adds:"}),
        ])
    ])
    chunks = run(doc)
    assert len(chunks) == 1
    assert chunks[0].chunk_type == "list"
    assert chunks[0].text.startswith("This study adds:")
    assert "item 1" in chunks[0].text


def test_table_hierarchy():
    table = {
        "table_id": "tbl1",
        "label": "Table 1",
        "caption": "Baseline characteristics.",
        "columns": [{"index": 0, "name": "Characteristic"},
                    {"index": 1, "name": "<380"},
                    {"index": 2, "name": ">=470"}],
        "categories": ["Demographics", "Sex, n (%)"],
        "rows": [
            {"row_label": "Male", "group_path": ["Demographics", "Sex, n (%)"],
             "values": {"<380": "10 (50)", ">=470": "12 (60)"},
             "source_block_ids": ["tbl1"]},
        ],
        "footnotes": [{"marker": "a", "text": "Note."}],
        "structure_degraded": False,
    }
    doc = make_document(sections=[
        section("Results", [Block("table", "Table 1", {"table": table, "block_id": "tbl1"})])
    ])
    chunks = run(doc)
    types = [c.chunk_type for c in chunks]
    assert "table_summary" in types
    assert "table_row" in types
    assert "table_footnotes" in types

    summary = next(c for c in chunks if c.chunk_type == "table_summary")
    row = next(c for c in chunks if c.chunk_type == "table_row")
    footnote = next(c for c in chunks if c.chunk_type == "table_footnotes")

    assert row.parent_id == summary.id
    assert footnote.parent_id == summary.id
    assert row.table_id == "tbl1"
    assert "Male" in row.text
    assert "10 (50)" in row.text
    assert summary.text.startswith("Table 1: Baseline characteristics.")


def test_figure_is_atomic_boundary():
    doc = make_document(sections=[
        section("Results", [
            paragraph("Before."),
            Block("figure", "Figure 1: A plot.", {"figure_id": "fig1"}),
            paragraph("After."),
        ])
    ])
    chunks = run(doc)
    types = [c.chunk_type for c in chunks]
    assert types == ["paragraph", "figure", "paragraph"]
    fig = chunks[1]
    assert fig.figure_id == "fig1"


def test_references_not_retrieval_eligible():
    doc = make_document(sections=[
        section("References", [
            Block("reference", "Author A. Title.", {"reference_id": "bib1"}),
            Block("reference", "Author B. Title.", {"reference_id": "bib2"}),
        ], section_type="references")
    ])
    chunks = run(doc)
    assert len(chunks) == 2
    assert all(c.chunk_type == "reference" for c in chunks)
    assert all(not c.retrieval_eligible for c in chunks)
    assert all(c.embedding_text == "" for c in chunks)


def test_administrative_not_retrieval_eligible():
    doc = make_document(sections=[
        section("Funding", [paragraph("Grant 123.")], section_type="administrative")
    ])
    chunks = run(doc)
    assert chunks[0].chunk_type == "administrative"
    assert not chunks[0].retrieval_eligible


def test_oversized_paragraph_split_on_sentences():
    sentence = "This is a sentence with enough words. "
    big = sentence * 150  # well over hard_max
    doc = make_document(sections=[section("A", [paragraph(big)])])
    chunks = run(doc)
    assert len(chunks) > 1
    assert all(c.chunk_type == "paragraph" for c in chunks)
    # no lost text
    joined = " ".join(c.text for c in chunks)
    assert "enough words" in joined
    # no mid-word splitting
    for c in chunks:
        assert not c.text.rstrip().endswith("wor")


def test_table_without_header_uses_fallback_columns():
    table = {
        "table_id": "tbl2",
        "label": "",
        "caption": "",
        "columns": [],
        "categories": [],
        "rows": [{"row_label": "X", "group_path": [], "values": {"column_1": "5"}}],
        "footnotes": [],
        "structure_degraded": True,
    }
    doc = make_document(sections=[
        section("R", [Block("table", "", {"table": table, "block_id": "tbl2"})])
    ])
    chunks = run(doc)
    row = next(c for c in chunks if c.chunk_type == "table_row")
    assert row.table_id == "tbl2"
    assert row.parent_id is not None


def test_duplicate_table_ids_do_not_collide():
    table1 = {"table_id": "tbl1", "label": "Table 1", "caption": "",
              "columns": [], "categories": [], "rows": [],
              "footnotes": [], "structure_degraded": True}
    table2 = {"table_id": "tbl1", "label": "Table 1", "caption": "",
              "columns": [], "categories": [], "rows": [],
              "footnotes": [], "structure_degraded": True}
    doc = make_document(sections=[
        section("R", [
            Block("table", "", {"table": table1, "block_id": "tbl1"}),
            Block("table", "", {"table": table2, "block_id": "tbl1"}),
        ])
    ])
    chunks = run(doc)
    ids = [c.id for c in chunks]
    assert len(ids) == len(set(ids))


def test_unknown_block_falls_back_without_loss():
    doc = make_document(sections=[
        section("S", [Block("weird_type", "Some content.", {"block_id": "x1"})])
    ])
    chunks = run(doc)
    assert len(chunks) == 1
    assert chunks[0].text == "Some content."
    assert chunks[0].metadata["original_block_type"] == "weird_type"


def test_determinism():
    doc = make_document(sections=[
        section("Results", [
            paragraph("One."),
            Block("list", "- a\n- b", {"label": "Key points:"}),
            Block("table", "t", {"table": {
                "table_id": "tbl1", "label": "Table 1", "caption": "Cap.",
                "columns": [], "categories": [],
                "rows": [{"row_label": "r", "group_path": [], "values": {"c": "1"}}],
                "footnotes": [], "structure_degraded": False,
            }, "block_id": "tbl1"}),
        ])
    ])
    chunks1 = ASTChunker().chunk(doc)
    chunks2 = ASTChunker().chunk(doc)

    assert [c.id for c in chunks1] == [c.id for c in chunks2]
    assert [c.text for c in chunks1] == [c.text for c in chunks2]
    assert [c.chunk_type for c in chunks1] == [c.chunk_type for c in chunks2]
    assert [c.document_position for c in chunks1] == [c.document_position for c in chunks2]
    assert [c.parent_id for c in chunks1] == [c.parent_id for c in chunks2]
