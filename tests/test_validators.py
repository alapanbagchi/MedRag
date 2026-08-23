"""Tests for ASTValidator and ChunkValidator."""

from __future__ import annotations

from medrag.models import Block, Chunk, Document, Section
from medrag.validators import ASTValidator
from medrag.validators import ChunkValidator


def make_document():
    return Document(
        pmcid="PMC1",
        title="T",
        sections=[Section("Results", 1, ["Results"], blocks=[
            Block("paragraph", "Text.")
        ], section_type="content")],
    )


def test_ast_validator_detects_missing_document_id():
    doc = make_document()
    doc.pmcid = ""
    report = ASTValidator().validate(doc)
    codes = [d.code for d in report.diagnostics]
    assert "MISSING_DOCUMENT_ID" in codes


def test_ast_validator_detects_table_without_rows():
    doc = make_document()
    doc.sections[0].blocks.append(Block(
        "table", "",
        {"table": {"table_id": "t1", "rows": [], "columns": [], "categories": [],
                   "footnotes": [], "structure_degraded": False}},
    ))
    report = ASTValidator().validate(doc)
    codes = [d.code for d in report.diagnostics]
    assert "TABLE_NO_ROWS" in codes


def test_ast_validator_detects_unknown_block_type():
    doc = make_document()
    doc.sections[0].blocks.append(Block("weird", "content"))
    report = ASTValidator().validate(doc)
    assert any(d.code == "UNKNOWN_BLOCK_TYPE" for d in report.diagnostics)


def make_chunk(**overrides):
    base = dict(
        id="PMC1_0",
        document_id="PMC1",
        text="hello",
        embedding_text="Results\n\nhello",
        chunk_type="paragraph",
        document_position=0,
        retrieval_eligible=True,
    )
    base.update(overrides)
    return Chunk(**base)


def test_chunk_validator_detects_duplicate_ids():
    chunks = [
        make_chunk(id="PMC1_0", document_position=0),
        make_chunk(id="PMC1_0", document_position=1),
    ]
    report = ChunkValidator().validate(chunks, "PMC1")
    assert any(d.code == "DUPLICATE_CHUNK_ID" for d in report.diagnostics)


def test_chunk_validator_detects_empty_embedding():
    chunks = [make_chunk(embedding_text="")]
    report = ChunkValidator().validate(chunks, "PMC1")
    assert any(d.code == "EMPTY_EMBEDDING_TEXT" for d in report.diagnostics)


def test_chunk_validator_detects_table_row_without_parent():
    chunks = [
        make_chunk(id="PMC1_summary", chunk_type="table_summary", table_id="t1"),
        make_chunk(id="PMC1_row0", chunk_type="table_row", table_id="t1",
                   parent_id="missing"),
    ]
    report = ChunkValidator().validate(chunks, "PMC1")
    assert any(d.code == "PARENT_NOT_FOUND" for d in report.diagnostics)


def test_chunk_validator_detects_reference_retrieval_eligible():
    chunks = [make_chunk(chunk_type="reference", retrieval_eligible=True,
                         embedding_text="")]
    report = ChunkValidator().validate(chunks, "PMC1")
    assert any(d.code == "REFERENCE_RETRIEVAL_ELIGIBLE" for d in report.diagnostics)


def test_chunk_validator_detects_raw_html_in_table():
    chunks = [make_chunk(chunk_type="table_summary", table_id="t1",
                         text="<table><tr><td>x</td></tr></table>")]
    report = ChunkValidator().validate(chunks, "PMC1")
    assert any(d.code == "TABLE_RAW_HTML" for d in report.diagnostics)


def test_chunk_validator_detects_ordering_violation():
    chunks = [
        make_chunk(id="PMC1_0", document_position=1),
        make_chunk(id="PMC1_1", document_position=0),
    ]
    report = ChunkValidator().validate(chunks, "PMC1")
    assert any(d.code == "ORDERING_VIOLATION" for d in report.diagnostics)


def test_chunk_validator_detects_figure_without_id():
    chunks = [make_chunk(chunk_type="figure", figure_id="")]
    report = ChunkValidator().validate(chunks, "PMC1")
    assert any(d.code == "FIGURE_NO_ID" for d in report.diagnostics)
