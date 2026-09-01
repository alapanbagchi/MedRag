"""Golden tests for the JATS -> AST parser."""

from __future__ import annotations

from pathlib import Path

from src.processing.parser import PMCASTParser


def make_xml(body: str, extra_front: str = "") -> str:
    return f"""<?xml version="1.0"?>
<article xmlns:xlink="http://www.w3.org/1999/xlink">
  <front>
    <journal-meta><journal-title>Test Journal</journal-title></journal-meta>
    <article-meta>
      <article-id pub-id-type="pmcid">PMC12345</article-id>
      <title-group><article-title>Test Article Title</article-title></title-group>
      {extra_front}
    </article-meta>
  </front>
  <body>{body}</body>
</article>"""


def parse(xml_text: str):
    path = Path("/tmp/test_article.xml")
    path.write_text(xml_text, encoding="utf-8")
    return PMCASTParser().parse(path)


def collect_blocks(document):
    blocks = []

    def walk(sections):
        for section in sections:
            blocks.extend(section.blocks)
            walk(section.children)

    walk(document.sections)
    return blocks


def test_simple_prose_section():
    doc = parse(make_xml(
        "<sec id='s1'><title>Introduction</title>"
        "<p>First paragraph.</p><p>Second paragraph.</p></sec>"
    ))
    assert doc.pmcid == "PMC12345"
    assert doc.title == "Test Article Title"
    assert len(doc.sections) == 1
    section = doc.sections[0]
    assert section.title == "Introduction"
    assert section.source_id == "s1"
    assert [b.block_type for b in section.blocks] == ["paragraph", "paragraph"]


def test_nested_section_hierarchy():
    doc = parse(make_xml(
        "<sec id='results'><title>Results</title>"
        "<p>Top level.</p>"
        "<sec id='sub'><title>Baseline</title><p>Nested.</p></sec>"
        "</sec>"
    ))
    results = doc.sections[0]
    assert results.title == "Results"
    assert len(results.blocks) == 1
    assert len(results.children) == 1
    child = results.children[0]
    assert child.title == "Baseline"
    assert child.breadcrumb == ["Results", "Baseline"]
    # parent and child blocks must remain separate
    assert results.blocks[0].content == "Top level."
    assert child.blocks[0].content == "Nested."


def test_labelled_list_detection():
    doc = parse(make_xml(
        "<sec><title>Box</title>"
        "<p><bold>This study adds:</bold></p>"
        "<list><list-item><p>Statement one.</p></list-item>"
        "<list-item><p>Statement two.</p></list-item></list>"
        "</sec>"
    ))
    section = doc.sections[0]
    lists = [b for b in section.blocks if b.block_type == "list"]
    assert len(lists) == 1
    assert lists[0].metadata["label"] == "This study adds:"
    assert "Statement one." in lists[0].content
    # no standalone paragraph block for the label
    assert not any(
        b.block_type == "paragraph" and "This study adds" in b.content
        for b in section.blocks
    )


def test_unrelated_paragraph_not_attached_to_list():
    doc = parse(make_xml(
        "<sec><title>S</title>"
        "<p>A long paragraph about study design that is definitely not a label "
        "and is much longer than any reasonable label would be.</p>"
        "<p><bold>Key points:</bold></p>"
        "<list><list-item><p>Item.</p></list-item></list>"
        "</sec>"
    ))
    section = doc.sections[0]
    lists = [b for b in section.blocks if b.block_type == "list"]
    assert lists[0].metadata["label"] == "Key points:"
    paragraphs = [b for b in section.blocks if b.block_type == "paragraph"]
    assert len(paragraphs) == 1
    assert "study design" in paragraphs[0].content


def test_table_structured_extraction():
    doc = parse(make_xml(
        "<sec><title>Results</title>"
        "<table-wrap id='tbl1'>"
        "<label>Table 1:</label>"
        "<caption><title>Baseline characteristics.</title></caption>"
        "<table>"
        "<thead><tr><th>Characteristics</th><th>&lt;380</th><th>&gt;=470</th></tr></thead>"
        "<tbody>"
        "<tr><td>Demographics</td><td>&#160;</td><td>&#160;</td></tr>"
        "<tr><td>Sex, n (%)</td><td>&#160;</td><td>&#160;</td></tr>"
        "<tr><td>&#8195;Male</td><td>10 (50)</td><td>12 (60)</td></tr>"
        "</tbody>"
        "</table>"
        "<table-wrap-foot><fn id='f1'><label>a</label><p>Note.</p></fn></table-wrap-foot>"
        "</table-wrap></sec>"
    ))
    section = doc.sections[0]
    tables = [b for b in section.blocks if b.block_type == "table"]
    assert len(tables) == 1
    table = tables[0].metadata["table"]
    assert table["table_id"] == "tbl1"
    assert table["caption"].startswith("Baseline characteristics")
    assert table["columns"][1]["name"] == "<380"
    assert table["categories"] == ["Demographics", "Sex, n (%)"]
    rows = table["rows"]
    assert len(rows) == 1
    assert rows[0]["row_label"] == "Male"
    assert rows[0]["group_path"] == ["Demographics", "Sex, n (%)"]
    assert rows[0]["values"]["<380"] == "10 (50)"
    assert table["footnotes"][0]["marker"] == "a"


def test_figure_extraction():
    doc = parse(make_xml(
        "<sec><title>Results</title>"
        "<fig id='fig1'><label>Figure 1</label>"
        "<caption><title>A plot.</title></caption>"
        "<graphic xlink:href='img.jpg'/></fig></sec>"
    ))
    figures = [b for b in collect_blocks(doc) if b.block_type == "figure"]
    assert len(figures) == 1
    assert figures[0].metadata["figure_id"] == "fig1"
    assert figures[0].metadata["caption"] == "A plot."
    assert figures[0].metadata["image_ref"].endswith("img.jpg")


def test_references_section():
    # references live in back matter; build a fuller document
    xml = """<?xml version="1.0"?>
<article xmlns:xlink="http://www.w3.org/1999/xlink">
<front><article-meta>
<article-id pub-id-type="pmcid">PMC9</article-id>
<title-group><article-title>T</article-title></title-group>
</article-meta></front>
<body><sec><title>Results</title><p>Body.</p></sec></body>
<back><ref-list><ref id="bib1">
<mixed-citation>Author A. <article-title>A title.</article-title> <source>J</source> <year>2020</year></mixed-citation>
</ref></ref-list></back>
</article>"""
    path = Path("/tmp/refs.xml")
    path.write_text(xml, encoding="utf-8")
    doc = PMCASTParser().parse(path)
    ref_section = doc.sections[-1]
    assert ref_section.title == "References"
    assert ref_section.classification == "references"
    refs = ref_section.blocks
    assert refs[0].block_type == "reference"
    assert refs[0].metadata["reference_id"] == "bib1"


def test_administrative_section_classification():
    doc = parse(make_xml(
        "<sec><title>FUNDING</title><p>Grant 123.</p></sec>"
    ))
    assert doc.sections[0].classification == "administrative"


def test_graphical_abstract_is_figure():
    xml = """<?xml version="1.0"?>
<article xmlns:xlink="http://www.w3.org/1999/xlink">
<front><article-meta>
<article-id pub-id-type="pmcid">PMC1</article-id>
<title-group><article-title>T</article-title></title-group>
<abstract abstract-type="graphical"><p><fig id="ga1">
<label>Graphical Abstract</label><graphic xlink:href="ga.jpg"/>
</fig></p></abstract>
</article-meta></front>
<body><sec><title>Results</title><p>Body.</p></sec></body>
</article>"""
    path = Path("/tmp/ga.xml")
    path.write_text(xml, encoding="utf-8")
    doc = PMCASTParser().parse(path)
    ga = doc.sections[0]
    assert ga.title == "Graphical Abstract"
    figures = [b for b in ga.blocks if b.block_type == "figure"]
    assert len(figures) == 1
    assert figures[0].metadata["figure_id"] == "ga1"


def test_unknown_block_type_not_lost():
    doc = parse(make_xml(
        "<sec><title>S</title><weird-tag>Some content here.</weird-tag></sec>"
    ))
    blocks = collect_blocks(doc)
    assert len(blocks) == 1
    assert blocks[0].content == "Some content here."
    assert blocks[0].metadata["original_block_type"] == "weird-tag"
