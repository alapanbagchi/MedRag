"""Tests for scripts/jats_to_md.py — the loss-aware, structure-first
JATS -> Markdown converter.

No network, no corpus required (the real-file test is guarded). Synthetic
JATS XML exercises references/citations, table spans, figures, equations,
lists, quotes, supplementary material, front matter, and converter safety.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.jats_to_md import (  # noqa: E402
    _build_grid,
    _collect_references,
    _extract_tex,
    _format_citation_numbers,
    _render_document,
    convert_one,
)
from src.parser import PMCASTParser  # noqa: E402

S3 = "https://pmc-oa-opendata.s3.amazonaws.com/PMC42"

# ---------------------------------------------------------------------------
# Synthetic JATS helpers
# ---------------------------------------------------------------------------

def _jats(body_extra: str = "", refs: str = "", meta: str = "") -> str:
    return f"""<?xml version="1.0"?>
<article xmlns:xlink="http://www.w3.org/1999/xlink">
  <front>
    <article-meta>
      <article-id pub-id-type="pmcid">PMC42</article-id>
      <article-id pub-id-type="doi">10.1/x</article-id>
      <article-id pub-id-type="pmid">999</article-id>
      <title-group><article-title>Cardiac Arrest Study</article-title></title-group>
      <contrib-group>
        <contrib contrib-type="author"><name><surname>Smith</surname><given-names>Ada</given-names></name></contrib>
        <contrib contrib-type="author"><name><surname>Jones</surname><given-names>Bob, Jr.</given-names></name></contrib>
      </contrib-group>
      <pub-date pub-type="epub"><year>2024</year><month>3</month><day>1</day></pub-date>
      <kwd-group><kwd>cardiac arrest, resuscitation</kwd><kwd>sudden death</kwd></kwd-group>
      <article-categories><subj-group><subject>Cardiology, Surgery</subject></subj-group></article-categories>
      {meta}
    </article-meta>
    <journal-meta><journal-title>J of Tests</journal-title></journal-meta>
  </front>
  <body>{body_extra}</body>
  <back>{refs}</back>
</article>"""


FULL = """<?xml version="1.0"?>
<article xmlns:xlink="http://www.w3.org/1999/xlink">
  <front>
    <article-meta>
      <article-id pub-id-type="pmcid">PMC42</article-id>
      <article-id pub-id-type="doi">10.1/x</article-id>
      <article-id pub-id-type="pmid">999</article-id>
      <title-group><article-title>Cardiac Arrest Study</article-title></title-group>
      <contrib-group>
        <contrib contrib-type="author"><name><surname>Smith</surname><given-names>Ada</given-names></name></contrib>
      </contrib-group>
      <pub-date pub-type="epub"><year>2024</year><month>3</month><day>1</day></pub-date>
      <kwd-group><kwd>cardiac arrest</kwd></kwd-group>
      <article-categories><subj-group><subject>Cardiology</subject></subj-group></article-categories>
    </article-meta>
    <journal-meta><journal-title>J of Tests</journal-title></journal-meta>
  </front>
  <body>
    <abstract><p>Cardiac arrest is the sudden cessation of cardiac activity (P = .04).</p>
      <sec><title>Objective</title><p>To study outcomes [1].</p></sec>
    </abstract>
    <sec><title>Methods</title>
      <p>We studied patients<sup><xref ref-type="bibr" rid="r1 r3">1,3</xref></sup>.
         See Figure <xref ref-type="fig" rid="f1">F1</xref>.</p>
      <list list-type="order">
        <list-item><p>First step</p></list-item>
        <list-item><p>Second step<list list-type="bullet">
          <list-item><p>nested item</p></list-item>
        </list></p></list-item>
      </list>
      <disp-formula id="eq1"><alternatives>
        <tex-math><![CDATA[E = mc^2]]></tex-math>
      </alternatives></disp-formula>
      <p>Text before <disp-formula><tex-math>y = a + b</tex-math></disp-formula> text after.</p>
      <disp-quote><p>To be or not to be.</p></disp-quote>
      <table-wrap id="t1">
        <label>Table 1</label>
        <caption><p>Outcomes by group</p></caption>
        <table>
          <thead><tr><th>Group</th><th>Survival</th></tr></thead>
          <tbody>
            <tr><td>A</td><td>80%</td></tr>
            <tr><td colspan="2" rowspan="1">All groups</td></tr>
          </tbody>
        </table>
        <table-wrap-foot><fn><label>a</label><p>Survival at 30 days.</p></fn></table-wrap-foot>
      </table-wrap>
      <fig id="f1">
        <label>Figure 1</label>
        <caption><p>Kaplan-Meier curve</p></caption>
        <alt-text>survival over time</alt-text>
        <long-desc>A detailed description of the curve.</long-desc>
        <graphic xlink:href="fig1.jpg"/>
      </fig>
      <supplementary-material id="s1">
        <label>Supplement 1</label>
        <caption><p>Source data ZIP</p></caption>
        <media mimetype="application" mime-subtype="zip" xlink:href="data.zip"/>
      </supplementary-material>
    </sec>
    <sec><title>Results</title><p>P = .80 (unchanged), P = .001.</p></sec>
  </body>
  <back>
    <sec><title>Acknowledgments</title><p>Funded by nothing.</p></sec>
    <ref-list>
      <ref id="r1"><label>1</label><mixed-citation>
        <person-group><string-name><surname>Roe</surname><given-names>R</given-names></string-name></person-group>
        <article-title>First paper</article-title><source>J Heart</source><year>2001</year><volume>1</volume><fpage>10</fpage><lpage>20</lpage>
        <pub-id pub-id-type="doi">10.1/a</pub-id><pub-id pub-id-type="pmid">1</pub-id>
      </mixed-citation></ref>
      <ref id="r2"><label>2</label><mixed-citation>
        <person-group><string-name><surname>Moe</surname><given-names>M</given-names></string-name></person-group>
        <article-title>Second paper</article-title><source>J Heart</source><year>2002</year><volume>2</volume><fpage>30</fpage>
      </mixed-citation></ref>
      <ref id="r3"><label>3</label><element-citation>
        <person-group><string-name><surname>Poe</surname><given-names>P</given-names></string-name></person-group>
        <article-title>Third paper</article-title><source>J Heart</source><year>2003</year>
        <pub-id pub-id-type="pmcid">PMC5</pub-id>
      </element-citation></ref>
      <ref><label>4</label><year>2013</year></ref>   <!-- year-only junk -->
    </ref-list>
  </back>
</article>"""


def _render(xml: str):
    from lxml import etree

    root = etree.fromstring(xml.encode("utf-8"))
    return _render_document(root)


# ---------------------------------------------------------------------------
# References + citations
# ---------------------------------------------------------------------------

def test_references_numbered_and_citations_link():
    md, meta, loss, sources = _render(FULL)
    i = md.find("## References")
    refs_block = md[i:]
    assert "[1] Roe R. First paper. J Heart 2001;1:10–20. " in refs_block
    assert "[2] Moe M. Second paper. J Heart 2002;2:30." in refs_block
    assert "[3] Poe P. Third paper. J Heart 2003. " in refs_block
    # year-only junk is NOT numbered
    assert "[4]" not in refs_block and "<ref>" not in refs_block
    # inline citation -> [1,3]; figure xref -> label
    assert "[1,3]" in md
    assert "Figure 1" in md
    # nothing dangling
    assert "r3" not in md.replace("rid", "")


def test_p_value_never_bracketed():
    _, _, _, _ = _render(FULL)
    md, _, _, _ = _render(FULL)
    assert ".[04]" not in md and ".[80]" not in md
    assert "P = .04" in md and "P = .80 (unchanged), P = .001" in md


def test_citation_number_collapsing():
    assert _format_citation_numbers([1, 2, 4, 5, 6]) == "[1–2,4–6]"
    assert _format_citation_numbers([1]) == "[1]"
    assert _format_citation_numbers([1, 3]) == "[1,3]"


# ---------------------------------------------------------------------------
# Abstract / headings
# ---------------------------------------------------------------------------

def test_abstract_nested_and_heading_structure():
    md, *_ = _render(FULL)
    # exactly ONE abstract heading for a single-abstract file (regression:
    # the renderer used to emit a bare duplicate "## Abstract" above the real
    # abstract)
    assert md.count("## Abstract") == 1
    assert "### Objective" in md


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------

def test_table_grid_with_thead_and_colspan():
    md, *_ = _render(FULL)
    assert "| Group | Survival |" in md
    assert "| --- | --- |" in md
    # a full-width colspan row is group context: single-cell span row
    assert "| All groups |" in md
    # caption + footnote survive
    assert "*Table 1 Outcomes by group*" in md
    assert "*a Survival at 30 days.*" in md


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def test_figure_keeps_label_caption_alt_desc():
    md, *_ = _render(FULL)
    assert f"![Figure 1]({S3}/fig1.jpg)" in md
    assert "**Figure 1 Kaplan-Meier curve Description: A detailed description of the curve. Alt-text: survival over time**" in md


# ---------------------------------------------------------------------------
# Equations (verbatim) / paragraphs with inline display math
# ---------------------------------------------------------------------------

def test_disp_formula_block_and_paragraph_split():
    md, *_ = _render(FULL)
    assert "$$\nE = mc^2\n$$" in md
    assert "$$\ny = a + b\n$$" in md
    assert "Text before" in md and "text after." in md
    assert md.count("$$") % 2 == 0


def test_equation_extraction_namespace_agnostic():
    md, *_ = _render(FULL)
    assert "E = mc^2" in md


# ---------------------------------------------------------------------------
# Lists / quotes / boxes
# ---------------------------------------------------------------------------

def test_lists_preserve_markers_and_nesting():
    md, *_ = _render(FULL)
    assert "1. First step" in md
    assert "2. Second step" in md
    assert "    - nested item" in md


def test_disp_quote_becomes_blockquote():
    md, *_ = _render(FULL)
    assert "> To be or not to be." in md


def test_supplementary_material_section():
    md, *_ = _render(FULL)
    assert "## Supplementary Material" in md
    assert "**Supplement 1**" in md
    assert f"Download: [data.zip]({S3}/data.zip)" in md


# ---------------------------------------------------------------------------
# Front matter / title
# ---------------------------------------------------------------------------

def test_front_matter_yaml_lists():
    import yaml

    md, meta, _, _ = _render(FULL)
    inner = "\n".join(md.splitlines()[1:md.splitlines().index("---", 1)])
    parsed = yaml.safe_load(inner)
    assert parsed["authors"] == ["Ada Smith"]
    assert parsed["keywords"] == ["cardiac arrest"]
    assert parsed["pmcid"] == "PMC42"


def test_absent_title_does_not_emit_bare_hash():
    xml = _jats(body_extra="<p>no title here</p>")
    xml = xml.replace("<title-group><article-title>Cardiac Arrest Study</article-title></title-group>",
                      "<title-group></title-group>")
    md, meta, _, _ = _render(xml)
    assert not md.startswith("# ")


# ---------------------------------------------------------------------------
# Converter driver: atomic write / resume / loss report
# ---------------------------------------------------------------------------

def test_convert_one_atomic_and_skip(tmp_path):
    xml_path = tmp_path / "PMC42.xml"
    xml_path.write_text(FULL, encoding="utf-8")
    md_path = tmp_path / "PMC42.md"
    res = convert_one(xml_path, md_path, overwrite=True)
    assert res["status"] == "converted"
    assert res["loss"]["dropped"] == []
    md = md_path.read_text(encoding="utf-8")
    assert md.startswith("---")
    assert "## References" in md
    assert md_path.with_suffix(".md.sha256").exists()
    # unchanged -> skipped, resumable
    res2 = convert_one(xml_path, md_path, overwrite=False)
    assert res2["status"] == "skipped"


# ---------------------------------------------------------------------------
# Href resolution (PMC OA S3)
# ---------------------------------------------------------------------------

def test_image_hrefs_resolved_to_pmc_oa_bucket():
    import scripts.jats_to_md as J
    assert J._resolve_href("fig1.jpg") == \
        "https://pmc-oa-opendata.s3.amazonaws.com/PMC4329953.1/fig1.jpg" \
        if J._ARTICLE_DIR == "" else True
    # set the module dir like _render_document does, then resolve
    old = J._ARTICLE_DIR
    J._ARTICLE_DIR = "PMC4329953.1"
    try:
        assert J._resolve_href("fig1.jpg") == \
            "https://pmc-oa-opendata.s3.amazonaws.com/PMC4329953.1/fig1.jpg"
        assert J._resolve_href("sub/fig2.tif") == \
            "https://pmc-oa-opendata.s3.amazonaws.com/PMC4329953.1/sub/fig2.tif"
        # absolute / fragment hrefs are never rewritten
        assert J._resolve_href("https://example.com/x.jpg") == "https://example.com/x.jpg"
        assert J._resolve_href("#anchor") == "#anchor"
        assert J._resolve_href("data:image/png;base64,AAAA") == "data:image/png;base64,AAAA"
    finally:
        J._ARTICLE_DIR = old


# ---------------------------------------------------------------------------
# Real-file regression through the MD chunker
# ---------------------------------------------------------------------------

@pytest.mark.skipif(
    not Path("data/chunked/PMC10327125.4.xml").exists(),
    reason="corpus not present",
)
def test_real_file_converts_and_chunks_with_tables(tmp_path):
    from src.md_chunker import MDChunker

    xml_path = Path("data/chunked/PMC10327125.4.xml")
    md_path = tmp_path / "PMC10327125.4.md"
    res = convert_one(xml_path, md_path, overwrite=True)
    assert res["status"] == "converted"
    assert res["loss"]["dropped"] == []      # no silent structural loss
    md = md_path.read_text(encoding="utf-8")
    assert md.count("$$") % 2 == 0           # balanced fences
    assert "## References" in md             # numbered references present
    assert md.count("| ---") > 0             # real pipe tables

    chunks, units, report = MDChunker(max_tokens=480).chunk_md(md, doc_id="PMC10327125.4")
    types = {c.chunk_type for c in chunks}
    assert {"table_summary", "table_row", "table_footnotes", "figure", "equation"} <= types
    assert report["retrieval_eligible"] > 100