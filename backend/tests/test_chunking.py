"""Unit tests for the chunking package (src/chunking) — Markdown-native chunking.

No LLM, no corpus, no network: synthetic Markdown documents exercise the
parser, parent/child units, references, and determinism through the public
library entry chunk_document().
"""

from __future__ import annotations

import re

import pytest

from src.chunking import chunk_document
from src.chunking.cli import main as cli_main
from src.chunking.parsing import _document_metadata, _parse_front_matter
from src.chunking.prose import split_long_paragraph_pieces, split_overlap_budget
from src.chunking.tokens import _estimate_tokens, _split_sentences


class Chunker:
    """Thin local wrapper that mirrors the old MDChunker.chunk_md() call shape
    so the tests stay readable against the functional chunk_document API."""

    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def chunk_md(self, text, doc_id=None):
        return chunk_document(text, doc_id=doc_id, **self.kwargs)


def make_md(**overrides):
    front = {
        "pmcid": "PMC42",
        "title": "A Cardiac Arrest Study",
        "journal": "Journal of Tests",
        "published": "2024-03-01",
        "doi": "10.9999/test",
        "keywords": "cardiac arrest, resuscitation",
        "authors": "Ada Lovelace, Grace Hopper",
        "categories": "Cardiology",
    }
    front.update(overrides)
    body = "\n".join([
        "## Abstract",
        "Cardiac arrest is the sudden cessation of cardiac activity.",
        "",
        "## Methods",
        "We studied patients. First sentence of methods. Second sentence here. "
        "Third sentence completes the paragraph.",
        "",
        "## Results",
        "Results showed improvement.",
        "- item one",
        "- item two",
        "",
        "*Table 1: outcomes by group*",
        "",
        "| Group | Survival |",
        "|-------|----------|",
        "| A     | 80%      |",
        "| B     | 70%      |",
        "",
        "*a* Survival at 30 days.",
        "",
        "![FIGURE 1](https://example.com/fig1.jpg)",
        "**FIGURE 1 Kaplan-Meier curve.**",
        "",
        "$$",
        "E = mc^2",
        "$$",
        "",
        "## Funding",
        "Supported by a grant.",
        "",
        "## References",
        "- Smith J. Cardiac Arrest. DOI: [10.1111/x](https://doi.org/10.1111/x)",
        "- Roe R. Outcomes. PMID: [12345](https://pubmed.ncbi.nlm.nih.gov/12345)",
    ])
    head = "---\n" + "\n".join(f"{k}: {v}" for k, v in front.items()) + "\n---\n"
    return head + body


@pytest.fixture()
def chunker():
    return Chunker(max_tokens=120)


def test_front_matter_parsing():
    md = make_md()
    front, body = _parse_front_matter(md)
    assert front["pmcid"] == "PMC42"
    assert front["title"] == "A Cardiac Arrest Study"
    assert "## Abstract" in body


def test_document_metadata_normalization():
    meta = _document_metadata(
        {"pmcid": "PMC1", "keywords": "a, b, c", "authors": ["X", "Y"]}, "PMC1"
    )
    assert meta["keywords"] == ["a", "b", "c"]
    assert meta["authors"] == ["X", "Y"]
    assert meta["pmcid"] == "PMC1"


def test_doc_title_not_a_section_breadcrumb_starts_at_sections(chunker):
    chunks, units, report = chunker.chunk_md(make_md(), doc_id="PMC42")
    sections = {c.section for c in chunks}
    assert "A Cardiac Arrest Study" not in sections
    assert "Abstract" in sections
    # every chunk's breadcrumb starts with the real section heading
    for c in chunks:
        assert c.breadcrumb[0] in ("Abstract", "Methods", "Results",
                                   "Funding", "References")


def test_types_and_eligibility(chunker):
    chunks, _, _ = chunker.chunk_md(make_md(), doc_id="PMC42")
    by_type = {c.chunk_type for c in chunks}
    assert {"paragraph", "table_summary", "table_row", "table_footnotes",
            "figure", "equation", "administrative", "reference"} <= by_type
    assert all(c.retrieval_eligible for c in chunks
               if c.chunk_type not in ("reference", "administrative"))
    assert all(not c.retrieval_eligible for c in chunks
               if c.chunk_type in ("reference", "administrative"))


def test_embedding_text_is_plain_chunk_text(chunker):
    # no embedding logic in the chunker: embedding_text is the chunk text
    # (encoding is a separate step that reads the stored column)
    chunks, _, _ = chunker.chunk_md(make_md(), doc_id="PMC42")
    for c in chunks:
        assert c.embedding_text == c.text
    abstract = [c for c in chunks if c.chunk_type == "paragraph"
                and c.section == "Abstract"][0]
    assert abstract.text.startswith("Cardiac arrest is the sudden cessation")


def test_table_summary_rows_footnotes_and_units(chunker):
    chunks, units, _ = chunker.chunk_md(make_md(), doc_id="PMC42")
    summary = [c for c in chunks if c.chunk_type == "table_summary"][0]
    rows = [c for c in chunks if c.chunk_type == "table_row"]
    fns = [c for c in chunks if c.chunk_type == "table_footnotes"]
    table_units = [u for u in units if u.kind == "table"]
    assert len(rows) == 2
    assert len(fns) == 1
    assert len(table_units) == 1
    tu = table_units[0]
    assert "outcomes by group" in summary.text
    # uniform parent_id semantics: every table chunk points at its TABLE unit
    assert summary.parent_id == tu.unit_id
    assert all(r.parent_id == tu.unit_id for r in rows)
    assert fns[0].parent_id == tu.unit_id
    # rows also carry the SECTION unit + order via metadata
    assert all(r.metadata.get("unit_id") for r in rows)
    assert [r.metadata["row_index"] for r in rows] == [0, 1]
    assert rows[0].row_label == "A"
    # row text carries the row label + one line per column value
    assert "A" in rows[0].text and "Survival: 80%" in rows[0].text
    assert summary.id in tu.chunk_ids
    assert all(r.id in tu.chunk_ids for r in rows)


def test_figure_caption_and_image_ref(chunker):
    chunks, _, _ = chunker.chunk_md(make_md(), doc_id="PMC42")
    fig = [c for c in chunks if c.chunk_type == "figure"][0]
    assert "Kaplan-Meier curve" in fig.text
    assert fig.figure_id
    assert fig.figure_link == "https://example.com/fig1.jpg"          # first-class link
    assert fig.metadata.get("image_ref") == "https://example.com/fig1.jpg"  # back-compat


def test_equation_fence(chunker):
    chunks, _, _ = chunker.chunk_md(make_md(), doc_id="PMC42")
    eq = [c for c in chunks if c.chunk_type == "equation"]
    assert len(eq) == 1
    assert "E = mc^2" in eq[0].text


def test_references_and_doi(chunker):
    chunks, _, _ = chunker.chunk_md(make_md(), doc_id="PMC42")
    refs = [c for c in chunks if c.chunk_type == "reference"]
    assert len(refs) == 2
    assert refs[0].reference_id
    assert refs[0].metadata["doi"] == "10.1111/x"
    assert refs[1].metadata["pmid"] == "12345"
    assert refs[0].embedding_text == refs[0].text  # no special embedding for refs


def test_split_paragraph_carry_between_consecutive_pieces():
    c = Chunker(max_tokens=60)
    text = "Sentence number one has some words to fill the token budget. " * 40
    md = "---\npmcid: PMC1\n---\n## Methods\n" + text
    chunks, _, _ = c.chunk_md(md, doc_id="PMC1")
    paras = [ch for ch in chunks if ch.chunk_type == "paragraph"]
    assert len(paras) >= 2
    # paragraph-first: the healthy carry means each piece after a split
    # opens with the tail of the previous piece (intra-paragraph overlap)
    shared = None
    for a, b in zip(paras, paras[1:]):
        tail_words = a.text.split()[-20:]
        if any(w in b.text for w in tail_words):
            shared = True
            break
    assert shared, "expected sentence carry between consecutive split pieces"


def test_deterministic_output():
    a = Chunker(max_tokens=120)
    b = Chunker(max_tokens=120)
    ca, ua, _ = a.chunk_md(make_md(), doc_id="PMC42")
    cb, ub, _ = b.chunk_md(make_md(), doc_id="PMC42")
    assert [c.id for c in ca] == [c.id for c in cb]
    assert [c.text for c in ca] == [c.text for c in cb]
    assert [u.unit_id for u in ua] == [u.unit_id for u in ub]


def test_units_cover_all_chunks(chunker):
    chunks, units, _ = chunker.chunk_md(make_md(), doc_id="PMC42")
    covered = set()
    for u in units:
        covered.update(u.chunk_ids)
    assert set(c.id for c in chunks) == covered


# -- contextual embeddings, healthy split overlap, late-chunking hooks --

def test_split_overlap_healthy_carry_white_box():
    para = " ".join(f"Visit number {i} recorded a precise measurement for the cohort."
                     for i in range(20))
    budget = split_overlap_budget(80, 10 ** 6)
    pieces = split_long_paragraph_pieces(para, 80, 640, 2, budget)
    assert len(pieces) >= 3
    s0 = re.split(r"(?<=[.!?])\s+", pieces[0])
    s1 = re.split(r"(?<=[.!?])\s+", pieces[1])
    # healthy carry: the next piece STARTS with the last 2 sentences of the
    # previous piece, and the carry respects the token budget
    assert s1[0] == s0[-2] and s1[1] == s0[-1]
    assert _estimate_tokens(s1[0]) + _estimate_tokens(s1[1]) <= budget


def test_split_overlap_zero_disables_carry():
    para = " ".join(f"Outcome {i} differed between the two randomized groups."
                     for i in range(20))
    budget = split_overlap_budget(80)
    pieces = split_long_paragraph_pieces(para, 80, 640, 0, budget)
    assert len(pieces) >= 2
    s0 = re.split(r"(?<=[.!?])\s+", pieces[0])
    s1 = re.split(r"(?<=[.!?])\s+", pieces[1])
    assert s1[0] != s0[-1]  # unrelated sentences, no carry


def test_split_paragraphs_emit_paragraph_units_and_late_chunk_hooks():
    para = " ".join(f"Measurement {i} was recorded precisely for the cohort group."
                     for i in range(40))
    md = f"---\npmcid: PMC1\n---\n## Results\n{para}"
    c = Chunker(max_tokens=80)
    chunks, units, report = c.chunk_md(md, doc_id="PMC1")
    splitters = [ch for ch in chunks if ch.chunk_type == "paragraph"
                 and ch.metadata.get("sentence_split")]
    assert splitters
    p_units = [u for u in units if u.kind == "paragraph"]
    assert len(p_units) == 1
    pu = p_units[0]
    assert pu.text == para  # FULL source paragraph text (late-chunking hook)
    assert pu.title == "Results"
    assert len(pu.chunk_ids) == len(splitters)
    pu_by_id = {u.unit_id: u for u in p_units}
    covered = set(pu.chunk_ids)
    idxs = []
    for ch in splitters:
        ss = ch.metadata["sentence_split"]
        assert ss["paragraph_unit_id"] in pu_by_id
        idxs.append(ss["piece_index"])
        assert ss["piece_count"] == len(splitters)
        assert ch.id in covered
    assert sorted(idxs) == list(range(len(splitters)))
    assert report["paragraphs_sentence_split"] == 1
    assert report["split_pieces"] == len(splitters)


def test_sentence_split_respects_abbreviations_and_decimals():
    text = ("We used e.g. the standard protocol, which is well known. "
            "Fig. 3 shows the results. No. of patients was 100. "
            "The dose was 5 mg/dL. Results confirm the hypothesis.")
    sents = _split_sentences(text)
    assert "We used e.g. the standard protocol, which is well known." in sents
    assert "Fig. 3 shows the results." in sents
    assert "No. of patients was 100." in sents
    assert "The dose was 5 mg/dL." in sents
    assert "Results confirm the hypothesis." in sents

def test_units_hierarchy_parent_links():
    # nested sections + a split paragraph + a table: every unit knows its parent
    para = " ".join(f"Measurement {i} was recorded precisely for the cohort group."
                     for i in range(40))
    md = ("---\npmcid: PMC9\n---\n"
          "# Title\n"
          "## Study Design\n"
          "### Cohort\n"
          f"{para}\n"
          "## Results\n"
          "*Table 3: outcomes*\n\n| Group | Value |\n|-------|-------|\n| A | 1 |\n")
    chunks, units, _ = Chunker(max_tokens=80).chunk_md(md, doc_id="PMC9")
    by_id = {u.unit_id: u for u in units}
    sections = [u for u in units if u.kind == "section"]
    assert len(sections) == 3  # Study Design, Cohort, Results
    design = by_id[sections[0].unit_id]
    cohort = by_id[sections[1].unit_id]
    results = by_id[sections[2].unit_id]
    # root section has no parent; the subsection hangs off it
    assert design.parent_unit_id is None
    assert cohort.parent_unit_id == design.unit_id
    assert results.parent_unit_id is None
    # paragraph unit hangs off its enclosing section (Cohort)
    paras = [u for u in units if u.kind == "paragraph"]
    assert paras and all(u.parent_unit_id == cohort.unit_id for u in paras)
    # table unit hangs off its enclosing section (Results)
    tables = [u for u in units if u.kind == "table"]
    assert tables and tables[0].parent_unit_id == results.unit_id
    # chunks still reference a parent unit that exists in the tree
    assert all(c.parent_id in by_id for c in chunks if c.chunk_type in
               ("paragraph", "table_summary", "table_row"))

# -- robustness regressions --------------------------------------------------

def test_single_line_equation_strips_trailing_dollars():
    c = Chunker(max_tokens=120)
    chunks, _, _ = c.chunk_md("---\npmcid: PMC1\n---\n## M\n$$E=mc^2$$\n",
                              doc_id="PMC1")
    eq = [ch for ch in chunks if ch.chunk_type == "equation"]
    assert len(eq) == 1
    assert eq[0].text == "E=mc^2"
    assert "$$" not in eq[0].text


def test_unclosed_equation_fence_does_not_swallow_document():
    md = ("---\npmcid: PMC1\n---\n## M\nintro before\n$$\nnever closed\n"
          "tail after\n")
    chunks, _, _ = Chunker(max_tokens=120).chunk_md(md, doc_id="PMC1")
    alltext = "\n".join(c.text for c in chunks)
    assert "intro before" in alltext
    assert "tail after" in alltext


def test_headingless_document_still_chunks():
    md = "---\npmcid: PMC1\n---\nJust one unheaded paragraph of body prose.\n"
    chunks, _, _ = Chunker(max_tokens=120).chunk_md(md, doc_id="PMC1")
    paras = [c for c in chunks if c.chunk_type == "paragraph"]
    assert paras
    assert "unheaded paragraph" in paras[0].text


def test_h1_direct_blocks_preserved():
    md = ("---\npmcid: PMC1\n---\n# Title\nUnheaded intro paragraph.\n\n"
          "## Section A\nSome body text.\n")
    chunks, _, _ = Chunker(max_tokens=120).chunk_md(md, doc_id="PMC1")
    assert any("Unheaded intro paragraph" in c.text for c in chunks)


def test_reference_chunks_carry_unit_id(chunker):
    chunks, units, _ = chunker.chunk_md(make_md(), doc_id="PMC42")
    refs = [c for c in chunks if c.chunk_type == "reference"]
    assert refs
    assert all(c.metadata["unit_id"] for c in refs)
    unit_ids = {u.unit_id for u in units}
    assert refs[0].metadata["unit_id"] in unit_ids


def test_cli_input_without_markdown_errors(tmp_path, capfd):
    empty = tmp_path / "empty"
    empty.mkdir()
    assert cli_main(["--input", str(empty)]) == 1
    out = capfd.readouterr()
    assert "No Markdown files found" in (out.out + out.err)



# -- table robustness: rowspan continuation, spans, subheaders -----------------

def test_table_continuation_rows_are_chunked_with_carried_label():
    # JATS rowspan flattening leaves continued rows' first cell blank
    # ("| | Felodipine | C | ..." continues "Calcium channelblockers").
    # They must be chunked as data rows with the carried label, never
    # swallowed as group context (PMC10001459 table regression).
    md = ("---\npmcid: PMC9\n---\n## Drugs\n"
          "*Table 1: drug recommendations.*\n\n"
          "| Category | Drug | Class | Recommendation |\n"
          "| --- | --- | --- | --- |\n"
          "| CCB | Amlodipine | C | Benefit vs. risk. |\n"
          "|  | Felodipine | C | Benefit vs. risk. |\n"
          "|  | Nifedipine | C | Present in milk. |\n"
          "| PDE5i | Sildenafil | B | Not indicated. |\n"
          "|  | Tadalafil | B | Not indicated. |\n")
    chunks, _, _ = Chunker(max_tokens=120).chunk_md(md, doc_id="PMC9")
    rows = [c for c in chunks if c.chunk_type == "table_row"]
    assert len(rows) == 5                      # every source row IS a chunk
    assert [r.row_label for r in rows] == ["CCB", "CCB", "CCB", "PDE5i", "PDE5i"]
    assert all(not r.group_path for r in rows)  # no swallowed-row leakage
    sildenafil = [r for r in rows if "Drug: Sildenafil" in r.text][0]
    assert sildenafil.row_label == "PDE5i"
    assert "Tadalafil" not in sildenafil.text  # next row NOT leaked into group


def test_table_spanning_subheader_becomes_group_path():
    md = ("---\npmcid: PMC9\n---\n## Results\n"
          "| Measure | Value |\n| --- | --- |\n"
          "| Net results (mmHg) | Net results (mmHg) |\n"
          "| Change 24h-SBP | -3 |\n"
          "| Change 24h-DBP | -2 |\n")
    chunks, _, _ = Chunker(max_tokens=120).chunk_md(md, doc_id="PMC9")
    rows = [c for c in chunks if c.chunk_type == "table_row"]
    assert len(rows) == 2                      # subheader row is context, not a chunk
    assert all(r.group_path == ["Net results (mmHg)"] for r in rows)


def test_table_single_cell_subheader_becomes_group_path():
    md = ("---\npmcid: PMC9\n---\n## Results\n"
          "| Sex | Value |\n| --- | --- |\n"
          "| Male | 1 |\n"
          "| Median (IQR) |\n"
          "| Female | 2 |\n")
    chunks, _, _ = Chunker(max_tokens=120).chunk_md(md, doc_id="PMC9")
    rows = [c for c in chunks if c.chunk_type == "table_row"]
    assert len(rows) == 2
    female = [r for r in rows if "Sex: Female" in r.text][0]
    assert female.group_path == ["Median (IQR)"]



def test_figure_footnotes_merge_into_figure_chunk():
    # jats_to_md emits the image + bold caption + italic footnote lines after
    # a figure; the footnote must travel WITH the figure (and its link), not
    # leak out as a standalone paragraph chunk.
    md = ("---\npmcid: PMC9\n---\n## Figures\n"
          "![Figure 1](https://example.com/fig1.jpg)\n\n"
          "**Figure 1 Loss of BMPR2 increases proliferation.**\n\n"
          "*BMPR2*, *ARRB2* or both genes were reduced in PASMC by siRNA.\n")
    chunks, _, _ = Chunker(max_tokens=120).chunk_md(md, doc_id="PMC9")
    figs = [c for c in chunks if c.chunk_type == "figure"]
    assert len(figs) == 1
    fig = figs[0]
    assert "Loss of BMPR2" in fig.text            # caption attached
    assert "both genes were reduced" in fig.text  # footnote merged in
    assert fig.figure_link == "https://example.com/fig1.jpg"  # first-class link
    paras = [c for c in chunks if c.chunk_type == "paragraph"]
    assert not any("both genes were reduced" in c.text for c in paras)
