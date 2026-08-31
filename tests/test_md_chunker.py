"""Unit tests for chunker v2 (src/md_chunker.py) — Markdown-native chunking.

No LLM, no corpus, no network: synthetic Markdown documents exercise the
parser, parent/child units, entity tagging, dedup, and determinism.
"""

from __future__ import annotations

import json

import pytest

from src.chunking.md_chunker import MDChunker, LexiconTagger
from src.chunking.md_chunker import _parse_front_matter, _document_metadata


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
    return MDChunker(max_tokens=120, overlap=0.1)


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


def test_embedding_text_prefix_and_breadcrumb(chunker):
    chunks, _, _ = chunker.chunk_md(make_md(), doc_id="PMC42")
    abstract = [c for c in chunks if c.chunk_type == "paragraph"
                and c.section == "Abstract"][0]
    # current v2 prefix: title-first (MedCPT ArticleEncoder ordering) +
    # breadcrumb; journal/date are metadata-only, not in the vector text
    assert "A Cardiac Arrest Study" in abstract.embedding_text
    assert "Abstract" in abstract.embedding_text
    assert abstract.text.startswith("Cardiac arrest is the sudden cessation")


def test_table_summary_rows_footnotes_and_units(chunker):
    chunks, units, _ = chunker.chunk_md(make_md(), doc_id="PMC42")
    summary = [c for c in chunks if c.chunk_type == "table_summary"][0]
    rows = [c for c in chunks if c.chunk_type == "table_row"]
    fns = [c for c in chunks if c.chunk_type == "table_footnotes"]
    assert len(rows) == 2
    assert len(fns) == 1
    assert "outcomes by group" in summary.text
    assert all(r.parent_id == summary.id for r in rows)
    assert fns[0].parent_id == summary.id
    assert rows[0].row_label == "A"
    # current v2 row text carries the row label + one line per column value
    assert "A" in rows[0].text and "Survival: 80%" in rows[0].text
    table_units = [u for u in units if u.kind == "table"]
    assert len(table_units) == 1
    tu = table_units[0]
    assert summary.id in tu.chunk_ids
    assert all(r.id in tu.chunk_ids for r in rows)


def test_figure_caption_and_image_ref(chunker):
    chunks, _, _ = chunker.chunk_md(make_md(), doc_id="PMC42")
    fig = [c for c in chunks if c.chunk_type == "figure"][0]
    assert "Kaplan-Meier curve" in fig.text
    assert fig.figure_id
    assert fig.metadata.get("image_ref") == "https://example.com/fig1.jpg"


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
    assert refs[0].embedding_text == ""


def test_prose_overlap_across_windows():
    c = MDChunker(max_tokens=60, overlap=0.25)
    text = "Sentence number one has some words to fill the token budget. " * 40
    md = "---\npmcid: PMC1\n---\n## Methods\n" + text
    chunks, _, _ = c.chunk_md(md, doc_id="PMC1")
    paras = [ch for ch in chunks if ch.chunk_type == "paragraph"]
    assert len(paras) >= 2
    # consecutive windows share a tail (overlap)
    shared = None
    for a, b in zip(paras, paras[1:]):
        tail_words = a.text.split()[-20:]
        if any(w in b.text for w in tail_words):
            shared = True
            break
    assert shared, "expected overlap between consecutive prose windows"


def test_entity_tagging_no_llm():
    tagger = LexiconTagger({"C0000731": ["cardiac arrest"],
                            "C0020538": ["hypertension"]})
    assert tagger.tag("patients with cardiac arrest") == ["C0000731"]
    assert tagger.tag("no concepts here") == []
    c = MDChunker(max_tokens=120, tagger=tagger)
    chunks, _, report = c.chunk_md(make_md(), doc_id="PMC42")
    tagged = [ch for ch in chunks if ch.concept_ids]
    assert tagged
    assert report["entities_tagged_chunks"] == len(tagged)


def test_exact_dup_within_document_suppressed():
    md = ("---\npmcid: PMC1\n---\n## A\nConsent was obtained from all patients.\n"
          "\n## B\nConsent was obtained from all patients.\n")
    c = MDChunker(max_tokens=200)
    chunks, _, report = c.chunk_md(md, doc_id="PMC1")
    paras = [ch for ch in chunks if ch.chunk_type == "paragraph"]
    assert len(paras) == 2
    dup = [ch for ch in paras if not ch.retrieval_eligible]
    assert len(dup) == 1
    assert dup[0].metadata.get("dedup_of") == paras[0].id
    assert report["dedup_suppressed"] == 1


def test_deterministic_output():
    a = MDChunker(max_tokens=120)
    b = MDChunker(max_tokens=120)
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


# -- robustness regressions (from the v2 critique review) --------------------

def test_single_line_equation_strips_trailing_dollars():
    c = MDChunker(max_tokens=120)
    chunks, _, _ = c.chunk_md("---\npmcid: PMC1\n---\n## M\n$$E=mc^2$$\n",
                              doc_id="PMC1")
    eq = [ch for ch in chunks if ch.chunk_type == "equation"]
    assert len(eq) == 1
    assert eq[0].text == "E=mc^2"
    assert "$$" not in eq[0].text


def test_unclosed_equation_fence_does_not_swallow_document():
    md = ("---\npmcid: PMC1\n---\n## M\nintro before\n$$\nnever closed\n"
          "tail after\n")
    chunks, _, _ = MDChunker(max_tokens=120).chunk_md(md, doc_id="PMC1")
    alltext = "\n".join(c.text for c in chunks)
    assert "intro before" in alltext
    assert "tail after" in alltext


def test_headingless_document_still_chunks():
    md = "---\npmcid: PMC1\n---\nJust one unheaded paragraph of body prose.\n"
    chunks, _, _ = MDChunker(max_tokens=120).chunk_md(md, doc_id="PMC1")
    paras = [c for c in chunks if c.chunk_type == "paragraph"]
    assert paras
    assert "unheaded paragraph" in paras[0].text


def test_h1_direct_blocks_preserved():
    md = ("---\npmcid: PMC1\n---\n# Title\nUnheaded intro paragraph.\n\n"
          "## Section A\nSome body text.\n")
    chunks, _, _ = MDChunker(max_tokens=120).chunk_md(md, doc_id="PMC1")
    assert any("Unheaded intro paragraph" in c.text for c in chunks)


def test_reference_chunks_carry_unit_id(chunker):
    chunks, units, _ = chunker.chunk_md(make_md(), doc_id="PMC42")
    refs = [c for c in chunks if c.chunk_type == "reference"]
    assert refs
    assert all(c.metadata["unit_id"] for c in refs)
    unit_ids = {u.unit_id for u in units}
    assert refs[0].metadata["unit_id"] in unit_ids


def test_global_dedup_deterministic_p1_wins(capfd, tmp_path):
    import pandas as pd
    from src.chunking.md_chunker import main as md_main

    md_in = tmp_path / "md"
    md_in.mkdir()
    body = "Common boilerplate sentence that repeats verbatim.\n"
    (md_in / "P1.md").write_text(f"---\npmcid: P1\n---\n## A\n{body}",
                                 encoding="utf-8")
    (md_in / "P2.md").write_text(f"---\npmcid: P2\n---\n## A\n{body}",
                                 encoding="utf-8")
    out = tmp_path / "chunks"
    assert md_main(["--input", str(md_in), "--chunks-out", str(out),
                    "--units-out", str(tmp_path / "units"),
                    "--global-dedup", "--workers", "2"]) == 0
    d1 = pd.read_parquet(out / "P1.parquet")
    d2 = pd.read_parquet(out / "P2.parquet")
    e1 = int(d1[d1.chunk_type == "paragraph"].retrieval_eligible.sum())
    e2 = int(d2[d2.chunk_type == "paragraph"].retrieval_eligible.sum())
    # deterministic: the earlier file (P1, sorted order) is kept
    assert (e1, e2) == (1, 0)
    dup_row = d2[d2.chunk_type == "paragraph"].iloc[0]
    assert not bool(dup_row.retrieval_eligible)   # np.False_ vs False
    assert json.loads(dup_row.metadata)["dedup_of"].startswith("P1_")


def test_cli_runner_skips_unchanged(capfd, tmp_path):
    from src.chunking.md_chunker import main as md_main
    md_in = tmp_path / "md"
    md_in.mkdir()
    (md_in / "PMC42.md").write_text(make_md(), encoding="utf-8")
    out = tmp_path / "chunks"
    units = tmp_path / "units"
    args = ["--input", str(md_in), "--chunks-out", str(out),
            "--units-out", str(units), "--workers", "2"]
    assert md_main(args) == 0
    assert (out / "PMC42.parquet").exists()
    first = out / "PMC42.parquet"
    second = md_main(args)   # unchanged -> skipped
    full = capfd.readouterr().out + capfd.readouterr().err
    assert second == 0
    assert "skipped=1" in full
    assert first.read_bytes() == (out / "PMC42.parquet").read_bytes()


def test_runner_writes_corpus_compatible_columns(capfd, tmp_path):
    import pandas as pd
    from src.chunking.md_chunker import main as md_main
    md_in = tmp_path / "md"
    md_in.mkdir()
    (md_in / "PMC42.md").write_text(make_md(), encoding="utf-8")
    out = tmp_path / "chunks"
    md_main(["--input", str(md_in), "--chunks-out", str(out),
             "--units-out", str(tmp_path / "units")])
    df = pd.read_parquet(out / "PMC42.parquet")
    required = {"id", "document_id", "text", "embedding_text", "chunk_type",
                "section", "subsection", "breadcrumb", "parent_id",
                "table_id", "figure_id", "document_position",
                "retrieval_eligible"}
    assert required <= set(df.columns)
    # pyarrow returns numpy arrays for list columns; a list (or array-like)
    # is what the corpus schema expects
    br = df.iloc[0]["breadcrumb"]
    assert hasattr(br, "tolist") or isinstance(br, list)