# MedRAG Retrieval V2 PageIndex

Offline builder and paper-local navigation (V2.1 parts 9-14, 24-28, 36-37).

## What it is

PageIndex is the PAPER-LOCAL structural navigation layer. Global
BM25 + pgvector discover PAPERS; PageIndex navigates WITHIN a paper over a
precomputed hierarchical tree that mirrors the existing XML-derived logical
structure (Abstract / Introduction / Methods / Results / Tables / Figures /
Discussion). It is:

- built OFFLINE from the same corpus the LogicalDocumentIndex reads
- never run over the global 883k-chunk corpus
- never used as a vector replacement
- OPTIONAL (missing artifacts fall back to BM25 + pgvector + doc index;
  trace records pageindex_status = success | unavailable | failed)

## Building artifacts (do once per paper)

    python -m medrag.retrieval_v2.pageindex_build --paper PMC11743609
    python -m medrag.retrieval_v2.pageindex_build --papers PMC11743609 PMC12009809 ...
    python -m medrag.retrieval_v2.pageindex_build --limit 2000      # incremental
    python -m medrag.retrieval_v2.pageindex_build --all --rebuild   # full rebuild

Artifacts: index/pageindex/{paper_id}.json  (tree + node_map + chunk_to_node)

The Markdown input to the official pageindex.page_index_md.md_to_tree is
generated from the XML-derived corpus structure (section/subsection hierarchy,
tables as single nodes carrying summary + rows + footnotes, figures), so the
PageIndex tree and the LogicalDocumentIndex share the SAME chunk identifiers:
PageIndex decides WHERE to look, the doc index retrieves the exact XML
evidence object.

## Navigation

    from medrag.retrieval_v2.pageindex_adapter import PageIndexAdapter
    adapter = PageIndexAdapter(doc_index, pageindex_dir=Path("index/pageindex"), config=cfg)
    hits = adapter.navigate("PMC11743609", navigation_objective, top_k=8)
    resolved = adapter.resolve_node("PMC11743609", hits[0].pageindex_node_id)
    # resolved["chunk_ids"] -> corpus chunk ids -> LogicalDocumentIndex

Trace events: PAGEINDEX_LOAD, PAGEINDEX_SEARCH, PAGEINDEX_NODE_SELECTED,
PAGEINDEX_NODE_RESOLVED.

## Example objective

Question: "Which surgical repair techniques were associated with recurrent
coarctation and what are the percentages and p-values?"

Navigation objective produced by the planner:

    Find the Results sections, tables, and table footnotes that report
    surgical repair techniques associated with recurrent coarctation
    including percentage, p-value.

Top PageIndex hits for PMC11743609:

    Table T2   (Results > Recurrent coarctation (re-CoA) > Table T2)   rel=1.00
    Table T1   (Results > Table T1)                                    rel=0.86
    "Data are shown in Table 2..." (Results > Pre-discharge BP...)      rel=0.86
    "Re-CoA occurred in 4 patients..." (Results > Recurrent re-CoA)     rel=0.55
    "All patients with re-CoA had repair via lateral thoracotomy..."
        (Results > Surgical technique and re-CoA)                       rel=0.55

PAGEINDEX_NODE_RESOLVED for Table T2 maps onto chunk ids:

    PMC11743609_T2_summary, PMC11743609_T2_row_0..21, PMC11743609_T2_footnotes

## Fallback

If the pageindex package is not installed, artifacts are absent, or a search
fails, the local retriever continues with BM25 + pgvector + the logical
document index. pageindex_status records what happened per paper.

## JATS XML -> Markdown -> PageIndex (V2.3)

The PageIndex tree can now be generated from the ORIGINAL ARTICLE XML using the
installed jats-to-markdown parser, instead of reconstructing hierarchy from
chunks:

    ORIGINAL XML  ->  jats (parse_jats_xml + convert_to_markdown)
                  ->  Markdown folder  (index/pageindex_md/{paper_id}.md)
                  ->  official pageindex.page_index_md.md_to_tree
                  ->  index/pageindex/{paper_id}.json

Convert the XML corpus to Markdown (one command):

    python -m medrag.retrieval_v2.jats_convert --all
    python -m medrag.retrieval_v2.jats_convert --papers PMC11743609 PMC11092466
    python -m medrag.retrieval_v2.jats_convert --paper PMC11743609 --rebuild

Markdown is saved to `index/pageindex_md`. The converter:

- keeps real XML section/subsection titles as headings (never paragraph
  first-sentences),
- keeps tables and figures at their source XML location,
- re-adds floats-group tables/figures (JATS end-of-document floats) without
  inventing a "Figures and Tables" branch: figures are placed at their first
  in-text citation, tables stay under the XML `Floats-group` element in
  document order,
- normalizes bold table/figure labels into `#### Table N / Figure N` headings
  so PageIndex's md_to_tree nests them correctly.

Then build the PageIndex artifacts from the Markdown folder:

    python -m medrag.retrieval_v2.pageindex_build --from-md --papers PMC11743609 PMC11092466

Known limit of MD mode: tables are rendered as HTML blocks, so row-level chunk
mapping (PMC..._T2_row_*) is not re-attached in this mode; paragraphs that
match exactly are mapped (chunk_to_node). The XML-tree builder
(medrag.retrieval_v2.xml_tree) retains full row/chunk mapping when needed.

### Supplementary-material (source data files)

JATS `<supplementary-material>` elements (source data for figures, raw blots,
supplementary datasets, movies) are no longer dropped from the Markdown. The
converter renders each as a `#### <label>` node carrying its caption/title text
and a `Media: <href>` line, inserted at the element's source section
(position = deepest ancestor `<sec>` title; otherwise appended under a top-level
"Supplementary data" section). Conversion is idempotent (re-running does not
duplicate entries). The XML-tree builder records the same label/caption/media
metadata on its supplementary-material section node.
