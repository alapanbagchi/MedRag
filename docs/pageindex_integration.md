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
