# MedRAG Retrieval V2 - paper-first, intent-aware architecture (V2.1)

The V2 pipeline (package `src/medrag/retrieval_v2/`) implements the
paper-first retrieval architecture with PageIndex paper-local navigation and
table-aware evidence context:

    question → structured query understanding (MedGemma / deterministic)
        → clinical_entities / query_targets / requested_fields / populations
        → requirements H1..Hn (ONE per genuine evidence obligation)
        → retrieval variants V0..Vn per requirement
        → per-query hybrid retrieval (BM25 + pgvector) with full provenance
        → query-local RRF (never global chunk fusion)
        → paper-level aggregation + per-branch shortlists (union)
        → cross-query paper ranking with branch guarantees
        → top papers → PageIndex paper-local navigation (precomputed trees)
        → logical document structure index → paper-local hybrid search
        → bounded parent/child expansion (tables, figures, footnotes)
        → table-aware evidence context (header + row + footnotes)
        → MedCPT cross-encoder on requirement ↔ contextual evidence
        → intent-aware contextual scoring with explicit penalties
        → greedy requirement-coverage selection + focused repair pass
        → honest coverage report + full trace (PAGEINDEX_* events)

## The /expand contract (V2.1)

POST {server}/expand now returns a structured query-intelligence object:

    {"query", "clinical_entities", "query_targets", "requested_fields",
     "relationships", "populations", "retrieval_variants",
     "reranker_intent", "umls"}

It does NOT return `evidence_requirements` and the planner NEVER slices a
question into H1..H4 statistical branches. One question = one requirement per
genuine evidence obligation (a multi-hop question = H1 (mechanism) + H2
(population difference) with hop dependencies).

## PageIndex integration

PageIndex is the PAPER-LOCAL structural navigation layer. Global BM25+pgvector
discovers PAPERS; PageIndex navigates WITHIN a paper over a precomputed
hierarchical tree (built offline by the official VectifyAI/PageIndex library
`pageindex.page_index_md.md_to_tree` from the existing XML-derived corpus
structure). It is never run over the global corpus and never used as a vector
replacement. Nodes carry existing chunk ids (one evidence identity per chunk).

Offline artifact build (do once per paper):

    python -m medrag.retrieval_v2.pageindex_build --paper PMC11743609
    python -m medrag.retrieval_v2.pageindex_build --papers PMC11743609 PMC12009809
    python -m medrag.retrieval_v2.pageindex_build --limit 2000        # incremental

Artifacts live in `index/pageindex/{paper_id}.json` (tree + node→chunk map).
PageIndex is OPTIONAL: missing artifacts fall back to BM25+pgvector+
LogicalDocumentIndex and the trace records pageindex_status =
success | unavailable | failed.

## Key invariants

- retrieval happens per requirement branch / variant; branches never compete
  at the chunk level before paper-level evidence is preserved
- a paper essential to ONE branch survives selection via the per-branch
  shortlist union (branch guarantee)
- PageIndex runs only after paper discovery, over precomputed per-paper trees
- table rows are scored WITH their assembled table context (summary + headers
  + row + footnotes + breadcrumb) and requested fields (percentage, p-value,
  CI, OR/HR) are detected from table headers (deterministic)
- medical terminology is preserved (INOCA is NEVER rewritten to MINOCA; the
  terminology guard preserves every surface form and acronym generically)
- the expensive cross-encoder runs only on 20-100 strong candidates
- coverage is honest: an evidence obligation is covered only when the
  contextual evidence contains the concept AND satisfies the requested
  fields/outcome - no generic substitution for missing evidence

## Usage

    # Surgical-technique numerical question (acceptance test 1)
    python -m medrag.retrieval_v2 "Which surgical repair techniques were \
        associated with recurrent coarctation and what are the percentages \
        and p-values?" --output retrieval_runs/surgical

    # Known multi-hop regression benchmark (spec section 41)
    python scripts/benchmark_v2.py [--no-rerank] [--out eval/v2.json]

    # Run any question through the V2 pipeline
    python -m medrag.retrieval_v2 "How does anemia affect cardiovascular \
        vulnerability in elderly hip-fracture patients?" --output retrieval_runs/run1

    # With an existing V2 plan (JSON)
    python -m medrag.retrieval_v2 --plan plan.json --output retrieval_runs/run2

## Outputs per run

- `v2_result.json` - full structured result (papers, evidence, metrics, pageindex)
- `v2_evidence.md` - readable evidence summary (context-aware)
- `v2_trace.log`   - step-by-step execution trace incl. PAGEINDEX_* events

## Tests

    python -m pytest tests/test_v2_*.py -q
