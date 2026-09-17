# src/chunking — Markdown → chunks → ParadeDB

One job: **read a directory of Markdown, chunk it by the document's markdown
structure, and store chunks + units in the `medpat` ParadeDB container.**
No LLM, no embedding (encoding is a separate step, see below).

## How to run it

    make chunks DIR="data/md"                 # backend/Makefile: starts medpat, chunks DIR, stores
    make chunks DIR="data/md" MEDPAT_WORKERS=8  # same, 8 files in parallel
    python -m src.chunking --input data/md --workers 4   # directly

Tuning flags: `--workers` (default 1 = strictly sequential; each worker holds its
own Postgres connection), `--max-tokens` (soft budget per prose chunk, default 320),
`--hard-max-tokens` (hard ceiling, default 2x), `--split-overlap-sentences`
(sentence carry, default 2), `--split-overlap-tokens` (carry budget),
`--overwrite`. Re-runs are idempotent: unchanged files are skipped via the
stored `md_sha256`.

## The pipeline, end to end

    cli.py            --input <dir> -> list of .md files (parallel-able via --workers)
      -> documents.py chunk_document(md)   one article
           parsing.py        markdown -> heading tree (RawBlock/MDNode) + metadata
           sections.py       recurse the tree, one section at a time
             classification.py  "references"/"administrative" vs content
             prose.py          paragraphs: sentence-split with carry, one chunk each
             tables.py         tables: summary + row + footnotes + TABLE unit
             chunks.py         ChunkFactory builds the stored Chunk records
           ids.py            per-document id/position counters (no collisions)
      -> store_pg.py store_file()  upsert documents/units/chunks/refs/citations
                                    into medpat (one transaction per document)

The recursion is the "chunk by md structure" part: each heading becomes a
section unit, its `##`/`###` children hang off it, and every block (prose,
lists, tables, figures, equations, references, admin) is chunked with the
section breadcrumb.

## The one piece of custom logic

`prose.split_long_paragraph_pieces` is the overlap: when a paragraph exceeds
`max_tokens` it is split at sentence boundaries and the last N sentences of the
flushed piece seed the next piece (the carry, token-budgeted to ~25% of
`max_tokens`). Overlap is applied **only** inside a split paragraph — boundaries
between distinct paragraphs are never overlapped.

## Figures

A figure = one `figure` chunk holding: the label and caption, any figure
footnotes (italic lines after the caption, e.g. "*p* < 0.01; ^✭✭✭^" — merged
into the chunk text so they travel with the figure), and the resolved image
URL in a first-class `figure_link` column (also in `metadata.image_ref`), so
retrieval results can render the actual image.

## What is stored (medpat schema)

- `medpat.documents` — one row per md file (title, metadata, body, report).
- `medpat.units` — hierarchy: one section unit per heading, one table unit per
  table, one paragraph unit per split paragraph. Every chunk's `parent_id` points
  at the unit that contains it.
- `medpat.chunks` — one row per chunk (`chunk_type`, breadcrumb, text,
  `embedding_text` = the chunk text, citation refs, position). Figure chunks carry
  `figure_link`; references and administrative sections are stored but
  `retrieval_eligible = false`, so search never returns them.
- `medpat.references` / `medpat.chunk_citations` — normalized reference list +
  which chunks cite which references.

`embedding_text` is plain chunk text; the *actual* encoding is a separate step:
`make medpat-embed` (python -m src.embedding) POSTs `embedding_text` in batches
to the OpenAI-compatible MedCPT embedding server (`EMBEDDING_BASE_URL`),
L2-normalizes the returned 768-dim vectors and upserts them into
`medpat.chunk_embeddings` (idempotent via `embedding_text_hash`). ParadeDB BM25
search also runs over `embedding_text`.

## Table handling (the fiddly part)

Tables are parsed by **markdown-it-py** (the `table` rule, GFM pipe tables) into
`RawBlock.header` + `RawBlock.rows`; `tables.py` maps rows onto chunks: one
`table_summary`, one `table_row` per data row (never dropped), one
`table_footnotes`, and a `table` unit holding the full table text. Awkward
markdown-table cases and how they are resolved:

1. **rowspan continuation.** JATS→markdown flattens a `rowspan` by leaving the
   continued row's first cell blank (`| | Felodipine | C | ...`). A blank first
   cell carries the previous row's first-column value forward and the row is
   still chunked (this fixed PMC10001459 Table 1, where 10 of 16 rows were
   previously swallowed).
2. **spanning subheader.** A colspan header flattened into every cell
   (`| Net results | Net results |`) or a row with a single populated cell
   (`| Median (IQR) |`) is group context, not a data row: it is recorded in the
   following rows' `group_path` and emits no chunk of its own.
3. **repeated header rows** inside multi-panel tables are likewise consumed as
   context, matching the column headings.

## File map (one concern per file)

| File | Concern |
|---|---|
| `cli.py` | folder → loop (parallel via --workers) → store (the only entry point) |
| `documents.py` | one article → (chunks, units, report); composes the rest |
| `parsing.py` | markdown → heading tree + blocks + front matter + figure/table context |
| `sections.py` | one heading node → its chunks + section unit |
| `classification.py` | section title → content / references / administrative |
| `prose.py` | paragraph-first chunking + the split-overlap carry |
| `tables.py` | tables → summary/row/footnote chunks + table unit + heuristics |
| `chunks.py` | ChunkFactory: build the Chunk record for every chunk type |
| `ids.py` | per-document unique ids, positions, counter state |
| `tokens.py` | chars/4 token estimate + abbreviation-aware sentence splitter |
| `models.py` | `UnitRecord` (Chunk lives in src/lib/models.py) |
| `store_pg.py` | medpat Postgres upserts (documents/units/chunks/refs/citations) |

## Tests

    pytest tests/test_chunking.py -q

Synthetic markdown, no DB/network: parser, units, references, split-overlap
carry, tables (incl. rowspan/span/subheader robustness), figures (caption +
footnotes + figure_link), determinism, and the CLI's no-input error.
