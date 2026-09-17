-- ParadoxDB pg_search 0.25: real BM25 over medpat.chunks.
-- Sparse leg for hybrid retrieval (replaces the on-disk Rank-BM25 files).
-- Only ONE ParadeDB index per table is allowed; id = key_field (unique).

CREATE EXTENSION IF NOT EXISTS pg_search;

-- retrieval_eligible must be part of the index: otherwise the planner applies
-- it as a heap_filter (one heap fetch per matching doc) instead of a term
-- filter, which is ~80x slower on broad queries.
CREATE INDEX IF NOT EXISTS chunks_bm25_idx ON medpat.chunks
    USING paradedb (id, retrieval_eligible, (embedding_text::pdb.whitespace))
    WITH (key_field = 'id');

COMMENT ON INDEX medpat.chunks_bm25_idx IS
  'ParadeDB BM25 over embedding_text (whitespace tokenizer - keeps gene/
   drug tokens like EGFR intact). Query via "embedding_text ||| q" and
   pdb.score(id). Same column the FTS tsv leg indexes; both may coexist.';

