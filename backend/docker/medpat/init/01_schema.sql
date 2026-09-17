-- medpat schema v1 - the corpus storage (documents, units, chunks, embeddings, references, citations, lexicon, load marks).
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pgcrypto;   -- sha256 digest() for embedding change detection
CREATE SCHEMA IF NOT EXISTS medpat;
GRANT ALL ON SCHEMA medpat TO medpat;
SET search_path TO medpat, public;
CREATE TABLE medpat.meta (key TEXT PRIMARY KEY, value JSONB NOT NULL, updated_at TIMESTAMPTZ NOT NULL DEFAULT now());
INSERT INTO medpat.meta (key, value) VALUES
  ('schema_version', to_jsonb('1'::text)),
  ('embedder_model', to_jsonb('ncbi/MedCPT-Article-Encoder'::text)),
  ('fts_config',     to_jsonb('simple'::text))
ON CONFLICT (key) DO NOTHING;
CREATE TABLE medpat.documents (
    id                TEXT PRIMARY KEY,
    pmcid             TEXT UNIQUE,
    title             TEXT NOT NULL DEFAULT '',
    journal           TEXT NOT NULL DEFAULT '',
    doi               TEXT NOT NULL DEFAULT '',
    pmid              TEXT NOT NULL DEFAULT '',
    publication_date  TEXT NOT NULL DEFAULT '',
    authors           JSONB NOT NULL DEFAULT '[]',
    keywords          JSONB NOT NULL DEFAULT '[]',
    categories        JSONB NOT NULL DEFAULT '[]',
    funding_sources   JSONB NOT NULL DEFAULT '[]',
    volume            TEXT NOT NULL DEFAULT '',
    issue             TEXT NOT NULL DEFAULT '',
    pages             TEXT NOT NULL DEFAULT '',
    source_format     TEXT NOT NULL DEFAULT 'md',
    front_matter      JSONB NOT NULL DEFAULT '{}',
    body_md           TEXT NOT NULL,
    md_sha256         TEXT NOT NULL DEFAULT '',
    raw_xml           TEXT,
    chunker_report    JSONB,
    status            TEXT NOT NULL DEFAULT 'pending' CHECK (status IN
                        ('pending','chunked','failed','stale')),
    last_error        TEXT,
    chunk_count       INT NOT NULL DEFAULT 0,
    unit_count        INT NOT NULL DEFAULT 0,
    eligible_count    INT NOT NULL DEFAULT 0,
    created_at        TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at        TIMESTAMPTZ NOT NULL DEFAULT now()
);
COMMENT ON COLUMN medpat.documents.body_md IS
  'Full Markdown source; the storage home of the corpus (no disk files).';
COMMENT ON COLUMN medpat.documents.md_sha256 IS
  'Fingerprint for incremental-skip re-ingestion (same role as the old sidecar).';
CREATE INDEX documents_status_idx  ON medpat.documents (status);
CREATE INDEX documents_journal_idx ON medpat.documents (journal);

CREATE TABLE medpat.units (
    unit_id         TEXT PRIMARY KEY,
    document_id     TEXT NOT NULL REFERENCES medpat.documents(id) ON DELETE CASCADE,
    kind            TEXT NOT NULL CHECK (kind IN ('section','table','paragraph')),
    title           TEXT NOT NULL DEFAULT '',
    breadcrumb      JSONB NOT NULL DEFAULT '[]',
    text            TEXT NOT NULL,
    parent_unit_id  TEXT REFERENCES medpat.units(unit_id) ON DELETE CASCADE,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX units_document_idx ON medpat.units (document_id);
CREATE INDEX units_kind_idx     ON medpat.units (kind);
CREATE INDEX units_parent_idx   ON medpat.units (parent_unit_id);

CREATE TABLE medpat.chunks (
    id                    TEXT PRIMARY KEY,
    document_id           TEXT NOT NULL REFERENCES medpat.documents(id) ON DELETE CASCADE,
    chunk_type            TEXT NOT NULL CHECK (chunk_type IN (
                            'paragraph','list','table_summary','table_row',
                            'table_footnotes','figure','equation','reference',
                            'administrative')),
    section               TEXT,
    subsection            TEXT,
    breadcrumb            JSONB NOT NULL DEFAULT '[]',
    parent_id             TEXT REFERENCES medpat.units(unit_id) ON DELETE CASCADE,
    object_id             TEXT,
    source_block_ids      JSONB NOT NULL DEFAULT '[]',
    table_id              TEXT,
    figure_id             TEXT,
    figure_link           TEXT,
    equation_id           TEXT,
    reference_id          TEXT,
    row_label             TEXT,
    group_path            JSONB NOT NULL DEFAULT '[]',
    text                  TEXT NOT NULL,
    embedding_text        TEXT NOT NULL,
    metadata              JSONB NOT NULL DEFAULT '{}',
    citation_refs         JSONB NOT NULL DEFAULT '[]',
    footnote_refs         JSONB NOT NULL DEFAULT '[]',
    concept_ids           JSONB NOT NULL DEFAULT '[]',
    embedding_token_count INT NOT NULL DEFAULT 0,
    document_position     INT NOT NULL DEFAULT 0,
    retrieval_eligible    BOOLEAN NOT NULL DEFAULT true,
    dedup_of              TEXT,
    fingerprint           TEXT,
    chunk_version         TEXT NOT NULL DEFAULT '1.0',
    tsv                   TSVECTOR,
    created_at            TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX chunks_document_idx     ON medpat.chunks (document_id, document_position);
CREATE INDEX chunks_type_idx         ON medpat.chunks (chunk_type);
CREATE INDEX chunks_parent_idx       ON medpat.chunks (parent_id);
CREATE INDEX chunks_table_idx        ON medpat.chunks (table_id);
CREATE INDEX chunks_eligible_idx     ON medpat.chunks (document_position)
    WHERE retrieval_eligible;
CREATE INDEX chunks_breadcrumb_gin   ON medpat.chunks USING gin (breadcrumb);
CREATE INDEX chunks_metadata_gin     ON medpat.chunks USING gin (metadata);
CREATE INDEX chunks_tsv_gin          ON medpat.chunks USING gin (tsv);
CREATE INDEX chunks_fingerprint_hash ON medpat.chunks USING hash (fingerprint)
    WHERE fingerprint IS NOT NULL;
CREATE INDEX chunks_table_row_order  ON medpat.chunks
    (((metadata->>'row_index')::int)) WHERE chunk_type = 'table_row';
COMMENT ON COLUMN medpat.chunks.tsv IS
  'Sparse leg (Postgres FTS). Deliberately to_tsvector("simple", embedding_text):
   simple avoids the English stemmer mangling biomedical terms (gene/drug
   names, abbreviations), and indexing embedding_text (not text) makes object
   labels like "Table: Table 2 ..." match. If stopword noise shows up in the
   query distribution, add a custom dictionary - do NOT switch to "english".';
COMMENT ON COLUMN medpat.chunks.fingerprint IS
  'Exact-duplicate key for paragraph/list chunks. The global dedup pass is a
   deterministic SQL window over this column.';

CREATE TABLE medpat.chunk_embeddings (
    chunk_id            TEXT PRIMARY KEY REFERENCES medpat.chunks(id) ON DELETE CASCADE,
    embedding           vector(768) NOT NULL,
    model               TEXT NOT NULL,
    embedding_text_hash TEXT NOT NULL,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at          TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX embeddings_vector_idx ON medpat.chunk_embeddings
    USING hnsw (embedding vector_ip_ops) WITH (m = 16, ef_construction = 128);
CREATE INDEX embeddings_hash_idx ON medpat.chunk_embeddings (embedding_text_hash);
COMMENT ON COLUMN medpat.chunk_embeddings.embedding_text_hash IS
  'sha256 of exactly what was embedded; the encoder skips rows whose hash is
   unchanged, so re-chunking never silently leaves stale vectors.';

CREATE TABLE medpat.references (
    document_id   TEXT NOT NULL REFERENCES medpat.documents(id) ON DELETE CASCADE,
    reference_id  TEXT NOT NULL,
    position      INT  NOT NULL,
    text          TEXT NOT NULL,
    doi           TEXT NOT NULL DEFAULT '',
    pmid          TEXT NOT NULL DEFAULT '',
    pmcid         TEXT NOT NULL DEFAULT '',
    chunk_id      TEXT REFERENCES medpat.chunks(id) ON DELETE SET NULL,
    PRIMARY KEY (document_id, reference_id),
    UNIQUE (document_id, position)
);
CREATE INDEX references_chunk_idx ON medpat.references (chunk_id);

CREATE TABLE medpat.chunk_citations (
    chunk_id      TEXT NOT NULL REFERENCES medpat.chunks(id) ON DELETE CASCADE,
    document_id   TEXT NOT NULL,
    reference_id  TEXT NOT NULL,
    PRIMARY KEY (chunk_id, reference_id),
    FOREIGN KEY (document_id, reference_id)
      REFERENCES medpat.references(document_id, reference_id)
);

CREATE TABLE medpat.lexicon_terms (
    concept_id  TEXT NOT NULL,
    term        TEXT NOT NULL,
    PRIMARY KEY (concept_id, term)
);

CREATE TABLE medpat.load_marks (
    source_key      TEXT PRIMARY KEY,
    status          TEXT NOT NULL DEFAULT 'pending',
    chunker_config  JSONB,
    chunks          INT NOT NULL DEFAULT 0,
    units           INT NOT NULL DEFAULT 0,
    embeddings      INT NOT NULL DEFAULT 0,
    started_at      TIMESTAMPTZ,
    finished_at     TIMESTAMPTZ,
    error           TEXT
);

CREATE FUNCTION medpat.set_tsv() RETURNS trigger LANGUAGE plpgsql AS $$
BEGIN
    NEW.tsv := to_tsvector('simple', coalesce(NEW.embedding_text, ''));
    RETURN NEW;
END $$;

CREATE TRIGGER chunks_tsv
    BEFORE INSERT OR UPDATE OF embedding_text ON medpat.chunks
    FOR EACH ROW EXECUTE FUNCTION medpat.set_tsv();

CREATE VIEW medpat.v_corpus_stats AS
SELECT d.id, d.pmcid, d.title, d.journal, d.status,
       d.chunk_count, d.unit_count, d.eligible_count,
       c.by_type, e.embedded, u.by_kind
FROM medpat.documents d
LEFT JOIN (SELECT document_id, jsonb_object_agg(chunk_type, n) AS by_type
           FROM (SELECT document_id, chunk_type, count(*) n
                 FROM medpat.chunks GROUP BY 1, 2) s GROUP BY 1) c
  ON c.document_id = d.id
LEFT JOIN (SELECT ch.document_id, count(*) embedded
           FROM medpat.chunk_embeddings e
           JOIN medpat.chunks ch ON ch.id = e.chunk_id
           GROUP BY 1) e ON e.document_id = d.id
LEFT JOIN (SELECT document_id, jsonb_object_agg(kind, n) AS by_kind
           FROM (SELECT document_id, kind, count(*) n
                 FROM medpat.units GROUP BY 1, 2) s2 GROUP BY 1) u
  ON u.document_id = d.id;
