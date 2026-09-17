"""PostgreSQL + pgvector storage backend for MedRAG embeddings.

Replaces the FAISS flat index with a persistent, queryable pgvector store.

Schema (medpat ParadeDB by default; select the legacy db with
``MEDPAT_PG_SCHEMA=medrag``):
    medpat.chunks            — chunk metadata (id, document_id, chunk_type, text, ...)
    medpat.chunk_embeddings  — vectors (chunk_id, embedding vector(768)), HNSW
    medrag.chunks / medrag.embeddings — legacy layout (IVFFlat)

The embeddings table uses pgvector's <#> (negative inner product) operator
for cosine similarity search (vectors are stored L2-normalized).

Environment variables:
    MEDPAT_PG_SCHEMA  target schema (default: medpat; medrag = legacy)
    MEDPAT_DSN        medpat libpq DSN (default: the local ParadeDB container)
    PGHOST / PGPORT / PGUSER / PGDATABASE / PGPASSWORD
                      legacy medrag connection (used when schema != medpat)

Usage:
    from src.retrieval.pgvector_store import PgVectorStore

    store = PgVectorStore.from_env()
    store upsert_embeddings(chunk_ids, embeddings)
    results = store.search(query_vector, top_k=10)
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from src.lib.trace import get_trace

EMBEDDING_DIM = 768

# Default DSN of the medpat ParadeDB container (docker/medpat). Matches
# src.retrieval.pg_search and the chunking/embedding writers, so every leg of
# the hybrid stack reads the same corpus.
DEFAULT_MEDPAT_DSN = "postgresql://medpat:medpat@localhost:5433/medpat"


def current_schema() -> str:
    """Corpus schema, resolved at call time rather than import time.

    ``AppConfig`` loads ``backend/.env`` *after* modules are imported, so a
    module-level constant could freeze the wrong value. Default is the medpat
    ParadeDB corpus; ``MEDPAT_PG_SCHEMA=medrag`` selects the legacy database.
    """
    return (os.environ.get("MEDPAT_PG_SCHEMA") or "medpat").strip() or "medpat"


def embeddings_table(schema: Optional[str] = None) -> str:
    """Vectors table for ``schema``.

    medpat stores vectors in ``chunk_embeddings`` (written by src.embedding,
    HNSW/vector_ip_ops); the legacy medrag schema uses ``embeddings``.
    """
    schema = schema or current_schema()
    override = (os.environ.get("MEDPAT_PG_EMBEDDINGS_TABLE") or "").strip()
    if override:
        return override
    return "chunk_embeddings" if schema == "medpat" else "embeddings"


# Legacy constant kept for callers that only need a schema name at import time.
# New code should call current_schema() so a late-loaded .env is honored.
SCHEMA = current_schema()


def _to_np(v: Any) -> np.ndarray:
    """Convert a fetched pgvector value (Vector / list / ndarray) to float32."""
    if hasattr(v, "to_list"):
        v = v.to_list()
    return np.asarray(v, dtype=np.float32)


@dataclass
class PgConfig:
    """PostgreSQL connection configuration.

    Defaults point to the Docker container started by scripts/setup_pgvector.sh.
    """
    host: str = "localhost"
    port: int = 5432
    user: str = "postgres"
    password: str = "medrag"
    database: str = "medrag"
    # Full libpq DSN; when set it wins over the host/port/... fields.
    dsn_url: str = ""

    @classmethod
    def from_env(cls) -> "PgConfig":
        # The medpat corpus lives in its own container (MEDPAT_DSN). When the
        # target schema is medpat, use that DSN so the dense leg reads the same
        # rows as the ParadeDB BM25 leg. Explicitly selecting a non-medpat
        # schema keeps the legacy PG* variables in charge.
        if current_schema() == "medpat":
            dsn_url = (os.environ.get("MEDPAT_DSN") or DEFAULT_MEDPAT_DSN).strip()
            if dsn_url:
                return cls(dsn_url=dsn_url)
        return cls(
            host=os.environ.get("PGHOST", "localhost"),
            port=int(os.environ.get("PGPORT", "5432")),
            user=os.environ.get("PGUSER", "postgres"),
            password=os.environ.get("PGPASSWORD", "medrag"),
            database=os.environ.get("PGDATABASE", "medrag"),
        )

    def dsn(self) -> str:
        if self.dsn_url:
            return self.dsn_url
        parts = [f"host={self.host}", f"port={self.port}", f"dbname={self.database}"]
        if self.user:
            parts.append(f"user={self.user}")
        if self.password:
            parts.append(f"password={self.password}")
        return " ".join(parts)


class PgVectorStore:
    """PostgreSQL + pgvector backend for chunk embeddings and metadata."""

    def __init__(self, config: Optional[PgConfig] = None) -> None:
        self.config = config or PgConfig.from_env()
        # Resolved per store (not per import) so a late-loaded .env wins.
        self.schema = current_schema()
        self.emb_table = embeddings_table(self.schema)
        self._conn = None
        # Connection whose pgvector typecasters are already registered.
        self._vector_registered = None

    # ==================================================================
    # Connection management
    # ==================================================================

    def connect(self):
        """Open a psycopg2 connection (lazy)."""
        if self._conn is not None and not self._conn.closed:
            return self._conn
        import psycopg2
        self._conn = psycopg2.connect(self.config.dsn())
        self._conn.autocommit = False
        return self._conn

    def _register_vector(self, conn) -> None:
        """Register pgvector typecasters on this connection exactly once.

        register_vector issues a pg_type lookup and installs the adapters, so
        calling it per query added a DB round trip to every dense search and
        metadata read. Guarded by connection identity, so a reconnect
        re-registers and a failure is retried next call (as before).
        """
        if self._vector_registered is conn:
            return
        from pgvector.psycopg2 import register_vector

        register_vector(conn)
        self._vector_registered = conn

    def close(self) -> None:
        if self._conn is not None and not self._conn.closed:
            self._conn.close()
        self._conn = None
        self._vector_registered = None

    def __enter__(self) -> "PgVectorStore":
        self.connect()
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # ==================================================================
    # Schema management
    # ==================================================================

    def ensure_schema(self) -> None:
        """Create the legacy medrag schema/tables if missing.

        Never issues DDL against medpat: that corpus is created and migrated by
        docker/medpat/init/*.sql and written by src.chunking / src.embedding.
        """
        if self.schema == "medpat":
            return
        conn = self.connect()
        cur = conn.cursor()

        # 1. Schema (commit immediately so rollback from later steps doesn't undo it)
        cur.execute(f"CREATE SCHEMA IF NOT EXISTS {self.schema}")
        conn.commit()

        # 2. Extension — try CREATE EXTENSION; if it fails, check for manual install
        try:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
            conn.commit()
        except Exception:
            conn.rollback()
            cur.execute("SELECT 1 FROM pg_type WHERE typname = 'vector'")
            if cur.fetchone() is None:
                raise RuntimeError(
                    "pgvector extension not available and vector type not found. "
                    "Install pgvector or run the vector SQL manually."
                )
            conn.commit()

        cur.execute(f"""
            CREATE TABLE IF NOT EXISTS {self.schema}.chunks (
                id              TEXT PRIMARY KEY,
                document_id     TEXT,
                chunk_type      TEXT,
                section         TEXT,
                subsection      TEXT,
                breadcrumb      JSONB,
                parent_id       TEXT,
                table_id        TEXT,
                figure_id       TEXT,
                document_position BIGINT,
                text            TEXT,
                embedding_text  TEXT,
                metadata        JSONB
            )
        """)

        cur.execute(f"""
            CREATE TABLE IF NOT EXISTS {self.schema}.{self.emb_table} (
                chunk_id    TEXT PRIMARY KEY REFERENCES {self.schema}.chunks(id) ON DELETE CASCADE,
                embedding   vector({EMBEDDING_DIM}),
                created_at  TIMESTAMPTZ DEFAULT now()
            )
        """)

        # IVFFlat index for fast approximate cosine search.
        # Rebuilt after bulk load with REINDEX.
        # Skip if vector index types aren't available (extension not fully loaded).
        try:
            cur.execute(f"""
                CREATE INDEX IF NOT EXISTS idx_embeddings_vector
                ON {self.schema}.{self.emb_table}
                USING ivfflat (embedding vector_cosine_ops)
                WITH (lists = 100)
            """)
        except Exception:
            conn.rollback()
            # IVFFlat not available — fall back to no index (search still works)
            pass

        cur.execute(f"""
            CREATE INDEX IF NOT EXISTS idx_chunks_document_id
            ON {self.schema}.chunks (document_id)
        """)

        cur.execute(f"""
            CREATE INDEX IF NOT EXISTS idx_chunks_chunk_type
            ON {self.schema}.chunks (chunk_type)
        """)

        conn.commit()
        cur.close()

    def rebuild_index(self) -> None:
        """Rebuild the legacy IVFFlat index (call after bulk inserts).

        medpat's HNSW index is managed by scripts/medpat_fullvec_migrate.py and
        the embedding CLI (--drop-index), never from the retrieval path.
        """
        if self.schema == "medpat":
            return
        conn = self.connect()
        cur = conn.cursor()
        try:
            cur.execute(f"REINDEX INDEX {self.schema}.idx_embeddings_vector")
            conn.commit()
        except Exception:
            conn.rollback()
        cur.close()

    # ==================================================================
    # Metadata upsert
    # ==================================================================

    def upsert_chunks(self, rows: List[Dict[str, Any]], batch_size: int = 5000) -> int:
        """Insert or update chunk metadata rows. Returns count inserted."""
        conn = self.connect()
        cur = conn.cursor()
        count = 0

        for start in range(0, len(rows), batch_size):
            batch = rows[start:start + batch_size]
            values = []
            for r in batch:
                breadcrumb = r.get("breadcrumb")
                if isinstance(breadcrumb, (list, tuple)):
                    breadcrumb = json.dumps(list(breadcrumb))
                elif breadcrumb is not None:
                    breadcrumb = json.dumps([str(breadcrumb)])
                else:
                    breadcrumb = "[]"

                values.append((
                    r.get("id", ""),
                    r.get("document_id", ""),
                    r.get("chunk_type", ""),
                    r.get("section", ""),
                    r.get("subsection", ""),
                    breadcrumb,
                    r.get("parent_id", ""),
                    r.get("table_id", ""),
                    r.get("figure_id", ""),
                    r.get("document_position", 0),
                    r.get("text", ""),
                    r.get("embedding_text", ""),
                    json.dumps(r.get("metadata", {})),
                ))

            cur.executemany(f"""
                INSERT INTO {self.schema}.chunks
                    (id, document_id, chunk_type, section, subsection,
                     breadcrumb, parent_id, table_id, figure_id,
                     document_position, text, embedding_text, metadata)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (id) DO UPDATE SET
                    document_id = EXCLUDED.document_id,
                    chunk_type = EXCLUDED.chunk_type,
                    section = EXCLUDED.section,
                    subsection = EXCLUDED.subsection,
                    breadcrumb = EXCLUDED.breadcrumb,
                    parent_id = EXCLUDED.parent_id,
                    table_id = EXCLUDED.table_id,
                    figure_id = EXCLUDED.figure_id,
                    document_position = EXCLUDED.document_position,
                    text = EXCLUDED.text,
                    embedding_text = EXCLUDED.embedding_text,
                    metadata = EXCLUDED.metadata
            """, values)
            count += len(batch)

        conn.commit()
        cur.close()
        return count

    # ==================================================================
    # Embedding upsert
    # ==================================================================

    def upsert_embeddings(
        self,
        chunk_ids: List[str],
        embeddings: np.ndarray,
        batch_size: int = 2000,
    ) -> int:
        """Insert or update embedding vectors. Returns count inserted.

        ``embeddings`` must be float32 of shape (len(chunk_ids), 768).
        Vectors are L2-normalized before storage for cosine similarity search.
        """
        trace = get_trace()

        conn = self.connect()
        self._register_vector(conn)
        cur = conn.cursor()

        n = len(chunk_ids)
        if n == 0:
            return 0

        trace.log("pg_upsert_embeddings", params={"n_vectors": n, "dim": embeddings.shape[1] if embeddings.ndim > 1 else 0, "batch_size": batch_size})

        vecs = np.asarray(embeddings, dtype=np.float32)
        if vecs.shape != (n, EMBEDDING_DIM):
            raise ValueError(
                f"embedding shape {vecs.shape} != expected ({n}, {EMBEDDING_DIM})"
            )

        # L2-normalize for cosine similarity via inner product
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        vecs = vecs / norms

        count = 0
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            batch_ids = chunk_ids[start:end]
            batch_vecs = vecs[start:end]

            values = list(zip(batch_ids, [v.tolist() for v in batch_vecs]))

            cur.executemany(f"""
                INSERT INTO {self.schema}.{self.emb_table} (chunk_id, embedding)
                VALUES (%s, %s)
                ON CONFLICT (chunk_id) DO UPDATE SET
                    embedding = EXCLUDED.embedding,
                    created_at = now()
            """, values)
            count += len(batch_ids)

        conn.commit()
        cur.close()
        return count

    # ==================================================================
    # Search
    # ==================================================================

    def search(
        self,
        query_vector: np.ndarray,
        top_k: int = 10,
        method: str = "cosine",
    ) -> List[Tuple[str, float]]:
        """Find the top_k most similar chunks to query_vector.

        Returns list of (chunk_id, score) tuples sorted by descending similarity.
        Scores are cosine similarities in [-1, 1].
        """
        import time as _time
        trace = get_trace()

        t0 = _time.perf_counter()
        conn = self.connect()
        self._register_vector(conn)
        cur = conn.cursor()

        q = np.asarray(query_vector, dtype=np.float32).flatten()
        # L2-normalize query for cosine similarity
        norm = np.linalg.norm(q)
        if norm > 0:
            q = q / norm

        # pgvector HNSW only explores ~ef_search candidates per query; the
        # default (40) would cap results below top_k. Raise it for this scan.
        cur.execute("SET LOCAL hnsw.ef_search = %s", (max(top_k + 64, 160),))
        # IVFFlat recall knob: how many of the ~2115 lists the scan touches.
        # It must NOT scale with top_k - the old max(top_k, 16) made a top_k=60
        # dense leg scan 60 lists (~0.4s). Each probe is ~7ms here, so 8 probes
        # is ~40ms and 16 is ~100ms. Tune with IVFFLAT_PROBES.
        try:
            probes = max(1, int(os.environ.get("IVFFLAT_PROBES", "16")))
        except (TypeError, ValueError):
            probes = 16
        cur.execute("SET LOCAL ivfflat.probes = %s", (probes,))

        if method == "cosine":
            # <#> returns negative inner product; negate for similarity
            cur.execute(f"""
                SELECT chunk_id, - (embedding <#> %s::vector) AS similarity
                FROM {self.schema}.{self.emb_table}
                ORDER BY embedding <#> %s::vector
                LIMIT %s
            """, (q.tolist(), q.tolist(), top_k))
        elif method == "l2":
            cur.execute(f"""
                SELECT chunk_id, -(embedding <-> %s::vector) AS similarity
                FROM {self.schema}.{self.emb_table}
                ORDER BY embedding <-> %s::vector
                LIMIT %s
            """, (q.tolist(), q.tolist(), top_k))
        else:
            raise ValueError(f"unknown method: {method!r}")

        results = [(row[0], float(row[1])) for row in cur.fetchall()]
        dur_ms = (_time.perf_counter() - t0) * 1000
        cur.close()

        if trace is not None:
            trace.log(
                "pg_search",
                params={
                    "method": method,
                    "top_k": top_k,
                    "query_shape": list(q.shape),
                    "sql": f"SELECT chunk_id, -(embedding <#> $1::vector) ... LIMIT {top_k}" if method == "cosine" else f"SELECT chunk_id, -(embedding <-> $1::vector) ... LIMIT {top_k}",
                },
                result={
                    "n_results": len(results),
                    "top_chunk_ids": [r[0] for r in results[:10]],
                    "top_scores": [round(r[1], 6) for r in results[:10]],
                },
                duration_ms=dur_ms,
            )
        return results

    def search_batch(
        self,
        query_vectors: np.ndarray,
        top_k: int = 10,
    ) -> List[List[Tuple[str, float]]]:
        """Search with multiple query vectors. Returns one result list per query."""
        results = []
        for q in query_vectors:
            results.append(self.search(q, top_k))
        return results

    def search_filtered(
        self,
        query_vector: np.ndarray,
        top_k: int = 10,
        method: str = "cosine",
        **filters: Any,
    ) -> List[Tuple[str, float]]:
        """Vector search restricted by chunk metadata filters.

        Used by V2 local retrieval (spec section 19): search only inside one
        selected paper (or a section / node type / table) instead of the whole
        corpus. The vector scan is exact on the filtered subset, which is
        small (a paper has tens-to-low-thousands of chunks), so the metadata
        constraint is applied first.

        Example:
            store.search_filtered(qvec, top_k=40, document_id="PMC11743015")
        """
        import time as _time

        t0 = _time.perf_counter()
        conn = self.connect()
        self._register_vector(conn)
        cur = conn.cursor()

        q = np.asarray(query_vector, dtype=np.float32).flatten()
        norm = np.linalg.norm(q)
        if norm > 0:
            q = q / norm
        cur.execute("SET LOCAL hnsw.ef_search = %s", (max(top_k + 64, 160),))
        # IVFFlat recall knob: how many of the ~2115 lists the scan touches.
        # It must NOT scale with top_k - the old max(top_k, 16) made a top_k=60
        # dense leg scan 60 lists (~0.4s). Each probe is ~7ms here, so 8 probes
        # is ~40ms and 16 is ~100ms. Tune with IVFFLAT_PROBES.
        try:
            probes = max(1, int(os.environ.get("IVFFLAT_PROBES", "16")))
        except (TypeError, ValueError):
            probes = 16
        cur.execute("SET LOCAL ivfflat.probes = %s", (probes,))

        where: List[str] = []
        params: List[Any] = []
        for col, val in filters.items():
            if val is None:
                continue
            where.append(f"c.{col} = %s")
            params.append(val)

        if method == "cosine":
            sql = (
                f"SELECT e.chunk_id, -(e.embedding <#> %s::vector) AS similarity "
                f"FROM {self.schema}.{self.emb_table} e JOIN {self.schema}.chunks c ON c.id = e.chunk_id "
            )
            if where:
                sql += "WHERE " + " AND ".join(where) + " "
            sql += "ORDER BY e.embedding <#> %s::vector LIMIT %s"
            params = [q.tolist(), *params, q.tolist(), max(0, int(top_k))]
        elif method == "l2":
            sql = (
                f"SELECT e.chunk_id, -(e.embedding <-> %s::vector) AS similarity "
                f"FROM {self.schema}.{self.emb_table} e JOIN {self.schema}.chunks c ON c.id = e.chunk_id "
            )
            if where:
                sql += "WHERE " + " AND ".join(where) + " "
            sql += "ORDER BY e.embedding <-> %s::vector LIMIT %s"
            params = [q.tolist(), *params, q.tolist(), max(0, int(top_k))]
        else:
            raise ValueError(f"unknown method: {method!r}")

        cur.execute(sql, params)
        results = [(row[0], float(row[1])) for row in cur.fetchall()]
        dur_ms = (_time.perf_counter() - t0) * 1000
        cur.close()
        return results

    # ==================================================================
    # Metadata resolution
    # ==================================================================

    def get_chunk(self, chunk_id: str) -> Optional[Dict[str, Any]]:
        """Fetch metadata for a single chunk."""
        conn = self.connect()
        cur = conn.cursor()
        cur.execute(f"""
            SELECT id, document_id, chunk_type, section, subsection,
                   breadcrumb, parent_id, document_position, text,
                   metadata->>'title' AS title,
                   metadata->>'journal' AS journal
            FROM {self.schema}.chunks
            WHERE id = %s
        """, (chunk_id,))
        row = cur.fetchone()
        cur.close()
        if row is None:
            return None
        bc = row[5]
        if isinstance(bc, (list, dict)):
            breadcrumb = bc
        elif bc:
            breadcrumb = json.loads(bc)
        else:
            breadcrumb = []
        return {
            "id": row[0],
            "document_id": row[1],
            "chunk_type": row[2],
            "section": row[3],
            "subsection": row[4],
            "breadcrumb": breadcrumb,
            "parent_id": row[6],
            "document_position": row[7],
            "text": row[8],
            "title": row[9] or "",
            "journal": row[10] or "",
        }

    def get_chunks(self, chunk_ids: List[str]) -> Dict[str, Dict[str, Any]]:
        """Fetch metadata for multiple chunks. Returns {chunk_id: metadata}."""
        if not chunk_ids:
            return {}
        conn = self.connect()
        cur = conn.cursor()
        cur.execute(f"""
            SELECT id, document_id, chunk_type, section, subsection,
                   breadcrumb, parent_id, document_position, text,
                   metadata->>'title' AS title,
                   metadata->>'journal' AS journal
            FROM {self.schema}.chunks
            WHERE id = ANY(%s)
        """, (chunk_ids,))
        results = {}
        for row in cur.fetchall():
            bc = row[5]
            if isinstance(bc, (list, dict)):
                breadcrumb = bc
            elif bc:
                breadcrumb = json.loads(bc)
            else:
                breadcrumb = []
            results[row[0]] = {
                "id": row[0],
                "document_id": row[1],
                "chunk_type": row[2],
                "section": row[3],
                "subsection": row[4],
                "breadcrumb": breadcrumb,
                "parent_id": row[6],
                "document_position": row[7],
                "text": row[8],
                "title": row[9] or "",
                "journal": row[10] or "",
            }
        cur.close()
        return results

    def get_embedding(self, chunk_id: str) -> Optional[np.ndarray]:
        """Fetch the embedding vector for a single chunk."""
        conn = self.connect()
        self._register_vector(conn)
        cur = conn.cursor()
        cur.execute(f"""
            SELECT embedding FROM {self.schema}.{self.emb_table}
            WHERE chunk_id = %s
        """, (chunk_id,))
        row = cur.fetchone()
        cur.close()
        if row is None:
            return None
        return _to_np(row[0])

    def get_embeddings(self, chunk_ids: List[str]) -> Dict[str, np.ndarray]:
        """Fetch embedding vectors for multiple chunks."""
        if not chunk_ids:
            return {}
        conn = self.connect()
        self._register_vector(conn)
        cur = conn.cursor()
        cur.execute(f"""
            SELECT chunk_id, embedding FROM {self.schema}.{self.emb_table}
            WHERE chunk_id = ANY(%s)
        """, (chunk_ids,))
        results = {}
        for row in cur.fetchall():
            results[row[0]] = _to_np(row[1])
        cur.close()
        return results

    # ==================================================================
    # Stats
    # ==================================================================

    def count_chunks(self) -> int:
        conn = self.connect()
        cur = conn.cursor()
        cur.execute(f"SELECT COUNT(*) FROM {self.schema}.chunks")
        n = cur.fetchone()[0]
        cur.close()
        return n

    def count_embeddings(self) -> int:
        conn = self.connect()
        cur = conn.cursor()
        cur.execute(f"SELECT COUNT(*) FROM {self.schema}.{self.emb_table}")
        n = cur.fetchone()[0]
        cur.close()
        return n

    def count_documents(self) -> int:
        conn = self.connect()
        cur = conn.cursor()
        cur.execute(f"SELECT COUNT(DISTINCT document_id) FROM {self.schema}.chunks")
        n = cur.fetchone()[0]
        cur.close()
        return n

    def stats(self) -> Dict[str, Any]:
        return {
            "chunks": self.count_chunks(),
            "embeddings": self.count_embeddings(),
            "documents": self.count_documents(),
            "embedding_dim": EMBEDDING_DIM,
            "host": self.config.host,
            "port": self.config.port,
            "database": self.config.database,
        }


# ==================================================================
# Convenience: load from existing FAISS/embedding files
# ==================================================================

def load_from_parquet_embeddings(
    store: PgVectorStore,
    embedding_dir: Path,
    corpus_path: Optional[Path] = None,
    batch_size: int = 2000,
    progress_callback=None,
) -> Dict[str, Any]:
    """Bulk-load embeddings from the existing .embeddings.parquet files into pgvector.

    Optionally also loads chunk metadata from corpus.parquet.
    Returns stats dict.
    """
    import pyarrow.compute as pc
    import pyarrow.parquet as pq

    embedding_dir = Path(embedding_dir)
    emb_files = sorted(embedding_dir.rglob("*.embeddings.parquet"))

    total_embeddings = 0
    total_files = 0
    start = time.time()

    for fi, emb_path in enumerate(emb_files):
        table = pq.read_table(str(emb_path), columns=["chunk_id", "embedding"])
        n = table.num_rows
        if n == 0:
            continue

        ids = table["chunk_id"].to_pylist()
        flat = pc.list_flatten(table["embedding"]).to_numpy()
        vecs = flat.reshape(n, EMBEDDING_DIM).astype(np.float32, copy=False)

        store.upsert_embeddings(ids, vecs, batch_size=batch_size)
        total_embeddings += n
        total_files += 1

        if progress_callback:
            progress_callback(fi + 1, len(emb_files), total_embeddings)

    # Load chunk metadata from corpus.parquet if provided
    total_chunks = 0
    if corpus_path and corpus_path.is_file():
        pf = pq.ParquetFile(str(corpus_path))
        for batch in pf.iter_batches(batch_size=50_000):
            df = batch.to_pandas()
            rows = df.to_dict("records")
            store.upsert_chunks(rows, batch_size=batch_size)
            total_chunks += len(rows)

    # Rebuild the IVFFlat index after bulk load
    try:
        store.rebuild_index()
    except Exception:
        pass  # IVFFlat requires >30 rows; skip if too small

    elapsed = time.time() - start

    return {
        "embedding_files": total_files,
        "total_embeddings": total_embeddings,
        "total_chunks": total_chunks,
        "elapsed_seconds": round(elapsed, 2),
    }
