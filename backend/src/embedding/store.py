"""medpat persistence for embeddings: pending-chunk reads + vector upsert."""

from __future__ import annotations

import hashlib
from typing import Iterator, List, Tuple

import numpy as np

UPSERT_EMBEDDING = """
INSERT INTO medpat.chunk_embeddings (chunk_id, embedding, model, embedding_text_hash)
VALUES (%s, %s::vector, %s, %s)
ON CONFLICT (chunk_id) DO UPDATE SET
    embedding=EXCLUDED.embedding, model=EXCLUDED.model,
    embedding_text_hash=EXCLUDED.embedding_text_hash, updated_at=now()
"""

# Bulk variant for psycopg2.extras.execute_values: %s is the whole VALUES list.
# Template keeps the ::vector cast per row (single round-trip per flush
# instead of one per row like executemany).
UPSERT_EMBEDDING_VALUES = """
INSERT INTO medpat.chunk_embeddings (chunk_id, embedding, model, embedding_text_hash)
VALUES %s
ON CONFLICT (chunk_id) DO UPDATE SET
    embedding=EXCLUDED.embedding, model=EXCLUDED.model,
    embedding_text_hash=EXCLUDED.embedding_text_hash, updated_at=now()
"""
_ROW_TEMPLATE = "(%s, %s::vector, %s, %s)"

DROP_VECTOR_INDEX = "DROP INDEX IF EXISTS embeddings_vector_idx"
DROP_VECTOR_INDEX_ALT = "DROP INDEX IF EXISTS medpat.embeddings_vector_idx"
CREATE_VECTOR_INDEX = (
    "CREATE INDEX IF NOT EXISTS embeddings_vector_idx "
    "ON medpat.chunk_embeddings "
    "USING hnsw (embedding vector_ip_ops) WITH (m = 16, ef_construction = 128)"
)


def upsert_embeddings(cur, rows, page_size: int = 500) -> None:
    """Bulk upsert pre-formatted (chunk_id, vector_literal, model, hash) rows.

    Uses a single multi-row INSERT per page via execute_values. Falls back to
    executemany when psycopg2.extras is unavailable (tests / bare envs).
    """
    if not rows:
        return
    try:
        from psycopg2.extras import execute_values  # type: ignore
    except ImportError:
        cur.executemany(UPSERT_EMBEDDING, rows)
        return
    execute_values(cur, UPSERT_EMBEDDING_VALUES, rows,
                   template=_ROW_TEMPLATE, page_size=page_size)

# Shared predicate: retrieval-eligible, non-empty text, and either no stored
# vector or a stale one (sha256 of embedding_text changed).
_PENDING = (
    " FROM medpat.chunks c"
    " LEFT JOIN medpat.chunk_embeddings e ON e.chunk_id = c.id"
    " WHERE c.retrieval_eligible AND c.embedding_text <> ''"
    "   AND (e.chunk_id IS NULL OR e.embedding_text_hash IS DISTINCT FROM"
    "        encode(digest(c.embedding_text, 'sha256'), 'hex'))"
)


def hash_text(text: str) -> str:
    return hashlib.sha256((text or "").encode("utf-8")).hexdigest()


def count_pending_chunks(conn) -> int:
    """Total chunks that still need encoding (the progress-bar total)."""
    cur = conn.cursor()
    cur.execute("SELECT count(*)" + _PENDING)
    n = int(cur.fetchone()[0])
    cur.close()
    return n


def load_pending_chunks(conn, limit: int) -> List[Tuple[str, str, str]]:
    """(chunk_id, embedding_text, existing_hash) for chunks needing attention.

    Small-limit helper (dry-run preview). The full run uses
    iter_pending_chunks, which streams instead of materialising millions of
    rows in memory.
    """
    sql = ("SELECT c.id, c.embedding_text, e.embedding_text_hash" + _PENDING
           + " ORDER BY c.document_position, c.id")
    if limit:
        sql += " LIMIT %s"
    cur = conn.cursor()
    cur.execute(sql, (limit,) if limit else ())
    rows = cur.fetchall()
    cur.close()
    return [(str(r[0]), str(r[1] or ""), str(r[2] or "") if r[2] else "") for r in rows]


def iter_pending_chunks(conn, batch_size: int,
                        limit: int = 0) -> Iterator[List[Tuple[str, str, str]]]:
    """Yield batches of pending rows in (document_position, id) order.

    Keyset pagination, not OFFSET: bounded client memory for the full corpus,
    and each page is its own short read so the batch writes can commit without
    a long-lived server-side cursor.
    """
    sql = ("SELECT c.id, c.embedding_text, e.embedding_text_hash, c.document_position"
           + _PENDING
           + " AND (c.document_position, c.id) > (%s, %s)"
           + " ORDER BY c.document_position, c.id LIMIT %s")
    cur = conn.cursor()
    last_pos, last_id = -1, ""
    fetched = 0
    try:
        while True:
            n = batch_size
            if limit:
                n = min(n, limit - fetched)
                if n <= 0:
                    return
            cur.execute(sql, (last_pos, last_id, n))
            rows = cur.fetchall()
            if not rows:
                return
            last_pos = int(rows[-1][3])
            last_id = str(rows[-1][0])
            fetched += len(rows)
            yield [(str(r[0]), str(r[1] or ""), str(r[2] or "") if r[2] else "")
                   for r in rows]
            if len(rows) < n:
                return
    finally:
        cur.close()


def format_vector(vec: np.ndarray) -> str:
    """vector literal: '[a,b,c]' with enough precision for float32."""
    return "[%s]" % ",".join(f"{v:.9g}" for v in vec)
