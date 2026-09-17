"""ParadeDB pg_search sparse leg - BM25 inside Postgres.

Drop-in replacement for the on-disk Rank-BM25 index (``BM25Index``) with the
same ``search`` / ``search_single`` / ``search_restricted`` interface, so the
hybrid retriever BM25 leg can point at a ParadeDB index instead of
``index/bm25_v2/*.npy`` files. Everything stays in the medpat database: no
parquet, no numpy, no disk artifacts.

Query shape: ``WHERE embedding_text ||| <query> ORDER BY pdb.score(id) DESC``.
``|||`` is match-any (disjunction over the query tokens) - the same OR
semantics as the legacy Rank-BM25 leg - scored with real Okapi-BM25.

Requires the ``chunks_bm25_idx`` ParadeDB index (see
docker/medpat/init/02_search.sql). Connection: MEDPAT_DSN.
"""

from __future__ import annotations

import os
import threading
from typing import Any, List, Optional, Sequence

from src.lib.models import ScoredChunk

DEFAULT_DSN = os.environ.get(
    "MEDPAT_DSN", "postgresql://medpat:CHANGEME@localhost:5433/medpat"
)


class PgSearchBM25:
    """BM25 sparse leg backed by ParadeDB pg_search."""

    def __init__(self, dsn: Optional[str] = None, schema: str = "medpat",
                 table: str = "chunks", index: str = "chunks_bm25_idx") -> None:
        self.dsn = dsn or DEFAULT_DSN
        self.schema = schema
        self.table = table
        self.index = index
        self._conn: Any = None
        self._lock = threading.Lock()

    # -- connection ----------------------------------------------------

    def _connect(self) -> Any:
        if self._conn is None or getattr(self._conn, "closed", 0) == 1:
            import psycopg2

            with self._lock:
                if self._conn is None or getattr(self._conn, "closed", 0) == 1:
                    self._conn = psycopg2.connect(self.dsn)
        return self._conn

    def ping(self) -> None:
        """Raise if the ParadeDB index is missing - fail loud, never silent."""
        cur = self._connect().cursor()
        try:
            cur.execute(
                "SELECT 1 FROM pg_indexes"
                " WHERE schemaname = %s AND indexname = %s"
                "   AND indexdef LIKE '%%paradedb%%'",
                (self.schema, self.index),
            )
            if cur.fetchone() is None:
                raise RuntimeError(
                    f"ParadeDB index {self.schema}.{self.index} not found - "
                    "run scripts/medpat_bm25_index.py or apply "
                    "docker/medpat/init/02_search.sql"
                )
        finally:
            cur.close()

    # -- same contract as BM25Index -----------------------------------

    def search(self, queries, top_k: int) -> List[List[ScoredChunk]]:
        query_list = [queries] if isinstance(queries, str) else list(queries)
        return [self.search_single(q, top_k) for q in query_list]

    def search_single(self, query: str, top_k: int) -> List[ScoredChunk]:
        return self._run(query, None, top_k)

    def search_restricted(self, query: str, chunk_ids: Sequence[str],
                          top_k: int) -> List[ScoredChunk]:
        """BM25 restricted to a subset of chunk ids (per-paper search)."""
        ids = [str(c) for c in chunk_ids]
        if not ids:
            return []
        return self._run(query, ids, top_k)

    # -- query ---------------------------------------------------------

    def _run(self, query: str, ids: Optional[List[str]],
             top_k: int) -> List[ScoredChunk]:
        if not (query or "").strip() or top_k <= 0:
            return []
        conn = self._connect()
        cur = conn.cursor()
        sql = (f"SELECT id, pdb.score(id) AS s"
               f" FROM {self.schema}.{self.table}"
               f" WHERE retrieval_eligible AND embedding_text ||| %s")
        params: list = [query]
        if ids is not None:
            sql += " AND id = ANY(%s)"
            params.append(ids)
        sql += " ORDER BY pdb.score(id) DESC LIMIT %s"
        params.append(int(top_k))
        try:
            cur.execute(sql, tuple(params))
            rows = cur.fetchall()
        except Exception:
            conn.rollback()  # never leave the shared connection aborted
            raise
        finally:
            cur.close()
        return [
            ScoredChunk(chunk_id=str(r[0]), score=float(r[1]),
                        retrieval_method="bm25", rank=i + 1)
            for i, r in enumerate(rows)
        ]

