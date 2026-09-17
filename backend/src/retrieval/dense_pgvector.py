"""Dense retrieval index backed by PostgreSQL + pgvector.

Drop-in replacement for ``medrag.retrieval.dense.DenseIndex`` that stores
vectors in pgvector instead of a local FAISS file.

The interface matches DenseIndex so the retrieval engine, hybrid retriever,
and orchestrator all work unchanged.

Usage:
    from src.retrieval.dense_pgvector import PgDenseIndex

    index = PgDenseIndex.from_env()
    results = index.search_single(query_vector, top_k=10)
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

from src.lib.models import ScoredChunk
from src.retrieval.pgvector_store import PgConfig, PgVectorStore, EMBEDDING_DIM


class PgDenseIndex:
    """pgvector-backed dense index with the same interface as DenseIndex.

    Unlike the FAISS version, this does NOT hold all vectors in memory.
    Queries go directly to PostgreSQL, which manages the vector index.
    """

    def __init__(self, store: PgVectorStore) -> None:
        self.store = store
        self._chunk_ids_cache: Optional[np.ndarray] = None

    # ==================================================================
    # Construction
    # ==================================================================

    @classmethod
    def from_env(cls, config: Optional[PgConfig] = None) -> "PgDenseIndex":
        """Create from environment variables (PGHOST, PGPORT, etc.)."""
        store = PgVectorStore(config)
        store.connect()
        store.ensure_schema()
        return cls(store)

    @classmethod
    def from_config(cls, host: str, port: int, user: str, password: str, database: str) -> "PgDenseIndex":
        config = PgConfig(host=host, port=port, user=user, password=password, database=database)
        return cls.from_env(config)

    # ==================================================================
    # Properties (matches DenseIndex interface)
    # ==================================================================

    @property
    def n_total(self) -> int:
        return self.store.count_embeddings()

    @property
    def dimension(self) -> int:
        return EMBEDDING_DIM

    @property
    def index_type(self) -> str:
        return "pgvector_ivfflat"

    @property
    def normalize(self) -> bool:
        return True  # pgvector store always normalizes

    @property
    def chunk_ids(self) -> np.ndarray:
        """Lazily load all chunk IDs (needed by orchestrator for embedding reconstruction)."""
        if self._chunk_ids_cache is None:
            conn = self.store.connect()
            cur = conn.cursor()
            cur.execute(
                f"SELECT chunk_id FROM {self.store.schema}.{self.store.emb_table} "
                "ORDER BY chunk_id"
            )
            self._chunk_ids_cache = np.array([row[0] for row in cur.fetchall()], dtype=object)
            cur.close()
        return self._chunk_ids_cache

    # ==================================================================
    # Build (bulk load from parquet files)
    # ==================================================================

    @classmethod
    def build(
        cls,
        embedding_files: Sequence[Path],
        out_dir: Path,
        corpus_path: Optional[Path] = None,
        index_type: str = "flat",
        normalize: bool = True,
        hnsw_m: int = 32,
        ef_construction: int = 200,
        ef_search: int = 64,
        batch_size: int = 65536,
        config: Optional[PgConfig] = None,
    ) -> "PgDenseIndex":
        """Bulk-load embeddings from parquet files into pgvector.

        This replaces the FAISS build step. The ``out_dir`` is used only
        for a metadata JSON file; the actual vectors live in PostgreSQL.
        """
        from src.retrieval.pgvector_store import load_from_parquet_embeddings

        store = PgVectorStore(config)
        store.connect()
        store.ensure_schema()

        def progress(i, total, count):
            print(f"  [{i}/{total}] {count:,} embeddings loaded", flush=True)

        stats = load_from_parquet_embeddings(
            store,
            embedding_dir=embedding_files[0].parent if embedding_files else Path("."),
            corpus_path=corpus_path,
            batch_size=batch_size,
            progress_callback=progress,
        )

        # Write metadata JSON to out_dir (replaces dense_meta.json)
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        meta = {
            "index_type": "pgvector_ivfflat",
            "metric": "cosine",
            "cosine": True,
            "normalize": True,
            "dimension": EMBEDDING_DIM,
            "n_total": stats["total_embeddings"],
            "build_seconds": stats["elapsed_seconds"],
            "built_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "source": "pgvector",
        }
        (out_dir / "dense_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

        return cls(store)

    # ==================================================================
    # Load (connect to existing pgvector)
    # ==================================================================

    @classmethod
    def load(cls, out_dir: Path, config: Optional[PgConfig] = None) -> "PgDenseIndex":
        """Load from an existing pgvector database.

        ``out_dir`` is read for dense_meta.json config hints, but the
        actual data comes from PostgreSQL.
        """
        meta_path = Path(out_dir) / "dense_meta.json"
        # We could read pgvector config from meta, but env vars take precedence
        store = PgVectorStore(config)
        store.connect()
        store.ensure_schema()
        return cls(store)

    # ==================================================================
    # Search (same interface as FAISS DenseIndex)
    # ==================================================================

    def search(self, query_vectors: np.ndarray, top_k: int) -> List[List[ScoredChunk]]:
        """Search with one or more query vectors.

        Returns one ranked list per query (same as DenseIndex.search).
        """
        q = np.asarray(query_vectors, dtype=np.float32)
        if q.ndim == 1:
            q = q.reshape(1, -1)
        if q.shape[1] != self.dimension:
            raise ValueError(f"query dimension {q.shape[1]} != index dimension {self.dimension}")

        top_k = min(top_k, self.n_total)
        if top_k == 0:
            return [[] for _ in range(q.shape[0])]

        # Batch search: pgvector handles each query individually
        # For batch efficiency, we could use a UNION query, but individual
        # queries are fast enough for the use cases here.
        out: List[List[ScoredChunk]] = []
        for row_idx in range(q.shape[0]):
            results = self.store.search(q[row_idx], top_k=top_k, method="cosine")
            ranked = [
                ScoredChunk(
                    chunk_id=cid,
                    score=score,
                    retrieval_method="dense",
                    rank=rank + 1,
                )
                for rank, (cid, score) in enumerate(results)
            ]
            out.append(ranked)

        return out

    def search_single(self, query_vector: np.ndarray, top_k: int) -> List[ScoredChunk]:
        return self.search(query_vector, top_k)[0]

    # ==================================================================
    # Embedding reconstruction (for orchestrator evidence matching)
    # ==================================================================

    def reconstruct(self, chunk_id: str) -> Optional[np.ndarray]:
        """Fetch embedding vector for a single chunk by ID.

        Used by the orchestrator for evidence matching and diversity computation.
        """
        return self.store.get_embedding(chunk_id)

    def reconstruct_batch(self, chunk_ids: List[str]) -> Dict[str, np.ndarray]:
        """Fetch embedding vectors for multiple chunks."""
        return self.store.get_embeddings(chunk_ids)
