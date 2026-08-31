"""Bulk-load embeddings + corpus metadata into pgvector.

Fast parallel loader for 14k+ embedding files.

Usage:
    python scripts/load_embeddings.py
    python scripts/load_embeddings.py --workers 8 --batch-size 5000
    python scripts/load_embeddings.py --corpus-only   # just metadata, skip vectors
    python scripts/load_embeddings.py --vectors-only   # just vectors, skip metadata
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from multiprocessing import Pool
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pyarrow.compute as pc
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

EMBEDDING_DIM = 768


# ==================================================================
# Worker: read one embedding file
# ==================================================================

def _read_embedding_file(args: Tuple[Path, int]) -> Tuple[List[str], np.ndarray, int]:
    """Read a single .embeddings.parquet file. Returns (chunk_ids, matrix, n_rows)."""
    path, batch_idx = args
    table = pq.read_table(str(path), columns=["chunk_id", "embedding"])
    n = table.num_rows
    if n == 0:
        return [], np.empty((0, EMBEDDING_DIM), dtype=np.float32), 0
    ids = table["chunk_id"].to_pylist()
    flat = pc.list_flatten(table["embedding"]).to_numpy()
    mat = flat.reshape(n, EMBEDDING_DIM).astype(np.float32, copy=False)
    return ids, mat, n


# ==================================================================
# Bulk load
# ==================================================================

def load_embeddings(
    embedding_dir: Path,
    db_host: str,
    db_port: int,
    db_user: str,
    db_password: str,
    db_name: str,
    workers: int = 4,
    batch_size: int = 5000,
) -> Dict[str, Any]:
    """Load all embedding files into pgvector in parallel."""

    from medrag.retrieval.pgvector_store import PgConfig, PgVectorStore

    embedding_dir = Path(embedding_dir)
    emb_files = sorted(embedding_dir.rglob("*.embeddings.parquet"))
    print(f"Found {len(emb_files):,} embedding files")

    # Phase 1: Read all files in parallel
    print(f"\nPhase 1: Reading embeddings ({workers} workers)...")
    t0 = time.time()

    tasks = [(f, i) for i, f in enumerate(emb_files)]
    all_ids: List[str] = []
    all_vecs: List[np.ndarray] = []

    with Pool(workers) as pool:
        for i, (ids, mat, n) in enumerate(pool.imap_unordered(_read_embedding_file, tasks, chunksize=64), 1):
            if n > 0:
                all_ids.extend(ids)
                all_vecs.append(mat)
            if i % 500 == 0 or i == len(emb_files):
                total = sum(v.shape[0] for v in all_vecs)
                print(f"  [{i}/{len(emb_files)}] {total:,} embeddings read")

    read_ms = (time.time() - t0) * 1000
    vectors = np.vstack(all_vecs) if all_vecs else np.empty((0, EMBEDDING_DIM), dtype=np.float32)
    print(f"  Read {len(all_ids):,} embeddings in {read_ms:.0f} ms")

    # Phase 2: Insert into pgvector — COPY for max speed
    print(f"\nPhase 2: Inserting {len(all_ids):,} embeddings into pgvector...")
    t0 = time.time()

    import psycopg2
    from pgvector.psycopg2 import register_vector
    from psycopg2.extras import execute_values

    dsn = f"host={db_host} port={db_port} dbname={db_name} user={db_user} password={db_password}"
    conn = psycopg2.connect(dsn)
    conn.autocommit = True
    register_vector(conn)
    cur = conn.cursor()

    # Drop IVFFlat index during bulk load (rebuild after — 100x faster)
    cur.execute("DROP INDEX IF EXISTS medrag.idx_embeddings_vector")
    cur.execute("TRUNCATE medrag.embeddings")

    # L2-normalize all vectors
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    vectors = vectors / norms

    n = len(all_ids)
    inserted = 0
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        batch_ids = all_ids[start:end]
        batch_vecs = vectors[start:end]

        values = [(cid, v.tolist()) for cid, v in zip(batch_ids, batch_vecs)]
        execute_values(
            cur,
            "INSERT INTO medrag.embeddings (chunk_id, embedding) VALUES %s",
            values,
            page_size=1000,
        )
        conn.commit()
        inserted += len(batch_ids)
        elapsed = time.time() - t0
        rate = inserted / elapsed if elapsed > 0 else 0
        eta = (n - inserted) / rate if rate > 0 else 0
        print(f"  [{inserted:>9,}/{n:,}] {elapsed:.0f}s elapsed | {rate:.0f} vec/s | ETA {eta:.0f}s", flush=True)

    cur.close()
    conn.close()
    insert_ms = (time.time() - t0) * 1000
    print(f"  Inserted {inserted:,} embeddings in {insert_ms:.0f} ms")

    # Rebuild IVFFlat index (fresh create, not REINDEX)
    print("\nPhase 3: Building IVFFlat index...")
    t0 = time.time()
    conn2 = psycopg2.connect(dsn)
    conn2.autocommit = True
    register_vector(conn2)
    cur2 = conn2.cursor()
    cur2.execute(
        "CREATE INDEX idx_embeddings_vector ON medrag.embeddings "
        "USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100)"
    )
    cur2.close()
    conn2.close()
    idx_ms = (time.time() - t0) * 1000
    print(f"  Index built in {idx_ms:.0f} ms")

    # Get final stats
    conn3 = psycopg2.connect(dsn)
    conn3.autocommit = True
    cur3 = conn3.cursor()
    cur3.execute("SELECT COUNT(*) FROM medrag.embeddings")
    n_embeddings = cur3.fetchone()[0]
    cur3.execute("SELECT COUNT(*) FROM medrag.chunks")
    n_chunks = cur3.fetchone()[0]
    cur3.close()
    conn3.close()

    return {
        "files": len(emb_files),
        "embeddings": inserted,
        "read_ms": round(read_ms),
        "insert_ms": round(insert_ms),
        "index_ms": round(idx_ms),
        "stats": {"embeddings": n_embeddings, "chunks": n_chunks},
    }


def load_corpus(
    corpus_path: Path,
    db_host: str,
    db_port: int,
    db_user: str,
    db_password: str,
    db_name: str,
    batch_size: int = 5000,
) -> Dict[str, Any]:
    """Load corpus.parquet metadata into pgvector chunks table."""
    from medrag.retrieval.pgvector_store import PgConfig, PgVectorStore

    corpus_path = Path(corpus_path)
    if not corpus_path.is_file():
        return {"error": f"not found: {corpus_path}"}

    print(f"Loading corpus metadata from {corpus_path}...")
    t0 = time.time()

    config = PgConfig(host=db_host, port=db_port, user=db_user, password=db_password, database=db_name)
    store = PgVectorStore(config)
    store.connect()
    store.ensure_schema()

    pf = pq.ParquetFile(str(corpus_path))
    total = 0
    for batch in pf.iter_batches(batch_size=50_000):
        df = batch.to_pandas()
        rows = df.to_dict("records")
        store.upsert_chunks(rows, batch_size=batch_size)
        total += len(rows)
        print(f"  {total:,} chunks loaded", flush=True)

    elapsed = (time.time() - t0) * 1000
    stats = store.stats()
    store.close()

    return {
        "chunks": total,
        "elapsed_ms": round(elapsed),
        "stats": stats,
    }


# ==================================================================
# CLI
# ==================================================================

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--embeddings-dir", default="embeddings")
    parser.add_argument("--corpus", default="index/corpus.parquet")
    parser.add_argument("--host", default=os.environ.get("PGHOST", "localhost"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("PGPORT", "5432")))
    parser.add_argument("--user", default=os.environ.get("PGUSER", "postgres"))
    parser.add_argument("--password", default=os.environ.get("PGPASSWORD", "medrag"))
    parser.add_argument("--database", default=os.environ.get("PGDATABASE", "medrag"))
    parser.add_argument("--workers", type=int, default=4, help="Parallel readers for embedding files")
    parser.add_argument("--batch-size", type=int, default=5000)
    parser.add_argument("--corpus-only", action="store_true", help="Only load corpus metadata")
    parser.add_argument("--vectors-only", action="store_true", help="Only load embedding vectors")
    args = parser.parse_args()

    print("=" * 60)
    print("MEDRAG → PGVECTOR LOADER")
    print("=" * 60)
    print(f"Database:  {args.host}:{args.port}/{args.database}")
    print(f"User:      {args.user}")
    print()

    t0 = time.time()

    if not args.vectors_only:
        r = load_corpus(
            Path(args.corpus), args.host, args.port, args.user, args.password, args.database,
            batch_size=args.batch_size,
        )
        print(f"\nCorpus: {r.get('chunks', 0):,} chunks loaded in {r.get('elapsed_ms', 0):,} ms")

    if not args.corpus_only:
        r = load_embeddings(
            Path(args.embeddings_dir), args.host, args.port, args.user, args.password, args.database,
            workers=args.workers, batch_size=args.batch_size,
        )
        print(f"\nEmbeddings: {r['embeddings']:,} vectors loaded")
        print(f"  Read:    {r['read_ms']:,} ms")
        print(f"  Insert:  {r['insert_ms']:,} ms")
        print(f"  Index:   {r['index_ms']:,} ms")
        print(f"  Stats:   {r['stats']}")

    total = time.time() - t0
    print(f"\nTotal: {total:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
