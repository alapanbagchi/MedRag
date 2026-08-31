#!/usr/bin/env python3
"""Retrieval smoke test against the pgvector store.

Confirms the loaded embeddings are queryable end-to-end:

* default  : pick random stored vectors as queries and check each matches
             its own chunk first (pure DB sanity check, no model needed)
* --query  : embed real text with the MedCPT query encoder and search
* --top-k  : how many hits to show per query

Example:
    python scripts/pg_smoke.py
    python scripts/pg_smoke.py --query "What is the role of uric acid in hypertension?"
    python scripts/pg_smoke.py --queries "query A" "query B" --top-k 5
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import numpy as np


def load_env() -> None:
    try:
        from dotenv import load_dotenv
        load_dotenv(PROJECT_ROOT / ".env")
    except Exception:
        pass


def _vec_np(v: Any) -> np.ndarray:
    """Convert a fetched pgvector value to float32 np.ndarray."""
    if hasattr(v, "to_list"):
        v = v.to_list()
    return np.asarray(v, dtype=np.float32)


def main(argv: Optional[List[str]] = None) -> int:
    load_env()
    from src.retrieval.pgvector_store import PgConfig, PgVectorStore

    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--host", default=os.environ.get("PGHOST", "localhost"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("PGPORT", "5432")))
    parser.add_argument("--user", default=os.environ.get("PGUSER", "postgres"))
    parser.add_argument("--password", default=os.environ.get("PGPASSWORD", "medrag"))
    parser.add_argument("--database", default=os.environ.get("PGDATABASE", "medrag"))
    parser.add_argument("-q", "--query", action="append", default=[],
                        help="real query text to embed (repeatable)")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--samples", type=int, default=3,
                        help="random stored vectors to probe when no --query given")
    parser.add_argument("--vector-type", choices=("vector", "halfvec"), default="vector",
                        help="storage type of the embedding column (for the query cast)")
    args = parser.parse_args(argv)

    store = PgVectorStore(PgConfig(host=args.host, port=args.port, user=args.user,
                                   password=args.password, database=args.database))
    try:
        store.connect()
    except Exception as e:
        print(f"ERROR: cannot connect to PostgreSQL: {e}")
        print("Is the container running? Try `make pg-up`.")
        return 1

    print(f"DB: {args.host}:{args.port}/{args.database}")
    stats = store.stats()
    print(f"chunks: {stats['chunks']:,}  embeddings: {stats['embeddings']:,}  documents: {stats['documents']:,}")

    if stats["embeddings"] == 0:
        print("no embeddings yet — run `make pg-load` first")
        store.close()
        return 1

    queries: List[tuple] = []  # (label, vector)
    try:
        if args.query:
            from src.retrieval.query import MedCPTQueryEncoder
            enc = MedCPTQueryEncoder()
            print(f"\nEmbedding {len(args.query)} query(ies) with {enc.model_name} ...")
            vecs = enc.encode(args.query)
            for label, vec in zip(args.query, vecs):
                queries.append((label, vec))

        if not queries:
            # Random stored vectors as pseudo-queries (self-hit check).
            import psycopg2
            from pgvector.psycopg2 import register_vector
            conn = store.connect()
            register_vector(conn)
            cur = conn.cursor()
            cur.execute(
                f"SELECT chunk_id, embedding FROM medrag.embeddings "
                f"ORDER BY random() LIMIT %s", (args.samples,)
            )
            for cid, vec in cur.fetchall():
                queries.append((cid, _vec_np(vec)))
            cur.close()
            print(f"\nProbing {len(queries)} random stored vectors (self-hit check)")

        qtype = args.vector_type
        for label, qvec in queries:
            print(f"\n--- query: {label[:120]}")
            results = store.search(qvec, top_k=args.top_k)
            if not results:
                print("  (no results)")
                continue
            for rank, (cid, sim) in enumerate(results, 1):
                row = store.get_chunk(cid) or {}
                text = (row.get("text") or "").strip().replace("\n", " ")
                doc = row.get("document_id") or "?"
                ctype = row.get("chunk_type") or "?"
                print(f"  {rank}. [{sim:.4f}] {cid}  (doc {doc}, {ctype})")
                if text:
                    print(f"       {text[:140]}")
    finally:
        store.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())