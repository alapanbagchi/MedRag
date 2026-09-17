#!/usr/bin/env python3
"""medpat fullvec migrate: chunk_embeddings halfvec(768) -> vector(768).

    python scripts/medpat_fullvec_migrate.py [--dsn DSN] [--reset-embeddings]

The medpat schema originally stored chunk embeddings as halfvec(768) (fp16).
This migration widens the column to full-precision vector(768) (fp32) and
rebuilds the HNSW index with vector_ip_ops. It is idempotent: re-running on an
already-migrated database is a no-op.

Widening an existing halfvec value preserves the ALREADY ROUNDED fp16 value -
the original fp32 precision is gone. Pass --reset-embeddings to TRUNCATE
medpat.chunk_embeddings so python -m src.embedding regenerates every vector at
full precision (only when the embedding job can be re-run).

NOTE: ALTER COLUMN TYPE rewrites the whole table under an ACCESS EXCLUSIVE
lock. Fine for an empty/small table; for a large corpus prefer migrating first
and then re-embedding.
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.chunking.store_pg import DEFAULT_DSN, connect  # noqa: E402

DROP_INDEX = "DROP INDEX IF EXISTS medpat.embeddings_vector_idx"
ALTER_COLUMN = (
    "ALTER TABLE medpat.chunk_embeddings "
    "ALTER COLUMN embedding TYPE vector(768) USING embedding::vector"
)
CREATE_INDEX = (
    "CREATE INDEX embeddings_vector_idx ON medpat.chunk_embeddings "
    "USING hnsw (embedding vector_ip_ops) "
    "WITH (m = 16, ef_construction = 128)"
)
TRUNCATE = "TRUNCATE medpat.chunk_embeddings"


def _is_vector_type(type_name: str) -> bool:
    """True for the full-precision target type (tolerates spacing)."""
    return (type_name or "").replace(" ", "") == "vector(768)"


def _embedding_type(conn) -> str:
    cur = conn.cursor()
    cur.execute(
        "SELECT format_type(atttypid, atttypmod) FROM pg_attribute "
        "WHERE attrelid = 'medpat.chunk_embeddings'::regclass "
        "AND attname = 'embedding'"
    )
    row = cur.fetchone()
    cur.close()
    return row[0] if row else ""


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="Widen medpat chunk embeddings from halfvec(768) to "
                    "full-precision vector(768)."
    )
    ap.add_argument("--dsn", default=os.environ.get("MEDPAT_DSN", DEFAULT_DSN))
    ap.add_argument("--reset-embeddings", action="store_true",
                    help="TRUNCATE chunk_embeddings after migrating so the next "
                         "medpat-embed run regenerates fp32 vectors")
    args = ap.parse_args(argv)

    conn = connect(args.dsn)
    try:
        before = _embedding_type(conn)
        if not before:
            print("medpat.chunk_embeddings.embedding not found; run "
                  "scripts/medpat_migrate.py first", file=sys.stderr)
            return 1

        if _is_vector_type(before):
            print("already vector(768) - nothing to migrate")
        else:
            print(f"migrating embedding column: {before} -> vector(768)")
            with conn:
                cur = conn.cursor()
                cur.execute(DROP_INDEX)
                cur.execute(ALTER_COLUMN)
                cur.execute(CREATE_INDEX)
                cur.close()
            print("column widened + HNSW index rebuilt (vector_ip_ops)")

        if args.reset_embeddings:
            with conn:
                cur = conn.cursor()
                cur.execute(TRUNCATE)
                cur.close()
            print("chunk_embeddings truncated - re-run make medpat-embed")

        cur = conn.cursor()
        cur.execute("SELECT count(*) FROM medpat.chunk_embeddings")
        n = cur.fetchone()[0]
        cur.execute(
            "SELECT format_type(atttypid, atttypmod) FROM pg_attribute "
            "WHERE attrelid = 'medpat.chunk_embeddings'::regclass "
            "AND attname = 'embedding'"
        )
        after = cur.fetchone()[0]
        cur.execute(
            "SELECT indexdef FROM pg_indexes WHERE schemaname = 'medpat' "
            "AND tablename = 'chunk_embeddings' "
            "AND indexname = 'embeddings_vector_idx'"
        )
        idx = cur.fetchone()
        cur.close()
        print(f"embedding type: {after}; rows: {n}")
        print(f"index: {idx[0] if idx else '(missing)'}")
        return 0
    finally:
        conn.close()


if __name__ == "__main__":
    sys.exit(main())
