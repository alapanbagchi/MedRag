"""Initialize PostgreSQL + pgvector database for MedRAG.

Creates the database, schema, extension, and tables.
Optionally bulk-loads from existing embedding parquet files.

Usage:
    python scripts/init_pgvector.py
    python scripts/init_pgvector.py --load-from embeddings --corpus index/corpus.parquet
    python scripts/init_pgvector.py --reset
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


def create_database(config) -> bool:
    """Create the medrag database if it doesn't exist."""
    import psycopg2
    import psycopg2.extensions

    # Connect to 'postgres' default DB to create our database
    dsn = f"host={config.host} port={config.port} dbname=postgres user={config.user}"
    if config.password:
        dsn += f" password={config.password}"

    conn = psycopg2.connect(dsn)
    conn.set_isolation_level(psycopg2.extensions.ISOLATION_LEVEL_AUTOCOMMIT)
    cur = conn.cursor()

    # Check if database exists
    cur.execute("SELECT 1 FROM pg_database WHERE datname = %s", (config.database,))
    exists = cur.fetchone() is not None

    if not exists:
        cur.execute(f"CREATE DATABASE {config.database}")
        print(f"Created database: {config.database}")
    else:
        print(f"Database already exists: {config.database}")

    cur.close()
    conn.close()
    return not exists


def main(argv: Optional[list] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default=os.environ.get("PGHOST", "localhost"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("PGPORT", "5432")))
    parser.add_argument("--user", default=os.environ.get("PGUSER", "postgres"))
    parser.add_argument("--password", default=os.environ.get("PGPASSWORD", ""))
    parser.add_argument("--database", default=os.environ.get("PGDATABASE", "medrag"))
    parser.add_argument("--load-from", type=Path, default=None,
                        help="Embeddings directory (e.g. embeddings/) to bulk-load from")
    parser.add_argument("--corpus", type=Path, default=None,
                        help="corpus.parquet path for chunk metadata")
    parser.add_argument("--reset", action="store_true",
                        help="Drop and recreate everything")
    parser.add_argument("--batch-size", type=int, default=2000)
    args = parser.parse_args(argv)

    from medrag.retrieval.pgvector_store import PgConfig, PgVectorStore

    config = PgConfig(host=args.host, port=args.port, user=args.user,
                      password=args.password, database=args.database)

    # Step 1: Create database
    print("=" * 60)
    print("MEDRAG PGVECTOR INITIALIZATION")
    print("=" * 60)
    print(f"Host:     {config.host}:{config.port}")
    print(f"User:     {config.user}")
    print(f"Database: {config.database}")
    print()

    try:
        create_database(config)
    except Exception as e:
        print(f"ERROR: Could not connect to PostgreSQL: {e}")
        print("Make sure PostgreSQL is running and accessible.")
        return 1

    # Step 2: Connect and create schema
    store = PgVectorStore(config)

    if args.reset:
        print("Resetting: dropping existing schema...")
        conn = store.connect()
        cur = conn.cursor()
        cur.execute("DROP SCHEMA IF EXISTS medrag CASCADE")
        conn.commit()
        cur.close()
        print("  Schema dropped.")

    print("Creating schema and tables...")
    store.ensure_schema()
    print("  Schema: medrag")
    print("  Tables: chunks, embeddings")
    print("  Index:  ivfflat (cosine)")

    # Step 3: Bulk load if requested
    if args.load_from:
        print()
        print(f"Bulk loading embeddings from {args.load_from}...")
        from medrag.retrieval.pgvector_store import load_from_parquet_embeddings

        def progress(i, total, count):
            print(f"  [{i}/{total}] {count:,} embeddings")

        stats = load_from_parquet_embeddings(
            store,
            embedding_dir=args.load_from,
            corpus_path=args.corpus,
            batch_size=args.batch_size,
            progress_callback=progress,
        )

        print()
        print(f"Loaded:")
        print(f"  Embedding files: {stats['embedding_files']}")
        print(f"  Total embeddings: {stats['total_embeddings']:,}")
        print(f"  Total chunks:     {stats['total_chunks']:,}")
        print(f"  Elapsed:          {stats['elapsed_seconds']:.1f}s")

    # Step 4: Print stats
    print()
    print("Database stats:")
    s = store.stats()
    print(f"  Chunks:     {s['chunks']:,}")
    print(f"  Embeddings: {s['embeddings']:,}")
    print(f"  Documents:  {s['documents']:,}")

    store.close()
    print()
    print("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
