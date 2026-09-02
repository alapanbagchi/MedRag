#!/usr/bin/env python3
"""Initialize the MedPat Memory + Context layer schema in PostgreSQL.

Creates (idempotently) the ``medrag_memory`` schema: conversations,
messages, conversation summaries, research sessions, research questions,
evidence references, claims, claim-evidence links, claim relations,
contradictions, research gaps, memory events and user preferences — plus
optional pgvector embedding columns when the extension is available.

The memory layer never stores PMC evidence itself; ``evidence_references``
holds only pointers (PMCID/PMID/DOI + chunk) with verification status.

Usage:
    python scripts/memory_init.py                 # create schema (idempotent)
    python scripts/memory_init.py --stats         # print current counts
    python scripts/memory_init.py --drop          # DROP SCHEMA medrag_memory CASCADE

PG connection: PGHOST/PGPORT/PGUSER/PGPASSWORD/PGDATABASE env vars
(or --host/--port/--user/--password/--database flags).
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default=os.environ.get("PGHOST", "localhost"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("PGPORT", "5432")))
    parser.add_argument("--user", default=os.environ.get("PGUSER", "postgres"))
    parser.add_argument("--password", default=os.environ.get("PGPASSWORD", "medrag"))
    parser.add_argument("--database", default=os.environ.get("PGDATABASE", "medrag"))
    parser.add_argument("--dim", type=int, default=int(os.environ.get("MEMORY_EMBED_DIM", "256")),
                        help="embedding width for the memory vector columns")
    parser.add_argument("--stats", action="store_true",
                        help="print current memory table counts and exit")
    parser.add_argument("--drop", action="store_true",
                        help="DROP SCHEMA medrag_memory CASCADE and exit")
    args = parser.parse_args(argv)

    dsn = " ".join([
        f"host={args.host}", f"port={args.port}", f"dbname={args.database}",
        f"user={args.user}", f"password={args.password}"])

    try:
        from src.memory.store import PostgresMemoryStore
    except ImportError:
        # allow running from the repo root without src on the path
        sys.path.insert(0, str(PROJECT_ROOT / "backend"))
        from src.memory.store import PostgresMemoryStore

    store = PostgresMemoryStore(dsn=dsn, dim=args.dim)

    try:
        store.connect()
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR: cannot connect to PostgreSQL: {exc}")
        print("Is the database up? (make pg-up / docker compose up)")
        return 1

    try:
        if args.drop:
            store._execute(
                f"DROP SCHEMA IF EXISTS {store.schema} CASCADE")
            print(f"Dropped schema {store.schema}")
            return 0

        store.ensure_schema()
        counts = store.counts()
        print(f"Schema {store.schema!r} ready "
              f"(pgvector {'available' if store.has_dense else 'NOT available'} — "
              f"memory recall degrades to lexical-only without it).")
        for label, n in counts.items():
            print(f"  {label:>16}: {n}")
        print("\nMemory layer is ready: attach it to the pipeline via")
        print("  python -m src --memory \"<query>\"   (or MEMORY_ENABLED=1)")
        return 0
    except Exception as exc:  # noqa: BLE001
        import traceback
        traceback.print_exc()
        print(f"ERROR: {exc}")
        return 1
    finally:
        store.close()


if __name__ == "__main__":
    sys.exit(main())