"""medpat migrate: rebuild the medpat schema from scratch.

    python scripts/medpat_migrate.py [--dsn DSN] [--yes] [--keep-data]

Drops the whole medpat schema (every table, view, index and trigger) and
re-applies docker/medpat/init/*.sql in order - 01_schema.sql then
02_search.sql (the ParadeDB BM25 index). This is the script to run after
editing the schema. DESTRUCTIVE: all corpus data is lost unless --keep-data
is given, which dumps every table to a temp schema, rebuilds, and copies the
rows back for tables that still exist.

Connection must be able to CREATE EXTENSION pg_search (the medpat superuser
in the docker container can).
"""

from __future__ import annotations

import argparse
import glob
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.chunking.store_pg import connect  # noqa: E402

INIT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "..", "docker", "medpat", "init")


def _sql_files(init_dir: str) -> list[str]:
    return sorted(glob.glob(os.path.join(init_dir, "*.sql")))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Rebuild the medpat schema from init/*.sql.")
    ap.add_argument("--dsn", default="", help="Postgres DSN (default: MEDPAT_DSN).")
    ap.add_argument("--yes", action="store_true", help="skip the confirmation prompt")
    ap.add_argument("--keep-data", action="store_true",
                    help="copy surviving tables' rows across the rebuild")
    ap.add_argument("--init-dir", default=INIT_DIR, help="directory holding the *.sql files")
    args = ap.parse_args(argv)

    files = _sql_files(args.init_dir)
    if not files:
        print(f"no .sql files found in {args.init_dir}", file=sys.stderr)
        return 1

    if not args.yes:
        reply = input("DROP the entire medpat schema and rebuild it? [y/N] ")
        if reply.strip().lower() not in ("y", "yes"):
            print("aborted")
            return 1

    conn = connect(args.dsn or None)
    cur = conn.cursor()

    # --keep-data: park every table in a throwaway schema, then copy back the
    # rows for tables that the new schema still defines.
    parked: list[str] = []
    if args.keep_data:
        cur.execute("DROP SCHEMA IF EXISTS medpat_backup CASCADE")
        cur.execute("CREATE SCHEMA medpat_backup")
        cur.execute("SELECT tablename FROM pg_tables WHERE schemaname='medpat'")
        for (table,) in cur.fetchall():
            # LIKE copies columns/indexes/constraints but NOT foreign keys, so
            # parking order is irrelevant (backup tables have no FKs)
            cur.execute(f"CREATE TABLE medpat_backup.{table} "
                        f"(LIKE medpat.{table} INCLUDING ALL)")
            cur.execute(f"INSERT INTO medpat_backup.{table} SELECT * FROM medpat.{table}")
            parked.append(table)
        print(f"parked {len(parked)} table(s) in medpat_backup")

    print(f"dropping schema medpat")
    cur.execute("DROP SCHEMA IF EXISTS medpat CASCADE")

    for path in files:
        print(f"applying {os.path.basename(path)}")
        cur.execute(open(path, encoding="utf-8").read())

    if parked:
        cur.execute("SELECT tablename FROM pg_tables WHERE schemaname='medpat'")
        new_tables = {t for (t,) in cur.fetchall()}
        # restore parents before children so FKs never dangle
        dep_order = ("documents", "units", "chunks", "references",
                     "chunk_citations", "chunk_embeddings",
                     "lexicon_terms", "load_marks", "meta")
        for table in dep_order:
            if table not in parked or table not in new_tables:
                if table in parked and table not in new_tables:
                    print(f"  skip {table}: not in the new schema")
                continue
            # init/*.sql may already seed rows (medpat.meta): copy only the
            # parked rows that the fresh schema does not already have
            cur.execute(f"SELECT count(*) FROM medpat.{table}")
            existing = cur.fetchone()[0]
            if existing:
                print(f"  skip {table}: new schema already has {existing} row(s)")
                continue
            cur.execute(f"INSERT INTO medpat.{table} "
                        f"SELECT * FROM medpat_backup.{table}")
            print(f"  restored {table}: {cur.rowcount} row(s)")
        cur.execute("DROP SCHEMA medpat_backup CASCADE")

    conn.commit()
    cur.execute("SELECT count(*) FROM medpat.documents")
    docs = cur.fetchone()[0]
    cur.close()
    conn.close()
    print(f"medpat schema rebuilt from {len(files)} file(s); documents={docs}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
