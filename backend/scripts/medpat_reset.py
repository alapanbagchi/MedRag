"""medpat reset: wipe ALL corpus data, keep the schema (tables stay, empty).

    python scripts/medpat_reset.py [--dsn DSN] [--yes]

Truncates every medpat data table and resets medpat.meta. Fast and safe to
re-run. Use medpat_migrate.py instead when the schema itself must be rebuilt.

Note: TRUNCATE on a parent table requires CASCADE when child tables reference
it; all corpus tables are truncated together here.
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.chunking.store_pg import connect  # noqa: E402

# Dependency order does not matter with CASCADE, but keep it readable.
TABLES = (
    "chunk_citations", "references", "chunk_embeddings", "chunks", "units",
    "documents", "lexicon_terms", "load_marks",
)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Wipe medpat corpus data (keep schema).")
    ap.add_argument("--dsn", default="", help="Postgres DSN (default: MEDPAT_DSN).")
    ap.add_argument("--yes", action="store_true", help="skip the confirmation prompt")
    args = ap.parse_args(argv)

    if not args.yes:
        reply = input("Wipe ALL medpat corpus data (schema is kept)? [y/N] ")
        if reply.strip().lower() not in ("y", "yes"):
            print("aborted")
            return 1

    conn = connect(args.dsn or None)
    before = _counts(conn)
    with conn:
        cur = conn.cursor()
        for table in TABLES:
            cur.execute(f"TRUNCATE medpat.{table} CASCADE")
        cur.execute("DELETE FROM medpat.meta WHERE key <> 'schema_version'")
        cur.close()
    after = _counts(conn)
    conn.close()
    print(f"reset medpat: {before} -> {after}")
    return 0


def _counts(conn) -> dict:
    cur = conn.cursor()
    out = {}
    for table in ("documents", "units", "chunks"):
        cur.execute(f"SELECT count(*) FROM medpat.{table}")
        out[table] = cur.fetchone()[0]
    cur.close()
    return out


if __name__ == "__main__":
    sys.exit(main())
