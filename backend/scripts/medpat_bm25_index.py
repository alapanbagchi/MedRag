#!/usr/bin/env python3
"""medpat BM25 index (re)build for the ParadeDB pg_search leg.

    python scripts/medpat_bm25_index.py [--dsn ...] [--tokenizer whitespace]

Drops and recreates the ParadeDB index over medpat.chunks.embedding_text.
Run after a tokenizer change or a schema reset; harmless when re-run.
"""

from __future__ import annotations

import argparse
import sys
from typing import Optional

DEFAULT_DSN = "postgresql://medpat:medpat@localhost:5433/medpat"


def main(argv: Optional[list] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dsn", default=None)
    parser.add_argument("--tokenizer", default="whitespace",
                        choices=["whitespace", "unicode", "literal", "ngram"],
                        help="Per-field tokenizer for embedding_text (whitespace keeps drug/gene terms like EGFR intact).")
    args = parser.parse_args(argv)

    import psycopg2

    conn = psycopg2.connect(args.dsn or DEFAULT_DSN)
    conn.autocommit = True
    cur = conn.cursor()
    cur.execute("CREATE EXTENSION IF NOT EXISTS pg_search")
    cur.execute("DROP INDEX IF EXISTS chunks_bm25_idx")
    cur.execute("CREATE INDEX chunks_bm25_idx ON medpat.chunks"
                " USING paradedb (id, retrieval_eligible, (embedding_text::pdb.%s))"
                " WITH (key_field = 'id')" % args.tokenizer)
    cur.execute("SELECT count(*) FROM pg_indexes WHERE indexname = 'chunks_bm25_idx'")
    ok = cur.fetchone()[0]
    cur.close()
    conn.close()
    print(f"chunks_bm25_idx ready (tokenizer={args.tokenizer}): {bool(ok)}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
